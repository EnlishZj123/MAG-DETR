"""Find confusion classes on a validation set.

This script runs the model on `val_dataloader`, matches predictions to ground-truth
using the configured Hungarian matcher, and accumulates a confusion matrix.

It prints:
- Top-K most frequent (gt -> pred) confusions
- A suggested set of class ids to treat as "confusable"
- k-coverage curve (cumulative confusion coverage as k grows)
- Recommended k and recommended ETF class ids from k-coverage

Usage (example):
  python tools/find_confusion_classes.py \
    -c configs/rtdetrv2/rtdetrv2_dinov3_vit_6x_coco2017_5k.yml \
    -r outputs/rtdetrv2_dinov3_vit_6x_coco2017_5k/best.pth \
        --device cuda:0 --topk-pairs 30 --coverage-threshold 0.8
"""

import os
import sys
from typing import List, Tuple

ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, ROOT)

import argparse
import torch

from src.core import YAMLConfig
from src.zoo.rtdetr.box_ops import box_cxcywh_to_xyxy, box_iou


def _load_state_dict_from_ckpt(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"]:
            return ckpt["ema"]["module"]
        if "model" in ckpt:
            return ckpt["model"]
    raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.3,
        help="IoU filter on Hungarian-matched pairs. Use <=0 to disable.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Score filter on Hungarian-matched pairs. Use <=0 to disable.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="0=all")
    parser.add_argument("--topk-pairs", type=int, default=30)
    parser.add_argument(
        "--max-classes",
        type=int,
        default=8,
        help="legacy cap for fixed-size suggestion; recommendation now comes from k-coverage",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.8,
        help="target cumulative confusion coverage to pick recommended k",
    )
    parser.add_argument(
        "--coverage-max-k",
        type=int,
        default=20,
        help="max k to print in k-coverage curve (0 = print all non-zero ranks)",
    )
    parser.add_argument(
        "--min-recommended-k",
        type=int,
        default=2,
        help="minimum recommended class count when confusion exists",
    )
    parser.add_argument("--etf-scale-init", type=float, default=10.0)
    parser.add_argument("--etf-seed", type=int, default=0)
    parser.add_argument(
        "--etf-scale-trainable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="whether ETF scale is trainable (prints into YAML block)",
    )
    parser.add_argument(
        "--auto-relax-filters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically relax min-iou/min-score if no matches survive filters.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    cfg = YAMLConfig(args.config)
    model = cfg.model
    state = _load_state_dict_from_ckpt(args.resume)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    criterion = cfg.criterion
    matcher = criterion.matcher
    matcher.eval()

    num_classes = int(getattr(criterion, "num_classes", 80))
    # Pre-collect matched tuples so we can retry with relaxed filters without another full forward pass.
    matched_tuples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    total_matches = 0

    # data_loader = cfg.val_dataloader
    data_loader = cfg.train_dataloader

    for idx_img, (samples, targets) in enumerate(data_loader):
        if args.max_images and idx_img >= args.max_images:
            break

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples)
        matched = matcher(outputs, targets)
        indices = matched["indices"]

        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() == 0:
                continue

            pred_boxes = outputs["pred_boxes"][b, pred_idx]  # [M,4] cxcywh
            tgt_boxes = targets[b]["boxes"][tgt_idx]  # [M,4] cxcywh
            ious, _ = box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(tgt_boxes))
            ious = torch.diag(ious).detach().cpu()

            logits = outputs["pred_logits"][b, pred_idx]  # [M,C]
            pred_cls = torch.argmax(logits, dim=-1)
            pred_score = torch.sigmoid(logits.gather(1, pred_cls.unsqueeze(1)).squeeze(1)).detach().cpu()
            gt_cls = targets[b]["labels"][tgt_idx].detach().cpu()

            matched_tuples.append((gt_cls, pred_cls.detach().cpu(), ious, pred_score))
            total_matches += int(pred_idx.numel())

    def build_confusion(min_iou: float, min_score: float):
        confusion_local = torch.zeros((num_classes, num_classes), dtype=torch.long)
        kept_local = 0
        for gt_cls, pred_cls, ious, pred_score in matched_tuples:
            keep = torch.ones_like(ious, dtype=torch.bool)
            if float(min_iou) > 0:
                keep = keep & (ious >= float(min_iou))
            if float(min_score) > 0:
                keep = keep & (pred_score >= float(min_score))
            if keep.any():
                kept_local += int(keep.sum().item())
                for g, p in zip(gt_cls[keep].tolist(), pred_cls[keep].tolist()):
                    confusion_local[g, p] += 1
        return confusion_local, kept_local

    trial_filters = [(float(args.min_iou), float(args.min_score))]
    if bool(args.auto_relax_filters):
        trial_filters.extend([
            (0.2, 0.0),
            (0.1, 0.0),
            (0.0, 0.0),
        ])

    confusion = None
    kept_matches = 0
    used_iou = float(args.min_iou)
    used_score = float(args.min_score)
    for min_iou, min_score in trial_filters:
        confusion_try, kept_try = build_confusion(min_iou=min_iou, min_score=min_score)
        if kept_try > 0 or (min_iou, min_score) == trial_filters[-1]:
            confusion = confusion_try
            kept_matches = kept_try
            used_iou = min_iou
            used_score = min_score
            break

    assert confusion is not None

    # remove diagonal for confusion pairs
    conf_offdiag = confusion.clone()
    conf_offdiag.fill_diagonal_(0)

    # list top confusion pairs
    flat = conf_offdiag.flatten()
    topk = min(int(args.topk_pairs), flat.numel())
    vals, inds = torch.topk(flat, k=topk)

    pairs: List[Tuple[int, int, int]] = []
    for v, i in zip(vals.tolist(), inds.tolist()):
        if v <= 0:
            continue
        gt = i // num_classes
        pr = i % num_classes
        pairs.append((gt, pr, v))

    print("=== Confusion summary ===")
    print(f"num_classes: {num_classes}")
    print(f"total matched pairs (before filters): {total_matches}")
    print(f"kept matched pairs (iou>= {used_iou}, score>= {used_score}): {kept_matches}")
    if (used_iou != float(args.min_iou)) or (used_score != float(args.min_score)):
        print(
            f"[auto-relax] requested (iou>={float(args.min_iou)}, score>={float(args.min_score)}) "
            f"had 0 matches; fallback used (iou>={used_iou}, score>={used_score})."
        )

    print("\nTop confusion pairs (gt -> pred : count):")
    for gt, pr, v in pairs[: int(args.topk_pairs)]:
        print(f"  {gt:>2} -> {pr:>2} : {v}")

    # Suggest a class id set from top pairs (legacy fixed-cap behavior)
    involved = []
    for gt, pr, _ in pairs:
        involved.append(gt)
        involved.append(pr)
    # frequency of involvement
    freq = torch.zeros(num_classes, dtype=torch.long)
    for c in involved:
        freq[c] += 1
    max_classes = int(args.max_classes)
    if max_classes > 0:
        top_vals, top_ids = torch.topk(freq, k=min(max_classes, num_classes))
        suggested = [int(c) for c, f in zip(top_ids.tolist(), top_vals.tolist()) if f > 0]
    else:
        suggested = []

    # Build k-coverage from the full off-diagonal confusion matrix.
    # involvement_strength[c] = outgoing confusion + incoming confusion for class c
    # (excluding diagonal true positives).
    involvement_strength = conf_offdiag.sum(dim=1) + conf_offdiag.sum(dim=0)
    nonzero_mask = involvement_strength > 0
    ranked_vals, ranked_ids = torch.sort(involvement_strength, descending=True)
    ranked_vals = ranked_vals[nonzero_mask[ranked_ids]]
    ranked_ids = ranked_ids[nonzero_mask[ranked_ids]]

    total_strength = int(involvement_strength.sum().item())
    rank_count = int(ranked_ids.numel())
    threshold = float(args.coverage_threshold)
    threshold = min(max(threshold, 0.0), 1.0)
    min_k = max(1, int(args.min_recommended_k))

    recommended_k = 0
    recommended_classes: List[int] = []
    coverage_rows: List[Tuple[int, float]] = []

    if total_strength > 0 and rank_count > 0:
        cumsum = torch.cumsum(ranked_vals, dim=0)
        coverage = cumsum.float() / float(total_strength)

        raw_k = int((coverage >= threshold).nonzero(as_tuple=False)[0].item() + 1)
        recommended_k = min(max(raw_k, min_k), rank_count)
        recommended_classes = [int(x) for x in ranked_ids[:recommended_k].tolist()]

        max_k_arg = int(args.coverage_max_k)
        if max_k_arg <= 0:
            max_k_to_print = rank_count
        else:
            max_k_to_print = min(max_k_arg, rank_count)

        for k in range(1, max_k_to_print + 1):
            coverage_rows.append((k, float(coverage[k - 1].item())))

    print("\nSuggested etf_confusion_classes (by involvement frequency):")
    print(suggested)

    print("\n=== k-coverage (cumulative confusion coverage vs number of classes) ===")
    if total_strength <= 0 or rank_count == 0:
        print("No off-diagonal confusion observed; cannot recommend k.")
    else:
        print(
            f"coverage_threshold={threshold:.2f}, min_recommended_k={min_k}, "
            f"nonzero_confusion_classes={rank_count}"
        )
        print("k -> coverage")
        for k, cov in coverage_rows:
            print(f"  {k:>2} -> {cov:.4f}")

        rec_cov = coverage_rows[recommended_k - 1][1] if coverage_rows and recommended_k <= len(coverage_rows) else None
        if rec_cov is None:
            cumsum = torch.cumsum(ranked_vals, dim=0)
            rec_cov = float((cumsum[recommended_k - 1].float() / float(total_strength)).item())

        print("\nRecommended k by k-coverage:")
        print(f"k={recommended_k}, coverage={rec_cov:.4f}")
        print("recommended_etf_confusion_classes:")
        print(recommended_classes)

    print("\nCopy-paste YAML (put this under your config root):")
    yaml_ids = recommended_classes if len(recommended_classes) >= 2 else suggested
    if len(yaml_ids) < 2:
        print("# Not enough confused classes found to enable ETF (need >=2).")
    else:
        # Keep YAML stable and one-line friendly.
        ids_str = ", ".join(str(x) for x in yaml_ids)
        scale_str = float(args.etf_scale_init)
        seed_str = int(args.etf_seed)
        trainable_str = "true" if bool(args.etf_scale_trainable) else "false"
        print("RTDETRTransformerv2:")
        print(f"  etf_confusion_classes: [{ids_str}]")
        print(f"  etf_scale_init: {scale_str}")
        print(f"  etf_scale_trainable: {trainable_str}")
        print(f"  etf_seed: {seed_str}")


if __name__ == "__main__":
    main()
