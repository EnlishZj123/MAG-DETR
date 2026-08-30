"""Find confusion-prone classes for ETF classification.

This script runs a trained baseline detector on a chosen split, matches predictions
to ground-truth using the configured Hungarian matcher, and accumulates a class
confusion matrix.

Compared with the original version, this version is stricter and less biased:
- Uses train split by default to avoid selecting ETF classes from the validation set.
- Uses stricter default filters: IoU >= 0.5 and score >= 0.05.
- Disables auto-relax by default, so bad thresholds are not silently weakened.
- Supports normalized/hybrid ranking to reduce the bias toward frequent classes.
- Adds --max-recommended-k to really control the final number of ETF classes.
- Keeps --max-classes as a backward-compatible alias for --max-recommended-k.

Single-GPU usage example:
  python tools/find_confusion_classes_fixed_multigpu.py \
    -c configs/rtdetrv2/rtdetrv2_dinov3_vit_6x_coco.yml \
    -r /data2/ZJ_output2/output22_oco2017_1k_vitl16_best/best.pth \
    --device cuda:4 \
    --split train \
    --min-iou 0.5 \
    --min-score 0.05 \
    --topk-pairs 50 \
    --coverage-threshold 0.7 \
    --selection-metric hybrid \
    --max-recommended-k 21 \
    --no-auto-relax-filters

Multi-GPU usage example:
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 \
    tools/find_confusion_classes_fixed_multigpu.py \
    -c configs/rtdetrv2/rtdetrv2_dinov3_vit_6x_coco.yml \
    -r /data2/ZJ_output2/output22_oco2017_1k_vitl16_best/best.pth \
    --split train \
    --min-iou 0.5 \
    --min-score 0.05 \
    --topk-pairs 50 \
    --coverage-threshold 0.7 \
    --selection-metric hybrid \
    --max-recommended-k 21 \
    --no-auto-relax-filters
"""

import argparse
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.distributed as dist

ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, ROOT)

from src.core import YAMLConfig
from src.zoo.rtdetr.box_ops import box_cxcywh_to_xyxy, box_iou


def _load_state_dict_from_ckpt(ckpt_path: str) -> dict:
    """Load model state dict from a checkpoint file."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"]:
            return ckpt["ema"]["module"]
        if "model" in ckpt:
            return ckpt["model"]
    raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")


def _move_target_to_device(target: dict, device: torch.device) -> dict:
    """Move tensor values in a target dict to device; keep non-tensor values unchanged."""
    out = {}
    for k, v in target.items():
        out[k] = v.to(device) if torch.is_tensor(v) else v
    return out


def _normalize_score(x: torch.Tensor) -> torch.Tensor:
    """Normalize a non-negative score vector to [0, 1]."""
    x = x.float()
    max_v = float(x.max().item()) if x.numel() > 0 else 0.0
    if max_v <= 0:
        return torch.zeros_like(x, dtype=torch.float32)
    return x / max_v


def _unwrap_dataset(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _get_dataloader(cfg: YAMLConfig, split: str):
    """Return configured dataloader by split name."""
    if split == "train":
        data_loader = getattr(cfg, "train_dataloader", None)
    elif split == "val":
        data_loader = getattr(cfg, "val_dataloader", None)
    else:
        raise ValueError(f"Unsupported split: {split}")

    if data_loader is None:
        raise AttributeError(f"cfg.{split}_dataloader does not exist. Please check your config.")
    return data_loader



def _setup_distributed(args):
    """Initialize torch.distributed when launched by torchrun.

    Returns:
        distributed, world_size, rank, local_rank, device, is_main_process
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"

        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
    else:
        device = torch.device(args.device)

    is_main_process = rank == 0
    return distributed, world_size, rank, local_rank, device, is_main_process


def _all_reduce_long(value: int, device: torch.device) -> int:
    """Sum an integer value across all distributed ranks."""
    tensor = torch.tensor([int(value)], device=device, dtype=torch.long)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return int(tensor.item())


def _all_reduce_confusion(confusion: torch.Tensor, kept: int, device: torch.device):
    """Sum confusion matrix and kept-match count across all distributed ranks."""
    confusion_device = confusion.to(device)
    kept_device = torch.tensor([int(kept)], device=device, dtype=torch.long)

    dist.all_reduce(confusion_device, op=dist.ReduceOp.SUM)
    dist.all_reduce(kept_device, op=dist.ReduceOp.SUM)

    return confusion_device.cpu(), int(kept_device.item())


def _print_class_mapping(rows: List[Tuple[int, int, str]]) -> None:
    print("\nClass id mapping (model label_id -> dataset category_id -> class_name):")
    if len(rows) == 0:
        print("  <empty>")
        return

    print("  label_id | category_id | class_name")
    for label_id, category_id, class_name in rows:
        print(f"  {label_id:>8} | {category_id:>11} | {class_name}")


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split used to mine confusion classes. Use train by default to avoid val-set selection bias.",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.5,
        help="IoU filter on Hungarian-matched pairs. Use <=0 to disable.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.05,
        help="Score filter on Hungarian-matched pairs. Use <=0 to disable.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="0=all")
    parser.add_argument("--topk-pairs", type=int, default=50)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Print progress every N processed batches. Set <=0 to disable periodic logs.",
    )

    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.7,
        help="Target cumulative selection-score coverage to pick recommended k.",
    )
    parser.add_argument(
        "--coverage-max-k",
        type=int,
        default=30,
        help="Max k to print in k-coverage curve. 0 = print all non-zero ranks.",
    )
    parser.add_argument(
        "--min-recommended-k",
        type=int,
        default=2,
        help="Minimum recommended class count when confusion exists.",
    )
    parser.add_argument(
        "--max-recommended-k",
        type=int,
        default=0,
        help="Maximum final recommended class count. 0 = no cap.",
    )
    parser.add_argument(
        "--max-classes",
        type=int,
        default=0,
        help="Deprecated alias of --max-recommended-k. Kept for backward compatibility.",
    )

    parser.add_argument(
        "--selection-metric",
        type=str,
        default="hybrid",
        choices=["count", "rate", "hybrid"],
        help=(
            "Metric for ranking confusion-prone classes. "
            "count = raw outgoing+incoming confusion; "
            "rate = normalized outgoing/incoming confusion rates; "
            "hybrid = weighted combination of normalized count and normalized rate."
        ),
    )
    parser.add_argument(
        "--count-weight",
        type=float,
        default=0.5,
        help="Weight of normalized count score when --selection-metric=hybrid. Clamped to [0, 1].",
    )

    parser.add_argument("--etf-scale-init", type=float, default=10.0)
    parser.add_argument("--etf-seed", type=int, default=0)
    parser.add_argument(
        "--etf-scale-trainable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether ETF scale is trainable. Only printed into the YAML block.",
    )
    parser.add_argument(
        "--auto-relax-filters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Automatically relax min-iou/min-score if no matches survive filters. Disabled by default.",
    )
    return parser


@torch.no_grad()
def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()

    if args.max_classes > 0 and args.max_recommended_k <= 0:
        args.max_recommended_k = args.max_classes

    args.count_weight = min(max(float(args.count_weight), 0.0), 1.0)

    # Keep data order and stochastic transforms as consistent as possible across ranks.
    torch.manual_seed(int(args.etf_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.etf_seed))

    distributed, world_size, rank, local_rank, device, is_main_process = _setup_distributed(args)

    # Only rank 0 prints final results. Other ranks still report errors through stderr.
    if distributed and not is_main_process:
        sys.stdout = open(os.devnull, "w")

    start_t = time.time()

    if distributed:
        print(
            f"[dist] world_size={world_size}, rank={rank}, "
            f"local_rank={local_rank}, device={device}"
        )
    print("[1/5] Loading YAML config...")

    cfg = YAMLConfig(args.config)
    print("[2/5] Building model and loading checkpoint...")
    model = cfg.model
    state = _load_state_dict_from_ckpt(args.resume)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    print("[3/5] Building criterion and matcher...")
    criterion = cfg.criterion
    matcher = criterion.matcher
    matcher.eval()

    num_classes = int(getattr(criterion, "num_classes", 80))

    # Store matched tuples so filters can be changed without another full forward pass.
    matched_tuples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    total_matches = 0

    print(f"[4/5] Building dataloader for split='{args.split}'...")
    data_loader = _get_dataloader(cfg, args.split)
    try:
        total_batches = len(data_loader)
    except TypeError:
        total_batches = -1

    if total_batches > 0:
        print(f"[5/5] Start scanning batches: total={total_batches}, max_images={args.max_images}")
    else:
        print(f"[5/5] Start scanning batches: total=<unknown>, max_images={args.max_images}")

    for idx_img, (samples, targets) in enumerate(data_loader):
        if args.max_images and idx_img >= args.max_images:
            break

        # Manual batch sharding for torchrun. Each rank runs forward only on its shard.
        # The dataloader is still iterated on every rank so existing project dataloader
        # code does not need to be modified.
        if distributed and (idx_img % world_size) != rank:
            continue

        if args.log_interval > 0 and idx_img > 0 and (idx_img % args.log_interval == 0):
            elapsed = time.time() - start_t
            if total_batches > 0:
                print(
                    f"[progress] batch={idx_img}/{total_batches}, "
                    f"matched_pairs_so_far={total_matches}, elapsed={elapsed:.1f}s"
                )
            else:
                print(
                    f"[progress] batch={idx_img}, "
                    f"matched_pairs_so_far={total_matches}, elapsed={elapsed:.1f}s"
                )

        samples = samples.to(device)
        targets = [_move_target_to_device(t, device) for t in targets]

        outputs = model(samples)
        matched = matcher(outputs, targets)
        indices = matched["indices"]

        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if pred_idx.numel() == 0:
                continue

            pred_boxes = outputs["pred_boxes"][b, pred_idx]  # [M, 4], cxcywh
            tgt_boxes = targets[b]["boxes"][tgt_idx]         # [M, 4], cxcywh

            ious, _ = box_iou(
                box_cxcywh_to_xyxy(pred_boxes),
                box_cxcywh_to_xyxy(tgt_boxes),
            )
            ious = torch.diag(ious).detach().cpu()

            logits = outputs["pred_logits"][b, pred_idx]  # [M, C]
            pred_cls = torch.argmax(logits, dim=-1)
            pred_score = torch.sigmoid(
                logits.gather(1, pred_cls.unsqueeze(1)).squeeze(1)
            ).detach().cpu()

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
                gt_keep = gt_cls[keep].long()
                pred_keep = pred_cls[keep].long()

                valid = (
                    (gt_keep >= 0)
                    & (gt_keep < num_classes)
                    & (pred_keep >= 0)
                    & (pred_keep < num_classes)
                )
                if valid.any():
                    gt_keep = gt_keep[valid]
                    pred_keep = pred_keep[valid]
                    kept_local += int(gt_keep.numel())

                    flat_idx = gt_keep * num_classes + pred_keep
                    counts = torch.bincount(
                        flat_idx,
                        minlength=num_classes * num_classes,
                    ).view(num_classes, num_classes)
                    confusion_local += counts

        return confusion_local, kept_local

    if distributed:
        total_matches = _all_reduce_long(total_matches, device)

    trial_filters = [(float(args.min_iou), float(args.min_score))]
    if bool(args.auto_relax_filters):
        trial_filters.extend([(0.2, 0.0), (0.1, 0.0), (0.0, 0.0)])

    confusion = None
    kept_matches = 0
    used_iou = float(args.min_iou)
    used_score = float(args.min_score)

    for min_iou, min_score in trial_filters:
        confusion_try, kept_try = build_confusion(min_iou=min_iou, min_score=min_score)

        if distributed:
            confusion_try, kept_try = _all_reduce_confusion(confusion_try, kept_try, device)

        if kept_try > 0 or (min_iou, min_score) == trial_filters[-1]:
            confusion = confusion_try
            kept_matches = kept_try
            used_iou = min_iou
            used_score = min_score
            break

    assert confusion is not None

    conf_offdiag = confusion.clone()
    conf_offdiag.fill_diagonal_(0)

    # Top confusion pairs by raw count.
    flat = conf_offdiag.flatten()
    topk = min(int(args.topk_pairs), flat.numel())
    vals, inds = torch.topk(flat, k=topk)

    pairs: List[Tuple[int, int, int]] = []
    for v, i in zip(vals.tolist(), inds.tolist()):
        if v <= 0:
            continue
        gt = i // num_classes
        pr = i % num_classes
        pairs.append((gt, pr, int(v)))

    # Class-level statistics.
    out_conf_count = conf_offdiag.sum(dim=1).float()  # gt c predicted as others
    in_conf_count = conf_offdiag.sum(dim=0).float()   # others predicted as c
    count_score = out_conf_count + in_conf_count

    gt_kept_count = confusion.sum(dim=1).float()
    pred_kept_count = confusion.sum(dim=0).float()

    out_conf_rate = out_conf_count / gt_kept_count.clamp(min=1.0)
    in_conf_rate = in_conf_count / pred_kept_count.clamp(min=1.0)
    rate_score = 0.5 * (out_conf_rate + in_conf_rate)

    if args.selection_metric == "count":
        selection_score = count_score
    elif args.selection_metric == "rate":
        selection_score = rate_score
    else:
        selection_score = (
            args.count_weight * _normalize_score(count_score)
            + (1.0 - args.count_weight) * _normalize_score(rate_score)
        )

    nonzero_mask = selection_score > 0
    ranked_vals_all, ranked_ids_all = torch.sort(selection_score, descending=True)
    ranked_ids = ranked_ids_all[nonzero_mask[ranked_ids_all]]
    ranked_vals = ranked_vals_all[nonzero_mask[ranked_ids_all]]

    total_score = float(selection_score.sum().item())
    rank_count = int(ranked_ids.numel())
    threshold = min(max(float(args.coverage_threshold), 0.0), 1.0)
    min_k = max(1, int(args.min_recommended_k))

    recommended_k = 0
    recommended_classes: List[int] = []
    coverage_rows: List[Tuple[int, float]] = []

    if total_score > 0 and rank_count > 0:
        cumsum = torch.cumsum(ranked_vals, dim=0)
        coverage = cumsum.float() / float(total_score)

        reached = (coverage >= threshold).nonzero(as_tuple=False)
        raw_k = int(reached[0].item() + 1) if reached.numel() > 0 else rank_count
        recommended_k = min(max(raw_k, min_k), rank_count)

        max_recommended_k = int(args.max_recommended_k)
        if max_recommended_k > 0:
            recommended_k = min(recommended_k, max_recommended_k, rank_count)

        recommended_classes = [int(x) for x in ranked_ids[:recommended_k].tolist()]

        max_k_arg = int(args.coverage_max_k)
        max_k_to_print = rank_count if max_k_arg <= 0 else min(max_k_arg, rank_count)
        for k in range(1, max_k_to_print + 1):
            coverage_rows.append((k, float(coverage[k - 1].item())))

    dataset = _unwrap_dataset(data_loader.dataset)
    label2category = getattr(dataset, "label2category", {})
    category2name = getattr(dataset, "category2name", {})
    mapping_rows: List[Tuple[int, int, str]] = []
    for lid in recommended_classes:
        cid = int(label2category.get(int(lid), int(lid)))
        cname = category2name.get(int(cid), str(cid))
        mapping_rows.append((int(lid), int(cid), str(cname)))

    total_elapsed = time.time() - start_t

    print("=== Confusion summary ===")
    print(f"split: {args.split}")
    print(f"num_classes: {num_classes}")
    print(f"total matched pairs before filters: {total_matches}")
    print(f"kept matched pairs: {kept_matches}  (iou >= {used_iou}, score >= {used_score})")
    print(f"elapsed_seconds: {total_elapsed:.1f}")
    print(f"selection_metric: {args.selection_metric}")
    if args.selection_metric == "hybrid":
        print(f"hybrid count_weight: {args.count_weight:.2f}, rate_weight: {1.0 - args.count_weight:.2f}")

    if (used_iou != float(args.min_iou)) or (used_score != float(args.min_score)):
        print(
            f"[auto-relax] requested (iou >= {float(args.min_iou)}, score >= {float(args.min_score)}) "
            f"had 0 matches; fallback used (iou >= {used_iou}, score >= {used_score})."
        )

    print("\nTop confusion pairs by raw count (gt -> pred : count):")
    if len(pairs) == 0:
        print("  No off-diagonal confusion pairs found.")
    else:
        for gt, pr, v in pairs[: int(args.topk_pairs)]:
            print(f"  {gt:>2} -> {pr:>2} : {v}")

    print("\nClass ranking:")
    if rank_count == 0:
        print("  No off-diagonal confusion observed; cannot rank classes.")
    else:
        max_rank_print = int(args.coverage_max_k)
        max_rank_print = rank_count if max_rank_print <= 0 else min(max_rank_print, rank_count)
        print("  rank | cls | score    | count | out_rate | in_rate")
        for r in range(max_rank_print):
            c = int(ranked_ids[r].item())
            print(
                f"  {r + 1:>4} | {c:>3} | {float(selection_score[c]):>8.4f} | "
                f"{int(count_score[c].item()):>5} | "
                f"{float(out_conf_rate[c]):>8.4f} | {float(in_conf_rate[c]):>7.4f}"
            )

    print("\n=== k-coverage ===")
    if total_score <= 0 or rank_count == 0:
        print("No off-diagonal confusion observed; cannot recommend k.")
    else:
        print(
            f"coverage_threshold={threshold:.2f}, min_recommended_k={min_k}, "
            f"max_recommended_k={int(args.max_recommended_k)}, nonzero_confusion_classes={rank_count}"
        )
        print("k -> cumulative selection-score coverage")
        for k, cov in coverage_rows:
            print(f"  {k:>2} -> {cov:.4f}")

        cumsum = torch.cumsum(ranked_vals, dim=0)
        rec_cov = float((cumsum[recommended_k - 1].float() / float(total_score)).item())

        print("\nRecommended ETF confusion classes:")
        print(f"k={recommended_k}, coverage={rec_cov:.4f}")
        print(recommended_classes)
        _print_class_mapping(mapping_rows)

    print("\nCopy-paste YAML:")
    yaml_ids = recommended_classes
    if len(yaml_ids) < 2:
        print("# Not enough confused classes found to enable ETF. Need at least 2 classes.")
    else:
        ids_str = ", ".join(str(x) for x in yaml_ids)
        scale_str = float(args.etf_scale_init)
        trainable_str = "true" if bool(args.etf_scale_trainable) else "false"
        seed_str = int(args.etf_seed)

        print("RTDETRTransformerv2:")
        print(f"  etf_confusion_classes: [{ids_str}]")
        print(f"  etf_scale_init: {scale_str}")
        print(f"  etf_scale_trainable: {trainable_str}")
        print(f"  etf_seed: {seed_str}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
