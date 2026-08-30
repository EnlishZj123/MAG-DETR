"""Evaluate RT-DETR on COCO occlusion subsets and count detected targets.

This script reuses the project's validation pipeline and evaluates three COCO
subset annotation files that split instances into light, medium, and heavy
occlusion groups.

Expected subset files:
- <subset-dir>/instances_val2017_light.json
- <subset-dir>/instances_val2017_medium.json
- <subset-dir>/instances_val2017_heavy.json

The standard COCO metrics come from the project's evaluator. In addition, this
script reports a small set of instance-level counting metrics that help compare
occlusion difficulty across the three subsets.

Example:
    python tools/eval_coco_occlusion_subsets.py \
        -c configs/rtdetrv2/rtdetrv2_dinov3_vit_6x_coco_ETF.yml \
        -r /data2/ZJ_output2/Ablation/ETF_amc_0.1/best.pth \
        --subset-dir ./occ_eval_outputs \
        --device cuda:0 \
        --score-thr 0.001 \
        --match-iou-thr 0.5 \
        --output-json ./occ_eval_outputs/occlusion_results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Set

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)

from src.core import YAMLConfig
from src.misc import dist_utils
from src.solver.det_engine import evaluate
from src.zoo.rtdetr.box_ops import box_iou


Annotation = Dict[str, Any]


def _load_state_dict_from_ckpt(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"]:
            return ckpt["ema"]["module"]
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")


def _extract_ap(stats: Dict) -> Dict[str, float]:
    bbox_stats = stats.get("coco_eval_bbox", None)
    if bbox_stats is None:
        raise RuntimeError("COCO bbox stats were not produced by evaluation.")
    return {
        "AP": float(bbox_stats[0]),
        "AP50": float(bbox_stats[1]),
        "AP75": float(bbox_stats[2]),
        "AP_small": float(bbox_stats[3]),
        "AP_medium": float(bbox_stats[4]),
        "AP_large": float(bbox_stats[5]),
        "AR_1": float(bbox_stats[6]),
        "AR_10": float(bbox_stats[7]),
        "AR_100": float(bbox_stats[8]),
        "AR_small": float(bbox_stats[9]),
        "AR_medium": float(bbox_stats[10]),
        "AR_large": float(bbox_stats[11]),
    }


def _make_overrides(subset_dir: Path, level: str) -> Dict[str, Any]:
    subset_ann = subset_dir / f"instances_val2017_{level}.json"
    if not subset_ann.exists():
        raise FileNotFoundError(f"missing subset annotation file: {subset_ann}")
    return {
        "val_dataloader": {
            "dataset": {
                "ann_file": str(subset_ann),
            }
        }
    }


def _greedy_match_counts(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    match_iou_thr: float,
) -> int:
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return 0

    ious, _ = box_iou(gt_boxes, pred_boxes)
    used_pred: Set[int] = set()
    matched = 0

    for gt_idx in range(int(gt_boxes.shape[0])):
        best_iou = 0.0
        best_pred = -1
        for pred_idx in range(int(pred_boxes.shape[0])):
            if pred_idx in used_pred:
                continue
            if int(gt_labels[gt_idx].item()) != int(pred_labels[pred_idx].item()):
                continue
            iou = float(ious[gt_idx, pred_idx].item())
            if iou > best_iou:
                best_iou = iou
                best_pred = pred_idx
        if best_pred >= 0 and best_iou >= match_iou_thr:
            used_pred.add(best_pred)
            matched += 1
    return matched


def _print_level_result(level: str, ap_stats: Dict[str, float], count_stats: Dict[str, Any], match_iou_thr: float) -> None:
    iou_tag = int(round(match_iou_thr * 100))
    print(
        f"{level:>8} | "
        f"AP: {ap_stats['AP']:.4f} | "
        f"AP50: {ap_stats['AP50']:.4f} | "
        f"AP75: {ap_stats['AP75']:.4f} | "
        f"AR100: {ap_stats['AR_100']:.4f} | "
        f"eval_images: {count_stats['num_eval_images']} | "
        f"gt_targets: {count_stats['num_gt_targets']} | "
        f"pred_boxes: {count_stats['num_pred_boxes']} | "
        f"matched_targets@IoU{match_iou_thr:.2f}: {count_stats[f'num_matched_targets_iou{iou_tag}']} | "
        f"missed_targets@IoU{match_iou_thr:.2f}: {count_stats[f'num_missed_targets_iou{iou_tag}']} | "
        f"target_recall@IoU{match_iou_thr:.2f}: {count_stats[f'target_recall_iou{iou_tag}']:.4f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a detector on COCO occlusion subsets.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Base training config")
    parser.add_argument("-r", "--resume", type=str, required=True, help="Checkpoint path")
    parser.add_argument(
        "--subset-dir",
        type=str,
        default="./occ_eval_outputs",
        help="Directory containing instances_val2017_{light,medium,heavy}.json",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--levels", nargs="+", default=["light", "medium", "heavy"])
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.0,
        help="Score threshold used only for counting predicted/matched targets. COCO AP evaluation is unchanged.",
    )
    parser.add_argument(
        "--match-iou-thr",
        type=float,
        default=0.5,
        help="IoU threshold used only for counting matched targets. COCO AP evaluation is unchanged.",
    )
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=0)

    device = torch.device(args.device)
    subset_dir = Path(args.subset_dir)
    results: Dict[str, Dict[str, Any]] = {}

    total_eval_images_sum = 0
    total_gt_targets_sum = 0
    total_pred_boxes_sum = 0
    total_matched_targets_sum = 0
    total_missed_targets_sum = 0
    total_matched_pred_boxes_sum = 0

    iou_tag = int(round(args.match_iou_thr * 100))

    state = _load_state_dict_from_ckpt(args.resume)

    for level in args.levels:
        overrides = _make_overrides(subset_dir, level)
        cfg = YAMLConfig(args.config, **overrides)
        subset_ann_path = Path(cfg.yaml_cfg["val_dataloader"]["dataset"]["ann_file"])

        model = cfg.model
        model.load_state_dict(state, strict=False)
        model.to(device)

        criterion = cfg.criterion
        postprocessor = cfg.postprocessor
        evaluator = cfg.evaluator
        data_loader = cfg.val_dataloader

        dataset_ids = list(getattr(data_loader.dataset, "ids", []))

        print(f"\n========== Evaluate {level} ==========")
        print(f"subset ann: {subset_ann_path}")
        print(f"images from dataloader: {len(dataset_ids)}")

        dataset = data_loader.dataset
        category2label = getattr(dataset, "category2label", {})
        use_category_ids = bool(getattr(postprocessor, "remap_mscoco_category", False))
        if use_category_ids and category2label:
            max_cat_id = max(category2label.keys())
            label_map = torch.full((max_cat_id + 1,), -1, dtype=torch.long)
            for cat_id, label in category2label.items():
                label_map[int(cat_id)] = int(label)
        else:
            label_map = None

        model.eval()
        criterion.eval()
        evaluator.cleanup()

        num_eval_images = 0
        num_gt_targets = 0
        num_pred_boxes = 0
        num_matched_targets = 0

        for batch_idx, (samples, targets) in enumerate(data_loader):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            batch_results = postprocessor(outputs, orig_target_sizes)

            res = {target["image_id"].item(): output for target, output in zip(targets, batch_results)}
            evaluator.update(res)

            for target, result in zip(targets, batch_results):
                gt_boxes = target.get("boxes", torch.zeros((0, 4), device=device)).detach().cpu()
                gt_labels = target.get("labels", torch.zeros((0,), device=device, dtype=torch.long)).detach().cpu()

                pred_boxes = result.get("boxes", torch.zeros((0, 4))).detach().cpu()
                pred_labels = result.get("labels", torch.zeros((0,), dtype=torch.long)).detach().cpu()
                pred_scores = result.get("scores", torch.zeros((0,))).detach().cpu()

                if use_category_ids and label_map is not None and pred_labels.numel() > 0:
                    valid = (pred_labels >= 0) & (pred_labels < label_map.numel())
                    pred_labels = torch.where(valid, label_map[pred_labels], pred_labels.new_full(pred_labels.shape, -1))

                keep = pred_scores >= float(args.score_thr)
                if pred_labels.numel() > 0:
                    keep = keep & (pred_labels >= 0)
                pred_boxes = pred_boxes[keep]
                pred_labels = pred_labels[keep]

                num_eval_images += 1
                num_gt_targets += int(gt_boxes.shape[0])
                num_pred_boxes += int(pred_boxes.shape[0])
                num_matched_targets += _greedy_match_counts(
                    gt_boxes,
                    gt_labels,
                    pred_boxes,
                    pred_labels,
                    match_iou_thr=float(args.match_iou_thr),
                )

            if (batch_idx + 1) % 50 == 0:
                print(
                    f"Progress [{level}] batch {batch_idx + 1}/{len(data_loader)} | "
                    f"pred_boxes={num_pred_boxes} matched_targets={num_matched_targets}",
                    flush=True,
                )

        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()

        stats = {
            "coco_eval_bbox": evaluator.coco_eval["bbox"].stats.tolist(),
        }
        ap_stats = _extract_ap(stats)

        num_missed_targets = max(0, num_gt_targets - num_matched_targets)
        target_recall = num_matched_targets / num_gt_targets if num_gt_targets > 0 else 0.0
        count_stats = {
            "num_eval_images": num_eval_images,
            "num_gt_targets": num_gt_targets,
            "num_pred_boxes": num_pred_boxes,
            f"num_matched_targets_iou{iou_tag}": num_matched_targets,
            f"num_missed_targets_iou{iou_tag}": num_missed_targets,
            f"target_recall_iou{iou_tag}": float(target_recall),
            "score_thr_for_counting": float(args.score_thr),
            "match_iou_thr_for_counting": float(args.match_iou_thr),
        }

        if len(dataset_ids) > 0 and len(dataset_ids) != count_stats["num_eval_images"]:
            print(
                "Warning: dataloader image count and annotation image count are different: "
                f"{len(dataset_ids)} vs {count_stats['num_eval_images']}"
            )

        level_result: Dict[str, Any] = {}
        level_result.update(ap_stats)
        level_result.update(count_stats)
        results[level] = level_result

        total_eval_images_sum += int(count_stats["num_eval_images"])
        total_gt_targets_sum += int(count_stats["num_gt_targets"])
        total_pred_boxes_sum += int(count_stats["num_pred_boxes"])
        total_matched_targets_sum += int(count_stats[f"num_matched_targets_iou{iou_tag}"])
        total_missed_targets_sum += int(count_stats[f"num_missed_targets_iou{iou_tag}"])

        _print_level_result(level, ap_stats, count_stats, args.match_iou_thr)

        if cfg.output_dir:
            out_dir = Path(cfg.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(evaluator.coco_eval["bbox"].eval, out_dir / f"eval_{level}.pth")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overall_target_recall = total_matched_targets_sum / total_gt_targets_sum if total_gt_targets_sum > 0 else 0.0

    summary: Dict[str, Any] = {
        "score_thr_for_counting": float(args.score_thr),
        "match_iou_thr_for_counting": float(args.match_iou_thr),
        "sum_eval_images": total_eval_images_sum,
        "sum_gt_targets": total_gt_targets_sum,
        "sum_pred_boxes": total_pred_boxes_sum,
        f"sum_matched_targets_iou{iou_tag}": total_matched_targets_sum,
        f"sum_missed_targets_iou{iou_tag}": total_missed_targets_sum,
        f"overall_target_recall_iou{iou_tag}": float(overall_target_recall),
    }
    results["_summary"] = summary

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results to: {out_path}")

    print("\n========== Summary ==========")
    for level in args.levels:
        level_result = results[level]
        _print_level_result(level, level_result, level_result, args.match_iou_thr)

    print("\n========== Target Count Summary ==========")
    print(f"score threshold for counting: {args.score_thr:g}")
    print(f"match IoU threshold for counting: {args.match_iou_thr:.2f}")
    print(f"sum eval images light+medium+heavy: {summary['sum_eval_images']}")
    print(f"sum GT targets light+medium+heavy: {summary['sum_gt_targets']}")
    print(f"sum predicted boxes light+medium+heavy: {summary['sum_pred_boxes']}")
    print(f"sum matched targets@IoU{args.match_iou_thr:.2f} light+medium+heavy: {summary[f'sum_matched_targets_iou{iou_tag}']}")
    print(f"sum missed targets@IoU{args.match_iou_thr:.2f} light+medium+heavy: {summary[f'sum_missed_targets_iou{iou_tag}']}")
    print(f"overall target recall@IoU{args.match_iou_thr:.2f}: {summary[f'overall_target_recall_iou{iou_tag}']:.4f}")


if __name__ == "__main__":
    main()