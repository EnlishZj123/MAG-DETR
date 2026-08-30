"""Analyze missed detections on a validation set.

This script measures missed detections with a class-aware one-to-one match:
- A GT is counted as matched only when a prediction with the same class,
  score >= threshold, and IoU >= threshold is assigned to it.
- A class-aware unmatched GT with no overlapping prediction is a pure miss.
- A class-aware unmatched GT with overlapping predictions only from other
  classes is a confusion-driven miss.
- A class-aware unmatched GT with an overlapping same-class prediction is
  counted separately as a same-class unmatched miss. This usually means a
  duplicate/assignment/merged-object issue rather than a pure miss or class
  confusion.

Outputs:
- Console summary
- summary.json
- per_class.csv
- confusions.csv
- confusion_changes.csv, when --compare-confusions is set
- worst_images.csv

Example:
  python tools/analyze_missed_detections.py \
    -c configs/rtdetrv2/rtdetrv2_dinov3_vit_6x_coco_ETF.yml \
    -r /data2/ZJ_output2/Ablation/ETF_amc_0.1/best.pth \
    --device cuda:7 --score-thr 0.15 --iou-thr 0.5
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, ROOT)

from src.core import YAMLConfig
from src.zoo.rtdetr.box_ops import box_iou


@torch.no_grad()
def _load_state_dict_from_ckpt(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "ema" in ckpt and isinstance(ckpt["ema"], dict) and "module" in ckpt["ema"]:
            return ckpt["ema"]["module"]
        if "model" in ckpt:
            return ckpt["model"]
    raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")


def _unwrap_dataset(dataset):
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset


def _safe_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _xywh_to_xyxy(box: Iterable[float]) -> List[float]:
    x, y, w, h = box
    return [float(x), float(y), float(x + w), float(y + h)]


def _area_bucket(area: float) -> str:
    if area < 32.0 * 32.0:
        return "small"
    if area < 96.0 * 96.0:
        return "medium"
    return "large"


def _load_gt_for_image(coco, image_id: int, use_category_ids: bool, category2label: Dict[int, int]):
    ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=False)
    anns = coco.loadAnns(ann_ids)

    boxes = []
    labels = []
    areas = []
    valid_anns = []
    for ann in anns:
        bbox = ann.get("bbox", None)
        if bbox is None or len(bbox) != 4:
            continue
        _, _, w, h = bbox
        if w <= 0 or h <= 0:
            continue
        boxes.append(_xywh_to_xyxy(bbox))
        category_id = int(ann["category_id"])
        label = category_id if use_category_ids else int(category2label[category_id])
        labels.append(label)
        areas.append(float(ann.get("area", w * h)))
        valid_anns.append(ann)

    if boxes:
        boxes_t = torch.tensor(boxes, dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64)
        areas_t = torch.tensor(areas, dtype=torch.float32)
    else:
        boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.zeros((0,), dtype=torch.int64)
        areas_t = torch.zeros((0,), dtype=torch.float32)

    return boxes_t, labels_t, areas_t, valid_anns


@torch.no_grad()
def _greedy_match(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    iou_thr: float,
    require_same_label: bool,
) -> List[int]:
    """COCO-style score-ordered one-to-one matching."""
    matched_pred_for_gt = [-1] * int(gt_boxes.shape[0])
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return matched_pred_for_gt

    ious, _ = box_iou(gt_boxes, pred_boxes)
    candidates: List[Tuple[float, float, int, int]] = []
    for gt_idx in range(gt_boxes.shape[0]):
        for pred_idx in range(pred_boxes.shape[0]):
            iou = float(ious[gt_idx, pred_idx].item())
            if iou < iou_thr:
                continue
            if require_same_label and int(gt_labels[gt_idx].item()) != int(pred_labels[pred_idx].item()):
                continue
            score = float(pred_scores[pred_idx].item())
            candidates.append((score, iou, gt_idx, pred_idx))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)

    used_pred = set()
    for _, _, gt_idx, pred_idx in candidates:
        if matched_pred_for_gt[gt_idx] != -1:
            continue
        if pred_idx in used_pred:
            continue
        matched_pred_for_gt[gt_idx] = pred_idx
        used_pred.add(pred_idx)

    return matched_pred_for_gt


def _is_better_candidate(
    score: float,
    iou: float,
    best_score: float,
    best_iou: float,
) -> bool:
    return (score, iou) > (best_score, best_iou)


@torch.no_grad()
def _overlap_diagnostics(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    pred_boxes: torch.Tensor,
    pred_labels: torch.Tensor,
    pred_scores: torch.Tensor,
    iou_thr: float,
) -> Dict[str, List]:
    """Find the best overlapping predictions per GT without global assignment.

    This is intentionally not one-to-one. It is diagnostic evidence for why an
    already-unmatched GT was missed, so a prediction should not disappear just
    because it was assigned to a different GT in another matching pass.
    """
    num_gt = int(gt_boxes.shape[0])
    result = {
        "best_any": [-1] * num_gt,
        "best_same": [-1] * num_gt,
        "best_wrong": [-1] * num_gt,
        "best_any_iou": [0.0] * num_gt,
        "best_same_iou": [0.0] * num_gt,
        "best_wrong_iou": [0.0] * num_gt,
    }
    if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
        return result

    ious, _ = box_iou(gt_boxes, pred_boxes)
    best_any_score = [-1.0] * num_gt
    best_same_score = [-1.0] * num_gt
    best_wrong_score = [-1.0] * num_gt

    for gt_idx in range(num_gt):
        gt_class = int(gt_labels[gt_idx].item())
        for pred_idx in range(int(pred_boxes.shape[0])):
            iou = float(ious[gt_idx, pred_idx].item())
            if iou < iou_thr:
                continue

            score = float(pred_scores[pred_idx].item())
            pred_class = int(pred_labels[pred_idx].item())

            if _is_better_candidate(score, iou, best_any_score[gt_idx], result["best_any_iou"][gt_idx]):
                result["best_any"][gt_idx] = pred_idx
                result["best_any_iou"][gt_idx] = iou
                best_any_score[gt_idx] = score

            if pred_class == gt_class:
                if _is_better_candidate(score, iou, best_same_score[gt_idx], result["best_same_iou"][gt_idx]):
                    result["best_same"][gt_idx] = pred_idx
                    result["best_same_iou"][gt_idx] = iou
                    best_same_score[gt_idx] = score
            else:
                if _is_better_candidate(score, iou, best_wrong_score[gt_idx], result["best_wrong_iou"][gt_idx]):
                    result["best_wrong"][gt_idx] = pred_idx
                    result["best_wrong_iou"][gt_idx] = iou
                    best_wrong_score[gt_idx] = score

    return result


def _resolve_save_dir(args) -> Path:
    if args.save_dir:
        return Path(args.save_dir)
    ckpt_stem = Path(args.resume).stem
    ckpt_parent = Path(args.resume).resolve().parent
    return ckpt_parent / "analysis" / f"missed_detections_{ckpt_stem}"


def _write_csv(path: Path, fieldnames: List[str], rows: List[Dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _bump(stats, key: str, value: int = 1) -> None:
    stats[key] += value


def _rate(numerator: float, denominator: float) -> float:
    return round(float(numerator) / max(float(denominator), 1.0), 6)


def _csv_int(row: Dict, key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return int(float(value))


def _csv_float(row: Dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in (None, ""):
        return default
    return float(value)


def _load_confusion_baseline(path: str) -> Dict[Tuple[int, int], Dict]:
    if not path:
        return {}

    baseline = {}
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (_csv_int(row, "gt_class_id"), _csv_int(row, "pred_class_id"))
            baseline[key] = row
    return baseline


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--score-thr", type=float, default=0.3)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--max-images", type=int, default=0, help="0 means all images")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--save-dir", type=str, default="")
    parser.add_argument(
        "--compare-confusions",
        type=str,
        default="",
        help="Optional previous confusions.csv to compute confusion count/rate deltas",
    )
    args = parser.parse_args()

    save_dir = _resolve_save_dir(args)
    save_dir.mkdir(parents=True, exist_ok=True)
    baseline_confusions = _load_confusion_baseline(args.compare_confusions)

    device = torch.device(args.device)
    cfg = YAMLConfig(args.config)

    model = cfg.model
    state = _load_state_dict_from_ckpt(args.resume)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    postprocessor = cfg.postprocessor
    postprocessor.eval()

    data_loader = cfg.val_dataloader
    dataset = _unwrap_dataset(data_loader.dataset)
    coco = dataset.coco

    use_category_ids = bool(getattr(postprocessor, "remap_mscoco_category", False))
    category2label = getattr(dataset, "category2label", {})
    category2name = getattr(dataset, "category2name", {})
    if use_category_ids:
        class_name_lookup = {int(k): v for k, v in category2name.items()}
    else:
        label2category = getattr(dataset, "label2category", {})
        class_name_lookup = {
            int(label): category2name.get(int(category_id), str(label))
            for label, category_id in label2category.items()
        }

    summary = {
        "config": args.config,
        "checkpoint": args.resume,
        "device": args.device,
        "score_thr": float(args.score_thr),
        "iou_thr": float(args.iou_thr),
        "images_processed": 0,
        "total_gt": 0,
        "matched_gt": 0,
        "missed_gt": 0,
        "pure_missed_gt": 0,
        "confused_gt": 0,
        "same_class_unmatched_gt": 0,
    }

    size_stats = {
        "small": defaultdict(int),
        "medium": defaultdict(int),
        "large": defaultdict(int),
    }
    per_class = defaultdict(lambda: defaultdict(int))
    confusion_counter = Counter()
    image_rows: List[Dict] = []

    for _, (samples, targets) in enumerate(data_loader):
        if args.max_images and summary["images_processed"] >= args.max_images:
            break

        samples = samples.to(device)
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0).to(device)

        outputs = model(samples)
        results = postprocessor(outputs, orig_target_sizes)

        batch_size = len(results)
        for i in range(batch_size):
            if args.max_images and summary["images_processed"] >= args.max_images:
                break

            image_id = _safe_int(targets[i]["image_id"])
            gt_boxes, gt_labels, gt_areas, _ = _load_gt_for_image(
                coco,
                image_id=image_id,
                use_category_ids=use_category_ids,
                category2label=category2label,
            )

            pred_boxes = results[i]["boxes"].detach().cpu()
            pred_labels = results[i]["labels"].detach().cpu().to(torch.int64)
            pred_scores = results[i]["scores"].detach().cpu()

            keep = pred_scores >= float(args.score_thr)
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            pred_scores = pred_scores[keep]

            class_aware_matches = _greedy_match(
                gt_boxes,
                gt_labels,
                pred_boxes,
                pred_labels,
                pred_scores,
                iou_thr=float(args.iou_thr),
                require_same_label=True,
            )
            diagnostics = _overlap_diagnostics(
                gt_boxes,
                gt_labels,
                pred_boxes,
                pred_labels,
                pred_scores,
                iou_thr=float(args.iou_thr),
            )

            image_stats = defaultdict(int)

            for gt_idx in range(int(gt_boxes.shape[0])):
                class_id = int(gt_labels[gt_idx].item())
                bucket = _area_bucket(float(gt_areas[gt_idx].item()))

                _bump(summary, "total_gt")
                _bump(size_stats[bucket], "total_gt")
                _bump(per_class[class_id], "total_gt")
                _bump(image_stats, "total_gt")

                if class_aware_matches[gt_idx] != -1:
                    _bump(summary, "matched_gt")
                    _bump(size_stats[bucket], "matched_gt")
                    _bump(per_class[class_id], "matched_gt")
                    _bump(image_stats, "matched_gt")
                    continue

                _bump(summary, "missed_gt")
                _bump(size_stats[bucket], "missed_gt")
                _bump(per_class[class_id], "missed_gt")
                _bump(image_stats, "missed_gt")

                same_pred_idx = diagnostics["best_same"][gt_idx]
                wrong_pred_idx = diagnostics["best_wrong"][gt_idx]

                if same_pred_idx != -1:
                    _bump(summary, "same_class_unmatched_gt")
                    _bump(size_stats[bucket], "same_class_unmatched_gt")
                    _bump(per_class[class_id], "same_class_unmatched_gt")
                    _bump(image_stats, "same_class_unmatched_gt")
                elif wrong_pred_idx != -1:
                    pred_class_id = int(pred_labels[wrong_pred_idx].item())
                    _bump(summary, "confused_gt")
                    _bump(size_stats[bucket], "confused_gt")
                    _bump(per_class[class_id], "confused_gt")
                    _bump(image_stats, "confused_gt")
                    confusion_counter[(class_id, pred_class_id)] += 1
                else:
                    _bump(summary, "pure_missed_gt")
                    _bump(size_stats[bucket], "pure_missed_gt")
                    _bump(per_class[class_id], "pure_missed_gt")
                    _bump(image_stats, "pure_missed_gt")

            img_info = coco.imgs.get(image_id, {})
            total_gt = int(image_stats.get("total_gt", 0))
            missed_gt = int(image_stats.get("missed_gt", 0))
            image_rows.append(
                {
                    "image_id": image_id,
                    "file_name": img_info.get("file_name", str(image_id)),
                    "total_gt": total_gt,
                    "matched_gt": int(image_stats.get("matched_gt", 0)),
                    "missed_gt": missed_gt,
                    "pure_missed_gt": int(image_stats.get("pure_missed_gt", 0)),
                    "confused_gt": int(image_stats.get("confused_gt", 0)),
                    "same_class_unmatched_gt": int(image_stats.get("same_class_unmatched_gt", 0)),
                    "miss_rate": round(missed_gt / max(total_gt, 1), 6),
                }
            )
            summary["images_processed"] += 1

    summary["miss_rate"] = summary["missed_gt"] / max(summary["total_gt"], 1)
    summary["pure_miss_rate"] = summary["pure_missed_gt"] / max(summary["total_gt"], 1)
    summary["confusion_miss_rate"] = summary["confused_gt"] / max(summary["total_gt"], 1)
    summary["same_class_unmatched_rate"] = summary["same_class_unmatched_gt"] / max(summary["total_gt"], 1)
    summary["size_breakdown"] = {}
    for bucket, bucket_stats in size_stats.items():
        total_gt = int(bucket_stats.get("total_gt", 0))
        summary["size_breakdown"][bucket] = {
            "total_gt": total_gt,
            "matched_gt": int(bucket_stats.get("matched_gt", 0)),
            "missed_gt": int(bucket_stats.get("missed_gt", 0)),
            "pure_missed_gt": int(bucket_stats.get("pure_missed_gt", 0)),
            "confused_gt": int(bucket_stats.get("confused_gt", 0)),
            "same_class_unmatched_gt": int(bucket_stats.get("same_class_unmatched_gt", 0)),
            "miss_rate": (bucket_stats.get("missed_gt", 0) / total_gt) if total_gt else 0.0,
            "pure_miss_rate": (bucket_stats.get("pure_missed_gt", 0) / total_gt) if total_gt else 0.0,
            "confusion_miss_rate": (bucket_stats.get("confused_gt", 0) / total_gt) if total_gt else 0.0,
            "same_class_unmatched_rate": (
                bucket_stats.get("same_class_unmatched_gt", 0) / total_gt
            ) if total_gt else 0.0,
        }

    per_class_rows = []
    for class_id, stats in per_class.items():
        total_gt = int(stats.get("total_gt", 0))
        missed_gt = int(stats.get("missed_gt", 0))
        pure_missed_gt = int(stats.get("pure_missed_gt", 0))
        confused_gt = int(stats.get("confused_gt", 0))
        same_class_unmatched_gt = int(stats.get("same_class_unmatched_gt", 0))
        row = {
            "class_id": class_id,
            "class_name": class_name_lookup.get(class_id, str(class_id)),
            "total_gt": total_gt,
            "matched_gt": int(stats.get("matched_gt", 0)),
            "missed_gt": missed_gt,
            "pure_missed_gt": pure_missed_gt,
            "confused_gt": confused_gt,
            "same_class_unmatched_gt": same_class_unmatched_gt,
            "miss_rate": _rate(missed_gt, total_gt),
            "pure_miss_rate": _rate(pure_missed_gt, total_gt),
            "confusion_rate_total_gt": _rate(confused_gt, total_gt),
            "confusion_share_in_misses": _rate(confused_gt, missed_gt),
            "same_class_unmatched_rate": _rate(same_class_unmatched_gt, total_gt),
        }
        per_class_rows.append(row)

    per_class_rows.sort(key=lambda row: (row["missed_gt"], row["miss_rate"], row["total_gt"]), reverse=True)

    confusion_rows = []
    for (gt_class, pred_class), count in confusion_counter.most_common():
        gt_stats = per_class[gt_class]
        gt_total = int(gt_stats.get("total_gt", 0))
        gt_missed = int(gt_stats.get("missed_gt", 0))
        gt_confused = int(gt_stats.get("confused_gt", 0))
        row = {
            "gt_class_id": gt_class,
            "gt_class_name": class_name_lookup.get(gt_class, str(gt_class)),
            "pred_class_id": pred_class,
            "pred_class_name": class_name_lookup.get(pred_class, str(pred_class)),
            "count": count,
            "gt_total_gt": gt_total,
            "gt_missed_gt": gt_missed,
            "gt_confused_gt": gt_confused,
            "confusion_rate_total_gt": _rate(count, gt_total),
            "confusion_rate_missed_gt": _rate(count, gt_missed),
            "share_of_gt_confusions": _rate(count, gt_confused),
            "share_of_all_confusions": _rate(count, summary["confused_gt"]),
        }
        if baseline_confusions:
            base = baseline_confusions.get((gt_class, pred_class), {})
            base_count = _csv_int(base, "count")
            base_gt_total = _csv_float(base, "gt_total_gt")
            base_gt_missed = _csv_float(base, "gt_missed_gt")
            base_gt_confused = _csv_float(base, "gt_confused_gt")
            base_rate_total = _csv_float(
                base,
                "confusion_rate_total_gt",
                _rate(base_count, base_gt_total) if base_gt_total else 0.0,
            )
            base_rate_missed = _csv_float(
                base,
                "confusion_rate_missed_gt",
                _rate(base_count, base_gt_missed) if base_gt_missed else 0.0,
            )
            base_share = _csv_float(
                base,
                "share_of_gt_confusions",
                _rate(base_count, base_gt_confused) if base_gt_confused else 0.0,
            )
            row.update(
                {
                    "baseline_count": base_count,
                    "delta_count": int(count) - base_count,
                    "baseline_confusion_rate_total_gt": base_rate_total,
                    "delta_confusion_rate_total_gt": round(row["confusion_rate_total_gt"] - base_rate_total, 6),
                    "baseline_confusion_rate_missed_gt": base_rate_missed,
                    "delta_confusion_rate_missed_gt": round(row["confusion_rate_missed_gt"] - base_rate_missed, 6),
                    "baseline_share_of_gt_confusions": base_share,
                    "delta_share_of_gt_confusions": round(row["share_of_gt_confusions"] - base_share, 6),
                }
            )
        confusion_rows.append(row)

    confusion_change_rows = []
    if baseline_confusions:
        confusion_change_rows = [dict(row) for row in confusion_rows]
        current_keys = set(confusion_counter.keys())
        for (gt_class, pred_class), base in baseline_confusions.items():
            if (gt_class, pred_class) in current_keys:
                continue
            base_count = _csv_int(base, "count")
            base_gt_total = _csv_float(base, "gt_total_gt")
            base_gt_missed = _csv_float(base, "gt_missed_gt")
            base_gt_confused = _csv_float(base, "gt_confused_gt")
            base_rate_total = _csv_float(
                base,
                "confusion_rate_total_gt",
                _rate(base_count, base_gt_total) if base_gt_total else 0.0,
            )
            base_rate_missed = _csv_float(
                base,
                "confusion_rate_missed_gt",
                _rate(base_count, base_gt_missed) if base_gt_missed else 0.0,
            )
            base_share = _csv_float(
                base,
                "share_of_gt_confusions",
                _rate(base_count, base_gt_confused) if base_gt_confused else 0.0,
            )
            gt_stats = per_class[gt_class]
            gt_total = int(gt_stats.get("total_gt", 0))
            gt_missed = int(gt_stats.get("missed_gt", 0))
            gt_confused = int(gt_stats.get("confused_gt", 0))
            confusion_change_rows.append(
                {
                    "gt_class_id": gt_class,
                    "gt_class_name": class_name_lookup.get(gt_class, base.get("gt_class_name", str(gt_class))),
                    "pred_class_id": pred_class,
                    "pred_class_name": class_name_lookup.get(pred_class, base.get("pred_class_name", str(pred_class))),
                    "count": 0,
                    "gt_total_gt": gt_total,
                    "gt_missed_gt": gt_missed,
                    "gt_confused_gt": gt_confused,
                    "confusion_rate_total_gt": 0.0,
                    "confusion_rate_missed_gt": 0.0,
                    "share_of_gt_confusions": 0.0,
                    "share_of_all_confusions": 0.0,
                    "baseline_count": base_count,
                    "delta_count": -base_count,
                    "baseline_confusion_rate_total_gt": base_rate_total,
                    "delta_confusion_rate_total_gt": round(-base_rate_total, 6),
                    "baseline_confusion_rate_missed_gt": base_rate_missed,
                    "delta_confusion_rate_missed_gt": round(-base_rate_missed, 6),
                    "baseline_share_of_gt_confusions": base_share,
                    "delta_share_of_gt_confusions": round(-base_share, 6),
                }
            )
        confusion_change_rows.sort(
            key=lambda row: (
                abs(row["delta_confusion_rate_total_gt"]),
                abs(row["delta_count"]),
                row["count"],
            ),
            reverse=True,
        )

    image_rows.sort(key=lambda row: (row["missed_gt"], row["miss_rate"], row["total_gt"]), reverse=True)

    summary_path = save_dir / "summary.json"
    per_class_path = save_dir / "per_class.csv"
    confusions_path = save_dir / "confusions.csv"
    confusion_changes_path = save_dir / "confusion_changes.csv"
    worst_images_path = save_dir / "worst_images.csv"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "top_missed_classes": per_class_rows[: int(args.topk)],
                "top_confusions": confusion_rows[: int(args.topk)],
                "top_confusion_changes": confusion_change_rows[: int(args.topk)],
                "worst_images": image_rows[: int(args.topk)],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    _write_csv(
        per_class_path,
        fieldnames=[
            "class_id",
            "class_name",
            "total_gt",
            "matched_gt",
            "missed_gt",
            "pure_missed_gt",
            "confused_gt",
            "same_class_unmatched_gt",
            "miss_rate",
            "pure_miss_rate",
            "confusion_rate_total_gt",
            "confusion_share_in_misses",
            "same_class_unmatched_rate",
        ],
        rows=per_class_rows,
    )
    confusion_fieldnames = [
        "gt_class_id",
        "gt_class_name",
        "pred_class_id",
        "pred_class_name",
        "count",
        "gt_total_gt",
        "gt_missed_gt",
        "gt_confused_gt",
        "confusion_rate_total_gt",
        "confusion_rate_missed_gt",
        "share_of_gt_confusions",
        "share_of_all_confusions",
    ]
    if baseline_confusions:
        confusion_fieldnames.extend(
            [
                "baseline_count",
                "delta_count",
                "baseline_confusion_rate_total_gt",
                "delta_confusion_rate_total_gt",
                "baseline_confusion_rate_missed_gt",
                "delta_confusion_rate_missed_gt",
                "baseline_share_of_gt_confusions",
                "delta_share_of_gt_confusions",
            ]
        )

    _write_csv(confusions_path, fieldnames=confusion_fieldnames, rows=confusion_rows)
    if baseline_confusions:
        _write_csv(confusion_changes_path, fieldnames=confusion_fieldnames, rows=confusion_change_rows)
    _write_csv(
        worst_images_path,
        fieldnames=[
            "image_id",
            "file_name",
            "total_gt",
            "matched_gt",
            "missed_gt",
            "pure_missed_gt",
            "confused_gt",
            "same_class_unmatched_gt",
            "miss_rate",
        ],
        rows=image_rows,
    )

    print("=== Missed Detection Summary ===")
    print(f"images_processed: {summary['images_processed']}")
    print(f"total_gt: {summary['total_gt']}")
    print(f"matched_gt: {summary['matched_gt']}")
    print(f"missed_gt: {summary['missed_gt']}")
    print(f"miss_rate: {summary['miss_rate']:.4%}")
    print(f"pure_missed_gt: {summary['pure_missed_gt']}")
    print(f"pure_miss_rate: {summary['pure_miss_rate']:.4%}")
    print(f"confused_gt: {summary['confused_gt']}")
    print(f"confusion_miss_rate: {summary['confusion_miss_rate']:.4%}")
    print(f"same_class_unmatched_gt: {summary['same_class_unmatched_gt']}")
    print(f"same_class_unmatched_rate: {summary['same_class_unmatched_rate']:.4%}")

    print("\nSize breakdown:")
    for bucket, stats in summary["size_breakdown"].items():
        print(
            f"  {bucket:<6} total={stats['total_gt']:<6} missed={stats['missed_gt']:<6} "
            f"pure={stats['pure_missed_gt']:<6} confused={stats['confused_gt']:<6} "
            f"same_cls_unmatched={stats['same_class_unmatched_gt']:<6} "
            f"miss_rate={stats['miss_rate']:.4%}"
        )

    print("\nTop missed classes:")
    for row in per_class_rows[: int(args.topk)]:
        print(
            f"  {row['class_id']:>3} {row['class_name']:<20} "
            f"missed={row['missed_gt']:<6} total={row['total_gt']:<6} rate={row['miss_rate']:.4%} "
            f"pure={row['pure_missed_gt']:<6} confused={row['confused_gt']:<6} "
            f"conf_rate={row['confusion_rate_total_gt']:.4%} "
            f"same_cls_unmatched={row['same_class_unmatched_gt']:<6}"
        )

    if confusion_rows:
        print("\nTop confusion-driven misses:")
        for row in confusion_rows[: int(args.topk)]:
            delta = ""
            if baseline_confusions:
                delta = (
                    f" delta_count={row['delta_count']:+d} "
                    f"delta_rate={row['delta_confusion_rate_total_gt']:+.4%}"
                )
            print(
                f"  {row['gt_class_name']} -> {row['pred_class_name']} : {row['count']} "
                f"rate_total={row['confusion_rate_total_gt']:.4%} "
                f"rate_missed={row['confusion_rate_missed_gt']:.4%} "
                f"share_in_gt_confusions={row['share_of_gt_confusions']:.4%}"
                f"{delta}"
            )

    if confusion_change_rows:
        print("\nTop confusion changes:")
        for row in confusion_change_rows[: int(args.topk)]:
            print(
                f"  {row['gt_class_name']} -> {row['pred_class_name']} : "
                f"count {row['baseline_count']} -> {row['count']} ({row['delta_count']:+d}), "
                f"rate_total {row['baseline_confusion_rate_total_gt']:.4%} -> "
                f"{row['confusion_rate_total_gt']:.4%} "
                f"({row['delta_confusion_rate_total_gt']:+.4%})"
            )

    print("\nSaved files:")
    print(f"  {summary_path}")
    print(f"  {per_class_path}")
    print(f"  {confusions_path}")
    if baseline_confusions:
        print(f"  {confusion_changes_path}")
    print(f"  {worst_images_path}")


if __name__ == "__main__":
    main()
