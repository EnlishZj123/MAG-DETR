"""Build a COCO2017 complex-scene validation subset.

This script ranks COCO val2017 images by a hand-crafted complexity score that
favors crowded, small-object-heavy, overlapped, multi-category, and crowd-tagged
scenes. It then copies or hard-links the top-K images into a new COCO-style
dataset layout without changing training data.

Expected source layout:

    <src-root>/
      val2017/
      annotations/instances_val2017.json

Output layout:

    <dst-root>/
      val2017/
      annotations/instances_val2017.json

Typical usage:

    python tools/prepare_coco2017_complex_val.py \
        --src-root /path/to/COCO2017 \
        --dst-root data/coco2017-complex \
        --top-k 1000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def box_iou_xywh(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, aw, ah = [float(v) for v in box_a]
    bx1, by1, bw, bh = [float(v) for v in box_b]
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _valid_annotations(annotations: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    valid = []
    for ann in annotations:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        width = float(bbox[2])
        height = float(bbox[3])
        if width <= 1.0 or height <= 1.0:
            continue
        valid.append(ann)
    return valid


def _count_overlap_pairs(boxes: List[List[float]], iou_thr: float) -> int:
    overlap_pairs = 0
    for idx in range(len(boxes)):
        for jdx in range(idx + 1, len(boxes)):
            if box_iou_xywh(boxes[idx], boxes[jdx]) > iou_thr:
                overlap_pairs += 1
    return overlap_pairs


def compute_complexity_metrics(annotations: List[Dict[str, Any]], small_area_thr: float, iou_thr: float) -> Dict[str, Any]:
    valid = _valid_annotations(annotations)
    if not valid:
        return {
            "num_instances": 0,
            "num_categories": 0,
            "small_count": 0,
            "small_ratio": 0.0,
            "crowd_count": 0,
            "overlap_pairs": 0,
            "pair_count": 0,
            "overlap_ratio": 0.0,
            "score": 0.0,
        }

    num_instances = len(valid)
    num_categories = len({int(ann["category_id"]) for ann in valid})
    small_count = sum(1 for ann in valid if float(ann["bbox"][2]) * float(ann["bbox"][3]) < small_area_thr)
    crowd_count = sum(1 for ann in valid if int(ann.get("iscrowd", 0)) == 1)
    boxes = [ann["bbox"] for ann in valid]
    overlap_pairs = _count_overlap_pairs(boxes, iou_thr)
    pair_count = math.comb(num_instances, 2) if num_instances >= 2 else 0
    overlap_ratio = overlap_pairs / pair_count if pair_count > 0 else 0.0
    small_ratio = small_count / num_instances

    instance_score = min(num_instances / 20.0, 1.0) * 0.35
    small_score = min(small_ratio, 1.0) * 0.25
    overlap_score = min(overlap_pairs / 10.0, 1.0) * 0.25
    category_score = min(num_categories / 6.0, 1.0) * 0.10
    crowd_score = min(crowd_count / 3.0, 1.0) * 0.05
    score = instance_score + small_score + overlap_score + category_score + crowd_score

    return {
        "num_instances": num_instances,
        "num_categories": num_categories,
        "small_count": small_count,
        "small_ratio": small_ratio,
        "crowd_count": crowd_count,
        "overlap_pairs": overlap_pairs,
        "pair_count": pair_count,
        "overlap_ratio": overlap_ratio,
        "score": score,
    }


def _build_selection_rank(
    images: List[Dict[str, Any]],
    anns_by_img: Dict[int, List[Dict[str, Any]]],
    small_area_thr: float,
    iou_thr: float,
) -> List[Dict[str, Any]]:
    ranked = []
    for image in images:
        image_id = int(image["id"])
        metrics = compute_complexity_metrics(anns_by_img.get(image_id, []), small_area_thr, iou_thr)
        ranked.append(
            {
                "image_id": image_id,
                "file_name": image["file_name"],
                "width": int(image.get("width", 0) or 0),
                "height": int(image.get("height", 0) or 0),
                **metrics,
            }
        )

    ranked.sort(
        key=lambda item: (
            item["score"],
            item["num_instances"],
            item["small_ratio"],
            item["overlap_pairs"],
            item["num_categories"],
            item["crowd_count"],
            -item["image_id"],
        ),
        reverse=True,
    )
    return ranked


def _clean_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _resolve_source_root(src_root: Path) -> Path:
    if src_root.exists():
        return src_root

    fallback_candidates = [Path("data/COCO2017"), Path("data/coco2017")]
    for candidate in fallback_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Source COCO root not found: {src_root}. Tried fallback roots: {', '.join(str(p) for p in fallback_candidates)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a COCO2017 complex-scene validation subset.")
    parser.add_argument("--src-root", type=str, default="data/COCO2017", help="Source COCO2017 root directory")
    parser.add_argument("--dst-root", type=str, default="data/coco2017-complex", help="Output root directory")
    parser.add_argument("--split", type=str, default="val2017", help="Image split name to sample from")
    parser.add_argument(
        "--ann-file",
        type=str,
        default="annotations/instances_val2017.json",
        help="Source annotation JSON relative to --src-root or an absolute path",
    )
    parser.add_argument("--top-k", type=int, default=1000, help="Number of images to keep")
    parser.add_argument("--small-area-thr", type=float, default=32.0 * 32.0, help="Area threshold for small objects")
    parser.add_argument("--iou-thr", type=float, default=0.1, help="IoU threshold for overlap pairs")
    parser.add_argument(
        "--report-name",
        type=str,
        default="complexity_ranking.json",
        help="Filename for the ranking report saved under the output annotations directory",
    )
    return parser.parse_args()


def _resolve_annotation_path(src_root: Path, ann_file: str) -> Path:
    ann_path = Path(ann_file)
    if ann_path.is_absolute():
        return ann_path
    return src_root / ann_path


def main() -> None:
    args = parse_args()

    src_root = _resolve_source_root(Path(args.src_root))
    dst_root = Path(args.dst_root)
    if src_root.resolve() == dst_root.resolve():
        raise ValueError("--dst-root must be different from --src-root to avoid deleting the source dataset")
    split_dir = src_root / args.split
    ann_path = _resolve_annotation_path(src_root, args.ann_file)

    if not split_dir.exists():
        raise FileNotFoundError(f"Missing source image folder: {split_dir}")
    if not ann_path.exists():
        raise FileNotFoundError(f"Missing source annotation file: {ann_path}")

    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    anns_by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_img[int(ann["image_id"])].append(ann)

    ranked = _build_selection_rank(
        coco.get("images", []),
        anns_by_img,
        small_area_thr=float(args.small_area_thr),
        iou_thr=float(args.iou_thr),
    )

    if not ranked:
        raise RuntimeError("No valid COCO images found to rank.")

    top_k = max(1, min(int(args.top_k), len(ranked)))
    selected = ranked[:top_k]
    selected_ids = {int(item["image_id"]) for item in selected}

    new_images = [img for img in coco.get("images", []) if int(img["id"]) in selected_ids]
    new_annotations = [ann for ann in coco.get("annotations", []) if int(ann["image_id"]) in selected_ids]

    out_img_dir = dst_root / args.split
    out_ann_dir = dst_root / "annotations"
    _clean_output_dir(out_img_dir)
    out_ann_dir.mkdir(parents=True, exist_ok=True)

    for image in new_images:
        src_img = split_dir / image["file_name"]
        dst_img = out_img_dir / image["file_name"]
        if not src_img.exists():
            raise FileNotFoundError(f"Missing source image file: {src_img}")
        link_or_copy(src_img, dst_img)

    out_ann = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco.get("categories", []),
    }

    out_ann_path = out_ann_dir / Path(args.ann_file).name
    with out_ann_path.open("w", encoding="utf-8") as f:
        json.dump(out_ann, f, ensure_ascii=False)

    report_path = out_ann_dir / args.report_name
    report = {
        "source_root": str(src_root),
        "source_annotation": str(ann_path),
        "output_root": str(dst_root),
        "split": args.split,
        "top_k": top_k,
        "small_area_thr": float(args.small_area_thr),
        "iou_thr": float(args.iou_thr),
        "selected_count": len(selected),
        "ranking": ranked,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    score_values = [item["score"] for item in selected]
    score_min = min(score_values) if score_values else 0.0
    score_max = max(score_values) if score_values else 0.0
    score_mean = sum(score_values) / len(score_values) if score_values else 0.0

    print(f"source: {src_root}")
    print(f"annotation: {ann_path}")
    print(f"output: {dst_root}")
    print(f"selected images: {len(new_images)}")
    print(f"selected annotations: {len(new_annotations)}")
    print(f"score range: [{score_min:.4f}, {score_max:.4f}], mean={score_mean:.4f}")
    print(f"saved annotation: {out_ann_path}")
    print(f"saved ranking report: {report_path}")


if __name__ == "__main__":
    main()