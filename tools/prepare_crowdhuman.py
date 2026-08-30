"""Convert CrowdHuman val annotations to COCO-style JSON.

This utility does not change the detector. It only rewrites CrowdHuman ground
truth into a COCO val annotation file that contains a single `person` category
and uses the full-body `fbox` box for each valid person instance.

Expected layout:

    data/crowdhuman/
      annotation_val.odgt
      val2017/

Output:

    data/crowdhuman/annotations/instances_val2017.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_odgt(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _as_int_image_id(record: Dict[str, Any], index: int) -> int:
    image_id = record.get("ID", record.get("id", index))
    try:
        return int(image_id)
    except (TypeError, ValueError):
        return index


def _as_file_name(record: Dict[str, Any], image_id: int) -> str:
    file_name = record.get("file_name")
    if file_name:
        return file_name
    return f"{record.get('ID', image_id)}.jpg"


def _bbox_area(bbox: List[float]) -> float:
    return float(max(bbox[2], 0.0) * max(bbox[3], 0.0))


def convert_odgt_to_coco(odgt_path: Path, image_root: Path, output_json: Path) -> None:
    records = _load_odgt(odgt_path)

    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    ann_id = 1

    for index, record in enumerate(records, start=1):
        image_id = _as_int_image_id(record, index)
        file_name = _as_file_name(record, image_id)
        width = int(record.get("width", 0) or 0)
        height = int(record.get("height", 0) or 0)

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
        )

        for gt in record.get("gtboxes", []):
            if gt.get("tag", "person") != "person":
                continue

            box = gt.get("fbox")
            if box is None:
                continue

            x, y, w, h = [float(v) for v in box]
            if w <= 0 or h <= 0:
                continue

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": _bbox_area([x, y, w, h]),
                    "iscrowd": int(gt.get("extra", {}).get("ignore", 0)),
                    "segmentation": [],
                }
            )
            ann_id += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person", "supercategory": "person"}],
    }
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"Wrote {output_json} with {len(images)} images and {len(annotations)} annotations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert CrowdHuman val annotations to COCO JSON")
    parser.add_argument("--data-root", default="data/crowdhuman", help="CrowdHuman dataset root")
    parser.add_argument("--split", default="val", choices=["val"], help="Split to convert")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    odgt_path = data_root / f"annotation_{args.split}.odgt"
    image_root = data_root / f"{args.split}2017"
    output_json = data_root / "annotations" / f"instances_{args.split}2017.json"

    if not odgt_path.exists():
        raise FileNotFoundError(f"Missing CrowdHuman annotation file: {odgt_path}")
    if not image_root.exists():
        raise FileNotFoundError(f"Missing CrowdHuman image folder: {image_root}")

    convert_odgt_to_coco(odgt_path, image_root, output_json)


if __name__ == "__main__":
    main()