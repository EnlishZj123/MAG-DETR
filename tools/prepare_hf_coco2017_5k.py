import argparse
import json
import os
from datetime import datetime


def coco_categories():
    # Standard COCO 80 categories (ids are the official MSCOCO ids)
    categories = [
        (1, "person"),
        (2, "bicycle"),
        (3, "car"),
        (4, "motorcycle"),
        (5, "airplane"),
        (6, "bus"),
        (7, "train"),
        (8, "truck"),
        (9, "boat"),
        (10, "traffic light"),
        (11, "fire hydrant"),
        (13, "stop sign"),
        (14, "parking meter"),
        (15, "bench"),
        (16, "bird"),
        (17, "cat"),
        (18, "dog"),
        (19, "horse"),
        (20, "sheep"),
        (21, "cow"),
        (22, "elephant"),
        (23, "bear"),
        (24, "zebra"),
        (25, "giraffe"),
        (27, "backpack"),
        (28, "umbrella"),
        (31, "handbag"),
        (32, "tie"),
        (33, "suitcase"),
        (34, "frisbee"),
        (35, "skis"),
        (36, "snowboard"),
        (37, "sports ball"),
        (38, "kite"),
        (39, "baseball bat"),
        (40, "baseball glove"),
        (41, "skateboard"),
        (42, "surfboard"),
        (43, "tennis racket"),
        (44, "bottle"),
        (46, "wine glass"),
        (47, "cup"),
        (48, "fork"),
        (49, "knife"),
        (50, "spoon"),
        (51, "bowl"),
        (52, "banana"),
        (53, "apple"),
        (54, "sandwich"),
        (55, "orange"),
        (56, "broccoli"),
        (57, "carrot"),
        (58, "hot dog"),
        (59, "pizza"),
        (60, "donut"),
        (61, "cake"),
        (62, "chair"),
        (63, "couch"),
        (64, "potted plant"),
        (65, "bed"),
        (67, "dining table"),
        (70, "toilet"),
        (72, "tv"),
        (73, "laptop"),
        (74, "mouse"),
        (75, "remote"),
        (76, "keyboard"),
        (77, "cell phone"),
        (78, "microwave"),
        (79, "oven"),
        (80, "toaster"),
        (81, "sink"),
        (82, "refrigerator"),
        (84, "book"),
        (85, "clock"),
        (86, "vase"),
        (87, "scissors"),
        (88, "teddy bear"),
        (89, "hair drier"),
        (90, "toothbrush"),
    ]
    return [{"id": cid, "name": name, "supercategory": "none"} for cid, name in categories]


def build_name_to_id():
    return {c["name"]: c["id"] for c in coco_categories()}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def export_split(ds_split, split_name: str, out_images_dir: str, out_ann_path: str, image_id_offset: int = 0) -> int:
    name_to_id = build_name_to_id()

    cat_feature = None
    try:
        cat_feature = ds_split.features["objects"]["category"]
        # Usually this is Sequence(ClassLabel). Unwrap to the underlying ClassLabel.
        if hasattr(cat_feature, "feature"):
            cat_feature = cat_feature.feature
    except Exception:
        cat_feature = None

    images = []
    annotations = []

    ann_id = 1
    for idx in range(len(ds_split)):
        ex = ds_split[idx]
        img = ex["image"]
        width, height = img.size

        image_id = image_id_offset + idx + 1
        file_name = f"{image_id:012d}.jpg"
        img_path = os.path.join(out_images_dir, file_name)
        img.save(img_path, format="JPEG", quality=95)

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": int(width),
                "height": int(height),
            }
        )

        objects = ex.get("objects", {})
        bboxes = objects.get("bbox", []) or []
        categories = objects.get("category", []) or []

        if len(bboxes) != len(categories):
            raise ValueError(
                f"[{split_name}] Example {idx}: bbox count {len(bboxes)} != category count {len(categories)}"
            )

        for bbox, cat in zip(bboxes, categories):
            if bbox is None or len(bbox) != 4:
                continue

            x, y, w, h = [float(v) for v in bbox]
            if w <= 0 or h <= 0:
                continue

            # 'category' is a ClassLabel; in examples it materializes as an int.
            if isinstance(cat, str):
                cat_name = cat
            elif cat_feature is not None and hasattr(cat_feature, "int2str"):
                cat_name = cat_feature.int2str(int(cat))
            else:
                cat_name = str(cat)

            if cat_name not in name_to_id:
                raise ValueError(f"Unknown category name '{cat_name}'")

            category_id = int(name_to_id[cat_name])
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        if (idx + 1) % 500 == 0:
            print(f"[{split_name}] exported {idx + 1}/{len(ds_split)}")

    coco = {
        "info": {
            "description": "coco2017-5k (HF: benjamintli/coco2017-5k) exported to COCO format",
            "version": "1.0",
            "year": 2017,
            "date_created": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": coco_categories(),
    }

    with open(out_ann_path, "w", encoding="utf-8") as f:
        json.dump(coco, f)

    return image_id_offset + len(ds_split)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download HF dataset benjamintli/coco2017-5k and export to standard COCO layout (train2017/val2017 + instances_*.json)."
    )
    parser.add_argument("--out-dir", type=str, default="data/coco2017-5k", help="Output root directory")
    parser.add_argument("--cache-dir", type=str, default=None, help="Optional HF datasets cache dir")
    parser.add_argument("--train-split", type=str, default="train", help="HF split name for training")
    parser.add_argument(
        "--val-split", type=str, default="validation", help="HF split name for validation"
    )
    args = parser.parse_args()

    from datasets import load_dataset

    out_dir = args.out_dir
    out_train_dir = os.path.join(out_dir, "train2017")
    out_val_dir = os.path.join(out_dir, "val2017")
    out_ann_dir = os.path.join(out_dir, "annotations")

    ensure_dir(out_train_dir)
    ensure_dir(out_val_dir)
    ensure_dir(out_ann_dir)

    print("Loading dataset from HF: benjamintli/coco2017-5k")
    ds = load_dataset("benjamintli/coco2017-5k", cache_dir=args.cache_dir)

    if args.train_split not in ds:
        raise ValueError(f"train split '{args.train_split}' not found. Available: {list(ds.keys())}")
    if args.val_split not in ds:
        raise ValueError(f"val split '{args.val_split}' not found. Available: {list(ds.keys())}")

    print(f"Exporting split '{args.train_split}' -> train2017")
    offset = export_split(
        ds[args.train_split],
        split_name=args.train_split,
        out_images_dir=out_train_dir,
        out_ann_path=os.path.join(out_ann_dir, "instances_train2017.json"),
        image_id_offset=0,
    )

    print(f"Exporting split '{args.val_split}' -> val2017")
    export_split(
        ds[args.val_split],
        split_name=args.val_split,
        out_images_dir=out_val_dir,
        out_ann_path=os.path.join(out_ann_dir, "instances_val2017.json"),
        image_id_offset=offset,
    )

    print("Done.")
    print(f"COCO root: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
