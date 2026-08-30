"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys

# ===== ensure project root in PYTHONPATH =====
ROOT = os.path.abspath(os.path.join(__file__, "../../.."))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torchvision.transforms as T

import numpy as np
from PIL import Image, ImageDraw

from src.core import YAMLConfig
from PIL import ImageFont


def _load_portable_font(font_size: int) -> ImageFont.ImageFont:
    """Load a crisp TrueType font without relying on OS font paths.

    Pillow typically ships with DejaVu fonts inside the package. Using that path
    makes visualization consistent across Windows/Linux/macOS.
    """
    try:
        import PIL

        pil_dir = os.path.dirname(PIL.__file__)
        font_path = os.path.join(pil_dir, "fonts", "DejaVuSans-Bold.ttf")
        return ImageFont.truetype(font_path, size=int(font_size))
    except Exception:
        return ImageFont.load_default()


def _text_bbox(drawer: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont):
    """Get text bounding box with broad Pillow compatibility."""
    if hasattr(drawer, "textbbox"):
        return drawer.textbbox((0, 0), text, font=font)
    if hasattr(font, "getbbox"):
        return font.getbbox(text)
    # Fallback for very old Pillow
    w, h = font.getsize(text)
    return (0, 0, w, h)


def _draw_text_with_stroke(
    drawer: ImageDraw.ImageDraw,
    xy,
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill,
    stroke_width: int,
    stroke_fill,
):
    """Draw text with stroke; falls back when Pillow doesn't support stroke_* args."""
    try:
        drawer.text(
            xy,
            text,
            fill=fill,
            font=font,
            stroke_width=int(stroke_width),
            stroke_fill=stroke_fill,
        )
        return
    except TypeError:
        # Manual stroke for older Pillow.
        sw = int(stroke_width)
        if sw > 0:
            x, y = xy
            for dx in range(-sw, sw + 1):
                for dy in range(-sw, sw + 1):
                    if dx == 0 and dy == 0:
                        continue
                    drawer.text((x + dx, y + dy), text, fill=stroke_fill, font=font)
        drawer.text(xy, text, fill=fill, font=font)

# ===== COCO 80 classes (official order) =====
COCO_CLASSES = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat',
    'traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat',
    'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack',
    'umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball',
    'kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket',
    'bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple',
    'sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair',
    'couch','potted plant','bed','dining table','toilet','tv','laptop','mouse',
    'remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
    'book','clock','vase','scissors','teddy bear','hair drier','toothbrush'
]

COCO_CLASS2ID = {name: idx for idx, name in enumerate(COCO_CLASSES)}


def get_color(name: str):
    """
    Generate a deterministic color for each class name
    """
    np.random.seed(abs(hash(name)) % (2**32))
    return tuple(np.random.randint(0, 255, size=3).tolist())


def _parse_class_thresh(spec: str):
    """Parse class thresholds like: 'car=0.25,truck=0.35,bus=0.30'."""
    mapping = {}
    if spec is None:
        return mapping

    spec = spec.strip()
    if not spec:
        return mapping

    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid class threshold token '{item}'. Use name=value format.")
        name, value = item.split("=", 1)
        name = name.strip().lower()
        value = value.strip()

        if name not in COCO_CLASS2ID:
            raise ValueError(f"Unknown class name '{name}'.")

        try:
            thr = float(value)
        except ValueError as e:
            raise ValueError(f"Invalid threshold '{value}' for class '{name}'.") from e

        if not (0.0 <= thr <= 1.0):
            raise ValueError(f"Threshold for class '{name}' must be in [0, 1], got {thr}.")

        mapping[COCO_CLASS2ID[name]] = thr

    return mapping


def _parse_class_max_area_ratio(spec: str):
    """Parse class area-ratio limits like: 'car=0.35,truck=0.45'."""
    mapping = {}
    if spec is None:
        return mapping

    spec = spec.strip()
    if not spec:
        return mapping

    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid class area token '{item}'. Use name=value format.")
        name, value = item.split("=", 1)
        name = name.strip().lower()
        value = value.strip()

        if name not in COCO_CLASS2ID:
            raise ValueError(f"Unknown class name '{name}'.")

        try:
            ratio = float(value)
        except ValueError as e:
            raise ValueError(f"Invalid area ratio '{value}' for class '{name}'.") from e

        if not (0.0 < ratio <= 1.0):
            raise ValueError(
                f"Area ratio for class '{name}' must be in (0, 1], got {ratio}."
            )

        mapping[COCO_CLASS2ID[name]] = ratio

    return mapping


def _intersection_area(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    return iw * ih


def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _box_center(box):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _center_in_box(center_xy, box):
    cx, cy = center_xy
    x1, y1, x2, y2 = box
    return (x1 <= cx <= x2) and (y1 <= cy <= y2)


# def draw(images, labels, boxes, scores, thrh=0.4):
#     """
#     Draw detection results with per-class colors
#     """
#     for i, im in enumerate(images):
#         drawer = ImageDraw.Draw(im)

#         scr = scores[i]
#         keep = scr > thrh

#         lab = labels[i][keep]
#         box = boxes[i][keep]
#         scrs = scores[i][keep]

#         for j, b in enumerate(box):
#             cls_id = lab[j].item()
#             cls_name = COCO_CLASSES[cls_id]
#             score = scrs[j].item()

#             color = get_color(cls_name)

#             # draw bounding box
#             drawer.rectangle(list(b), outline=color, width=3)

#             # draw label text
#             text = f"{cls_name} {score:.2f}"
#             drawer.text(
#                 (b[0] + 2, b[1] + 2),
#                 text,
#                 fill=color
#             )

#         im.save(f"results_{i}.jpg")
def draw(
    images,
    labels,
    boxes,
    scores,
    thrh=0.3,
    class_thrh=None,
    class_max_area_ratio=None,
    max_area_ratio=1.0,
    suppress_containing_boxes=True,
    contain_iou_small=0.9,
    contain_area_ratio=2.0,
):
    """
    Draw detection results with large, readable class names + scores
    (PIL-based, paper-quality visualization)
    """
    for i, im in enumerate(images):
        # Adaptive font size for readability across resolutions.
        font_size = max(20, int(round(im.size[1] * 0.03)))
        font = _load_portable_font(font_size)

        drawer = ImageDraw.Draw(im)

        scr = scores[i]
        lab_all = labels[i]
        box_all = boxes[i]

        use_geom_filter = (max_area_ratio < 1.0) or bool(class_max_area_ratio)
        if class_thrh or use_geom_filter:
            keep_list = []
            img_area = float(im.size[0] * im.size[1])
            for j in range(scr.shape[0]):
                cls_id = int(lab_all[j].item())
                cls_thr = float(class_thrh.get(cls_id, thrh))
                score_keep = bool(scr[j].item() > cls_thr)

                x1, y1, x2, y2 = box_all[j].tolist()
                bw = max(0.0, float(x2 - x1))
                bh = max(0.0, float(y2 - y1))
                area_ratio = (bw * bh) / max(img_area, 1.0)

                area_keep = True
                if use_geom_filter:
                    area_keep = area_ratio <= float(max_area_ratio)
                    if class_max_area_ratio and (cls_id in class_max_area_ratio):
                        area_keep = area_ratio <= float(class_max_area_ratio[cls_id])

                keep_list.append(score_keep and area_keep)
            keep = torch.tensor(keep_list, dtype=torch.bool, device=scr.device)
        else:
            keep = scr > thrh

        lab = lab_all[keep]
        box = box_all[keep]
        scrs = scr[keep]

        if suppress_containing_boxes and box.shape[0] > 1:
            final_keep = torch.ones((box.shape[0],), dtype=torch.bool, device=box.device)
            box_np = box.detach().cpu().tolist()
            lab_np = lab.detach().cpu().tolist()
            scr_np = scrs.detach().cpu().tolist()

            for a in range(len(box_np)):
                if not final_keep[a]:
                    continue
                area_a = _box_area(box_np[a])
                if area_a <= 0:
                    final_keep[a] = False
                    continue
                for b in range(len(box_np)):
                    if a == b:
                        continue
                    if lab_np[a] != lab_np[b]:
                        continue
                    if scr_np[b] <= scr_np[a]:
                        continue

                    area_b = _box_area(box_np[b])
                    if area_b <= 0:
                        continue
                    if area_a < float(contain_area_ratio) * area_b:
                        continue

                    center_b = _box_center(box_np[b])
                    if not _center_in_box(center_b, box_np[a]):
                        continue

                    inter = _intersection_area(box_np[a], box_np[b])
                    io_small = inter / max(area_b, 1e-12)
                    if io_small >= float(contain_iou_small):
                        final_keep[a] = False
                        break

            lab = lab[final_keep]
            box = box[final_keep]
            scrs = scrs[final_keep]

        for j, b in enumerate(box):
            cls_id = lab[j].item()
            cls_name = COCO_CLASSES[cls_id]
            score = scrs[j].item()

            color = get_color(cls_name)

            x1, y1, x2, y2 = b.tolist()

            # ===== draw bounding box =====
            drawer.rectangle(
                [x1, y1, x2, y2],
                outline=color,
                width=20
            )

            # ===== label text =====
            text = f"{cls_name} {score:.2f}"

            # text size
            tb = _text_bbox(drawer, text, font)
            tw = tb[2] - tb[0]
            th = tb[3] - tb[1]

            pad = 4

            # label background (same as box color)
            drawer.rectangle(
                [x1, y1, x1 + tw + pad * 2, y1 + th + pad * 2],
                fill=color
            )

            # draw text (white text + black stroke)
            _draw_text_with_stroke(
                drawer,
                (x1 + pad, y1 + pad),
                text,
                font=font,
                fill=(255, 255, 255),
                stroke_width=max(1, int(round(font_size * 0.08))),
                stroke_fill=(0, 0, 0),
            )

        # Save with higher JPEG quality to avoid text artifacts.
        try:
            im.save(f"results_{i}.jpg", quality=95, subsampling=0)
        except TypeError:
            # Older Pillow doesn't accept subsampling kwarg
            im.save(f"results_{i}.jpg", quality=95)

def main(args):
    """
    main inference entry
    """
    cfg = YAMLConfig(args.config, resume=args.resume)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume to load model.state_dict by now.")

    # ===== load weights (train -> deploy) =====
    incompatible = cfg.model.load_state_dict(state, strict=False)
    # When the checkpoint doesn't match the config (e.g., different backbone/encoder),
    # strict=False can silently skip many parameters and hurt recall.
    if hasattr(incompatible, "missing_keys") and hasattr(incompatible, "unexpected_keys"):
        mk, uk = incompatible.missing_keys, incompatible.unexpected_keys
        print(f"[load_state_dict] missing_keys={len(mk)} unexpected_keys={len(uk)}")
        if len(mk) > 0:
            print("[load_state_dict] first missing_keys:")
            for k in mk[:30]:
                print("  -", k)
        if len(uk) > 0:
            print("[load_state_dict] first unexpected_keys:")
            for k in uk[:30]:
                print("  -", k)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model().to(args.device)
    model.eval()

    # ===== load image =====
    im_pil = Image.open(args.im_file).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(args.device)

    eval_size = getattr(getattr(cfg, 'model', None), 'decoder', None)
    eval_size = getattr(eval_size, 'eval_spatial_size', None)
    if eval_size is None:
        eval_size = cfg.yaml_cfg.get('eval_spatial_size', [640, 640])

    if not isinstance(eval_size, (list, tuple)) or len(eval_size) != 2:
        raise ValueError(f"Invalid eval_spatial_size: {eval_size}")

    resize_h, resize_w = int(eval_size[0]), int(eval_size[1])

    transforms = T.Compose([
        T.Resize((resize_h, resize_w)),
        T.ToTensor(),
    ])

    im_data = transforms(im_pil)[None].to(args.device)

    with torch.no_grad():
        labels, boxes, scores = model(im_data, orig_size)

    class_thrh = _parse_class_thresh(args.class_thresh)
    if class_thrh:
        print("[class_thresh] using per-class thresholds:")
        for cls_id, thr in sorted(class_thrh.items(), key=lambda x: x[0]):
            print(f"  - {COCO_CLASSES[cls_id]}({cls_id}): {thr}")

    class_max_area_ratio = _parse_class_max_area_ratio(args.class_max_area_ratio)
    if class_max_area_ratio:
        print("[class_max_area_ratio] using per-class limits:")
        for cls_id, ratio in sorted(class_max_area_ratio.items(), key=lambda x: x[0]):
            print(f"  - {COCO_CLASSES[cls_id]}({cls_id}): {ratio}")

    if not (0.0 < float(args.max_area_ratio) <= 1.0):
        raise ValueError(f"--max-area-ratio must be in (0, 1], got {args.max_area_ratio}")

    draw(
        [im_pil],
        labels,
        boxes,
        scores,
        thrh=float(args.thresh),
        class_thrh=class_thrh,
        class_max_area_ratio=class_max_area_ratio,
        max_area_ratio=float(args.max_area_ratio),
        suppress_containing_boxes=bool(args.suppress_containing_boxes),
        contain_iou_small=float(args.contain_iou_small),
        contain_area_ratio=float(args.contain_area_ratio),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-r", "--resume", type=str, required=True)
    parser.add_argument("-f", "--im-file", type=str, required=True)
    parser.add_argument("-d", "--device", type=str, default="cpu")
    parser.add_argument("--thresh", type=float, default=0.5, help="Score threshold for visualization.")
    parser.add_argument(
        "--class-thresh",
        type=str,
        default="",
        help="Per-class thresholds, e.g. 'car=0.25,truck=0.35,bus=0.30'.",
    )
    parser.add_argument(
        "--class-max-area-ratio",
        type=str,
        default="",
        help="Per-class max box area ratio, e.g. 'car=0.35,truck=0.45'.",
    )
    parser.add_argument(
        "--max-area-ratio",
        type=float,
        default=0.35,
        help="Global max box area ratio for all classes, in (0,1].",
    )
    parser.add_argument(
        "--suppress-containing-boxes",
        action="store_true",
        help="Suppress large low-score boxes that contain higher-score same-class boxes.",
    )
    parser.add_argument(
        "--no-suppress-containing-boxes",
        action="store_false",
        dest="suppress_containing_boxes",
        help="Disable suppression of containing boxes.",
    )
    parser.set_defaults(suppress_containing_boxes=True)
    parser.add_argument(
        "--contain-iou-small",
        type=float,
        default=0.9,
        help="Containment threshold on intersection-over-smaller-box area.",
    )
    parser.add_argument(
        "--contain-area-ratio",
        type=float,
        default=2.0,
        help="Suppress large box when its area is at least this multiple of smaller contained box.",
    )
    args = parser.parse_args()

    main(args)
