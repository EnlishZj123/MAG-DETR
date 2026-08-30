"""Validation-only CrowdHuman evaluation entrypoint.

This script keeps the COCO-80 detector unchanged, evaluates on the CrowdHuman
val split converted to COCO format, and saves:

- raw COCO-style predictions JSON
- person-only predictions JSON
- COCO metrics JSON
- a short markdown summary
- a log.txt line for the run

Example:

    torchrun --nproc_per_node=4 tools/eval_crowdhuman.py \
      -c configs/rtdetrv2/rtdetrv2_dinov3_vit_6x_crowdhuman.yml \
      -r /data2/ZJ_output2/Ablation/ETF_amc_0.1/best.pth \
      --device cuda:0 \
      --output-dir /data2/ZJ_output2/CrowdHuman_eval
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    import torch
    from src.core import YAMLConfig

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, ROOT)


def _load_state_dict_from_ckpt(ckpt_path: str) -> Dict[str, "torch.Tensor"]:
    import torch

    resolved_path = Path(ckpt_path)
    if resolved_path.is_dir():
        for candidate_name in ("best.pth", "last.pth", "latest.pth"):
            candidate_path = resolved_path / candidate_name
            if candidate_path.is_file():
                resolved_path = candidate_path
                break
        else:
            raise FileNotFoundError(
                f"Checkpoint path is a directory and no known checkpoint file was found inside: {ckpt_path}. "
                "Expected one of best.pth, last.pth, or latest.pth."
            )

    if not resolved_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {resolved_path}")

    checkpoint = torch.load(str(resolved_path), map_location="cpu")
    if isinstance(checkpoint, dict):
        if "ema" in checkpoint and isinstance(checkpoint["ema"], dict) and "module" in checkpoint["ema"]:
            return checkpoint["ema"]["module"]
        if "model" in checkpoint:
            return checkpoint["model"]
        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint format: {resolved_path}")


def _to_coco_predictions(results: List[Dict[str, torch.Tensor]], targets: List[Dict[str, torch.Tensor]]) -> List[dict]:
    predictions: List[dict] = []
    for target, result in zip(targets, results):
        image_id = int(target["image_id"].item())
        boxes = result.get("boxes")
        labels = result.get("labels")
        scores = result.get("scores")

        if boxes is None or labels is None or scores is None:
            continue

        boxes = boxes.detach().cpu()
        labels = labels.detach().cpu()
        scores = scores.detach().cpu()

        if boxes.numel() == 0:
            continue

        boxes_xywh = boxes.clone()
        boxes_xywh[:, 2:] -= boxes_xywh[:, :2]

        for box, label, score in zip(boxes_xywh, labels, scores):
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(label.item()),
                    "bbox": [float(v) for v in box.tolist()],
                    "score": float(score.item()),
                }
            )
    return predictions


def _extract_ap_stats(stats: Dict[str, Any]) -> Dict[str, float]:
    bbox = stats.get("coco_eval_bbox")
    if bbox is None:
        raise RuntimeError("Missing coco_eval_bbox stats")
    return {
        "AP": float(bbox[0]),
        "AP50": float(bbox[1]),
        "AP75": float(bbox[2]),
        "AP_small": float(bbox[3]),
        "AP_medium": float(bbox[4]),
        "AP_large": float(bbox[5]),
        "AR_1": float(bbox[6]),
        "AR_10": float(bbox[7]),
        "AR_100": float(bbox[8]),
        "AR_small": float(bbox[9]),
        "AR_medium": float(bbox[10]),
        "AR_large": float(bbox[11]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a COCO-trained detector on CrowdHuman val")
    parser.add_argument("-c", "--config", required=True, help="CrowdHuman evaluation config")
    parser.add_argument("-r", "--resume", required=True, help="checkpoint path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default="", help="directory to save predictions and logs")
    parser.add_argument("--score-thr", type=float, default=0.0, help="filter saved predictions by score")
    parser.add_argument("-u", "--update", nargs="+", help="yaml override")
    parser.add_argument("--print-method", type=str, default="builtin")
    parser.add_argument("--print-rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-rank", type=int)
    return parser.parse_args()


def _resolve_device(args: argparse.Namespace, torch) -> "torch.device":
    local_rank = args.local_rank
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        if args.device == "cuda" or args.device == "cuda:0":
            torch.cuda.set_device(local_rank)
            return torch.device(f"cuda:{local_rank}")
        return torch.device(args.device)

    return torch.device(args.device)


def _write_outputs(output_dir: Path, stats: Dict[str, Any], predictions: List[dict], args: argparse.Namespace, cfg: YAMLConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_json = output_dir / "crowdhuman_predictions.json"
    person_json = output_dir / "crowdhuman_person_predictions.json"
    stats_json = output_dir / "crowdhuman_metrics.json"
    summary_md = output_dir / "summary.md"
    log_txt = output_dir / "log.txt"

    with raw_json.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)

    person_predictions = [p for p in predictions if int(p["category_id"]) == 1]
    with person_json.open("w", encoding="utf-8") as f:
        json.dump(person_predictions, f, ensure_ascii=False)

    ap_stats = _extract_ap_stats(stats)
    payload = {
        "config": args.config,
        "resume": args.resume,
        "output_dir": str(output_dir),
        "device": args.device,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "metrics": ap_stats,
        "num_predictions": len(predictions),
        "num_person_predictions": len(person_predictions),
    }
    with stats_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with summary_md.open("w", encoding="utf-8") as f:
        f.write("# CrowdHuman Validation Summary\n\n")
        f.write(f"- Config: `{args.config}`\n")
        f.write(f"- Checkpoint: `{args.resume}`\n")
        f.write(f"- Output dir: `{output_dir}`\n")
        f.write(f"- Device: `{args.device}`\n")
        f.write("- Dataset: CrowdHuman val converted to COCO format\n")
        f.write("- Evaluation: person-only GT, COCO-style metrics\n\n")
        f.write("## Detection Counts\n\n")
        f.write(f"- Detected targets: {len(predictions)}\n")
        f.write(f"- Person detections: {len(person_predictions)}\n\n")
        f.write("## Metrics\n\n")
        for key in ["AP", "AP50", "AP75", "AR_1", "AR_10", "AR_100", "AP_small", "AP_medium", "AP_large", "AR_small", "AR_medium", "AR_large"]:
            f.write(f"- {key}: {ap_stats[key]:.4f}\n")
        f.write("\n## Artifacts\n\n")
        f.write(f"- Raw predictions: `{raw_json}`\n")
        f.write(f"- Person-only predictions: `{person_json}`\n")
        f.write(f"- Metrics JSON: `{stats_json}`\n")
        f.write(f"- Log file: `{log_txt}`\n")

    log_entry = {
        "time": _dt.datetime.now().isoformat(timespec="seconds"),
        "config": args.config,
        "resume": args.resume,
        "metrics": ap_stats,
        "num_predictions": len(predictions),
        "num_person_predictions": len([p for p in predictions if int(p["category_id"]) == 1]),
    }
    with log_txt.open("a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


def _format_key_list(keys: List[str], limit: int = 20) -> str:
    if not keys:
        return "[]"
    shown = keys[:limit]
    suffix = "" if len(keys) <= limit else f" ... (+{len(keys) - limit} more)"
    return "[" + ", ".join(shown) + "]" + suffix

def main() -> None:
    args = parse_args()

    import torch

    from src.core import YAMLConfig, yaml_utils
    from src.misc import dist_utils
    from src.zoo.rtdetr import box_ops  # noqa: F401

    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    update_dict.update({k: v for k, v in args.__dict__.items() if k not in ["update"] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    output_dir = Path(args.output_dir or cfg.output_dir or "./crowdhuman_eval")

    with torch.no_grad():
        device = _resolve_device(args, torch)
        model = cfg.model
        state_dict = _load_state_dict_from_ckpt(args.resume)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if dist_utils.is_main_process():
            print(f"Loaded checkpoint from {args.resume}")
            print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
            if missing:
                print(f"Missing key names: {_format_key_list(list(missing))}")
            if unexpected:
                print(f"Unexpected key names: {_format_key_list(list(unexpected))}")

        model = dist_utils.warp_model(model.to(device), sync_bn=cfg.sync_bn, find_unused_parameters=cfg.find_unused_parameters)
        criterion = cfg.criterion.to(device)
        postprocessor = cfg.postprocessor.to(device)
        evaluator = cfg.evaluator
        data_loader = dist_utils.warp_loader(cfg.val_dataloader, shuffle=cfg.val_dataloader.shuffle)

        model.eval()
        criterion.eval()
        evaluator.cleanup()

        all_predictions: List[dict] = []

        for samples, targets in data_loader:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(samples)
            orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = postprocessor(outputs, orig_target_sizes)

            res = {target["image_id"].item(): output for target, output in zip(targets, results)}
            evaluator.update(res)

            batch_predictions = _to_coco_predictions(results, targets)
            if args.score_thr > 0:
                batch_predictions = [p for p in batch_predictions if p["score"] >= float(args.score_thr)]
            all_predictions.extend(batch_predictions)

        evaluator.synchronize_between_processes()
        evaluator.accumulate()
        evaluator.summarize()

        stats: Dict[str, Any] = {}
        if "bbox" in evaluator.coco_eval:
            stats["coco_eval_bbox"] = evaluator.coco_eval["bbox"].stats.tolist()

        gathered_predictions = dist_utils.all_gather(all_predictions)
        if dist_utils.is_main_process():
            flat_predictions = [item for sublist in gathered_predictions for item in sublist]
            _write_outputs(output_dir, stats, flat_predictions, args, cfg)
            print(f"Saved CrowdHuman predictions and summary to {output_dir}")

        dist_utils.cleanup()


if __name__ == "__main__":
    main()