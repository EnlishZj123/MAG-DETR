"""Compare confusion-class AP between two checkpoints.

Usage example:
python tools/compare_confusion_ap.py \
    --base-config configs/rtdetrv2/rtdetrv2_dinov3_vit_6x_coco.yml \
    --etf-config configs/rtdetrv2/rtdetrv2_dinov3_vit_6x_coco_ETF.yml\
  --base /data2/ZJ_output2/output22_oco2017_1k_vitl16_best/best.pth \
  --etf /data2/ZJ_output2/Ablation/ETF_amc_0.1/best.pth \
  --confusion-label-ids 0,1,2,3,5,6,7,8,9,13,14,24,25,26,32,33,39,40,41,43,45,50,51,56,57,58,59,60,62,65,69,71,72,73 \
  --device cuda:7
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(__file__, "..", ".."))
sys.path.insert(0, ROOT)

from src.core import YAMLConfig
from src.solver.det_engine import evaluate


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


def _parse_ids(text: str) -> List[int]:
    text = text.strip()
    if not text:
        return []
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _eval_ckpt(config: str, ckpt: str, device: str):
    cfg = YAMLConfig(config)

    model = cfg.model
    state = _load_state_dict_from_ckpt(ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device)

    criterion = cfg.criterion
    criterion.to(device)

    postprocessor = cfg.postprocessor
    postprocessor.to(device)

    data_loader = cfg.val_dataloader
    coco_evaluator = cfg.evaluator

    stats, coco_evaluator = evaluate(
        model=model,
        criterion=criterion,
        postprocessor=postprocessor,
        data_loader=data_loader,
        coco_evaluator=coco_evaluator,
        device=torch.device(device),
    )
    return cfg, stats, coco_evaluator


def _per_class_ap_from_coco_eval(coco_eval_bbox) -> Tuple[Dict[int, float], List[int]]:
    # precision shape: [TxRxKxAxM]
    precision = coco_eval_bbox.eval["precision"]
    cat_ids = list(coco_eval_bbox.params.catIds)

    # area=all -> index 0, maxDets=100 -> usually index 2 for COCO
    area_idx = 0
    maxdet_idx = len(coco_eval_bbox.params.maxDets) - 1

    out = {}
    for k, cat_id in enumerate(cat_ids):
        p = precision[:, :, k, area_idx, maxdet_idx]
        p = p[p > -1]
        ap = float(np.mean(p)) if p.size > 0 else float("nan")
        out[int(cat_id)] = ap
    return out, cat_ids


def _mean_ignore_nan(values: List[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="",
                        help="shared config for both checkpoints (backward compatible)")
    parser.add_argument("--base-config", type=str, default="",
                        help="config used for --base checkpoint")
    parser.add_argument("--etf-config", type=str, default="",
                        help="config used for --etf checkpoint")
    parser.add_argument("--base", type=str, required=True, help="non-ETF checkpoint")
    parser.add_argument("--etf", type=str, required=True, help="ETF checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--confusion-label-ids", type=str, required=True,
                        help="comma-separated label ids (0-based), e.g. 0,1,2")
    parser.add_argument("--save-dir", type=str, default="")
    args = parser.parse_args()

    save_dir = Path(args.save_dir) if args.save_dir else Path(os.path.dirname(args.etf)) / "analysis" / "confusion_ap_compare"
    save_dir.mkdir(parents=True, exist_ok=True)

    confusion_label_ids = _parse_ids(args.confusion_label_ids)
    if len(confusion_label_ids) == 0:
        raise ValueError("No confusion label ids provided.")

    base_config = args.base_config or args.config
    etf_config = args.etf_config or args.config
    if not base_config or not etf_config:
        raise ValueError("Please provide --base-config and --etf-config, or provide shared --config.")

    cfg_base, stats_base, eval_base = _eval_ckpt(base_config, args.base, args.device)
    _, stats_etf, eval_etf = _eval_ckpt(etf_config, args.etf, args.device)

    coco_base = eval_base.coco_eval["bbox"]
    coco_etf = eval_etf.coco_eval["bbox"]

    ap_base_by_cat, _ = _per_class_ap_from_coco_eval(coco_base)
    ap_etf_by_cat, _ = _per_class_ap_from_coco_eval(coco_etf)

    dataset = _unwrap_dataset(cfg_base.val_dataloader.dataset)
    label2category = getattr(dataset, "label2category", {})
    category2name = getattr(dataset, "category2name", {})

    remap_on = bool(getattr(cfg_base.postprocessor, "remap_mscoco_category", False))
    if remap_on:
        confusion_cat_ids = [int(label2category[int(lid)]) for lid in confusion_label_ids]
    else:
        confusion_cat_ids = [int(lid) for lid in confusion_label_ids]

    rows = []
    for lid, cid in zip(confusion_label_ids, confusion_cat_ids):
        ap_b = ap_base_by_cat.get(cid, float("nan"))
        ap_e = ap_etf_by_cat.get(cid, float("nan"))
        rows.append({
            "label_id": int(lid),
            "category_id": int(cid),
            "class_name": category2name.get(int(cid), str(cid)),
            "ap_base": ap_b,
            "ap_etf": ap_e,
            "delta": ap_e - ap_b if (not np.isnan(ap_b) and not np.isnan(ap_e)) else float("nan"),
        })

    confusion_ap_base = _mean_ignore_nan([r["ap_base"] for r in rows])
    confusion_ap_etf = _mean_ignore_nan([r["ap_etf"] for r in rows])
    confusion_ap_delta = confusion_ap_etf - confusion_ap_base

    global_ap_base = float(stats_base["coco_eval_bbox"][0])
    global_ap_etf = float(stats_etf["coco_eval_bbox"][0])
    global_ap_delta = global_ap_etf - global_ap_base

    summary = {
        "base_config": base_config,
        "etf_config": etf_config,
        "base_ckpt": args.base,
        "etf_ckpt": args.etf,
        "device": args.device,
        "confusion_label_ids": confusion_label_ids,
        "confusion_category_ids": confusion_cat_ids,
        "global_ap_base": global_ap_base,
        "global_ap_etf": global_ap_etf,
        "global_ap_delta": global_ap_delta,
        "confusion_ap_base": confusion_ap_base,
        "confusion_ap_etf": confusion_ap_etf,
        "confusion_ap_delta": confusion_ap_delta,
        "delta_ratio_conf_over_global": (
            confusion_ap_delta / (global_ap_delta + 1e-12)
            if not np.isnan(confusion_ap_delta) else float("nan")
        ),
    }

    csv_path = save_dir / "confusion_class_ap_compare.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label_id", "category_id", "class_name", "ap_base", "ap_etf", "delta"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    json_path = save_dir / "summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== Confusion AP Comparison ===")
    print(f"Global AP: base={global_ap_base:.4f}, etf={global_ap_etf:.4f}, delta={global_ap_delta:+.4f}")
    print(f"Confusion AP(mean): base={confusion_ap_base:.4f}, etf={confusion_ap_etf:.4f}, delta={confusion_ap_delta:+.4f}")
    print(f"Delta ratio (conf/global): {summary['delta_ratio_conf_over_global']:.4f}")
    print("Saved:")
    print(f"  {csv_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    main()
