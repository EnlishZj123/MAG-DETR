"""Draw a dumbbell chart for top confusion pairs before and after ETF.

Example:
  python tools/DrawConfusionPairs_Dumbbell.py \
    --before-csv dataset/confusions_withoutETF.csv \
    --after-csv dataset/confusions_ETF.csv \
    --top-k 15 \
    --metric count
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "figure_outputs"
MERGE_COLS = ["gt_class_id", "pred_class_id"]


def _validate_columns(frame: pd.DataFrame, required_columns: list[str], csv_path: Path) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")


def _load_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    return pd.read_csv(csv_path)


def _format_change(before_value: float, after_value: float) -> str:
    if before_value <= 0:
        if after_value <= 0:
            return "0.0%"
        return "n/a"

    reduction = (before_value - after_value) / before_value * 100.0
    if reduction >= 0:
        return f"-{reduction:.1f}%"
    return f"+{abs(reduction):.1f}%"


def _metric_label(metric: str) -> str:
    if metric == "count":
        return "Number of Confused GT Instances"
    if metric == "confusion_rate_total_gt":
        return "Confusion Rate w.r.t. Total Ground Truth"
    if metric == "share_of_all_confusions":
        return "Share of All Confusions"
    return metric


def _prepare_plot_data(
    before: pd.DataFrame,
    after: pd.DataFrame,
    before_csv: Path,
    after_csv: Path,
    metric: str,
    top_k: int,
) -> pd.DataFrame:
    _validate_columns(before, MERGE_COLS + ["gt_class_name", "pred_class_name", metric], before_csv)
    _validate_columns(after, MERGE_COLS + [metric], after_csv)

    before = before.copy()
    after = after.copy()

    before["pair"] = before["gt_class_name"].astype(str) + " $\\rightarrow$ " + before["pred_class_name"].astype(str)
    after["pair"] = after["gt_class_name"].astype(str) + " $\\rightarrow$ " + after["pred_class_name"].astype(str)

    plot_df = before.sort_values(metric, ascending=False)[
        MERGE_COLS + ["gt_class_name", "pred_class_name", "pair", metric]
    ].rename(columns={metric: "before"})

    after_sub = after[MERGE_COLS + [metric]].rename(columns={metric: "after"})
    plot_df = plot_df.merge(after_sub, on=MERGE_COLS, how="left")
    plot_df["after"] = plot_df["after"].fillna(0)

    # Only keep pairs whose confusion count actually dropped after ETF.
    plot_df = plot_df[plot_df["after"] < plot_df["before"]].copy()

    # Keep only the top-K reduced pairs from the full filtered set.
    plot_df = plot_df.head(top_k).copy()

    plot_df["abs_change"] = plot_df["after"] - plot_df["before"]
    plot_df["relative_reduction"] = np.where(
        plot_df["before"] > 0,
        (plot_df["before"] - plot_df["after"]) / plot_df["before"] * 100.0,
        np.nan,
    )
    return plot_df.sort_values("before", ascending=True).reset_index(drop=True)


def draw_dumbbell(plot_df: pd.DataFrame, metric: str, before_name: str, after_name: str, output_dir: Path, show: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    fig_height = 0.45 * max(len(plot_df), 1) + 2.2
    fig, ax = plt.subplots(figsize=(9, fig_height))

    y = np.arange(len(plot_df))

    for idx in range(len(plot_df)):
        ax.plot(
            [plot_df.loc[idx, "before"], plot_df.loc[idx, "after"]],
            [y[idx], y[idx]],
            linewidth=2,
            alpha=0.6,
            color="#7f8c8d",
        )

    ax.scatter(plot_df["before"], y, s=70, label=before_name, zorder=3, color="#1f77b4")
    ax.scatter(plot_df["after"], y, s=70, label=after_name, zorder=3, color="#d62728")

    x_max = float(max(plot_df["before"].max(), plot_df["after"].max(), 1.0))
    offset = x_max * 0.03

    for idx in range(len(plot_df)):
        before_value = float(plot_df.loc[idx, "before"])
        after_value = float(plot_df.loc[idx, "after"])
        text = _format_change(before_value, after_value)
        ax.text(
            max(before_value, after_value) + offset,
            y[idx],
            text,
            va="center",
            fontsize=10,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["pair"], fontsize=11)
    ax.set_xlabel(_metric_label(metric), fontsize=12)
    ax.set_title("Top Confusion Pairs With Reduced Confusion After ETF", fontsize=14, pad=12)
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(0, x_max * 1.25)

    fig.tight_layout()
    fig.savefig(output_dir / "top_confusion_pairs_dumbbell.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "top_confusion_pairs_dumbbell.png", dpi=600, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw a dumbbell chart for top confusion pairs before and after ETF.")
    parser.add_argument("--before-csv", type=Path, default=Path("/mnt/data/confusions(2).csv"))
    parser.add_argument("--after-csv", type=Path, default=Path("/mnt/data/confusions(3).csv"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument(
        "--metric",
        type=str,
        default="count",
        choices=["count", "confusion_rate_total_gt", "share_of_all_confusions"],
    )
    parser.add_argument("--before-name", type=str, default="Before ETF")
    parser.add_argument("--after-name", type=str, default="After ETF")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    before = _load_csv(args.before_csv)
    after = _load_csv(args.after_csv)

    plot_df = _prepare_plot_data(before, after, args.before_csv, args.after_csv, args.metric, args.top_k)
    if plot_df.empty:
        raise ValueError("No confusion pairs available for plotting.")

    draw_dumbbell(
        plot_df=plot_df,
        metric=args.metric,
        before_name=args.before_name,
        after_name=args.after_name,
        output_dir=args.output_dir,
        show=args.show,
    )

    table_out = plot_df.sort_values("before", ascending=False).copy()
    table_out["relative_reduction"] = table_out["relative_reduction"].round(2)

    csv_path = args.output_dir / "top_confusion_pairs_comparison.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table_out[["gt_class_name", "pred_class_name", "before", "after", "abs_change", "relative_reduction"]].to_csv(
        csv_path,
        index=False,
    )

    print(table_out[["gt_class_name", "pred_class_name", "before", "after", "abs_change", "relative_reduction"]])
    print(f"Saved figure to: {args.output_dir / 'top_confusion_pairs_dumbbell.pdf'}")
    print(f"Saved table to: {csv_path}")


if __name__ == "__main__":
    main()