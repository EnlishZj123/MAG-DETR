import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_log_txt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "figure_outputs"

METRIC_NAMES = ["AP", "AP50", "AP75", "APs", "APm", "APl"]
MODEL_FILES = {
    "Layerwise_fused": "110k_未加入ETF.txt",
    "Baseline": "layerwise_110k.txt",
    "ETF": "ETF_amc.txt",
}
MODEL_COLORS = {
    "Layerwise_fused": "#4c78a8",
    "Baseline": "#f58518",
    "ETF": "#d62728",
}
MODEL_STYLES = {
    "Layerwise_fused": {"linestyle": "-", "marker": "o"},
    "Baseline": {"linestyle": "--", "marker": "s"},
    "ETF": {"linestyle": "-.", "marker": "^"},
}


def apply_paper_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
        }
    )


def load_best_metrics(log_path):
    best_record = None
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if "test_coco_eval_bbox" in record and len(record["test_coco_eval_bbox"]) >= 6:
                if best_record is None or record["test_coco_eval_bbox"][0] > best_record["test_coco_eval_bbox"][0]:
                    best_record = record

    if best_record is None:
        raise ValueError(f"No evaluation record found in {log_path}")

    values = best_record["test_coco_eval_bbox"][:6]
    return values, best_record.get("epoch")


def build_radar_angles(num_metrics):
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]
    return angles


def close_series(values):
    return values + values[:1]


def plot_radar_series(ax, angles, series, labels, scale=1.0):
    for model_name, values in series.items():
        closed_values = close_series(values)
        style = MODEL_STYLES[model_name]
        ax.plot(
            angles,
            closed_values,
            color=MODEL_COLORS[model_name],
            linewidth=2.4 * scale,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4.5 * scale,
            markevery=list(range(len(angles) - 1)),
            label=model_name,
        )
        ax.fill(
            angles,
            closed_values,
            color=MODEL_COLORS[model_name],
            alpha=0.10,
        )


def annotate_radar_values(ax, angles, values, color, radial_offset):
    for angle, value in zip(angles[:-1], values):
        ax.text(
            angle,
            value + radial_offset,
            f"{value:.4f}",
            ha="center",
            va="center",
            fontsize=10,
            color=color,
            bbox={"facecolor": "white", "edgecolor": color, "alpha": 0.75, "pad": 0.15},
            clip_on=False,
        )


def configure_radar_axis(ax, angles, show_metric_labels=True, y_min=0.0, y_max=1.0, y_step=0.2, tick_min=None):
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    if show_metric_labels:
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(METRIC_NAMES)
    else:
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([])
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis="x", pad=14)

    tick_start = y_min if tick_min is None else tick_min
    tick_values = np.arange(tick_start, y_max + 1e-9, y_step)
    ax.set_yticks(tick_values)
    ax.set_yticklabels([f"{value:.2f}" for value in tick_values])
    ax.tick_params(axis="y", pad=10)
    ax.grid(alpha=0.35)


def plot_radar(output_dir, show=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()

    series = {}
    for model_name, file_name in MODEL_FILES.items():
        metrics, _ = load_best_metrics(DATA_DIR / file_name)
        series[model_name] = metrics

    angles = build_radar_angles(len(METRIC_NAMES))

    fig = plt.figure(figsize=(6.8, 6.8))
    ax = fig.add_subplot(111, polar=True)
    configure_radar_axis(
        ax,
        angles,
        show_metric_labels=True,
        y_min=0.10,
        y_max=0.90,
        y_step=0.10,
        tick_min=0.10,
    )
    plot_radar_series(ax, angles, series, METRIC_NAMES)

    ax.set_title("Best-Epoch Metrics Radar Chart", pad=18)
    ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.12), frameon=True)
    fig.savefig(output_dir / "final_metrics_radar.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "final_metrics_radar.svg", bbox_inches="tight")
    fig.savefig(output_dir / "final_metrics_radar.png", bbox_inches="tight")

    pair_series = {
        "Layerwise_fused": series["Layerwise_fused"],
        "ETF": series["ETF"],
    }
    pair_min = min(min(values) for values in pair_series.values())
    pair_max = max(max(values) for values in pair_series.values())
    pair_padding = max((pair_max - pair_min) * 0.15, 0.005)
    pair_y_min = max(0.0, pair_min - pair_padding)
    pair_y_max = min(1.0, pair_max + pair_padding)
    pair_step = 0.02
    pair_start = np.floor(pair_y_min * 100) / 100
    pair_end = np.ceil(pair_y_max * 100) / 100

    pair_diff = [etf - base for base, etf in zip(pair_series["Layerwise_fused"], pair_series["ETF"])]

    pair_fig = plt.figure(figsize=(6.8, 6.8))
    diff_ax = pair_fig.add_subplot(111)
    x_positions = np.arange(len(METRIC_NAMES))
    light_blue = "#8ecae6"
    diff_bars = diff_ax.bar(x_positions, pair_diff, color=light_blue, edgecolor="#5b9bd5", width=0.62)
    diff_ax.axhline(0, color="#333333", linewidth=1.0)
    diff_ax.set_xticks(x_positions)
    diff_ax.set_xticklabels(METRIC_NAMES)
    diff_ax.set_ylabel("ETF - Layerwise_fused")
    diff_ax.set_title("Layerwise_fused vs ETF: Metric Difference")
    diff_ax.grid(axis="y", alpha=0.3)
    diff_ax.set_ylim(min(0.0, min(pair_diff) - 0.002), max(pair_diff) + 0.002)
    diff_ax.tick_params(axis="both", labelsize=12)
    for bar, value in zip(diff_bars, pair_diff):
        diff_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + (0.0005 if value >= 0 else -0.0005),
            f"{value:.4f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
            color="#222222",
        )

    pair_fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.16)
    pair_fig.savefig(output_dir / "final_metrics_radar_layerwise_fused_vs_etf.pdf", bbox_inches="tight")
    pair_fig.savefig(output_dir / "final_metrics_radar_layerwise_fused_vs_etf.svg", bbox_inches="tight")
    pair_fig.savefig(output_dir / "final_metrics_radar_layerwise_fused_vs_etf.png", bbox_inches="tight")

    if show:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot a radar chart for final metrics.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    plot_radar(args.output_dir, show=args.show)


if __name__ == "__main__":
    main()