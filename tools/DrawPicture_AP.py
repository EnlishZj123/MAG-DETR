import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_log_txt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "figure_outputs"
PAPER_COLORS = {
    "layerwise_fused": "#4c78a8",
    "layerwise": "#f58518",
    "etf_amc": "#54a24b",
}


def apply_paper_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "axes.linewidth": 0.9,
        }
    )

def load_log(file_path, last_n=None):
    epochs, AP, AP50, AP75 = [], [], [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if "test_coco_eval_bbox" in data and len(data["test_coco_eval_bbox"]) >= 3:
                epochs.append(data["epoch"])
                AP.append(data["test_coco_eval_bbox"][0])
                AP50.append(data["test_coco_eval_bbox"][1])
                AP75.append(data["test_coco_eval_bbox"][2])

    if last_n is not None:
        return epochs[-last_n:], AP[-last_n:], AP50[-last_n:], AP75[-last_n:]
    return epochs, AP, AP50, AP75


def normalize_epochs(epochs):
    if not epochs:
        return epochs
    start_epoch = epochs[0]
    return [epoch - start_epoch for epoch in epochs]


def get_last_values(values):
    return values[-1] if values else None


def add_boxplot(ax, data, labels, title):
    box = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        showmeans=True,
        meanline=True,
        medianprops={"color": "black", "linewidth": 1.4},
        meanprops={"color": "#7a7a7a", "linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )

    colors = [PAPER_COLORS["layerwise_fused"], PAPER_COLORS["layerwise"], PAPER_COLORS["etf_amc"]]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)

    ax.set_title(title)
    ax.set_ylabel("AP")
    ax.grid(axis="y", alpha=0.35)


def plot_figures(output_dir, show=False):
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_log = DATA_DIR / "110k_未加入ETF.txt"
    layerwise_log = DATA_DIR / "layerwise_110k.txt"
    etf_amc_log = DATA_DIR / "ETF_amc.txt"

    if not baseline_log.exists():
        raise FileNotFoundError(f"Missing log file: {baseline_log}")
    if not layerwise_log.exists():
        raise FileNotFoundError(f"Missing log file: {layerwise_log}")
    if not etf_amc_log.exists():
        raise FileNotFoundError(f"Missing log file: {etf_amc_log}")

    e1, ap1, ap50_1, ap75_1 = load_log(baseline_log, last_n=60)
    e1 = normalize_epochs(e1)
    e2, ap2, ap50_2, ap75_2 = load_log(layerwise_log)
    e3, ap3, ap50_3, ap75_3 = load_log(etf_amc_log)

    final_ap50 = [get_last_values(ap50_1), get_last_values(ap50_2), get_last_values(ap50_3)]
    final_ap75 = [get_last_values(ap75_1), get_last_values(ap75_2), get_last_values(ap75_3)]
    method_names = ["Layerwise_fused", "Baseline", "ETF"]
    comparison_labels = ["Layerwise_fused", "ETF"]
    comparison_data = [ap1, ap3]

    apply_paper_style()

    line_kwargs = {"linewidth": 2.0}

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1.0])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    add_boxplot(ax1, [ap1, ap2, ap3], method_names, "(a) AP Comparison")

    add_boxplot(ax2, comparison_data, comparison_labels, "(b) Layerwise_fused vs ETF")

    fig.tight_layout()
    fig.savefig(output_dir / "ap_triptych.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "ap_triptych.svg", bbox_inches="tight")
    fig.savefig(output_dir / "ap_triptych.png", bbox_inches="tight")

    fig1, ax1 = plt.subplots(figsize=(5.8, 4.6))
    add_boxplot(ax1, [ap1, ap2, ap3], method_names, "AP Comparison")
    fig1.tight_layout()
    fig1.savefig(output_dir / "ap_comparison.pdf", bbox_inches="tight")
    fig1.savefig(output_dir / "ap_comparison.svg", bbox_inches="tight")
    fig1.savefig(output_dir / "ap_comparison.png", bbox_inches="tight")

    fig3, ax3 = plt.subplots(figsize=(7.2, 4.8))
    x = range(len(method_names))
    bar_width = 0.36
    ax3.bar([index - bar_width / 2 for index in x], final_ap50, width=bar_width, label="AP50")
    ax3.bar([index + bar_width / 2 for index in x], final_ap75, width=bar_width, label="AP75")
    ax3.set_xticks(list(x))
    ax3.set_xticklabels(method_names)
    ax3.set_ylabel("Score")
    ax3.set_title("AP50 / AP75 Comparison")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(output_dir / "ap50_vs_ap75.pdf", bbox_inches="tight")
    fig3.savefig(output_dir / "ap50_vs_ap75.svg", bbox_inches="tight")
    fig3.savefig(output_dir / "ap50_vs_ap75.png", bbox_inches="tight")

    if show:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot AP curves from training logs.")
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
    plot_figures(args.output_dir, show=args.show)


if __name__ == "__main__":
    main()