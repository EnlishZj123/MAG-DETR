import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_log_txt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "figure_outputs"

DEFAULT_SERIES = [
    ("Layer_6_12_18", DATA_DIR / "layer_6_12_18.txt"),
    ("Layer_7_15_23", DATA_DIR / "layer_7_15_23.txt"),
    ("Layer_11_17_23", DATA_DIR / "layer_11_17_23.txt"),
    ("Layer_15_23", DATA_DIR / "layer_15_23.txt"),
    ("Layer_23", DATA_DIR / "layer_23.txt"),
]

PAPER_COLORS = ["#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2"]


def apply_paper_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.9,
        }
    )


def load_ap_series(log_path):
    ap_values = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if "test_coco_eval_bbox" in record and len(record["test_coco_eval_bbox"]) >= 1:
                ap_values.append(record["test_coco_eval_bbox"][0])

    if not ap_values:
        raise ValueError(f"No AP values found in {log_path}")

    return ap_values


def resolve_series(custom_series=None):
    if custom_series:
        return [(label, Path(path)) for label, path in custom_series]
    return DEFAULT_SERIES


def plot_layer_boxplot(output_dir, series=None, show=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()

    resolved_series = resolve_series(series)
    labels = [label for label, _ in resolved_series]
    data = []

    missing_paths = []
    for label, log_path in resolved_series:
        if not log_path.exists():
            missing_paths.append(str(log_path))
            continue
        data.append(load_ap_series(log_path))

    if missing_paths:
        missing_text = "\n".join(missing_paths)
        raise FileNotFoundError(f"Missing log file(s):\n{missing_text}")

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    box = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.55,
        showmeans=True,
        meanline=True,
        medianprops={"color": "black", "linewidth": 1.4},
        meanprops={"color": "#6f6f6f", "linewidth": 1.2},
        whiskerprops={"linewidth": 1.2},
        capprops={"linewidth": 1.2},
    )

    for patch, color in zip(box["boxes"], PAPER_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.32)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.2)

    ax.set_title("AP Comparison Across Layer Configurations")
    ax.set_ylabel("AP")
    ax.set_xlabel("Layer Configuration")
    ax.grid(axis="y", alpha=0.35)

    y_min = min(min(series_values) for series_values in data)
    y_max = max(max(series_values) for series_values in data)
    padding = max((y_max - y_min) * 0.08, 0.01)
    ax.set_ylim(max(0.0, y_min - padding), min(1.0, y_max + padding))

    fig.tight_layout()
    fig.savefig(output_dir / "layer_ap_boxplot.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "layer_ap_boxplot.svg", bbox_inches="tight")
    fig.savefig(output_dir / "layer_ap_boxplot.png", bbox_inches="tight")

    if show:
        plt.show()


def parse_series_arguments(series_args):
    if not series_args:
        return None

    parsed = []
    for item in series_args:
        if len(item) != 2:
            raise ValueError("Each --series entry must contain exactly a label and a file path.")
        parsed.append((item[0], item[1]))
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Plot AP boxplots for layer comparison logs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--series",
        action="append",
        nargs=2,
        metavar=("LABEL", "LOG_PATH"),
        help="Optional label and log path pairs. Repeat this argument for each series.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figures interactively after saving them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    custom_series = parse_series_arguments(args.series)
    plot_layer_boxplot(args.output_dir, series=custom_series, show=args.show)


if __name__ == "__main__":
    main()