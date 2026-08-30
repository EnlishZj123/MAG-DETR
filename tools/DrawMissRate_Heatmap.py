import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "figure_outputs"

ROW_LABELS = ["Miss Rate", "Small Miss Rate", "Medium Miss Rate", "Large Miss Rate"]
COL_LABELS = ["Ours", "RT-DETR", "YOLOv8", "YOLOv11"]

# Values are percentages from the four provided summary screenshots.
DEFAULT_MISS_RATE_MATRIX = np.array(
    [
        [8.7, 13.2875, 43.3439, 42.6311],
        [17.2432, 22.6677, 67.2628, 67.1253],
        [3.44, 8.0617, 32.6904, 31.3993],
        [1.4979, 4.2516, 16.4881, 15.5894],
    ],
    dtype=float,
)


def apply_paper_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
        }
    )


def draw_heatmap(output_dir: Path, show: bool = False) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()

    data = DEFAULT_MISS_RATE_MATRIX

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    heatmap = ax.imshow(data, cmap="Blues", aspect="auto")

    ax.set_xticks(np.arange(len(COL_LABELS)), labels=COL_LABELS)
    ax.set_yticks(np.arange(len(ROW_LABELS)), labels=ROW_LABELS)
    ax.set_title("Miss Rate Heatmap")
    ax.set_xlabel("Model")
    ax.set_ylabel("Object Size Metric")
    ax.grid(False)

    # Annotate each heatmap cell with the precise miss rate.
    text_threshold = float(data.max()) * 0.55
    for row in range(data.shape[0]):
        for col in range(data.shape[1]):
            value = data[row, col]
            text_color = "black" if value < text_threshold else "white"
            ax.text(col, row, f"{value:.2f}%", ha="center", va="center", color=text_color, fontsize=10)

    cbar = fig.colorbar(heatmap, ax=ax)
    cbar.set_label("Miss Rate (%)")
    cbar.ax.grid(False)

    fig.tight_layout()
    fig.savefig(output_dir / "miss_rate_heatmap.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "miss_rate_heatmap.svg", bbox_inches="tight")
    fig.savefig(output_dir / "miss_rate_heatmap.png", bbox_inches="tight")

    if show:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot miss-rate heatmap for Ours, RT-DETR, YOLOv8, and YOLOv11."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save figures.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show figure interactively after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    draw_heatmap(args.output_dir, show=args.show)


if __name__ == "__main__":
    main()