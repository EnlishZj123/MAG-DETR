import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_log_txt"
DEFAULT_OUTPUT_DIR = BASE_DIR / "figure_outputs"

MODEL_FILES = {
    "Layerwise": "layerwise_110k.txt",
    "ETF+AMC": "ETF_amc.txt",
    "Layerwise_fused": "110k_未加入ETF.txt",
}
LOSS_FIELDS = ["train_loss_bbox", "train_loss_giou", "train_loss_vfl"]
LOSS_LABELS = ["bbox_loss", "giou_loss", "vfl_loss"]
LOSS_COLORS = {
    "bbox_loss": "#4c78a8",
    "giou_loss": "#f58518",
    "vfl_loss": "#d62728",
}


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
        }
    )


def load_loss_components(log_path):
    epochs = []
    components = {field: [] for field in LOSS_FIELDS}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if all(field in record for field in LOSS_FIELDS):
                epochs.append(record.get("epoch"))
                for field in LOSS_FIELDS:
                    components[field].append(record[field])

    return epochs, components


def plot_loss_components(output_dir, show=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()

    log_data = {}
    for model_name, file_name in MODEL_FILES.items():
        epochs, components = load_loss_components(DATA_DIR / file_name)
        log_data[model_name] = {"epochs": epochs, "components": components}

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 9.2), sharex=False)
    panel_order = ["Layerwise", "ETF+AMC", "Layerwise_fused"]

    for axis, model_name in zip(axes, panel_order):
        epochs = log_data[model_name]["epochs"]
        components = log_data[model_name]["components"]

        for field, label in zip(LOSS_FIELDS, LOSS_LABELS):
            axis.plot(
                epochs,
                components[field],
                label=label,
                color=LOSS_COLORS[label],
                linewidth=1.8,
            )

        axis.set_title(model_name)
        axis.set_ylabel("Loss")
        axis.grid(axis="y", alpha=0.35)
        axis.legend(loc="upper right", ncol=3, frameon=True)

        y_values = [value for field in LOSS_FIELDS for value in components[field]]
        y_min = min(y_values)
        y_max = max(y_values)
        padding = max((y_max - y_min) * 0.06, 0.15)
        axis.set_ylim(y_min - padding, y_max + padding)

    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Training Loss Evolution Across Three Models", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(output_dir / "train_loss_components.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "train_loss_components.svg", bbox_inches="tight")
    fig.savefig(output_dir / "train_loss_components.png", bbox_inches="tight")

    if show:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot bbox/giou/vfl loss evolution for three models.")
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
    plot_loss_components(args.output_dir, show=args.show)


if __name__ == "__main__":
    main()