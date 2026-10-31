"""Redraw training curves and confusion matrix from outputs/metrics.json.

Does not retrain or change reported numbers.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "outputs" / "metrics.json"
OUT = ROOT / "outputs"
NAVY = "#071426"
TEAL = "#0f766e"
CYAN = "#37CFFF"
SLATE = "#334155"
LABELS = {
    "leak": "Leak",
    "no_leak": "No leak",
    "environmental_noise": "Noise",
}


def main() -> None:
    metrics = json.loads(METRICS.read_text(encoding="utf-8"))
    history = metrics["history"]
    classes = metrics["class_order"]
    matrix = np.asarray(metrics["test"]["confusion_matrix"])
    best_epoch = 15
    best_val = metrics["training"]["best_val_accuracy"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": SLATE,
            "text.color": NAVY,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
        }
    )

    epochs = [row["epoch"] for row in history]
    train = [row["train_accuracy"] for row in history]
    val = [row["val_accuracy"] for row in history]

    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=160)
    ax.plot(epochs, train, color=TEAL, linewidth=2.2, label="Train")
    ax.plot(epochs, val, color=CYAN, linewidth=2.2, label="Validation")
    ax.scatter([best_epoch], [best_val], color=NAVY, zorder=5, s=36)
    ax.annotate(
        f"Best val {best_val * 100:.2f}% · epoch {best_epoch}",
        xy=(best_epoch, best_val),
        xytext=(best_epoch + 1.2, best_val - 0.12),
        arrowprops={"arrowstyle": "->", "color": NAVY},
        color=NAVY,
        fontsize=9,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Training and validation accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.set_xlim(1, max(epochs))
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "training_curves.png")
    plt.close(fig)

    names = [LABELS.get(name, name) for name in classes]
    fig, ax = plt.subplots(figsize=(5.8, 5.1), dpi=160)
    cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list("wg", ["#ecfeff", TEAL, NAVY])
    image = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Test confusion matrix (150 clips)")
    peak = float(matrix.max())
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            color = "white" if value > peak * 0.55 else NAVY
            ax.text(j, i, str(int(value)), ha="center", va="center", color=color, fontsize=12, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png")
    plt.close(fig)
    print("Wrote outputs/training_curves.png and outputs/confusion_matrix.png")


if __name__ == "__main__":
    main()
