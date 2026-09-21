"""Report-ready plots."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_history(history, model_name: str, outputs: Path):
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, hist["accuracy"], label="Training Accuracy")
    ax.plot(epochs, hist["val_accuracy"], label="Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{model_name}: Training vs Validation Accuracy")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outputs / f"curves_{model_name}_accuracy.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, hist["loss"], label="Training Loss")
    ax.plot(epochs, hist["val_loss"], label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary Cross-Entropy Loss")
    ax.set_title(f"{model_name}: Training vs Validation Loss")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outputs / f"curves_{model_name}_loss.png", dpi=200)
    plt.close(fig)


def plot_metric_bars(results_rows: list[dict], outputs: Path, baseline_accuracy: float):
    model_rows = [r for r in results_rows if r["Model"] != "Majority Baseline"]
    names = [r["Model"] for r in model_rows]
    specs = [
        ("Acc_Test", "Test Accuracy", "fig3_accuracy.png"),
        ("F1", "Test F1-score", "fig3_f1.png"),
        ("ROC_AUC", "Test ROC-AUC", "fig3_rocauc.png"),
    ]

    for key, ylabel, filename in specs:
        values = [float(r[key]) for r in model_rows]
        fig, ax = plt.subplots(figsize=(11, 6))
        x = np.arange(len(names))
        bars = ax.bar(x, values)
        ax.axhline(
            baseline_accuracy,
            linestyle="--",
            linewidth=1.5,
            label=f"Majority baseline ({baseline_accuracy:.2%})",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Model Comparison — {ylabel}")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(value + 0.015, 1.02),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        fig.tight_layout()
        fig.savefig(outputs / filename, dpi=200)
        plt.close(fig)
