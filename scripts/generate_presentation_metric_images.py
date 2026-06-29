from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
OUTPUT_DIR = ROOT_DIR / "reports" / "presentation_metrics"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from generate_results_report import evaluate_model, load_training_frame, plot_decision_matrix  # noqa: E402


INK = "#17201c"
MUTED = "#65716b"
SURFACE = "#f4f7f4"
TEAL = "#007f73"
BLUE = "#2563eb"
AMBER = "#b55a00"
GREEN = "#13845f"
LINE = "#d9e0da"


def make_metric_card(path: Path, title: str, value: str, unit: str, meaning: str, speaker_note: str, color: str) -> None:
    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.add_patch(plt.Rectangle((0.04, 0.08), 0.92, 0.84, facecolor="white", edgecolor=LINE, linewidth=2))
    ax.add_patch(plt.Rectangle((0.04, 0.08), 0.018, 0.84, facecolor=color, edgecolor=color))

    ax.text(0.085, 0.78, "MODEL EVALUATION METRIC", color=MUTED, fontsize=18, fontweight="bold")
    ax.text(0.085, 0.68, title, color=INK, fontsize=44, fontweight="bold")
    ax.text(0.085, 0.42, value, color=color, fontsize=96, fontweight="bold")
    ax.text(0.43, 0.44, unit, color=MUTED, fontsize=30, fontweight="bold")

    ax.text(0.085, 0.28, meaning, color=INK, fontsize=24, wrap=True)
    ax.text(0.085, 0.17, speaker_note, color=MUTED, fontsize=18, wrap=True)

    ax.text(0.82, 0.12, "Fish Kiln AI", color=MUTED, fontsize=15, fontweight="bold")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_matrix_card(path: Path, matrix) -> None:
    matrix_values = matrix.to_numpy()
    correct = int(np.trace(matrix_values))
    total = int(matrix_values.sum())
    operational_accuracy = correct / total * 100

    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(0.055, 0.88, "CONFUSION MATRIX STYLE RESULT", color=MUTED, fontsize=18, fontweight="bold")
    ax.text(0.055, 0.80, "Operational Decision Matrix", color=INK, fontsize=42, fontweight="bold")
    ax.text(
        0.055,
        0.735,
        "Regression outputs were grouped into classes: Ready/complete, Short wait, Needs drying.",
        color=MUTED,
        fontsize=19,
    )

    heat_ax = fig.add_axes([0.08, 0.15, 0.52, 0.52])
    im = heat_ax.imshow(matrix_values, cmap="YlGnBu")
    heat_ax.set_xticks(range(len(matrix.columns)))
    heat_ax.set_yticks(range(len(matrix.index)))
    heat_ax.set_xticklabels(matrix.columns, fontsize=14)
    heat_ax.set_yticklabels(matrix.index, fontsize=14)
    heat_ax.set_xlabel("Predicted class", fontsize=15, fontweight="bold")
    heat_ax.set_ylabel("Actual class", fontsize=15, fontweight="bold")
    heat_ax.tick_params(axis="x", rotation=15)

    for i in range(matrix_values.shape[0]):
        for j in range(matrix_values.shape[1]):
            value = int(matrix_values[i, j])
            text_color = "white" if value > matrix_values.max() * 0.55 else INK
            heat_ax.text(j, i, str(value), ha="center", va="center", fontsize=22, fontweight="bold", color=text_color)

    for spine in heat_ax.spines.values():
        spine.set_visible(False)

    fig.colorbar(im, ax=heat_ax, fraction=0.045, pad=0.04)

    ax.add_patch(plt.Rectangle((0.66, 0.42), 0.27, 0.22, facecolor="white", edgecolor=LINE, linewidth=2))
    ax.text(0.69, 0.58, f"{operational_accuracy:.1f}%", color=GREEN, fontsize=54, fontweight="bold")
    ax.text(0.69, 0.51, "binned decision agreement", color=INK, fontsize=17, fontweight="bold")
    ax.text(0.69, 0.46, f"{correct} of {total} test rows landed in the same operational class.", color=MUTED, fontsize=14)

    ax.add_patch(plt.Rectangle((0.66, 0.18), 0.27, 0.18, facecolor="white", edgecolor=LINE, linewidth=2))
    ax.text(0.69, 0.31, "Important wording", color=AMBER, fontsize=18, fontweight="bold")
    ax.text(
        0.69,
        0.24,
        "This is not a native classification confusion matrix. It is a regression output binned for presentation.",
        color=INK,
        fontsize=14,
        wrap=True,
    )

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_training_frame()
    evaluation = evaluate_model(data)
    metrics = evaluation["metrics"]
    _, matrix = plot_decision_matrix(evaluation["y_test"], evaluation["predictions"])

    make_metric_card(
        OUTPUT_DIR / "01_mae.png",
        "MAE - Mean Absolute Error",
        f"{metrics['mae']:.3f}",
        "minutes",
        "Average size of the prediction error on the held-out synthetic test set.",
        "Say: on average, the model is about 14 minutes away from the synthetic target remaining time.",
        TEAL,
    )
    make_metric_card(
        OUTPUT_DIR / "02_rmse.png",
        "RMSE - Root Mean Squared Error",
        f"{metrics['rmse']:.3f}",
        "minutes",
        "Error metric that punishes large mistakes more strongly than MAE.",
        "Say: RMSE is higher than MAE because bigger mistakes receive extra penalty.",
        BLUE,
    )
    make_metric_card(
        OUTPUT_DIR / "03_r2.png",
        "R² - Explained Variance",
        f"{metrics['r2']:.3f}",
        "score",
        "How much variation in the synthetic test target is explained by the model.",
        "Say: 0.979 means about 97.9% explained variance on synthetic test data; real data may be noisier.",
        GREEN,
    )
    make_matrix_card(OUTPUT_DIR / "04_confusion_matrix_equivalent.png", matrix)

    print("Presentation metric images written to:")
    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(path)


if __name__ == "__main__":
    main()
