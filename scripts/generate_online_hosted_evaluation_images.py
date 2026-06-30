from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "reports" / "app_evaluation"

INK = "#17201c"
MUTED = "#65716b"
SURFACE = "#f4f7f4"
CARD = "#ffffff"
LINE = "#d9e0da"
TEAL = "#007f73"
BLUE = "#2563eb"
GREEN = "#13845f"
AMBER = "#b55a00"
RED = "#bd342f"

FRONTEND_URL = "https://frontend-chi-tawny-61.vercel.app"
BACKEND_URL = "https://fish-kiln-backend.onrender.com"
MEASURED_AT = "2026-06-30"

WARM_RESULTS = [
    {
        "label": "Vercel frontend page",
        "endpoint": FRONTEND_URL,
        "successes": 5,
        "failures": 0,
        "codes": [200, 200, 200, 200, 200],
        "avg_total_ms": 1445.9,
        "median_total_ms": 627.2,
        "min_total_ms": 382.5,
        "max_total_ms": 4812.8,
    },
    {
        "label": "Render health",
        "endpoint": f"{BACKEND_URL}/api/health",
        "successes": 5,
        "failures": 0,
        "codes": [200, 200, 200, 200, 200],
        "avg_total_ms": 1052.2,
        "median_total_ms": 1039.9,
        "min_total_ms": 732.5,
        "max_total_ms": 1339.2,
    },
    {
        "label": "Render model metadata",
        "endpoint": f"{BACKEND_URL}/api/model-info",
        "successes": 5,
        "failures": 0,
        "codes": [200, 200, 200, 200, 200],
        "avg_total_ms": 2729.0,
        "median_total_ms": 1008.7,
        "min_total_ms": 731.4,
        "max_total_ms": 9527.7,
    },
    {
        "label": "Render prediction",
        "endpoint": f"{BACKEND_URL}/api/predict",
        "successes": 5,
        "failures": 0,
        "codes": [200, 200, 200, 200, 200],
        "avg_total_ms": 1351.1,
        "median_total_ms": 1240.4,
        "min_total_ms": 1037.9,
        "max_total_ms": 2041.8,
    },
    {
        "label": "Render latest reading",
        "endpoint": f"{BACKEND_URL}/api/readings/latest",
        "successes": 5,
        "failures": 0,
        "codes": [200, 200, 200, 200, 200],
        "avg_total_ms": 1232.5,
        "median_total_ms": 933.0,
        "min_total_ms": 901.9,
        "max_total_ms": 2153.3,
    },
]

COLD_START_RESULTS = [
    {
        "label": "Render health after sleep",
        "endpoint": f"{BACKEND_URL}/api/health",
        "http_status": 200,
        "total_seconds": 42.833746,
    },
    {
        "label": "Render latest reading after sleep",
        "endpoint": f"{BACKEND_URL}/api/readings/latest",
        "http_status": 200,
        "total_seconds": 42.773751,
    },
]


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def seconds(ms: float) -> str:
    return f"{ms / 1000:.2f}s"


def write_measurements() -> Path:
    path = OUTPUT_DIR / "online_hosted_measurements.json"
    data = {
        "measured_at": MEASURED_AT,
        "measurement_origin": "This PC/network to hosted Vercel frontend and hosted Render backend",
        "warm_measurement_method": "Python urllib total elapsed time, 5 requests per route",
        "frontend_url": FRONTEND_URL,
        "backend_url": BACKEND_URL,
        "cold_start_results": COLD_START_RESULTS,
        "warm_results": WARM_RESULTS,
        "note": "Warm results exclude Render free-tier cold start. Cold-start results show an earlier first request after the service had slept.",
    }
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def draw_warm_response_time() -> Path:
    path = OUTPUT_DIR / "06_online_hosted_response_time.png"
    labels = [item["label"] for item in WARM_RESULTS]
    values = [float(item["avg_total_ms"]) for item in WARM_RESULTS]
    medians = [float(item["median_total_ms"]) for item in WARM_RESULTS]
    max_value = max(values)

    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    ax = fig.add_axes([0.10, 0.16, 0.78, 0.58])
    y = np.arange(len(labels))
    colors = [TEAL, BLUE, BLUE, GREEN, AMBER]
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=15)
    ax.invert_yaxis()
    ax.set_xlabel("Average total response time, warm requests", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", color=LINE)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for index, (value, median) in enumerate(zip(values, medians)):
        ax.text(
            value + max_value * 0.025,
            index,
            f"avg {seconds(value)} | median {seconds(median)}",
            va="center",
            fontsize=13,
            fontweight="bold",
            color=INK,
        )

    fig.text(0.055, 0.89, "HOSTED APP EVALUATION", fontsize=18, fontweight="bold", color=MUTED)
    fig.text(0.055, 0.82, "Online Response Time", fontsize=42, fontweight="bold", color=INK)
    fig.text(
        0.055,
        0.765,
        "Measured from this PC/network to the hosted Vercel frontend and Render backend; n=5 warm requests per route.",
        fontsize=17,
        color=MUTED,
    )
    fig.text(
        0.055,
        0.095,
        "Interpretation: user-facing requests were successful. Render backend latency varied because it is running on a free shared service.",
        fontsize=14,
        color=MUTED,
    )

    save(fig, path)
    return path


def draw_cold_start_summary() -> Path:
    path = OUTPUT_DIR / "07_render_cold_start_summary.png"
    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    fig.text(0.055, 0.89, "HOSTED APP EVALUATION", fontsize=18, fontweight="bold", color=MUTED)
    fig.text(0.055, 0.82, "Render Cold Start vs Warm API", fontsize=42, fontweight="bold", color=INK)
    fig.text(
        0.055,
        0.765,
        "Render free instances sleep after inactivity. The first request can be slow; subsequent requests are much faster.",
        fontsize=17,
        color=MUTED,
    )

    cards = [
        ("Cold start: /api/health", "42.83s", RED, "First request after backend was asleep"),
        ("Cold start: latest reading", "42.77s", RED, "First latest-reading request after sleep"),
        ("Warm prediction average", "1.35s", GREEN, "POST /api/predict, n=5"),
        ("Warm latest reading average", "1.23s", GREEN, "GET /api/readings/latest, n=5"),
    ]
    for index, (title, value, color, note) in enumerate(cards):
        col = index % 2
        row = index // 2
        x0 = 0.075 + col * 0.43
        y0 = 0.49 - row * 0.23
        ax.add_patch(plt.Rectangle((x0, y0), 0.36, 0.16, facecolor=CARD, edgecolor=LINE, linewidth=2))
        ax.add_patch(plt.Rectangle((x0, y0), 0.014, 0.16, facecolor=color, edgecolor=color))
        ax.text(x0 + 0.035, y0 + 0.105, title, fontsize=15, color=MUTED, fontweight="bold")
        ax.text(x0 + 0.035, y0 + 0.052, value, fontsize=34, color=INK, fontweight="bold")
        ax.text(x0 + 0.035, y0 + 0.025, note, fontsize=12.5, color=MUTED)

    ax.add_patch(plt.Rectangle((0.075, 0.12), 0.79, 0.09, facecolor="#eaf3ef", edgecolor="#bfd7ce", linewidth=1.5))
    ax.text(
        0.095,
        0.17,
        "Presentation line: the deployed app works online, but the free Render backend has a cold-start delay. "
        "After wake-up, the API usually responds around 1-3 seconds on the measured network.",
        fontsize=15,
        color=INK,
        va="center",
    )

    save(fig, path)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [write_measurements(), draw_warm_response_time(), draw_cold_start_summary()]
    print("Online hosted evaluation assets written to:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
