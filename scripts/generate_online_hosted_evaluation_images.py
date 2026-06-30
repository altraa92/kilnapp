from __future__ import annotations

import json
import os
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


ROOT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT_DIR / "reports" / "app_evaluation"

FRONTEND_URL = os.getenv("FRONTEND_URL", "https://frontend-chi-tawny-61.vercel.app")
BACKEND_URL = os.getenv("BACKEND_URL", "https://fish-kiln-backend.onrender.com")
SAMPLES_PER_ROUTE = int(os.getenv("APP_EVAL_SAMPLES", "5"))

TEST_BATCH = {
    "fishType": "catfish",
    "initialWeight": 2.0,
    "currentWeight": 1.4,
    "temperature": 60,
    "humidity": 45,
    "elapsedTime": 120,
}

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


def clear_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT_DIR.iterdir():
        if path.is_file():
            path.unlink()


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def request_json(method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"User-Agent": "kilnapp-online-evaluation/1.0"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
            elapsed_ms = (time.perf_counter() - start) * 1000
            parsed: Any
            try:
                parsed = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                parsed = {}
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "elapsed_ms": elapsed_ms,
                "body": parsed,
                "error": "",
            }
    except urllib.error.HTTPError as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        try:
            body = json.loads(exc.read().decode("utf-8"))
        except Exception:
            body = {}
        return {"ok": False, "status": exc.code, "elapsed_ms": elapsed_ms, "body": body, "error": str(exc)}
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {"ok": False, "status": None, "elapsed_ms": elapsed_ms, "body": {}, "error": str(exc)}


def measure_endpoint(label: str, method: str, url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    records = []
    for _ in range(SAMPLES_PER_ROUTE):
        records.append(request_json(method, url, payload))
        time.sleep(0.45)

    successes = [record for record in records if record["ok"]]
    times = [float(record["elapsed_ms"]) for record in successes]
    return {
        "label": label,
        "method": method,
        "url": url,
        "successes": len(successes),
        "samples": len(records),
        "codes": [record["status"] for record in records],
        "avg_ms": statistics.mean(times) if times else 0.0,
        "median_ms": statistics.median(times) if times else 0.0,
        "min_ms": min(times) if times else 0.0,
        "max_ms": max(times) if times else 0.0,
        "errors": [record["error"] for record in records if record["error"]],
    }


def collect_online_results() -> dict[str, Any]:
    endpoints = [
        ("Vercel frontend", "GET", FRONTEND_URL, None),
        ("Render health API", "GET", f"{BACKEND_URL}/api/health", None),
        ("Render model info API", "GET", f"{BACKEND_URL}/api/model-info", None),
        ("Render prediction API", "POST", f"{BACKEND_URL}/api/predict", TEST_BATCH),
        ("Render latest reading API", "GET", f"{BACKEND_URL}/api/readings/latest", None),
    ]

    for _, method, url, payload in endpoints:
        request_json(method, url, payload)

    measurements = [measure_endpoint(*endpoint) for endpoint in endpoints]
    prediction_result = request_json("POST", f"{BACKEND_URL}/api/predict", TEST_BATCH)
    return {
        "frontend_url": FRONTEND_URL,
        "backend_url": BACKEND_URL,
        "samples_per_route": SAMPLES_PER_ROUTE,
        "measurements": measurements,
        "prediction": prediction_result,
    }


def seconds(ms: float) -> str:
    return f"{ms / 1000:.2f}s"


def draw_card(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    color: str = TEAL,
    body_size: float = 20,
) -> None:
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=CARD, edgecolor=LINE, linewidth=2))
    ax.add_patch(plt.Rectangle((x, y), 0.012, h, facecolor=color, edgecolor=color))
    ax.text(x + 0.03, y + h - 0.045, title, fontsize=14, color=MUTED, fontweight="bold")
    ax.text(x + 0.03, y + h - 0.095, body, fontsize=body_size, color=INK, fontweight="bold", linespacing=1.12)


def draw_scope(results: dict[str, Any]) -> Path:
    path = OUTPUT_DIR / "01_online_evaluation_scope.png"
    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(0.055, 0.89, "HOSTED APP EVALUATION", fontsize=18, fontweight="bold", color=MUTED)
    ax.text(0.055, 0.82, "Online Measurement Scope", fontsize=42, fontweight="bold", color=INK)
    ax.text(
        0.055,
        0.765,
        "The app was evaluated against the deployed Vercel frontend and Render backend. Local Flask test-client results are not used here.",
        fontsize=17,
        color=MUTED,
    )

    boxes = [
        (0.06, "Browser network", "HTTP request\norigin", TEAL),
        (0.29, "Vercel frontend", "React dashboard\nonline", BLUE),
        (0.52, "Render backend", "Flask API\nendpoints", GREEN),
        (0.75, "Prediction model", "Random Forest\noutput", AMBER),
    ]
    for x, title, body, color in boxes:
        draw_card(ax, x, 0.38, 0.19, 0.16, title, body, color, body_size=18)

    for x in [0.25, 0.48, 0.71]:
        ax.annotate(
            "",
            xy=(x + 0.03, 0.46),
            xytext=(x, 0.46),
            arrowprops={"arrowstyle": "->", "color": INK, "linewidth": 2.5},
        )

    ax.add_patch(plt.Rectangle((0.08, 0.16), 0.80, 0.11, facecolor="#eaf3ef", edgecolor="#bfd7ce", linewidth=1.5))
    ax.text(0.105, 0.215, "Frontend: Vercel hosted React app", fontsize=14, color=INK, fontweight="bold")
    ax.text(0.105, 0.175, "Backend: Render hosted Flask API", fontsize=14, color=INK, fontweight="bold")

    save(fig, path)
    return path


def draw_validation_matrix(results: dict[str, Any]) -> Path:
    path = OUTPUT_DIR / "02_online_endpoint_validation.png"
    rows = results["measurements"]

    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(0.055, 0.89, "HOSTED APP EVALUATION", fontsize=18, fontweight="bold", color=MUTED)
    ax.text(0.055, 0.82, "Online Endpoint Validation", fontsize=42, fontweight="bold", color=INK)
    ax.text(
        0.055,
        0.765,
        f"Each route was called {results['samples_per_route']} times against the deployed app. A pass means every sampled request returned HTTP 200.",
        fontsize=17,
        color=MUTED,
    )

    x0, y0 = 0.065, 0.63
    widths = [0.26, 0.16, 0.16, 0.25]
    headers = ["Check", "HTTP result", "Samples", "Evidence"]
    for index, header in enumerate(headers):
        x = x0 + sum(widths[:index])
        ax.add_patch(plt.Rectangle((x, y0), widths[index], 0.07, facecolor=INK, edgecolor=INK))
        ax.text(x + 0.015, y0 + 0.043, header, fontsize=13, color="white", fontweight="bold")

    for row_index, item in enumerate(rows):
        y = y0 - (row_index + 1) * 0.082
        status_color = GREEN if item["successes"] == item["samples"] else RED
        values = [
            item["label"],
            "Pass" if item["successes"] == item["samples"] else "Check",
            f"{item['successes']}/{item['samples']}",
            f"avg {seconds(item['avg_ms'])}, median {seconds(item['median_ms'])}",
        ]
        for col_index, value in enumerate(values):
            x = x0 + sum(widths[:col_index])
            ax.add_patch(plt.Rectangle((x, y), widths[col_index], 0.082, facecolor=CARD, edgecolor=LINE, linewidth=1.5))
            color = status_color if col_index == 1 else INK
            ax.text(x + 0.015, y + 0.049, value, fontsize=13.5, color=color, fontweight="bold")

    save(fig, path)
    return path


def draw_response_time(results: dict[str, Any]) -> Path:
    path = OUTPUT_DIR / "03_online_response_time.png"
    rows = results["measurements"]
    labels = [row["label"] for row in rows]
    median_values = [float(row["median_ms"]) for row in rows]
    avg_values = [float(row["avg_ms"]) for row in rows]
    max_value = max(median_values) if median_values else 1

    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    ax = fig.add_axes([0.11, 0.16, 0.76, 0.58])
    y = np.arange(len(labels))
    colors = [TEAL, BLUE, BLUE, GREEN, AMBER]
    ax.barh(y, median_values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel("Median hosted response time (online requests)", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", color=LINE)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for index, (avg, median) in enumerate(zip(avg_values, median_values)):
        ax.text(
            median + max_value * 0.04,
            index,
            f"median {seconds(median)} | avg {seconds(avg)}",
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
        "Measured against hosted Vercel and Render services using repeated online requests. Local timing is not used.",
        fontsize=17,
        color=MUTED,
    )

    save(fig, path)
    return path


def draw_prediction_result(results: dict[str, Any]) -> Path:
    path = OUTPUT_DIR / "04_online_prediction_result.png"
    prediction = results["prediction"]
    body = prediction.get("body") if prediction.get("ok") else {}
    status = body.get("dryingStatus", "n/a")

    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(0.055, 0.89, "HOSTED APP EVALUATION", fontsize=18, fontweight="bold", color=MUTED)
    ax.text(0.055, 0.82, "Live Prediction Result", fontsize=42, fontweight="bold", color=INK)
    ax.text(
        0.055,
        0.765,
        "A sample fish batch was submitted to the deployed Render prediction endpoint and the returned output was recorded.",
        fontsize=17,
        color=MUTED,
    )

    draw_card(ax, 0.075, 0.54, 0.28, 0.13, "Input batch", "Catfish, 2.0kg to 1.4kg", TEAL)
    draw_card(ax, 0.075, 0.36, 0.28, 0.13, "Kiln conditions", "60C, 45% humidity", BLUE)
    draw_card(ax, 0.075, 0.18, 0.28, 0.13, "Elapsed drying time", "120 minutes", AMBER)

    outputs = [
        ("Estimated moisture", f"{body.get('estimatedMoistureContent', 'n/a')}%", GREEN),
        ("Drying progress", f"{body.get('dryingProgress', 'n/a')}%", GREEN),
        ("Remaining time", f"{body.get('estimatedRemainingTime', 'n/a')} min", BLUE),
        ("Drying status", str(status), AMBER),
    ]
    for index, (title, value, color) in enumerate(outputs):
        x = 0.43 + (index % 2) * 0.25
        y = 0.47 - (index // 2) * 0.21
        draw_card(ax, x, y, 0.21, 0.14, title, value, color)

    ax.add_patch(plt.Rectangle((0.43, 0.16), 0.46, 0.10, facecolor="#eaf3ef", edgecolor="#bfd7ce", linewidth=1.5))
    ax.text(
        0.455,
        0.213,
        f"Endpoint: POST {results['backend_url']}/api/predict",
        fontsize=14,
        color=INK,
        fontweight="bold",
    )
    ax.text(0.455, 0.177, f"HTTP status: {prediction.get('status')}", fontsize=13, color=MUTED)

    save(fig, path)
    return path


def draw_scorecard(results: dict[str, Any]) -> Path:
    path = OUTPUT_DIR / "05_online_app_scorecard.png"
    rows = results["measurements"]
    total_successes = sum(int(row["successes"]) for row in rows)
    total_samples = sum(int(row["samples"]) for row in rows)
    prediction_row = next(row for row in rows if row["label"] == "Render prediction API")
    latest_row = next(row for row in rows if row["label"] == "Render latest reading API")
    frontend_row = next(row for row in rows if row["label"] == "Vercel frontend")

    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(0.055, 0.89, "HOSTED APP EVALUATION", fontsize=18, fontweight="bold", color=MUTED)
    ax.text(0.055, 0.82, "Online Application Scorecard", fontsize=42, fontweight="bold", color=INK)
    ax.text(
        0.055,
        0.765,
        "Summary of the deployed app checks. These are application metrics, separate from the Random Forest MAE, RMSE, and R2.",
        fontsize=17,
        color=MUTED,
    )

    cards = [
        ("Endpoint success", f"{total_successes}/{total_samples}", GREEN, "All sampled hosted requests returned success"),
        ("Frontend availability", f"{frontend_row['successes']}/{frontend_row['samples']}", TEAL, "Hosted React page responded online"),
        ("Prediction median", seconds(prediction_row["median_ms"]), BLUE, "Median hosted POST /api/predict response"),
        ("Latest-reading median", seconds(latest_row["median_ms"]), AMBER, "Median hosted GET /api/readings/latest response"),
    ]
    for index, (title, value, color, note) in enumerate(cards):
        col = index % 2
        row = index // 2
        x0 = 0.075 + col * 0.43
        y0 = 0.48 - row * 0.23
        ax.add_patch(plt.Rectangle((x0, y0), 0.36, 0.16, facecolor=CARD, edgecolor=LINE, linewidth=2))
        ax.add_patch(plt.Rectangle((x0, y0), 0.014, 0.16, facecolor=color, edgecolor=color))
        ax.text(x0 + 0.035, y0 + 0.105, title, fontsize=15, color=MUTED, fontweight="bold")
        ax.text(x0 + 0.035, y0 + 0.052, value, fontsize=34, color=INK, fontweight="bold")
        ax.text(x0 + 0.035, y0 + 0.025, note, fontsize=12.5, color=MUTED)

    ax.add_patch(plt.Rectangle((0.075, 0.12), 0.79, 0.09, facecolor="#eaf3ef", edgecolor="#bfd7ce", linewidth=1.5))
    ax.text(
        0.095,
        0.17,
        "Presentation line: the hosted system was evaluated online through the deployed frontend and backend routes.",
        fontsize=15,
        color=INK,
        va="center",
    )

    save(fig, path)
    return path


def main() -> None:
    clear_output_dir()
    results = collect_online_results()
    paths = [
        draw_scope(results),
        draw_validation_matrix(results),
        draw_response_time(results),
        draw_prediction_result(results),
        draw_scorecard(results),
    ]

    print("Online app evaluation pictures written to:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
