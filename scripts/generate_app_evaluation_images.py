from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from textwrap import fill

import matplotlib
import numpy as np


matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
OUTPUT_DIR = ROOT_DIR / "reports" / "app_evaluation"

sys.path.insert(0, str(BACKEND_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

import app as backend_app  # noqa: E402
from generate_results_report import TEST_BATCH  # noqa: E402


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


def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def timed_request(client, method: str, path: str, **kwargs):
    start = time.perf_counter()
    response = getattr(client, method)(path, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return response, elapsed_ms


def run_frontend_build() -> dict[str, object]:
    start = time.perf_counter()
    result = subprocess.run(
        ["npm", "run", "build"],
        cwd=FRONTEND_DIR,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed_s = time.perf_counter() - start

    assets_dir = FRONTEND_DIR / "dist" / "assets"
    asset_kb = 0.0
    if assets_dir.exists():
        asset_kb = sum(path.stat().st_size for path in assets_dir.iterdir() if path.is_file()) / 1024

    return {
        "pass": result.returncode == 0,
        "build_seconds": elapsed_s,
        "asset_kb": asset_kb,
        "stderr": result.stderr[-180:],
    }


def run_app_evaluation() -> dict[str, object]:
    build = run_frontend_build()

    rows: list[dict[str, object]] = [
        {
            "area": "Frontend",
            "check": "Production build",
            "status": "Pass" if build["pass"] else "Fail",
            "evidence": f"Vite build in {build['build_seconds']:.2f}s",
        },
        {
            "area": "Frontend",
            "check": "Compiled asset size",
            "status": "Recorded",
            "evidence": f"{build['asset_kb']:.1f} KB in dist/assets",
        },
    ]
    latencies: list[dict[str, object]] = []
    demo_snapshots: list[dict[str, object]] = []

    with backend_app.app.test_client() as client:
        # Warm the model so /api/predict latency is not dominated by one-time loading.
        client.post("/api/predict", json=TEST_BATCH)

        for label, method, path, kwargs in [
            ("Health check", "get", "/api/health", {}),
            ("Model metadata", "get", "/api/model-info", {}),
            ("Prediction", "post", "/api/predict", {"json": TEST_BATCH}),
        ]:
            response, elapsed_ms = timed_request(client, method, path, **kwargs)
            ok = response.status_code == 200
            rows.append(
                {
                    "area": "Backend API",
                    "check": label,
                    "status": "Pass" if ok else "Fail",
                    "evidence": f"HTTP {response.status_code}; {elapsed_ms:.1f} ms",
                }
            )
            latencies.append({"endpoint": label, "latency_ms": elapsed_ms, "status": response.status_code})

        invalid_humidity = dict(TEST_BATCH)
        invalid_humidity["humidity"] = 140
        response, elapsed_ms = timed_request(client, "post", "/api/predict", json=invalid_humidity)
        rows.append(
            {
                "area": "Validation",
                "check": "Reject humidity above 100%",
                "status": "Pass" if response.status_code == 400 else "Fail",
                "evidence": f"HTTP {response.status_code}; {elapsed_ms:.1f} ms",
            }
        )

        bad_fish = dict(TEST_BATCH)
        bad_fish["fishType"] = "unknown"
        response, elapsed_ms = timed_request(client, "post", "/api/predict", json=bad_fish)
        rows.append(
            {
                "area": "Validation",
                "check": "Reject unsupported fish type",
                "status": "Pass" if response.status_code == 400 else "Fail",
                "evidence": f"HTTP {response.status_code}; {elapsed_ms:.1f} ms",
            }
        )

        negative_time = dict(TEST_BATCH)
        negative_time["elapsedTime"] = -5
        response, elapsed_ms = timed_request(client, "post", "/api/predict", json=negative_time)
        rows.append(
            {
                "area": "Validation",
                "check": "Reject negative elapsed time",
                "status": "Pass" if response.status_code == 400 else "Fail",
                "evidence": f"HTTP {response.status_code}; {elapsed_ms:.1f} ms",
            }
        )

        demo_start_payload = {
            "fishType": "catfish",
            "initialWeight": 2.0,
            "currentWeight": 2.0,
            "temperature": 60,
            "humidity": 45,
            "elapsedTime": 0,
        }
        response, elapsed_ms = timed_request(client, "post", "/api/demo/start", json=demo_start_payload)
        start_data = response.get_json() or {}
        demo_snapshots.append(start_data)
        rows.append(
            {
                "area": "Demo feed",
                "check": "Start backend sensor feed",
                "status": "Pass" if response.status_code == 200 and start_data.get("reading", {}).get("running") else "Fail",
                "evidence": f"HTTP {response.status_code}; source={start_data.get('reading', {}).get('source', 'n/a')}",
            }
        )

        # Let the backend demo feed advance by one configured tick.
        reading = backend_app._latest_reading
        if reading is not None:
            reading["lastTickAt"] -= reading["tickSeconds"]

        response, elapsed_ms = timed_request(client, "get", "/api/readings/latest")
        latest_data = response.get_json() or {}
        demo_snapshots.append(latest_data)
        latest_reading = latest_data.get("reading") or {}
        rows.append(
            {
                "area": "Demo feed",
                "check": "Weight changes after one tick",
                "status": "Pass" if latest_reading.get("currentWeight", 2.0) < 2.0 else "Fail",
                "evidence": (
                    f"{start_data.get('reading', {}).get('currentWeight', 'n/a')}kg -> "
                    f"{latest_reading.get('currentWeight', 'n/a')}kg"
                ),
            }
        )
        latencies.append({"endpoint": "Latest reading", "latency_ms": elapsed_ms, "status": response.status_code})

        hardware_payload = dict(TEST_BATCH)
        response, elapsed_ms = timed_request(client, "post", "/api/readings", json=hardware_payload)
        hardware_data = response.get_json() or {}
        rows.append(
            {
                "area": "Hardware-ready route",
                "check": "Accept hardware-style reading",
                "status": "Pass" if response.status_code == 200 and hardware_data.get("reading", {}).get("source") == "hardware" else "Fail",
                "evidence": f"HTTP {response.status_code}; source={hardware_data.get('reading', {}).get('source', 'n/a')}",
            }
        )
        latencies.append({"endpoint": "Hardware reading", "latency_ms": elapsed_ms, "status": response.status_code})

    pass_count = sum(1 for row in rows if row["status"] == "Pass")
    evaluated_count = sum(1 for row in rows if row["status"] in {"Pass", "Fail"})

    return {
        "rows": rows,
        "latencies": latencies,
        "demo_snapshots": demo_snapshots,
        "pass_count": pass_count,
        "evaluated_count": evaluated_count,
        "pass_rate": pass_count / evaluated_count * 100 if evaluated_count else 0,
        "build": build,
    }


def draw_scope_diagram(results: dict[str, object]) -> Path:
    path = OUTPUT_DIR / "01_app_evaluation_scope.png"
    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(0.055, 0.88, "APP EVALUATION SCOPE", fontsize=18, fontweight="bold", color=MUTED)
    ax.text(0.055, 0.80, "What Was Evaluated?", fontsize=44, fontweight="bold", color=INK)
    ax.text(
        0.055,
        0.73,
        "One application was evaluated: the fish kiln dashboard made of React frontend, Flask backend, prediction API, and demo sensor-feed workflow.",
        fontsize=19,
        color=MUTED,
    )

    blocks = [
        ("React dashboard", "Form, metrics, alerts, live feed display", 0.07, 0.42, TEAL),
        ("Flask API", "/api/predict, /api/readings, /api/health", 0.31, 0.42, BLUE),
        ("AI integration", "Loads model.pkl and returns drying prediction", 0.55, 0.42, GREEN),
        ("Validation", "Rejects invalid fish, humidity, time, weight", 0.79, 0.42, AMBER),
    ]
    for title, body, x, y, color in blocks:
        ax.add_patch(plt.Rectangle((x, y), 0.16, 0.19, facecolor=CARD, edgecolor=LINE, linewidth=2))
        ax.add_patch(plt.Rectangle((x, y + 0.165), 0.16, 0.025, facecolor=color, edgecolor=color))
        ax.text(x + 0.015, y + 0.12, title, fontsize=18, fontweight="bold", color=INK)
        ax.text(x + 0.015, y + 0.065, fill(body, 24), fontsize=12.5, color=MUTED, va="top")

    for x in [0.24, 0.48, 0.72]:
        ax.annotate("", xy=(x + 0.055, 0.515), xytext=(x - 0.015, 0.515), arrowprops=dict(arrowstyle="->", lw=2, color=MUTED))

    ax.add_patch(plt.Rectangle((0.07, 0.16), 0.86, 0.12, facecolor=CARD, edgecolor=LINE, linewidth=2))
    ax.text(0.095, 0.225, f"{results['pass_count']} / {results['evaluated_count']} checks passed", fontsize=30, fontweight="bold", color=GREEN)
    ax.text(0.095, 0.18, "This is application testing, not ML-model accuracy testing.", fontsize=16, color=MUTED)
    ax.text(0.79, 0.18, "Fish Kiln App", fontsize=16, fontweight="bold", color=MUTED)

    save(fig, path)
    return path


def draw_validation_matrix(results: dict[str, object]) -> Path:
    path = OUTPUT_DIR / "02_app_validation_matrix.png"
    rows = results["rows"]

    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    ax.text(0.055, 0.89, "APP TEST RESULTS", fontsize=18, fontweight="bold", color=MUTED)
    ax.text(0.055, 0.82, "Application Validation Matrix", fontsize=42, fontweight="bold", color=INK)

    table_ax = fig.add_axes([0.055, 0.10, 0.89, 0.64])
    table_ax.set_axis_off()

    headers = ["Area", "Check", "Status", "Evidence"]
    cell_text = [[row["area"], row["check"], row["status"], row["evidence"]] for row in rows]
    table = table_ax.table(
        cellText=cell_text,
        colLabels=headers,
        colWidths=[0.18, 0.34, 0.12, 0.36],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.8)
    table.scale(1, 1.55)

    for (row_idx, col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(LINE)
        if row_idx == 0:
            cell.set_facecolor(INK)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif col_idx == 2:
            status = cell.get_text().get_text()
            if status == "Pass":
                cell.set_facecolor("#e3f5ed")
                cell.get_text().set_color(GREEN)
            elif status == "Fail":
                cell.set_facecolor("#fde9e7")
                cell.get_text().set_color(RED)
            else:
                cell.set_facecolor("#fff1dc")
                cell.get_text().set_color(AMBER)
            cell.get_text().set_fontweight("bold")
        elif row_idx % 2 == 0:
            cell.set_facecolor("#f9fbf9")
        else:
            cell.set_facecolor("white")

    save(fig, path)
    return path


def draw_response_time_chart(results: dict[str, object]) -> Path:
    path = OUTPUT_DIR / "03_api_response_time.png"
    latencies = results["latencies"]
    labels = [item["endpoint"] for item in latencies]
    values = [float(item["latency_ms"]) for item in latencies]

    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    ax = fig.add_axes([0.09, 0.16, 0.78, 0.60])
    colors = [TEAL, BLUE, GREEN, AMBER, "#5b6770"][: len(values)]
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=15)
    ax.invert_yaxis()
    ax.set_xlabel("Local backend test-client response time (ms)", fontsize=14, fontweight="bold")
    ax.grid(True, axis="x", color=LINE)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i, value in enumerate(values):
        ax.text(value + max(values) * 0.02, i, f"{value:.1f} ms", va="center", fontsize=14, fontweight="bold", color=INK)

    fig.text(0.055, 0.89, "APP PERFORMANCE CHECK", fontsize=18, fontweight="bold", color=MUTED)
    fig.text(0.055, 0.82, "Backend API Response Time", fontsize=42, fontweight="bold", color=INK)
    fig.text(
        0.055,
        0.77,
        "Measured with Flask test client on the local prototype environment; deployed internet latency can be higher.",
        fontsize=17,
        color=MUTED,
    )

    save(fig, path)
    return path


def draw_demo_feed_chart(results: dict[str, object]) -> Path:
    path = OUTPUT_DIR / "04_demo_sensor_feed.png"
    snapshots = [item for item in results["demo_snapshots"] if item.get("reading")]
    if len(snapshots) < 2:
        return path

    start = snapshots[0]
    latest = snapshots[-1]
    start_reading = start["reading"]
    latest_reading = latest["reading"]
    latest_prediction = latest["prediction"]

    labels = ["Start", "+5 simulated min"]
    weights = [float(start_reading["currentWeight"]), float(latest_reading["currentWeight"])]
    elapsed = [float(start_reading["elapsedTime"]), float(latest_reading["elapsedTime"])]
    progress = [0.0, float(latest_prediction["dryingProgress"])]

    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.text(0.055, 0.89, "DEMO SENSOR-FEED RESULT", fontsize=18, fontweight="bold", color=MUTED)
    ax.text(0.055, 0.82, "Backend Simulated Reading Update", fontsize=42, fontweight="bold", color=INK)
    ax.text(
        0.055,
        0.76,
        "For demonstration, the backend temporarily behaves like a sensor feed. Real hardware can later POST readings to the same backend.",
        fontsize=17,
        color=MUTED,
    )

    chart = fig.add_axes([0.08, 0.18, 0.48, 0.48])
    x = np.arange(len(labels))
    chart.plot(x, weights, marker="o", linewidth=4, markersize=12, color=TEAL)
    chart.set_xticks(x)
    chart.set_xticklabels(labels, fontsize=15)
    chart.set_ylabel("Current weight (kg)", fontsize=14, fontweight="bold")
    chart.grid(True, axis="y", color=LINE)
    for spine in chart.spines.values():
        spine.set_visible(False)
    for i, value in enumerate(weights):
        chart.text(i, value + 0.01, f"{value:.3f} kg", ha="center", fontsize=15, fontweight="bold", color=INK)

    cards = [
        ("Elapsed time", f"{elapsed[0]:.0f} -> {elapsed[1]:.0f} min", BLUE),
        ("Drying progress", f"{progress[-1]:.2f}%", GREEN),
        ("Remaining time", f"{latest_prediction['estimatedRemainingTime']} min", AMBER),
        ("Status", latest_prediction["dryingStatus"], TEAL),
    ]
    for index, (title, value, color) in enumerate(cards):
        x0 = 0.62
        y0 = 0.55 - index * 0.105
        ax.add_patch(plt.Rectangle((x0, y0), 0.30, 0.078, facecolor=CARD, edgecolor=LINE, linewidth=2))
        ax.add_patch(plt.Rectangle((x0, y0), 0.012, 0.078, facecolor=color, edgecolor=color))
        ax.text(x0 + 0.025, y0 + 0.045, title, fontsize=13, color=MUTED, fontweight="bold")
        ax.text(x0 + 0.025, y0 + 0.016, str(value), fontsize=18, color=INK, fontweight="bold")

    save(fig, path)
    return path


def draw_scorecard(results: dict[str, object]) -> Path:
    path = OUTPUT_DIR / "05_app_scorecard.png"
    rows = results["rows"]
    categories = {
        "Frontend": ["Frontend"],
        "Backend API": ["Backend API"],
        "Validation": ["Validation"],
        "Feed readiness": ["Demo feed", "Hardware-ready route"],
    }
    scores = []
    for category, areas in categories.items():
        category_rows = [row for row in rows if row["area"] in areas and row["status"] in {"Pass", "Fail"}]
        passed = sum(1 for row in category_rows if row["status"] == "Pass")
        score = passed / len(category_rows) * 100 if category_rows else 0
        scores.append((category, score, passed, len(category_rows)))

    fig = plt.figure(figsize=(16, 9), facecolor=SURFACE)
    ax = fig.add_axes([0.08, 0.18, 0.82, 0.55])
    labels = [item[0] for item in scores]
    values = [item[1] for item in scores]
    colors = [TEAL, BLUE, GREEN, AMBER]
    bars = ax.bar(labels, values, color=colors, width=0.56)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Pass rate (%)", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", color=LINE)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="x", labelsize=15)
    ax.tick_params(axis="y", labelsize=12)
    for bar, (_, score, passed, total) in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 3,
            f"{score:.0f}%\n{passed}/{total}",
            ha="center",
            fontsize=15,
            fontweight="bold",
            color=INK,
        )

    fig.text(0.055, 0.89, "APP EVALUATION SUMMARY", fontsize=18, fontweight="bold", color=MUTED)
    fig.text(0.055, 0.82, "Prototype App Scorecard", fontsize=42, fontweight="bold", color=INK)
    fig.text(
        0.055,
        0.76,
        "These are software checks for the prototype application; they are separate from the ML model metrics.",
        fontsize=17,
        color=MUTED,
    )
    fig.text(0.73, 0.09, f"Overall: {results['pass_count']}/{results['evaluated_count']} checks passed", fontsize=18, fontweight="bold", color=GREEN)

    save(fig, path)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = run_app_evaluation()
    paths = [
        draw_scope_diagram(results),
        draw_validation_matrix(results),
        draw_response_time_chart(results),
        draw_demo_feed_chart(results),
        draw_scorecard(results),
    ]

    print("Application evaluation images written to:")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
