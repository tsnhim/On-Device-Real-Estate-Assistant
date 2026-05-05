from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from textwrap import wrap


ROOT = Path(__file__).resolve().parent
INPUT_JSON = ROOT / "all_benchmarks_final.json"
OUT_DIR = ROOT / "tradeoff_plots"

COLORS = {
    "Baseline": "#3b5b92",
    "Quantization": "#087f5b",
    "Pruning": "#d9480f",
    "Combined": "#7048e8",
}


def load_rows() -> list[dict]:
    records = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    rows = []
    baseline = next(r for r in records if r["model_name"] == "Baseline")
    base_f1 = baseline["quality"]["token_f1"]
    base_size = baseline["efficiency"]["disk_size_mb"]
    base_latency = baseline["efficiency"]["latency_ms_mean"]

    for record in records:
        name = record["model_name"]
        q = record["quality"]
        e = record["efficiency"]
        row = {
            "model_name": name,
            "family": classify_family(name),
            "token_f1": float(q["token_f1"] or 0),
            "disk_size_mb": float(e["disk_size_mb"]),
            "param_count": int(e["param_count"]),
            "latency_ms_mean": float(e["latency_ms_mean"]),
            "latency_ms_p95": float(e["latency_ms_p95"]),
            "examples_per_second": float(e["examples_per_second"]),
            "tokens_per_second": float(e["generated_tokens_per_second"]),
            "load_time_s": float(e["load_time_s"]),
        }
        row["quality_retention_pct"] = 100 * row["token_f1"] / base_f1 if base_f1 else 0
        row["size_reduction_pct"] = 100 * (base_size - row["disk_size_mb"]) / base_size
        row["latency_reduction_pct"] = 100 * (base_latency - row["latency_ms_mean"]) / base_latency
        row["latency_speedup"] = base_latency / row["latency_ms_mean"] if row["latency_ms_mean"] else 0
        rows.append(row)
    return rows


def classify_family(name: str) -> str:
    if name == "Baseline":
        return "Baseline"
    has_quant = any(term in name for term in ["INT8", "FP16", "BF16"])
    has_prune = "Prune" in name
    if has_quant and has_prune:
        return "Combined"
    if has_quant:
        return "Quantization"
    if has_prune:
        return "Pruning"
    return "Other"


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def short_label(label: str, width: int = 18) -> list[str]:
    return wrap(label, width=width, break_long_words=False) or [label]


def svg_page(width: int, height: int, title: str, body: str) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <style>
    text {{ font-family: Arial, Helvetica, sans-serif; fill: #1f2933; }}
    .title {{ font-size: 22px; font-weight: 700; }}
    .subtitle {{ font-size: 12px; fill: #52606d; }}
    .axis {{ stroke: #9aa5b1; stroke-width: 1; }}
    .grid {{ stroke: #d9e2ec; stroke-width: 1; stroke-dasharray: 4 5; }}
    .label {{ font-size: 11px; fill: #334e68; }}
    .tick {{ font-size: 10px; fill: #52606d; }}
    .legend {{ font-size: 12px; fill: #334e68; }}
  </style>
  <text class="title" x="36" y="34">{esc(title)}</text>
{body}
</svg>
"""


def linear_scale(value: float, domain: tuple[float, float], range_: tuple[float, float]) -> float:
    d0, d1 = domain
    r0, r1 = range_
    if d1 == d0:
        return (r0 + r1) / 2
    return r0 + (value - d0) * (r1 - r0) / (d1 - d0)


def log_scale(value: float, domain: tuple[float, float], range_: tuple[float, float]) -> float:
    safe_value = max(value, 1e-9)
    d0, d1 = math.log10(domain[0]), math.log10(domain[1])
    return linear_scale(math.log10(safe_value), (d0, d1), range_)


def legend(x: int, y: int) -> str:
    parts = []
    dx = 0
    for family, color in COLORS.items():
        parts.append(f'<circle cx="{x + dx}" cy="{y}" r="6" fill="{color}"/>')
        parts.append(f'<text class="legend" x="{x + dx + 10}" y="{y + 4}">{esc(family)}</text>')
        dx += 128
    return "\n".join(parts)


def write_accuracy_and_retention(rows: list[dict], manifest: list[dict]) -> None:
    data = sorted(rows, key=lambda row: row["token_f1"], reverse=True)
    width = 1180
    row_h = 44
    height = 120 + row_h * len(data)
    left = 320
    top = 78
    f1_w = 360
    retention_w = 300
    max_f1 = max(row["token_f1"] for row in data)
    max_retention = max(row["quality_retention_pct"] for row in data)
    body = [
        '<text class="subtitle" x="36" y="55">Shows actual Token F1 and the same score as percentage retained versus baseline.</text>',
        f'<text class="label" x="{left}" y="70">Token F1</text>',
        f'<text class="label" x="{left + f1_w + 112}" y="70">Retention vs baseline</text>',
    ]
    for index, row in enumerate(data):
        y = top + index * row_h
        color = COLORS.get(row["family"], "#6b7280")
        f1_bar_w = linear_scale(row["token_f1"], (0, max_f1), (0, f1_w))
        retention_bar_w = linear_scale(row["quality_retention_pct"], (0, max_retention), (0, retention_w))
        label_lines = short_label(row["model_name"], 31)
        for line_index, line in enumerate(label_lines[:2]):
            body.append(f'<text class="label" x="28" y="{y + 16 + line_index * 13}">{esc(line)}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{f1_w}" height="24" fill="#eef2f7" stroke="#d9e2ec"/>')
        body.append(f'<rect x="{left}" y="{y}" width="{f1_bar_w:.1f}" height="24" fill="{color}" fill-opacity="0.84"/>')
        body.append(f'<text class="label" x="{left + f1_bar_w + 8:.1f}" y="{y + 16}">{row["token_f1"]:.4f}</text>')
        retention_x = left + f1_w + 112
        body.append(f'<rect x="{retention_x}" y="{y}" width="{retention_w}" height="24" fill="#eef2f7" stroke="#d9e2ec"/>')
        body.append(f'<rect x="{retention_x}" y="{y}" width="{retention_bar_w:.1f}" height="24" fill="{color}" fill-opacity="0.68"/>')
        body.append(f'<text class="label" x="{retention_x + retention_bar_w + 8:.1f}" y="{y + 16}">{row["quality_retention_pct"]:.1f}%</text>')

    path = OUT_DIR / "01_accuracy_token_f1_and_retention.svg"
    path.write_text(svg_page(width, height, "Accuracy: Token F1 and Baseline Retention", "\n".join(body)), encoding="utf-8")
    manifest.append({"file": path.name, "title": "Accuracy: Token F1 and baseline retention"})


def write_quality_latency_scatter(rows: list[dict], manifest: list[dict]) -> None:
    width, height = 1120, 720
    left, right, top, bottom = 96, 48, 78, 92
    plot_w = width - left - right
    plot_h = height - top - bottom
    latencies = [r["latency_ms_mean"] for r in rows]
    max_size = max(r["disk_size_mb"] for r in rows)
    x_domain = (min(latencies) * 0.75, max(latencies) * 1.25)
    y_domain = (0, max(r["token_f1"] for r in rows) * 1.12)
    body = [f'<text class="subtitle" x="36" y="55">Bubble size = disk size. X-axis uses log scale because latency spans two orders of magnitude.</text>']
    body.append(legend(650, 34))
    body.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')

    for tick in [100, 200, 500, 1000, 2000, 5000, 10000, 20000]:
        if x_domain[0] <= tick <= x_domain[1]:
            x = log_scale(tick, x_domain, (left, left + plot_w))
            body.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>')
            body.append(f'<text class="tick" x="{x - 15:.1f}" y="{top + plot_h + 22}">{tick}</text>')
    for tick in [0, 0.05, 0.10, 0.15, 0.20, 0.25]:
        y = linear_scale(tick, y_domain, (top + plot_h, top))
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        body.append(f'<text class="tick" x="{left - 44}" y="{y + 4:.1f}">{tick:.2f}</text>')

    for row in sorted(rows, key=lambda r: r["disk_size_mb"], reverse=True):
        x = log_scale(row["latency_ms_mean"], x_domain, (left, left + plot_w))
        y = linear_scale(row["token_f1"], y_domain, (top + plot_h, top))
        radius = 6 + 16 * math.sqrt(row["disk_size_mb"] / max_size)
        color = COLORS.get(row["family"], "#6b7280")
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.72" stroke="#1f2933" stroke-width="0.7"/>')
        body.append(f'<text class="label" x="{x + radius + 4:.1f}" y="{y + 4:.1f}">{esc(row["model_name"])}</text>')

    body.append(f'<text class="label" x="{left + plot_w / 2 - 80:.1f}" y="{height - 30}">Mean latency (ms/example, log scale) - lower is better</text>')
    body.append(f'<text class="label" transform="translate(24 {top + plot_h / 2 + 70:.1f}) rotate(-90)">Token F1 - higher is better</text>')
    path = OUT_DIR / "03_quality_vs_latency_bubble.svg"
    path.write_text(svg_page(width, height, "Quality vs Latency Tradeoff", "\n".join(body)), encoding="utf-8")
    manifest.append({"file": path.name, "title": "Quality vs latency bubble plot"})


def write_retention_size_scatter(rows: list[dict], manifest: list[dict]) -> None:
    width, height = 1040, 660
    left, right, top, bottom = 86, 52, 78, 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_domain = (min(r["size_reduction_pct"] for r in rows) - 3, max(r["size_reduction_pct"] for r in rows) + 5)
    y_domain = (0, max(r["quality_retention_pct"] for r in rows) * 1.12)
    body = ['<text class="subtitle" x="36" y="55">Best region is upper-right: high quality retention and high model-size reduction.</text>']
    body.append(legend(594, 34))
    body.append(f'<line class="axis" x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}"/>')
    body.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}"/>')
    for tick in [0, 10, 20, 30, 40, 50, 60, 70]:
        x = linear_scale(tick, x_domain, (left, left + plot_w))
        body.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}"/>')
        body.append(f'<text class="tick" x="{x - 10:.1f}" y="{top + plot_h + 22}">{tick}%</text>')
    for tick in [0, 25, 50, 75, 100, 125]:
        y = linear_scale(tick, y_domain, (top + plot_h, top))
        body.append(f'<line class="grid" x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}"/>')
        body.append(f'<text class="tick" x="{left - 45}" y="{y + 4:.1f}">{tick}%</text>')
    for row in rows:
        x = linear_scale(row["size_reduction_pct"], x_domain, (left, left + plot_w))
        y = linear_scale(row["quality_retention_pct"], y_domain, (top + plot_h, top))
        color = COLORS.get(row["family"], "#6b7280")
        body.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="{color}" fill-opacity="0.82"/>')
        body.append(f'<text class="label" x="{x + 12:.1f}" y="{y + 4:.1f}">{esc(row["model_name"])}</text>')
    body.append(f'<text class="label" x="{left + plot_w / 2 - 88:.1f}" y="{height - 28}">Disk size reduction vs baseline</text>')
    body.append(f'<text class="label" transform="translate(24 {top + plot_h / 2 + 76:.1f}) rotate(-90)">Token F1 retention vs baseline</text>')
    path = OUT_DIR / "04_quality_retention_vs_size_reduction.svg"
    path.write_text(svg_page(width, height, "Quality Retention vs Size Reduction", "\n".join(body)), encoding="utf-8")
    manifest.append({"file": path.name, "title": "Quality retention vs size reduction"})


def write_horizontal_bar(
    rows: list[dict],
    manifest: list[dict],
    filename: str,
    title: str,
    metric: str,
    axis_label: str,
    lower_better: bool = False,
    top_n: int | None = None,
) -> None:
    data = sorted(rows, key=lambda r: r[metric], reverse=not lower_better)
    if top_n:
        data = data[:top_n]
    width = 1120
    row_h = 42
    height = 112 + row_h * len(data)
    left, right, top = 316, 70, 72
    bar_w = width - left - right
    max_val = max(r[metric] for r in data) or 1
    body = [f'<text class="subtitle" x="36" y="55">{esc(axis_label)}</text>']
    for idx, row in enumerate(data):
        y = top + idx * row_h
        color = COLORS.get(row["family"], "#6b7280")
        w = linear_scale(row[metric], (0, max_val), (0, bar_w))
        label_lines = short_label(row["model_name"], 31)
        for line_idx, line in enumerate(label_lines[:2]):
            body.append(f'<text class="label" x="28" y="{y + 17 + line_idx * 13}">{esc(line)}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{w:.1f}" height="25" fill="{color}" fill-opacity="0.84"/>')
        value = row[metric]
        text_value = f"{value:.2f}" if value < 100 else f"{value:,.1f}"
        body.append(f'<text class="label" x="{left + w + 8:.1f}" y="{y + 17}">{text_value}</text>')
    path = OUT_DIR / filename
    path.write_text(svg_page(width, height, title, "\n".join(body)), encoding="utf-8")
    manifest.append({"file": path.name, "title": title})


def write_latency_detail(rows: list[dict], manifest: list[dict]) -> None:
    data = sorted(rows, key=lambda r: r["latency_ms_mean"])
    width = 1120
    row_h = 42
    height = 118 + row_h * len(data)
    left, right, top = 316, 80, 78
    bar_w = width - left - right
    max_val = max(r["latency_ms_p95"] for r in data)
    body = ['<text class="subtitle" x="36" y="55">Mean latency bars with P95 marker. Lower is better.</text>']
    body.append('<rect x="760" y="28" width="18" height="12" fill="#1971c2" fill-opacity="0.78"/><text class="legend" x="784" y="39">Mean</text>')
    body.append('<line x1="858" y1="34" x2="890" y2="34" stroke="#c92a2a" stroke-width="3"/><text class="legend" x="898" y="39">P95</text>')
    for idx, row in enumerate(data):
        y = top + idx * row_h
        mean_w = linear_scale(row["latency_ms_mean"], (0, max_val), (0, bar_w))
        p95_x = left + linear_scale(row["latency_ms_p95"], (0, max_val), (0, bar_w))
        label_lines = short_label(row["model_name"], 31)
        for line_idx, line in enumerate(label_lines[:2]):
            body.append(f'<text class="label" x="28" y="{y + 17 + line_idx * 13}">{esc(line)}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{mean_w:.1f}" height="25" fill="#1971c2" fill-opacity="0.78"/>')
        body.append(f'<line x1="{p95_x:.1f}" y1="{y - 2}" x2="{p95_x:.1f}" y2="{y + 29}" stroke="#c92a2a" stroke-width="3"/>')
        body.append(f'<text class="label" x="{left + mean_w + 8:.1f}" y="{y + 17}">{row["latency_ms_mean"]:,.1f} ms</text>')
    path = OUT_DIR / "05_latency_mean_and_p95.svg"
    path.write_text(svg_page(width, height, "Latency: Mean and P95", "\n".join(body)), encoding="utf-8")
    manifest.append({"file": path.name, "title": "Latency mean and P95"})


def write_pruning_structure_comparison(rows: list[dict], manifest: list[dict]) -> None:
    prune_rows = [
        row
        for row in rows
        if row["family"] == "Pruning"
        and ("Structured" in row["model_name"] or "Unstructured" in row["model_name"])
    ]
    data = sorted(
        prune_rows,
        key=lambda row: (
            0 if row["model_name"].startswith("Unstructured") else 1,
            -row["token_f1"],
        ),
    )
    metrics = [
        ("token_f1", "Token F1", False),
        ("examples_per_second", "Throughput", False),
        ("latency_ms_mean", "Latency", True),
        ("disk_size_mb", "Size", True),
    ]
    ranges = {}
    for metric, _, _ in metrics:
        values = [row[metric] for row in data]
        ranges[metric] = (min(values), max(values))

    def normalized(value: float, low: float, high: float, lower_better: bool) -> float:
        if high == low:
            score = 0.5
        else:
            score = (value - low) / (high - low)
        return 1 - score if lower_better else score

    width = 1180
    row_h = 56
    height = 128 + row_h * len(data)
    left = 328
    top = 88
    metric_w = 180
    gap = 34
    body = [
        '<text class="subtitle" x="36" y="55">Prune-only models. Bars are normalized within this subset; longer is better for every metric.</text>',
        '<rect x="760" y="28" width="14" height="14" fill="#d9480f" fill-opacity="0.84"/><text class="legend" x="782" y="40">Unstructured</text>',
        '<rect x="890" y="28" width="14" height="14" fill="#7048e8" fill-opacity="0.84"/><text class="legend" x="912" y="40">Structured</text>',
    ]
    for index, (_, label, _) in enumerate(metrics):
        x = left + index * (metric_w + gap)
        body.append(f'<text class="label" x="{x}" y="76">{esc(label)}</text>')

    for row_index, row in enumerate(data):
        y = top + row_index * row_h
        is_structured = row["model_name"].startswith("Structured")
        color = "#7048e8" if is_structured else "#d9480f"
        label_lines = short_label(row["model_name"], 34)
        for line_index, line in enumerate(label_lines[:2]):
            body.append(f'<text class="label" x="28" y="{y + 17 + line_index * 13}">{esc(line)}</text>')

        for metric_index, (metric, _, lower_better) in enumerate(metrics):
            x = left + metric_index * (metric_w + gap)
            low, high = ranges[metric]
            score = normalized(row[metric], low, high, lower_better)
            bar_w = score * metric_w
            visible_w = max(4.0, bar_w)
            body.append(f'<rect x="{x}" y="{y}" width="{metric_w}" height="25" fill="#eef2f7" stroke="#d9e2ec"/>')
            body.append(f'<rect x="{x}" y="{y}" width="{visible_w:.1f}" height="25" fill="{color}" fill-opacity="0.84"/>')
            value = row[metric]
            if metric == "token_f1":
                text_value = f"{value:.4f}"
            elif metric == "examples_per_second":
                text_value = f"{value:.2f}/s"
            elif metric == "latency_ms_mean":
                text_value = f"{value:.1f} ms"
            else:
                text_value = f"{value:.1f} MB"
            body.append(f'<text class="tick" x="{x}" y="{y + 43}">{text_value}</text>')

    path = OUT_DIR / "08_structured_vs_unstructured_pruning.svg"
    path.write_text(svg_page(width, height, "Structured vs Unstructured Pruning", "\n".join(body)), encoding="utf-8")
    manifest.append({"file": path.name, "title": "Structured vs unstructured pruning"})


ACCURACY_GATE_RETENTION = 85.0


def score_rows(rows: list[dict]) -> list[dict]:
    def norm(value: float, low: float, high: float, lower_better: bool = False) -> float:
        if high == low:
            return 0.5
        score = (value - low) / (high - low)
        return 1 - score if lower_better else score

    passed = [row for row in rows if row["quality_retention_pct"] >= ACCURACY_GATE_RETENTION]
    ranges = {}
    for metric in ["examples_per_second", "latency_ms_mean", "disk_size_mb"]:
        values = [r[metric] for r in passed]
        ranges[metric] = (min(values), max(values))

    scored = []
    for row in rows:
        out = dict(row)
        out["accuracy_gate_pass"] = row["quality_retention_pct"] >= ACCURACY_GATE_RETENTION
        if out["accuracy_gate_pass"]:
            out["efficiency_score"] = (
                0.40 * norm(row["latency_ms_mean"], *ranges["latency_ms_mean"], lower_better=True)
                + 0.35 * norm(row["examples_per_second"], *ranges["examples_per_second"])
                + 0.25 * norm(row["disk_size_mb"], *ranges["disk_size_mb"], lower_better=True)
            )
        else:
            out["efficiency_score"] = 0.0
        out["balanced_score"] = out["efficiency_score"]
        scored.append(out)
    return sorted(
        scored,
        key=lambda r: (
            not r["accuracy_gate_pass"],
            -r["efficiency_score"],
            -r["quality_retention_pct"],
        ),
    )


def write_gated_balanced_ranking(scored: list[dict], manifest: list[dict]) -> None:
    width = 1180
    row_h = 42
    height = 132 + row_h * len(scored)
    left = 360
    top = 88
    bar_w = 390
    body = [
        f'<text class="subtitle" x="36" y="55">Accuracy gate: Token F1 retention must be at least {ACCURACY_GATE_RETENTION:.0f}% of baseline. Passing models are ranked by efficiency: latency 40%, throughput 35%, disk size 25%.</text>',
        '<rect x="835" y="28" width="14" height="14" fill="#087f5b" fill-opacity="0.84"/><text class="legend" x="856" y="40">Pass gate</text>',
        '<rect x="940" y="28" width="14" height="14" fill="#9aa5b1" fill-opacity="0.84"/><text class="legend" x="961" y="40">Filtered out</text>',
        f'<text class="label" x="{left}" y="76">Efficiency score</text>',
        f'<text class="label" x="{left + bar_w + 120}" y="76">Token F1 retention</text>',
    ]
    for index, row in enumerate(scored):
        y = top + index * row_h
        passed = row["accuracy_gate_pass"]
        color = COLORS.get(row["family"], "#6b7280") if passed else "#9aa5b1"
        score_w = row["efficiency_score"] * bar_w
        label_lines = short_label(row["model_name"], 34)
        for line_index, line in enumerate(label_lines[:2]):
            body.append(f'<text class="label" x="28" y="{y + 16 + line_index * 13}">{esc(line)}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{bar_w}" height="24" fill="#eef2f7" stroke="#d9e2ec"/>')
        if passed:
            body.append(f'<rect x="{left}" y="{y}" width="{score_w:.1f}" height="24" fill="{color}" fill-opacity="0.84"/>')
            body.append(f'<text class="label" x="{left + score_w + 8:.1f}" y="{y + 16}">{row["efficiency_score"]:.3f}</text>')
        else:
            body.append(f'<rect x="{left}" y="{y}" width="8" height="24" fill="{color}" fill-opacity="0.84"/>')
            body.append(f'<text class="label" x="{left + 16}" y="{y + 16}">filtered</text>')
        retention_x = left + bar_w + 120
        retention_w = min(260, row["quality_retention_pct"] / 100 * 260)
        body.append(f'<rect x="{retention_x}" y="{y}" width="260" height="24" fill="#eef2f7" stroke="#d9e2ec"/>')
        body.append(f'<rect x="{retention_x}" y="{y}" width="{retention_w:.1f}" height="24" fill="{color}" fill-opacity="0.62"/>')
        body.append(f'<text class="label" x="{retention_x + retention_w + 8:.1f}" y="{y + 16}">{row["quality_retention_pct"]:.1f}%</text>')

    path = OUT_DIR / "09_balanced_score_ranking.svg"
    path.write_text(svg_page(width, height, "Gated Balanced Score Ranking", "\n".join(body)), encoding="utf-8")
    manifest.append({"file": path.name, "title": "Gated balanced score ranking"})


def write_scorecard(rows: list[dict], manifest: list[dict]) -> None:
    scored = score_rows(rows)
    path = OUT_DIR / "model_tradeoff_scorecard.csv"
    fields = [
        "model_name",
        "family",
        "balanced_score",
        "efficiency_score",
        "accuracy_gate_pass",
        "token_f1",
        "quality_retention_pct",
        "latency_ms_mean",
        "latency_ms_p95",
        "latency_speedup",
        "disk_size_mb",
        "size_reduction_pct",
        "examples_per_second",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in scored:
            writer.writerow({field: row[field] for field in fields})
    manifest.append({"file": path.name, "title": "Model tradeoff scorecard CSV"})

    write_gated_balanced_ranking(scored, manifest)


def write_html(manifest: list[dict]) -> None:
    cards = []
    for item in manifest:
        file = item["file"]
        title = item["title"]
        if file.endswith(".svg"):
            cards.append(f'<section><h2>{esc(title)}</h2><img src="{esc(file)}" alt="{esc(title)}"></section>')
        else:
            cards.append(f'<section><h2>{esc(title)}</h2><p><a href="{esc(file)}">{esc(file)}</a></p></section>')
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FLAN-T5 Optimization Tradeoff Plots</title>
  <style>
    body {{ margin: 24px; font-family: Arial, Helvetica, sans-serif; color: #1f2933; background: #f8fafc; }}
    h1 {{ margin-bottom: 4px; }}
    p {{ color: #52606d; }}
    section {{ margin: 28px 0; padding: 18px; background: white; border: 1px solid #d9e2ec; border-radius: 8px; }}
    img {{ width: 100%; height: auto; display: block; }}
    a {{ color: #0b7285; }}
  </style>
</head>
<body>
  <h1>FLAN-T5 Optimization Tradeoff Plots</h1>
  <p>Generated from <code>06_Visualize/all_benchmarks_final.json</code>.</p>
  {''.join(cards)}
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html_text, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    manifest: list[dict] = []
    write_accuracy_and_retention(rows, manifest)
    write_quality_latency_scatter(rows, manifest)
    write_retention_size_scatter(rows, manifest)
    write_latency_detail(rows, manifest)
    write_horizontal_bar(
        rows,
        manifest,
        "06_disk_size_comparison.svg",
        "Disk Size Comparison",
        "disk_size_mb",
        "Disk size in MB. Lower is better for on-device deployment.",
        lower_better=True,
    )
    write_horizontal_bar(
        rows,
        manifest,
        "07_throughput_comparison.svg",
        "Throughput Comparison",
        "examples_per_second",
        "Examples per second. Higher is better, but only useful if quality remains acceptable.",
    )
    write_pruning_structure_comparison(rows, manifest)
    write_scorecard(rows, manifest)
    with (OUT_DIR / "plot_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "title"])
        writer.writeheader()
        writer.writerows(manifest)
    write_html(manifest)
    print(f"Wrote {len(manifest)} assets to {OUT_DIR.relative_to(ROOT.parent)}")


if __name__ == "__main__":
    main()
