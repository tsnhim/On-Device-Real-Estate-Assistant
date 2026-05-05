from __future__ import annotations

import argparse
import csv
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_OPTIMIZED_ROOT = Path("benchmarks/optimized/flan_t5")
DEFAULT_BASELINE_ROOT = Path("benchmarks/runs")
DEFAULT_OUTPUT_DIR = Path("benchmarks/visualizations/flan_t5")

PALETTE = {
    "baseline": "#4C6A92",
    "quant_only": "#1B998B",
    "prune_only": "#E67E22",
    "prune_then_quant": "#C0392B",
    "quant_then_prune": "#8E44AD",
    "other": "#7F8C8D",
}

QUANT_ONLY_COLORS = {
    "dynamic_int8_cpu_ptq": "#136F63",
    "bf16_cast_gpu": "#2D6CDF",
    "fp16_cast_gpu": "#F39C12",
}


@dataclass
class RunRecord:
    name: str
    family: str
    group_label: str
    source: str
    status: str
    eval_count: int | None
    device: str | None
    runtime_target: str | None
    quantization_dtype: str | None
    layer_policy: str | None
    tags: list[str]
    disk_size_mb: float | None
    param_count: int | None
    eval_loss: float | None
    exact_match: float | None
    token_f1: float | None
    rouge_l_f1: float | None
    examples_per_second: float | None
    tokens_per_second: float | None
    latency_ms_mean: float | None
    avg_generated_tokens: float | None
    metrics_path: str | None
    manifest_path: str | None
    artifact_dir: str | None
    parent_names: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create FLAN-T5 benchmark comparison plots and a lightweight HTML dashboard."
    )
    parser.add_argument("--optimized-root", default=str(DEFAULT_OPTIMIZED_ROOT))
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--min-eval-count",
        type=int,
        default=1000,
        help="Only include runs with at least this many evaluation examples in the charts.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_family(raw_family: str | None) -> str:
    if not raw_family:
        return "other"
    if raw_family == "pruning":
        return "prune_only"
    return raw_family


def classify_group(record: RunRecord) -> str:
    family = record.family
    if family == "baseline":
        return "Baseline"
    if family == "quant_only":
        if record.device == "cpu" or (record.quantization_dtype or "").lower() == "qint8":
            return "Quant Only (CPU)"
        return "Quant Only (GPU)"
    if family == "prune_only":
        if "structured" in record.name:
            return "Prune Only (Structured)"
        return "Prune Only (Unstructured)"
    if family == "prune_then_quant":
        return "Prune -> Quant"
    if family == "quant_then_prune":
        return "Quant -> Prune"
    return "Other"


def read_optimized_runs(root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for strategy_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        manifest_path = strategy_dir / "strategy_manifest.json"
        metrics_path = strategy_dir / "benchmark" / "metrics.json"
        smoke_metrics_path = strategy_dir / "benchmark_smoke1" / "metrics.json"
        manifest = load_json(manifest_path) if manifest_path.exists() else {}
        metrics = load_json(metrics_path) if metrics_path.exists() else None
        status = "evaluated" if metrics else "artifact_only" if manifest else "unknown"
        if not metrics and smoke_metrics_path.exists():
            status = "smoke_only"
        parent_names = [item.get("strategy_name") for item in manifest.get("parent_sources", []) if item.get("strategy_name")]

        record = RunRecord(
            name=strategy_dir.name,
            family=normalize_family(manifest.get("strategy_family")),
            group_label="",
            source="optimized",
            status=status,
            eval_count=metrics.get("split", {}).get("eval_count") if metrics else None,
            device=metrics.get("model", {}).get("device") if metrics else manifest.get("benchmark_device"),
            runtime_target=metrics.get("model", {}).get("runtime_target") if metrics else manifest.get("runtime_target"),
            quantization_dtype=manifest.get("quantization_dtype"),
            layer_policy=manifest.get("layer_policy"),
            tags=list(manifest.get("tags", [])),
            disk_size_mb=metrics.get("model", {}).get("disk_size_mb") if metrics else None,
            param_count=metrics.get("model", {}).get("param_count") if metrics else None,
            eval_loss=metrics.get("quality", {}).get("eval_loss") if metrics else None,
            exact_match=metrics.get("quality", {}).get("exact_match") if metrics else None,
            token_f1=metrics.get("quality", {}).get("token_f1") if metrics else None,
            rouge_l_f1=metrics.get("quality", {}).get("rouge_l_f1") if metrics else None,
            examples_per_second=metrics.get("efficiency", {}).get("examples_per_second") if metrics else None,
            tokens_per_second=metrics.get("efficiency", {}).get("generated_tokens_per_second") if metrics else None,
            latency_ms_mean=metrics.get("efficiency", {}).get("latency_ms_mean") if metrics else None,
            avg_generated_tokens=metrics.get("quality", {}).get("avg_generated_tokens") if metrics else None,
            metrics_path=str(metrics_path).replace("\\", "/") if metrics else None,
            manifest_path=str(manifest_path).replace("\\", "/") if manifest else None,
            artifact_dir=str(strategy_dir).replace("\\", "/"),
            parent_names=parent_names,
        )
        record.group_label = classify_group(record)
        records.append(record)
    return records


def read_baseline_runs(root: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        if "optimized" in metrics_path.parts:
            continue
        metrics = load_json(metrics_path)
        run_name = f"{metrics_path.parent.parent.name}__{metrics_path.parent.name}"
        record = RunRecord(
            name=run_name,
            family="baseline",
            group_label="Baseline",
            source="benchmark",
            status="evaluated",
            eval_count=metrics.get("split", {}).get("eval_count"),
            device=metrics.get("model", {}).get("device"),
            runtime_target="baseline",
            quantization_dtype=None,
            layer_policy=None,
            tags=["baseline"],
            disk_size_mb=metrics.get("model", {}).get("disk_size_mb"),
            param_count=metrics.get("model", {}).get("param_count"),
            eval_loss=metrics.get("quality", {}).get("eval_loss"),
            exact_match=metrics.get("quality", {}).get("exact_match"),
            token_f1=metrics.get("quality", {}).get("token_f1"),
            rouge_l_f1=metrics.get("quality", {}).get("rouge_l_f1"),
            examples_per_second=metrics.get("efficiency", {}).get("examples_per_second"),
            tokens_per_second=metrics.get("efficiency", {}).get("generated_tokens_per_second"),
            latency_ms_mean=metrics.get("efficiency", {}).get("latency_ms_mean"),
            avg_generated_tokens=metrics.get("quality", {}).get("avg_generated_tokens"),
            metrics_path=str(metrics_path).replace("\\", "/"),
            manifest_path=None,
            artifact_dir=str(metrics_path.parent).replace("\\", "/"),
            parent_names=[],
        )
        records.append(record)
    return records


def to_number(value: float | int | None) -> float | None:
    if value is None:
        return None
    return float(value)


def filter_full_runs(records: list[RunRecord], min_eval_count: int) -> list[RunRecord]:
    return [
        record
        for record in records
        if record.status == "evaluated"
        and (record.eval_count or 0) >= min_eval_count
        and record.token_f1 is not None
        and record.disk_size_mb is not None
        and record.examples_per_second is not None
    ]


def write_csv(records: list[RunRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "name",
                "family",
                "group_label",
                "source",
                "status",
                "eval_count",
                "device",
                "runtime_target",
                "quantization_dtype",
                "layer_policy",
                "disk_size_mb",
                "param_count",
                "eval_loss",
                "exact_match",
                "token_f1",
                "rouge_l_f1",
                "examples_per_second",
                "tokens_per_second",
                "latency_ms_mean",
                "avg_generated_tokens",
                "metrics_path",
                "manifest_path",
                "artifact_dir",
                "parent_names",
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record.name,
                    record.family,
                    record.group_label,
                    record.source,
                    record.status,
                    record.eval_count,
                    record.device,
                    record.runtime_target,
                    record.quantization_dtype,
                    record.layer_policy,
                    record.disk_size_mb,
                    record.param_count,
                    record.eval_loss,
                    record.exact_match,
                    record.token_f1,
                    record.rouge_l_f1,
                    record.examples_per_second,
                    record.tokens_per_second,
                    record.latency_ms_mean,
                    record.avg_generated_tokens,
                    record.metrics_path,
                    record.manifest_path,
                    record.artifact_dir,
                    "|".join(record.parent_names),
                ]
            )


def axis_scale(values: list[float], padding_ratio: float = 0.08) -> tuple[float, float]:
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        delta = max(abs(low) * 0.1, 1.0)
        return low - delta, high + delta
    pad = (high - low) * padding_ratio
    return low - pad, high + pad


def svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        "text { font-family: Arial, sans-serif; fill: #18212f; }",
        ".title { font-size: 20px; font-weight: 700; }",
        ".subtitle { font-size: 11px; fill: #5B6576; }",
        ".axis { stroke: #5B6576; stroke-width: 1; }",
        ".grid { stroke: #DCE3EC; stroke-width: 1; }",
        ".label { font-size: 12px; }",
        ".tick { font-size: 10px; fill: #5B6576; }",
        ".legend { font-size: 11px; }",
        ".point-label { font-size: 10px; fill: #233142; }",
        ".bar-label { font-size: 10px; fill: #233142; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="#FBFCFE" />',
        f'<text class="title" x="28" y="34">{html.escape(title)}</text>',
    ]


def write_text(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def quant_color(record: RunRecord) -> str:
    return QUANT_ONLY_COLORS.get(record.name, PALETTE["quant_only"])


def scatter_svg(
    records: list[RunRecord],
    x_attr: str,
    y_attr: str,
    title: str,
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    width = 1080
    height = 720
    margin_left = 90
    margin_right = 240
    margin_top = 70
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    points = []
    for record in records:
        x_value = to_number(getattr(record, x_attr))
        y_value = to_number(getattr(record, y_attr))
        if x_value is None or y_value is None:
            continue
        points.append((record, x_value, y_value))

    xs = [x for _, x, _ in points]
    ys = [y for _, _, y in points]
    x_min, x_max = axis_scale(xs)
    y_min, y_max = axis_scale(ys)

    def sx(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return margin_top + plot_height - (value - y_min) / (y_max - y_min) * plot_height

    lines = svg_header(width, height, title)
    lines.append(
        '<text class="subtitle" x="28" y="54">Higher on the chart means better quality. Labels are strategy folder names.</text>'
    )

    tick_count = 5
    for idx in range(tick_count + 1):
        ratio = idx / tick_count
        x = margin_left + ratio * plot_width
        y = margin_top + ratio * plot_height
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_height}" />')
        lines.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_width}" y2="{y:.1f}" />')
        x_value = x_min + ratio * (x_max - x_min)
        y_value = y_max - ratio * (y_max - y_min)
        lines.append(f'<text class="tick" x="{x:.1f}" y="{margin_top + plot_height + 22:.1f}" text-anchor="middle">{x_value:.2f}</text>')
        lines.append(f'<text class="tick" x="{margin_left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end">{y_value:.2f}</text>')

    lines.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" />')
    lines.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" />')
    lines.append(f'<text class="label" x="{margin_left + plot_width / 2:.1f}" y="{height - 28}" text-anchor="middle">{html.escape(x_label)}</text>')
    lines.append(
        f'<text class="label" transform="translate(24,{margin_top + plot_height / 2:.1f}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>'
    )

    for record, x_value, y_value in points:
        color = PALETTE.get(record.family, PALETTE["other"])
        cx = sx(x_value)
        cy = sy(y_value)
        lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7" fill="{color}" fill-opacity="0.88" stroke="#ffffff" stroke-width="1.5" />')
        lines.append(f'<text class="point-label" x="{cx + 9:.1f}" y="{cy - 8:.1f}">{html.escape(record.name)}</text>')

    legend_y = 108
    legend_x = width - margin_right + 26
    for family in ["baseline", "quant_only", "prune_only", "prune_then_quant", "quant_then_prune"]:
        color = PALETTE.get(family, PALETTE["other"])
        label = family.replace("_", " ")
        lines.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="6" fill="{color}" />')
        lines.append(f'<text class="legend" x="{legend_x + 14}" y="{legend_y + 4}">{html.escape(label)}</text>')
        legend_y += 24

    lines.append("</svg>")
    write_text(lines, output_path)


def horizontal_bar_svg(
    records: list[RunRecord],
    title: str,
    metric_attr: str,
    metric_label: str,
    output_path: Path,
) -> None:
    ordered = sorted(
        [record for record in records if to_number(getattr(record, metric_attr)) is not None],
        key=lambda record: float(getattr(record, metric_attr)),
        reverse=True,
    )
    width = 1180
    bar_step = 34
    margin_top = 90
    margin_left = 320
    margin_right = 120
    margin_bottom = 60
    plot_height = max(len(ordered) * bar_step, 140)
    height = margin_top + plot_height + margin_bottom
    plot_width = width - margin_left - margin_right

    values = [float(getattr(record, metric_attr)) for record in ordered]
    _, x_max = axis_scale(values, padding_ratio=0.12)
    x_min = 0.0

    def sx(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_width

    lines = svg_header(width, height, title)
    lines.append(f'<text class="subtitle" x="28" y="54">Sorted by {html.escape(metric_label)}. Best fully evaluated runs appear at the top.</text>')

    tick_count = 5
    for idx in range(tick_count + 1):
        ratio = idx / tick_count
        x = margin_left + ratio * plot_width
        value = x_min + ratio * (x_max - x_min)
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_height}" />')
        lines.append(f'<text class="tick" x="{x:.1f}" y="{margin_top + plot_height + 22:.1f}" text-anchor="middle">{value:.2f}</text>')

    for index, record in enumerate(ordered):
        y = margin_top + index * bar_step
        bar_y = y + 6
        value = float(getattr(record, metric_attr))
        bar_end = sx(value)
        color = PALETTE.get(record.family, PALETTE["other"])
        lines.append(f'<rect x="{margin_left}" y="{bar_y}" width="{bar_end - margin_left:.1f}" height="20" rx="4" fill="{color}" fill-opacity="0.88" />')
        lines.append(f'<text class="label" x="{margin_left - 10}" y="{bar_y + 14}" text-anchor="end">{html.escape(record.name)}</text>')
        lines.append(f'<text class="bar-label" x="{bar_end + 8:.1f}" y="{bar_y + 14}">{value:.4f}</text>')

    lines.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" />')
    lines.append(f'<text class="label" x="{margin_left + plot_width / 2:.1f}" y="{height - 24}" text-anchor="middle">{html.escape(metric_label)}</text>')
    lines.append("</svg>")
    write_text(lines, output_path)


def summary_heatmap_svg(records: list[RunRecord], output_path: Path) -> None:
    top_records = sorted(records, key=lambda record: (record.token_f1 or -1), reverse=True)[:10]
    width = 1180
    row_height = 34
    col_widths = [280, 120, 120, 140, 120, 120]
    cols = [
        ("Strategy", "name"),
        ("Family", "group_label"),
        ("Token F1", "token_f1"),
        ("ROUGE-L F1", "rouge_l_f1"),
        ("Size MB", "disk_size_mb"),
        ("Ex/s", "examples_per_second"),
    ]
    height = 100 + (len(top_records) + 1) * row_height + 40
    x_positions = [28]
    for width_value in col_widths[:-1]:
        x_positions.append(x_positions[-1] + width_value)

    numeric_attrs = {"token_f1", "rouge_l_f1", "disk_size_mb", "examples_per_second"}
    minmax: dict[str, tuple[float, float]] = {}
    for _, attr in cols:
        if attr not in numeric_attrs:
            continue
        values = [float(getattr(record, attr)) for record in top_records if getattr(record, attr) is not None]
        minmax[attr] = (min(values), max(values)) if values else (0.0, 1.0)

    def score(attr: str, value: float) -> float:
        low, high = minmax[attr]
        if math.isclose(low, high):
            return 1.0
        base = (value - low) / (high - low)
        if attr == "disk_size_mb":
            base = 1.0 - base
        return max(0.0, min(base, 1.0))

    def color_for(attr: str, value: Any) -> str:
        if attr not in numeric_attrs or value is None:
            return "#F4F7FB"
        blend = score(attr, float(value))
        red = round(236 - blend * 104)
        green = round(111 + blend * 94)
        blue = round(99 + blend * 74)
        return f"rgb({red},{green},{blue})"

    lines = svg_header(width, height, "Benchmark Scorecard Heatmap")
    lines.append('<text class="subtitle" x="28" y="54">Quick read: greener cells are better, except size where smaller is better and therefore greener.</text>')

    header_y = 84
    for index, (col_name, _) in enumerate(cols):
        x = x_positions[index]
        w = col_widths[index]
        lines.append(f'<rect x="{x}" y="{header_y}" width="{w}" height="{row_height}" fill="#DCE7F5" stroke="#ffffff" />')
        lines.append(f'<text class="label" x="{x + 10}" y="{header_y + 22}">{html.escape(col_name)}</text>')

    for row_index, record in enumerate(top_records):
        y = header_y + row_height * (row_index + 1)
        for col_index, (_, attr) in enumerate(cols):
            x = x_positions[col_index]
            w = col_widths[col_index]
            value = getattr(record, attr)
            fill = color_for(attr, value)
            lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{row_height}" fill="{fill}" stroke="#ffffff" />')
            if isinstance(value, float):
                text = f"{value:.4f}" if value < 10 else f"{value:.2f}"
            else:
                text = str(value)
            lines.append(f'<text class="label" x="{x + 10}" y="{y + 22}">{html.escape(text)}</text>')

    lines.append("</svg>")
    write_text(lines, output_path)


def small_multiple_tradeoff_svg(
    records: list[RunRecord],
    family: str,
    family_title: str,
    output_path: Path,
) -> None:
    chosen = [record for record in records if record.family == family]
    metrics = [
        ("disk_size_mb", "token_f1", "Disk Size (MB)", "Token F1"),
        ("param_count", "token_f1", "Parameter Count", "Token F1"),
        ("latency_ms_mean", "token_f1", "Mean Latency (ms)", "Token F1"),
        ("disk_size_mb", "rouge_l_f1", "Disk Size (MB)", "ROUGE-L F1"),
        ("param_count", "rouge_l_f1", "Parameter Count", "ROUGE-L F1"),
        ("latency_ms_mean", "rouge_l_f1", "Mean Latency (ms)", "ROUGE-L F1"),
    ]
    width = 1380
    height = 980
    panel_w = 400
    panel_h = 330
    start_x = 40
    start_y = 90
    gap_x = 30
    gap_y = 34

    lines = svg_header(width, height, f"{family_title}: Quality vs Efficiency Tradeoffs")
    lines.append(
        '<text class="subtitle" x="28" y="54">Each panel shows how a quality metric moves against one important efficiency metric within the same strategy family.</text>'
    )

    for idx, (x_attr, y_attr, x_label, y_label) in enumerate(metrics):
        col = idx % 3
        row = idx // 3
        panel_x = start_x + col * (panel_w + gap_x)
        panel_y = start_y + row * (panel_h + gap_y)
        inner_left = panel_x + 58
        inner_right = panel_x + panel_w - 18
        inner_top = panel_y + 34
        inner_bottom = panel_y + panel_h - 44
        plot_w = inner_right - inner_left
        plot_h = inner_bottom - inner_top

        points = []
        for record in chosen:
            x_value = to_number(getattr(record, x_attr))
            y_value = to_number(getattr(record, y_attr))
            if x_value is None or y_value is None:
                continue
            points.append((record, x_value, y_value))
        xs = [x for _, x, _ in points]
        ys = [y for _, _, y in points]
        x_min, x_max = axis_scale(xs)
        y_min, y_max = axis_scale(ys)

        def sx(value: float) -> float:
            return inner_left + (value - x_min) / (x_max - x_min) * plot_w

        def sy(value: float) -> float:
            return inner_top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

        lines.append(f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="12" fill="#ffffff" stroke="#D8E1EC" />')
        lines.append(f'<text class="label" x="{panel_x + 16}" y="{panel_y + 22}">{html.escape(y_label)} vs {html.escape(x_label)}</text>')

        for tick in range(5):
            ratio = tick / 4
            gx = inner_left + ratio * plot_w
            gy = inner_top + ratio * plot_h
            x_value = x_min + ratio * (x_max - x_min)
            y_value = y_max - ratio * (y_max - y_min)
            lines.append(f'<line class="grid" x1="{gx:.1f}" y1="{inner_top}" x2="{gx:.1f}" y2="{inner_bottom}" />')
            lines.append(f'<line class="grid" x1="{inner_left}" y1="{gy:.1f}" x2="{inner_right}" y2="{gy:.1f}" />')
            lines.append(f'<text class="tick" x="{gx:.1f}" y="{inner_bottom + 18:.1f}" text-anchor="middle">{x_value:.2f}</text>')
            lines.append(f'<text class="tick" x="{inner_left - 8:.1f}" y="{gy + 4:.1f}" text-anchor="end">{y_value:.2f}</text>')

        lines.append(f'<line class="axis" x1="{inner_left}" y1="{inner_bottom}" x2="{inner_right}" y2="{inner_bottom}" />')
        lines.append(f'<line class="axis" x1="{inner_left}" y1="{inner_top}" x2="{inner_left}" y2="{inner_bottom}" />')
        lines.append(f'<text class="tick" x="{inner_left + plot_w / 2:.1f}" y="{panel_y + panel_h - 12}">{html.escape(x_label)}</text>')
        lines.append(
            f'<text class="tick" transform="translate({panel_x + 14},{inner_top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">{html.escape(y_label)}</text>'
        )

        for record, x_value, y_value in points:
            color = PALETTE.get(record.family, PALETTE["other"])
            cx = sx(x_value)
            cy = sy(y_value)
            lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6.5" fill="{color}" fill-opacity="0.9" stroke="#ffffff" stroke-width="1.3" />')
            lines.append(f'<text class="point-label" x="{cx + 8:.1f}" y="{cy - 7:.1f}">{html.escape(record.name)}</text>')

    lines.append("</svg>")
    write_text(lines, output_path)


def pareto_frontier(
    records: list[RunRecord],
    x_attr: str,
    y_attr: str,
    *,
    maximize_x: bool,
    maximize_y: bool,
) -> list[RunRecord]:
    frontier: list[RunRecord] = []
    for candidate in records:
        candidate_x = to_number(getattr(candidate, x_attr))
        candidate_y = to_number(getattr(candidate, y_attr))
        if candidate_x is None or candidate_y is None:
            continue
        dominated = False
        for other in records:
            if other is candidate:
                continue
            other_x = to_number(getattr(other, x_attr))
            other_y = to_number(getattr(other, y_attr))
            if other_x is None or other_y is None:
                continue
            x_better_or_equal = other_x >= candidate_x if maximize_x else other_x <= candidate_x
            y_better_or_equal = other_y >= candidate_y if maximize_y else other_y <= candidate_y
            x_strict = other_x > candidate_x if maximize_x else other_x < candidate_x
            y_strict = other_y > candidate_y if maximize_y else other_y < candidate_y
            if x_better_or_equal and y_better_or_equal and (x_strict or y_strict):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier


def quant_only_overview_svg(records: list[RunRecord], output_path: Path) -> None:
    chosen = [record for record in records if record.family == "quant_only"]
    width = 1120
    height = 760
    margin_left = 96
    margin_right = 240
    margin_top = 84
    margin_bottom = 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    points = [
        record
        for record in chosen
        if record.token_f1 is not None and record.latency_ms_mean is not None and record.disk_size_mb is not None
    ]
    xs = [float(record.latency_ms_mean) for record in points]
    ys = [float(record.token_f1) for record in points]
    size_min = min(float(record.disk_size_mb) for record in points)
    size_max = max(float(record.disk_size_mb) for record in points)
    x_min, x_max = axis_scale(xs, padding_ratio=0.12)
    y_min, y_max = axis_scale(ys, padding_ratio=0.12)

    def sx(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return margin_top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    def sr(value: float) -> float:
        if math.isclose(size_min, size_max):
            return 14.0
        return 10.0 + (value - size_min) / (size_max - size_min) * 20.0

    lines = svg_header(width, height, "Quant-Only Overview: Quality, Latency, and Size")
    lines.append(
        '<text class="subtitle" x="28" y="54">Left and up is better. Bubble size shows artifact size so you can spot the fastest or smallest option without losing the quality view.</text>'
    )

    for idx in range(6):
        ratio = idx / 5
        x = margin_left + ratio * plot_w
        y = margin_top + ratio * plot_h
        xv = x_min + ratio * (x_max - x_min)
        yv = y_max - ratio * (y_max - y_min)
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_h}" />')
        lines.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" />')
        lines.append(f'<text class="tick" x="{x:.1f}" y="{margin_top + plot_h + 22:.1f}" text-anchor="middle">{xv:.0f}</text>')
        lines.append(f'<text class="tick" x="{margin_left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end">{yv:.3f}</text>')

    lines.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" />')
    lines.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" />')
    lines.append(f'<text class="label" x="{margin_left + plot_w / 2:.1f}" y="{height - 26}" text-anchor="middle">Mean Latency (ms, lower is better)</text>')
    lines.append(
        f'<text class="label" transform="translate(28,{margin_top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">Token F1 (higher is better)</text>'
    )

    best_quality = max(points, key=lambda record: record.token_f1 or -1)
    fastest = min(points, key=lambda record: record.latency_ms_mean or float("inf"))
    smallest = min(points, key=lambda record: record.disk_size_mb or float("inf"))

    for record in points:
        color = quant_color(record)
        cx = sx(float(record.latency_ms_mean))
        cy = sy(float(record.token_f1))
        radius = sr(float(record.disk_size_mb))
        lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.42" stroke="{color}" stroke-width="2.4" />')
        lines.append(f'<text class="point-label" x="{cx + radius + 6:.1f}" y="{cy - radius - 3:.1f}">{html.escape(record.name)}</text>')

    summary_x = width - margin_right + 22
    legend_y = 116
    for record in points:
        color = quant_color(record)
        lines.append(f'<circle cx="{summary_x}" cy="{legend_y}" r="7" fill="{color}" stroke="{color}" stroke-width="2" />')
        lines.append(f'<text class="legend" x="{summary_x + 14}" y="{legend_y + 4}">{html.escape(record.name)}</text>')
        legend_y += 24
    legend_y += 12
    lines.append(f'<text class="legend" x="{summary_x}" y="{legend_y}">Best quality: {html.escape(best_quality.name)}</text>')
    lines.append(f'<text class="legend" x="{summary_x}" y="{legend_y + 18}">Fastest: {html.escape(fastest.name)}</text>')
    lines.append(f'<text class="legend" x="{summary_x}" y="{legend_y + 36}">Smallest: {html.escape(smallest.name)}</text>')
    lines.append(f'<text class="legend" x="{summary_x}" y="{legend_y + 64}">Bubble size = disk size</text>')
    lines.append("</svg>")
    write_text(lines, output_path)


def quant_only_frontier_svg(records: list[RunRecord], output_path: Path) -> None:
    chosen = [
        record
        for record in records
        if record.family == "quant_only"
        and record.token_f1 is not None
        and record.disk_size_mb is not None
        and record.latency_ms_mean is not None
        and record.examples_per_second is not None
    ]
    width = 1380
    height = 760
    panel_w = 620
    panel_h = 520
    start_x = 42
    start_y = 120
    gap_x = 34

    panels = [
        ("disk_size_mb", "Artifact Size (MB, lower is better)", False, "Size frontier"),
        ("latency_ms_mean", "Mean Latency (ms, lower is better)", False, "Latency frontier"),
    ]

    lines = svg_header(width, height, "Quant-Only Pareto Frontiers")
    lines.append(
        '<text class="subtitle" x="28" y="54">Points on the frontier are not strictly beaten by another quant-only run on both quality and the efficiency metric in that panel.</text>'
    )

    for idx, (x_attr, x_label, maximize_x, panel_title) in enumerate(panels):
        panel_x = start_x + idx * (panel_w + gap_x)
        panel_y = start_y
        inner_left = panel_x + 68
        inner_right = panel_x + panel_w - 20
        inner_top = panel_y + 38
        inner_bottom = panel_y + panel_h - 54
        plot_w = inner_right - inner_left
        plot_h = inner_bottom - inner_top

        frontier = pareto_frontier(chosen, x_attr, "token_f1", maximize_x=maximize_x, maximize_y=True)
        points = [(record, float(getattr(record, x_attr)), float(record.token_f1)) for record in chosen]
        xs = [x for _, x, _ in points]
        ys = [y for _, _, y in points]
        x_min, x_max = axis_scale(xs, padding_ratio=0.12)
        y_min, y_max = axis_scale(ys, padding_ratio=0.12)

        def sx(value: float) -> float:
            return inner_left + (value - x_min) / (x_max - x_min) * plot_w

        def sy(value: float) -> float:
            return inner_top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

        lines.append(f'<rect x="{panel_x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" rx="14" fill="#ffffff" stroke="#D8E1EC" />')
        lines.append(f'<text class="label" x="{panel_x + 18}" y="{panel_y + 24}">{html.escape(panel_title)}</text>')

        for tick in range(6):
            ratio = tick / 5
            gx = inner_left + ratio * plot_w
            gy = inner_top + ratio * plot_h
            xv = x_min + ratio * (x_max - x_min)
            yv = y_max - ratio * (y_max - y_min)
            lines.append(f'<line class="grid" x1="{gx:.1f}" y1="{inner_top}" x2="{gx:.1f}" y2="{inner_bottom}" />')
            lines.append(f'<line class="grid" x1="{inner_left}" y1="{gy:.1f}" x2="{inner_right}" y2="{gy:.1f}" />')
            lines.append(f'<text class="tick" x="{gx:.1f}" y="{inner_bottom + 20:.1f}" text-anchor="middle">{xv:.1f}</text>')
            lines.append(f'<text class="tick" x="{inner_left - 8:.1f}" y="{gy + 4:.1f}" text-anchor="end">{yv:.3f}</text>')

        ordered_frontier = sorted(frontier, key=lambda record: float(getattr(record, x_attr)), reverse=maximize_x)
        if len(ordered_frontier) >= 2:
            coords = " ".join(f"{sx(float(getattr(record, x_attr))):.1f},{sy(float(record.token_f1)):.1f}" for record in ordered_frontier)
            lines.append(f'<polyline points="{coords}" fill="none" stroke="#243447" stroke-width="2.5" stroke-dasharray="8 6" />')

        lines.append(f'<line class="axis" x1="{inner_left}" y1="{inner_bottom}" x2="{inner_right}" y2="{inner_bottom}" />')
        lines.append(f'<line class="axis" x1="{inner_left}" y1="{inner_top}" x2="{inner_left}" y2="{inner_bottom}" />')
        lines.append(f'<text class="tick" x="{inner_left + plot_w / 2:.1f}" y="{panel_y + panel_h - 16}">{html.escape(x_label)}</text>')
        lines.append(
            f'<text class="tick" transform="translate({panel_x + 16},{inner_top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">Token F1 (higher is better)</text>'
        )

        for record, x_value, y_value in points:
            color = quant_color(record)
            cx = sx(x_value)
            cy = sy(y_value)
            is_frontier = record in frontier
            radius = 8.5 if is_frontier else 6.5
            stroke = "#17212B" if is_frontier else "#ffffff"
            stroke_width = 2.2 if is_frontier else 1.4
            lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.9" stroke="{stroke}" stroke-width="{stroke_width}" />')
            lines.append(f'<text class="point-label" x="{cx + 8:.1f}" y="{cy - 8:.1f}">{html.escape(record.name)}</text>')

        lines.append(f'<text class="legend" x="{panel_x + 18}" y="{panel_y + panel_h - 18}">Dark outline + dashed line = Pareto frontier</text>')

    lines.append("</svg>")
    write_text(lines, output_path)


def quant_only_scorecard_svg(records: list[RunRecord], output_path: Path) -> None:
    chosen = [record for record in records if record.family == "quant_only"]
    cols = [
        ("Strategy", "name"),
        ("Runtime", "runtime_target"),
        ("Token F1", "token_f1"),
        ("ROUGE-L", "rouge_l_f1"),
        ("Size MB", "disk_size_mb"),
        ("Latency", "latency_ms_mean"),
        ("Ex/s", "examples_per_second"),
    ]
    col_widths = [270, 100, 120, 120, 110, 120, 110]
    row_height = 38
    width = 1080
    height = 122 + row_height * (len(chosen) + 1) + 28
    x_positions = [30]
    for width_value in col_widths[:-1]:
        x_positions.append(x_positions[-1] + width_value)

    numeric_attrs = {"token_f1", "rouge_l_f1", "disk_size_mb", "latency_ms_mean", "examples_per_second"}
    lower_is_better = {"disk_size_mb", "latency_ms_mean"}
    minmax: dict[str, tuple[float, float]] = {}
    for _, attr in cols:
        if attr not in numeric_attrs:
            continue
        values = [float(getattr(record, attr)) for record in chosen if getattr(record, attr) is not None]
        minmax[attr] = (min(values), max(values)) if values else (0.0, 1.0)

    def score(attr: str, value: float) -> float:
        low, high = minmax[attr]
        if math.isclose(low, high):
            return 1.0
        base = (value - low) / (high - low)
        if attr in lower_is_better:
            base = 1.0 - base
        return max(0.0, min(base, 1.0))

    def fill_color(attr: str, value: Any) -> str:
        if attr not in numeric_attrs or value is None:
            return "#F4F7FB"
        blend = score(attr, float(value))
        red = round(242 - blend * 96)
        green = round(118 + blend * 92)
        blue = round(101 + blend * 78)
        return f"rgb({red},{green},{blue})"

    lines = svg_header(width, height, "Quant-Only Scorecard")
    lines.append(
        '<text class="subtitle" x="28" y="54">Greener means stronger for that metric. Size and latency are inverted so smaller values score greener there.</text>'
    )

    header_y = 86
    for index, (col_name, _) in enumerate(cols):
        x = x_positions[index]
        w = col_widths[index]
        lines.append(f'<rect x="{x}" y="{header_y}" width="{w}" height="{row_height}" fill="#DCE7F5" stroke="#ffffff" />')
        lines.append(f'<text class="label" x="{x + 10}" y="{header_y + 24}">{html.escape(col_name)}</text>')

    ordered = sorted(chosen, key=lambda record: record.token_f1 or -1, reverse=True)
    for row_index, record in enumerate(ordered):
        y = header_y + row_height * (row_index + 1)
        for col_index, (_, attr) in enumerate(cols):
            x = x_positions[col_index]
            w = col_widths[col_index]
            value = getattr(record, attr)
            lines.append(f'<rect x="{x}" y="{y}" width="{w}" height="{row_height}" fill="{fill_color(attr, value)}" stroke="#ffffff" />')
            if attr == "name":
                text = str(value)
            elif attr == "runtime_target":
                text = str(value).upper() if value else "-"
            elif isinstance(value, float):
                text = f"{value:.4f}" if value < 10 else f"{value:.2f}"
            else:
                text = str(value)
            lines.append(f'<text class="label" x="{x + 10}" y="{y + 24}">{html.escape(text)}</text>')

    lines.append("</svg>")
    write_text(lines, output_path)


def combined_story_svg(records: list[RunRecord], output_path: Path) -> None:
    by_name = {record.name: record for record in records}
    combined = [
        record
        for record in records
        if record.family in {"prune_then_quant", "quant_then_prune"} and len(record.parent_names) >= 2
    ]
    width = 1360
    height = 260 + len(combined) * 220
    lines = svg_header(width, height, "Combined Strategy Story: Why These Parents Were Continued")
    lines.append(
        '<text class="subtitle" x="28" y="54">Each block shows the selected parents and the resulting combined artifact so you can see the quality and efficiency tradeoff behind the next step.</text>'
    )

    block_y = 92
    for record in combined:
        quant_parent = None
        prune_parent = None
        for name in record.parent_names:
            parent = by_name.get(name)
            if not parent:
                continue
            if parent.family == "quant_only":
                quant_parent = parent
            elif parent.family == "prune_only":
                prune_parent = parent
        if not quant_parent and not prune_parent:
            continue

        lines.append(f'<rect x="24" y="{block_y}" width="1300" height="188" rx="16" fill="#ffffff" stroke="#D8E1EC" />')
        lines.append(f'<text class="label" x="42" y="{block_y + 28}">{html.escape(record.name)}</text>')

        nodes = [
            ("Prune Parent", prune_parent, 120),
            ("Quant Parent", quant_parent, 470),
            ("Combined Result", record, 890),
        ]
        for label, node, x in nodes:
            if node is None:
                continue
            y = block_y + 54
            color = PALETTE.get(node.family, PALETTE["other"])
            lines.append(f'<rect x="{x}" y="{y}" width="290" height="102" rx="12" fill="#F9FBFE" stroke="{color}" stroke-width="2" />')
            lines.append(f'<text class="label" x="{x + 14}" y="{y + 20}" fill="#5B6576">{html.escape(label)}</text>')
            lines.append(f'<text class="label" x="{x + 14}" y="{y + 42}">{html.escape(node.name)}</text>')
            lines.append(f'<text class="tick" x="{x + 14}" y="{y + 64}">Token F1: {(node.token_f1 or 0.0):.4f}</text>')
            lines.append(f'<text class="tick" x="{x + 14}" y="{y + 80}">Size: {(node.disk_size_mb or 0.0):.2f} MB</text>')
            lines.append(f'<text class="tick" x="{x + 150}" y="{y + 64}">Latency: {(node.latency_ms_mean or 0.0):.1f} ms</text>')
            lines.append(f'<text class="tick" x="{x + 150}" y="{y + 80}">ROUGE-L: {(node.rouge_l_f1 or 0.0):.4f}</text>')

        lines.append(f'<line x1="410" y1="{block_y + 105}" x2="470" y2="{block_y + 105}" stroke="#AAB7C8" stroke-width="2.5" marker-end="url(#arrow)" />')
        lines.append(f'<line x1="760" y1="{block_y + 105}" x2="890" y2="{block_y + 105}" stroke="#AAB7C8" stroke-width="2.5" marker-end="url(#arrow)" />')

        if prune_parent is not None:
            delta_quality = (record.token_f1 or 0.0) - (prune_parent.token_f1 or 0.0)
            delta_size = (record.disk_size_mb or 0.0) - (prune_parent.disk_size_mb or 0.0)
            lines.append(
                f'<text class="tick" x="898" y="{block_y + 176}">vs prune parent: Δ Token F1 {delta_quality:+.4f}, Δ Size {delta_size:+.2f} MB</text>'
            )
        elif quant_parent is not None:
            delta_quality = (record.token_f1 or 0.0) - (quant_parent.token_f1 or 0.0)
            delta_size = (record.disk_size_mb or 0.0) - (quant_parent.disk_size_mb or 0.0)
            lines.append(
                f'<text class="tick" x="898" y="{block_y + 176}">vs quant parent: Δ Token F1 {delta_quality:+.4f}, Δ Size {delta_size:+.2f} MB</text>'
            )
        block_y += 220

    lines.insert(
        1,
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L8,3 z" fill="#AAB7C8" /></marker></defs>',
    )
    lines.append("</svg>")
    write_text(lines, output_path)


def all_models_overview_svg(records: list[RunRecord], output_path: Path) -> None:
    width = 1160
    height = 760
    margin_left = 90
    margin_right = 230
    margin_top = 84
    margin_bottom = 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    points = [record for record in records if record.token_f1 is not None and record.latency_ms_mean is not None and record.disk_size_mb is not None]
    xs = [float(record.latency_ms_mean) for record in points]
    ys = [float(record.token_f1) for record in points]
    x_min, x_max = axis_scale(xs)
    y_min, y_max = axis_scale(ys)
    size_min = min(float(record.disk_size_mb) for record in points)
    size_max = max(float(record.disk_size_mb) for record in points)

    def sx(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return margin_top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    def sr(value: float) -> float:
        if math.isclose(size_min, size_max):
            return 12.0
        return 8.0 + (value - size_min) / (size_max - size_min) * 18.0

    lines = svg_header(width, height, "All Models Overview: Quality, Latency, and Size Together")
    lines.append(
        '<text class="subtitle" x="28" y="54">X is latency, Y is Token F1, and bubble size represents disk size. This is the broadest single view of quality and efficiency together.</text>'
    )

    for idx in range(6):
        ratio = idx / 5
        x = margin_left + ratio * plot_w
        y = margin_top + ratio * plot_h
        xv = x_min + ratio * (x_max - x_min)
        yv = y_max - ratio * (y_max - y_min)
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{margin_top}" x2="{x:.1f}" y2="{margin_top + plot_h}" />')
        lines.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{margin_left + plot_w}" y2="{y:.1f}" />')
        lines.append(f'<text class="tick" x="{x:.1f}" y="{margin_top + plot_h + 22:.1f}" text-anchor="middle">{xv:.0f}</text>')
        lines.append(f'<text class="tick" x="{margin_left - 10:.1f}" y="{y + 4:.1f}" text-anchor="end">{yv:.2f}</text>')

    lines.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" />')
    lines.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" />')
    lines.append(f'<text class="label" x="{margin_left + plot_w / 2:.1f}" y="{height - 26}" text-anchor="middle">Mean Latency (ms, lower is better)</text>')
    lines.append(
        f'<text class="label" transform="translate(28,{margin_top + plot_h / 2:.1f}) rotate(-90)" text-anchor="middle">Token F1 (higher is better)</text>'
    )

    for record in points:
        color = PALETTE.get(record.family, PALETTE["other"])
        cx = sx(float(record.latency_ms_mean))
        cy = sy(float(record.token_f1))
        radius = sr(float(record.disk_size_mb))
        lines.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}" fill="{color}" fill-opacity="0.42" stroke="{color}" stroke-width="2" />')
        lines.append(f'<text class="point-label" x="{cx + radius + 4:.1f}" y="{cy - radius - 2:.1f}">{html.escape(record.name)}</text>')

    legend_x = width - margin_right + 24
    legend_y = 112
    for family in ["baseline", "quant_only", "prune_only", "prune_then_quant", "quant_then_prune"]:
        color = PALETTE.get(family, PALETTE["other"])
        lines.append(f'<circle cx="{legend_x}" cy="{legend_y}" r="7" fill="{color}" fill-opacity="0.5" stroke="{color}" stroke-width="2" />')
        lines.append(f'<text class="legend" x="{legend_x + 14}" y="{legend_y + 4}">{html.escape(family.replace("_", " "))}</text>')
        legend_y += 24
    lines.append(f'<text class="legend" x="{legend_x}" y="{legend_y + 14}">Bubble size = disk size</text>')
    lines.append("</svg>")
    write_text(lines, output_path)


def build_dashboard(records: list[RunRecord], full_runs: list[RunRecord], output_dir: Path) -> None:
    best_quality = max(full_runs, key=lambda record: record.token_f1 or -1)
    best_speed = max(full_runs, key=lambda record: record.examples_per_second or -1)
    smallest = min(full_runs, key=lambda record: record.disk_size_mb or float("inf"))
    failed = [record for record in full_runs if (record.token_f1 or 0.0) < 0.1]

    lines = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8" />',
        "<title>FLAN-T5 Benchmark Dashboard</title>",
        "<style>",
        "body { font-family: Arial, sans-serif; margin: 24px; color: #1d2736; background: #f6f8fb; }",
        "h1, h2 { margin: 0 0 12px; }",
        "p { line-height: 1.5; }",
        ".cards { display: flex; gap: 16px; flex-wrap: wrap; margin: 20px 0 28px; }",
        ".card { background: #ffffff; border-radius: 14px; padding: 16px 18px; box-shadow: 0 8px 24px rgba(25, 35, 52, 0.08); min-width: 240px; }",
        ".label { font-size: 12px; text-transform: uppercase; color: #66758c; letter-spacing: 0.05em; }",
        ".value { font-size: 22px; font-weight: 700; margin: 8px 0 4px; }",
        ".caption { font-size: 14px; color: #435063; }",
        ".chart { background: #ffffff; border-radius: 14px; padding: 12px; margin: 18px 0; box-shadow: 0 8px 24px rgba(25, 35, 52, 0.08); overflow-x: auto; }",
        "code { background: #eef3fa; padding: 2px 5px; border-radius: 5px; }",
        "ul { line-height: 1.5; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>FLAN-T5 Benchmark Dashboard</h1>",
        "<p>This dashboard compares fully evaluated optimization strategies using the shared benchmark contract. It is built from saved <code>strategy_manifest.json</code> and <code>metrics.json</code> files.</p>",
        '<div class="cards">',
        f'<div class="card"><div class="label">Best Quality</div><div class="value">{html.escape(best_quality.name)}</div><div class="caption">Token F1 {best_quality.token_f1:.4f}, size {best_quality.disk_size_mb:.2f} MB</div></div>',
        f'<div class="card"><div class="label">Best Throughput</div><div class="value">{html.escape(best_speed.name)}</div><div class="caption">{best_speed.examples_per_second:.4f} examples/s, Token F1 {best_speed.token_f1:.4f}</div></div>',
        f'<div class="card"><div class="label">Smallest Artifact</div><div class="value">{html.escape(smallest.name)}</div><div class="caption">{smallest.disk_size_mb:.2f} MB, Token F1 {smallest.token_f1:.4f}</div></div>',
        "</div>",
        "<h2>Recommended Views</h2>",
        "<ul>",
        "<li>Start with the quant-only overview, Pareto frontiers, and scorecard if your immediate decision is only among quantized models.</li>",
        "<li>Then use the quant-only and prune-only tradeoff boards for the broader quality versus disk size, parameter count, and latency read.</li>",
        "<li>Then read the combined strategy story chart to understand why certain parents were continued into prune-plus-quant experiments.</li>",
        "<li>Use the all-models overview bubble chart as the final summary across quality, latency, and size.</li>",
        "<li>Use the ranking bars and heatmap for a compact backup summary.</li>",
        "</ul>",
        '<div class="chart"><object data="quant_only_overview.svg" type="image/svg+xml"></object></div>',
        '<div class="chart"><object data="quant_only_pareto_frontiers.svg" type="image/svg+xml"></object></div>',
        '<div class="chart"><object data="quant_only_scorecard.svg" type="image/svg+xml"></object></div>',
        '<div class="chart"><object data="quant_only_tradeoff_board.svg" type="image/svg+xml"></object></div>',
        '<div class="chart"><object data="prune_only_tradeoff_board.svg" type="image/svg+xml"></object></div>',
        '<div class="chart"><object data="combined_strategy_story.svg" type="image/svg+xml"></object></div>',
        '<div class="chart"><object data="all_models_overview.svg" type="image/svg+xml"></object></div>',
        '<div class="chart"><object data="all_methods_quality_vs_size.svg" type="image/svg+xml"></object></div>',
        '<div class="chart"><object data="all_methods_quality_vs_speed.svg" type="image/svg+xml"></object></div>',
        '<div class="chart"><object data="full_run_token_f1_ranking.svg" type="image/svg+xml"></object></div>',
        '<div class="chart"><object data="quant_only_token_f1_ranking.svg" type="image/svg+xml"></object></div>',
        '<div class="chart"><object data="prune_only_token_f1_ranking.svg" type="image/svg+xml"></object></div>',
        '<div class="chart"><object data="combined_token_f1_ranking.svg" type="image/svg+xml"></object></div>',
        '<div class="chart"><object data="top10_scorecard_heatmap.svg" type="image/svg+xml"></object></div>',
        "<h2>Notes</h2>",
        "<ul>",
        f"<li>Charts include runs with at least {min(record.eval_count or 0 for record in full_runs)} evaluation examples after filtering.</li>",
        f"<li>{len(failed)} runs have Token F1 below 0.10 and should be treated as failed or severely degraded candidates.</li>",
        "<li>Structured pruning is separated from unstructured pruning in the labels because the quality behavior is very different in these results.</li>",
        "</ul>",
        "</body>",
        "</html>",
    ]
    write_text(lines, output_dir / "index.html")


def main() -> None:
    args = parse_args()
    optimized_root = Path(args.optimized_root)
    baseline_root = Path(args.baseline_root)
    output_dir = Path(args.output_dir)

    records = read_optimized_runs(optimized_root) + read_baseline_runs(baseline_root)
    full_runs = filter_full_runs(records, args.min_eval_count)

    write_csv(records, output_dir / "benchmark_summary.csv")
    scatter_svg(
        full_runs,
        x_attr="disk_size_mb",
        y_attr="token_f1",
        title="All Methods: Quality vs Size",
        x_label="Artifact size (MB, lower is better)",
        y_label="Token F1 (higher is better)",
        output_path=output_dir / "all_methods_quality_vs_size.svg",
    )
    scatter_svg(
        full_runs,
        x_attr="examples_per_second",
        y_attr="token_f1",
        title="All Methods: Quality vs Throughput",
        x_label="Examples per second (higher is better)",
        y_label="Token F1 (higher is better)",
        output_path=output_dir / "all_methods_quality_vs_speed.svg",
    )
    horizontal_bar_svg(
        full_runs,
        title="Full Run Ranking by Token F1",
        metric_attr="token_f1",
        metric_label="Token F1",
        output_path=output_dir / "full_run_token_f1_ranking.svg",
    )
    horizontal_bar_svg(
        [record for record in full_runs if record.family == "quant_only"],
        title="Quant-Only Ranking by Token F1",
        metric_attr="token_f1",
        metric_label="Token F1",
        output_path=output_dir / "quant_only_token_f1_ranking.svg",
    )
    horizontal_bar_svg(
        [record for record in full_runs if record.family == "prune_only"],
        title="Prune-Only Ranking by Token F1",
        metric_attr="token_f1",
        metric_label="Token F1",
        output_path=output_dir / "prune_only_token_f1_ranking.svg",
    )
    horizontal_bar_svg(
        [record for record in full_runs if record.family in {"prune_then_quant", "quant_then_prune"}],
        title="Combined Methods Ranking by Token F1",
        metric_attr="token_f1",
        metric_label="Token F1",
        output_path=output_dir / "combined_token_f1_ranking.svg",
    )
    small_multiple_tradeoff_svg(
        full_runs,
        family="quant_only",
        family_title="Quant Only",
        output_path=output_dir / "quant_only_tradeoff_board.svg",
    )
    quant_only_overview_svg(full_runs, output_dir / "quant_only_overview.svg")
    quant_only_frontier_svg(full_runs, output_dir / "quant_only_pareto_frontiers.svg")
    quant_only_scorecard_svg(full_runs, output_dir / "quant_only_scorecard.svg")
    small_multiple_tradeoff_svg(
        full_runs,
        family="prune_only",
        family_title="Prune Only",
        output_path=output_dir / "prune_only_tradeoff_board.svg",
    )
    combined_story_svg(full_runs, output_dir / "combined_strategy_story.svg")
    all_models_overview_svg(full_runs, output_dir / "all_models_overview.svg")
    summary_heatmap_svg(full_runs, output_dir / "top10_scorecard_heatmap.svg")
    build_dashboard(records, full_runs, output_dir)

    print(f"Wrote dashboard assets to {output_dir}")


if __name__ == "__main__":
    main()
