"""
Generate report-ready JPG plots for model optimization experiments.

Input:
    model_summary_metrics.csv

Output:
    model_optimization_report_assets/*.jpg
    model_optimization_report_assets/model_selection_scorecard.csv
    model_optimization_report_assets/plot_manifest.csv

How to run:
    python generate_model_optimization_plots.py --csv model_summary_metrics.csv

Optional:
    python generate_model_optimization_plots.py --csv model_summary_metrics.csv --outdir my_plots
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------

ACCURACY_COLS = ["token_f1", "rouge_l_f1"]
EFFICIENCY_COLS = ["disk_size_mb", "latency_ms_mean", "examples_per_second", "generated_tokens_per_second", "rss_after_load_mb"]

# Balanced score weights. Adjust these if your report values speed/storage more than quality.
SCORE_WEIGHTS = {
    "token_f1": 0.45,
    "rouge_l_f1": 0.15,
    "examples_per_second": 0.15,
    "latency_ms_mean": 0.15,  # lower is better
    "disk_size_mb": 0.10,     # lower is better
}

LOWER_IS_BETTER = {"latency_ms_mean", "disk_size_mb", "rss_after_load_mb", "load_time_s", "total_eval_time_s"}

FIG_DPI = 300


# -----------------------------
# Utility functions
# -----------------------------

def ensure_outdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(fig: plt.Figure, outdir: Path, filename: str, title: str, manifest: List[Dict[str, str]]) -> None:
    path = outdir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight", format="jpg")
    plt.close(fig)
    manifest.append({"file": filename, "title": title, "path": str(path)})


def short_name(name: str) -> str:
    """Make long experiment names readable in plot labels."""
    mapping = {
        "baseline_reference": "Baseline\nFP32",
        "bf16_cast_gpu": "BF16\nGPU",
        "fp16_cast_gpu": "FP16\nGPU",
        "dynamic_int8_cpu_ptq": "INT8\nCPU",
        "prune_unstructured_attention_l1_10": "Unstruct.\nAttn 10%",
        "prune_unstructured_mlp_l1_10": "Unstruct.\nMLP 10%",
        "prune_unstructured_global_l1_10": "Unstruct.\nGlobal 10%",
        "prune_unstructured_global_l1_20": "Unstruct.\nGlobal 20%",
        "prune_structured_attention_heads_1": "Struct.\nAttn Head",
        "prune_structured_mlp_intermediate_10": "Struct.\nMLP 10%",
        "combined_quant_then_prune_selected__bf16_cast_gpu__prune_unstructured_mlp_l1_10": "BF16 →\nMLP prune",
        "combined_structured_prune_then_quant__prune_structured_attention_heads_1__bf16_cast_gpu": "Struct. Attn →\nBF16",
        "combined_unstructured_prune_then_quant__prune_unstructured_mlp_l1_10__dynamic_int8_cpu_ptq": "MLP prune →\nINT8",
    }
    if name in mapping:
        return mapping[name]

    s = name
    s = s.replace("combined_quant_then_prune_selected__", "")
    s = s.replace("combined_unstructured_prune_then_quant__", "")
    s = s.replace("combined_structured_prune_then_quant__", "")
    s = s.replace("prune_unstructured_", "unstruct_")
    s = s.replace("prune_structured_", "struct_")
    s = s.replace("dynamic_int8_cpu_ptq", "int8_cpu")
    s = s.replace("bf16_cast_gpu", "bf16_gpu")
    s = s.replace("fp16_cast_gpu", "fp16_gpu")
    s = s.replace("_", " ")
    return "\n".join(re.findall(r".{1,18}(?:\s|$)", s)) or s


def add_point_labels(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, label_col: str = "model_name") -> None:
    for _, r in df.iterrows():
        if pd.notna(r.get(x)) and pd.notna(r.get(y)):
            ax.annotate(
                short_name(str(r[label_col])),
                (r[x], r[y]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
                alpha=0.90,
            )


def add_grid(ax: plt.Axes) -> None:
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_axisbelow(True)


def pct_delta(new: float, old: float, lower_is_better: bool = False) -> float:
    """
    Percent improvement from old to new.
    If lower_is_better, lower values are positive improvements.
    """
    if old == 0 or pd.isna(old) or pd.isna(new):
        return np.nan
    if lower_is_better:
        return (old - new) / old * 100.0
    return (new - old) / old * 100.0


def minmax_score(series: pd.Series, lower_is_better: bool = False) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mn, mx = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(np.ones(len(s)) * 0.5, index=s.index)
    norm = (s - mn) / (mx - mn)
    if lower_is_better:
        norm = 1 - norm
    return norm.fillna(0)


def compute_balanced_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    score = pd.Series(np.zeros(len(out)), index=out.index, dtype=float)
    for col, w in SCORE_WEIGHTS.items():
        if col in out.columns:
            score += w * minmax_score(out[col], lower_is_better=(col in LOWER_IS_BETTER))
    out["balanced_score"] = score
    return out.sort_values("balanced_score", ascending=False)


def first_row(df: pd.DataFrame, name: str) -> Optional[pd.Series]:
    hit = df[df["model_name"] == name]
    if hit.empty:
        return None
    return hit.iloc[0]


def safe_family(df: pd.DataFrame, family: str) -> pd.DataFrame:
    if "strategy_family" not in df.columns:
        return df.iloc[0:0].copy()
    return df[df["strategy_family"].eq(family)].copy()


def selected_baseline(df: pd.DataFrame) -> Optional[pd.Series]:
    """Prefer baseline_reference if present, otherwise use the largest eval_count baseline row."""
    row = first_row(df, "baseline_reference")
    if row is not None:
        return row
    baselines = safe_family(df, "baseline")
    if baselines.empty:
        return None
    return baselines.sort_values("eval_count", ascending=False).iloc[0]


# -----------------------------
# Plot functions
# -----------------------------

def plot_quant_accuracy_vs_size(df: pd.DataFrame, outdir: Path, manifest: List[Dict[str, str]]) -> None:
    q = safe_family(df, "quant_only")
    if q.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(q["disk_size_mb"], q["token_f1"], s=120)
    add_point_labels(ax, q, "disk_size_mb", "token_f1")
    add_grid(ax)
    ax.set_title("Quantization Only: Accuracy vs Model Size")
    ax.set_xlabel("Disk size (MB) — lower is better")
    ax.set_ylabel("Token F1 — higher is better")
    save_fig(fig, outdir, "01_quant_only_accuracy_vs_size.jpg", "Quantization only: accuracy vs model size", manifest)


def plot_quant_accuracy_vs_latency(df: pd.DataFrame, outdir: Path, manifest: List[Dict[str, str]]) -> None:
    q = safe_family(df, "quant_only")
    if q.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(q["latency_ms_mean"], q["token_f1"], s=120)
    add_point_labels(ax, q, "latency_ms_mean", "token_f1")
    add_grid(ax)
    ax.set_title("Quantization Only: Accuracy vs Latency")
    ax.set_xlabel("Mean latency (ms/example) — lower is better")
    ax.set_ylabel("Token F1 — higher is better")
    save_fig(fig, outdir, "02_quant_only_accuracy_vs_latency.jpg", "Quantization only: accuracy vs latency", manifest)


def plot_prune_accuracy_vs_latency(df: pd.DataFrame, outdir: Path, manifest: List[Dict[str, str]]) -> None:
    p = safe_family(df, "prune_only")
    if p.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5.8))
    ax.scatter(p["latency_ms_mean"], p["token_f1"], s=120)
    add_point_labels(ax, p, "latency_ms_mean", "token_f1")
    add_grid(ax)
    ax.set_title("Pruning Only: Accuracy vs Latency")
    ax.set_xlabel("Mean latency (ms/example) — lower is better")
    ax.set_ylabel("Token F1 — higher is better")
    save_fig(fig, outdir, "03_prune_only_accuracy_vs_latency.jpg", "Pruning only: accuracy vs latency", manifest)


def plot_unstructured_sparsity_vs_accuracy(df: pd.DataFrame, outdir: Path, manifest: List[Dict[str, str]]) -> None:
    p = safe_family(df, "prune_only")
    p = p[p["pruning_ops"].fillna("").str.contains("unstructured", case=False)]
    if p.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.scatter(p["global_linear_sparsity"] * 100, p["token_f1"], s=120)
    for _, r in p.iterrows():
        ax.annotate(
            short_name(str(r["model_name"])),
            (r["global_linear_sparsity"] * 100, r["token_f1"]),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
        )
    add_grid(ax)
    ax.set_title("Unstructured Pruning: Sparsity vs Accuracy")
    ax.set_xlabel("Global linear sparsity (%)")
    ax.set_ylabel("Token F1 — higher is better")
    save_fig(fig, outdir, "04_prune_only_unstructured_accuracy_vs_sparsity.jpg", "Unstructured pruning: sparsity vs accuracy", manifest)


def plot_prune_disk_reduction(df: pd.DataFrame, outdir: Path, manifest: List[Dict[str, str]]) -> None:
    p = safe_family(df, "prune_only")
    if p.empty:
        return

    if "size_reduction_pct_vs_source" not in p.columns or p["size_reduction_pct_vs_source"].isna().all():
        src_size = df["source_model_disk_size_mb"].dropna()
        source_size = src_size.iloc[0] if not src_size.empty else df["disk_size_mb"].max()
        p = p.copy()
        p["size_reduction_pct_vs_source"] = (source_size - p["disk_size_mb"]) / source_size * 100

    p = p.sort_values("size_reduction_pct_vs_source", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.bar([short_name(x) for x in p["model_name"]], p["size_reduction_pct_vs_source"])
    add_grid(ax)
    ax.set_title("Pruning Only: Actual Disk Size Reduction")
    ax.set_xlabel("Pruning method")
    ax.set_ylabel("Disk reduction vs source (%)")
    ax.tick_params(axis="x", rotation=0, labelsize=8)
    save_fig(fig, outdir, "05_prune_only_disk_reduction.jpg", "Pruning only: actual disk savings", manifest)


def plot_combined_parent_to_pipeline(df: pd.DataFrame, outdir: Path, manifest: List[Dict[str, str]]) -> None:
    combined = df[df["strategy_family"].isin(["quant_then_prune", "prune_then_quant"])].copy()
    if combined.empty:
        return

    # Manually map each combined pipeline to the closest single-step parent.
    parent_map = {
        "combined_quant_then_prune_selected__bf16_cast_gpu__prune_unstructured_mlp_l1_10": "bf16_cast_gpu",
        "combined_structured_prune_then_quant__prune_structured_attention_heads_1__bf16_cast_gpu": "prune_structured_attention_heads_1",
        "combined_unstructured_prune_then_quant__prune_unstructured_mlp_l1_10__dynamic_int8_cpu_ptq": "dynamic_int8_cpu_ptq",
    }

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title("Stacked Optimization: Parent Method → Combined Pipeline")
    ax.set_xlabel("Mean latency (ms/example) — lower is better")
    ax.set_ylabel("Token F1 — higher is better")
    add_grid(ax)

    for child_name, parent_name in parent_map.items():
        child = first_row(df, child_name)
        parent = first_row(df, parent_name)
        if child is None or parent is None:
            continue

        ax.scatter(parent["latency_ms_mean"], parent["token_f1"], s=120, marker="o")
        ax.scatter(child["latency_ms_mean"], child["token_f1"], s=120, marker="s")
        ax.annotate(
            "",
            xy=(child["latency_ms_mean"], child["token_f1"]),
            xytext=(parent["latency_ms_mean"], parent["token_f1"]),
            arrowprops=dict(arrowstyle="->", lw=1.4, alpha=0.8),
        )
        ax.annotate(short_name(parent_name), (parent["latency_ms_mean"], parent["token_f1"]), textcoords="offset points", xytext=(6, 5), fontsize=8)
        ax.annotate(short_name(child_name), (child["latency_ms_mean"], child["token_f1"]), textcoords="offset points", xytext=(6, -16), fontsize=8)

    save_fig(fig, outdir, "06_combined_parent_to_pipeline_tradeoff.jpg", "Combined pipelines: parent to pipeline tradeoff", manifest)


def plot_combined_relative_deltas(df: pd.DataFrame, outdir: Path, manifest: List[Dict[str, str]]) -> None:
    parent_map = {
        "BF16 → MLP prune\nvs BF16": (
            "combined_quant_then_prune_selected__bf16_cast_gpu__prune_unstructured_mlp_l1_10",
            "bf16_cast_gpu",
        ),
        "Struct. Attn → BF16\nvs Struct. Attn": (
            "combined_structured_prune_then_quant__prune_structured_attention_heads_1__bf16_cast_gpu",
            "prune_structured_attention_heads_1",
        ),
        "MLP prune → INT8\nvs INT8": (
            "combined_unstructured_prune_then_quant__prune_unstructured_mlp_l1_10__dynamic_int8_cpu_ptq",
            "dynamic_int8_cpu_ptq",
        ),
    }

    rows = []
    for label, (child_name, parent_name) in parent_map.items():
        child = first_row(df, child_name)
        parent = first_row(df, parent_name)
        if child is None or parent is None:
            continue
        rows.append({
            "pipeline": label,
            "Token F1 Δ%": pct_delta(child["token_f1"], parent["token_f1"]),
            "Latency improvement %": pct_delta(child["latency_ms_mean"], parent["latency_ms_mean"], lower_is_better=True),
            "Disk reduction improvement %": pct_delta(child["disk_size_mb"], parent["disk_size_mb"], lower_is_better=True),
            "RSS improvement %": pct_delta(child["rss_after_load_mb"], parent["rss_after_load_mb"], lower_is_better=True),
        })

    delta_df = pd.DataFrame(rows)
    if delta_df.empty:
        return

    metrics = ["Token F1 Δ%", "Latency improvement %", "Disk reduction improvement %", "RSS improvement %"]
    x = np.arange(len(delta_df))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, metric in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, delta_df[metric], width, label=metric)

    ax.axhline(0, linewidth=1)
    add_grid(ax)
    ax.set_title("Stacked Optimization: What the Second Step Adds")
    ax.set_ylabel("Relative improvement vs parent (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(delta_df["pipeline"], fontsize=8)
    ax.legend(fontsize=8, ncol=2)
    save_fig(fig, outdir, "07_combined_relative_to_parent_deltas.jpg", "Combined pipelines: relative deltas vs parent", manifest)


def plot_overall_ranking(df: pd.DataFrame, outdir: Path, manifest: List[Dict[str, str]]) -> pd.DataFrame:
    candidates = df[df["strategy_family"].isin(["quant_only", "prune_only", "quant_then_prune", "prune_then_quant"])].copy()
    if candidates.empty:
        return pd.DataFrame()

    ranked = compute_balanced_score(candidates)
    ranked.to_csv(outdir / "model_selection_scorecard.csv", index=False)

    top = ranked.head(10).sort_values("balanced_score", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh([short_name(x) for x in top["model_name"]], top["balanced_score"])
    add_grid(ax)
    ax.set_title("Overall Ranking Across Optimized Candidates")
    ax.set_xlabel("Balanced score, accuracy-weighted")
    ax.set_ylabel("Model")
    save_fig(fig, outdir, "08_overall_balanced_score_ranking.jpg", "Overall ranking across optimized candidates", manifest)
    return ranked


def plot_selected_vs_baseline(df: pd.DataFrame, outdir: Path, manifest: List[Dict[str, str]], selected_name: Optional[str] = None) -> None:
    base = selected_baseline(df)
    if base is None:
        return

    if selected_name is None:
        ranked = compute_balanced_score(df[df["strategy_family"].isin(["quant_only", "prune_only", "quant_then_prune", "prune_then_quant"])]).copy()
        if ranked.empty:
            return
        selected = ranked.iloc[0]
    else:
        selected = first_row(df, selected_name)
        if selected is None:
            return

    metrics = [
        ("Token F1", "token_f1", False),
        ("ROUGE-L F1", "rouge_l_f1", False),
        ("Disk size MB", "disk_size_mb", True),
        ("Latency ms", "latency_ms_mean", True),
        ("Examples/s", "examples_per_second", False),
    ]

    labels = [m[0] for m in metrics]
    baseline_vals = [float(base[m[1]]) for m in metrics]
    selected_vals = [float(selected[m[1]]) for m in metrics]

    # Normalize each pair to make mixed units comparable in one report figure.
    baseline_norm, selected_norm = [], []
    for (_, _, lower_better), b, s in zip(metrics, baseline_vals, selected_vals):
        if lower_better:
            best = min(b, s)
            baseline_norm.append(best / b if b else np.nan)
            selected_norm.append(best / s if s else np.nan)
        else:
            best = max(b, s)
            baseline_norm.append(b / best if best else np.nan)
            selected_norm.append(s / best if best else np.nan)

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, baseline_norm, width, label=short_name(str(base["model_name"])).replace("\n", " "))
    ax.bar(x + width / 2, selected_norm, width, label=short_name(str(selected["model_name"])).replace("\n", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Normalized score within each metric\n(best of the two = 1.0)")
    ax.set_title("Selected Final Model vs Baseline Reference")
    ax.legend(fontsize=8)
    add_grid(ax)

    # Add raw values below the plot area as compact text.
    note = "Raw selected values: " + ", ".join([f"{label}={val:.4g}" for label, val in zip(labels, selected_vals)])
    ax.text(0.0, -0.28, note, transform=ax.transAxes, fontsize=8, va="top")

    save_fig(fig, outdir, "09_selected_model_vs_baseline_reference.jpg", "Selected model vs baseline reference", manifest)


# -----------------------------
# Analysis text
# -----------------------------

def write_analysis(df: pd.DataFrame, ranked: pd.DataFrame, outdir: Path) -> None:
    q = safe_family(df, "quant_only")
    p = safe_family(df, "prune_only")
    combined = df[df["strategy_family"].isin(["quant_then_prune", "prune_then_quant"])]

    lines = []
    lines.append("MODEL OPTIMIZATION ANALYSIS SUMMARY")
    lines.append("=" * 42)
    lines.append("")

    if not q.empty:
        best_q_acc = q.sort_values("token_f1", ascending=False).iloc[0]
        best_q_speed = q.sort_values("latency_ms_mean", ascending=True).iloc[0]
        best_q_size = q.sort_values("disk_size_mb", ascending=True).iloc[0]
        lines.append("Quantization-only findings:")
        lines.append(f"- Best quant-only accuracy: {best_q_acc['model_name']} with Token F1={best_q_acc['token_f1']:.4f}.")
        lines.append(f"- Fastest quant-only latency: {best_q_speed['model_name']} with mean latency={best_q_speed['latency_ms_mean']:.1f} ms.")
        lines.append(f"- Smallest quant-only disk size: {best_q_size['model_name']} with disk size={best_q_size['disk_size_mb']:.2f} MB.")
        lines.append("- Interpretation: BF16/FP16 preserve quality better because they keep more numerical information than INT8. INT8 compresses strongly, but the aggressive quantization can hurt generation quality.")
        lines.append("")

    if not p.empty:
        best_p_acc = p.sort_values("token_f1", ascending=False).iloc[0]
        best_p_speed = p.sort_values("latency_ms_mean", ascending=True).iloc[0]
        lines.append("Pruning-only findings:")
        lines.append(f"- Best prune-only accuracy: {best_p_acc['model_name']} with Token F1={best_p_acc['token_f1']:.4f}.")
        lines.append(f"- Fastest prune-only model: {best_p_speed['model_name']} with mean latency={best_p_speed['latency_ms_mean']:.1f} ms.")
        lines.append("- Interpretation: unstructured targeted pruning is safer than structured pruning here. Structured pruning can remove whole functional blocks and greatly improve speed, but it can also collapse quality if important heads or MLP dimensions are removed.")
        lines.append("- Disk-size caveat: unstructured pruning may not reduce file size unless the model is saved/executed using a sparse-aware format.")
        lines.append("")

    if not combined.empty:
        lines.append("Combined pipeline findings:")
        lines.append("- Quant-then-prune and prune-then-quant should be judged by whether the second step creates a new Pareto improvement.")
        lines.append("- In these results, most stacked methods do not create enough extra value: several keep the same disk size or lose accuracy while increasing memory.")
        lines.append("- Interpretation: stacking optimization methods can compound approximation error. It is useful only when the second step improves a deployment bottleneck without a large quality penalty.")
        lines.append("")

    if not ranked.empty:
        selected = ranked.iloc[0]
        lines.append("Final selection:")
        lines.append(f"- Selected by balanced score: {selected['model_name']} with score={selected['balanced_score']:.4f}.")
        lines.append("- The balanced score weights Token F1 most heavily, then ROUGE-L, throughput, latency, and disk size.")
        lines.append("- This makes the selected model a quality-preserving deployment choice rather than simply the smallest or fastest model.")
        lines.append("")

    base = selected_baseline(df)
    if base is not None and "eval_count" in df.columns:
        opt_eval_counts = df[df["strategy_family"].isin(["quant_only", "prune_only", "quant_then_prune", "prune_then_quant"])] ["eval_count"].dropna().unique()
        lines.append("Baseline caveat:")
        lines.append(f"- Baseline reference eval_count={base.get('eval_count', np.nan)}.")
        lines.append(f"- Optimized candidate eval_count values={list(opt_eval_counts)}.")
        lines.append("- If these counts differ, baseline comparison should be described as illustrative rather than fully apples-to-apples.")
        lines.append("")

    (outdir / "analysis_summary.txt").write_text("\n".join(lines), encoding="utf-8")


# -----------------------------
# Main runner
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate model optimization plots from model_summary_metrics.csv")
    parser.add_argument("--csv", type=str, default="model_summary_metrics.csv", help="Path to model_summary_metrics.csv")
    parser.add_argument("--outdir", type=str, default="model_optimization_report_assets", help="Output directory for JPG plots")
    parser.add_argument("--selected", type=str, default=None, help="Optional model_name to use as final selected model")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    outdir = ensure_outdir(Path(args.outdir))

    df = pd.read_csv(csv_path)

    # Make sure numeric columns are numeric even if loaded as strings.
    for col in ACCURACY_COLS + EFFICIENCY_COLS + ["size_reduction_pct_vs_source", "global_linear_sparsity", "eval_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    manifest: List[Dict[str, str]] = []

    # Quantization story
    plot_quant_accuracy_vs_size(df, outdir, manifest)
    plot_quant_accuracy_vs_latency(df, outdir, manifest)

    # Pruning story
    plot_prune_accuracy_vs_latency(df, outdir, manifest)
    plot_unstructured_sparsity_vs_accuracy(df, outdir, manifest)
    plot_prune_disk_reduction(df, outdir, manifest)

    # Combined optimization story
    plot_combined_parent_to_pipeline(df, outdir, manifest)
    plot_combined_relative_deltas(df, outdir, manifest)

    # Overall selection story
    ranked = plot_overall_ranking(df, outdir, manifest)
    plot_selected_vs_baseline(df, outdir, manifest, selected_name=args.selected)

    # Write helper files
    pd.DataFrame(manifest).to_csv(outdir / "plot_manifest.csv", index=False)
    write_analysis(df, ranked, outdir)

    print(f"Done. Exported {len(manifest)} JPG plots to: {outdir.resolve()}")
    print(f"Also wrote: {outdir / 'model_selection_scorecard.csv'}")
    print(f"Also wrote: {outdir / 'plot_manifest.csv'}")
    print(f"Also wrote: {outdir / 'analysis_summary.txt'}")


if __name__ == "__main__":
    main()
