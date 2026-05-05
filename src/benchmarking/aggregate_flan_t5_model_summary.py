from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OPTIMIZED_ROOT = Path("04_Experiments/optimized/flan_t5")
DEFAULT_BASELINE_ROOT = Path("04_Experiments/benchmark")
DEFAULT_OUTPUT_CSV = Path("04_Experiments/benchmark/plots/flan_t5/model_summary_metrics.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate one-row-per-model FLAN-T5 benchmark summaries into a single CSV."
    )
    parser.add_argument("--optimized-root", default=str(DEFAULT_OPTIMIZED_ROOT))
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    parser.add_argument(
        "--include-unevaluated",
        action="store_true",
        help="Include artifact directories even when benchmark metrics are missing.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_path(path: Path | str | None) -> str | None:
    if path is None:
        return None
    return str(path).replace("\\", "/")


def safe_get(mapping: dict[str, Any] | None, *keys: str) -> Any:
    current: Any = mapping or {}
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def to_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def detect_status(strategy_dir: Path, has_manifest: bool, has_metrics: bool) -> str:
    if has_metrics:
        return "evaluated"
    if (strategy_dir / "benchmark_smoke1" / "metrics.json").exists():
        return "smoke_only"
    if has_manifest:
        return "artifact_only"
    return "unknown"


def summarize_pipeline_step(step: dict[str, Any]) -> dict[str, Any]:
    op = step.get("op") or step.get("operation")
    summary = {
        "op": op,
        "criterion": step.get("criterion"),
        "family": step.get("family"),
        "amount": step.get("amount"),
        "dtype": step.get("dtype"),
        "heads_to_prune": step.get("heads_to_prune"),
        "global_linear_sparsity": safe_get(step, "audit", "global_linear_sparsity"),
        "zero_weights": safe_get(step, "audit", "zero_weights"),
        "total_weights": safe_get(step, "audit", "total_weights"),
        "linear_layer_count": safe_get(step, "audit", "linear_layer_count"),
        "mlp_neurons_pruned": safe_get(step, "audit", "neurons_pruned"),
        "mlp_neurons_total": safe_get(step, "audit", "neurons_total"),
        "attention_modules_pruned": safe_get(step, "audit", "pruned_module_count"),
        "attention_modules_total": safe_get(step, "audit", "total_module_count"),
        "attention_heads_removed": safe_get(step, "audit", "total_heads_removed"),
        "attention_heads_original": safe_get(step, "audit", "total_heads_original"),
    }
    return summary


def build_row(
    *,
    run_name: str,
    source: str,
    artifact_dir: Path,
    metrics_path: Path | None,
    manifest_path: Path | None,
    status: str,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    pipeline = safe_get(manifest, "optimization", "transform_pipeline") or []
    step_summaries = [summarize_pipeline_step(step) for step in pipeline if isinstance(step, dict)]

    quant_steps = [step for step in step_summaries if step.get("op") == "dynamic_quantize_linear"]
    prune_steps = [step for step in step_summaries if str(step.get("op", "")).startswith("prune_")]

    model_disk_mb = safe_get(metrics, "model", "disk_size_mb")
    source_disk_mb = safe_get(manifest, "source_model", "disk_size_mb")
    size_reduction_mb = None
    size_reduction_pct = None
    if isinstance(source_disk_mb, (int, float)) and isinstance(model_disk_mb, (int, float)):
        size_reduction_mb = round(source_disk_mb - model_disk_mb, 2)
        if source_disk_mb:
            size_reduction_pct = round(((source_disk_mb - model_disk_mb) / source_disk_mb) * 100.0, 3)

    return {
        "model_name": run_name,
        "source": source,
        "status": status,
        "strategy_family": manifest.get("strategy_family", "baseline"),
        "strategy_name": manifest.get("strategy_name", run_name),
        "runtime_target": manifest.get("runtime_target") or safe_get(metrics, "model", "runtime_target"),
        "benchmark_device": manifest.get("benchmark_device") or safe_get(metrics, "model", "device"),
        "torch_dtype": safe_get(metrics, "model", "torch_dtype"),
        "quantization_dtype": manifest.get("quantization_dtype"),
        "loader_kind": safe_get(metrics, "model", "loader_kind"),
        "requires_retraining": manifest.get("requires_retraining"),
        "requires_data": manifest.get("requires_data"),
        "tags": "|".join(manifest.get("tags", [])) if isinstance(manifest.get("tags"), list) else None,
        "quant_step_count": len(quant_steps),
        "prune_step_count": len(prune_steps),
        "optimization_pipeline_ops": "|".join(str(step.get("op")) for step in step_summaries if step.get("op")),
        "pruning_ops": "|".join(str(step.get("op")) for step in prune_steps if step.get("op")),
        "pruning_families": "|".join(str(step.get("family")) for step in prune_steps if step.get("family")),
        "pruning_criteria": "|".join(str(step.get("criterion")) for step in prune_steps if step.get("criterion")),
        "pruning_amounts": "|".join(str(step.get("amount")) for step in prune_steps if step.get("amount") is not None),
        "quant_dtypes_in_pipeline": "|".join(str(step.get("dtype")) for step in quant_steps if step.get("dtype")),
        "global_linear_sparsity": next(
            (step.get("global_linear_sparsity") for step in prune_steps if step.get("global_linear_sparsity") is not None),
            None,
        ),
        "zero_weights": next((step.get("zero_weights") for step in prune_steps if step.get("zero_weights") is not None), None),
        "total_weights": next((step.get("total_weights") for step in prune_steps if step.get("total_weights") is not None), None),
        "linear_layer_count": next(
            (step.get("linear_layer_count") for step in prune_steps if step.get("linear_layer_count") is not None),
            None,
        ),
        "mlp_neurons_pruned": next(
            (step.get("mlp_neurons_pruned") for step in prune_steps if step.get("mlp_neurons_pruned") is not None),
            None,
        ),
        "mlp_neurons_total": next(
            (step.get("mlp_neurons_total") for step in prune_steps if step.get("mlp_neurons_total") is not None),
            None,
        ),
        "attention_modules_pruned": next(
            (step.get("attention_modules_pruned") for step in prune_steps if step.get("attention_modules_pruned") is not None),
            None,
        ),
        "attention_modules_total": next(
            (step.get("attention_modules_total") for step in prune_steps if step.get("attention_modules_total") is not None),
            None,
        ),
        "attention_heads_removed": next(
            (step.get("attention_heads_removed") for step in prune_steps if step.get("attention_heads_removed") is not None),
            None,
        ),
        "attention_heads_original": next(
            (step.get("attention_heads_original") for step in prune_steps if step.get("attention_heads_original") is not None),
            None,
        ),
        "param_count": safe_get(metrics, "model", "param_count") or safe_get(manifest, "source_model", "param_count"),
        "disk_size_mb": model_disk_mb or safe_get(manifest, "source_model", "disk_size_mb"),
        "disk_size_bytes": safe_get(metrics, "model", "disk_size_bytes") or safe_get(manifest, "source_model", "disk_size_bytes"),
        "source_model_disk_size_mb": source_disk_mb,
        "size_reduction_mb_vs_source": size_reduction_mb,
        "size_reduction_pct_vs_source": size_reduction_pct,
        "eval_count": safe_get(metrics, "split", "eval_count") or safe_get(manifest, "dataset_contract", "eval_count"),
        "eval_loss": safe_get(metrics, "quality", "eval_loss"),
        "exact_match": safe_get(metrics, "quality", "exact_match"),
        "token_f1": safe_get(metrics, "quality", "token_f1"),
        "rouge_l_f1": safe_get(metrics, "quality", "rouge_l_f1"),
        "avg_generated_tokens": safe_get(metrics, "quality", "avg_generated_tokens"),
        "load_time_s": safe_get(metrics, "efficiency", "load_time_s"),
        "total_eval_time_s": safe_get(metrics, "efficiency", "total_eval_time_s"),
        "examples_per_second": safe_get(metrics, "efficiency", "examples_per_second"),
        "generated_tokens_per_second": safe_get(metrics, "efficiency", "generated_tokens_per_second"),
        "latency_ms_mean": safe_get(metrics, "efficiency", "latency_ms_mean"),
        "latency_ms_p50": safe_get(metrics, "efficiency", "latency_ms_p50"),
        "latency_ms_p95": safe_get(metrics, "efficiency", "latency_ms_p95"),
        "latency_ms_first_batch_per_example": safe_get(metrics, "efficiency", "latency_ms_first_batch_per_example"),
        "rss_before_load_mb": safe_get(metrics, "efficiency", "rss_before_load_mb"),
        "rss_after_load_mb": safe_get(metrics, "efficiency", "rss_after_load_mb"),
        "rss_delta_load_mb": safe_get(metrics, "efficiency", "rss_delta_load_mb"),
        "batch_size": safe_get(metrics, "generation", "batch_size"),
        "num_beams": safe_get(metrics, "generation", "num_beams"),
        "artifact_dir": normalize_path(artifact_dir),
        "metrics_path": normalize_path(metrics_path),
        "manifest_path": normalize_path(manifest_path),
        "pipeline_summary_json": to_json(step_summaries) if step_summaries else None,
    }


def read_optimized_runs(root: Path, include_unevaluated: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest_path = strategy_dir / "strategy_manifest.json"
        metrics_path = strategy_dir / "benchmark" / "metrics.json"

        has_manifest = manifest_path.exists()
        has_metrics = metrics_path.exists()
        if not include_unevaluated and not has_metrics:
            continue

        manifest = load_json(manifest_path) if has_manifest else {}
        metrics = load_json(metrics_path) if has_metrics else {}
        status = detect_status(strategy_dir, has_manifest, has_metrics)

        rows.append(
            build_row(
                run_name=strategy_dir.name,
                source="optimized",
                artifact_dir=strategy_dir,
                metrics_path=metrics_path if has_metrics else None,
                manifest_path=manifest_path if has_manifest else None,
                status=status,
                metrics=metrics,
                manifest=manifest,
            )
        )
    return rows


def read_baseline_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        if "optimized" in metrics_path.parts:
            continue

        metrics = load_json(metrics_path)
        run_name = f"{metrics_path.parent.parent.name}__{metrics_path.parent.name}"
        rows.append(
            build_row(
                run_name=run_name,
                source="benchmark",
                artifact_dir=metrics_path.parent,
                metrics_path=metrics_path,
                manifest_path=None,
                status="evaluated",
                metrics=metrics,
                manifest={},
            )
        )
    return rows


def all_columns(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "model_name",
        "strategy_name",
        "strategy_family",
        "source",
        "status",
        "runtime_target",
        "benchmark_device",
        "torch_dtype",
        "quantization_dtype",
        "quant_step_count",
        "prune_step_count",
        "optimization_pipeline_ops",
        "pruning_ops",
        "pruning_families",
        "pruning_criteria",
        "pruning_amounts",
        "global_linear_sparsity",
        "attention_heads_removed",
        "mlp_neurons_pruned",
        "param_count",
        "disk_size_mb",
        "source_model_disk_size_mb",
        "size_reduction_mb_vs_source",
        "size_reduction_pct_vs_source",
        "eval_count",
        "eval_loss",
        "exact_match",
        "token_f1",
        "rouge_l_f1",
        "load_time_s",
        "total_eval_time_s",
        "examples_per_second",
        "generated_tokens_per_second",
        "latency_ms_mean",
        "latency_ms_p50",
        "latency_ms_p95",
        "rss_delta_load_mb",
        "artifact_dir",
        "metrics_path",
        "manifest_path",
    ]
    discovered = sorted({key for row in rows for key in row.keys() if key not in preferred})
    return preferred + discovered


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    columns = all_columns(rows)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def main() -> None:
    args = parse_args()
    optimized_rows = read_optimized_runs(Path(args.optimized_root), include_unevaluated=args.include_unevaluated)
    baseline_rows = read_baseline_runs(Path(args.baseline_root))
    rows = baseline_rows + optimized_rows
    write_csv(rows, Path(args.output_csv))
    print(
        json.dumps(
            {
                "row_count": len(rows),
                "output_csv": normalize_path(args.output_csv),
                "included_unevaluated": bool(args.include_unevaluated),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
