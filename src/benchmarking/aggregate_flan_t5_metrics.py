from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_OPTIMIZED_ROOT = Path("04_Experiments/optimized/flan_t5")
DEFAULT_BASELINE_ROOT = Path("04_Experiments/benchmark")
DEFAULT_OUTPUT_CSV = Path("04_Experiments/benchmark/plots/flan_t5/all_model_metrics.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate FLAN-T5 runs, metrics, manifests, and predictions into one CSV."
    )
    parser.add_argument("--optimized-root", default=str(DEFAULT_OPTIMIZED_ROOT))
    parser.add_argument("--baseline-root", default=str(DEFAULT_BASELINE_ROOT))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_path(path: Path | str | None) -> str | None:
    if path is None:
        return None
    return str(path).replace("\\", "/")


def scalar_or_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def flatten_mapping(value: dict[str, Any], prefix: str, target: dict[str, Any]) -> None:
    for key, item in value.items():
        column = f"{prefix}{key}" if prefix else key
        if isinstance(item, dict):
            flatten_mapping(item, f"{column}__", target)
            continue
        target[column] = scalar_or_json(item)


def detect_status(strategy_dir: Path, has_manifest: bool, has_metrics: bool) -> str:
    if has_metrics:
        return "evaluated"
    if (strategy_dir / "benchmark_smoke1" / "metrics.json").exists():
        return "smoke_only"
    if has_manifest:
        return "artifact_only"
    return "unknown"


def build_common_row(
    *,
    run_name: str,
    source: str,
    artifact_dir: Path,
    metrics_path: Path | None,
    predictions_path: Path | None,
    manifest_path: Path | None,
    status: str,
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "source": source,
        "status": status,
        "artifact_dir": normalize_path(artifact_dir),
        "metrics_path": normalize_path(metrics_path),
        "predictions_path": normalize_path(predictions_path),
        "manifest_path": normalize_path(manifest_path),
    }


def build_rows_for_run(
    *,
    common: dict[str, Any],
    family: str | None,
    group_name: str,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    base_row = dict(common)
    base_row["family"] = family
    base_row["group_name"] = group_name
    base_row["row_type"] = "prediction" if predictions else "run_summary"
    flatten_mapping(manifest, "manifest__", base_row)
    flatten_mapping(metrics, "metrics__", base_row)

    if not predictions:
        return [base_row]

    rows: list[dict[str, Any]] = []
    for prediction in predictions:
        row = dict(base_row)
        flatten_mapping(prediction, "prediction__", row)
        rows.append(row)
    return rows


def read_optimized_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for strategy_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest_path = strategy_dir / "strategy_manifest.json"
        metrics_path = strategy_dir / "benchmark" / "metrics.json"
        predictions_path = strategy_dir / "benchmark" / "predictions.jsonl"

        has_manifest = manifest_path.exists()
        has_metrics = metrics_path.exists()

        manifest = load_json(manifest_path) if has_manifest else {}
        metrics = load_json(metrics_path) if has_metrics else {}
        predictions = load_jsonl(predictions_path) if predictions_path.exists() else []

        common = build_common_row(
            run_name=strategy_dir.name,
            source="optimized",
            artifact_dir=strategy_dir,
            metrics_path=metrics_path if has_metrics else None,
            predictions_path=predictions_path if predictions_path.exists() else None,
            manifest_path=manifest_path if has_manifest else None,
            status=detect_status(strategy_dir, has_manifest, has_metrics),
        )
        rows.extend(
            build_rows_for_run(
                common=common,
                family=manifest.get("strategy_family"),
                group_name=manifest.get("strategy_name", strategy_dir.name),
                manifest=manifest,
                metrics=metrics,
                predictions=predictions,
            )
        )
    return rows


def read_baseline_runs(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.rglob("metrics.json")):
        if "optimized" in metrics_path.parts:
            continue

        predictions_path = metrics_path.parent / "predictions.jsonl"
        metrics = load_json(metrics_path)
        predictions = load_jsonl(predictions_path) if predictions_path.exists() else []
        run_name = f"{metrics_path.parent.parent.name}__{metrics_path.parent.name}"

        common = build_common_row(
            run_name=run_name,
            source="benchmark",
            artifact_dir=metrics_path.parent,
            metrics_path=metrics_path,
            predictions_path=predictions_path if predictions_path.exists() else None,
            manifest_path=None,
            status="evaluated",
        )
        rows.extend(
            build_rows_for_run(
                common=common,
                family="baseline",
                group_name=metrics_path.parent.parent.name,
                manifest={},
                metrics=metrics,
                predictions=predictions,
            )
        )
    return rows


def collect_rows(optimized_root: Path, baseline_root: Path) -> list[dict[str, Any]]:
    return read_baseline_runs(baseline_root) + read_optimized_runs(optimized_root)


def all_columns(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "run_name",
        "group_name",
        "family",
        "source",
        "status",
        "row_type",
        "artifact_dir",
        "metrics_path",
        "predictions_path",
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
    rows = collect_rows(Path(args.optimized_root), Path(args.baseline_root))
    write_csv(rows, Path(args.output_csv))
    print(
        json.dumps(
            {
                "row_count": len(rows),
                "output_csv": normalize_path(args.output_csv),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
