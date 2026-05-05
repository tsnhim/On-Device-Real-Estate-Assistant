from __future__ import annotations

import argparse
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.benchmarking.benchmark_flan_t5 import BenchmarkConfig, run_benchmark
from src.benchmarking.flan_t5_data import load_split_manifest
from src.optimization.flan_t5_artifacts import (
    build_baseline_manifest,
    normalize_path,
    save_state_dict,
    utc_now_iso,
    write_json,
    write_strategy_manifest,
)
from src.optimization.flan_t5_registry import (
    DEFAULT_SELECTION_POLICY,
    StrategySpec,
    combined_strategy_templates,
    curated_v1_direct_specs,
)
from src.optimization.flan_t5_transforms import apply_pipeline_step_for_artifact


DEFAULT_MODEL_PATH = "models/flan_t5_zillow_final1"
DEFAULT_OUTPUT_ROOT = "benchmarks/optimized/flan_t5"
DEFAULT_SPLIT_MANIFEST = "benchmarks/data/flan_t5_baseline/split_manifest.json"


@dataclass
class OptimizerContext:
    model_path: Path
    output_root: Path
    split_manifest: Path
    benchmark_batch_size: int
    benchmark_max_eval_samples: int
    benchmark_max_source_length: int
    benchmark_max_target_length: int
    benchmark_max_label_length: int
    benchmark_num_beams: int
    benchmark_repetition_penalty: float
    benchmark_no_repeat_ngram_size: int
    selection_policy: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Registry-driven FLAN-T5 optimization framework with reproducible artifacts and benchmark integration."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_cmd = subparsers.add_parser("list-strategies", help="Show the curated direct and combined strategy matrix.")
    _add_shared_args(list_cmd)
    list_cmd.add_argument("--json", action="store_true", help="Print the strategy registry as JSON.")

    run_one = subparsers.add_parser("run-strategy", help="Create one strategy artifact and optionally benchmark it.")
    _add_shared_args(run_one)
    run_one.add_argument("--name", required=True, help="Strategy name from the curated matrix.")
    run_one.add_argument("--skip-benchmark", action="store_true", help="Create the artifact but do not run evaluation.")

    run_matrix = subparsers.add_parser(
        "run-curated-v1",
        help="Execute the full curated v1 matrix, benchmark direct strategies, then build selected combined artifacts.",
    )
    _add_shared_args(run_matrix)
    run_matrix.add_argument("--skip-benchmark", action="store_true", help="Build artifacts but skip benchmarking.")

    qat = subparsers.add_parser(
        "scaffold-qat",
        help="Write a phase-2 QAT recommendation scaffold tied to the saved train/eval split contract.",
    )
    _add_shared_args(qat)
    qat.add_argument(
        "--trigger-strategy",
        default="dynamic_int8_cpu_ptq",
        help="Parent strategy name that would trigger QAT if quality degradation is unacceptable.",
    )

    return parser.parse_args()


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Source FLAN-T5 checkpoint path.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Root directory for optimization artifacts.")
    parser.add_argument(
        "--split-manifest",
        default=DEFAULT_SPLIT_MANIFEST,
        help="Canonical split manifest used for all artifact evaluation and any future QAT training.",
    )
    parser.add_argument("--benchmark-batch-size", type=int, default=1)
    parser.add_argument("--benchmark-max-eval-samples", type=int, default=0)
    parser.add_argument("--benchmark-max-source-length", type=int, default=256)
    parser.add_argument("--benchmark-max-target-length", type=int, default=200)
    parser.add_argument("--benchmark-max-label-length", type=int, default=256)
    parser.add_argument("--benchmark-num-beams", type=int, default=4)
    parser.add_argument("--benchmark-repetition-penalty", type=float, default=1.3)
    parser.add_argument("--benchmark-no-repeat-ngram-size", type=int, default=3)


def build_context(args: argparse.Namespace) -> OptimizerContext:
    return OptimizerContext(
        model_path=Path(args.model_path),
        output_root=Path(args.output_root),
        split_manifest=Path(args.split_manifest),
        benchmark_batch_size=args.benchmark_batch_size,
        benchmark_max_eval_samples=args.benchmark_max_eval_samples,
        benchmark_max_source_length=args.benchmark_max_source_length,
        benchmark_max_target_length=args.benchmark_max_target_length,
        benchmark_max_label_length=args.benchmark_max_label_length,
        benchmark_num_beams=args.benchmark_num_beams,
        benchmark_repetition_penalty=args.benchmark_repetition_penalty,
        benchmark_no_repeat_ngram_size=args.benchmark_no_repeat_ngram_size,
        selection_policy=deepcopy(DEFAULT_SELECTION_POLICY),
    )


def baseline_model_summary(model: torch.nn.Module, model_path: Path) -> dict[str, Any]:
    disk_size = sum(path.stat().st_size for path in model_path.rglob("*") if path.is_file())
    param_count = sum(param.numel() for param in model.parameters())
    return {
        "source_model_path": normalize_path(model_path),
        "param_count": int(param_count),
        "disk_size_bytes": int(disk_size),
        "disk_size_mb": round(disk_size / (1024 * 1024), 2),
    }


def load_base_model_and_tokenizer(model_path: Path) -> tuple[torch.nn.Module, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    model.eval()
    return model, tokenizer


def ensure_dataset_contract(context: OptimizerContext) -> dict[str, Any]:
    manifest = load_split_manifest(context.split_manifest)
    pair_cache_path = context.split_manifest.parent / "qa_pairs.jsonl"
    if not pair_cache_path.exists():
        raise FileNotFoundError(
            f"Expected pair cache at {pair_cache_path}. Rebuild it with `python -m src.benchmarking.build_flan_t5_split --source hf`."
        )
    return {
        "split_manifest_path": normalize_path(context.split_manifest),
        "pair_cache_path": normalize_path(pair_cache_path),
        "seed": manifest["split_seed"],
        "train_count": manifest["train_count"],
        "eval_count": manifest["eval_count"],
        "split_test_size": manifest["split_test_size"],
        "source_name": manifest["source_name"],
    }


def ensure_baseline_manifest(context: OptimizerContext) -> Path:
    artifact_dir = context.output_root / "baseline_reference"
    manifest_path = artifact_dir / "strategy_manifest.json"
    if manifest_path.exists():
        return manifest_path

    model, _ = load_base_model_and_tokenizer(context.model_path)
    source_summary = baseline_model_summary(model, context.model_path)
    manifest = build_baseline_manifest(
        artifact_dir=artifact_dir,
        model_path=context.model_path,
        split_manifest_path=context.split_manifest,
        source_model_summary=source_summary,
    )
    manifest["selection_policy"] = deepcopy(context.selection_policy)
    manifest["benchmark_request"] = benchmark_request_payload(context, artifact_dir / "benchmark")
    write_strategy_manifest(manifest_path, manifest)
    write_json(artifact_dir / "benchmark_request.json", manifest["benchmark_request"])
    return manifest_path


def strategy_output_dir(context: OptimizerContext, strategy_name: str) -> Path:
    return context.output_root / strategy_name


def benchmark_request_payload(context: OptimizerContext, output_dir: Path, device: str = "auto") -> dict[str, Any]:
    return {
        "split_manifest": normalize_path(context.split_manifest),
        "output_dir": normalize_path(output_dir),
        "batch_size": context.benchmark_batch_size,
        "max_eval_samples": context.benchmark_max_eval_samples,
        "max_source_length": context.benchmark_max_source_length,
        "max_target_length": context.benchmark_max_target_length,
        "max_label_length": context.benchmark_max_label_length,
        "num_beams": context.benchmark_num_beams,
        "repetition_penalty": context.benchmark_repetition_penalty,
        "no_repeat_ngram_size": context.benchmark_no_repeat_ngram_size,
        "device": device,
    }


def build_manifest_base(
    *,
    context: OptimizerContext,
    spec: StrategySpec,
    source_model_summary: dict[str, Any],
    parent_sources: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    artifact_dir = strategy_output_dir(context, spec.name)
    loader: dict[str, Any]
    if spec.output_loader_kind == "transformers_checkpoint":
        loader = {
            "kind": "transformers_checkpoint",
            "artifact_model_path": normalize_path(context.model_path),
            "tokenizer_source_path": normalize_path(context.model_path),
        }
    else:
        loader = {
            "kind": spec.output_loader_kind,
            "reconstruction_base_model_path": normalize_path(context.model_path),
            "artifact_state_dict_path": normalize_path(artifact_dir / "artifact_state_dict.pt"),
            "tokenizer_source_path": normalize_path(artifact_dir / "tokenizer"),
            "pipeline": [],
        }

    benchmark_device = spec.benchmark_device
    if benchmark_device == "cuda" and not torch.cuda.is_available():
        benchmark_device = "cpu" if spec.runtime_target == "cpu" else "auto"

    return {
        "created_at_utc": utc_now_iso(),
        "status": "ready",
        "strategy_name": spec.name,
        "strategy_family": spec.family,
        "execution_order": spec.execution_order,
        "runtime_target": spec.runtime_target,
        "benchmark_device": benchmark_device,
        "quantization_dtype": spec.quantization_dtype,
        "layer_policy": spec.layer_policy,
        "requires_data": spec.requires_data,
        "requires_retraining": spec.requires_retraining,
        "source_selector": spec.source_selector,
        "strategy_notes": spec.notes,
        "tags": list(spec.tags),
        "dataset_contract": ensure_dataset_contract(context),
        "selection_policy": deepcopy(context.selection_policy),
        "source_model": deepcopy(source_model_summary),
        "parent_sources": parent_sources or [],
        "optimization": {
            "transform_pipeline": [],
            "quantization": None,
            "pruning": None,
            "combined_selection": None,
        },
        "loader": loader,
        "artifact_dir": normalize_path(artifact_dir),
    }


def support_status_for_spec(spec: StrategySpec) -> tuple[str, str | None]:
    if spec.name == "bf16_cast_gpu":
        if not torch.cuda.is_available():
            return "unsupported", "BF16 GPU strategy skipped because CUDA is not available in the active environment."
        if not torch.cuda.is_bf16_supported():
            return "unsupported", "BF16 GPU strategy skipped because the active CUDA runtime does not report BF16 support."
    if spec.benchmark_device == "cuda" and not torch.cuda.is_available():
        return "ready", "CUDA is unavailable, so benchmarking will fall back to auto or CPU while keeping the artifact definition intact."
    return "ready", None


def apply_strategy_pipeline(model: torch.nn.Module, spec: StrategySpec) -> tuple[torch.nn.Module, list[dict[str, Any]]]:
    audit_steps: list[dict[str, Any]] = []
    for step in spec.transform_pipeline:
        model, audit = apply_pipeline_step_for_artifact(model, step)
        if audit is None:
            continue
        if "op" not in audit:
            audit = {"op": audit.get("operation", step["op"]), **audit}
        audit_steps.append(audit)
    return model, audit_steps


def summarize_pipeline(audits: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    quantization = None
    pruning_steps: list[dict[str, Any]] = []
    for audit in audits:
        op = audit["op"]
        if op in {"cast_dtype", "dynamic_quantize_linear"}:
            quantization = deepcopy(audit)
        elif op.startswith("prune_"):
            pruning_steps.append(deepcopy(audit))
    pruning = pruning_steps if pruning_steps else None
    return quantization, pruning


def save_pipeline_artifact(
    *,
    context: OptimizerContext,
    spec: StrategySpec,
    parent_sources: list[dict[str, Any]] | None = None,
    benchmark_enabled: bool = True,
) -> Path:
    artifact_dir = strategy_output_dir(context, spec.name)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "strategy_manifest.json"

    model, tokenizer = load_base_model_and_tokenizer(context.model_path)
    source_summary = baseline_model_summary(model, context.model_path)
    status, status_reason = support_status_for_spec(spec)
    manifest = build_manifest_base(
        context=context,
        spec=spec,
        source_model_summary=source_summary,
        parent_sources=parent_sources,
    )

    benchmark_dir = artifact_dir / "benchmark"
    manifest["benchmark_request"] = benchmark_request_payload(
        context,
        benchmark_dir,
        device=manifest["benchmark_device"],
    )
    write_json(artifact_dir / "benchmark_request.json", manifest["benchmark_request"])

    if status == "unsupported":
        manifest["status"] = status
        manifest["status_reason"] = status_reason
        write_strategy_manifest(manifest_path, manifest)
        return manifest_path

    optimized_model, audits = apply_strategy_pipeline(model, spec)
    tokenizer_dir = artifact_dir / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)
    save_state_dict(optimized_model, artifact_dir / "artifact_state_dict.pt")

    quantization, pruning = summarize_pipeline(audits)
    manifest["status"] = "ready"
    if status_reason:
        manifest["status_reason"] = status_reason
    manifest["optimization"]["transform_pipeline"] = audits
    manifest["optimization"]["quantization"] = quantization
    manifest["optimization"]["pruning"] = pruning
    manifest["loader"]["pipeline"] = audits
    manifest["benchmark_enabled"] = benchmark_enabled and spec.benchmark_enabled
    manifest["artifact_files"] = {
        "state_dict": normalize_path(artifact_dir / "artifact_state_dict.pt"),
        "tokenizer_dir": normalize_path(tokenizer_dir),
    }
    write_strategy_manifest(manifest_path, manifest)
    return manifest_path


def build_combined_spec(template: dict[str, Any], first: StrategySpec, second: StrategySpec) -> StrategySpec:
    runtime_target = first.runtime_target if "quant" in template["family"] and template["family"] == "quant_then_prune" else second.runtime_target
    benchmark_device = second.benchmark_device
    if template["family"] == "quant_then_prune":
        benchmark_device = second.benchmark_device
    elif template["family"] == "prune_then_quant":
        benchmark_device = second.benchmark_device

    combined_name = f"{template['name']}__{first.name}__{second.name}"
    return StrategySpec(
        name=combined_name,
        family=template["family"],
        execution_order=max(first.execution_order, second.execution_order) + 100,
        runtime_target=runtime_target,
        benchmark_device=benchmark_device,
        quantization_dtype=second.quantization_dtype or first.quantization_dtype,
        layer_policy=f"{first.layer_policy}+{second.layer_policy}",
        requires_data=first.requires_data or second.requires_data,
        requires_retraining=first.requires_retraining or second.requires_retraining,
        output_loader_kind="pipeline_state_dict",
        source_selector=f"{template['selection_role']}:{first.name}->{second.name}",
        transform_pipeline=deepcopy(first.transform_pipeline) + deepcopy(second.transform_pipeline),
        tags=tuple(sorted(set(first.tags + second.tags + ("combined",)))),
        notes=template["notes"],
    )


def metrics_path_for_strategy(context: OptimizerContext, strategy_name: str) -> Path:
    return strategy_output_dir(context, strategy_name) / "benchmark" / "metrics.json"


def load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def passes_quality_gate(candidate_metrics: dict[str, Any], baseline_metrics: dict[str, Any], policy: dict[str, Any]) -> bool:
    baseline_token_f1 = baseline_metrics["quality"]["token_f1"]
    baseline_eval_loss = baseline_metrics["quality"]["eval_loss"]
    candidate_token_f1 = candidate_metrics["quality"]["token_f1"]
    candidate_eval_loss = candidate_metrics["quality"]["eval_loss"]

    if baseline_token_f1 > 0:
        token_ratio = candidate_token_f1 / baseline_token_f1
        if token_ratio < policy["min_token_f1_ratio"]:
            return False
    if baseline_eval_loss > 0:
        loss_ratio = candidate_eval_loss / baseline_eval_loss
        if loss_ratio > policy["max_eval_loss_ratio"]:
            return False
    return True


def candidate_efficiency_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    return (
        metrics["model"]["disk_size_bytes"],
        metrics["efficiency"]["latency_ms_mean"],
        -metrics["quality"]["token_f1"],
    )


def select_best_strategy_name(
    *,
    context: OptimizerContext,
    candidate_names: list[str],
    baseline_metrics: dict[str, Any],
) -> str:
    eligible: list[tuple[tuple[float, float, float], str]] = []
    fallback: list[tuple[tuple[float, float, float], str]] = []
    for name in candidate_names:
        metrics_file = metrics_path_for_strategy(context, name)
        if not metrics_file.exists():
            continue
        metrics = load_metrics(metrics_file)
        sort_key = candidate_efficiency_key(metrics)
        fallback.append((sort_key, name))
        if passes_quality_gate(metrics, baseline_metrics, context.selection_policy):
            eligible.append((sort_key, name))

    if eligible:
        eligible.sort()
        return eligible[0][1]
    if fallback:
        fallback.sort()
        return fallback[0][1]
    raise FileNotFoundError(f"No benchmark metrics found for candidates: {candidate_names}")


def direct_specs_by_name() -> dict[str, StrategySpec]:
    return {spec.name: spec for spec in curated_v1_direct_specs()}


def benchmark_from_manifest(context: OptimizerContext, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_dir = Path(manifest["benchmark_request"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config = BenchmarkConfig(
        strategy_manifest=normalize_path(manifest_path),
        split_manifest=manifest["benchmark_request"]["split_manifest"],
        output_dir=manifest["benchmark_request"]["output_dir"],
        batch_size=manifest["benchmark_request"]["batch_size"],
        max_eval_samples=manifest["benchmark_request"]["max_eval_samples"],
        max_source_length=manifest["benchmark_request"]["max_source_length"],
        max_target_length=manifest["benchmark_request"]["max_target_length"],
        max_label_length=manifest["benchmark_request"]["max_label_length"],
        num_beams=manifest["benchmark_request"]["num_beams"],
        repetition_penalty=manifest["benchmark_request"]["repetition_penalty"],
        no_repeat_ngram_size=manifest["benchmark_request"]["no_repeat_ngram_size"],
        device=manifest["benchmark_request"]["device"],
    )
    return run_benchmark(config)


def benchmark_baseline(context: OptimizerContext, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = BenchmarkConfig(
        model_path=context.model_path.as_posix(),
        split_manifest=manifest["benchmark_request"]["split_manifest"],
        output_dir=manifest["benchmark_request"]["output_dir"],
        batch_size=manifest["benchmark_request"]["batch_size"],
        max_eval_samples=manifest["benchmark_request"]["max_eval_samples"],
        max_source_length=manifest["benchmark_request"]["max_source_length"],
        max_target_length=manifest["benchmark_request"]["max_target_length"],
        max_label_length=manifest["benchmark_request"]["max_label_length"],
        num_beams=manifest["benchmark_request"]["num_beams"],
        repetition_penalty=manifest["benchmark_request"]["repetition_penalty"],
        no_repeat_ngram_size=manifest["benchmark_request"]["no_repeat_ngram_size"],
        device=manifest["benchmark_request"]["device"],
    )
    return run_benchmark(config)


def maybe_benchmark_strategy(context: OptimizerContext, manifest_path: Path, skip_benchmark: bool) -> dict[str, Any] | None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if skip_benchmark:
        return None
    if manifest.get("status") == "unsupported":
        return None
    if manifest["strategy_name"] == "baseline_reference":
        return benchmark_baseline(context, manifest_path)
    return benchmark_from_manifest(context, manifest_path)


def run_direct_strategy(context: OptimizerContext, spec: StrategySpec, skip_benchmark: bool) -> Path:
    if spec.name == "baseline_reference":
        manifest_path = ensure_baseline_manifest(context)
    else:
        manifest_path = save_pipeline_artifact(context=context, spec=spec)
    maybe_benchmark_strategy(context, manifest_path, skip_benchmark)
    return manifest_path


def resolve_combined_specs(context: OptimizerContext) -> list[StrategySpec]:
    baseline_metrics = load_metrics(metrics_path_for_strategy(context, "baseline_reference"))
    policy = context.selection_policy
    best_gpu_quant = select_best_strategy_name(
        context=context,
        candidate_names=policy["fallback_order_quant_gpu"],
        baseline_metrics=baseline_metrics,
    )
    best_cpu_quant = select_best_strategy_name(
        context=context,
        candidate_names=policy["fallback_order_quant_cpu"],
        baseline_metrics=baseline_metrics,
    )
    best_unstructured_prune = select_best_strategy_name(
        context=context,
        candidate_names=policy["fallback_order_unstructured_prune"],
        baseline_metrics=baseline_metrics,
    )
    best_structured_prune = select_best_strategy_name(
        context=context,
        candidate_names=policy["fallback_order_structured_prune"],
        baseline_metrics=baseline_metrics,
    )

    direct = direct_specs_by_name()
    templates = {item["selection_role"]: item for item in combined_strategy_templates()}
    return [
        build_combined_spec(templates["quant_then_prune"], direct[best_gpu_quant], direct[best_unstructured_prune]),
        build_combined_spec(templates["unstructured_prune_then_quant"], direct[best_unstructured_prune], direct[best_cpu_quant]),
        build_combined_spec(templates["structured_prune_then_quant"], direct[best_structured_prune], direct[best_gpu_quant]),
    ]


def run_combined_strategy(context: OptimizerContext, spec: StrategySpec, skip_benchmark: bool) -> Path:
    steps = spec.source_selector.split(":", 1)[1]
    first_name, second_name = steps.split("->", 1)
    parent_sources = [
        {
            "strategy_name": first_name,
            "manifest_path": normalize_path(strategy_output_dir(context, first_name) / "strategy_manifest.json"),
            "metrics_path": normalize_path(metrics_path_for_strategy(context, first_name)),
        },
        {
            "strategy_name": second_name,
            "manifest_path": normalize_path(strategy_output_dir(context, second_name) / "strategy_manifest.json"),
            "metrics_path": normalize_path(metrics_path_for_strategy(context, second_name)),
        },
    ]
    manifest_path = save_pipeline_artifact(
        context=context,
        spec=spec,
        parent_sources=parent_sources,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["optimization"]["combined_selection"] = {
        "selection_role": spec.source_selector.split(":", 1)[0],
        "parents": parent_sources,
        "selection_policy": deepcopy(context.selection_policy),
    }
    write_strategy_manifest(manifest_path, manifest)
    maybe_benchmark_strategy(context, manifest_path, skip_benchmark)
    return manifest_path


def scaffold_qat(context: OptimizerContext, trigger_strategy: str) -> Path:
    baseline_manifest_path = ensure_baseline_manifest(context)
    artifact_dir = context.output_root / "qat_recommendation"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "strategy_manifest.json"
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    payload = {
        "created_at_utc": utc_now_iso(),
        "status": "planned",
        "strategy_name": "qat_phase2_scaffold",
        "strategy_family": "qat",
        "runtime_target": "phase2",
        "requires_data": True,
        "requires_retraining": True,
        "trigger_parent_strategy": trigger_strategy,
        "dataset_contract": baseline_manifest["dataset_contract"],
        "selection_policy": deepcopy(context.selection_policy),
        "notes": (
            "QAT is reserved as a phase-2 fallback for the most promising INT8 path. "
            "Promote this scaffold into a trainable experiment only if PTQ or prune-plus-quant breaches the quality gate."
        ),
        "expected_outputs": [
            "trained_checkpoint",
            "benchmark_results",
            "strategy_manifest",
            "parent_strategy_reference",
        ],
    }
    write_strategy_manifest(manifest_path, payload)
    return manifest_path


def render_registry_payload() -> dict[str, Any]:
    return {
        "selection_policy": deepcopy(DEFAULT_SELECTION_POLICY),
        "direct_strategies": [spec.to_manifest_summary() for spec in curated_v1_direct_specs()],
        "combined_templates": combined_strategy_templates(),
        "qat_family": {
            "strategy_family": "qat",
            "phase": "phase_2_fallback",
            "notes": "Reserved for the most promising INT8 path after benchmark-driven quality gating.",
        },
    }


def main() -> None:
    args = parse_args()
    context = build_context(args)

    if args.command == "list-strategies":
        payload = render_registry_payload()
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            for item in payload["direct_strategies"]:
                print(f"{item['strategy_name']}: {item['strategy_family']} [{item['runtime_target']}]")
            for item in payload["combined_templates"]:
                print(f"{item['name']}: {item['family']} [selected]")
            print("qat_phase2_scaffold: qat [planned]")
        return

    if args.command == "scaffold-qat":
        manifest_path = scaffold_qat(context, args.trigger_strategy)
        print(f"Saved QAT scaffold manifest to {normalize_path(manifest_path)}")
        return

    direct = direct_specs_by_name()

    if args.command == "run-strategy":
        if args.name == "qat_phase2_scaffold":
            manifest_path = scaffold_qat(context, "dynamic_int8_cpu_ptq")
            print(f"Saved QAT scaffold manifest to {normalize_path(manifest_path)}")
            return

        if args.name in direct:
            manifest_path = run_direct_strategy(context, direct[args.name], args.skip_benchmark)
            print(f"Saved strategy manifest to {normalize_path(manifest_path)}")
            return

        combined_specs = {spec.name: spec for spec in resolve_combined_specs(context)}
        if args.name not in combined_specs:
            raise ValueError(f"Unknown strategy: {args.name}")
        manifest_path = run_combined_strategy(context, combined_specs[args.name], args.skip_benchmark)
        print(f"Saved strategy manifest to {normalize_path(manifest_path)}")
        return

    if args.command == "run-curated-v1":
        manifest_paths: list[Path] = []
        for spec in curated_v1_direct_specs():
            manifest_paths.append(run_direct_strategy(context, spec, args.skip_benchmark))
        if not args.skip_benchmark:
            combined_specs = resolve_combined_specs(context)
            for spec in combined_specs:
                manifest_paths.append(run_combined_strategy(context, spec, args.skip_benchmark))
        qat_path = scaffold_qat(context, "dynamic_int8_cpu_ptq")
        manifest_paths.append(qat_path)
        for path in manifest_paths:
            print(normalize_path(path))
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
