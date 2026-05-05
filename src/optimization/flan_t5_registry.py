from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


DEFAULT_SELECTION_POLICY = {
    "min_token_f1_ratio": 0.5,
    "max_eval_loss_ratio": 1.75,
    "fallback_order_quant_gpu": ["fp16_cast_gpu", "bf16_cast_gpu"],
    "fallback_order_quant_cpu": ["dynamic_int8_cpu_ptq"],
    "fallback_order_unstructured_prune": [
        "prune_unstructured_mlp_l1_10",
        "prune_unstructured_attention_l1_10",
        "prune_unstructured_global_l1_10",
        "prune_unstructured_global_l1_20",
    ],
    "fallback_order_structured_prune": [
        "prune_structured_mlp_intermediate_10",
        "prune_structured_attention_heads_1",
    ],
}


@dataclass(frozen=True)
class StrategySpec:
    name: str
    family: str
    execution_order: int
    runtime_target: str
    benchmark_device: str
    quantization_dtype: str | None
    layer_policy: str
    requires_data: bool
    requires_retraining: bool
    output_loader_kind: str
    source_selector: str
    transform_pipeline: list[dict[str, Any]] = field(default_factory=list)
    tags: tuple[str, ...] = ()
    notes: str = ""
    benchmark_enabled: bool = True

    def to_manifest_summary(self) -> dict[str, Any]:
        return {
            "strategy_name": self.name,
            "strategy_family": self.family,
            "execution_order": self.execution_order,
            "runtime_target": self.runtime_target,
            "benchmark_device": self.benchmark_device,
            "quantization_dtype": self.quantization_dtype,
            "layer_policy": self.layer_policy,
            "requires_data": self.requires_data,
            "requires_retraining": self.requires_retraining,
            "output_loader_kind": self.output_loader_kind,
            "source_selector": self.source_selector,
            "transform_pipeline": self.transform_pipeline,
            "tags": list(self.tags),
            "notes": self.notes,
            "benchmark_enabled": self.benchmark_enabled,
        }


def curated_v1_direct_specs() -> list[StrategySpec]:
    return [
        StrategySpec(
            name="baseline_reference",
            family="baseline",
            execution_order=0,
            runtime_target="baseline",
            benchmark_device="auto",
            quantization_dtype=None,
            layer_policy="none",
            requires_data=False,
            requires_retraining=False,
            output_loader_kind="transformers_checkpoint",
            source_selector="baseline",
            notes="Reference full-precision checkpoint benchmark.",
        ),
        StrategySpec(
            name="fp16_cast_gpu",
            family="quant_only",
            execution_order=10,
            runtime_target="gpu",
            benchmark_device="cuda",
            quantization_dtype="float16",
            layer_policy="all_weights",
            requires_data=False,
            requires_retraining=False,
            output_loader_kind="pipeline_state_dict",
            source_selector="baseline",
            transform_pipeline=[{"op": "cast_dtype", "dtype": "float16"}],
            tags=("quant", "gpu", "fp16"),
            notes="GPU-first cast/export strategy using FP16 weights.",
        ),
        StrategySpec(
            name="bf16_cast_gpu",
            family="quant_only",
            execution_order=11,
            runtime_target="gpu",
            benchmark_device="cuda",
            quantization_dtype="bfloat16",
            layer_policy="all_weights",
            requires_data=False,
            requires_retraining=False,
            output_loader_kind="pipeline_state_dict",
            source_selector="baseline",
            transform_pipeline=[{"op": "cast_dtype", "dtype": "bfloat16"}],
            tags=("quant", "gpu", "bf16"),
            notes="GPU-first cast/export strategy using BF16 weights when runtime support exists.",
        ),
        StrategySpec(
            name="dynamic_int8_cpu_ptq",
            family="quant_only",
            execution_order=12,
            runtime_target="cpu",
            benchmark_device="cpu",
            quantization_dtype="qint8",
            layer_policy="linear_only",
            requires_data=False,
            requires_retraining=False,
            output_loader_kind="pipeline_state_dict",
            source_selector="baseline",
            transform_pipeline=[{"op": "dynamic_quantize_linear", "dtype": "qint8"}],
            tags=("quant", "cpu", "int8", "ptq"),
            notes="Retained CPU PTQ baseline using dynamic INT8 over Linear layers.",
        ),
        StrategySpec(
            name="prune_unstructured_global_l1_10",
            family="prune_only",
            execution_order=20,
            runtime_target="gpu",
            benchmark_device="cuda",
            quantization_dtype=None,
            layer_policy="all_linear",
            requires_data=False,
            requires_retraining=False,
            output_loader_kind="pipeline_state_dict",
            source_selector="baseline",
            transform_pipeline=[
                {"op": "prune_unstructured", "criterion": "l1_global", "family": "all_linear", "amount": 0.10}
            ],
            tags=("prune", "unstructured"),
            notes="Global L1 unstructured pruning over all linear layers at 10%.",
        ),
        StrategySpec(
            name="prune_unstructured_global_l1_20",
            family="prune_only",
            execution_order=21,
            runtime_target="gpu",
            benchmark_device="cuda",
            quantization_dtype=None,
            layer_policy="all_linear",
            requires_data=False,
            requires_retraining=False,
            output_loader_kind="pipeline_state_dict",
            source_selector="baseline",
            transform_pipeline=[
                {"op": "prune_unstructured", "criterion": "l1_global", "family": "all_linear", "amount": 0.20}
            ],
            tags=("prune", "unstructured"),
            notes="Global L1 unstructured pruning over all linear layers at 20%.",
        ),
        StrategySpec(
            name="prune_unstructured_attention_l1_10",
            family="prune_only",
            execution_order=22,
            runtime_target="gpu",
            benchmark_device="cuda",
            quantization_dtype=None,
            layer_policy="attention_linear",
            requires_data=False,
            requires_retraining=False,
            output_loader_kind="pipeline_state_dict",
            source_selector="baseline",
            transform_pipeline=[
                {"op": "prune_unstructured", "criterion": "l1_global", "family": "attention_linear", "amount": 0.10}
            ],
            tags=("prune", "unstructured", "attention"),
            notes="Attention-only unstructured pruning at 10%.",
        ),
        StrategySpec(
            name="prune_unstructured_mlp_l1_10",
            family="prune_only",
            execution_order=23,
            runtime_target="gpu",
            benchmark_device="cuda",
            quantization_dtype=None,
            layer_policy="mlp_linear",
            requires_data=False,
            requires_retraining=False,
            output_loader_kind="pipeline_state_dict",
            source_selector="baseline",
            transform_pipeline=[
                {"op": "prune_unstructured", "criterion": "l1_global", "family": "mlp_linear", "amount": 0.10}
            ],
            tags=("prune", "unstructured", "mlp"),
            notes="MLP-only unstructured pruning at 10%.",
        ),
        StrategySpec(
            name="prune_structured_mlp_intermediate_10",
            family="prune_only",
            execution_order=24,
            runtime_target="gpu",
            benchmark_device="cuda",
            quantization_dtype=None,
            layer_policy="mlp_intermediate",
            requires_data=False,
            requires_retraining=False,
            output_loader_kind="pipeline_state_dict",
            source_selector="baseline",
            transform_pipeline=[
                {"op": "prune_structured_mlp", "criterion": "l1_neuron", "family": "mlp_linear", "amount": 0.10}
            ],
            tags=("prune", "structured", "mlp"),
            notes="Structured pruning over MLP intermediate neurons at a conservative 10%.",
        ),
        StrategySpec(
            name="prune_structured_attention_heads_1",
            family="prune_only",
            execution_order=25,
            runtime_target="gpu",
            benchmark_device="cuda",
            quantization_dtype=None,
            layer_policy="attention_heads",
            requires_data=False,
            requires_retraining=False,
            output_loader_kind="pipeline_state_dict",
            source_selector="baseline",
            transform_pipeline=[
                {"op": "prune_structured_attention", "criterion": "l1_head", "family": "attention_linear", "heads_to_prune": 1}
            ],
            tags=("prune", "structured", "attention"),
            notes="Structured attention-head pruning removing one head per attention module when reconstruction is safe.",
        ),
    ]


def combined_strategy_templates() -> list[dict[str, Any]]:
    return [
        {
            "name": "combined_quant_then_prune_selected",
            "family": "quant_then_prune",
            "selection_role": "quant_then_prune",
            "notes": "Exploratory path using the best GPU quant candidate followed by the selected unstructured prune family.",
        },
        {
            "name": "combined_unstructured_prune_then_quant",
            "family": "prune_then_quant",
            "selection_role": "unstructured_prune_then_quant",
            "notes": "Primary combined CPU path using the best unstructured prune candidate then the best CPU quant candidate.",
        },
        {
            "name": "combined_structured_prune_then_quant",
            "family": "prune_then_quant",
            "selection_role": "structured_prune_then_quant",
            "notes": "Primary combined GPU path using the best structured prune candidate then the best GPU quant candidate.",
        },
    ]

