from __future__ import annotations

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.optimization.flan_t5_transforms import apply_pipeline_step_for_loading, quantize_dynamic_linear_layers


def utc_now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def read_strategy_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_strategy_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_state_dict(model: torch.nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def normalize_path(path: Path | str) -> str:
    return str(Path(path)).replace("\\", "/")


def load_model_and_tokenizer_from_manifest(
    manifest_path: Path,
    device: str,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    manifest = read_strategy_manifest(manifest_path)
    loader = manifest["loader"]
    tokenizer = AutoTokenizer.from_pretrained(loader["tokenizer_source_path"])

    kind = loader["kind"]
    if kind == "transformers_checkpoint":
        model = AutoModelForSeq2SeqLM.from_pretrained(loader["artifact_model_path"])
    elif kind == "dynamic_int8_state_dict":
        model = AutoModelForSeq2SeqLM.from_pretrained(loader["reconstruction_base_model_path"])
        model = quantize_dynamic_linear_layers(model)
        state_dict = torch.load(loader["artifact_state_dict_path"], map_location="cpu")
        model.load_state_dict(state_dict)
    elif kind == "pipeline_state_dict":
        model = AutoModelForSeq2SeqLM.from_pretrained(loader["reconstruction_base_model_path"])
        for step in loader.get("pipeline", []):
            model = apply_pipeline_step_for_loading(model, step)
        state_dict = torch.load(loader["artifact_state_dict_path"], map_location="cpu")
        model.load_state_dict(state_dict)
    else:
        raise ValueError(f"Unsupported loader kind: {kind}")

    model.to(device)
    model.eval()
    return model, tokenizer, manifest


def build_baseline_manifest(
    *,
    artifact_dir: Path,
    model_path: Path,
    split_manifest_path: Path,
    source_model_summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "created_at_utc": utc_now_iso(),
        "status": "ready",
        "strategy_name": "baseline_reference",
        "strategy_family": "baseline",
        "execution_order": 0,
        "runtime_target": "baseline",
        "source_model": deepcopy(source_model_summary),
        "dataset_contract": {
            "split_manifest_path": normalize_path(split_manifest_path),
            "pair_cache_path": normalize_path(split_manifest_path.parent / "qa_pairs.jsonl"),
            "seed": read_strategy_manifest(split_manifest_path)["split_seed"],
        },
        "optimization": {
            "quantization": None,
            "pruning": None,
        },
        "loader": {
            "kind": "transformers_checkpoint",
            "artifact_model_path": normalize_path(model_path),
            "tokenizer_source_path": normalize_path(model_path),
        },
        "artifact_dir": normalize_path(artifact_dir),
    }

