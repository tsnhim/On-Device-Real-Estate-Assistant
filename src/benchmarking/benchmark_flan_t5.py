from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.benchmarking.flan_t5_data import TRAINING_PREFIX, load_pairs_jsonl, load_split_manifest
from src.benchmarking.flan_t5_metrics import exact_match, mean_or_zero, percentile, rouge_l_f1, token_f1
from src.optimization.flan_t5_artifacts import load_model_and_tokenizer_from_manifest


@dataclass
class BenchmarkConfig:
    model_path: str = "models/flan_t5_zillow_final1"
    strategy_manifest: str = ""
    split_manifest: str = "benchmarks/data/flan_t5_baseline/split_manifest.json"
    output_dir: str = "benchmarks/runs/flan_t5_baseline/results"
    batch_size: int = 1
    max_eval_samples: int = 0
    max_source_length: int = 256
    max_target_length: int = 200
    max_label_length: int = 256
    num_beams: int = 4
    repetition_penalty: float = 1.3
    no_repeat_ngram_size: int = 3
    device: str = "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the retained FLAN-T5 checkpoint on a saved eval split.")
    parser.add_argument("--model-path", default="models/flan_t5_zillow_final1")
    parser.add_argument("--strategy-manifest", default="")
    parser.add_argument("--split-manifest", default="benchmarks/data/flan_t5_baseline/split_manifest.json")
    parser.add_argument("--output-dir", default="benchmarks/runs/flan_t5_baseline/results")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=200)
    parser.add_argument("--max-label-length", type=int, default=256)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    return parser.parse_args()


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return name


def model_disk_size_bytes(model_path: Path) -> int:
    return sum(path.stat().st_size for path in model_path.rglob("*") if path.is_file())


def build_eval_examples(split_manifest_path: Path) -> list[dict[str, Any]]:
    manifest = load_split_manifest(split_manifest_path)
    pair_cache = load_pairs_jsonl(Path(manifest["source_pairs_path"]))
    pair_by_id = {pair.example_id: pair for pair in pair_cache}
    eval_examples = []
    for example_id in manifest["eval_ids"]:
        pair = pair_by_id[example_id]
        eval_examples.append(
            {
                "example_id": pair.example_id,
                "input_text": pair.input_text,
                "target_text": pair.target_text,
                "topic": pair.topic,
            }
        )
    return eval_examples


def chunked(items: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def infer_model_dtype(model: torch.nn.Module, strategy_manifest: dict[str, Any] | None) -> str:
    first_param = next(model.parameters(), None)
    if first_param is not None:
        return str(first_param.dtype)
    first_buffer = next(model.buffers(), None)
    if first_buffer is not None:
        return str(first_buffer.dtype)
    if strategy_manifest:
        quant_dtype = strategy_manifest.get("quantization_dtype")
        if quant_dtype:
            return quant_dtype
        quant_block = strategy_manifest.get("optimization", {}).get("quantization")
        if quant_block and quant_block.get("dtype"):
            return str(quant_block["dtype"])
    return "unknown"


def _config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        model_path=args.model_path,
        strategy_manifest=args.strategy_manifest,
        split_manifest=args.split_manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_eval_samples=args.max_eval_samples,
        max_source_length=args.max_source_length,
        max_target_length=args.max_target_length,
        max_label_length=args.max_label_length,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        device=args.device,
    )


def run_benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_examples = build_eval_examples(Path(config.split_manifest))
    if config.max_eval_samples:
        eval_examples = eval_examples[: config.max_eval_samples]

    process = psutil.Process()
    device = resolve_device(config.device)

    pre_load_rss = process.memory_info().rss
    load_start = time.perf_counter()
    loaded_from_strategy = False
    strategy_manifest_path = config.strategy_manifest.strip()
    if strategy_manifest_path:
        model, tokenizer, strategy_manifest = load_model_and_tokenizer_from_manifest(
            Path(strategy_manifest_path),
            device,
        )
        loaded_from_strategy = True
    else:
        tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(config.model_path)
        model.to(device)
        model.eval()
        strategy_manifest = None
    load_end = time.perf_counter()
    post_load_rss = process.memory_info().rss

    param_count = sum(param.numel() for param in model.parameters())
    if loaded_from_strategy:
        loader = strategy_manifest["loader"]
        if loader["kind"] == "transformers_checkpoint":
            model_size_bytes = model_disk_size_bytes(Path(loader["artifact_model_path"]))
        else:
            model_size_bytes = Path(loader["artifact_state_dict_path"]).stat().st_size
    else:
        model_size_bytes = model_disk_size_bytes(Path(config.model_path))

    latency_ms: list[float] = []
    total_generated_tokens = 0
    total_elapsed = 0.0
    total_loss = 0.0
    total_loss_examples = 0
    predictions: list[dict[str, Any]] = []

    batches = chunked(eval_examples, config.batch_size)
    with torch.inference_mode():
        for batch_index, batch in enumerate(batches):
            prompts = [TRAINING_PREFIX + item["input_text"] for item in batch]
            encoded = tokenizer(
                prompts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=config.max_source_length,
            ).to(device)
            label_batch = tokenizer(
                [item["target_text"] for item in batch],
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=config.max_label_length,
            )
            label_ids = label_batch["input_ids"]
            label_ids[label_ids == tokenizer.pad_token_id] = -100
            label_ids = label_ids.to(device)

            loss_outputs = model(**encoded, labels=label_ids)
            total_loss += float(loss_outputs.loss.item()) * len(batch)
            total_loss_examples += len(batch)

            start = time.perf_counter()
            generated = model.generate(
                **encoded,
                max_new_tokens=config.max_target_length,
                num_beams=config.num_beams,
                repetition_penalty=config.repetition_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
                do_sample=False,
            )
            elapsed = time.perf_counter() - start

            batch_latency_ms = (elapsed * 1000.0) / max(len(batch), 1)
            latency_ms.extend([batch_latency_ms] * len(batch))
            total_elapsed += elapsed

            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for item, prediction, token_ids in zip(batch, decoded, generated):
                generated_tokens = int(token_ids.shape[-1])
                total_generated_tokens += generated_tokens
                predictions.append(
                    {
                        "example_id": item["example_id"],
                        "topic": item["topic"],
                        "prompt": item["input_text"],
                        "reference": item["target_text"],
                        "prediction": prediction,
                        "exact_match": exact_match(prediction, item["target_text"]),
                        "token_f1": token_f1(prediction, item["target_text"]),
                        "rouge_l_f1": rouge_l_f1(prediction, item["target_text"]),
                        "generated_tokens": generated_tokens,
                        "batch_index": batch_index,
                    }
                )

    metrics = {
        "model": {
            "model_path": str(Path(config.model_path)).replace("\\", "/"),
            "strategy_manifest": strategy_manifest_path or None,
            "device": device,
            "param_count": int(param_count),
            "disk_size_bytes": int(model_size_bytes),
            "disk_size_mb": round(model_size_bytes / (1024 * 1024), 2),
            "loader_kind": strategy_manifest["loader"]["kind"] if strategy_manifest else "transformers_checkpoint",
            "runtime_target": strategy_manifest.get("runtime_target") if strategy_manifest else "baseline",
            "torch_dtype": infer_model_dtype(model, strategy_manifest),
        },
        "split": {
            "split_manifest": str(Path(config.split_manifest)).replace("\\", "/"),
            "eval_count": len(eval_examples),
        },
        "quality": {
            "eval_loss": round(total_loss / total_loss_examples, 6) if total_loss_examples else 0.0,
            "exact_match": round(mean_or_zero(item["exact_match"] for item in predictions), 6),
            "token_f1": round(mean_or_zero(item["token_f1"] for item in predictions), 6),
            "rouge_l_f1": round(mean_or_zero(item["rouge_l_f1"] for item in predictions), 6),
            "avg_generated_tokens": round(mean_or_zero(item["generated_tokens"] for item in predictions), 3),
            "avg_reference_chars": round(mean_or_zero(len(item["reference"]) for item in predictions), 3),
            "avg_prediction_chars": round(mean_or_zero(len(item["prediction"]) for item in predictions), 3),
        },
        "efficiency": {
            "load_time_s": round(load_end - load_start, 6),
            "rss_before_load_mb": round(pre_load_rss / (1024 * 1024), 2),
            "rss_after_load_mb": round(post_load_rss / (1024 * 1024), 2),
            "rss_delta_load_mb": round((post_load_rss - pre_load_rss) / (1024 * 1024), 2),
            "total_eval_time_s": round(total_elapsed, 6),
            "examples_per_second": round(len(eval_examples) / total_elapsed, 6) if total_elapsed else 0.0,
            "generated_tokens_per_second": round(total_generated_tokens / total_elapsed, 6) if total_elapsed else 0.0,
            "latency_ms_mean": round(statistics.fmean(latency_ms), 6) if latency_ms else 0.0,
            "latency_ms_p50": round(percentile(latency_ms, 0.5), 6),
            "latency_ms_p95": round(percentile(latency_ms, 0.95), 6),
            "latency_ms_first_batch_per_example": round(latency_ms[0], 6) if latency_ms else 0.0,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "generation": {
            "max_source_length": config.max_source_length,
            "max_target_length": config.max_target_length,
            "max_label_length": config.max_label_length,
            "num_beams": config.num_beams,
            "repetition_penalty": config.repetition_penalty,
            "no_repeat_ngram_size": config.no_repeat_ngram_size,
            "batch_size": config.batch_size,
        },
    }

    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "predictions.jsonl"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with predictions_path.open("w", encoding="utf-8") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")

    return {
        "metrics": metrics,
        "metrics_path": str(metrics_path).replace("\\", "/"),
        "predictions_path": str(predictions_path).replace("\\", "/"),
    }


def main() -> None:
    config = _config_from_args(parse_args())
    result = run_benchmark(config)
    print(f"Saved metrics to {result['metrics_path']}")
    print(f"Saved predictions to {result['predictions_path']}")
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
