# Benchmarking

## Goal

Keep one reproducible evaluation path for every FLAN-T5 artifact:

- baseline full precision
- GPU cast variants such as `FP16` or `BF16`
- CPU `INT8` PTQ
- prune-only artifacts
- prune-plus-quant artifacts
- future QAT checkpoints

## Saved data contract

The repo now treats the benchmark dataset as a fixed contract, not a fresh split every run.

The canonical files are:

- `04_Experiments/benchmark/flan_t5_baseline/qa_pairs.jsonl`
- `04_Experiments/benchmark/flan_t5_baseline/split_manifest.json`

The split manifest preserves:

- train IDs
- eval IDs
- split seed
- split test size
- source pair cache path

Every optimization strategy manifest references this same contract so later comparisons stay fair.

## Recovered training/eval logic

The historical notebook behavior was reconstructed as:

1. load `zillow/real_estate_v1`
2. extract consecutive `user -> assistant` QA pairs
3. clean text
4. prefix inputs with `answer question: `
5. apply a `0.1` eval split

The original notebook did not save the historical split seed, so the exact legacy membership is not recoverable. The repo therefore standardizes on the saved canonical split with seed `42`.

## Metrics

### Quality

- eval loss
- exact match
- token F1
- ROUGE-L F1

### Efficiency

- artifact disk size
- parameter count
- load time
- RSS memory before and after load
- mean latency
- P50 latency
- P95 latency
- examples per second
- generated tokens per second

### Runtime metadata

- device used
- loader kind
- runtime target
- torch dtype
- torch version
- CUDA availability

## Baseline usage

Rebuild the canonical split if needed:

```powershell
python -m src.benchmarking.build_flan_t5_split --source hf
```

Run the baseline checkpoint directly:

```powershell
python -m src.benchmarking.benchmark_flan_t5 `
  --model-path 02_Models/PyTorch/flan_t5_zillow_final1 `
  --split-manifest 04_Experiments/benchmark/flan_t5_baseline/split_manifest.json `
  --output-dir 04_Experiments/benchmark/flan_t5_baseline/results_full `
  --device auto
```

## Strategy-manifest usage

Run any optimized artifact through its saved manifest:

```powershell
python -m src.benchmarking.benchmark_flan_t5 `
  --strategy-manifest 04_Experiments/optimized/flan_t5/dynamic_int8_cpu_ptq/strategy_manifest.json `
  --output-dir 04_Experiments/optimized/flan_t5/dynamic_int8_cpu_ptq/benchmark `
  --device cpu
```

In normal use, the optimizer already writes the benchmark request and can call the benchmark runner for you:

```powershell
python -m src.optimization.optimize_flan_t5 run-strategy `
  --name dynamic_int8_cpu_ptq `
  --benchmark-max-eval-samples 64
```

## Notes

- Use `cuda`, `cpu`, or `auto` as the benchmark device.
- GPU-first strategies may still be saved even if the current environment cannot benchmark them yet.
- Unsupported strategies such as unavailable `BF16` are recorded in their manifests instead of failing silently.

## Visualization

Generate a comparison dashboard and SVG charts from the saved benchmark results:

```powershell
python -m src.benchmarking.plot_flan_t5_benchmarks
```

Aggregate all saved model runs, per-example results, metrics, and strategy metadata into one flat CSV:

```powershell
python -m src.benchmarking.aggregate_flan_t5_metrics
```

The default output directory is:

- `04_Experiments/benchmark/plots/flan_t5`

The generated assets include:

- `benchmark_summary.csv`
- `all_model_metrics.csv`
- `all_methods_quality_vs_size.svg`
- `all_methods_quality_vs_speed.svg`
- `quant_only_overview.svg`
- `quant_only_pareto_frontiers.svg`
- `quant_only_scorecard.svg`
- `quant_only_tradeoff_board.svg`
- `prune_only_tradeoff_board.svg`
- `combined_strategy_story.svg`
- `all_models_overview.svg`
- `full_run_token_f1_ranking.svg`
- `quant_only_token_f1_ranking.svg`
- `prune_only_token_f1_ranking.svg`
- `combined_token_f1_ranking.svg`
- `top10_scorecard_heatmap.svg`
- `index.html`

The dashboard is meant to answer slightly different questions:

- which quant-only model is the best balance of quality, latency, and artifact size
- which quant-only models sit on the Pareto frontier instead of being clearly dominated
- how quant-only candidates compare side by side across quality and efficiency metrics
- quant-only quality versus size, parameter count, and latency
- prune-only quality versus size, parameter count, and latency
- why selected parent strategies were continued into combined methods
- one global quality-efficiency overview across all evaluated models
- ranking by `Token F1` for a simple winner list
- heatmap for a compact multi-metric overview
