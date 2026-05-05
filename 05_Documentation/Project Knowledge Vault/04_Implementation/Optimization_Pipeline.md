# Optimization Pipeline

## Current framework

The FLAN-T5 optimizer is now a registry-driven framework centered on one saved data contract and one benchmark contract.

Every optimization run starts from:

- the retained baseline checkpoint at `02_Models/PyTorch/flan_t5_zillow_final1`
- the canonical pair cache at `04_Experiments/benchmark/flan_t5_baseline/qa_pairs.jsonl`
- the canonical split manifest at `04_Experiments/benchmark/flan_t5_baseline/split_manifest.json`
- the shared selection policy embedded into each `strategy_manifest.json`

Artifacts are written under:

- `04_Experiments/optimized/flan_t5/<strategy_name>/`

Each strategy directory contains:

- `strategy_manifest.json`
- `benchmark_request.json`
- saved artifact files
- `benchmark/` results when evaluated

## Curated v1 matrix

### Baseline

- `baseline_reference`

### Quant only

- `fp16_cast_gpu`
- `bf16_cast_gpu`
- `dynamic_int8_cpu_ptq`

### Prune only

- `prune_unstructured_global_l1_10`
- `prune_unstructured_global_l1_20`
- `prune_unstructured_attention_l1_10`
- `prune_unstructured_mlp_l1_10`
- `prune_structured_mlp_intermediate_10`
- `prune_structured_attention_heads_1`

### Combined

- selected best quant then selected prune
- selected best unstructured prune then selected quant
- selected best structured prune then selected quant

### QAT

- `qat_phase2_scaffold`

This is scaffolded only. It records the training-data contract and trigger policy for a later QAT experiment, but does not train in phase 1.

## Theory-backed defaults

- GPU-first casts use `FP16` first and `BF16` only when CUDA support is available.
- `INT8` PTQ is retained as the conservative CPU transformer baseline.
- `INT16` is intentionally not part of the main matrix because it is rarely the best practical transformer deployment target.
- Unstructured pruning is kept as a sparsity experiment.
- Structured MLP pruning is treated as the main structured efficiency candidate because feed-forward blocks usually dominate parameter count.
- Structured attention pruning is included as a secondary architecture-aware path.

## Commands

List the registry:

```powershell
python -m src.optimization.optimize_flan_t5 list-strategies
```

Create one strategy artifact and benchmark it:

```powershell
python -m src.optimization.optimize_flan_t5 run-strategy `
  --name dynamic_int8_cpu_ptq `
  --benchmark-max-eval-samples 64
```

Build the full curated v1 artifact set:

```powershell
python -m src.optimization.optimize_flan_t5 run-curated-v1 --skip-benchmark
```

Write the QAT fallback scaffold:

```powershell
python -m src.optimization.optimize_flan_t5 scaffold-qat `
  --trigger-strategy dynamic_int8_cpu_ptq
```

## Selection policy

Combined strategies are not hard-coded from manual judgment. The framework selects “best” candidates using saved benchmark metrics plus the default quality gate:

- minimum token-F1 ratio: `0.5` of baseline
- maximum eval-loss ratio: `1.75` of baseline

Among strategies that pass the gate, the current tie-break priority is:

1. smaller artifact size
2. lower mean latency
3. higher token F1

If no candidate passes the gate, the framework falls back to the best-efficiency candidate inside the configured family order.

## Benchmark integration

The optimizer does not use ad hoc test data. Each strategy manifest stores:

- split manifest path
- pair cache path
- split seed
- train and eval counts

The benchmark runner can evaluate either:

- the baseline checkpoint directly
- or any saved strategy manifest

This keeps baseline, quantized, pruned, combined, and future QAT results comparable.
