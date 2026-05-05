# On-Device-Real-Estate-Assistant

On-Device-Real-Estate-Assistant is an on-device real-estate assistant prototype. The project keeps a domain FLAN-T5 question-answering model, exports it to ONNX for Android inference, and benchmarks multiple optimization strategies on an Android ARM64 environment.

Team: Phong Cao, Trang Tran, Mai Do  
School: Worcester Polytechnic Institute

The current repository is organized as a runnable project, not as a notebook dump. The final benchmark aggregate is [results/all_benchmarks.json](results/all_benchmarks.json), and the generated report plots are in [benchmarks/visualizations/tradeoff_plots](benchmarks/visualizations/tradeoff_plots).

## Project Goals

- Run a real-estate question-answering model locally on a phone which is limited resources.
- Compare model optimization strategies for on-device deployment.
- Measure both answer quality and device efficiency.
- Package the Android inference path with ONNX Runtime.
- Keep the final benchmark result reproducible and easy to inspect.

## Repository Structure

```text
app/android/                         Android app project
models/
  flan_t5_zillow_final1/              Hugging Face FLAN-T5 model assets
  whisper_model/                      Whisper speech model assets
  export_to_onnx.py                   PyTorch/Hugging Face -> ONNX export script
benchmarks/
  data/flan_t5_baseline/              fixed QA pair cache and eval split
  requirements.txt                    Python benchmark dependencies
  visualizations/                     plot generator and final SVG charts
results/
  all_benchmarks.json                 final Android benchmark aggregate
src/
  benchmarking/                       benchmark runner, split builder, metrics
  optimization/                       pruning/quantization strategy code
```

## System Overview

The full project pipeline is:

```text
User input
  -> typed text
  -> voice input -> phone speech-to-text -> text

Text prompt
  -> FLAN-T5 real-estate question-answering model
  -> optional optimization experiments
       -> quantization: FP16, BF16, INT8
       -> pruning: attention, MLP, global unstructured pruning
       -> combined pruning + quantization

Selected / exported model
  -> models/export_to_onnx.py
  -> ONNX encoder and decoder files
  -> app/android/app/src/main/assets/onnx_model/

Android phone
  -> ONNX Runtime Android
  -> local inference
  -> generated answer displayed in the app
```

The Android app does not run PyTorch or TensorFlow directly. It loads the exported `.onnx` encoder and decoder files through ONNX Runtime Android. Because ONNX Runtime expects numeric tensors instead of raw text, the app also bundles the matching FLAN-T5 tokenizer file, `spiece.model`. A small native C++ SentencePiece bridge loads that file, converts user text into the token IDs expected by the ONNX model, and decodes generated token IDs back into readable text.

## Methodology

The benchmark compares optimization families that are common for on-device transformer deployment:

- Quantization: `fp16`, `bf16`, and `int8`
- Pruning: unstructured attention, MLP, and global pruning
- Combined pipelines: pruning plus quantization

Each model is evaluated against the same fixed benchmark split:

- Pair cache: [benchmarks/data/flan_t5_baseline/qa_pairs.jsonl](benchmarks/data/flan_t5_baseline/qa_pairs.jsonl)
- Split manifest: [benchmarks/data/flan_t5_baseline/split_manifest.json](benchmarks/data/flan_t5_baseline/split_manifest.json)
- Source dataset: `zillow/real_estate_v1`
- Eval split: `10%`
- Split seed: `42`

Measured quality metrics:

- Token F1
- ROUGE-L F1
- Exact match
- Eval loss when available

Measured efficiency metrics:

- Disk size
- Parameter count
- Model load time
- RSS memory before/after load
- Mean, P50, and P95 latency
- Examples per second
- Generated tokens per second

## Results

The final retained benchmark file is:

```text
results/all_benchmarks.json
```

### Accuracy And Retention

![Accuracy and retention](benchmarks/visualizations/tradeoff_plots/01_accuracy_token_f1_and_retention.svg)

### Quality vs Latency

![Quality vs latency](benchmarks/visualizations/tradeoff_plots/03_quality_vs_latency_bubble.svg)

### Size Reduction Tradeoff

![Quality retention vs size reduction](benchmarks/visualizations/tradeoff_plots/04_quality_retention_vs_size_reduction.svg)

### Latency

![Latency mean and P95](benchmarks/visualizations/tradeoff_plots/05_latency_mean_and_p95.svg)

### Disk Size

![Disk size comparison](benchmarks/visualizations/tradeoff_plots/06_disk_size_comparison.svg)

### Throughput

![Throughput comparison](benchmarks/visualizations/tradeoff_plots/07_throughput_comparison.svg)

### Pruning Comparison

![Structured vs unstructured pruning](benchmarks/visualizations/tradeoff_plots/08_structured_vs_unstructured_pruning.svg)

### Balanced Ranking

![Balanced score ranking](benchmarks/visualizations/tradeoff_plots/09_balanced_score_ranking.svg)

## Result Analysis

The fastest model in the final Android benchmark is:

```text
FP16 + Unstructured MLP L1 10 Prune
```

It reaches `120.6 ms` mean latency, `180.8 ms` P95 latency, and `8.29 examples/sec`. Its Token F1 is `0.1851`, which retains about `81.0%` of the baseline Token F1 while cutting disk size by about `50.5%`.

The highest-quality model is the full baseline:

```text
Baseline
```

It has the best Token F1 at `0.2285`, but it is large and slow on Android ARM64 Termux: `311.11 MB`, `5476.5 ms` mean latency, and only `0.1826 examples/sec`.

The smallest models are:

```text
Dynamic INT8 PTQ
Unstructured MLP L1 10 Prune + Dynamic INT8 PTQ
```

Both are `126.58 MB`, a `59.3%` disk-size reduction from baseline. `Dynamic INT8 PTQ` preserves `0.2193` Token F1, or about `96.0%` of baseline quality, but it is much slower than the baseline in this run at `15236.9 ms` mean latency.

The best simple balanced choice is:

```text
FP16
```

It keeps `0.2208` Token F1, or `96.6%` of baseline, reduces disk size by `50.5%`, and improves latency from `5476.5 ms` to `162.9 ms`.

The best practical interpretation is:

- `FP16` and `BF16` provide the strongest quality/latency/size tradeoff in the retained results.
- `FP16 + Unstructured MLP L1 10 Prune` is the fastest model, but it gives up more quality than plain `FP16`.
- `Dynamic INT8 PTQ` gives the best compression while preserving quality, but the Android ARM64 CPU latency is too high for interactive use.
- Structured pruning reduces parameter count and can improve latency, but the quality/size tradeoff is weaker than plain `FP16`.
- The current best deployment candidate is `FP16` if latency and quality are both important.

## Benchmark Summary

| Model | Family | Token F1 | Retention | Size MB | Size Red. | Mean ms | P95 ms | Examples/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `FP16 + Unstructured MLP L1 10 Prune` | combined | 0.1851 | 81.0% | 154.01 | 50.5% | 120.6 | 180.8 | 8.2905 |
| `Unstructured MLP L1 10 Prune` | prune | 0.2146 | 93.9% | 307.93 | 1.0% | 135.7 | 190.7 | 7.3673 |
| `Structured MLP Intermediate 10 Prune + FP16` | combined | 0.1776 | 77.7% | 149.00 | 52.1% | 162.9 | 273.0 | 6.1389 |
| `FP16` | quant | 0.2208 | 96.6% | 154.01 | 50.5% | 162.9 | 271.5 | 6.1392 |
| `Unstructured Attention L1 10 Prune` | prune | 0.2074 | 90.8% | 307.93 | 1.0% | 164.1 | 235.4 | 6.0929 |
| `BF16 + Unstructured MLP L1 10 Prune` | combined | 0.1868 | 81.8% | 154.01 | 50.5% | 169.1 | 287.4 | 5.9147 |
| `Structured Attention Head 1 Prune` | prune | 0.2052 | 89.8% | 295.35 | 5.1% | 192.5 | 394.1 | 5.1953 |
| `Structured MLP Intermediate 10 Prune` | prune | 0.2017 | 88.3% | 297.90 | 4.2% | 192.7 | 295.4 | 5.1901 |
| `BF16` | quant | 0.2221 | 97.2% | 154.01 | 50.5% | 241.1 | 394.2 | 4.1471 |
| `Unstructured Global L1 10 Prune` | prune | 0.1976 | 86.5% | 307.93 | 1.0% | 335.8 | 1739.0 | 2.9779 |
| `Structured Attention Head 1 Prune + BF16` | combined | 0.1809 | 79.2% | 147.72 | 52.5% | 459.6 | 920.1 | 2.1756 |
| `Unstructured Global L1 20 Prune` | prune | 0.1664 | 72.8% | 307.93 | 1.0% | 1237.9 | 6331.2 | 0.8078 |
| `Baseline` | baseline | 0.2285 | 100.0% | 311.11 | 0.0% | 5476.5 | 9020.4 | 0.1826 |
| `Dynamic INT8 PTQ` | quant | 0.2193 | 96.0% | 126.58 | 59.3% | 15236.9 | 19735.6 | 0.0656 |
| `Unstructured MLP L1 10 Prune + Dynamic INT8 PTQ` | combined | 0.1884 | 82.5% | 126.58 | 59.3% | 15430.8 | 19503.6 | 0.0648 |

## How To Run

### 1. Install Python Benchmark Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r benchmarks/requirements.txt
```

### 2. Regenerate Benchmark Plots

```bash
python benchmarks/visualizations/generate_tradeoff_plots.py
```

Output:

```text
benchmarks/visualizations/tradeoff_plots/
```

Open the generated HTML report:

```text
benchmarks/visualizations/tradeoff_plots/index.html
```

### 3. Run The FLAN-T5 Benchmark Harness

```bash
python -m src.benchmarking.benchmark_flan_t5 \
  --model-path models/flan_t5_zillow_final1 \
  --split-manifest benchmarks/data/flan_t5_baseline/split_manifest.json \
  --output-dir benchmarks/runs/flan_t5_baseline/results \
  --device auto
```

### 4. Rebuild The Benchmark Split

This downloads/rebuilds the benchmark pair cache from `zillow/real_estate_v1`.

```bash
python -m src.benchmarking.build_flan_t5_split \
  --output-dir benchmarks/data/flan_t5_baseline
```

### 5. Export FLAN-T5 To ONNX For Android

```bash
python models/export_to_onnx.py
```

Output:

```text
app/android/app/src/main/assets/onnx_model/
```

### 6. Build The Android App

```bash
cd app/android
./gradlew assembleDebug
```

The Android app loads:

```text
app/android/app/src/main/assets/onnx_model/encoder_model.onnx
app/android/app/src/main/assets/onnx_model/decoder_model.onnx
app/android/app/src/main/assets/onnx_model/decoder_with_past_model.onnx
```

## Android Runtime

The Android app uses:

```kotlin
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.17.3")
```

At runtime, `MainActivity.kt` copies the `onnx_model` asset folder into app storage, creates ONNX Runtime sessions for the encoder and decoder, tokenizes with SentencePiece, and decodes greedily token by token.

## Current Limitations

- The app currently runs FLAN-T5 ONNX inference, but the retained benchmark results include separate Android/Termux-style PyTorch measurements for optimized artifacts.
- Whisper assets are retained, but the current Android app uses Android speech recognition for microphone input rather than the bundled Whisper models.
- The measured INT8 models preserve better Token F1 but are too slow in the retained benchmark.
- The faster FP16/BF16-style variants need quality debugging before they are useful.
- The benchmark quality scores are low overall, so future work should improve prompt parity, decoding settings, and evaluation data quality.

## Recommended Next Steps

- Align Android decoding settings with the Python benchmark settings.
- Benchmark the exact ONNX Android app path, not only Android/Termux model artifacts.
- Investigate why FP16/BF16 variants produce near-zero retained Token F1.
- Add an end-to-end scripted smoke test for export -> Android asset validation.
- Evaluate ONNX Runtime execution providers and decoder-with-past usage for latency.
- Decide whether Whisper should be integrated directly or removed from the active scope.
