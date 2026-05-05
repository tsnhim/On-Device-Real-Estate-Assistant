import os
import json
import time
import statistics
import gc
import torch
torch.backends.quantized.engine = 'qnnpack'
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config

BASE_CONFIG = os.path.expanduser("~/OnDeviceDeepLearning/flan_t5_zillow_final1")
MODELS_DIR  = os.path.expanduser("~/models")
RECORDS     = os.path.join(MODELS_DIR, "zillow_records.json")
OUTPUT_DIR  = os.path.join(MODELS_DIR, "my_benchmark_results")
EVAL_COUNT  = 10
MAX_SRC     = 256
MAX_TGT     = 128
NUM_BEAMS   = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load eval data ──
with open(RECORDS) as f:
    all_records = json.load(f)
import random
random.seed(42)
eval_data = random.sample(all_records, min(EVAL_COUNT, len(all_records)))
print(f"Eval examples: {len(eval_data)}")

# ── Token F1 ──
def token_f1(pred, ref):
    p = set(pred.lower().split())
    r = set(ref.lower().split())
    if not p or not r:
        return 0.0
    common    = p & r
    precision = len(common) / len(p)
    recall    = len(common) / len(r)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

# ── ROUGE-L ──
def rouge_l(pred, ref):
    p = pred.lower().split()
    r = ref.lower().split()
    if not p or not r:
        return 0.0
    m, n = len(p), len(r)
    dp = [[0]*(n+1) for _ in range(2)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if p[i-1] == r[j-1]:
                dp[i%2][j] = dp[(i-1)%2][j-1] + 1
            else:
                dp[i%2][j] = max(dp[(i-1)%2][j], dp[i%2][j-1])
    lcs = dp[m%2][n]
    prec = lcs / m
    rec  = lcs / n
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

# ── RAM usage ──
def get_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if "VmRSS" in line:
                    return int(line.split()[1]) / 1024
    except:
        return 0.0
    return 0.0

# ── Benchmark one model ──
def benchmark(model_name, model_path, tokenizer):
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")

    # Load model weights
    pt_file = os.path.join(model_path, "artifact_state_dict.pt")
    if not os.path.exists(pt_file):
        print(f"No artifact_state_dict.pt found — skipping")
        return None

    disk_mb = os.path.getsize(pt_file) / 1e6
    print(f"Disk size: {disk_mb:.1f} MB")

    # Load model
    rss_before = get_rss_mb()
    t0 = time.time()
    try:
        BASE_CONFIG = os.path.expanduser(
            "~/OnDeviceDeepLearning/flan_t5_zillow_final1"
        )
        config = T5Config.from_pretrained(BASE_CONFIG, local_files_only=True)

        # Check if quantized model by inspecting keys
        state = torch.load(pt_file, map_location="cpu")
        keys  = list(state.keys()) if isinstance(state, dict) else []
        is_quantized = any("packed_params" in k or "scale" in k for k in keys)

        if is_quantized:
            print("Detected quantized model — loading with qnnpack...")
            from torch.ao.quantization import quantize_dynamic
            model = T5ForConditionalGeneration(config)
            model = quantize_dynamic(
                model,
                {torch.nn.Linear},
                dtype=torch.qint8,
            )
            model.load_state_dict(state, strict=False)
        else:
            print("Detected FLOAT32 model — loading normally...")
            model = T5ForConditionalGeneration(config)
            model.load_state_dict(state, strict=False)

        model.eval()

    except Exception as e:
        print(f"Error loading: {e}")
        return None

    load_time  = time.time() - t0
    rss_after  = get_rss_mb()
    param_count = sum(p.numel() for p in model.parameters())

    print(f"Load time:  {load_time:.2f}s")
    print(f"Params:     {param_count:,}")
    print(f"RAM delta:  {rss_after - rss_before:.0f} MB")

    # Run inference
    latencies   = []
    tf1_scores  = []
    rl_scores   = []
    gen_tokens  = []

    print(f"Running {len(eval_data)} examples...")
    for i, rec in enumerate(eval_data):
        q   = rec.get("question", "")
        ref = rec.get("answer", "")

        prompt = f"You are a real estate expert.\nQuestion: {q}\nAnswer:"
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=MAX_SRC,
            truncation=True,
        )

        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_TGT,
                num_beams=NUM_BEAMS,
                no_repeat_ngram_size=3,
                repetition_penalty=2.0,
            )
        lat = (time.time() - t0) * 1000
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        n_tok = len(out[0])

        latencies.append(lat)
        tf1_scores.append(token_f1(pred, ref))
        rl_scores.append(rouge_l(pred, ref))
        gen_tokens.append(n_tok)

        if (i+1) % 5 == 0:
            print(f"  [{i+1}/{len(eval_data)}] "
                  f"lat={lat:.0f}ms "
                  f"tf1={tf1_scores[-1]:.3f}")

    # Summary
    sorted_lat  = sorted(latencies)
    mean_lat    = statistics.mean(latencies)
    mean_tok    = statistics.mean(gen_tokens)

    result = {
        "model_name": model_name,
        "quality": {
            "token_f1":    round(statistics.mean(tf1_scores), 4),
            "rouge_l_f1":  round(statistics.mean(rl_scores),  4),
            "exact_match": 0.0,
            "eval_loss":   None,
        },
        "efficiency": {
            "disk_size_mb":               round(disk_mb, 2),
            "param_count":                param_count,
            "load_time_s":                round(load_time, 3),
            "rss_before_load_mb":         round(rss_before, 1),
            "rss_after_load_mb":          round(rss_after, 1),
            "rss_delta_load_mb":          round(rss_after - rss_before, 1),
            "latency_ms_mean":            round(mean_lat, 1),
            "latency_ms_p50":             round(sorted_lat[len(sorted_lat)//2], 1),
            "latency_ms_p95":             round(sorted_lat[int(len(sorted_lat)*0.95)], 1),
            "examples_per_second":        round(1000/mean_lat, 4),
            "generated_tokens_per_second":round(mean_tok/(mean_lat/1000), 2),
        },
        "generation": {
            "eval_count":           len(eval_data),
            "avg_generated_tokens": round(mean_tok, 2),
            "avg_reference_chars":  round(statistics.mean(len(r.get("answer","")) for r in eval_data), 1),
            "avg_prediction_chars": 0,
            "num_beams":            NUM_BEAMS,
            "batch_size":           1,
            "max_source_length":    MAX_SRC,
            "max_target_length":    MAX_TGT,
        },
        "runtime_metadata": {
            "device":         "Android ARM64 Termux",
            "loader_kind":    "artifact_state_dict",
            "runtime_target": "cpu",
            "torch_dtype":    "float32",
            "torch_version":  torch.__version__,
            "cuda_available": False,
        }
    }

    print(f"\nResults:")
    print(f"  token_f1:   {result['quality']['token_f1']:.4f}")
    print(f"  rouge_l:    {result['quality']['rouge_l_f1']:.4f}")
    print(f"  lat_mean:   {result['efficiency']['latency_ms_mean']:.0f}ms")
    print(f"  lat_p95:    {result['efficiency']['latency_ms_p95']:.0f}ms")
    print(f"  ex/s:       {result['efficiency']['examples_per_second']:.4f}")
    print(f"  tok/s:      {result['efficiency']['generated_tokens_per_second']:.1f}")

    # Save individual result
    out_path = os.path.join(OUTPUT_DIR, f"{model_name}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")

    # Free memory
    del model
    gc.collect()

    return result


# ── Main ──
print("Loading tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(
    os.path.join(MODELS_DIR, "tokenizer"),
    legacy=True,
)
print("Tokenizer loaded!")

# Find all models with artifact_state_dict.pt
model_list = []
for name in sorted(os.listdir(MODELS_DIR)):
    pt = os.path.join(MODELS_DIR, name, "artifact_state_dict.pt")
    if os.path.exists(pt):
        model_list.append((name, os.path.join(MODELS_DIR, name)))

print(f"\nFound {len(model_list)} models to benchmark:")
for name, _ in model_list:
    print(f"  - {name}")

# Run all benchmarks
all_results = []
for name, path in model_list:
    # Redirect output to model-specific log file
    log_path = os.path.join(OUTPUT_DIR, f"{name}_log.txt")
    print(f"\nBenchmarking {name}")
    import sys
    original_stdout = sys.stdout

    # Write to both terminal and log file
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_file = open(log_path, "w")
    sys.stdout = Tee(original_stdout, log_file)

    result = benchmark(name, path, tokenizer)

    # Restore stdout
    sys.stdout = original_stdout
    log_file.close()
    print(f"Log saved: {log_path}")

    if result:
        all_results.append(result)

# Save combined
with open(os.path.join(OUTPUT_DIR, "all_benchmarks.json"), "w") as f:
    json.dump(all_results, f, indent=2)

# Print comparison table
print("\n" + "="*75)
print("FINAL COMPARISON TABLE")
print("="*75)
print(f"{'Model':<40} {'tf1':>6} {'rl':>6} {'disk':>7} {'lat':>8} {'ex/s':>7}")
print("-"*75)
for r in all_results:
    q = r["quality"]
    e = r["efficiency"]
    print(
        f"{r['model_name']:<40}"
        f"{q['token_f1']:>6.4f}"
        f"{q['rouge_l_f1']:>6.4f}"
        f"{e['disk_size_mb']:>6.1f}MB"
        f"{e['latency_ms_mean']:>8.0f}ms"
        f"{e['examples_per_second']:>7.4f}"
    )
print("="*75)
print(f"\nAll results in: {OUTPUT_DIR}")
