import os
import json
import time
import statistics
import gc
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config

# Set ARM64 quantization engine
torch.backends.quantized.engine = 'qnnpack'

MODELS_DIR  = os.path.expanduser("~/models")
RECORDS     = os.path.join(MODELS_DIR, "zillow_records.json")
OUTPUT_DIR  = os.path.join(MODELS_DIR, "my_benchmark_results")
EVAL_COUNT  = 10
MAX_SRC     = 256
MAX_TGT     = 128
NUM_BEAMS   = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Only structured pruning models ──
STRUCTURED_MODELS = [
    "combined_structured_prune_then_quant__prune_structured_attention_heads_1__bf16_cast_gpu",
    "combined_structured_prune_then_quant__prune_structured_mlp_intermediate_10__fp16_cast_gpu",
    "prune_structured_attention_heads_1",
    "prune_structured_mlp_intermediate_10",
    "qat_recommendation",
]

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
    lcs  = dp[m%2][n]
    prec = lcs / m
    rec  = lcs / n
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

# ── RAM ──
def get_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if "VmRSS" in line:
                    return int(line.split()[1]) / 1024
    except:
        return 0.0
    return 0.0

# ── Load config with correct dimensions ──
def load_config(model_path):
    """Load T5Config with correct dimensions for structured pruned models."""
    base_path = os.path.expanduser(
        "~/OnDeviceDeepLearning/flan_t5_zillow_final1"
    )
    config = T5Config.from_pretrained(base_path, local_files_only=True)

    # Infer actual dimensions from weights
    pt_file = os.path.join(model_path, "artifact_state_dict.pt")
    state   = torch.load(pt_file, map_location="cpu")

    if not isinstance(state, dict):
        return config

    # Check attention head dimension
    for key in state.keys():
        if "SelfAttention.q.weight" in key:
            actual = state[key].shape[0]
            if actual != config.d_kv * config.num_heads:
                new_heads = actual // config.d_kv
                print(f"  Pruned heads: {config.num_heads} → {new_heads}")
                config.num_heads = new_heads
            break

    # Check FFN dimension
    for key in state.keys():
        if "DenseReluDense.wi_0.weight" in key:
            actual = state[key].shape[0]
            if actual != config.d_ff:
                print(f"  Pruned FFN: {config.d_ff} → {actual}")
                config.d_ff = actual
            break
        if "DenseReluDense.wi.weight" in key:
            actual = state[key].shape[0]
            if actual != config.d_ff:
                print(f"  Pruned FFN: {config.d_ff} → {actual}")
                config.d_ff = actual
            break

    return config

# ── Benchmark one model ──
def benchmark(model_name, model_path, tokenizer):
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")

    pt_file = os.path.join(model_path, "artifact_state_dict.pt")
    if not os.path.exists(pt_file):
        print(f"No pt file found — skipping")
        return None

    disk_mb = os.path.getsize(pt_file) / 1e6
    print(f"Disk size: {disk_mb:.1f} MB")

    # Load config with correct dimensions
    rss_before = get_rss_mb()
    t0 = time.time()
    try:
        config = load_config(model_path)
        state  = torch.load(pt_file, map_location="cpu")

        # Check if quantized
        keys = list(state.keys()) if isinstance(state, dict) else []
        is_quantized = any(
            "packed_params" in k or "zero_point" in k for k in keys
        )

        if is_quantized:
            print("Quantized model — using qnnpack...")
            from torch.ao.quantization import quantize_dynamic
            model = T5ForConditionalGeneration(config)
            model = quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            model = T5ForConditionalGeneration(config)

        model.load_state_dict(state, strict=False)
        model.eval()

    except Exception as e:
        print(f"Error loading: {e}")
        return None

    load_time   = time.time() - t0
    rss_after   = get_rss_mb()
    param_count = sum(p.numel() for p in model.parameters())

    print(f"Load time:  {load_time:.2f}s")
    print(f"Params:     {param_count:,}")
    print(f"RAM delta:  {rss_after - rss_before:.0f} MB")

    # Run inference
    latencies  = []
    tf1_scores = []
    rl_scores  = []
    gen_tokens = []

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
        lat  = (time.time() - t0) * 1000
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        n_tok = len(out[0])

        latencies.append(lat)
        tf1_scores.append(token_f1(pred, ref))
        rl_scores.append(rouge_l(pred, ref))
        gen_tokens.append(n_tok)

        if (i+1) % 5 == 0:
            print(f"  [{i+1}/{len(eval_data)}] "
                  f"lat={lat:.0f}ms tf1={tf1_scores[-1]:.3f}")

    # Summary
    sorted_lat = sorted(latencies)
    mean_lat   = statistics.mean(latencies)
    mean_tok   = statistics.mean(gen_tokens)

    result = {
        "model_name": model_name,
        "quality": {
            "token_f1":    round(statistics.mean(tf1_scores), 4),
            "rouge_l_f1":  round(statistics.mean(rl_scores),  4),
            "exact_match": 0.0,
            "eval_loss":   None,
        },
        "efficiency": {
            "disk_size_mb":                round(disk_mb, 2),
            "param_count":                 param_count,
            "load_time_s":                 round(load_time, 3),
            "rss_before_load_mb":          round(rss_before, 1),
            "rss_after_load_mb":           round(rss_after,  1),
            "rss_delta_load_mb":           round(rss_after - rss_before, 1),
            "latency_ms_mean":             round(mean_lat, 1),
            "latency_ms_p50":              round(sorted_lat[len(sorted_lat)//2], 1),
            "latency_ms_p95":              round(sorted_lat[int(len(sorted_lat)*0.95)], 1),
            "examples_per_second":         round(1000/mean_lat, 4),
            "generated_tokens_per_second": round(mean_tok/(mean_lat/1000), 2),
        },
        "generation": {
            "eval_count":           len(eval_data),
            "avg_generated_tokens": round(mean_tok, 2),
            "avg_reference_chars":  round(
                statistics.mean(len(r.get("answer","")) for r in eval_data), 1
            ),
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
    print(f"  token_f1:  {result['quality']['token_f1']:.4f}")
    print(f"  rouge_l:   {result['quality']['rouge_l_f1']:.4f}")
    print(f"  lat_mean:  {result['efficiency']['latency_ms_mean']:.0f}ms")
    print(f"  lat_p95:   {result['efficiency']['latency_ms_p95']:.0f}ms")
    print(f"  ex/s:      {result['efficiency']['examples_per_second']:.4f}")

    # Save
    out_path = os.path.join(OUTPUT_DIR, f"{model_name}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {out_path}")

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

# Find structured models
model_list = []
for name in STRUCTURED_MODELS:
    path = os.path.join(MODELS_DIR, name)
    pt   = os.path.join(path, "artifact_state_dict.pt")
    if os.path.exists(pt):
        model_list.append((name, path))
    else:
        print(f"Skipping {name} — no pt file found")

print(f"\nFound {len(model_list)} structured models:")
for name, _ in model_list:
    print(f"  - {name}")

# Run benchmarks
all_results = []
for name, path in model_list:
    log_path = os.path.join(OUTPUT_DIR, f"{name}_log.txt")

    import sys
    original_stdout = sys.stdout

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

    log_file   = open(log_path, "w")
    sys.stdout = Tee(original_stdout, log_file)

    result = benchmark(name, path, tokenizer)

    sys.stdout = original_stdout
    log_file.close()
    print(f"Log saved: {log_path}")

    if result:
        all_results.append(result)

# Save combined
out = os.path.join(OUTPUT_DIR, "structured_benchmarks.json")
with open(out, "w") as f:
    json.dump(all_results, f, indent=2)

# Print table
print("\n" + "="*70)
print("STRUCTURED MODELS COMPARISON")
print("="*70)
print(f"{'Model':<45} {'tf1':>6} {'rl':>6} {'lat':>8}")
print("-"*70)
for r in all_results:
    q = r["quality"]
    e = r["efficiency"]
    print(
        f"{r['model_name'][:45]:<45}"
        f"{q['token_f1']:>6.4f}"
        f"{q['rouge_l_f1']:>6.4f}"
        f"{e['latency_ms_mean']:>8.0f}ms"
    )
print("="*70)
print(f"Results saved to: {out}")
