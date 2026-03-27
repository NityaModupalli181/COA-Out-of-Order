"""
benchmark.py
Full benchmark sweep — produces all numbers needed for the paper.

Tests all 5 configurations × 3 sequence lengths × 3 runs each.
Saves results to results.json for use by plots.py.

Configurations:
  1. Target only (baseline)
  2. Draft only  (baseline)
  3. Base speculative decoder
  4. Speculative + KV-cache
  5. Full decoder (KV + Adaptive K)
"""

import torch
import json
import time
from models import load_models, autoregressive_baseline, measure_beta
from speculative_decoder import SpeculativeDecoder
from kv_cache import KVCacheSpeculativeDecoder
from adaptive_k import FullSpeculativeDecoder

# ── Config ────────────────────────────────────────────────────
N_TOKENS_LIST = [50, 100, 200]
K_VALUE       = 4           # fixed k for base + KV variants
N_RUNS        = 3           # runs per config (averaged)
PROMPTS = [
    "The first digits of pi are",
    "The capital of France is",
    "Once upon a time in a land far away",
    "def fibonacci(n):",
]
# Use first prompt for main benchmark, all prompts for α sensitivity
MAIN_PROMPT = PROMPTS[0]


def run_n_times(fn, n):
    """Run fn() n times, return averaged result dict."""
    results = [fn() for _ in range(n)]
    keys    = results[0].keys()
    avg     = {}
    for k in keys:
        if isinstance(results[0][k], (int, float)):
            avg[k] = sum(r[k] for r in results) / n
        else:
            avg[k] = results[0][k]   # take first for non-numeric (text)
    return avg


def benchmark_target_only(model, tokenizer, n_tokens, n_runs):
    def fn():
        t0 = time.perf_counter()
        inputs = tokenizer(MAIN_PROMPT, return_tensors="pt").to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=n_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        elapsed = time.perf_counter() - t0
        return {"tps": n_tokens / elapsed, "acceptance_rate": 1.0,
                "rollbacks": 0, "k": 0}
    return run_n_times(fn, n_runs)


def benchmark_spec(decoder_cls, draft_model, target_model, tokenizer,
                   n_tokens, k, n_runs, **kwargs):
    def fn():
        dec = decoder_cls(draft_model, target_model, tokenizer, k=k, **kwargs)
        return dec.generate(MAIN_PROMPT, max_new_tokens=n_tokens)
    return run_n_times(fn, n_runs)


def benchmark_full(draft_model, target_model, tokenizer,
                   n_tokens, n_runs):
    def fn():
        dec = FullSpeculativeDecoder(
            draft_model, target_model, tokenizer, k_init=4
        )
        return dec.generate(MAIN_PROMPT, max_new_tokens=n_tokens)
    return run_n_times(fn, n_runs)


def main():
    print("=" * 60)
    print("  Speculative Decoding Benchmark")
    print("  GTX 1650 Ti | GPT-2 Small (draft) + GPT-2 XL (target)")
    print("=" * 60)

    draft_model, draft_tok, target_model, target_tok = load_models()
    beta = measure_beta(draft_model, target_model, target_tok)

    results = {
        "beta": beta,
        "configs": {},
        "alpha_sensitivity": {},
    }

    config_names = [
        "target_only",
        "draft_only",
        "base_speculative",
        "speculative_kv",
        "full_decoder",
    ]
    for cfg in config_names:
        results["configs"][cfg] = {}

    print(f"\n{'Config':<25} {'n':>5} {'tok/s':>8} {'α':>6} {'speedup':>8}")
    print("-" * 55)

    target_tps = {}   # store baseline tps per n

    for n in N_TOKENS_LIST:

        # ── 1. Target only ───────────────────────────────────
        r = benchmark_target_only(target_model, target_tok, n, N_RUNS)
        target_tps[n] = r["tps"]
        results["configs"]["target_only"][n] = r
        print(f"{'Target only':<25} {n:>5} {r['tps']:>8.1f} {'—':>6} {'1.00x':>8}")

        # ── 2. Draft only ────────────────────────────────────
        r = benchmark_target_only(draft_model, draft_tok, n, N_RUNS)
        results["configs"]["draft_only"][n] = r
        su = r["tps"] / target_tps[n]
        print(f"{'Draft only':<25} {n:>5} {r['tps']:>8.1f} {'—':>6} {su:>7.2f}x")

        # ── 3. Base speculative ──────────────────────────────
        r = benchmark_spec(
            SpeculativeDecoder, draft_model, target_model, target_tok,
            n, K_VALUE, N_RUNS
        )
        results["configs"]["base_speculative"][n] = r
        su = r["tps"] / target_tps[n]
        print(f"{'Base speculative':<25} {n:>5} {r['tps']:>8.1f} "
              f"{r['acceptance_rate']:>6.2f} {su:>7.2f}x")

        # ── 4. Speculative + KV-cache ────────────────────────
        r = benchmark_spec(
            KVCacheSpeculativeDecoder, draft_model, target_model, target_tok,
            n, K_VALUE, N_RUNS
        )
        results["configs"]["speculative_kv"][n] = r
        su = r["tps"] / target_tps[n]
        print(f"{'Speculative + KV-Cache':<25} {n:>5} {r['tps']:>8.1f} "
              f"{r['acceptance_rate']:>6.2f} {su:>7.2f}x")

        # ── 5. Full decoder (KV + Adaptive K) ────────────────
        r = benchmark_full(draft_model, target_model, target_tok, n, N_RUNS)
        results["configs"]["full_decoder"][n] = r
        su = r["tps"] / target_tps[n]
        print(f"{'Full decoder':<25} {n:>5} {r['tps']:>8.1f} "
              f"{r['acceptance_rate']:>6.2f} {su:>7.2f}x")

        print()

    # ── α sensitivity: vary k, measure α at n=100 ────────────
    print("\n── Acceptance rate sensitivity (n=100, vary k) ──")
    print(f"{'k':>4} {'tok/s':>8} {'α':>6} {'speedup':>8}")
    print("-" * 30)

    baseline_100 = target_tps[100]
    for k in [2, 3, 4, 5, 6, 8]:
        r = benchmark_spec(
            SpeculativeDecoder, draft_model, target_model, target_tok,
            100, k, N_RUNS
        )
        su = r["tps"] / baseline_100
        results["alpha_sensitivity"][k] = r
        print(f"{k:>4} {r['tps']:>8.1f} {r['acceptance_rate']:>6.2f} {su:>7.2f}x")

    # Save results
    with open("results.json", "w") as f:
        # Convert int keys to str for JSON
        serializable = {
            "beta": results["beta"],
            "configs": {
                cfg: {str(n): v for n, v in vals.items()}
                for cfg, vals in results["configs"].items()
            },
            "alpha_sensitivity": {
                str(k): v for k, v in results["alpha_sensitivity"].items()
            },
        }
        json.dump(serializable, f, indent=2)

    print("\nResults saved to results.json")
    print("Run plots.py to generate charts.")


if __name__ == "__main__":
    main()