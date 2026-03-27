"""
run_all.py
Single entry point to run the complete experiment end-to-end.

Usage:
    python run_all.py           # full benchmark + mamba comparison
    python run_all.py --quick   # smoke test only (no mamba)
    python run_all.py --mamba   # run mamba comparison only
    python run_all.py --bench   # run benchmark + plots only (no mamba)
"""

import sys
import torch


def check_environment():
    print("=" * 55)
    print("  Environment Check")
    print("=" * 55)
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {torch.version.cuda}")
    if torch.cuda.is_available():
        p    = torch.cuda.get_device_properties(0)
        vram = p.total_memory / 1e9
        print(f"  GPU     : {p.name}")
        print(f"  VRAM    : {vram:.1f} GB")
        if vram < 3.5:
            print("\n  WARNING: VRAM < 3.5 GB — GPT-2 XL may not fit.")
            print("  Try reducing to: TARGET_MODEL = 'gpt2-large'")
        else:
            print("  Status  : ✓ Sufficient VRAM for GPT-2 XL")
    else:
        print("  GPU     : NOT FOUND — running on CPU (very slow)")
    print()


def quick_test():
    """Smoke test — verify everything works before full benchmark."""
    print("Running quick smoke test (GPT-2 Small + GPT-2 XL, n=50)...\n")
    from models import load_models, autoregressive_baseline
    from speculative_decoder import SpeculativeDecoder
    from kv_cache import KVCacheSpeculativeDecoder
    from adaptive_k import FullSpeculativeDecoder

    dm, dt, tm, tt = load_models(verbose=False)
    PROMPT = "The first digits of pi are"

    print("[1/4] Target baseline...")
    base_tps, _ = autoregressive_baseline(tm, tt, PROMPT, 50, "GPT-2 XL")

    print("\n[2/4] Base speculative decoder...")
    dec = SpeculativeDecoder(dm, tm, tt, k=4)
    r   = dec.generate(PROMPT, max_new_tokens=50)
    print(f"  {r['tps']:.1f} tok/s | α={r['acceptance_rate']:.2f} | "
          f"speedup={r['tps']/base_tps:.2f}x")

    print("\n[3/4] Speculative + KV-Cache...")
    dec = KVCacheSpeculativeDecoder(dm, tm, tt, k=4)
    r   = dec.generate(PROMPT, max_new_tokens=50)
    print(f"  {r['tps']:.1f} tok/s | α={r['acceptance_rate']:.2f} | "
          f"speedup={r['tps']/base_tps:.2f}x")

    print("\n[4/4] Full decoder (KV + Adaptive K)...")
    dec = FullSpeculativeDecoder(dm, tm, tt, k_init=4)
    r   = dec.generate(PROMPT, max_new_tokens=50)
    print(f"  {r['tps']:.1f} tok/s | α={r['acceptance_rate']:.2f} | "
          f"speedup={r['tps']/base_tps:.2f}x")

    print(f"\n✓ Smoke test passed — all decoders working.")
    print(f"  Run without --quick for full benchmark + Mamba comparison.\n")


def run_benchmark_and_plots():
    """Run full benchmark sweep and generate all charts."""
    import benchmark
    import plots

    print("─" * 55)
    print("  Phase 1: Full Benchmark Sweep")
    print("─" * 55)
    benchmark.main()

    print("\n" + "─" * 55)
    print("  Phase 2: Generating Charts")
    print("─" * 55)
    data = plots.load()
    plots.print_summary_table(data)
    plots.plot_throughput_by_method(data)
    plots.plot_throughput_line(data)
    plots.plot_speedup(data)
    plots.plot_alpha_sensitivity(data)

    print("\nCharts saved:")
    print("  plot_throughput.png")
    print("  plot_throughput_line.png")
    print("  plot_speedup.png")
    print("  plot_alpha_sensitivity.png")


def run_mamba_comparison():
    """Run the Mamba-130m vs GPT-2 Small draft model comparison."""
    print("─" * 55)
    print("  Phase 3: Mamba-130m vs GPT-2 Small Draft Comparison")
    print("─" * 55)
    print("  Research question: Does an SSM draft model outperform")
    print("  a same-size Transformer draft model?")
    print("  (Proxy experiment for Mamba-3 theoretical claim)\n")

    try:
        import mamba_draft
        results = mamba_draft.run_comparison()

        import json
        with open("mamba_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("\n  Mamba results saved to mamba_results.json")

    except ImportError as e:
        print(f"\n  Could not import mamba_draft: {e}")
        print("  Make sure mamba_draft.py is in the same folder.")

    except Exception as e:
        print(f"\n  Mamba comparison failed: {e}")
        print("  This is non-critical — main benchmark results are unaffected.")
        print("  Common causes:")
        print("    - Mamba model download failed (check internet)")
        print("    - VRAM too low after loading GPT-2 XL")
        print("    - Transformers version does not support Mamba")
        print("  Try running standalone:  python mamba_draft.py")


def full_run():
    """Run everything: benchmark + plots + mamba comparison."""
    print("Starting full experiment...\n")

    run_benchmark_and_plots()

    import torch, gc
    torch.cuda.empty_cache()
    gc.collect()
    print()

    run_mamba_comparison()

    print("\n" + "=" * 55)
    print("  ALL DONE")
    print("=" * 55)
    print("  Output files:")
    print("    results.json              benchmark numbers")
    print("    mamba_results.json        mamba vs GPT-2 comparison")
    print("    plot_throughput.png       bar chart: tok/s by method")
    print("    plot_throughput_line.png  line: tok/s vs sequence length")
    print("    plot_speedup.png          speedup x over baseline")
    print("    plot_alpha_sensitivity.png  alpha and speedup vs k")
    print()
    print("  These files go directly into your paper and presentation.")


if __name__ == "__main__":
    check_environment()

    if "--quick" in sys.argv:
        quick_test()

    elif "--mamba" in sys.argv:
        run_mamba_comparison()

    elif "--bench" in sys.argv:
        print("Running benchmark + plots (skipping Mamba comparison)...\n")
        run_benchmark_and_plots()

    else:
        full_run()