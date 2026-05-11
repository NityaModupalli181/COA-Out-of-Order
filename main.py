"""
main.py — Entry point for CECS 530 Speculative Decoding project.
Delegates to run_all.py inside speculative_decoding/.

Usage:
    python main.py              # Full benchmark + Mamba comparison
    python main.py --quick      # Smoke test only (~5 min)
    python main.py --bench      # Benchmark + plots, skip Mamba (~25 min)
    python main.py --mamba      # Mamba comparison only (~10 min)
    python main.py --test       # Run correctness and edge-case tests
"""

import sys
import os
import subprocess


def check_environment():
    """Verify GPU and dependencies before running."""
    print("=" * 60)
    print("CECS 530 — Speculative Decoding & Multi-Token Pipelines")
    print("California State University, Long Beach — Team 10")
    print("=" * 60)

    try:
        import torch
        print(f"PyTorch version : {torch.__version__}")
        print(f"CUDA available  : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU             : {gpu}")
            print(f"VRAM            : {vram:.1f} GB")
            if vram < 3.8:
                print("WARNING: Less than 3.8 GB VRAM. GPT-2 XL requires ~3.0 GB.")
                print("         Use TARGET_MODEL = 'gpt2-large' in models.py as fallback.")
        else:
            print("WARNING: No CUDA GPU found. Benchmarks will be very slow on CPU.")
    except ImportError:
        print("ERROR: PyTorch not installed.")
        print("Install: pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121")
        sys.exit(1)

    try:
        import transformers
        print(f"Transformers    : {transformers.__version__}")
    except ImportError:
        print("ERROR: transformers not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    print()


def run_tests():
    """Run the test suite."""
    print("\nRunning test suite...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    sys.exit(result.returncode)


def main():
    check_environment()

    if "--test" in sys.argv:
        run_tests()
        return

    # Forward all other args to run_all.py
    run_all_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "speculative_decoding",
        "run_all.py"
    )

    if not os.path.exists(run_all_path):
        print(f"ERROR: run_all.py not found at {run_all_path}")
        print("Make sure the speculative_decoding/ directory is present.")
        sys.exit(1)

    args = [sys.executable, run_all_path] + sys.argv[1:]
    result = subprocess.run(args, cwd=os.path.dirname(os.path.abspath(__file__)))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
