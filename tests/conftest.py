"""
conftest.py — pytest shared fixtures for CECS 530 speculative decoding tests.
"""
import sys
import os
import pytest
import torch

# Add speculative_decoding module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'speculative_decoding'))


@pytest.fixture(scope="session")
def device():
    """Return best available device (CUDA preferred)."""
    if torch.cuda.is_available():
        d = torch.device("cuda")
        print(f"\n[conftest] Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"[conftest] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        d = torch.device("cpu")
        print("\n[conftest] WARNING: CUDA not available. Using CPU. Results may differ.")
    return d


@pytest.fixture(scope="session")
def vocab_size_gpt2():
    return 50257


@pytest.fixture(scope="session")
def vocab_size_mamba():
    return 50280


@pytest.fixture(scope="session")
def factual_prompt():
    return "The first digits of pi are"


@pytest.fixture(scope="session")
def expected_pi_prefix():
    return "3.14159"


@pytest.fixture(scope="session")
def paper_results():
    """
    Ground-truth results from the paper (Table 9, 10, 12, 13).
    Used to validate that measured results are consistent with reported values.
    """
    return {
        "beta": {
            "P1_n50": 0.066, "P1_n100": 0.064, "P1_n200": 0.062, "P1_mean": 0.064,
            "P3_n50": 0.102, "P3_n100": 0.099, "P3_n200": 0.097, "P3_mean": 0.099,
        },
        "throughput": {
            "target_only_n100": 19.9,
            "draft_only_n100":  85.1,
            "base_spec_n100":    1.4,
            "spec_kv_n100":      1.5,
            "spec_adaptk_n100":  1.5,
            "full_decoder_n100": 1.6,
            "pipeline_n100":     1.0,
        },
        "alpha_vs_k": {
            2: 0.77, 3: 0.69, 4: 0.33, 5: 0.76, 6: 0.65, 8: 0.79
        },
        "alpha_by_prompt": {
            "factual": 0.66,
            "creative": 0.58,
            "structured": 0.72,
        },
        "mamba": {
            "alpha_k4": 0.285,
            "beta_mean": 0.099,
            "beta_constant": True,
        },
        "theoretical_speedup_k4_alpha065_beta064": 2.01,
    }
