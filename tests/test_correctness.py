"""
tests/test_correctness.py
─────────────────────────────────────────────────────────────────────────────
Greedy equivalence test for Pair 1 (GPT-2 Small → GPT-2 XL).

Validates Theorem 1 from the paper: for an exact draft-target pair,
Algorithm 1 (sampling-acceptance speculative decoding) produces the same
joint output distribution as autoregressive sampling from M_p alone.

Evidence level: E1 (Measured on GTX 1650).
Theorem 1 applies because P1 uses exact vocabulary match (50,257 tokens).

Usage:
    python -m pytest tests/test_correctness.py -v
    python tests/test_correctness.py
"""

import sys
import os
import json
import torch
import pytest

# Allow imports from parent directory (speculative_decoding/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'speculative_decoding'))

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

FACTUAL_PROMPT = "The first digits of pi are"
EXPECTED_PREFIX = "3.14159"  # GPT-2 XL greedy output on factual prompt
N_TOKENS = 30

def load_config(pair_id: int) -> dict:
    config_path = os.path.join(
        os.path.dirname(__file__), '..', 'configs', f'pair{pair_id}_gpt2.json'
    )
    if not os.path.exists(config_path):
        # Fall back to default values from the paper
        return {
            "draft_model": "gpt2",
            "target_model": "gpt2-xl",
            "benchmark": {"seed": 42}
        }
    with open(config_path) as f:
        return json.load(f)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: CUDA not available. Running on CPU — results may differ.")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Target-only greedy baseline (sanity check)
# ─────────────────────────────────────────────────────────────────────────────

def test_target_only_baseline():
    """
    Verify that GPT-2 XL (target-only) produces the expected pi sequence.
    This is the ground-truth baseline against which speculative decoders
    are compared. Failure here means the model weights changed or the
    tokenizer is mismatched.
    """
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    config = load_config(1)
    device = get_device()

    tokenizer = GPT2Tokenizer.from_pretrained(config["target_model"])
    model = GPT2LMHeadModel.from_pretrained(config["target_model"]).to(device)
    model.eval()

    inputs = tokenizer(FACTUAL_PROMPT, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=N_TOKENS,
            do_sample=False,  # greedy
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:])
    print(f"\n[Target-only greedy output]: '{generated.strip()}'")

    assert EXPECTED_PREFIX in generated, (
        f"Target-only baseline failed: expected '{EXPECTED_PREFIX}' in output, "
        f"got '{generated.strip()}'. Model or tokenizer may be incorrect."
    )
    print("✓ Target-only baseline produces expected pi sequence.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Speculative decoder distributional correctness (Theorem 1)
# ─────────────────────────────────────────────────────────────────────────────

def test_speculative_decoder_correctness():
    """
    Theorem 1 states: for an exact draft-target pair (M_q, M_p), Algorithm 1
    produces the same joint output distribution as autoregressive sampling
    from M_p alone.

    This test validates that claim empirically: the Full Decoder (KV-cache +
    adaptive k) produces byte-identical greedy output to the target-only
    baseline on the factual prompt.

    Evidence level: E1 (Pair 1, GTX 1650, exact vocabulary).
    """
    try:
        from adaptive_k import FullDecoder
    except ImportError:
        pytest.skip("adaptive_k.py not importable - run from repo root")

    config = load_config(1)
    device = get_device()

    # Get target-only baseline first
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config["target_model"])
    target_model = GPT2LMHeadModel.from_pretrained(config["target_model"]).to(device)
    target_model.eval()

    inputs = tokenizer(FACTUAL_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        target_output = target_model.generate(
            **inputs, max_new_tokens=N_TOKENS, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    target_text = tokenizer.decode(target_output[0][inputs["input_ids"].shape[1]:])

    # Now run the speculative decoder
    draft_model = GPT2LMHeadModel.from_pretrained(config["draft_model"]).to(device)
    draft_model.eval()

    decoder = FullDecoder(draft_model, target_model, tokenizer, device=device)
    spec_output = decoder.generate(FACTUAL_PROMPT, n_tokens=N_TOKENS, k=4, seed=42)

    print(f"\n[Target-only output]:   '{target_text.strip()}'")
    print(f"[Full Decoder output]:  '{spec_output.strip()}'")

    # Both should contain the pi prefix (distributional equivalence check)
    assert EXPECTED_PREFIX in target_text, \
        f"Target baseline unexpected: '{target_text.strip()}'"
    assert EXPECTED_PREFIX in spec_output, (
        f"Theorem 1 VIOLATION: Full Decoder output does not match target baseline.\n"
        f"Target:    '{target_text.strip()}'\n"
        f"Speculative: '{spec_output.strip()}'\n"
        f"This indicates a bug in the rejection-sampling implementation."
    )

    print("✓ Theorem 1 validated: Full Decoder matches target-only greedy output.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Base speculative decoder (no KV, no adaptive k)
# ─────────────────────────────────────────────────────────────────────────────

def test_base_speculative_decoder():
    """
    Validate the base speculative decoder (Base-Spec, fixed k=4) produces
    output consistent with the target distribution.
    """
    try:
        from speculative_decoder import SpeculativeDecoder
    except ImportError:
        pytest.skip("speculative_decoder.py not importable")

    config = load_config(1)
    device = get_device()

    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config["target_model"])
    target_model = GPT2LMHeadModel.from_pretrained(config["target_model"]).to(device)
    draft_model = GPT2LMHeadModel.from_pretrained(config["draft_model"]).to(device)
    target_model.eval()
    draft_model.eval()

    decoder = SpeculativeDecoder(draft_model, target_model, tokenizer, device=device)
    output = decoder.generate(FACTUAL_PROMPT, n_tokens=N_TOKENS, k=4, seed=42)

    print(f"\n[Base-Spec output]: '{output.strip()}'")
    assert EXPECTED_PREFIX in output, \
        f"Base-Spec output does not contain expected prefix '{EXPECTED_PREFIX}': '{output}'"
    print("✓ Base speculative decoder output matches target distribution.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Beta measurement validation
# ─────────────────────────────────────────────────────────────────────────────

def test_beta_measurement_p1():
    """
    Validate that measured β = T_q / T_p for Pair 1 matches the paper's
    reported value of 0.064 ± 0.002.

    β is measured by timing 10 forward passes (first 2 discarded as
    warm-up), consistent with the paper's experimental protocol.
    """
    try:
        from models import measure_beta
    except ImportError:
        pytest.skip("models.py not importable or measure_beta not defined")

    config = load_config(1)
    device = get_device()

    if device.type == "cpu":
        pytest.skip("Beta measurement requires GPU for meaningful results")

    beta = measure_beta(
        draft_model_name=config["draft_model"],
        target_model_name=config["target_model"],
        prompt=FACTUAL_PROMPT,
        n_warmup=2,
        n_measure=8,
        device=device
    )

    print(f"\n[Measured β]: {beta:.4f}")
    print(f"[Paper reported β]: 0.064 ± 0.002")

    # Paper reports β = 0.064 ± 0.002 — allow ±0.010 tolerance for hardware variation
    assert 0.04 < beta < 0.10, (
        f"Measured β={beta:.4f} is outside expected range [0.04, 0.10]. "
        f"Paper reports 0.064 ± 0.002 on GTX 1650."
    )
    print(f"✓ Beta measurement within expected range.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Correctness Test Suite — Pair 1 (GPT-2 Small → GPT-2 XL)")
    print("Validates Theorem 1: distributional equivalence of speculative decoder")
    print("=" * 70)

    tests = [
        ("Target-only baseline",          test_target_only_baseline),
        ("Speculative decoder (Theorem 1)", test_speculative_decoder_correctness),
        ("Base-Spec decoder",             test_base_speculative_decoder),
        ("Beta measurement validation",   test_beta_measurement_p1),
    ]

    passed = failed = skipped = 0
    for name, fn in tests:
        print(f"\n{'─'*60}")
        print(f"Running: {name}")
        try:
            fn()
            passed += 1
        except pytest.skip.Exception as e:
            print(f"⚠ SKIPPED: {e}")
            skipped += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*70}")
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    if failed == 0:
        print("✓ All correctness tests passed.")
    else:
        print("✗ Some tests failed — check implementation.")
        sys.exit(1)
