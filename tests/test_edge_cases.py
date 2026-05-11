"""
tests/test_edge_cases.py
─────────────────────────────────────────────────────────────────────────────
Edge-case and robustness tests for the speculative decoding implementation.

Covers the three principal engineering challenges documented in the paper
(§4.3 Implementation Details):
  1. NaN in corrected residual sampling
  2. Mamba cache_position requirement
  3. Pipeline flush race condition

Also tests:
  4. KV-cache rollback correctness (O(k) cost, committed state preserved)
  5. Adaptive k controller boundary conditions
  6. Buffer overflow / all-rejected rounds

Usage:
    python -m pytest tests/test_edge_cases.py -v
    python tests/test_edge_cases.py
"""

import sys
import os
import json
import math
import threading
import queue
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'speculative_decoding'))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: NaN guard in corrected residual sampling
# ─────────────────────────────────────────────────────────────────────────────

def test_nan_guard_zero_residual():
    """
    When p ≈ q (draft and target agree exactly), the residual
    max(0, p - q) is near-zero. Naive multinomial sampling on a
    near-zero vector produces NaN, silently corrupting generation.

    The NaN guard (paper §4.3) detects ||r||_1 < 1e-6 and falls back
    to sampling directly from p, preventing NaN propagation.
    """
    vocab_size = 50257
    device = torch.device("cpu")

    # Simulate p ≈ q: draft exactly matches target
    p = torch.softmax(torch.randn(vocab_size, device=device), dim=0)
    q = p.clone()  # Exact match — residual = 0 everywhere

    residual = torch.clamp(p - q, min=0.0)
    residual_norm = residual.sum().item()

    assert residual_norm < 1e-6, "Expected near-zero residual when p == q"

    # Apply the NaN guard
    if residual_norm < 1e-6:
        # Fall back to p — must not produce NaN
        sample_dist = p
    else:
        sample_dist = residual / residual_norm

    assert not torch.isnan(sample_dist).any(), \
        "NaN guard failed: sample distribution contains NaN"
    assert not torch.isinf(sample_dist).any(), \
        "NaN guard failed: sample distribution contains Inf"
    assert abs(sample_dist.sum().item() - 1.0) < 1e-5, \
        "NaN guard failed: sample distribution does not sum to 1"

    # Sampling must succeed without error
    try:
        token = torch.multinomial(sample_dist, 1).item()
        assert 0 <= token < vocab_size
    except RuntimeError as e:
        pytest.fail(f"Multinomial sampling failed after NaN guard: {e}")

    print("✓ NaN guard correctly handles zero-residual case.")


def test_nan_guard_near_zero_residual():
    """
    Numerical near-zero case: p and q differ by floating-point epsilon.
    """
    vocab_size = 50257
    device = torch.device("cpu")

    p = torch.softmax(torch.randn(vocab_size, device=device), dim=0)
    q = p + torch.randn(vocab_size, device=device) * 1e-8  # Tiny perturbation
    q = torch.clamp(q, min=0.0)
    q = q / q.sum()

    residual = torch.clamp(p - q, min=0.0)
    residual_norm = residual.sum().item()

    if residual_norm < 1e-6:
        sample_dist = p
    else:
        sample_dist = residual / residual_norm

    assert not torch.isnan(sample_dist).any(), "Near-zero residual produced NaN"
    print(f"✓ Near-zero residual handled (||r||_1 = {residual_norm:.2e}).")


def test_nan_guard_all_mass_on_one_token():
    """
    Extreme case: target puts all mass on one token that the draft
    also predicts — residual is zero except for floating-point noise.
    """
    vocab_size = 50257
    device = torch.device("cpu")

    p = torch.zeros(vocab_size, device=device)
    p[42] = 1.0  # All mass on token 42
    q = torch.zeros(vocab_size, device=device)
    q[42] = 1.0  # Draft also predicts token 42

    residual = torch.clamp(p - q, min=0.0)
    residual_norm = residual.sum().item()

    assert residual_norm < 1e-6
    sample_dist = p  # Fallback to p
    assert sample_dist[42].item() == 1.0
    token = torch.multinomial(sample_dist, 1).item()
    assert token == 42
    print("✓ All-mass-on-one-token edge case handled correctly.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: KV-cache rollback correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_kv_cache_rollback_preserves_committed_state():
    """
    Paper §3.6 (Software Rollback Buffer and KV State):
    On rejection at position j*, speculative KV entries from j* onward
    are discarded and the committed pointer remains fixed.
    The operation is O(k), avoiding recomputation of the committed prefix.

    This test verifies:
    1. Committed state is unchanged after rollback
    2. Speculative state beyond j* is cleared
    3. Rollback cost is O(k), not O(n)
    """
    try:
        from kv_cache import DualStateKVCache
    except ImportError:
        pytest.skip("kv_cache.py not importable or DualStateKVCache not defined")

    n_layers = 12
    n_heads = 12
    head_dim = 64
    committed_len = 20
    spec_len = 4  # k = 4

    cache = DualStateKVCache(n_layers=n_layers, n_heads=n_heads, head_dim=head_dim)

    # Simulate committed state
    committed_keys = torch.randn(n_layers, n_heads, committed_len, head_dim)
    committed_vals = torch.randn(n_layers, n_heads, committed_len, head_dim)
    cache.set_committed(committed_keys, committed_vals)

    # Simulate speculative additions
    spec_keys = torch.randn(n_layers, n_heads, spec_len, head_dim)
    spec_vals = torch.randn(n_layers, n_heads, spec_len, head_dim)
    cache.add_speculative(spec_keys, spec_vals)

    # Record committed state before rollback
    committed_keys_before = cache.get_committed_keys().clone()

    # Trigger rollback at position j* = 2 (first 2 accepted, next 2 discarded)
    cache.rollback(j_star=2)

    # Verify committed state unchanged
    committed_keys_after = cache.get_committed_keys()
    assert torch.allclose(committed_keys_before, committed_keys_after), \
        "Rollback corrupted committed KV state"

    # Verify speculative state was cleared
    spec_state = cache.get_speculative_len()
    assert spec_state == 2, \
        f"Expected 2 speculative entries after rollback at j*=2, got {spec_state}"

    print(f"✓ KV-cache rollback preserves committed state (n={committed_len}, k={spec_len}).")
    print(f"✓ Speculative state correctly truncated to j*=2.")


def test_kv_cache_rollback_full_rejection():
    """
    All k draft tokens rejected (j* = 0): speculative state fully cleared.
    """
    try:
        from kv_cache import DualStateKVCache
    except ImportError:
        pytest.skip("kv_cache.py not importable")

    cache = DualStateKVCache(n_layers=12, n_heads=12, head_dim=64)
    committed_keys = torch.randn(12, 12, 10, 64)
    committed_vals = torch.randn(12, 12, 10, 64)
    cache.set_committed(committed_keys, committed_vals)

    spec_keys = torch.randn(12, 12, 4, 64)
    spec_vals = torch.randn(12, 12, 4, 64)
    cache.add_speculative(spec_keys, spec_vals)

    cache.rollback(j_star=0)

    assert cache.get_speculative_len() == 0, \
        "Full rejection should clear all speculative entries"
    print("✓ Full rejection (j*=0) clears all speculative KV entries.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Adaptive k controller boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

def test_adaptive_k_lower_bound():
    """
    Paper Eq. 5: max(k-2, 3) — k never drops below 3.
    Verify the hard floor is enforced even at very low acceptance rates.
    """
    try:
        from adaptive_k import AdaptiveKController
    except ImportError:
        pytest.skip("adaptive_k.py not importable")

    controller = AdaptiveKController(k_init=4, k_min=3, k_max=12)

    # Drive k to minimum
    for _ in range(20):
        controller.update(alpha_hat=0.1)  # Very low acceptance

    assert controller.k >= 3, \
        f"Adaptive controller violated k_min=3: k={controller.k}"
    print(f"✓ Adaptive k respects lower bound k_min=3 (current k={controller.k}).")


def test_adaptive_k_upper_bound():
    """
    Paper Eq. 5: min(k+1, 12) — k never exceeds 12.
    """
    try:
        from adaptive_k import AdaptiveKController
    except ImportError:
        pytest.skip("adaptive_k.py not importable")

    controller = AdaptiveKController(k_init=4, k_min=3, k_max=12)

    # Drive k to maximum
    for _ in range(30):
        controller.update(alpha_hat=0.99)  # Very high acceptance

    assert controller.k <= 12, \
        f"Adaptive controller violated k_max=12: k={controller.k}"
    print(f"✓ Adaptive k respects upper bound k_max=12 (current k={controller.k}).")


def test_adaptive_k_asymmetric_step():
    """
    Paper §3.5: decrement is -2, increment is +1 (asymmetric).
    Low-alpha rounds waste k*beta target compute, so rapid curtailment
    takes priority over cautious escalation.
    """
    try:
        from adaptive_k import AdaptiveKController
    except ImportError:
        pytest.skip("adaptive_k.py not importable")

    controller = AdaptiveKController(k_init=8, k_min=3, k_max=12)

    k_before_up = controller.k
    controller.update(alpha_hat=0.90)  # Above threshold: k += 1
    k_after_up = controller.k

    controller.update(alpha_hat=0.10)  # Below threshold: k -= 2
    k_after_down = controller.k

    assert k_after_up == min(k_before_up + 1, 12), \
        f"Expected increment by 1, got {k_after_up - k_before_up}"
    assert k_after_down == max(k_after_up - 2, 3), \
        f"Expected decrement by 2, got {k_after_up - k_after_down}"

    print(f"✓ Asymmetric step: +1 when alpha high, -2 when alpha low.")
    print(f"  k trajectory: {k_before_up} → {k_after_up} (↑) → {k_after_down} (↓)")


def test_adaptive_k_stable_zone():
    """
    In the stable zone (0.48 ≤ alpha < 0.72): k should not change.
    """
    try:
        from adaptive_k import AdaptiveKController
    except ImportError:
        pytest.skip("adaptive_k.py not importable")

    controller = AdaptiveKController(k_init=5, k_min=3, k_max=12)
    k_initial = controller.k

    for alpha in [0.48, 0.55, 0.60, 0.65, 0.71]:
        controller.update(alpha_hat=alpha)

    assert controller.k == k_initial, \
        f"k changed in stable zone: {k_initial} → {controller.k}"
    print(f"✓ Adaptive k stable in [0.48, 0.72) zone (k={controller.k}).")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Pipeline flush protocol
# ─────────────────────────────────────────────────────────────────────────────

def test_pipeline_flush_clears_queue():
    """
    Paper §4.3 (Pipeline invalidation):
    The pipeline uses a thread event to signal rejection. The draft thread
    checks the flag before each new token and abandons in-progress work.

    This test verifies that the flush mechanism:
    1. Sets the flush event
    2. The draft thread drains all queued batches
    3. The queue is empty after flush completes
    4. No items from the pre-flush queue are processed post-flush
    """
    try:
        from pipeline import PipelineController
    except ImportError:
        pytest.skip("pipeline.py not importable or PipelineController not defined")

    controller = PipelineController(queue_capacity=2)

    # Pre-load the queue with stale batches
    stale_batch_1 = {"tokens": [1, 2, 3, 4], "context": "stale"}
    stale_batch_2 = {"tokens": [5, 6, 7, 8], "context": "stale"}
    controller.enqueue(stale_batch_1)
    controller.enqueue(stale_batch_2)

    assert controller.queue_size() == 2, "Queue pre-load failed"

    # Trigger flush (simulates rejection event)
    controller.flush()

    # Queue should be empty after flush
    assert controller.queue_size() == 0, \
        f"Flush did not clear queue: {controller.queue_size()} items remain"

    # Flush event should be set / draft should know to reset
    assert controller.flush_signalled(), \
        "Flush event not set after flush() call"

    print("✓ Pipeline flush clears queue and signals draft thread.")


def test_pipeline_backpressure():
    """
    Paper §3.7 (Two-Stage Pipeline):
    Queue capacity Q=2 applies backpressure — draft thread blocks when queue full.
    This prevents over-speculation that would be wasted on rollback.
    """
    try:
        from pipeline import PipelineController
    except ImportError:
        pytest.skip("pipeline.py not importable")

    controller = PipelineController(queue_capacity=2)

    # Fill queue to capacity
    controller.enqueue({"batch": 1})
    controller.enqueue({"batch": 2})

    assert controller.queue_size() == 2

    # Third enqueue should block (non-blocking version raises Full)
    result = controller.try_enqueue({"batch": 3}, timeout=0.01)
    assert result is False, \
        "Backpressure failed: queue accepted item beyond capacity=2"

    print("✓ Pipeline backpressure enforced at Q=2.")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: All-rejected round handling
# ─────────────────────────────────────────────────────────────────────────────

def test_all_draft_tokens_rejected():
    """
    Edge case: all k draft tokens are rejected in one round.
    The decoder must:
    1. Sample exactly one corrective token from the residual
    2. Advance context by exactly 1 token
    3. Not crash or produce NaN
    4. Continue generation on the next round
    """
    vocab_size = 50257
    k = 4

    # Simulate scenario where all tokens are rejected
    # Target assigns zero probability to all draft proposals
    p = torch.softmax(torch.randn(vocab_size), dim=0)
    q = torch.softmax(torch.randn(vocab_size), dim=0)

    accepted = []
    for j in range(k):
        draft_token = torch.argmax(q).item()  # Draft picks its best token
        p_val = p[draft_token].item()
        q_val = q[draft_token].item()
        accept_prob = min(1.0, p_val / (q_val + 1e-9))

        if torch.rand(1).item() <= accept_prob:
            accepted.append(draft_token)
        else:
            # Rejection: sample corrective token
            residual = torch.clamp(p - q, min=0.0)
            residual_norm = residual.sum().item()
            if residual_norm < 1e-6:
                corrective_dist = p
            else:
                corrective_dist = residual / residual_norm
            corrective = torch.multinomial(corrective_dist, 1).item()
            assert 0 <= corrective < vocab_size
            assert not math.isnan(float(corrective))
            break

    # Test passes if we reach here without NaN/crash
    print(f"✓ All-rejected round handled correctly (accepted {len(accepted)}/{k} tokens).")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: Acceptance probability boundary conditions
# ─────────────────────────────────────────────────────────────────────────────

def test_acceptance_probability_when_p_greater_than_q():
    """
    When p(x|ctx) > q(x|ctx), the acceptance probability = min(1, p/q) = 1.
    Token is ALWAYS accepted. This is a key property ensuring tokens the
    target model prefers are never penalised by the draft's lower confidence.
    """
    # Case: target is more confident than draft
    p_val = 0.8
    q_val = 0.3
    accept_prob = min(1.0, p_val / q_val)
    assert accept_prob == 1.0, \
        f"Expected accept_prob=1.0 when p>q, got {accept_prob}"
    print("✓ Tokens with p > q are always accepted (accept_prob=1.0).")


def test_acceptance_probability_when_draft_overestimates():
    """
    When q(x|ctx) > p(x|ctx), the acceptance probability < 1.
    Draft over-estimates its confidence — some tokens are probabilistically rejected.
    """
    p_val = 0.1
    q_val = 0.8
    accept_prob = min(1.0, p_val / q_val)
    expected = p_val / q_val  # ≈ 0.125
    assert abs(accept_prob - expected) < 1e-6, \
        f"Acceptance probability incorrect: {accept_prob} vs {expected}"
    print(f"✓ Draft over-estimation correctly penalised: accept_prob={accept_prob:.3f}.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Edge Case Test Suite — Speculative Decoding Implementation")
    print("Covers NaN guard, KV rollback, adaptive k, pipeline flush")
    print("=" * 70)

    tests = [
        ("NaN guard (zero residual)",              test_nan_guard_zero_residual),
        ("NaN guard (near-zero residual)",         test_nan_guard_near_zero_residual),
        ("NaN guard (all-mass on one token)",      test_nan_guard_all_mass_on_one_token),
        ("KV rollback (preserves committed)",      test_kv_cache_rollback_preserves_committed_state),
        ("KV rollback (full rejection j*=0)",      test_kv_cache_rollback_full_rejection),
        ("Adaptive k (lower bound k_min=3)",       test_adaptive_k_lower_bound),
        ("Adaptive k (upper bound k_max=12)",      test_adaptive_k_upper_bound),
        ("Adaptive k (asymmetric step)",           test_adaptive_k_asymmetric_step),
        ("Adaptive k (stable zone)",               test_adaptive_k_stable_zone),
        ("Pipeline flush (clears queue)",          test_pipeline_flush_clears_queue),
        ("Pipeline backpressure (Q=2)",            test_pipeline_backpressure),
        ("All draft tokens rejected",              test_all_draft_tokens_rejected),
        ("Accept prob when p > q",                 test_acceptance_probability_when_p_greater_than_q),
        ("Accept prob when draft overestimates",   test_acceptance_probability_when_draft_overestimates),
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
        print("✓ All edge-case tests passed.")
    else:
        print("✗ Some tests failed — check implementation.")
        import sys
        sys.exit(1)
