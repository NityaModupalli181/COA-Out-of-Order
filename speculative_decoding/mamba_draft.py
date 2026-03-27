"""
mamba_draft.py
Compare two draft model architectures against the same target (GPT-2 XL):
  1. GPT-2 Small  (117M Transformer)  — standard baseline draft
  2. Mamba-130m   (130M SSM)          — state space model draft

Key research question:
  Does an SSM draft model achieve higher acceptance rate or
  better throughput than a same-size Transformer draft model?

This validates the theoretical Mamba-3 claim:
  "SSMs are architecturally ideal as draft models due to
   O(1) decode, no KV-cache, and fixed hidden state."

VRAM usage on GTX 1650 Ti:
  GPT-2 XL    (target) : ~3.0 GB
  Mamba-130m  (draft)  : ~0.26 GB
  Total                : ~3.26 GB — safe on 4 GB card
"""

import torch
import torch.nn.functional as F
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, MambaForCausalLM

# ── Model names ───────────────────────────────────────────────
TARGET_MODEL      = "gpt2-xl"                    # 1.5B Transformer (target)
GPT2_DRAFT_MODEL  = "gpt2"                       # 117M Transformer (draft 1)
MAMBA_DRAFT_MODEL = "state-spaces/mamba-130m-hf" # 130M SSM        (draft 2)


# ── VRAM check ────────────────────────────────────────────────
def vram_status(label=""):
    if not torch.cuda.is_available():
        return
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    used  = torch.cuda.memory_allocated(0) / 1e9
    tag   = f"[{label}] " if label else ""
    print(f"{tag}VRAM used: {used:.2f} / {total:.1f} GB")


# ── Load target model (shared for both experiments) ───────────
def load_target(device="cuda"):
    print(f"Loading target: {TARGET_MODEL}")
    tok   = AutoTokenizer.from_pretrained(TARGET_MODEL)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL, torch_dtype=torch.float16
    ).to(device).eval()
    vram_status("after target")
    return model, tok


# ── Load Transformer draft (GPT-2 Small) ─────────────────────
def load_gpt2_draft(device="cuda"):
    print(f"Loading Transformer draft: {GPT2_DRAFT_MODEL}")
    tok   = AutoTokenizer.from_pretrained(GPT2_DRAFT_MODEL)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        GPT2_DRAFT_MODEL, torch_dtype=torch.float16
    ).to(device).eval()
    vram_status("after GPT-2 draft")
    return model, tok


# ── Load Mamba draft (Mamba-130m) ─────────────────────────────
def load_mamba_draft(device="cuda"):
    print(f"Loading Mamba draft: {MAMBA_DRAFT_MODEL}")
    tok   = AutoTokenizer.from_pretrained(MAMBA_DRAFT_MODEL)
    tok.pad_token = tok.eos_token

    # MambaForCausalLM is supported in transformers >= 4.39
    model = AutoModelForCausalLM.from_pretrained(
        MAMBA_DRAFT_MODEL,
        torch_dtype=torch.float16,
    ).to(device).eval()
    vram_status("after Mamba draft")
    return model, tok


def draft_with_mamba(mamba_model, tokenizer, input_ids, k,
                     target_vocab_size):
    """
    Generate k draft tokens using Mamba's recurrent state.
    O(1) per token — no KV-cache growth.
    """
    draft_ids   = []
    draft_probs = []
    device      = input_ids.device
    seq_len     = input_ids.shape[1]

    with torch.no_grad():

        # ── First pass: process full input, get recurrent state ──
        out         = mamba_model(input_ids, use_cache=True)
        mamba_state = out.cache_params
        logits      = out.logits[:, -1, :]

        # Align vocab size with target
        if logits.shape[-1] < target_vocab_size:
            pad    = torch.full((1, target_vocab_size - logits.shape[-1]),
                                float('-inf'), device=device, dtype=logits.dtype)
            logits = torch.cat([logits, pad], dim=-1)
        elif logits.shape[-1] > target_vocab_size:
            logits = logits[:, :target_vocab_size]

        probs = F.softmax(logits, dim=-1)
        token = torch.argmax(probs, dim=-1)
        p     = probs[0, token.item()].item()
        draft_ids.append(token.item())
        draft_probs.append(max(p, 1e-9))

        # ── Subsequent tokens: one token at a time with state ──
        for step in range(k - 1):
            # cache_position = position of the NEW token in the sequence
            cache_pos = torch.tensor(
                [seq_len + step],
                device=device, dtype=torch.long
            )

            out         = mamba_model(
                token.unsqueeze(0),
                cache_params=mamba_state,
                cache_position=cache_pos,   # ← this is what was missing
                use_cache=True,
            )
            mamba_state = out.cache_params
            logits      = out.logits[:, -1, :]

            # Vocab alignment
            if logits.shape[-1] < target_vocab_size:
                pad    = torch.full((1, target_vocab_size - logits.shape[-1]),
                                    float('-inf'), device=device, dtype=logits.dtype)
                logits = torch.cat([logits, pad], dim=-1)
            elif logits.shape[-1] > target_vocab_size:
                logits = logits[:, :target_vocab_size]

            probs = F.softmax(logits, dim=-1)
            token = torch.argmax(probs, dim=-1)
            p     = probs[0, token.item()].item()
            draft_ids.append(token.item())
            draft_probs.append(max(p, 1e-9))

    return draft_ids, draft_probs

# ── Generic speculative decoding loop ─────────────────────────
def speculative_generate(target_model, target_tok,
                         draft_fn,        # callable: (input_ids, k) → (ids, probs)
                         prompt, k=4, max_new_tokens=100):
    """
    Runs speculative decoding with any draft function.
    draft_fn is pluggable — works for both GPT-2 and Mamba drafters.
    """
    device    = next(target_model.parameters()).device
    input_ids = target_tok(
        prompt, return_tensors="pt"
    ).input_ids.to(device)
    output_ids = input_ids.clone()

    total_drafted  = 0
    total_accepted = 0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    tokens_generated = 0

    while tokens_generated < max_new_tokens:

        # Draft k tokens
        draft_ids, draft_probs = draft_fn(output_ids, k)

        # Verify with target (one parallel pass)
        draft_tensor = torch.tensor(draft_ids, device=device).unsqueeze(0)
        full_seq     = torch.cat([output_ids, draft_tensor], dim=1)

        with torch.no_grad():
            out = target_model(full_seq)

        prefix_len   = output_ids.shape[1]
        accepted_ids = []
        bonus_token  = None

        for j in range(k):
            pos          = prefix_len - 1 + j
            target_probs = F.softmax(out.logits[:, pos, :], dim=-1)
            p_target     = target_probs[0, draft_ids[j]].item()
            p_draft      = draft_probs[j]
            accept_prob  = min(1.0, p_target / p_draft)
            accepted     = torch.rand(1).item() < accept_prob

            total_drafted += 1
            if accepted:
                total_accepted += 1
                accepted_ids.append(draft_ids[j])
            else:
                corrected   = torch.clamp(
                    target_probs[0] - p_draft, min=0.0
                )
                s           = corrected.sum()
                corrected   = corrected / s if s > 1e-8 else target_probs[0]
                bonus_token = torch.multinomial(corrected, 1).item()
                break

        if bonus_token is None:
            last_probs  = F.softmax(
                out.logits[:, prefix_len - 1 + k, :], dim=-1
            )
            bonus_token = torch.argmax(last_probs, dim=-1).item()

        new_ids    = accepted_ids + [bonus_token]
        new_tensor = torch.tensor(new_ids, device=device).unsqueeze(0)
        output_ids = torch.cat([output_ids, new_tensor], dim=1)
        tokens_generated += len(new_ids)

        if bonus_token == target_tok.eos_token_id:
            break

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    alpha = total_accepted / total_drafted if total_drafted > 0 else 0
    text  = target_tok.decode(
        output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
    )

    return {
        "text":            text,
        "tokens":          tokens_generated,
        "time":            elapsed,
        "tps":             tokens_generated / elapsed,
        "acceptance_rate": alpha,
    }


# ── Measure time for one Mamba forward pass ───────────────────
def measure_mamba_beta(mamba_model, target_model, tokenizer,
                       prompt, n_runs=10):
    """Measure β = T_mamba / T_target (should be even lower than GPT-2 Small)"""
    device = next(mamba_model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    def time_model(model, use_mamba=False, n=n_runs):
        times = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(**inputs)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        return sum(times[1:]) / (n - 1)

    t_mamba  = time_model(mamba_model)
    t_target = time_model(target_model)
    beta     = t_mamba / t_target

    print(f"\n  Mamba-130m forward : {t_mamba*1000:.1f} ms")
    print(f"  GPT-2 XL forward   : {t_target*1000:.1f} ms")
    print(f"  β (mamba/target)   : {beta:.3f}")
    return beta


# ── Main comparison ───────────────────────────────────────────
def run_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    print("=" * 60)

    # Load target (shared)
    target_model, target_tok = load_target(device)
    target_vocab = target_model.config.vocab_size

    # Load GPT-2 Small draft
    gpt2_model, gpt2_tok = load_gpt2_draft(device)

    # Load Mamba-130m draft
    mamba_model, mamba_tok = load_mamba_draft(device)

    vram_status("all models loaded")

    # Measure beta for both drafts
    print("\n── Beta measurement ──")
    print("GPT-2 Small vs GPT-2 XL:")
    inputs_gpt2 = gpt2_tok(
        "The first digits of pi are", return_tensors="pt"
    ).to(device)
    t_gpt2 = []
    t_target = []
    for _ in range(10):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            gpt2_model(**inputs_gpt2)
        torch.cuda.synchronize()
        t_gpt2.append(time.perf_counter() - t0)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            target_model(
                target_tok(
                    "The first digits of pi are", return_tensors="pt"
                ).to(device).input_ids[:, :inputs_gpt2.input_ids.shape[1]]
            )
        torch.cuda.synchronize()
        t_target.append(time.perf_counter() - t0)

    beta_gpt2  = (sum(t_gpt2[1:]) / 9) / (sum(t_target[1:]) / 9)
    print(f"  β GPT-2 Small : {beta_gpt2:.3f}")

    beta_mamba = measure_mamba_beta(
        mamba_model, target_model,
        target_tok, "The first digits of pi are"
    )

    # ── Run experiments ───────────────────────────────────────
    PROMPT       = "The first digits of pi are"
    N_TOKENS     = 100
    K            = 4
    N_RUNS       = 3

    print("\n" + "=" * 60)
    print(f"  Comparison: GPT-2 Small vs Mamba-130m as draft model")
    print(f"  Target: GPT-2 XL | k={K} | n={N_TOKENS} tokens")
    print("=" * 60)

    # Baseline: target alone
    print("\n[1/3] Target-only baseline...")
    baseline_times = []
    for _ in range(N_RUNS):
        inputs = target_tok(PROMPT, return_tensors="pt").to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = target_model.generate(
                **inputs, max_new_tokens=N_TOKENS,
                do_sample=False, pad_token_id=target_tok.eos_token_id
            )
        torch.cuda.synchronize()
        baseline_times.append(N_TOKENS / (time.perf_counter() - t0))
    baseline_tps = sum(baseline_times) / N_RUNS
    print(f"  Target alone: {baseline_tps:.1f} tok/s")

    # GPT-2 Small draft
    print("\n[2/3] GPT-2 Small (Transformer) as draft...")

    def gpt2_draft_fn(input_ids, k):
        ids, probs = [], []
        current    = input_ids.clone()
        with torch.no_grad():
            for _ in range(k):
                out   = gpt2_model(current)
                probs_ = F.softmax(out.logits[:, -1, :], dim=-1)
                token  = torch.argmax(probs_, dim=-1)
                ids.append(token.item())
                probs.append(max(probs_[0, token.item()].item(), 1e-9))
                current = torch.cat([current, token.unsqueeze(0)], dim=1)
        return ids, probs

    gpt2_results = []
    for _ in range(N_RUNS):
        r = speculative_generate(
            target_model, target_tok, gpt2_draft_fn,
            PROMPT, k=K, max_new_tokens=N_TOKENS
        )
        gpt2_results.append(r)

    gpt2_tps    = sum(r["tps"] for r in gpt2_results) / N_RUNS
    gpt2_alpha  = sum(r["acceptance_rate"] for r in gpt2_results) / N_RUNS
    gpt2_speedup = gpt2_tps / baseline_tps
    print(f"  GPT-2 Small draft  : {gpt2_tps:.1f} tok/s | "
          f"α={gpt2_alpha:.2f} | {gpt2_speedup:.2f}x speedup")

    # Mamba draft
    print("\n[3/3] Mamba-130m (SSM) as draft...")

    def mamba_draft_fn(input_ids, k):
        return draft_with_mamba(
            mamba_model, mamba_tok, input_ids, k, target_vocab
        )

    mamba_results = []
    for _ in range(N_RUNS):
        r = speculative_generate(
            target_model, target_tok, mamba_draft_fn,
            PROMPT, k=K, max_new_tokens=N_TOKENS
        )
        mamba_results.append(r)

    mamba_tps     = sum(r["tps"] for r in mamba_results) / N_RUNS
    mamba_alpha   = sum(r["acceptance_rate"] for r in mamba_results) / N_RUNS
    mamba_speedup = mamba_tps / baseline_tps
    print(f"  Mamba-130m draft   : {mamba_tps:.1f} tok/s | "
          f"α={mamba_alpha:.2f} | {mamba_speedup:.2f}x speedup")

    # ── Summary table ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Method':<30} {'tok/s':>7} {'α':>6} {'speedup':>8}")
    print(f"  {'─'*55}")
    print(f"  {'Target only (GPT-2 XL)':<30} "
          f"{baseline_tps:>7.1f} {'—':>6} {'1.00x':>8}")
    print(f"  {'GPT-2 Small draft (Transformer)':<30} "
          f"{gpt2_tps:>7.1f} {gpt2_alpha:>6.2f} {gpt2_speedup:>7.2f}x")
    print(f"  {'Mamba-130m draft (SSM)':<30} "
          f"{mamba_tps:>7.1f} {mamba_alpha:>6.2f} {mamba_speedup:>7.2f}x")
    print(f"\n  β GPT-2 Small : {beta_gpt2:.3f}")
    print(f"  β Mamba-130m  : {beta_mamba:.3f}")

    if mamba_tps > gpt2_tps:
        diff = ((mamba_tps - gpt2_tps) / gpt2_tps) * 100
        print(f"\n  ★ Mamba draft is {diff:.1f}% faster than GPT-2 draft")
        print(f"    — validates the SSM-as-drafter hypothesis")
    else:
        diff = ((gpt2_tps - mamba_tps) / mamba_tps) * 100
        print(f"\n  GPT-2 draft is {diff:.1f}% faster than Mamba draft")
        print(f"  — Mamba needs pretrained alignment to show full benefit")
        print(f"  — Mamba-3 (with MIMO + complex states) would improve this")

    print("\n  PAPER NOTE:")
    print("  Mamba-3 (ICLR 2026) has no pretrained weights yet.")
    print("  These results use Mamba-1 (130m) as a proxy.")
    print("  Mamba-3 improves over Mamba-1 with:")
    print("    - MIMO: higher GPU utilization during drafting")
    print("    - Complex states: better state tracking → higher α")
    print("    - Half state size: even smaller memory footprint")

    return {
        "baseline_tps":  baseline_tps,
        "gpt2_tps":      gpt2_tps,
        "gpt2_alpha":    gpt2_alpha,
        "mamba_tps":     mamba_tps,
        "mamba_alpha":   mamba_alpha,
        "beta_gpt2":     beta_gpt2,
        "beta_mamba":    beta_mamba,
    }


if __name__ == "__main__":
    results = run_comparison()