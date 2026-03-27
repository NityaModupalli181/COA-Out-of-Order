"""
models.py
Load draft + target model pair for GTX 1650 Ti (4 GB VRAM).
Draft  : GPT-2 Small (117M) — ~0.3 GB
Target : GPT-2 XL   (1.5B)  — ~3.0 GB
Total  : ~3.3 GB — safe on 4 GB card
"""

import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

print("CUDA available:", torch.cuda.is_available())
print("PyTorch version:", torch.__version__)

if not torch.cuda.is_available():
    raise RuntimeError(
        f"\n\nCUDA not available!\n"
        f"PyTorch: {torch.__version__}\n"
        f"CUDA compiled: {torch.version.cuda}\n"
        f"Try: pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121"
    )

device = "cuda"
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

DRAFT_MODEL  = "gpt2"        # 117M  ~0.3 GB fp16
TARGET_MODEL = "gpt2-xl"     # 1.5B  ~3.0 GB fp16

# ── VRAM monitor ─────────────────────────────────────────────
def vram_status(label=""):
    if not torch.cuda.is_available():
        print("CUDA not available")
        return
    total     = torch.cuda.get_device_properties(0).total_memory / 1e9
    reserved  = torch.cuda.memory_reserved(0)  / 1e9
    allocated = torch.cuda.memory_allocated(0) / 1e9
    free      = total - reserved
    tag = f"[{label}] " if label else ""
    print(f"{tag}VRAM | Total: {total:.1f}GB  "
          f"Used: {allocated:.2f}GB  "
          f"Reserved: {reserved:.2f}GB  "
          f"Free: {free:.2f}GB")

# ── Load both models ──────────────────────────────────────────
def load_models(device="cuda", verbose=True):
    if not torch.cuda.is_available():
        print("WARNING: CUDA not found — running on CPU (very slow)")
        device = "cpu"

    if verbose:
        print(f"Device : {torch.cuda.get_device_name(0)}")
        vram_status("before load")

    # ── Draft model ──────────────────────────────────────────
    if verbose:
        print(f"\nLoading draft model: {DRAFT_MODEL}")
    draft_tok = AutoTokenizer.from_pretrained(DRAFT_MODEL)
    draft_tok.pad_token = draft_tok.eos_token

    draft_model = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL,
        torch_dtype=torch.float16,
    ).to(device).eval()

    if verbose:
        vram_status("after draft")

    # ── Target model ─────────────────────────────────────────
    if verbose:
        print(f"\nLoading target model: {TARGET_MODEL}")
    target_tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    target_tok.pad_token = target_tok.eos_token

    target_model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL,
        torch_dtype=torch.float16,
    ).to(device).eval()

    if verbose:
        vram_status("after target")
        print("\nBoth models loaded successfully.\n")

    return draft_model, draft_tok, target_model, target_tok

# ── Measure β = T_draft / T_target ───────────────────────────
def measure_beta(draft_model, target_model, tokenizer,
                 prompt="The first digits of pi are", n_runs=10):
    """
    β is the cost ratio: time for one draft pass / time for one target pass.
    A small β means drafting is cheap — good for speculative decoding.
    """
    device = next(draft_model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    def time_forward(model, n):
        times = []
        for _ in range(n):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(**inputs)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        # drop first (cold) run
        return sum(times[1:]) / max(len(times) - 1, 1)

    t_draft  = time_forward(draft_model,  n_runs)
    t_target = time_forward(target_model, n_runs)
    beta = t_draft / t_target

    print(f"\nBeta measurement:")
    print(f"  Draft  forward pass : {t_draft*1000:.1f} ms")
    print(f"  Target forward pass : {t_target*1000:.1f} ms")
    print(f"  β = {beta:.3f}  (draft costs {beta:.1%} of target)")
    return beta

# ── Standalone autoregressive baseline ───────────────────────
def autoregressive_baseline(model, tokenizer, prompt,
                             n_tokens=100, label="model"):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=n_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    generated = tokenizer.decode(
        out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    )
    tps = n_tokens / elapsed
    print(f"[{label}] {tps:.1f} tok/s | {elapsed:.2f}s | "
          f'"{generated[:60]}..."')
    return tps, generated


if __name__ == "__main__":
    draft_model, draft_tok, target_model, target_tok = load_models()
    beta = measure_beta(draft_model, target_model, target_tok)

    PROMPT = "The first digits of pi are"
    print("\n=== BASELINES ===")
    autoregressive_baseline(draft_model,  draft_tok,  PROMPT, 100, "GPT-2 Small (draft)")
    autoregressive_baseline(target_model, target_tok, PROMPT, 100, "GPT-2 XL   (target)")