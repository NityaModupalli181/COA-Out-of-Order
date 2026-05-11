# Speculative Decoding & Multi-Token Pipelines
**CECS 530 — Advanced Computer Architecture | Team 10**
**California State University, Long Beach | May 2025**

> Nitya Modupalli · Varun S. Manik

---

## Overview

A complete implementation of speculative decoding for LLM inference
acceleration, evaluated on a memory-constrained NVIDIA GTX 1650 (4 GB GDDR6).

The project implements **six decoder configurations** and evaluates them
across **four model pairs** spanning two draft architectures (Transformer and
SSM) and two target scales.

The key finding is architectural, not algorithmic: speculative decoding is
theoretically sound (Theorem 1 guarantees distribution preservation), but the
4 GB VRAM constraint causes bandwidth contention that suppresses real-world
speedup below the 2× analytical ceiling. The Mamba-130m experiment confirms
O(1) recurrent-state decode via a constant β across all tested sequence lengths.

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/NityaModupalli181/COA-Out-of-Order.git
cd COA-Out-of-Order

# 2. Install PyTorch with CUDA (required first)
pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Verify GPU and run smoke test (~5 min)
python main.py --quick
```

**Expected smoke test output:**
```
GPU  : NVIDIA GeForce GTX 1650
VRAM : 4.3 GB  ✓
[1/4] Target baseline...     ~19.9 tok/s
[2/4] Base speculative...     ~1.4 tok/s  α=0.52
[3/4] Full decoder...         ~1.6 tok/s  α=0.66
[4/4] Correctness check...   ✓ pi sequence matches
```

---

## Run Commands

| Command | Description | Time |
|---------|-------------|------|
| `python main.py` | Full benchmark + Mamba comparison + all plots | ~40 min |
| `python main.py --quick` | Smoke test: verify GPU + all decoders run | ~5 min |
| `python main.py --bench` | Full benchmark + plots, skip Mamba | ~25 min |
| `python main.py --mamba` | Mamba-130m vs GPT-2 Small comparison | ~10 min |
| `python main.py --test` | Run full test suite (correctness + edge cases) | ~8 min |

### Individual module entry points

```bash
cd speculative_decoding/

python models.py              # GPU check + β measurement
python speculative_decoder.py # Test base speculative decoder (Pair 1)
python kv_cache.py            # Test dual-state KV-cache rollback
python adaptive_k.py          # Test adaptive k controller + full decoder
python pipeline.py            # Test two-stage buffered pipeline
python mamba_draft.py         # Mamba-130m O(1) decode comparison (Pair 3)
python benchmark.py           # Full sweep → results/results.json
python plots.py               # Generate all 5 charts from results.json
python run_all.py             # Everything above in sequence
```

### Run tests

```bash
# All tests
python main.py --test

# Or with pytest directly
python -m pytest tests/ -v

# Individual test files
python -m pytest tests/test_correctness.py -v   # Theorem 1 + beta validation
python -m pytest tests/test_edge_cases.py -v    # NaN, rollback, pipeline flush
python -m pytest tests/test_mamba_proxy.py -v   # Mamba O(1) decode + scope tests
```

---

## Repository Structure

```
COA-Out-of-Order/
│
├── main.py                          ← Top-level entry point
├── requirements.txt                 ← Python dependencies
├── README.md                        ← This file
│
├── speculative_decoding/            ← All implementation files
│   ├── models.py                    ← Load models, measure β, run baselines
│   ├── speculative_decoder.py       ← Core algorithm: draft → verify → accept/reject
│   ├── kv_cache.py                  ← Dual-state KV-cache with O(k) rollback
│   ├── adaptive_k.py                ← Adaptive speculation depth + Full decoder
│   ├── pipeline.py                  ← Two-stage FIFO-buffered pipeline (threaded)
│   ├── mamba_draft.py               ← Mamba-130m SSM vs GPT-2 Small (E2 proxy)
│   ├── benchmark.py                 ← Full benchmark sweep → results.json
│   ├── plots.py                     ← All 5 charts → PNG files
│   └── run_all.py                   ← Single entry point for all experiments
│
├── configs/                         ← Model-pair parameter definitions
│   ├── pair1_gpt2.json              ← P1: GPT-2 Sm → XL (E1 Measured)
│   ├── pair2_tinyllama.json         ← P2: TinyLlama → Llama2-7B (E3 Projected)
│   ├── pair3_mamba_gpt2.json        ← P3: Mamba → GPT-2 XL (E2 Proxy)
│   └── pair4_mamba_llama.json       ← P4: Mamba → Llama2-7B (E3 Proxy Projected)
│
├── tests/                           ← Test suite
│   ├── conftest.py                  ← Shared pytest fixtures + paper ground truth
│   ├── test_correctness.py          ← Theorem 1 + β measurement validation
│   ├── test_edge_cases.py           ← NaN guard, KV rollback, adaptive k, flush
│   └── test_mamba_proxy.py          ← Mamba cache_position, logit truncation, O(1)
│
└── results/                         ← Generated benchmark output (after running)
    ├── README.md                    ← Explains expected output files
    ├── results.json                 ← Generated by benchmark.py
    ├── mamba_results.json           ← Generated by mamba_draft.py
    └── *.png                        ← Generated by plots.py
```

---

## Model Pairs and Evidence Levels

The paper uses three evidence levels to prevent projections from being mistaken
for experiments:

| P | Draft | Target | Arch | Vocab | HW | Evidence |
|---|-------|--------|------|-------|-----|---------|
| 1 | GPT-2 Small (117M) | GPT-2 XL (1.5B) | T→T | Exact | GTX 1650 | **E1 Measured** |
| 2 | TinyLlama (1.1B) | Llama2-7B (7B) | T→T | Exact | T4 (Colab) | **E3 Projected** |
| 3 | Mamba-130m (130M) | GPT-2 XL (1.5B) | S→T | Proxy | GTX 1650 | **E2 Proxy** |
| 4 | Mamba-130m (130M) | Llama2-7B (7B) | S→T | Proxy | T4 (Colab) | **E3 Projected** |

- **E1**: Fully measured. Theorem 1 applies (exact vocabulary).
- **E2**: Executed, but distribution preservation NOT claimed (vocabulary mismatch). Measures β and O(1) decode only.
- **E3**: Analytical projection via `S = (1 - α^(k+1)) / ((1-α)(kβ+1))`. Not an experiment.

---

## Six Decoder Configurations

| Configuration | Description |
|--------------|-------------|
| `Target-only` | Autoregressive GPT-2 XL baseline (no speculation) |
| `Base-Spec` | Fixed k=4, sampling acceptance, no KV-cache |
| `Spec+KV` | Base-Spec + dual-state KV-cache with O(k) rollback |
| `Spec+AdaptK` | Base-Spec + adaptive k controller (Eq. 5 from paper) |
| `Full Decoder` | KV-cache + adaptive k combined |
| `Pipeline` | Two-stage FIFO-buffered producer-consumer pipeline |

---

## Key Results (Pair 1, GTX 1650)

### Measured cost ratios (β = T_q / T_p)

| Pair | n=50 | n=100 | n=200 | Mean |
|------|------|-------|-------|------|
| P1: GPT-2 Small / XL | 0.066 | 0.064 | 0.062 | **0.064 ± 0.002** |
| P3: Mamba-130m / XL | 0.102 | 0.099 | 0.097 | **0.099 ± 0.002** |

Mamba's constant β across all n confirms **O(1) recurrent-state decode**.

### Throughput at n=100 (tokens/s)

| Configuration | n=50 | n=100 | n=200 |
|--------------|------|-------|-------|
| Target-only | 18.0 | 19.9 | 20.5 |
| Draft-only | 81.4 | 85.1 | 86.9 |
| Base-Spec | 3.5 | 1.4 | 2.0 |
| Spec+KV | 1.5 | 1.5 | 1.5 |
| Spec+AdaptK | 1.1 | 1.5 | 1.5 |
| Full Decoder | 1.6 | 1.5 | 1.5 |
| Pipeline | 1.3 | 1.0 | 0.7 |

### Acceptance rate by prompt type (Full Decoder, k=4, n=100)

| Prompt type | tok/s | α |
|------------|-------|---|
| Factual ("digits of pi") | 1.6 | 0.66 |
| Creative ("once upon a time") | 1.2 | 0.58 |
| Structured (Python code) | 1.7 | 0.72 |

The 24-point spread in α demonstrates why adaptive k is essential for
heterogeneous serving workloads.

---

## Hardware Requirements

| Component | Minimum | Used |
|-----------|---------|------|
| GPU VRAM | 3.8 GB | GTX 1650 Ti — 4.3 GB ✓ |
| CUDA | 11.8 or 12.x | CUDA 12.1 ✓ |
| Python | 3.8–3.12 | Python 3.11.9 ✓ |
| RAM | 8 GB | Recommended |

Models download automatically from HuggingFace (~1.7 GB total).

---

## Three Key Implementation Challenges

### 1. NaN in corrected residual sampling
When p ≈ q (draft and target agree), `max(0, p - q)` approaches zero.
Multinomial sampling on a near-zero vector produces NaN.
**Fix**: if `||r||_1 < 1e-6`, fall back to sampling from p directly.

### 2. Mamba `cache_position` requirement
Mamba uses recurrent state (`cache_params`), not a KV-cache.
Each incremental generation step requires an explicit `cache_position` tensor.
Omitting it produces coherent but state-inconsistent output with no crash.
**Fix**: maintain a running position counter alongside the recurrent state.

### 3. Pipeline flush race condition
When the verify thread triggers rollback, the draft thread may be mid-batch.
**Fix**: Python `threading.Event` flush signal; draft thread checks flag before
each new token in a batch and abandons in-progress work immediately.

---

## Troubleshooting

### CUDA not available
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### Out of memory
In `speculative_decoding/models.py`, change:
```python
TARGET_MODEL = "gpt2-large"   # 774M uses ~1.5 GB instead of 3.0 GB
```

### Mamba model fails to load
```bash
pip install transformers==4.44.0
```

### Pipeline is slow (~1 tok/s)
Expected on single GPU. Python GIL prevents true thread parallelism.
Both threads share one CUDA context — this is a runtime limitation, not
an architectural flaw. See paper §4.3 for full analysis.

---

## References

1. Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (ICML 2023)
2. Chen et al. — Accelerating LLM Decoding with Speculative Sampling (arXiv 2023)
3. Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (ICML 2024)
4. Gloeckle et al. — Better & Faster LLMs via Multi-Token Prediction (ICML 2024)
5. Bachmann et al. — Judge Decoding: Faster Speculative Sampling via Token-Level Judge (ICLR 2025)
6. Gu & Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (ICLR 2024)
7. Lahoti et al. — Mamba-3: Improved Sequence Modeling using State Space Principles (ICLR 2026)
8. DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv 2024)
9. Miao et al. — SpecInfer: Tree-Based Speculative Inference and Verification (PPoPP 2024)
10. McDanel — AMUSD: Asynchronous Multi-Device Speculative Decoding (ISCAS 2025)

---

## External Dependencies

All models sourced from HuggingFace Model Hub (Apache 2.0 / Meta Llama licence):
- `gpt2` and `gpt2-xl`: OpenAI GPT-2 (MIT)
- `state-spaces/mamba-130m-hf`: Mamba SSM (Apache 2.0)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`: TinyLlama (Apache 2.0)
- `meta-llama/Llama-2-7b-hf`: Llama 2 (Meta Llama 2 Community Licence)

PyTorch, Transformers, and Accelerate are used under their respective
open-source licences.
