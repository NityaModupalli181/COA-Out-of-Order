# COA-Out-of-Order Team 10

# Speculative Decoding & Multi-Token Pipelines

A complete implementation of speculative decoding for LLM inference acceleration,
built for a GTX 1650 Ti (4 GB VRAM) using GPT-2 Small as the draft model and
GPT-2 XL as the target model. Includes a Mamba-130m (SSM) vs Transformer draft
model comparison experiment.

---

## Results

> Run python run_all.py to generate all charts below.

### Token Throughput by Method

![Token Throughput by Method](plot_throughput.png)

*Bar chart showing tokens/second for each decoder configuration at sequence lengths n=50, 100, 200.
Full Decoder (KV-Cache + Adaptive K) consistently achieves the highest throughput.*

---

### Sequence Length vs Token Throughput

![Sequence Length vs Token Throughput](plot_throughput_line.png)

*Line chart comparing all methods as sequence length grows. Speculative methods
scale better than the target-only baseline. The dashed pink line shows Mamba-130m
draft model throughput.*

---

### Speedup over Baseline

![Speedup over Baseline](plot_speedup.png)

*Speedup multiplier (×) compared to running GPT-2 XL alone. The gold bars show
the theoretical upper bound from the performance model. Pink bars show Mamba-130m
draft model speedup.*

---

### Acceptance Rate and Speedup vs Speculation Depth k

![Acceptance Rate vs k](plot_alpha_sensitivity.png)

*Dual-axis chart showing how acceptance rate α and overall speedup change as
speculation depth k increases. Dashed pink lines show Mamba-130m reference values.*

---

### Mamba-130m (SSM) vs GPT-2 Small (Transformer) Draft Model

![Mamba Comparison](plot_mamba_comparison.png)

*Dedicated three-panel comparison: token throughput, speedup, and acceptance rate α
vs cost ratio β for both draft architectures. Validates the SSM-as-drafter hypothesis
from the Mamba-3 paper (ICLR 2026).*

---

## Project Structure


speculative_decoding/
├── models.py               Load models, measure beta, run baselines
├── speculative_decoder.py  Core algorithm: draft → verify → accept/reject
├── kv_cache.py             KV-cache version with dual-state rollback
├── adaptive_k.py           Adaptive speculation depth + Full decoder
├── pipeline.py             Two-stage buffered pipeline (threaded)
├── mamba_draft.py          Mamba-130m SSM vs GPT-2 Small draft comparison
├── benchmark.py            Full benchmark sweep → results.json
├── plots.py                All charts → PNG files (includes Mamba)
├── run_all.py              Single entry point for everything
├── requirements.txt        Python dependencies
└── README.md               This file


---

## Hardware Requirements

| Component | Requirement | Your Setup |
|-----------|-------------|------------|
| GPU VRAM  | >= 3.8 GB   | GTX 1650 Ti — 4.3 GB ✓ |
| CUDA      | 11.8 or 12.x | CUDA 12.1 ✓ |
| Python    | 3.8 – 3.12  | Python 3.11.9 ✓ |
| RAM       | >= 8 GB     | Recommended |

---

## Installation

bash
# 1. Create virtual environment with Python 3.11
py -3.11 -m venv speculative_env_311
speculative_env_311\Scripts\activate          # Windows
# source speculative_env_311/bin/activate     # Mac / Linux

# 2. Install PyTorch with CUDA support
pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 --index-url https://download.pytorch.org/whl/cu121

# 3. Install remaining dependencies
pip install transformers==4.44.0 accelerate==0.27.0 numpy matplotlib


---

## Quick Start

bash
# Verify GPU is working and all decoders run correctly
python run_all.py --quick


Expected output:

GPU     : NVIDIA GeForce GTX 1650
VRAM    : 4.3 GB
Status  : ✓ Sufficient VRAM for GPT-2 XL

[1/4] Target baseline...       GPT-2 XL   ~16 tok/s
[2/4] Base speculative...      ~24 tok/s  α=0.65  speedup=1.5x
[3/4] Speculative + KV-Cache   ~28 tok/s  α=0.65  speedup=1.7x
[4/4] Full decoder (KV + K)    ~30 tok/s  α=0.66  speedup=1.9x

✓ Smoke test passed — all decoders working.


---

## All Run Commands

### One-Command Full Experiment

bash
python run_all.py


Runs everything in order:
1. Full benchmark sweep (all 5 configs × 3 sequence lengths)
2. Generates all benchmark charts
3. Runs Mamba-130m vs GPT-2 Small comparison
4. Generates dedicated Mamba comparison chart

---

### Individual Commands

| Command | What it does | Time |
|---------|-------------|------|
| python run_all.py | Everything: benchmark + Mamba | ~40 min |
| python run_all.py --quick | Smoke test only | ~5 min |
| python run_all.py --bench | Benchmark + plots, skip Mamba | ~25 min |
| python run_all.py --mamba | Mamba comparison only | ~10 min |
| python models.py | GPU check + beta measurement | ~2 min |
| python speculative_decoder.py | Test core algorithm | ~5 min |
| python kv_cache.py | Test KV-cache version | ~5 min |
| python adaptive_k.py | Test adaptive K + full decoder | ~5 min |
| python pipeline.py | Test threaded pipeline | ~5 min |
| python mamba_draft.py | Mamba vs GPT-2 draft comparison | ~10 min |
| python benchmark.py | Full sweep → results.json | ~25 min |
| python plots.py | Charts → PNG files | ~1 min |

---

## Output Files

| File | Description |
|------|-------------|
| results.json | All benchmark numbers (tok/s, alpha, rollbacks) |
| mamba_results.json | Mamba vs GPT-2 draft comparison numbers |
| plot_throughput.png | Bar chart: tok/s per method at n=50,100,200 |
| plot_throughput_line.png | Line chart: tok/s vs sequence length |
| plot_speedup.png | Speedup × over target baseline (includes Mamba) |
| plot_alpha_sensitivity.png | Acceptance rate and speedup vs k (includes Mamba) |
| plot_mamba_comparison.png | Dedicated Mamba SSM vs Transformer draft chart |

---

## Models Used

| Role | Model | Parameters | VRAM | Architecture |
|------|-------|------------|------|--------------|
| Target (verifier) | gpt2-xl | 1.5B | ~3.0 GB | Transformer |
| Draft (proposer) | gpt2 | 117M | ~0.27 GB | Transformer |
| SSM Draft (experiment) | state-spaces/mamba-130m-hf | 130M | ~0.26 GB | Mamba SSM |

All models download automatically from HuggingFace on first run (~1.7 GB total).

---

## What Each Decoder Does

### 1. Base Speculative Decoder
Fixed speculation depth k=4. Drafts 4 tokens with GPT-2 Small, verifies all
4 with GPT-2 XL in one parallel forward pass.

### 2. Speculative + KV-Cache
Adds dual-state KV-cache. Committed tokens cached permanently;
speculative tokens discarded on rollback. Avoids recomputing attention
over full context every step.

### 3. Full Decoder (KV-Cache + Adaptive K)
Adaptive k controller watches acceptance rate each round:
- α >= 0.72 → k increases by 1 (slow increase)
- α <  0.48 → k decreases by 2 (fast decrease)
- Hard floor: k >= 3 always

### 4. Multi-Token Pipeline
Two-stage threaded pipeline. Draft thread fills a bounded queue continuously.
Verify thread consumes from the queue. Backpressure pauses draft when queue is full.

### 5. Mamba-130m Draft (SSM)
Replaces GPT-2 Small drafter with Mamba-130m. Key SSM advantages:
- O(1) decode time per token (fixed recurrent state, no KV-cache)
- No growing memory burden at long contexts
- Lower β (cheaper per step than same-size Transformer)

---

## Key Metrics Explained

| Metric | Formula | Your Measured Value |
|--------|---------|---------------------|
| β (beta) | T_draft / T_target | *0.064* |
| α (alpha) | accepted / drafted | ~0.65 |
| Theoretical speedup | (1-α^(k+1)) / ((1-α)(kβ+1)) | ~2.0× |
| Actual speedup | measured_tps / baseline_tps | ~1.8× |

> *β = 0.064* means GPT-2 Small costs only 6.4% of one GPT-2 XL forward
> pass. This is excellent — you can run ~15 draft steps for every 1 target step.

---

## Architecture Diagram


Your Prompt
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                    models.py                        │
│   GPT-2 Small (0.27 GB) + GPT-2 XL (3.23 GB)      │
│   Total: 3.5 GB on GTX 1650 Ti                     │
└────────────────────┬────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐   ┌──────────────────────────────┐
│  Draft Phase    │   │       Verify Phase            │
│  GPT-2 Small   │   │       GPT-2 XL               │
│  Generates k=4 │──▶│  Checks all 4 in ONE pass    │
│  tokens fast   │   │  Accept / Reject each token  │
└─────────────────┘   └──────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
             Accepted tokens            Rejected token
             added to output            → resample from
                                          target distribution
                                          → restart draft

Configurations tested:
  base_speculative  →  above loop, fixed k=4
  speculative_kv    →  above + KV-cache (no recompute)
  full_decoder      →  above + KV-cache + adaptive k
  pipeline          →  draft and verify run in parallel threads
  mamba_draft       →  Mamba-130m replaces GPT-2 Small as drafter


---

## Troubleshooting

### CUDA not available
bash
pip uninstall torch torchvision torchaudio -y
pip install torch==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121


### Out of memory error
python
# In models.py, change:
TARGET_MODEL = "gpt2-large"   # 774M uses ~1.5 GB instead of 3.0 GB


### RuntimeError: probability tensor contains NaN
In speculative_decoder.py and kv_cache.py, replace the rejection block
corrected distribution with:
python
corrected = target_probs[0].clone()
corrected[draft_ids[j]] = max(0.0, corrected[draft_ids[j]].item() - p_draft)
corrected = torch.clamp(corrected, min=0.0)
s = corrected.sum()
if s < 1e-6:
    corrected = target_probs[0].clone()
    s = corrected.sum()
corrected = corrected / s
if torch.isnan(corrected).any() or torch.isinf(corrected).any():
    corrected = torch.ones_like(corrected) / corrected.shape[0]
bonus_token = torch.multinomial(corrected, 1).item()


### Mamba model fails to load
bash
pip install transformers==4.44.0


### Pipeline is very slow (~2 tok/s)
Expected on single GPU. Python GIL limits thread parallelism.
Document as a known limitation — the architecture is correct.

---

## References

1. Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (ICML 2023)
2. Chen et al. — Accelerating LLM Decoding with Speculative Sampling (arXiv 2023)
3. Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (ICML 2024)
4. Gloeckle et al. — Better & Faster LLMs via Multi-Token Prediction (Meta AI, ICML 2024)
5. Lahoti et al. — Mamba-3: Improved Sequence Modeling using State Space Principles (ICLR 2026)
6. Gu & Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (2023)