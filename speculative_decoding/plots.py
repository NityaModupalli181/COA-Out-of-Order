"""Generate all benchmark charts from results.json + mamba_results.json.
Run AFTER benchmark.py and mamba_draft.py have completed.
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "figure.facecolor": "#1A0533",
    "axes.facecolor":   "#1A0533",
    "axes.edgecolor":   "#B0A0CC",
    "axes.labelcolor":  "#B0A0CC",
    "xtick.color":      "#B0A0CC",
    "ytick.color":      "#B0A0CC",
    "text.color":       "#FFFFFF",
    "grid.color":       "#2E0D55",
    "grid.alpha":       0.6,
    "legend.facecolor": "#250845",
    "legend.edgecolor": "#B0A0CC",
    "font.family":      "DejaVu Sans",
})

COLORS = {
    "target_only":      "#FF6B6B",
    "draft_only":       "#60A5FA",
    "base_speculative": "#9B59F5",
    "speculative_kv":   "#FF9A3C",
    "full_decoder":     "#4ADE80",
    "theoretical":      "#F5C842",
    "mamba_draft":      "#F472B6",   # pink — Mamba SSM
    "gpt2_draft":       "#38BDF8",   # sky blue — GPT-2 Small draft
}

LABELS = {
    "target_only":      "Target only (GPT-2 XL)",
    "draft_only":       "Draft only (GPT-2 Small)",
    "base_speculative": "Base Speculative",
    "speculative_kv":   "Speculative + KV-Cache",
    "full_decoder":     "Full Decoder (KV + Adaptive K)",
    "mamba_draft":      "Mamba-130m Draft (SSM)",
    "gpt2_draft":       "GPT-2 Small Draft (Transformer)",
}

N_LIST = [50, 100, 200]


# ── Load helpers ──────────────────────────────────────────────
def load():
    with open("results.json") as f:
        return json.load(f)

def load_mamba():
    if os.path.exists("mamba_results.json"):
        with open("mamba_results.json") as f:
            return json.load(f)
    return None

def theoretical_speedup(alpha, k, beta):
    numerator   = (1 - alpha ** (k + 1)) / (1 - alpha)
    denominator = k * beta + 1
    return numerator / denominator


# ── Plot 1: Throughput bar chart ──────────────────────────────
def plot_throughput_by_method(data, mamba=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Token Throughput by Method", fontsize=14,
                 color="white", y=1.02)

    cfg_order = ["target_only", "draft_only", "base_speculative",
                 "speculative_kv", "full_decoder"]

    for ax_i, n in enumerate(N_LIST):
        ax = axes[ax_i]
        ax.set_title(f"n = {n} tokens", fontsize=12)
        ax.set_ylabel("Tokens / second" if ax_i == 0 else "")
        ax.grid(axis="y", alpha=0.4)
        ax.set_axisbelow(True)

        names, tps_vals, bar_colors = [], [], []
        for cfg in cfg_order:
            r = data["configs"][cfg].get(str(n))
            if r:
                short = LABELS[cfg].split("(")[0].strip()
                names.append(short)
                tps_vals.append(r["tps"])
                bar_colors.append(COLORS[cfg])

        # Add Mamba result if available (use n=100 value for all bars)
        if mamba and "mamba_tps" in mamba:
            names.append("Mamba-130m\nDraft (SSM)")
            tps_vals.append(mamba["mamba_tps"])
            bar_colors.append(COLORS["mamba_draft"])

        bars = ax.bar(range(len(names)), tps_vals,
                      color=bar_colors, edgecolor="#250845", linewidth=0.5)
        for bar, v in zip(bars, tps_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8.5, color="white")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig("plot_throughput.png", dpi=150, bbox_inches="tight",
                facecolor="#1A0533")
    print("Saved: plot_throughput.png")
    plt.close()


# ── Plot 2: Throughput line chart ─────────────────────────────
def plot_throughput_line(data, mamba=None):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("Sequence Length vs Token Throughput", fontsize=13)
    ax.set_xlabel("Sequence length (n)")
    ax.set_ylabel("Tokens / second")
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)

    for cfg in ["target_only", "draft_only", "base_speculative",
                "speculative_kv", "full_decoder"]:
        ys = [data["configs"][cfg][str(n)]["tps"] for n in N_LIST]
        ax.plot(N_LIST, ys, "o-", color=COLORS[cfg],
                label=LABELS[cfg], linewidth=2, markersize=7)

    # Theoretical line
    beta      = data.get("beta", 0.064)
    alpha_avg = 0.65
    k         = 4
    theory    = [
        theoretical_speedup(alpha_avg, k, beta) *
        data["configs"]["target_only"][str(n)]["tps"]
        for n in N_LIST
    ]
    ax.plot(N_LIST, theory, "s--", color=COLORS["theoretical"],
            label="Theoretical upper bound", linewidth=1.5, markersize=6)

    # Mamba horizontal line (single value, shown as dashed across all n)
    if mamba and "mamba_tps" in mamba:
        mamba_val = mamba["mamba_tps"]
        ax.axhline(mamba_val, color=COLORS["mamba_draft"],
                   linestyle=(0, (5, 3)), linewidth=2,
                   label=f"Mamba-130m Draft (SSM) — {mamba_val:.1f} tok/s")
        ax.annotate(f"Mamba: {mamba_val:.1f}",
                    xy=(N_LIST[-1], mamba_val),
                    xytext=(-60, 8), textcoords="offset points",
                    color=COLORS["mamba_draft"], fontsize=9)

    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig("plot_throughput_line.png", dpi=150, bbox_inches="tight",
                facecolor="#1A0533")
    print("Saved: plot_throughput_line.png")
    plt.close()


# ── Plot 3: Speedup bar chart ─────────────────────────────────
def plot_speedup(data, mamba=None):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_title("Speedup over Target-Only Baseline", fontsize=13)
    ax.set_xlabel("Sequence length (n)")
    ax.set_ylabel("Speedup (×)")
    ax.axhline(1.0, color="#FF6B6B", linestyle="--",
               alpha=0.5, label="Baseline (1.0×)")
    ax.grid(alpha=0.4, axis="y")
    ax.set_axisbelow(True)

    cfg_order  = ["draft_only", "base_speculative",
                  "speculative_kv", "full_decoder"]
    bar_width  = 0.14
    x          = np.arange(len(N_LIST))

    for i, cfg in enumerate(cfg_order):
        speedups = []
        for n in N_LIST:
            tgt = data["configs"]["target_only"][str(n)]["tps"]
            r   = data["configs"][cfg].get(str(n))
            speedups.append(r["tps"] / tgt if r else 0)
        offset = (i - len(cfg_order) / 2 + 0.5) * bar_width
        bars   = ax.bar(x + offset, speedups, bar_width,
                        color=COLORS[cfg], label=LABELS[cfg],
                        edgecolor="#250845", linewidth=0.5)
        for bar, v in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{v:.2f}×", ha="center", va="bottom",
                    fontsize=7.5, color="white")

    # Theoretical speedup bars
    beta       = data.get("beta", 0.064)
    theory_su  = [theoretical_speedup(0.65, 4, beta)] * len(N_LIST)
    offset     = (len(cfg_order) - len(cfg_order) / 2 + 0.5) * bar_width
    ax.bar(x + offset, theory_su, bar_width,
           color=COLORS["theoretical"], label="Theoretical",
           edgecolor="#250845", linewidth=0.5, alpha=0.8)

    # Mamba speedup bars
    if mamba and "mamba_tps" in mamba and "baseline_tps" in mamba:
        mamba_su  = [mamba["mamba_tps"] / mamba["baseline_tps"]] * len(N_LIST)
        offset_m  = (len(cfg_order) + 1 - len(cfg_order) / 2 + 0.5) * bar_width
        bars_m    = ax.bar(x + offset_m, mamba_su, bar_width,
                           color=COLORS["mamba_draft"],
                           label="Mamba-130m Draft (SSM)",
                           edgecolor="#250845", linewidth=0.5)
        for bar, v in zip(bars_m, mamba_su):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{v:.2f}×", ha="center", va="bottom",
                    fontsize=7.5, color=COLORS["mamba_draft"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in N_LIST])
    ax.legend(fontsize=8.5, loc="upper left")
    plt.tight_layout()
    plt.savefig("plot_speedup.png", dpi=150, bbox_inches="tight",
                facecolor="#1A0533")
    print("Saved: plot_speedup.png")
    plt.close()


# ── Plot 4: α sensitivity ─────────────────────────────────────
def plot_alpha_sensitivity(data, mamba=None):
    alpha_data = data.get("alpha_sensitivity", {})
    if not alpha_data:
        print("No alpha sensitivity data. Run benchmark.py first.")
        return

    ks       = sorted(int(k) for k in alpha_data.keys())
    alphas   = [alpha_data[str(k)]["acceptance_rate"] for k in ks]
    tps_vals = [alpha_data[str(k)]["tps"] for k in ks]
    baseline = data["configs"]["target_only"]["100"]["tps"]
    speedups = [t / baseline for t in tps_vals]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_title(
        "Acceptance Rate and Speedup vs Speculation Depth k", fontsize=12
    )
    ax1.set_xlabel("Speculation depth k")
    ax1.set_ylabel("Acceptance rate α", color="#2EE0C0")
    ax1.tick_params(axis="y", labelcolor="#2EE0C0")
    ax1.grid(alpha=0.3)

    l1 = ax1.plot(ks, alphas, "o-", color="#2EE0C0",
                  linewidth=2, markersize=8, label="Acceptance rate α (GPT-2 Small)")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Speedup (×)", color=COLORS["theoretical"])
    ax2.tick_params(axis="y", labelcolor=COLORS["theoretical"])
    l2 = ax2.plot(ks, speedups, "s--", color=COLORS["theoretical"],
                  linewidth=2, markersize=8, label="Speedup × (GPT-2 Small)")

    # Mamba point — single horizontal reference lines
    if mamba and "mamba_alpha" in mamba and "mamba_tps" in mamba:
        m_alpha   = mamba["mamba_alpha"]
        m_speedup = mamba["mamba_tps"] / mamba["baseline_tps"]
        l3 = ax1.axhline(m_alpha, color=COLORS["mamba_draft"],
                         linestyle=":", linewidth=2,
                         label=f"Mamba-130m α = {m_alpha:.2f}")
        l4 = ax2.axhline(m_speedup, color=COLORS["mamba_draft"],
                         linestyle="-.", linewidth=2,
                         label=f"Mamba-130m speedup = {m_speedup:.2f}×")
        ax1.annotate(f"Mamba α={m_alpha:.2f}",
                     xy=(ks[-1], m_alpha), xytext=(-100, 8),
                     textcoords="offset points",
                     color=COLORS["mamba_draft"], fontsize=9)
        lines = l1 + l2 + [l3, l4]
    else:
        lines = l1 + l2

    labels = [ln.get_label() if hasattr(ln, "get_label") else "" for ln in lines]
    ax1.legend(lines, labels, fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig("plot_alpha_sensitivity.png", dpi=150, bbox_inches="tight",
                facecolor="#1A0533")
    print("Saved: plot_alpha_sensitivity.png")
    plt.close()


# ── Plot 5: Mamba vs GPT-2 draft comparison ───────────────────
def plot_mamba_comparison(data, mamba):
    """Dedicated chart comparing Mamba-130m vs GPT-2 Small as draft models."""
    if not mamba:
        print("No mamba_results.json found — skipping Mamba comparison chart.")
        return

    baseline  = mamba.get("baseline_tps", 16.0)
    gpt2_tps  = mamba.get("gpt2_tps",    0)
    mamba_tps = mamba.get("mamba_tps",   0)
    gpt2_a    = mamba.get("gpt2_alpha",  0)
    mamba_a   = mamba.get("mamba_alpha", 0)
    beta_g    = mamba.get("beta_gpt2",   0)
    beta_m    = mamba.get("beta_mamba",  0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Mamba-130m (SSM) vs GPT-2 Small (Transformer) as Draft Model",
        fontsize=13, color="white", y=1.02
    )

    # ── Sub-chart 1: Throughput comparison ───────────────────
    ax = axes[0]
    ax.set_title("Token Throughput (tok/s)", fontsize=11)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)

    methods = ["Target\nOnly\n(baseline)", "GPT-2 Small\nDraft\n(Transformer)",
               "Mamba-130m\nDraft\n(SSM)"]
    tps     = [baseline, gpt2_tps, mamba_tps]
    colors  = [COLORS["target_only"], COLORS["gpt2_draft"], COLORS["mamba_draft"]]

    bars = ax.bar(methods, tps, color=colors, edgecolor="#250845", linewidth=0.5)
    for bar, v in zip(bars, tps):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom",
                fontsize=11, color="white", fontweight="bold")

    # ── Sub-chart 2: Speedup comparison ──────────────────────
    ax = axes[1]
    ax.set_title("Speedup over Target Baseline", fontsize=11)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)
    ax.axhline(1.0, color=COLORS["target_only"], linestyle="--",
               alpha=0.5, label="Baseline")

    speedups = [1.0,
                gpt2_tps  / baseline if baseline else 0,
                mamba_tps / baseline if baseline else 0]
    bars = ax.bar(methods, speedups,
                  color=colors, edgecolor="#250845", linewidth=0.5)
    for bar, v in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{v:.2f}×", ha="center", va="bottom",
                fontsize=11, color="white", fontweight="bold")

    # ── Sub-chart 3: Acceptance rate + β ─────────────────────
    ax = axes[2]
    ax.set_title("Acceptance Rate α and Cost Ratio β", fontsize=11)
    ax.grid(axis="y", alpha=0.4)
    ax.set_axisbelow(True)

    x      = np.array([0, 1])
    width  = 0.35
    alphas = [gpt2_a, mamba_a]
    betas  = [beta_g, beta_m]

    b1 = ax.bar(x - width/2, alphas, width,
                color=[COLORS["gpt2_draft"], COLORS["mamba_draft"]],
                label="Acceptance rate α", edgecolor="#250845")
    b2 = ax.bar(x + width/2, betas, width,
                color=[COLORS["gpt2_draft"], COLORS["mamba_draft"]],
                alpha=0.5, label="Cost ratio β", edgecolor="#250845",
                hatch="//")

    for bar, v in zip(list(b1) + list(b2), alphas + betas):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}", ha="center", va="bottom",
                fontsize=9, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels(["GPT-2 Small\n(Transformer)", "Mamba-130m\n(SSM)"])
    ax.legend(fontsize=9)

    # Annotation box
    if mamba_tps > gpt2_tps:
        diff   = ((mamba_tps - gpt2_tps) / gpt2_tps) * 100
        winner = f"Mamba is {diff:.1f}% faster"
        wcolor = COLORS["mamba_draft"]
    else:
        diff   = ((gpt2_tps - mamba_tps) / mamba_tps) * 100
        winner = f"GPT-2 is {diff:.1f}% faster"
        wcolor = COLORS["gpt2_draft"]

    fig.text(0.5, -0.04, winner,
             ha="center", fontsize=13, color=wcolor, fontweight="bold")
    fig.text(
        0.5, -0.09,
        "Mamba-1 used as proxy for Mamba-3  |  "
        "Mamba-3 adds MIMO efficiency, complex states & half state size",
        ha="center", fontsize=9, color="#B0A0CC", style="italic"
    )

    plt.tight_layout()
    plt.savefig("plot_mamba_comparison.png", dpi=150, bbox_inches="tight",
                facecolor="#1A0533")
    print("Saved: plot_mamba_comparison.png")
    plt.close()


# ── Summary table ─────────────────────────────────────────────
def print_summary_table(data, mamba=None):
    beta = data.get("beta", "N/A")
    print(f"\n{'='*65}")
    print(f"  Summary Table  |  β = {beta:.3f}")
    print(f"{'='*65}")
    print(f"{'Config':<30} {'n=50':>8} {'n=100':>8} {'n=200':>8}  {'unit'}")
    print(f"{'─'*65}")

    cfgs = [
        ("Target only",         "target_only"),
        ("Draft only",          "draft_only"),
        ("Base speculative",    "base_speculative"),
        ("Speculative + KV",    "speculative_kv"),
        ("Full decoder",        "full_decoder"),
    ]

    for label, cfg in cfgs:
        row = f"{label:<30}"
        for n in N_LIST:
            r    = data["configs"][cfg].get(str(n))
            row += f"{r['tps']:>8.1f}" if r else f"{'—':>8}"
        print(row + "  tok/s")

    if mamba and "mamba_tps" in mamba:
        print(f"\n{'─'*65}")
        print(f"  Mamba-130m draft (SSM)        "
              f"{'—':>8} {mamba['mamba_tps']:>8.1f} {'—':>8}  tok/s")
        print(f"  GPT-2 Small draft (Transformer)"
              f"{'—':>8} {mamba['gpt2_tps']:>8.1f} {'—':>8}  tok/s")
        print(f"  Mamba α                       "
              f"{'—':>8} {mamba['mamba_alpha']:>8.2f} {'—':>8}  acceptance")
        print(f"  GPT-2 Small α                 "
              f"{'—':>8} {mamba['gpt2_alpha']:>8.2f} {'—':>8}  acceptance")

    print(f"{'='*65}\n")


# ── Main ──────────────────────────────────────────────────────
if _name_ == "_main_":
    data  = load()
    mamba = load_mamba()

    if mamba:
        print("Mamba results found — including in all charts.\n")
    else:
        print("No mamba_results.json found — charts will not include Mamba.\n"
              "Run: python mamba_draft.py  to generate it.\n")

    print_summary_table(data, mamba)
    plot_throughput_by_method(data, mamba)
    plot_throughput_line(data, mamba)
    plot_speedup(data, mamba)
    plot_alpha_sensitivity(data, mamba)
    plot_mamba_comparison(data, mamba)

    print("\nAll charts generated:")
    print("  plot_throughput.png          bar chart: tok/s by method")
    print("  plot_throughput_line.png     line: tok/s vs sequence length")
    print("  plot_speedup.png             speedup x over baseline")
    print("  plot_alpha_sensitivity.png   alpha and speedup vs k")
    print("  plot_mamba_comparison.png    Mamba SSM vs GPT-2 draft (dedicated)")