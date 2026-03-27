"""
plots.py
Generate all benchmark charts from results.json.
Run AFTER benchmark.py has completed.
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
    "target_only":     "#FF6B6B",
    "draft_only":      "#60A5FA",
    "base_speculative":"#9B59F5",
    "speculative_kv":  "#FF9A3C",
    "full_decoder":    "#4ADE80",
    "theoretical":     "#F5C842",
}
LABELS = {
    "target_only":     "Target only (GPT-2 XL)",
    "draft_only":      "Draft only (GPT-2 Small)",
    "base_speculative":"Base Speculative",
    "speculative_kv":  "Speculative + KV-Cache",
    "full_decoder":    "Full Decoder (KV + Adaptive K)",
}
N_LIST = [50, 100, 200]


def theoretical_speedup(alpha, k, beta):
    """Equation (4) from the paper."""
    numerator   = (1 - alpha**(k+1)) / (1 - alpha)
    denominator = k * beta + 1
    return numerator / denominator


def load():
    with open("results.json") as f:
        return json.load(f)


def plot_throughput_by_method(data):
    """Bar chart: tok/s by method at each n."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Token Throughput by Method", fontsize=14, color="white", y=1.01)

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
                names.append(LABELS[cfg].split("(")[0].strip())
                tps_vals.append(r["tps"])
                bar_colors.append(COLORS[cfg])

        bars = ax.bar(range(len(names)), tps_vals,
                      color=bar_colors, edgecolor="#250845", linewidth=0.5)
        for bar, v in zip(bars, tps_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{v:.1f}", ha="center", va="bottom",
                    fontsize=8.5, color="white")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)

    plt.tight_layout()
    plt.savefig("plot_throughput.png", dpi=150, bbox_inches="tight",
                facecolor="#1A0533")
    print("Saved: plot_throughput.png")


def plot_throughput_line(data):
    """Line chart: tok/s vs n for all methods."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Sequence Length vs Token Throughput", fontsize=13)
    ax.set_xlabel("Sequence length (n)")
    ax.set_ylabel("Tokens / second")
    ax.grid(alpha=0.4)
    ax.set_axisbelow(True)

    for cfg in ["target_only", "draft_only", "base_speculative",
                "speculative_kv", "full_decoder"]:
        ys = []
        for n in N_LIST:
            r = data["configs"][cfg].get(str(n))
            ys.append(r["tps"] if r else 0)
        ax.plot(N_LIST, ys, "o-", color=COLORS[cfg],
                label=LABELS[cfg], linewidth=2, markersize=7)

    # Theoretical line
    beta = data.get("beta", 0.25)
    alpha_avg = 0.6
    k = 4
    theory = [theoretical_speedup(alpha_avg, k, beta) *
              data["configs"]["target_only"][str(n)]["tps"]
              for n in N_LIST]
    ax.plot(N_LIST, theory, "s--", color=COLORS["theoretical"],
            label="Theoretical upper bound", linewidth=1.5, markersize=6)

    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig("plot_throughput_line.png", dpi=150, bbox_inches="tight",
                facecolor="#1A0533")
    print("Saved: plot_throughput_line.png")


def plot_speedup(data):
    """Bar chart: speedup over target baseline at each n."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Speedup over Target-Only Baseline", fontsize=13)
    ax.set_xlabel("Sequence length (n)")
    ax.set_ylabel("Speedup (×)")
    ax.axhline(1.0, color="#FF6B6B", linestyle="--", alpha=0.5,
               label="Baseline (1.0×)")
    ax.grid(alpha=0.4, axis="y")
    ax.set_axisbelow(True)

    bar_width = 0.15
    x = np.arange(len(N_LIST))
    cfg_order = ["draft_only", "base_speculative", "speculative_kv", "full_decoder"]

    for i, cfg in enumerate(cfg_order):
        speedups = []
        for n in N_LIST:
            tgt = data["configs"]["target_only"][str(n)]["tps"]
            r   = data["configs"][cfg].get(str(n))
            speedups.append(r["tps"] / tgt if r else 0)

        offset = (i - len(cfg_order)/2 + 0.5) * bar_width
        bars = ax.bar(x + offset, speedups, bar_width,
                      color=COLORS[cfg], label=LABELS[cfg],
                      edgecolor="#250845", linewidth=0.5)
        for bar, v in zip(bars, speedups):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f"{v:.2f}×", ha="center", va="bottom",
                    fontsize=8, color="white")

    # Theoretical bars
    beta = data.get("beta", 0.25)
    theory_su = [theoretical_speedup(0.6, 4, beta) for _ in N_LIST]
    offset = (len(cfg_order) - len(cfg_order)/2 + 0.5) * bar_width
    ax.bar(x + offset, theory_su, bar_width,
           color=COLORS["theoretical"], label="Theoretical",
           edgecolor="#250845", linewidth=0.5, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in N_LIST])
    ax.legend(fontsize=9, loc="upper left")
    plt.tight_layout()
    plt.savefig("plot_speedup.png", dpi=150, bbox_inches="tight",
                facecolor="#1A0533")
    print("Saved: plot_speedup.png")


def plot_alpha_sensitivity(data):
    """Dual-axis: acceptance rate α and speedup vs k."""
    alpha_data = data.get("alpha_sensitivity", {})
    if not alpha_data:
        print("No alpha sensitivity data found. Run benchmark.py first.")
        return

    ks      = sorted(int(k) for k in alpha_data.keys())
    alphas  = [alpha_data[str(k)]["acceptance_rate"] for k in ks]
    tps_vals = [alpha_data[str(k)]["tps"] for k in ks]
    baseline = data["configs"]["target_only"]["100"]["tps"]
    speedups = [t / baseline for t in tps_vals]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.set_title("Acceptance Rate and Speedup vs Speculation Depth k",
                  fontsize=12)
    ax1.set_xlabel("Speculation depth k")
    ax1.set_ylabel("Acceptance rate α", color="#2EE0C0")
    ax1.tick_params(axis="y", labelcolor="#2EE0C0")
    ax1.grid(alpha=0.3)

    l1 = ax1.plot(ks, alphas, "o-", color="#2EE0C0", linewidth=2,
                  markersize=8, label="Acceptance rate α")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Speedup (×)", color=COLORS["theoretical"])
    ax2.tick_params(axis="y", labelcolor=COLORS["theoretical"])
    l2 = ax2.plot(ks, speedups, "s--", color=COLORS["theoretical"],
                  linewidth=2, markersize=8, label="Speedup ×")

    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, fontsize=10, loc="upper right")

    plt.tight_layout()
    plt.savefig("plot_alpha_sensitivity.png", dpi=150, bbox_inches="tight",
                facecolor="#1A0533")
    print("Saved: plot_alpha_sensitivity.png")


def print_summary_table(data):
    beta = data.get("beta", "N/A")
    print(f"\n{'='*65}")
    print(f"  Summary Table  |  β = {beta:.3f}")
    print(f"{'='*65}")
    print(f"{'Config':<28} {'n=50':>8} {'n=100':>8} {'n=200':>8}  {'unit'}")
    print(f"{'─'*65}")

    cfgs = [
        ("Target only",           "target_only"),
        ("Draft only",            "draft_only"),
        ("Base speculative",      "base_speculative"),
        ("Speculative + KV",      "speculative_kv"),
        ("Full decoder",          "full_decoder"),
    ]

    for label, cfg in cfgs:
        row = f"{label:<28}"
        for n in N_LIST:
            r = data["configs"][cfg].get(str(n))
            row += f"{r['tps']:>8.1f}" if r else f"{'—':>8}"
        print(row + "  tok/s")

    print(f"\n{'Config':<28} {'n=50':>8} {'n=100':>8} {'n=200':>8}  {'unit'}")
    print(f"{'─'*65}")
    tgt = {n: data["configs"]["target_only"][str(n)]["tps"] for n in N_LIST}
    for label, cfg in cfgs[1:]:
        row = f"{label:<28}"
        for n in N_LIST:
            r = data["configs"][cfg].get(str(n))
            su = r["tps"] / tgt[n] if r else 0
            row += f"{su:>7.2f}x"
        print(row + "  speedup")


if __name__ == "__main__":
    data = load()
    print_summary_table(data)
    plot_throughput_by_method(data)
    plot_throughput_line(data)
    plot_speedup(data)
    plot_alpha_sensitivity(data)
    print("\nAll plots generated.")