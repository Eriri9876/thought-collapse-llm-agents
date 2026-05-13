"""
Figure 1: Scaffold-level gap (EM_Full - EM_Direct) vs. coverage estimate (c_hat).
Positive gap = scaffold helps; negative = scaffold hurts.

25 points: Qwen2.5-14B/32B + DeepSeek-V3 + cross-family Llama-3.1-8B/70B,
each on 5 tasks (GSM8K, MATH-hard, HotpotQA, WebQ, TriviaQA).

Saves to figures/coverage_gap.{pdf,png}.

Label placement uses adjustText for automatic collision avoidance.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from adjustText import adjust_text

# ── data ──────────────────────────────────────────────────────────────────────
# c_hat: 20-example calibration-set Direct EM (seed=42 cal split)
# gap  : EM_Full - EM_Direct  (n=100 eval set; L70B uses 3-seed mean)

DATA = {
    "L8B": {
        "GSM8K":     (0.000, 0.070 - 0.060),   # +0.010
        "MATH-hard": (0.000, 0.140 - 0.160),   # -0.020
        "HotpotQA":  (0.150, 0.160 - 0.180),   # -0.020
        "WebQ":      (0.300, 0.160 - 0.210),   # -0.050
        "TriviaQA":  (0.600, 0.450 - 0.610),   # -0.160
    },
    "14B": {
        "GSM8K":     (0.10, 0.65 - 0.10),
        "MATH-hard": (0.15, 0.435 - 0.150),
        "HotpotQA":  (0.10, 0.20 - 0.28),
        "WebQ":      (0.40, 0.08 - 0.29),
        "TriviaQA":  (0.65, 0.434 - 0.630),
    },
    "32B": {
        "GSM8K":     (0.15, 0.44 - 0.20),
        "MATH-hard": (0.22, 0.350 - 0.220),
        "HotpotQA":  (0.20, 0.21 - 0.28),
        "WebQ":      (0.35, 0.12 - 0.30),
        "TriviaQA":  (0.70, 0.440 - 0.660),
    },
    "V3": {
        "GSM8K":     (0.25, 0.96 - 0.26),
        "MATH-hard": (0.36, 0.855 - 0.360),
        "HotpotQA":  (0.30, 0.32 - 0.29),
        "WebQ":      (0.25, 0.14 - 0.22),
        "TriviaQA":  (0.80, 0.739 - 0.780),
    },
    "L70B": {
        "GSM8K":     (0.15, 0.743 - 0.183),   # +0.560
        "MATH-hard": (0.05, 0.287 - 0.163),   # +0.124
        "HotpotQA":  (0.25, 0.253 - 0.330),   # -0.077
        "WebQ":      (0.25, 0.180 - 0.200),   # -0.020
        "TriviaQA":  (0.80, 0.680 - 0.823),   # -0.143
    },
}

MODEL_ORDER = ["L8B", "14B", "32B", "V3", "L70B"]

MODEL_STYLE = {
    "L8B":  {"color": "#888888", "marker": "x", "label": "Llama-3.1-8B",
             "size": 65, "lw": 1.5},
    "14B":  {"color": "#2166ac", "marker": "o", "label": "Qwen2.5-14B",
             "size": 70, "lw": 0.0},
    "32B":  {"color": "#f4a582", "marker": "s", "label": "Qwen2.5-32B",
             "size": 70, "lw": 0.0},
    "V3":   {"color": "#d6604d", "marker": "^", "label": "DeepSeek-V3",
             "size": 75, "lw": 0.0},
    "L70B": {"color": "#2E8B57", "marker": "*", "label": "Llama-3.1-70B",
             "size": 140, "lw": 0.0},
}

TASK_ABBREV = {
    "GSM8K":    "GSM",
    "MATH-hard":"MATH",
    "HotpotQA": "HQA",
    "WebQ":     "WebQ",
    "TriviaQA": "TQA",
}

# adjustText collision-avoidance parameters
EXPAND_POINTS = (1.1, 1.2)

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.4, 4.2))

texts = []  # collected for adjustText
for model in MODEL_ORDER:
    style = MODEL_STYLE[model]
    xs, ys, labels = [], [], []
    for task, (c, gap) in DATA[model].items():
        xs.append(c)
        ys.append(gap)
        labels.append(TASK_ABBREV[task])

    if style["marker"] == "x":
        ax.scatter(xs, ys, color=style["color"], marker=style["marker"],
                   s=style["size"], zorder=3, label=style["label"],
                   linewidths=style["lw"])
    else:
        ax.scatter(xs, ys, color=style["color"], marker=style["marker"],
                   s=style["size"], zorder=3, label=style["label"],
                   edgecolors="white", linewidths=0.6)

    for x, y, lbl in zip(xs, ys, labels):
        t = ax.text(x, y, lbl, color=style["color"], fontsize=8,
                    alpha=0.95, ha="center", va="center", zorder=4)
        texts.append(t)

# Run adjustText to resolve overlaps automatically.
adjust_text(
    texts,
    ax=ax,
    arrowprops=dict(arrowstyle="-", color="gray", alpha=0.6, lw=0.5),
    expand_points=EXPAND_POINTS,
)

# zero line
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)

# shaded regions
ax.axhspan(0, 0.78, alpha=0.04, color="steelblue")
ax.axhspan(-0.30, 0, alpha=0.06, color="firebrick")

# vertical separator (CAR threshold region)
ax.axvline(0.30, color="gray", linewidth=0.7, linestyle=":", alpha=0.6)
ax.text(0.31, 0.62, r"$\tau$=0.30", fontsize=7.5, color="gray", va="top")

# region labels
ax.text(0.04, 0.70, "Scaffold helps\n(Full > Direct)", fontsize=7,
        color="steelblue", alpha=0.85, va="top")
ax.text(0.55, -0.16, "Scaffold hurts\n(Direct > Full)", fontsize=7,
        color="firebrick", alpha=0.85, va="top")

ax.set_xlabel("Coverage estimate $\\hat{c}$ (calibration Direct EM)", fontsize=9)
ax.set_ylabel("Scaffold gap  (EM$_{\\mathrm{Full}}$ $-$ EM$_{\\mathrm{Direct}}$)", fontsize=9)
ax.set_xlim(-0.04, 0.94)
ax.set_ylim(-0.30, 0.78)
ax.legend(fontsize=7.5, loc="upper right", framealpha=0.92, ncol=1)
ax.grid(True, linestyle=":", alpha=0.3)
ax.tick_params(labelsize=8)

plt.tight_layout()
Path("figures").mkdir(exist_ok=True)
out_pdf = Path("figures/coverage_gap.pdf")
out_png = Path("figures/coverage_gap.png")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=300)
print(f"Saved -> {out_pdf}")
print(f"Saved -> {out_png}")
print(f"adjustText expand_points = {EXPAND_POINTS}")
