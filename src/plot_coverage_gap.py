"""
Figure 1: Scaffold-level gap (EM_Full - EM_Direct) vs. coverage estimate (ĉ).
Positive gap = scaffold helps; negative = scaffold hurts.
Saves to figures/coverage_gap.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── data ──────────────────────────────────────────────────────────────────────
# ĉ values: calibration-set Direct EM from routing_analysis.json / probe logs
# gap = EM_Full - EM_Direct (scaffold gap; negative = scaffold hurts)

DATA = {
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
}

MODEL_STYLE = {
    "14B": {"color": "#2166ac", "marker": "o", "label": "Qwen2.5-14B"},
    "32B": {"color": "#f4a582", "marker": "s", "label": "Qwen2.5-32B"},
    "V3":  {"color": "#d6604d", "marker": "^", "label": "DeepSeek-V3"},
}

TASK_ABBREV = {
    "GSM8K":    "GSM",
    "MATH-hard":"MATH",
    "HotpotQA": "HQA",
    "WebQ":     "WebQ",
    "TriviaQA": "TQA",
}

LABEL_OFFSET = {
    # (model, task): (dx, dy) in data coords
    ("14B", "GSM8K"):     (+0.01, +0.03),
    ("14B", "TriviaQA"):  (+0.01, -0.04),
    ("14B", "WebQ"):      (+0.01, -0.04),
    ("V3",  "GSM8K"):     (+0.01, +0.03),
    ("V3",  "MATH-hard"): (+0.01, +0.02),
}

# ── plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5.5, 3.8))

for model, style in MODEL_STYLE.items():
    xs, ys, labels = [], [], []
    for task, (c, gap) in DATA[model].items():
        xs.append(c)
        ys.append(gap)
        labels.append(TASK_ABBREV[task])
    ax.scatter(xs, ys, color=style["color"], marker=style["marker"],
               s=70, zorder=3, label=style["label"])
    for x, y, lbl in zip(xs, ys, labels):
        key = (model, [t for t, (c, g) in DATA[model].items() if c == x and g == y][0])
        dx, dy = LABEL_OFFSET.get(key, (+0.01, +0.015))
        ax.annotate(lbl, (x, y), xytext=(x + dx, y + dy),
                    fontsize=7, color=style["color"], va="center")

# zero line
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)

# shaded regions
ax.axhspan(0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 0.8,
           alpha=0.04, color="steelblue")
ax.axhspan(ax.get_ylim()[0] if ax.get_ylim()[0] < 0 else -0.3, 0,
           alpha=0.06, color="firebrick")

# vertical separator (CAR threshold region)
ax.axvline(0.30, color="gray", linewidth=0.7, linestyle=":", alpha=0.6)
ax.text(0.31, 0.55, r"$\tau$=0.30", fontsize=7.5, color="gray", va="top")

# region labels
ax.text(0.05, 0.68, "Scaffold helps\n(Full > Direct)", fontsize=7,
        color="steelblue", alpha=0.8, va="top")
ax.text(0.55, -0.08, "Scaffold hurts\n(Direct > Full)", fontsize=7,
        color="firebrick", alpha=0.8, va="top")

ax.set_xlabel("Coverage estimate $\\hat{c}$ (calibration Direct EM)", fontsize=9)
ax.set_ylabel("Scaffold gap  (EM$_{\\mathrm{Full}}$ $-$ EM$_{\\mathrm{Direct}}$)", fontsize=9)
ax.set_xlim(-0.02, 0.92)
ax.set_ylim(-0.30, 0.78)
ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
ax.grid(True, linestyle=":", alpha=0.3)
ax.tick_params(labelsize=8)

plt.tight_layout()
Path("figures").mkdir(exist_ok=True)
out = Path("figures/coverage_gap.pdf")
fig.savefig(out, bbox_inches="tight")
print(f"Saved -> {out}")
