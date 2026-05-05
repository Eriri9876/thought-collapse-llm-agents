"""
Token Cost Analysis

For each (model, task, variant) cell, compute:
  - avg steps per question
  - avg total tokens generated (proxy: whitespace-split words)
  - avg Thought tokens specifically (Full only)
  - EM score

Then derive per (model, task):
  - Thought overhead = avg_tokens(Full) - avg_tokens(None)
  - EM gain (Gap) = EM(Full) - EM(None)
  - Token efficiency = Gap / Thought_overhead  (EM gain per extra 100 tokens)

Produces a summary table + data for a Pareto scatter plot.
"""
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

LOG_DIR = Path("logs")

MODELS = {
    "7B":  "Qwen_Qwen2.5-7B-Instruct",
    "14B": "Qwen_Qwen2.5-14B-Instruct",
    "32B": "Qwen_Qwen2.5-32B-Instruct",
    "V3":  "deepseek-chat",
}
TASKS    = ["gsm8k", "hotpotqa", "webquestions"]
VARIANTS = ["full", "none", "compressed"]


def _count_tokens(text: str) -> int:
    """Rough token count: whitespace-split words (≈ 75% of BPE tokens for English)."""
    return len(text.split()) if text else 0


def _extract_thought_tokens(response: str) -> int:
    """Count tokens in the Thought portion of a response."""
    m = re.search(r"Thought:\s*(.*?)(?:\nAction:|$)", response, re.S | re.I)
    return _count_tokens(m.group(1)) if m else 0


def load_cell(model_slug: str, task: str, variant: str, seed: int = 42, n: int = 100):
    path = LOG_DIR / f"pilot_{variant}_{model_slug}_{task}_n{n}_seed{seed}.jsonl"
    if not path.exists() and task == "hotpotqa" and model_slug == "deepseek-chat":
        path = LOG_DIR / f"pilot_{variant}_{model_slug}_n{n}_seed{seed}.jsonl"
    if not path.exists():
        return None

    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    if not records:
        return None

    em_list, steps_list, total_tok_list, thought_tok_list = [], [], [], []

    for rec in records:
        em_list.append(rec.get("em", 0))
        steps_list.append(rec.get("steps", 0))

        traj = rec.get("trajectory", [])
        total_tok = 0
        thought_tok = 0
        for step in traj:
            resp = step.get("response", "")
            total_tok   += _count_tokens(resp)
            thought_tok += _extract_thought_tokens(resp)
        total_tok_list.append(total_tok)
        thought_tok_list.append(thought_tok)

    n_rec = len(records)
    return {
        "em":          round(sum(em_list)          / n_rec, 4),
        "avg_steps":   round(sum(steps_list)        / n_rec, 2),
        "avg_total_tok": round(sum(total_tok_list)  / n_rec, 1),
        "avg_thought_tok": round(sum(thought_tok_list) / n_rec, 1),
        "n": n_rec,
    }


def main():
    seed = 42

    # ── collect all cells ────────────────────────────────────────────
    data = {}
    for mkey, mslug in MODELS.items():
        for task in TASKS:
            for variant in VARIANTS:
                cell = load_cell(mslug, task, variant, seed=seed)
                if cell:
                    data[(mkey, task, variant)] = cell

    # ── summary table ────────────────────────────────────────────────
    print("TOKEN COST ANALYSIS  (seed=42, n=100)")
    print("=" * 82)
    print(f"  {'Model':<6} {'Task':<14} {'Variant':<12} {'EM':>6} {'Steps':>7} "
          f"{'TotalTok':>10} {'ThoughtTok':>11}")
    print("  " + "-" * 70)
    for mkey in MODELS:
        for task in TASKS:
            for variant in VARIANTS:
                c = data.get((mkey, task, variant))
                if c:
                    th = f"{c['avg_thought_tok']:>11.0f}" if variant == "full" else f"{'—':>11}"
                    print(f"  {mkey:<6} {task:<14} {variant:<12} "
                          f"{c['em']:>6.3f} {c['avg_steps']:>7.1f} "
                          f"{c['avg_total_tok']:>10.0f} {th}")
            print()

    # ── Pareto analysis ──────────────────────────────────────────────
    print("\nPARETO ANALYSIS: Token Overhead vs EM Gain (Full vs None)")
    print("=" * 72)
    print(f"  {'Model':<6} {'Task':<14} {'Overhead(tok)':>14} {'Gap(EM)':>9} "
          f"{'Efficiency':>12} {'Verdict'}")
    print("  " + "-" * 68)

    pareto_rows = []
    for mkey in MODELS:
        for task in TASKS:
            full = data.get((mkey, task, "full"))
            none = data.get((mkey, task, "none"))
            if not full or not none:
                continue
            overhead = full["avg_total_tok"] - none["avg_total_tok"]
            gap      = round(full["em"] - none["em"], 4)
            # EM gain per 100 extra tokens
            efficiency = round(gap / overhead * 100, 4) if overhead > 5 else float("nan")

            if gap > 0.05:
                verdict = "Thought worthwhile"
            elif gap > 0:
                verdict = "Marginal gain"
            elif gap >= -0.02:
                verdict = "Break-even / noise"
            else:
                verdict = "Thought harmful"

            pareto_rows.append({
                "model": mkey, "task": task,
                "overhead": overhead, "gap": gap,
                "efficiency": efficiency, "verdict": verdict,
                "thought_tok": full["avg_thought_tok"],
            })
            eff_str = f"{efficiency:>12.4f}" if efficiency == efficiency else f"{'—':>12}"
            print(f"  {mkey:<6} {task:<14} {overhead:>14.0f} {gap:>+9.3f} "
                  f"{eff_str} {verdict}")

    # ── Thought token budget ─────────────────────────────────────────
    print("\nTHOUGHT TOKEN BUDGET  (Full variant only)")
    print("=" * 55)
    print(f"  {'Model':<6} {'Task':<14} {'Thought tok':>12} {'Total tok':>10} {'Thought%':>10}")
    print("  " + "-" * 48)
    for mkey in MODELS:
        for task in TASKS:
            c = data.get((mkey, task, "full"))
            if c and c["avg_total_tok"] > 0:
                pct = c["avg_thought_tok"] / c["avg_total_tok"] * 100
                print(f"  {mkey:<6} {task:<14} {c['avg_thought_tok']:>12.0f} "
                      f"{c['avg_total_tok']:>10.0f} {pct:>9.1f}%")

    # ── save for plotting ────────────────────────────────────────────
    out = Path("results") / "token_analysis.json"
    out.parent.mkdir(exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"cells": {str(k): v for k, v in data.items()},
                   "pareto": pareto_rows}, f, ensure_ascii=False, indent=2)
    print(f"\nData saved → {out}")


if __name__ == "__main__":
    main()
