"""Aggregate TCI v2 cells across seeds and emit summary tables + anomaly list.

Reads all results/tci_v2_*.json and produces:
  - per-cell (model, task) 3-seed mean ± SE for sim_mismatched and
    adv_follows_question_rate
  - cross-seed stability (max - min within cell)
  - anomalies: |new_seed - seed42| > 0.05; follows_q < 0.50; max-min > 0.15
"""
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

RESULTS = Path("results")
SEEDS_3 = [42, 7, 123]

MODEL_LABELS = {
    "Qwen_Qwen2.5-14B-Instruct":          ("14B",  3),
    "Qwen_Qwen2.5-32B-Instruct":          ("32B",  3),
    "deepseek-chat":                      ("V3",   3),
    "meta-llama_llama-3.1-70b-instruct":  ("L70B", 3),
    "meta-llama_llama-3.1-8b-instruct":   ("L8B",  1),
}
TASKS = ["gsm8k", "math_hard", "hotpotqa", "webquestions", "triviaqa"]

# regex: tci_v2_<slug>_<task>_n<N>_seed<S>.json
PAT = re.compile(r"tci_v2_(.+?)_(gsm8k|math_hard|hotpotqa|webquestions|triviaqa)_n(\d+)_seed(\d+)\.json$")


def load_all() -> dict:
    """Return {(model_label, task, seed): {n, sim_mismatched, adv_follows_q_rate, ...}}."""
    out = {}
    for fp in sorted(RESULTS.glob("tci_v2_*.json")):
        m = PAT.search(fp.name)
        if not m:
            continue
        slug, task, n, seed = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        if slug not in MODEL_LABELS:
            continue
        label, _ = MODEL_LABELS[slug]
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        out[(label, task, seed)] = {
            "n":                          n,
            "sim_mismatched":             d["sim_mismatched"],
            "sim_scrambled":              d["sim_scrambled"],
            "sim_empty":                  d["sim_empty"],
            "adv_follows_q_rate":         d["adv_follows_question_rate"],
            "adv_sim_to_question":        d["adv_sim_to_question"],
            "adv_sim_to_misdirection":    d["adv_sim_to_misdirection"],
            "file":                       fp.name,
        }
    return out


def mean_se(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, float("nan")
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    se = math.sqrt(var / n)
    return m, se


def main():
    cells = load_all()
    print(f"Loaded {len(cells)} cell-results from {RESULTS}")

    # ── per (model, task) aggregation ──────────────────────────────────────
    rows = []
    anomalies = []
    for label, n_seed in sorted(set((lab, ns) for lab, ns in MODEL_LABELS.values()),
                                key=lambda x: ["L8B", "14B", "32B", "V3", "L70B"].index(x[0])):
        for task in TASKS:
            mis = []
            fq  = []
            sw  = []
            ns  = []
            seed_list = SEEDS_3 if n_seed == 3 else [42]
            seeds_present = []
            for s in seed_list:
                if (label, task, s) in cells:
                    c = cells[(label, task, s)]
                    mis.append(c["sim_mismatched"])
                    fq.append(c["adv_follows_q_rate"])
                    sw.append(c["adv_sim_to_misdirection"])
                    ns.append(c["n"])
                    seeds_present.append(s)
            if not mis:
                continue
            m_mis, se_mis = mean_se(mis)
            m_fq, se_fq   = mean_se(fq)
            m_sw, se_sw   = mean_se(sw)
            spread_mis = max(mis) - min(mis) if len(mis) > 1 else 0.0
            spread_fq  = max(fq)  - min(fq)  if len(fq)  > 1 else 0.0
            rows.append({
                "model":       label,
                "task":        task,
                "k_seeds":     len(seeds_present),
                "seeds":       seeds_present,
                "n_total":     sum(ns),
                "mismatched":  (m_mis, se_mis, spread_mis),
                "follows_q":   (m_fq,  se_fq,  spread_fq),
                "sim_wrong":   (m_sw,  se_sw),
                "per_seed_mismatched": dict(zip(seeds_present, mis)),
                "per_seed_followsq":   dict(zip(seeds_present, fq)),
            })

            # anomaly checks
            if 42 in seeds_present and len(seeds_present) > 1:
                v42 = cells[(label, task, 42)]["sim_mismatched"]
                for s, v in zip(seeds_present, mis):
                    if s == 42:
                        continue
                    if abs(v - v42) > 0.05:
                        anomalies.append(
                            f"  [DEVIATION] {label} × {task} seed{s} sim_mismatched={v:.3f} "
                            f"vs seed42={v42:.3f} (Δ={v - v42:+.3f})"
                        )
            if m_fq < 0.50:
                anomalies.append(
                    f"  [LOW follows_q] {label} × {task} mean={m_fq:.3f} "
                    f"(seeds: " + ", ".join(f"s{s}={v:.2f}" for s, v in zip(seeds_present, fq)) + ")"
                )
            if spread_mis > 0.15:
                anomalies.append(
                    f"  [UNSTABLE sim_mismatched] {label} × {task} spread={spread_mis:.3f} "
                    f"(values: " + ", ".join(f"s{s}={v:.3f}" for s, v in zip(seeds_present, mis)) + ")"
                )
            if spread_fq > 0.15:
                anomalies.append(
                    f"  [UNSTABLE follows_q] {label} × {task} spread={spread_fq:.3f} "
                    f"(values: " + ", ".join(f"s{s}={v:.2f}" for s, v in zip(seeds_present, fq)) + ")"
                )

    # ── output ────────────────────────────────────────────────────────────
    print("\n" + "=" * 102)
    print("TCI v2 AGGREGATED — 25 (model, task) cells; multi-seed cells show mean ± SE across seeds")
    print("=" * 102)
    print(f"  {'model':<6} {'task':<14} {'k':>3} {'n_tot':>5}  "
          f"{'sim_mismatched (mean±SE)':<26}  {'follows_q% (mean±SE)':<22}  "
          f"{'sim_wrong (mean)':<16}  spread_mis  spread_fq")
    print("  " + "-" * 100)
    for r in rows:
        m_mis, se_mis, sp_mis = r["mismatched"]
        m_fq,  se_fq,  sp_fq  = r["follows_q"]
        m_sw,  se_sw          = r["sim_wrong"]
        mis_txt = f"{m_mis:.3f}±{se_mis:.3f}" if r["k_seeds"] > 1 else f"{m_mis:.3f}  (single)"
        fq_txt  = f"{m_fq*100:.1f}%±{se_fq*100:.1f}%" if r["k_seeds"] > 1 else f"{m_fq*100:.1f}%  (single)"
        print(f"  {r['model']:<6} {r['task']:<14} {r['k_seeds']:>3} {r['n_total']:>5}  "
              f"{mis_txt:<26}  {fq_txt:<22}  {m_sw:<16.3f}  "
              f"{sp_mis:>9.3f}  {sp_fq:>9.3f}")

    print("\n" + "=" * 102)
    print(f"ANOMALIES ({len(anomalies)})")
    print("=" * 102)
    if not anomalies:
        print("  (none)")
    else:
        for a in anomalies:
            print(a)

    # ── persist as JSON for downstream ─────────────────────────────────────
    out_path = RESULTS / "tci_v2_aggregated.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "cells_total": len(cells),
            "rows":        rows,
            "anomalies":   anomalies,
        }, f, ensure_ascii=False, indent=2, default=lambda o: list(o) if isinstance(o, dict) else str(o))
    print(f"\n  saved → {out_path}")


if __name__ == "__main__":
    main()
