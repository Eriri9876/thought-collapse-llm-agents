"""
Coverage-Aware Routing (CAR)

Design:
  For each (model, task):
    1. Estimate coverage = Direct Probe EM on first n_cal questions
    2. Route:
         coverage > theta  →  Direct mode  (skip Thought + tools)
         coverage <= theta →  Full ReAct
    3. Evaluate routed policy on held-out questions (indices n_cal..n)
    4. Compare: Routed vs Full vs None vs Direct

Retrospective simulation — uses existing logs, no new API calls.
Results saved to results/routing_analysis.json.
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

LOG_DIR = Path("logs")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

MODELS = {
    "14B":  "Qwen_Qwen2.5-14B-Instruct",
    "32B":  "Qwen_Qwen2.5-32B-Instruct",
    "V3":   "deepseek-chat",
    "L70B": "meta-llama_llama-3.1-70b-instruct",
}
TASKS = ["gsm8k", "hotpotqa", "webquestions", "triviaqa", "math_hard"]

DEFAULT_THRESHOLDS = [0.15, 0.20, 0.25, 0.30]
DEFAULT_N_CAL      = 20   # calibration set size
DEFAULT_N          = 100  # total questions per cell
DEFAULT_SEED       = 42


# ── data loading ──────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def load_direct(model_slug: str, task: str, n: int, seed: int) -> list[dict]:
    path = LOG_DIR / f"probe_direct_{model_slug}_{task}_n{n}_seed{seed}.jsonl"
    return _load_jsonl(path)


def load_react(model_slug: str, variant: str, task: str, n: int, seed: int) -> list[dict]:
    path = LOG_DIR / f"pilot_{variant}_{model_slug}_{task}_n{n}_seed{seed}.jsonl"
    if not path.exists() and task == "hotpotqa" and model_slug == "deepseek-chat":
        path = LOG_DIR / f"pilot_{variant}_{model_slug}_n{n}_seed{seed}.jsonl"
    return _load_jsonl(path)


def _count_tokens(text: str) -> int:
    return len(text.split()) if text else 0


def _traj_tokens(record: dict) -> int:
    total = 0
    for step in record.get("trajectory", []):
        total += _count_tokens(step.get("response", ""))
    return total


# ── align records by question id ──────────────────────────────────────────────

def align_by_id(
    direct: list[dict],
    full:   list[dict],
    none:   list[dict],
) -> list[dict]:
    """Return list of dicts with fields from all three sources, matched by id."""
    full_map  = {r["id"]: r for r in full}
    none_map  = {r["id"]: r for r in none}
    aligned   = []
    for d in direct:
        qid = d["id"]
        f   = full_map.get(qid)
        no  = none_map.get(qid)
        if f is None or no is None:
            continue
        aligned.append({
            "id":            qid,
            "question":      d["question"],
            # EM
            "em_direct":     d.get("em", 0),
            "em_full":       f.get("em", 0),
            "em_none":       no.get("em", 0),
            # token proxies
            "tok_direct":    _count_tokens(d.get("pred", "")),
            "tok_full":      _traj_tokens(f),
            "tok_none":      _traj_tokens(no),
        })
    return aligned


# ── routing simulation ─────────────────────────────────────────────────────────

def simulate_routing(
    aligned:  list[dict],
    theta:    float,
    n_cal:    int,
) -> dict:
    """
    Split aligned records into calibration and evaluation sets.
    Calibration: first n_cal items → estimate coverage.
    Evaluation: remaining items → apply routing, compute metrics.
    """
    if len(aligned) <= n_cal:
        return {}

    cal_set  = aligned[:n_cal]
    eval_set = aligned[n_cal:]

    # ── step 1: estimate coverage from calibration set ────────────────────────
    coverage_est = sum(r["em_direct"] for r in cal_set) / len(cal_set)

    # ── step 2: routing decision ──────────────────────────────────────────────
    route_to_direct = coverage_est > theta

    # ── step 3: evaluate routed policy on held-out set ───────────────────────
    n_eval = len(eval_set)

    em_routed  = []
    tok_routed = []
    for r in eval_set:
        if route_to_direct:
            em_routed.append(r["em_direct"])
            tok_routed.append(r["tok_direct"])
        else:
            em_routed.append(r["em_full"])
            tok_routed.append(r["tok_full"])

    avg_em_routed  = round(sum(em_routed)  / n_eval, 4)
    avg_tok_routed = round(sum(tok_routed) / n_eval, 1)

    # baselines on same eval set
    avg_em_full    = round(sum(r["em_full"]   for r in eval_set) / n_eval, 4)
    avg_em_none    = round(sum(r["em_none"]   for r in eval_set) / n_eval, 4)
    avg_em_direct  = round(sum(r["em_direct"] for r in eval_set) / n_eval, 4)
    avg_tok_full   = round(sum(r["tok_full"]  for r in eval_set) / n_eval, 1)
    avg_tok_direct = round(sum(r["tok_direct"] for r in eval_set) / n_eval, 1)

    # oracle: best of Full vs Direct per question
    oracle_em = round(sum(max(r["em_full"], r["em_direct"]) for r in eval_set) / n_eval, 4)

    # token savings vs always-Full
    tok_savings_abs = round(avg_tok_full - avg_tok_routed, 1)
    tok_savings_pct = round(tok_savings_abs / avg_tok_full * 100, 1) if avg_tok_full > 0 else 0.0

    return {
        "theta":             theta,
        "n_cal":             n_cal,
        "n_eval":            n_eval,
        "coverage_est":      round(coverage_est, 4),
        "route_to_direct":   route_to_direct,
        "routing_label":     "Direct" if route_to_direct else "Full-ReAct",
        # EM
        "em_routed":         avg_em_routed,
        "em_full":           avg_em_full,
        "em_none":           avg_em_none,
        "em_direct":         avg_em_direct,
        "em_oracle":         oracle_em,
        # delta vs Full
        "delta_vs_full":     round(avg_em_routed - avg_em_full, 4),
        # tokens
        "tok_routed":        avg_tok_routed,
        "tok_full":          avg_tok_full,
        "tok_direct":        avg_tok_direct,
        "tok_savings_abs":   tok_savings_abs,
        "tok_savings_pct":   tok_savings_pct,
    }


# ── leave-one-task-out τ selection ────────────────────────────────────────────

def run_loto(all_aligned: dict[str, list[dict]], candidates: list[float], n_cal: int) -> list[dict]:
    """
    Leave-one-task-out τ selection.

    For each held-out task T:
      - Select τ* = argmax_{τ} mean_over_{other tasks} delta_vs_full(τ)
      - Evaluate τ* on T
    Returns list of result dicts with 'selected_tau' and 'loto_task' fields.
    """
    results = []
    tasks = list(all_aligned.keys())
    for held_out in tasks:
        best_tau, best_score = candidates[0], -float("inf")
        for tau in candidates:
            scores = []
            for task, aligned in all_aligned.items():
                if task == held_out:
                    continue
                r = simulate_routing(aligned, tau, n_cal)
                if r:
                    scores.append(r["delta_vs_full"])
            if scores:
                avg = sum(scores) / len(scores)
                if avg > best_score:
                    best_score, best_tau = avg, tau
        r = simulate_routing(all_aligned[held_out], best_tau, n_cal)
        if r:
            r["selected_tau"] = best_tau
            r["loto_task"] = held_out
            results.append(r)
    return results


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",          type=int,   default=DEFAULT_N)
    parser.add_argument("--seed",       type=int,   default=DEFAULT_SEED)
    parser.add_argument("--n_cal",      type=int,   default=DEFAULT_N_CAL)
    parser.add_argument("--thresholds", nargs="+",  type=float,
                        default=DEFAULT_THRESHOLDS)
    args = parser.parse_args()

    all_results = []

    for mkey, mslug in MODELS.items():
        for task in TASKS:
            direct_recs = load_direct(mslug, task, args.n, args.seed)
            full_recs   = load_react(mslug, "full", task, args.n, args.seed)
            none_recs   = load_react(mslug, "none", task, args.n, args.seed)

            if not direct_recs or not full_recs or not none_recs:
                print(f"  [{mkey} × {task}] missing logs, skipping")
                continue

            aligned = align_by_id(direct_recs, full_recs, none_recs)
            if len(aligned) < args.n_cal + 10:
                print(f"  [{mkey} × {task}] only {len(aligned)} aligned records, skipping")
                continue

            for theta in args.thresholds:
                result = simulate_routing(aligned, theta, args.n_cal)
                if result:
                    result.update({"model": mkey, "task": task})
                    all_results.append(result)

    # ── print summary table (best theta per cell) ─────────────────────────────
    print("\nCOVERAGE-AWARE ROUTING — Best θ per Cell")
    print("=" * 88)
    print(f"  {'Model':<6} {'Task':<14} {'Coverage':>9} {'Route':>12} "
          f"{'Routed':>8} {'Full':>8} {'Direct':>8} {'Δ vs Full':>10} {'TokSave%':>10}")
    print("  " + "-" * 80)

    # pick best theta = highest routed EM per (model, task)
    best = {}
    for r in all_results:
        key = (r["model"], r["task"])
        if key not in best or r["em_routed"] > best[key]["em_routed"]:
            best[key] = r

    for mkey in MODELS:
        for task in TASKS:
            r = best.get((mkey, task))
            if not r:
                continue
            sign = "+" if r["delta_vs_full"] >= 0 else ""
            print(f"  {mkey:<6} {task:<14} "
                  f"{r['coverage_est']:>9.3f} "
                  f"{r['routing_label']:>12} "
                  f"{r['em_routed']:>8.3f} "
                  f"{r['em_full']:>8.3f} "
                  f"{r['em_direct']:>8.3f} "
                  f"{sign}{r['delta_vs_full']:>+9.3f} "
                  f"{r['tok_savings_pct']:>9.1f}%")
        print()

    # ── threshold sensitivity ─────────────────────────────────────────────────
    print("\nTHRESHOLD SENSITIVITY  (EM Routed by θ)")
    print("=" * 72)
    header_parts = [f"  {'Model':<6} {'Task':<14}"]
    for theta in args.thresholds:
        header_parts.append(f"  θ={theta:.2f}")
    print("".join(header_parts))
    print("  " + "-" * 60)

    for mkey in MODELS:
        for task in TASKS:
            row_results = {r["theta"]: r for r in all_results
                           if r["model"] == mkey and r["task"] == task}
            if not row_results:
                continue
            parts = [f"  {mkey:<6} {task:<14}"]
            for theta in args.thresholds:
                r = row_results.get(theta)
                if r:
                    marker = "*" if r["route_to_direct"] else " "
                    parts.append(f"  {r['em_routed']:.3f}{marker}")
                else:
                    parts.append("      —")
            print("".join(parts))
        print()
    print("  * = routed to Direct mode")

    # ── token savings summary ─────────────────────────────────────────────────
    print("\nTOKEN SAVINGS  (best θ, Routed vs Full ReAct)")
    print("=" * 60)
    print(f"  {'Model':<6} {'Task':<14} {'Saved(tok)':>12} {'Saved%':>8} {'Verdict'}")
    print("  " + "-" * 52)
    for mkey in MODELS:
        for task in TASKS:
            r = best.get((mkey, task))
            if not r:
                continue
            verdict = ("Direct mode saves tokens" if r["route_to_direct"]
                       else "Full ReAct kept (low coverage)")
            print(f"  {mkey:<6} {task:<14} "
                  f"{r['tok_savings_abs']:>12.0f} "
                  f"{r['tok_savings_pct']:>7.1f}% "
                  f"  {verdict}")

    # ── oracle gap ────────────────────────────────────────────────────────────
    print("\nORACLE GAP  (Routed vs Oracle = best-of-Full/Direct per question)")
    print("=" * 52)
    print(f"  {'Model':<6} {'Task':<14} {'Routed':>8} {'Oracle':>8} {'Gap':>8}")
    print("  " + "-" * 44)
    for mkey in MODELS:
        for task in TASKS:
            r = best.get((mkey, task))
            if not r:
                continue
            gap = round(r["em_routed"] - r["em_oracle"], 4)
            print(f"  {mkey:<6} {task:<14} "
                  f"{r['em_routed']:>8.3f} "
                  f"{r['em_oracle']:>8.3f} "
                  f"{gap:>+8.3f}")

    # ── leave-one-task-out τ selection ────────────────────────────────────────
    print("\n\nLEAVE-ONE-TASK-OUT τ SELECTION")
    print("=" * 88)
    print(f"  {'Model':<6} {'Held-out task':<16} {'τ*':>5} {'Routed':>8} {'Full':>8} "
          f"{'Δ vs Full':>10} {'TokSave%':>10}")
    print("  " + "-" * 68)

    loto_results = {}
    for mkey, mslug in MODELS.items():
        model_aligned = {}
        for task in TASKS:
            direct_recs = load_direct(mslug, task, args.n, args.seed)
            full_recs   = load_react(mslug, "full",  task, args.n, args.seed)
            none_recs   = load_react(mslug, "none",  task, args.n, args.seed)
            if not direct_recs or not full_recs or not none_recs:
                continue
            aligned = align_by_id(direct_recs, full_recs, none_recs)
            if len(aligned) >= args.n_cal + 10:
                model_aligned[task] = aligned

        if len(model_aligned) < 2:
            continue

        loto_rows = run_loto(model_aligned, args.thresholds, args.n_cal)
        loto_results[mkey] = loto_rows
        for r in loto_rows:
            sign = "+" if r["delta_vs_full"] >= 0 else ""
            print(f"  {mkey:<6} {r['loto_task']:<16} {r['selected_tau']:>5.2f} "
                  f"{r['em_routed']:>8.3f} {r['em_full']:>8.3f} "
                  f"{sign}{r['delta_vs_full']:>+9.3f} "
                  f"{r['tok_savings_pct']:>9.1f}%")
        print()

    # ── save ──────────────────────────────────────────────────────────────────
    out = OUT_DIR / "routing_analysis.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "results": all_results,
            "best": {str(k): v for k, v in best.items()},
            "loto": {m: rows for m, rows in loto_results.items()},
        }, f, ensure_ascii=False, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
