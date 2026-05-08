"""
InfiniGram-channel parallel of :mod:`correlation_min_vs_head`.

For each (model, task) cell we read the multi-entity InfiniGram log
file (``infinigram_min_<task>_n50_seed42.jsonl``) and correlate
head_count / min_count against:

  - Direct EM      (single seed, n=100)
  - Thought-Gap sign ∈ {-1, 0, +1}, mean over seeds {42, 7, 123}

Statistics per (signal × target) pair:
  - Pearson r + 95 % bootstrap CI (2000 reps, by qid) + two-sided p
  - Spearman ρ + 95 % bootstrap CI + p
  - For thought_gap_sign: ordered-logit coefficient + p
    (statsmodels OrderedModel)

Mirror of the pageview-side analysis so the two channels are
directly comparable. Writes::

    experiments/coverage_signals/outputs/correlation_infinigram_min_vs_head.csv

Usage::

    python -m src.coverage_signals.correlation_infinigram_min_vs_head
"""
from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

LOG_DIR = Path("logs")
SIG_DIR = Path("experiments/coverage_signals/outputs")
OUT_CSV = SIG_DIR / "correlation_infinigram_min_vs_head.csv"

MODELS = {
    "14B": "Qwen_Qwen2.5-14B-Instruct",
    "32B": "Qwen_Qwen2.5-32B-Instruct",
    "V3":  "deepseek-chat",
}
TASKS = ["webquestions", "triviaqa", "hotpotqa", "gsm8k", "math_hard"]
SEEDS = [42, 7, 123]
N_BOOT = 2000


# ─────────────────────────────────────────────────────────────────────────────
# IO
# ─────────────────────────────────────────────────────────────────────────────

def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def load_signal(task: str, n: int = 50, seed: int = 42) -> dict[str, dict]:
    """qid → {head_ct, min_ct, max_ct, mean_ct, n_ents}"""
    pv = _jsonl(SIG_DIR / f"infinigram_min_{task}_n{n}_seed{seed}.jsonl")
    return {r["id"]: {
        "head_ct": r.get("head_count"),
        "min_ct":  r.get("min_count"),
        "max_ct":  r.get("max_count"),
        "mean_ct": r.get("mean_count"),
        "n_ents":  r.get("n_entities"),
    } for r in pv}


def load_direct_em(model_slug: str, task: str,
                   n: int = 100, seed: int = 42) -> dict[str, int]:
    p = LOG_DIR / f"probe_direct_{model_slug}_{task}_n{n}_seed{seed}.jsonl"
    return {r["id"]: int(r["em"]) for r in _jsonl(p)}


def _pilot_path(variant: str, model_slug: str, task: str,
                n: int, seed: int) -> Path:
    p = LOG_DIR / f"pilot_{variant}_{model_slug}_{task}_n{n}_seed{seed}.jsonl"
    if not p.exists() and task == "hotpotqa" and model_slug == "deepseek-chat":
        p = LOG_DIR / f"pilot_{variant}_{model_slug}_n{n}_seed{seed}.jsonl"
    return p


def load_pilot_em(variant: str, model_slug: str, task: str,
                  seeds: list[int]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for seed in seeds:
        for n in (200, 100):
            p = _pilot_path(variant, model_slug, task, n, seed)
            if p.exists():
                for r in _jsonl(p):
                    out.setdefault(r["id"], []).append(int(r["em"]))
                break
    return out


def thought_gap_sign(qid: str,
                     em_full: dict[str, list[int]],
                     em_none: dict[str, list[int]]) -> int | None:
    ef = em_full.get(qid)
    en = em_none.get(qid)
    if not ef or not en:
        return None
    delta = (sum(ef) / len(ef)) - (sum(en) / len(en))
    if delta > 0:
        return +1
    elif delta < 0:
        return -1
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

def _pearson(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    if len(x) < 5 or np.std(x) == 0 or np.std(y) == 0:
        return None
    r, p = stats.pearsonr(x, y)
    return float(r), float(p)


def _spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float] | None:
    if len(x) < 5:
        return None
    res = stats.spearmanr(x, y)
    if np.isnan(res.correlation):
        return None
    return float(res.correlation), float(res.pvalue)


def _boot_ci(x: np.ndarray, y: np.ndarray, fn, n_boot: int = N_BOOT,
             rng_seed: int = 7) -> tuple[float, float] | None:
    rng = np.random.default_rng(rng_seed)
    n = len(x)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bx, by = x[idx], y[idx]
        r = fn(bx, by)
        if r is not None:
            out.append(r[0])
    if not out:
        return None
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def _ordered_logit(signal: np.ndarray, target: np.ndarray):
    try:
        from statsmodels.miscmodels.ordinal_model import OrderedModel
        if len(np.unique(target)) < 2:
            return None
        t_arr = np.asarray(target, dtype=int)
        x_arr = np.asarray(signal, dtype=float).reshape(-1, 1)
        mod = OrderedModel(t_arr, x_arr, distr="logit")
        res = mod.fit(method="bfgs", disp=False)
        coef = float(res.params[0])
        pval = float(res.pvalues[0])
        return coef, pval
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-cell loop
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_FIELDS = {"head": "head_ct", "min": "min_ct"}


def analyze_cell(model_key: str, model_slug: str, task: str,
                 signal_data: dict[str, dict],
                 em_full: dict, em_none: dict,
                 em_direct: dict) -> list[dict]:
    rows = []

    targets = {
        "direct_em":         lambda qid: em_direct.get(qid),
        "thought_gap_sign":  lambda qid: thought_gap_sign(qid, em_full, em_none),
    }

    for sig_name, sig_field in SIGNAL_FIELDS.items():
        for tname, getter in targets.items():
            x_list, y_list = [], []
            for qid, sigs in signal_data.items():
                sv = sigs[sig_field]
                if sv is None:
                    continue
                tv = getter(qid)
                if tv is None:
                    continue
                x_list.append(sv)
                y_list.append(tv)
            n = len(x_list)
            base = {"model": model_key, "task": task, "signal": sig_name,
                    "target": tname, "n": n}
            if n < 5:
                rows.append(base)
                continue

            x = np.asarray(x_list, dtype=float)
            y = np.asarray(y_list, dtype=float)

            pe = _pearson(x, y)
            sp = _spearman(x, y)
            pe_ci = _boot_ci(x, y, _pearson) if pe is not None else None
            sp_ci = _boot_ci(x, y, _spearman) if sp is not None else None

            row = {**base}
            if pe is not None:
                row.update({
                    "pearson_r":      round(pe[0], 4),
                    "pearson_p":      round(pe[1], 4),
                })
                if pe_ci is not None:
                    row["pearson_ci_low"]  = round(pe_ci[0], 4)
                    row["pearson_ci_high"] = round(pe_ci[1], 4)
            if sp is not None:
                row.update({
                    "spearman_r":     round(sp[0], 4),
                    "spearman_p":     round(sp[1], 4),
                })
                if sp_ci is not None:
                    row["spearman_ci_low"]  = round(sp_ci[0], 4)
                    row["spearman_ci_high"] = round(sp_ci[1], 4)

            if tname == "thought_gap_sign":
                ol = _ordered_logit(x, y)
                if ol is not None:
                    row["ordered_logit_coef"] = round(ol[0], 4)
                    row["ordered_logit_p"]    = round(ol[1], 4)

            rows.append(row)

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

CSV_FIELDS = [
    "model", "task", "signal", "target", "n",
    "pearson_r", "pearson_ci_low", "pearson_ci_high", "pearson_p",
    "spearman_r", "spearman_ci_low", "spearman_ci_high", "spearman_p",
    "ordered_logit_coef", "ordered_logit_p",
]


def main() -> None:
    all_rows: list[dict] = []
    for task in TASKS:
        sig = load_signal(task)
        if not sig:
            print(f"  [{task}] no infinigram_min file; skip")
            continue
        for mkey, mslug in MODELS.items():
            em_direct = load_direct_em(mslug, task)
            em_full   = load_pilot_em("full", mslug, task, SEEDS)
            em_none   = load_pilot_em("none", mslug, task, SEEDS)
            cell_rows = analyze_cell(mkey, mslug, task, sig,
                                     em_full, em_none, em_direct)
            all_rows.extend(cell_rows)

    SIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"\n  saved → {OUT_CSV}\n")

    print_table(all_rows)


def print_table(rows: list[dict]) -> None:
    print("=" * 110)
    print("  per-cell correlation: head_count / min_count vs target  (InfiniGram)")
    print("=" * 110)

    targets = ["direct_em", "thought_gap_sign"]
    for tgt in targets:
        print(f"\n--- Target: {tgt} ---")
        print(f"  {'task':<14} {'model':<5} {'sig':<5} {'n':>4} "
              f"{'r_p':>7} {'CI_p':>16} {'p_p':>7} "
              f"{'r_s':>7} {'CI_s':>16} {'p_s':>7}", end="")
        if tgt == "thought_gap_sign":
            print(f" {'olr':>7} {'p_olr':>7}")
        else:
            print()
        for r in rows:
            if r["target"] != tgt:
                continue
            if r.get("pearson_r") is None:
                print(f"  {r['task']:<14} {r['model']:<5} {r['signal']:<5} "
                      f"{r['n']:>4}  insufficient data")
                continue
            ci_p = (f"[{r['pearson_ci_low']:+.2f},{r['pearson_ci_high']:+.2f}]"
                    if 'pearson_ci_low' in r else " " * 16)
            ci_s = (f"[{r['spearman_ci_low']:+.2f},{r['spearman_ci_high']:+.2f}]"
                    if 'spearman_ci_low' in r else " " * 16)
            line = (f"  {r['task']:<14} {r['model']:<5} {r['signal']:<5} "
                    f"{r['n']:>4} {r['pearson_r']:>+7.3f} {ci_p:>16} "
                    f"{r['pearson_p']:>7.3f} {r['spearman_r']:>+7.3f} "
                    f"{ci_s:>16} {r['spearman_p']:>7.3f}")
            if tgt == "thought_gap_sign":
                if "ordered_logit_coef" in r:
                    line += (f" {r['ordered_logit_coef']:>+7.3f} "
                             f"{r['ordered_logit_p']:>7.3f}")
                else:
                    line += " " * 16
            print(line)


if __name__ == "__main__":
    main()
