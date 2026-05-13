"""
Bootstrap 95% CI for Thought-Gap and per-variant EM.

Multi-seed cells use CLUSTER bootstrap (resample question IDs, keeping all
seed observations for each resampled question) to respect within-question
correlation across seeds.  Single-seed cells use plain bootstrap.

Usage:
    python -m src.bootstrap_ci
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

LOGS = Path("logs")
N_BOOT = 2000
RNG_SEED = 0
DEFAULT_MAX_N = 100

# ---------------------------------------------------------------------------
# CONFIGS: (label, model_slug, dataset_tag, hotpotqa_type, n_file, seeds)
# ---------------------------------------------------------------------------
_S3 = [42, 7, 123]

CONFIGS = [
    # HotpotQA full
    ("7B",  "Qwen_Qwen2.5-7B-Instruct",  "hotpotqa", None, 100, [42]),
    ("14B", "Qwen_Qwen2.5-14B-Instruct", "hotpotqa", None, 100, _S3),
    ("32B", "Qwen_Qwen2.5-32B-Instruct", "hotpotqa", None, 100, _S3),
    ("V3",  "deepseek-chat",              "hotpotqa", None, 100, _S3),

    # HotpotQA comparison-only
    ("14B", "Qwen_Qwen2.5-14B-Instruct", "hotpotqa_comparison", None, 100, [42]),
    ("32B", "Qwen_Qwen2.5-32B-Instruct", "hotpotqa_comparison", None, 100, [42]),
    ("V3",  "deepseek-chat",             "hotpotqa_comparison", None, 100, [42]),

    # GSM8K
    ("7B",  "Qwen_Qwen2.5-7B-Instruct",  "gsm8k", None, 100, [42]),
    ("14B", "Qwen_Qwen2.5-14B-Instruct", "gsm8k", None, 100, _S3),
    ("32B", "Qwen_Qwen2.5-32B-Instruct", "gsm8k", None, 100, _S3),
    ("V3",  "deepseek-chat",             "gsm8k", None, 100, _S3),

    # WebQuestions
    ("7B",  "Qwen_Qwen2.5-7B-Instruct",  "webquestions", None, 100, [42]),
    ("14B", "Qwen_Qwen2.5-14B-Instruct", "webquestions", None, 200, _S3),
    ("32B", "Qwen_Qwen2.5-32B-Instruct", "webquestions", None, 200, _S3),
    ("V3",  "deepseek-chat",             "webquestions", None, 200, _S3),

    # TriviaQA
    ("14B", "Qwen_Qwen2.5-14B-Instruct", "triviaqa", None, 100, _S3),
    ("32B", "Qwen_Qwen2.5-32B-Instruct", "triviaqa", None, 200, _S3),
    ("V3",  "deepseek-chat",             "triviaqa", None, 100, _S3),

    # MATH hard (14B/32B/V3: all 3 seeds; V3 补做 2026-05-12)
    ("14B", "Qwen_Qwen2.5-14B-Instruct", "math_hard", None, 100, _S3),
    ("32B", "Qwen_Qwen2.5-32B-Instruct", "math_hard", None, 100, _S3),
    ("V3",  "deepseek-chat",             "math_hard", None, 100, _S3),

    # Llama-3.1-70B (cross-family, Stage 4 — 3 seeds × 5 tasks)
    ("L70B", "meta-llama_llama-3.1-70b-instruct", "gsm8k",        None, 100, _S3),
    ("L70B", "meta-llama_llama-3.1-70b-instruct", "math_hard",    None, 100, _S3),
    ("L70B", "meta-llama_llama-3.1-70b-instruct", "hotpotqa",     None, 100, _S3),
    ("L70B", "meta-llama_llama-3.1-70b-instruct", "webquestions", None, 100, _S3),
    ("L70B", "meta-llama_llama-3.1-70b-instruct", "triviaqa",     None, 100, _S3),
]

V3_HOTPOTQA_SLUG = "deepseek-chat"


def _n_for_seed(seed: int, n_file: int) -> int:
    return n_file if seed == 42 else min(n_file, 100)


def _log_path(variant: str, model_slug: str, dataset: str, n: int = 100, seed: int = 42) -> Path:
    standard = LOGS / f"pilot_{variant}_{model_slug}_{dataset}_n{n}_seed{seed}.jsonl"
    # Legacy: V3 hotpotqa seed=42 was logged without the dataset suffix
    if model_slug == V3_HOTPOTQA_SLUG and dataset == "hotpotqa":
        legacy = LOGS / f"pilot_{variant}_{model_slug}_n{n}_seed{seed}.jsonl"
        if legacy.exists():
            return legacy
    return standard


def load_em_by_qid(variant: str, model_slug: str, dataset: str,
                   n_file: int, seeds: list[int]) -> dict[str, list[int]]:
    """
    Returns {qid: [em_seed1, em_seed2, ...]} — one EM entry per seed that loaded.
    Used for cluster bootstrap.
    """
    result: dict[str, list[int]] = {}
    for s in seeds:
        n = _n_for_seed(s, n_file)
        path = _log_path(variant, model_slug, dataset, n=n, seed=s)
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                qid = obj["id"]
                result.setdefault(qid, []).append(int(obj.get("em", 0)))
    return result


# ---------------------------------------------------------------------------
# Bootstrap functions
# ---------------------------------------------------------------------------

def _plain_bootstrap_gap(a: np.ndarray, b: np.ndarray,
                         rng: np.random.Generator) -> tuple[float, float, float]:
    """Paired bootstrap CI for mean(a) - mean(b) (single-seed path)."""
    n = len(a)
    obs = float(a.mean() - b.mean())
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        diffs[i] = a[idx].mean() - b[idx].mean()
    return obs, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _plain_bootstrap_mean(a: np.ndarray,
                          rng: np.random.Generator) -> tuple[float, float, float]:
    n = len(a)
    obs = float(a.mean())
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        means[i] = a[idx].mean()
    return obs, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _cluster_bootstrap_gap(full_by_qid: dict[str, list[int]],
                            none_by_qid: dict[str, list[int]],
                            rng: np.random.Generator) -> tuple[float, float, float]:
    """
    Cluster bootstrap CI for Gap, resampling QUESTION IDs.
    Respects within-question correlation across seeds.
    """
    qids = sorted(set(full_by_qid) & set(none_by_qid))
    n_q = len(qids)
    if n_q == 0:
        return 0.0, 0.0, 0.0

    all_full = [v for qid in qids for v in full_by_qid[qid]]
    all_none = [v for qid in qids for v in none_by_qid[qid]]
    obs = float(np.mean(all_full) - np.mean(all_none))

    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sampled = rng.integers(0, n_q, size=n_q)
        s_full = [v for j in sampled for v in full_by_qid[qids[j]]]
        s_none = [v for j in sampled for v in none_by_qid[qids[j]]]
        diffs[i] = np.mean(s_full) - np.mean(s_none)
    return obs, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _cluster_bootstrap_mean(by_qid: dict[str, list[int]],
                             rng: np.random.Generator) -> tuple[float, float, float]:
    """Cluster bootstrap CI for mean EM, resampling question IDs."""
    qids = sorted(by_qid)
    n_q = len(qids)
    all_vals = [v for qid in qids for v in by_qid[qid]]
    obs = float(np.mean(all_vals))
    means = np.empty(N_BOOT)
    for i in range(N_BOOT):
        sampled = rng.integers(0, n_q, size=n_q)
        s_vals = [v for j in sampled for v in by_qid[qids[j]]]
        means[i] = np.mean(s_vals)
    return obs, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def fmt(val: float, lo: float, hi: float) -> str:
    sig = "*" if not (lo <= 0 <= hi) else ""
    return f"{val:+.3f} [{lo:+.3f},{hi:+.3f}]{sig}"


def fmt_em(val: float, lo: float, hi: float) -> str:
    return f"{val:.3f} [{lo:.3f},{hi:.3f}]"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(RNG_SEED)
    current_dataset = None

    for label, model_slug, dataset, _, n_file, seeds in CONFIGS:
        multi = len(seeds) > 1

        if dataset != current_dataset:
            current_dataset = dataset
            print(f"\n{'='*80}")
            print(f"  Dataset: {dataset}  [x3→cluster-bootstrap on qid; single→plain bootstrap]")
            print(f"{'='*80}")
            print(f"{'Model':<6} {'seeds':>6} {'full EM':>22} {'none EM':>22} {'comp EM':>22} "
                  f"{'Gap(full-none)':>26} {'Gap(comp-none)':>26} {'n_q':>6}")
            print("-" * 150)

        full_d  = load_em_by_qid("full",       model_slug, dataset, n_file, seeds)
        none_d  = load_em_by_qid("none",       model_slug, dataset, n_file, seeds)
        comp_d  = load_em_by_qid("compressed", model_slug, dataset, n_file, seeds)

        if not full_d and not none_d:
            print(f"{label:<6}  [missing data]")
            continue

        # common question IDs present in all three variants
        common_qids = sorted(set(full_d) & set(none_d) & set(comp_d))
        if not common_qids:
            common_qids = sorted(set(full_d) & set(none_d))
        if not common_qids:
            print(f"{label:<6}  [no common IDs]")
            continue

        # restrict dicts to common questions (keep all seed observations per qid)
        full_c = {q: full_d[q] for q in common_qids}
        none_c = {q: none_d[q] for q in common_qids}
        comp_c = {q: comp_d.get(q, []) for q in common_qids if comp_d.get(q)}
        # re-restrict to questions present in all three
        if comp_c:
            triple = sorted(set(full_c) & set(none_c) & set(comp_c))
            if triple:
                full_c = {q: full_c[q] for q in triple}
                none_c = {q: none_c[q] for q in triple}
                comp_c = {q: comp_c[q] for q in triple}

        n_q = len(full_c)
        seed_str = f"x{len(seeds)}" if multi else "42"

        if multi:
            f_em, f_lo, f_hi = _cluster_bootstrap_mean(full_c, rng)
            n_em, n_lo, n_hi = _cluster_bootstrap_mean(none_c, rng)
            c_em, c_lo, c_hi = _cluster_bootstrap_mean(comp_c, rng) if comp_c else (float('nan'),)*3
            gap_fn, gap_fn_lo, gap_fn_hi = _cluster_bootstrap_gap(full_c, none_c, rng)
            gap_cn, gap_cn_lo, gap_cn_hi = (_cluster_bootstrap_gap(comp_c, none_c, rng)
                                             if comp_c else (float('nan'),)*3)
        else:
            # single seed: plain bootstrap on flat arrays
            full_arr = np.array([v for vals in full_c.values() for v in vals], dtype=float)
            none_arr = np.array([v for vals in none_c.values() for v in vals], dtype=float)
            comp_arr = np.array([v for vals in comp_c.values() for v in vals], dtype=float) if comp_c else np.array([])
            f_em, f_lo, f_hi = _plain_bootstrap_mean(full_arr, rng)
            n_em, n_lo, n_hi = _plain_bootstrap_mean(none_arr, rng)
            c_em, c_lo, c_hi = (_plain_bootstrap_mean(comp_arr, rng)
                                 if len(comp_arr) else (float('nan'),)*3)
            gap_fn, gap_fn_lo, gap_fn_hi = _plain_bootstrap_gap(full_arr, none_arr, rng)
            gap_cn, gap_cn_lo, gap_cn_hi = (_plain_bootstrap_gap(comp_arr, none_arr, rng)
                                             if len(comp_arr) else (float('nan'),)*3)

        def _fmt_em(v, lo, hi):
            return "   —" if np.isnan(v) else fmt_em(v, lo, hi)
        def _fmt_gap(v, lo, hi):
            return "   —" if np.isnan(v) else fmt(v, lo, hi)

        print(
            f"{label:<6} {seed_str:>6} "
            f"{_fmt_em(f_em, f_lo, f_hi):>22} "
            f"{_fmt_em(n_em, n_lo, n_hi):>22} "
            f"{_fmt_em(c_em, c_lo, c_hi):>22} "
            f"{_fmt_gap(gap_fn, gap_fn_lo, gap_fn_hi):>26} "
            f"{_fmt_gap(gap_cn, gap_cn_lo, gap_cn_hi):>26} "
            f"{n_q:>6}"
        )

    print()
    print("* = CI excludes 0 (significant at 95% level)")
    print("x3 = cluster-bootstrap over question IDs (pooled across seeds 42/7/123)")
    print("     CI reflects both question-level and prompt-level variance correctly")


if __name__ == "__main__":
    main()
