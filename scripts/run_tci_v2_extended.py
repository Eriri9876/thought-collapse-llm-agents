"""Stage runner for TCI v2 multi-seed expansion (50 new cells).

Monkey-patches src.tci.MODELS to add Llama-3.1-8B/70B, then invokes the
unchanged run_tci_cell() for each (model, task, seed). All single-step
Action semantics preserved; no perturbed-rollout / final-EM extension.

Usage:
    python -m scripts.run_tci_v2_extended --stage 1
    python -m scripts.run_tci_v2_extended --stage 2
    ...

Stage 1: L8B × 5 task × seed=42                       (5 cells, ~$0.01)
Stage 2: Qwen-14B × 5 task × {seed 7, 123}            (10 cells, ~$0.10)
Stage 3: Qwen-32B × 5 task × {seed 7, 123}            (10 cells, ~$0.17)
Stage 4: DeepSeek-V3 × 5 task × {seed 7, 123}         (10 cells, ~$0.29)
Stage 5: L70B × 5 task × {seed 42, 7, 123}            (15 cells, ~$0.59)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import src.tci as tci
from src import cost_tracker

# ── extend MODELS dict in-place (no edit to src/tci.py) ─────────────────────
tci.MODELS["L8B"]  = ("meta-llama/llama-3.1-8b-instruct",  "meta-llama_llama-3.1-8b-instruct")
tci.MODELS["L70B"] = ("meta-llama/llama-3.1-70b-instruct", "meta-llama_llama-3.1-70b-instruct")

TASKS_5 = ["gsm8k", "math_hard", "hotpotqa", "webquestions", "triviaqa"]

# Stage definition: ordered list of (model_key, task, seed). Stages run cheap → expensive.
STAGES: dict[int, list[tuple[str, str, int]]] = {
    1: [("L8B", t, 42) for t in TASKS_5],
    2: [(m, t, s) for m in ["14B"]          for t in TASKS_5 for s in [7, 123]],
    3: [(m, t, s) for m in ["32B"]          for t in TASKS_5 for s in [7, 123]],
    4: [(m, t, s) for m in ["V3"]           for t in TASKS_5 for s in [7, 123]],
    5: [(m, t, s) for m in ["L70B"]         for t in TASKS_5 for s in [42, 7, 123]],
}


def _expected_outfile(model_key: str, task: str, seed: int) -> Path:
    _, slug = tci.MODELS[model_key]
    # Filename uses {actual n} which we cannot know upfront — match by prefix/suffix
    return tci.OUT_DIR / f"tci_v2_{slug}_{task}_n*_seed{seed}.json"


def _existing_match(model_key: str, task: str, seed: int) -> Path | None:
    _, slug = tci.MODELS[model_key]
    matches = sorted(tci.OUT_DIR.glob(f"tci_v2_{slug}_{task}_n*_seed{seed}.json"))
    return matches[0] if matches else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True, choices=list(STAGES))
    ap.add_argument("--n",     type=int, default=50,
                    help="Cap on TCI sample size per cell (default 50, "
                         "matching existing seed=42 protocol). Actual n "
                         "may be smaller after first-step filter.")
    ap.add_argument("--cap",   type=float, default=3.0,
                    help="Hard cumulative USD cap (default 3.0). Crossing aborts.")
    args = ap.parse_args()

    cost_tracker.set_hard_cap(args.cap)
    print(f"[runner] Stage {args.stage}: {len(STAGES[args.stage])} cells, hard cap ${args.cap:.2f}")
    print(f"[runner] cells: {STAGES[args.stage]}")

    completed: list[tuple[str, str, int, Path | None]] = []
    skipped:   list[tuple[str, str, int, str]] = []
    failed:    list[tuple[str, str, int, str]] = []

    t0 = time.time()
    for model_key, task, seed in STAGES[args.stage]:
        existing = _existing_match(model_key, task, seed)
        if existing is not None:
            print(f"\n[runner] SKIP existing: {existing.name}")
            skipped.append((model_key, task, seed, existing.name))
            continue

        print(f"\n[runner] ▶ {model_key} × {task} × seed={seed}")
        try:
            result = tci.run_tci_cell(model_key, task, n=args.n, seed=seed)
        except cost_tracker.BudgetExceeded as e:
            print(f"[runner] BUDGET CAP HIT: {e}")
            failed.append((model_key, task, seed, f"BudgetExceeded: {e}"))
            break
        except Exception as e:
            print(f"[runner] CELL FAILED ({type(e).__name__}): {e}")
            failed.append((model_key, task, seed, f"{type(e).__name__}: {e}"))
            continue

        if result is None:
            failed.append((model_key, task, seed, "run_tci_cell returned None"))
            continue

        out = _existing_match(model_key, task, seed)
        completed.append((model_key, task, seed, out))

    dt = time.time() - t0
    snap = cost_tracker.summary()

    print("\n" + "=" * 72)
    print(f"Stage {args.stage} done in {dt/60:.1f} min")
    print("=" * 72)
    print(f"completed: {len(completed)}  skipped: {len(skipped)}  failed: {len(failed)}")
    print(f"cumulative spend: ${snap['cost_usd']:.4f}  "
          f"in_tok={snap['in_tokens']}  out_tok={snap['out_tokens']}")
    if failed:
        print("\nFAILED CELLS:")
        for m, t, s, msg in failed:
            print(f"  {m} × {t} × seed={s}: {msg}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
