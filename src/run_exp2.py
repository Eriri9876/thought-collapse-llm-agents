"""
Experiment 2 runner: cross-family Capacity Domination validation.

Sweeps:
  models      = {meta-llama/llama-3.1-8b-instruct, meta-llama/llama-3.1-70b-instruct}
  tasks       = {gsm8k, math_hard, hotpotqa, webquestions, triviaqa}
  conditions  = {full, none, direct}
  seeds       = {42, 7, 123}     (subset per stage)
  n           = 100

Conditions reuse Experiment 1 code paths verbatim:
  full / none → src.react.run_react   (with Llama-specific hard caps)
  direct      → src.probe._ask_direct (single-turn, no scaffold)

Hard caps for ReAct (Llama-only — avoids burning budget on Search→Search loops):
  max_steps               = 6
  max_tokens_per_call     = 256
  max_total_output_tokens = 1500
  temperature             = 0.0     (already the default in chat())

Cost protection:
  - Per-stage hard cap is set on cost_tracker before running.
  - Crossing it raises BudgetExceeded *after* the current sample's record has
    been flushed to the log file, so resumes pick up cleanly.
  - Each completed (model, task, condition, seed) cell appends one line to
    logs/experiment2_cost_log.jsonl.

Output schema is identical to Experiment 1 (logs/pilot_*.jsonl and
logs/probe_direct_*.jsonl), so bootstrap_ci.py et al. need no changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src import cost_tracker
from src.cost_tracker import BudgetExceeded
from src.data import get_samples
from src.llm import chat
from src.probe import DIRECT_SYSTEM
from src.react import run_react
from src.run import evaluate

sys.stdout.reconfigure(encoding="utf-8")

LOG_DIR = Path("logs")
COST_LOG_PATH = LOG_DIR / "experiment2_cost_log.jsonl"

LLAMA_MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "meta-llama/llama-3.1-70b-instruct",
]
TASKS = ["gsm8k", "math_hard", "hotpotqa", "webquestions", "triviaqa"]
CONDITIONS = ["full", "none", "direct"]

# ReAct hard caps for Llama runs (see module docstring).
LLAMA_REACT_KWARGS = dict(
    max_steps=6,
    max_tokens_per_call=256,
    max_total_output_tokens=1500,
)
# Direct probe only generates a final-answer string; 256 is plenty.
LLAMA_DIRECT_MAX_TOKENS = 256


def _model_slug(model: str) -> str:
    return model.replace("/", "_")


def _log_path(condition: str, model: str, task: str, n: int, seed: int) -> Path:
    slug = _model_slug(model)
    if condition == "direct":
        return LOG_DIR / f"probe_direct_{slug}_{task}_n{n}_seed{seed}.jsonl"
    # full / none ReAct → same naming convention as Experiment 1.
    return LOG_DIR / f"pilot_{condition}_{slug}_{task}_n{n}_seed{seed}.jsonl"


def _load_done_ids(path: Path) -> tuple[set, list]:
    done_ids: set = set()
    results: list = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done_ids.add(rec["id"])
                results.append(rec)
    return done_ids, results


def _append_cost_log(entry: dict) -> None:
    LOG_DIR.mkdir(exist_ok=True)
    with open(COST_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _ask_direct_llama(question: str, model: str) -> str | None:
    messages = [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user",   "content": f"Question: {question}"},
    ]
    return chat(messages, model=model, max_tokens=LLAMA_DIRECT_MAX_TOKENS)


def run_cell(
    *,
    stage: int,
    model: str,
    task: str,
    condition: str,
    seed: int,
    n: int,
) -> dict:
    """Run one (model, task, condition, seed) cell. Resumes from existing log.

    Returns a cost-log entry summarising what happened. Re-raises
    BudgetExceeded after flushing the current sample so the caller can
    persist state and exit cleanly."""
    assert condition in CONDITIONS, condition
    log_path = _log_path(condition, model, task, n, seed)
    samples = get_samples(n, seed=seed, dataset=task)
    done_ids, results = _load_done_ids(log_path)
    cost_before = cost_tracker.summary()["cost_usd"]
    print(
        f"\n=== stage{stage} | {model} | {task} | {condition} | seed={seed} | "
        f"resume={len(done_ids)}/{n} | cumulative=${cost_before:.4f} ==="
    )

    status_counts: dict[str, int] = {}
    parse_failed_n = 0
    output_budget_n = 0
    max_steps_n = 0
    aborted = False

    LOG_DIR.mkdir(exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        for i, sample in enumerate(samples):
            if sample["id"] in done_ids:
                continue
            try:
                t0 = time.time()
                if condition == "direct":
                    pred = _ask_direct_llama(sample["question"], model)
                    elapsed = round(time.time() - t0, 2)
                    metrics = evaluate(pred, sample["aliases"]) if pred else {"em": 0, "f1": 0.0}
                    record = {
                        "id":       sample["id"],
                        "question": sample["question"],
                        "gold":     sample["answer"],
                        "pred":     pred,
                        "elapsed":  elapsed,
                        **metrics,
                    }
                else:
                    result = run_react(
                        sample["question"],
                        variant=condition,
                        model=model,
                        **LLAMA_REACT_KWARGS,
                    )
                    elapsed = round(time.time() - t0, 2)
                    metrics = evaluate(result["answer"], sample["aliases"])
                    record = {
                        "id": sample["id"],
                        "question": sample["question"],
                        "gold": sample["answer"],
                        "pred": result["answer"],
                        "status": result["status"],
                        "steps": result["steps"],
                        "elapsed": elapsed,
                        **metrics,
                        "trajectory": result["trajectory"],
                    }
                    status_counts[result["status"]] = status_counts.get(result["status"], 0) + 1
                    if result["status"] == "output_budget_exceeded":
                        output_budget_n += 1
                    if result["status"] == "max_steps_reached":
                        max_steps_n += 1
                    if any("error" in t and t["error"] == "parse_failed"
                           for t in result["trajectory"]):
                        parse_failed_n += 1

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                results.append(record)
                pred_str = str(record.get("pred"))[:40]
                gold_str = str(sample["answer"])[:40]
                print(
                    f"  [{i+1}/{n}] EM={record['em']} "
                    f"pred={pred_str!r} gold={gold_str!r}"
                )
            except BudgetExceeded:
                aborted = True
                print(f"  [BUDGET HIT] last sample written, aborting cell")
                break

    em = round(sum(r["em"] for r in results) / max(1, len(results)), 4)
    f1 = round(sum(r["f1"] for r in results) / max(1, len(results)), 4)
    cost_after = cost_tracker.summary()
    delta_cost = round(cost_after["cost_usd"] - cost_before, 6)

    cost_entry = {
        "stage":          stage,
        "model":          model,
        "task":           task,
        "condition":      condition,
        "seed":           seed,
        "n_target":       n,
        "n_done":         len(results),
        "em":             em,
        "f1":             f1,
        "cost_usd_cell":  delta_cost,
        "cost_usd_cumul": round(cost_after["cost_usd"], 6),
        "in_tokens":      cost_after["in_tokens"],
        "out_tokens":     cost_after["out_tokens"],
        "status_counts":  status_counts,
        "parse_failed":   parse_failed_n,
        "output_budget":  output_budget_n,
        "max_steps":      max_steps_n,
        "aborted":        aborted,
        "log_path":       str(log_path),
    }
    _append_cost_log(cost_entry)
    print(
        f"  >> EM={em} F1={f1} cell=${delta_cost:.4f} "
        f"cumul=${cost_after['cost_usd']:.4f}"
        + (" [ABORTED]" if aborted else "")
    )
    if aborted:
        raise BudgetExceeded(
            f"stage {stage} budget exhausted at cell "
            f"{model}/{task}/{condition}/seed={seed}"
        )
    return cost_entry


def run_stage(*, stage: int, models: list[str], seeds: list[int],
              tasks: list[str], conditions: list[str], n: int,
              hard_cap_usd: float) -> None:
    cost_tracker.set_hard_cap(hard_cap_usd)
    print(f"=== STAGE {stage} START | hard_cap=${hard_cap_usd:.2f} | "
          f"models={models} | seeds={seeds} ===")
    try:
        for model in models:
            for task in tasks:
                for condition in conditions:
                    for seed in seeds:
                        run_cell(
                            stage=stage, model=model, task=task,
                            condition=condition, seed=seed, n=n,
                        )
    except BudgetExceeded as e:
        print(f"\n[STAGE {stage} ABORTED] {e}")
        sys.exit(2)
    finally:
        cost_tracker.print_summary()
    print(f"=== STAGE {stage} COMPLETE ===")


# Pre-baked stage configs matching the Experiment 2 plan.
# Hard caps are *cumulative-from-zero* on cost_tracker. The runner expects to
# be invoked one stage at a time in a single process — caching means re-runs
# are free. cost_tracker has process-singleton state, so a fresh process
# starts at $0, and the cap is the total budget we want this process to spend.
STAGE_CONFIGS: dict[int, dict] = {
    1: dict(models=["meta-llama/llama-3.1-8b-instruct"],
            seeds=[42], hard_cap_usd=3.0),
    2: dict(models=["meta-llama/llama-3.1-8b-instruct"],
            seeds=[7, 123], hard_cap_usd=2.0),
    3: dict(models=["meta-llama/llama-3.1-70b-instruct"],
            seeds=[42], hard_cap_usd=5.0),
    4: dict(models=["meta-llama/llama-3.1-70b-instruct"],
            seeds=[7, 123], hard_cap_usd=5.0),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2, 3, 4])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--tasks", nargs="+", default=TASKS)
    parser.add_argument("--conditions", nargs="+", default=CONDITIONS)
    parser.add_argument(
        "--cap-usd", type=float, default=None,
        help="Override the stage's default hard cap (USD).",
    )
    args = parser.parse_args()
    cfg = STAGE_CONFIGS[args.stage]
    cap = args.cap_usd if args.cap_usd is not None else cfg["hard_cap_usd"]
    run_stage(
        stage=args.stage,
        models=cfg["models"],
        seeds=cfg["seeds"],
        tasks=args.tasks,
        conditions=args.conditions,
        n=args.n,
        hard_cap_usd=cap,
    )


if __name__ == "__main__":
    main()
