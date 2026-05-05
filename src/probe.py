"""
Answer-First Probe (Physical Layer Experiment)

Tests H2: "Strong models give the answer directly from memory;
Thought is post-hoc rationalization, not causal."

Method: ask the question with no tools and no ReAct loop.
If direct-answer EM ≈ full-ReAct EM on a task, the model already
"knew" the answer before any reasoning — Thought is decorative.
If direct-answer EM << full-ReAct EM, Thought + tools are genuinely
driving performance.

Expected pattern (Capacity Domination):
  WebQ  : direct ≈ full  for 32B/V3  (Thought is post-hoc)
  GSM8K : direct << full for all models (Thought is causal)
  HotpotQA: intermediate, model-dependent
"""
import json
import sys
import time
from pathlib import Path

from src.data import get_samples
from src.llm import chat
from src.run import evaluate

sys.stdout.reconfigure(encoding="utf-8")

LOG_DIR = Path("logs")

MODELS = {
    "14B": "Qwen/Qwen2.5-14B-Instruct",
    "32B": "Qwen/Qwen2.5-32B-Instruct",
    "V3":  "deepseek-chat",
}

DIRECT_SYSTEM = (
    "Answer the question as concisely as possible. "
    "Give only the final answer — no explanation, no steps."
)


def _ask_direct(question: str, model: str) -> str | None:
    messages = [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user",   "content": f"Question: {question}"},
    ]
    try:
        return chat(messages, model=model)
    except Exception as e:
        print(f"    [API error: {e}]")
        return None


def run_probe(
    model_key: str,
    task: str,
    n: int = 100,
    seed: int = 42,
):
    model_api = MODELS[model_key]
    model_slug = model_api.replace("/", "_")
    log_path = LOG_DIR / f"probe_direct_{model_slug}_{task}_n{n}_seed{seed}.jsonl"

    samples = get_samples(n, seed=seed, dataset=task)

    # resume support
    done_ids = set()
    results = []
    if log_path.exists():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done_ids.add(rec["id"])
                results.append(rec)
        if done_ids:
            print(f"  Resuming: {len(done_ids)} done, {n - len(done_ids)} remaining")

    print(f"\n=== Direct Probe: {model_key} × {task} (n={n}, seed={seed}) ===")
    with open(log_path, "a", encoding="utf-8") as f:
        for i, sample in enumerate(samples):
            if sample["id"] in done_ids:
                continue
            print(f"  [{i+1}/{n}] {sample['question'][:60]}...")
            t0 = time.time()
            raw = _ask_direct(sample["question"], model_api)
            elapsed = round(time.time() - t0, 2)

            metrics = evaluate(raw, sample["aliases"]) if raw else {"em": 0, "f1": 0.0}
            record = {
                "id":       sample["id"],
                "question": sample["question"],
                "gold":     sample["answer"],
                "pred":     raw,
                "elapsed":  elapsed,
                **metrics,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            results.append(record)
            print(f"    pred={str(raw)[:40]!r}  gold={sample['answer']!r}  EM={metrics['em']}")

    em = round(sum(r["em"] for r in results) / len(results), 4) if results else 0
    f1 = round(sum(r["f1"] for r in results) / len(results), 4) if results else 0
    print(f"  >> Direct EM={em}  F1={f1}  ({model_key} × {task})")
    return {"model": model_key, "task": task, "em": em, "f1": f1}


def _load_react_em(model_slug: str, variant: str, task: str, n: int, seed: int) -> float | None:
    path = LOG_DIR / f"pilot_{variant}_{model_slug}_{task}_n{n}_seed{seed}.jsonl"
    if not path.exists() and task == "hotpotqa" and model_slug == "deepseek-chat":
        path = LOG_DIR / f"pilot_{variant}_{model_slug}_n{n}_seed{seed}.jsonl"
    if not path.exists():
        return None
    records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return round(sum(r["em"] for r in records) / len(records), 4) if records else None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int,  default=100)
    parser.add_argument("--seed",   type=int,  default=42)
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--tasks",  nargs="+", default=["gsm8k", "hotpotqa", "webquestions"])
    args = parser.parse_args()

    probe_results = []
    for model_key in args.models:
        for task in args.tasks:
            r = run_probe(model_key, task, n=args.n, seed=args.seed)
            probe_results.append(r)

    # comparison table
    print("\n" + "=" * 70)
    print("ANSWER-FIRST PROBE  vs  REACT FULL/NONE  (EM, seed=42)")
    print("=" * 70)
    slugs = {"14B": "Qwen_Qwen2.5-14B-Instruct", "32B": "Qwen_Qwen2.5-32B-Instruct", "V3": "deepseek-chat"}
    header = f"  {'Model':<6} {'Task':<14} {'Direct':>8} {'Full':>8} {'None':>8} {'Gap(F-N)':>10} {'Direct≈Full?':>14}"
    print(header)
    print("  " + "-" * 62)
    for r in probe_results:
        slug = slugs.get(r["model"], "")
        full = _load_react_em(slug, "full", r["task"], args.n, args.seed)
        none = _load_react_em(slug, "none", r["task"], args.n, args.seed)
        gap  = round(full - none, 3) if (full and none) else float("nan")
        close = "YES ← H2" if full and abs(r["em"] - full) <= 0.05 else "no"
        f_str = f"{full:.3f}" if full is not None else "  —  "
        n_str = f"{none:.3f}" if none is not None else "  —  "
        print(f"  {r['model']:<6} {r['task']:<14} {r['em']:>8.3f} {f_str:>8} {n_str:>8} {gap:>+10.3f} {close:>14}")

    print("\nH2 (post-hoc rationalization) is supported when Direct ≈ Full.")
    print("H2 is rejected when Direct << Full (Thought+tools are genuinely needed).")


if __name__ == "__main__":
    main()
