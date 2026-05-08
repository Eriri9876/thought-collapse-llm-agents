"""
Per-question InfiniGram frequency signal extractor.

Loads N samples for one task, runs NER + InfiniGram count, writes one
JSONL row per question to::

    experiments/coverage_signals/outputs/infinigram_<task>_n<n>_seed<seed>.jsonl

Sequential. No LLM calls. Resume-by-id mirrors src.coverage_signals.extract_pageview.

Usage::

    python -m src.coverage_signals.extract_infinigram --task webquestions --n 50 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.data import get_samples
from src.coverage_signals import infinigram, ner

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path("experiments/coverage_signals/outputs")


def run(task: str, n: int, seed: int,
        index: str = infinigram.DEFAULT_INDEX) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"infinigram_{task}_n{n}_seed{seed}.jsonl"

    samples = get_samples(n, seed=seed, dataset=task)
    done_ids: set = set()
    rows: list[dict] = []
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                done_ids.add(rec["id"])
                rows.append(rec)
        if done_ids:
            print(f"  Resuming: {len(done_ids)} done, "
                  f"{n - len(done_ids)} remaining")

    print(f"\n=== InfiniGram signal: {task} "
          f"(n={n}, seed={seed}, index={index}) ===")
    with open(out_path, "a", encoding="utf-8") as f:
        for i, sample in enumerate(samples):
            if sample["id"] in done_ids:
                continue
            t0 = time.time()
            head = ner.extract_head_entity(sample["question"])
            sig = (infinigram.frequency_signal(head, index=index)
                   if head
                   else {"index": index, "query": None, "count": None,
                         "log10_count": None, "approx": None,
                         "n_tokens": None})
            elapsed = round(time.time() - t0, 2)
            row = {
                "id":          sample["id"],
                "question":    sample["question"],
                "head_entity": head,
                "index":       sig["index"],
                "count":       sig["count"],
                "log10_count": sig["log10_count"],
                "approx":      sig["approx"],
                "n_tokens":    sig["n_tokens"],
                "elapsed":     elapsed,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            log_str = (f"{sig['log10_count']:.2f}"
                       if sig['log10_count'] is not None else "N/A")
            print(f"  [{i+1:>3}/{n}] head={str(head)[:32]:32s} "
                  f"log10={log_str:>5}  ({elapsed}s)")

    _print_summary(rows, n, out_path)
    return out_path


def _print_summary(rows: list[dict], n: int, out_path: Path) -> None:
    n_total = len(rows)
    if n_total == 0:
        print("  (no rows)")
        return
    n_with_head = sum(1 for r in rows if r["head_entity"])
    log10s = [r["log10_count"] for r in rows if r["log10_count"] is not None]
    n_with_count = len(log10s)

    print(f"\n  ── summary ──")
    print(f"  total:           {n_total}")
    print(f"  NER head found:  {n_with_head}/{n_total} ({100*n_with_head/n_total:.0f}%)")
    print(f"  count returned:  {n_with_count}/{n_total} ({100*n_with_count/n_total:.0f}%)")
    if log10s:
        srt = sorted(log10s)
        median = srt[len(srt) // 2]
        mean = sum(srt) / len(srt)
        print(f"  log10_count:     min={srt[0]:.2f}  median={median:.2f}  "
              f"mean={mean:.2f}  max={srt[-1]:.2f}")
    print(f"  saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index", default=infinigram.DEFAULT_INDEX)
    args = parser.parse_args()
    run(args.task, args.n, args.seed, index=args.index)


if __name__ == "__main__":
    main()
