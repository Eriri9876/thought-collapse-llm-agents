"""
Per-question Wikipedia pageview signal extractor.

Loads N samples for one task (via src.data.get_samples), runs NER +
pageview lookup on each question, and writes a JSONL row per question
to experiments/coverage_signals/outputs/pageview_<task>_n<n>_seed<seed>.jsonl.

Sequential, no LLM calls. Resume-by-id mirrors src/run.py.

Usage::

    python -m src.coverage_signals.extract_pageview --task webquestions --n 50 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.data import get_samples
from src.coverage_signals import ner, pageview

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path("experiments/coverage_signals/outputs")


def run(task: str, n: int, seed: int) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"pageview_{task}_n{n}_seed{seed}.jsonl"

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
            print(f"  Resuming: {len(done_ids)} done, {n - len(done_ids)} remaining")

    print(f"\n=== Pageview signal: {task} (n={n}, seed={seed}) ===")
    with open(out_path, "a", encoding="utf-8") as f:
        for i, sample in enumerate(samples):
            if sample["id"] in done_ids:
                continue
            t0 = time.time()
            head = ner.extract_head_entity(sample["question"])
            sig = (pageview.popularity_signal(head) if head
                   else {"title": None, "monthly_views": [],
                         "mean_views": None, "log10_views": None})
            elapsed = round(time.time() - t0, 2)
            row = {
                "id":           sample["id"],
                "question":     sample["question"],
                "head_entity":  head,
                "title":        sig["title"],
                "mean_views":   sig["mean_views"],
                "log10_views":  sig["log10_views"],
                "elapsed":      elapsed,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            log_str = (f"{sig['log10_views']:.2f}"
                       if sig['log10_views'] is not None else "N/A")
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
    n_with_title = sum(1 for r in rows if r["title"])
    log10s = [r["log10_views"] for r in rows if r["log10_views"] is not None]
    n_with_views = len(log10s)

    print(f"\n  ── summary ──")
    print(f"  total:           {n_total}")
    print(f"  NER head found:  {n_with_head}/{n_total} ({100*n_with_head/n_total:.0f}%)")
    print(f"  title resolved:  {n_with_title}/{n_total} ({100*n_with_title/n_total:.0f}%)")
    print(f"  log10 computed:  {n_with_views}/{n_total} ({100*n_with_views/n_total:.0f}%)")
    if log10s:
        srt = sorted(log10s)
        median = srt[len(srt) // 2]
        mean = sum(srt) / len(srt)
        print(f"  log10_views:     min={srt[0]:.2f}  "
              f"median={median:.2f}  mean={mean:.2f}  max={srt[-1]:.2f}")
    print(f"  saved → {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.task, args.n, args.seed)


if __name__ == "__main__":
    main()
