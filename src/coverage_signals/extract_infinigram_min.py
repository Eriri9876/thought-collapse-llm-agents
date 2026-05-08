"""
Per-question MULTI-entity InfiniGram frequency signal extractor.

Mirrors :mod:`src.coverage_signals.extract_pageview_min` for the
InfiniGram (Dolma v1.7) corpus-occurrence signal. For each question we
run NER, take every filtered entity, count its occurrences in the
chosen InfiniGram index, and record::

    head_count    log10(1 + count) of the head entity (= legacy single-entity signal)
    min_count     log10(1 + count) of the LEAST-frequent surviving entity
    max_count     log10(1 + count) of the MOST-frequent surviving entity
    mean_count    arithmetic mean of log10(1 + count) over all surviving entities

Tests the bottleneck-entity hypothesis on the InfiniGram channel: a
question is hard for the model when its rarest entity is rare in the
training corpus, regardless of how frequent the head is. Direct
companion to ``extract_pageview_min`` so the two signals can be
compared symmetrically (head vs. min).

Resume-by-id like ``extract_pageview_min``. Output::

    experiments/coverage_signals/outputs/infinigram_min_<task>_n<n>_seed<seed>.jsonl

Usage::

    python -m src.coverage_signals.extract_infinigram_min --task hotpotqa --n 50 --seed 42
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.coverage_signals import infinigram, ner
from src.data import get_samples

sys.stdout.reconfigure(encoding="utf-8")

OUT_DIR = Path("experiments/coverage_signals/outputs")


def _entity_log10(entity: str, index: str) -> tuple[int | None, float | None,
                                                    bool | None, int | None]:
    sig = infinigram.frequency_signal(entity, index=index)
    return (sig.get("count"), sig.get("log10_count"),
            sig.get("approx"), sig.get("n_tokens"))


def run(task: str, n: int, seed: int,
        index: str = infinigram.DEFAULT_INDEX) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"infinigram_min_{task}_n{n}_seed{seed}.jsonl"

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

    print(f"\n=== Multi-entity InfiniGram: {task} "
          f"(n={n}, seed={seed}, index={index}) ===")
    with open(out_path, "a", encoding="utf-8") as f:
        for i, sample in enumerate(samples):
            if sample["id"] in done_ids:
                continue
            t0 = time.time()
            ents = ner.extract_all_entities_filtered(sample["question"])
            head_ent = ner.extract_head_entity(sample["question"])

            ent_results = []
            for e in ents:
                cnt, log10, approx, ntok = _entity_log10(e, index)
                ent_results.append({
                    "entity":      e,
                    "count":       cnt,
                    "log10":       log10,
                    "approx":      approx,
                    "n_tokens":    ntok,
                })
            log10_list = [r["log10"] for r in ent_results
                          if r["log10"] is not None]

            head_count, head_log10 = (None, None)
            head_approx, head_ntok = (None, None)
            if head_ent:
                head_count, head_log10, head_approx, head_ntok = (
                    _entity_log10(head_ent, index)
                )

            min_log10 = min(log10_list) if log10_list else None
            max_log10 = max(log10_list) if log10_list else None
            mean_log10 = (sum(log10_list) / len(log10_list)
                          if log10_list else None)

            elapsed = round(time.time() - t0, 2)
            row = {
                "id":             sample["id"],
                "question":       sample["question"],
                "head_entity":    head_ent,
                "all_entities":   ents,
                "n_entities":     len(ents),
                "n_resolved":     len(log10_list),
                "index":          index,
                "entity_results": ent_results,
                "head_count":     head_log10,
                "min_count":      min_log10,
                "max_count":      max_log10,
                "mean_count":     mean_log10,
                "head_count_raw": head_count,
                "elapsed":        elapsed,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)

            head_str = f"{head_log10:.2f}" if head_log10 is not None else "N/A"
            min_str = f"{min_log10:.2f}" if min_log10 is not None else "N/A"
            print(f"  [{i+1:>3}/{n}] head={str(head_ent)[:24]:24s} "
                  f"({head_str:>5})  n_ents={len(ents):>2}  "
                  f"min={min_str:>5}  ({elapsed}s)")

    _print_summary(rows, n, out_path)
    return out_path


def _print_summary(rows: list[dict], n: int, out_path: Path) -> None:
    n_total = len(rows)
    if n_total == 0:
        print("  (no rows)")
        return
    n_with_head = sum(1 for r in rows if r["head_count"] is not None)
    n_with_min = sum(1 for r in rows if r["min_count"] is not None)

    n_ents_list = sorted([r["n_entities"] for r in rows])
    if n_ents_list:
        ents_min = n_ents_list[0]
        ents_med = n_ents_list[len(n_ents_list) // 2]
        ents_max = n_ents_list[-1]
        ents_mean = sum(n_ents_list) / len(n_ents_list)

    head_log10s = sorted([r["head_count"] for r in rows
                          if r["head_count"] is not None])
    min_log10s = sorted([r["min_count"] for r in rows
                         if r["min_count"] is not None])

    print(f"\n  ── summary ──")
    print(f"  total:                {n_total}")
    print(f"  head count avail:     {n_with_head}/{n_total} "
          f"({100*n_with_head/n_total:.0f}%)")
    print(f"  min count avail:      {n_with_min}/{n_total} "
          f"({100*n_with_min/n_total:.0f}%)")
    print(f"  entities/question:    min={ents_min}  median={ents_med}  "
          f"mean={ents_mean:.1f}  max={ents_max}")
    if head_log10s:
        print(f"  head log10:  min={head_log10s[0]:.2f}  "
              f"med={head_log10s[len(head_log10s)//2]:.2f}  "
              f"mean={sum(head_log10s)/len(head_log10s):.2f}  "
              f"max={head_log10s[-1]:.2f}")
    if min_log10s:
        print(f"  min  log10:  min={min_log10s[0]:.2f}  "
              f"med={min_log10s[len(min_log10s)//2]:.2f}  "
              f"mean={sum(min_log10s)/len(min_log10s):.2f}  "
              f"max={min_log10s[-1]:.2f}")
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
