"""
Post-hoc analysis script for Thought Collapse experiments.
Three analyses:
  1. HotpotQA bridge vs comparison subset Gap split
  2. Action sequence length across all tasks/models/variants
  3. Compressed Thought content analysis (focus on WebQ-V3)
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

LOG_DIR = Path("logs")
MODELS = {
    "7B":  "Qwen_Qwen2.5-7B-Instruct",
    "14B": "Qwen_Qwen2.5-14B-Instruct",
    "32B": "Qwen_Qwen2.5-32B-Instruct",
    "V3":  "deepseek-chat",
}
VARIANTS = ["full", "none", "compressed"]
TASKS = ["hotpotqa", "gsm8k", "webquestions"]


def load_log(variant: str, model_slug: str, task: str, n: int = 100) -> list[dict]:
    seed = 42
    path = LOG_DIR / f"pilot_{variant}_{model_slug}_{task}_n{n}_seed{seed}.jsonl"
    # V3 HotpotQA was logged without task name (early run)
    if not path.exists() and task == "hotpotqa" and model_slug == "deepseek-chat":
        path = LOG_DIR / f"pilot_{variant}_{model_slug}_n{n}_seed{seed}.jsonl"
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


def em_avg(records):
    if not records:
        return None
    return round(sum(r["em"] for r in records) / len(records), 4)


# ─────────────────────────────────────────────────────────────
# Analysis 1: HotpotQA bridge vs comparison Gap split
# ─────────────────────────────────────────────────────────────
def analysis_bridge_comparison():
    print("\n" + "="*60)
    print("ANALYSIS 1: HotpotQA bridge vs comparison Gap split")
    print("="*60)

    # Load HotpotQA dataset to get type field
    try:
        from datasets import load_dataset
        ds = load_dataset("hotpot_qa", "fullwiki", split="validation")
        id_to_type = {row["id"]: row["type"] for row in ds}
        print(f"Loaded HotpotQA: {len(id_to_type)} records, types: {set(id_to_type.values())}")
    except Exception as e:
        print(f"Could not load HotpotQA dataset: {e}")
        return

    for model_key, model_slug in MODELS.items():
        full_recs = load_log("full", model_slug, "hotpotqa")
        none_recs = load_log("none", model_slug, "hotpotqa")
        if not full_recs or not none_recs:
            continue

        # Index by id
        full_by_id = {r["id"]: r for r in full_recs}
        none_by_id = {r["id"]: r for r in none_recs}

        bridge_full, bridge_none = [], []
        comparison_full, comparison_none = [], []

        for qid, qtype in id_to_type.items():
            if qid in full_by_id:
                if qtype == "bridge":
                    bridge_full.append(full_by_id[qid])
                    if qid in none_by_id:
                        bridge_none.append(none_by_id[qid])
                elif qtype == "comparison":
                    comparison_full.append(full_by_id[qid])
                    if qid in none_by_id:
                        comparison_none.append(none_by_id[qid])

        bf = em_avg(bridge_full)
        bn = em_avg(bridge_none)
        cf = em_avg(comparison_full)
        cn = em_avg(comparison_none)

        bg = round(bf - bn, 4) if bf is not None and bn is not None else None
        cg = round(cf - cn, 4) if cf is not None and cn is not None else None

        print(f"\n  {model_key}:")
        print(f"    bridge     n={len(bridge_full):3d}  full={bf}  none={bn}  Gap={bg}")
        print(f"    comparison n={len(comparison_full):3d}  full={cf}  none={cn}  Gap={cg}")


# ─────────────────────────────────────────────────────────────
# Analysis 2: Action sequence length (steps) across all configs
# ─────────────────────────────────────────────────────────────
def analysis_action_length():
    print("\n" + "="*60)
    print("ANALYSIS 2: Mean action steps per (task, model, variant)")
    print("="*60)

    task_n = {"hotpotqa": 100, "gsm8k": 100, "webquestions": 100}

    for task in TASKS:
        print(f"\n  Task: {task}")
        print(f"  {'Model':<6} {'full steps':>12} {'none steps':>12} {'comp steps':>12}  {'full succ':>10} {'none succ':>10}")
        print(f"  {'-'*70}")
        for model_key, model_slug in MODELS.items():
            row = []
            succ_row = []
            for variant in VARIANTS:
                recs = load_log(variant, model_slug, task, n=task_n[task])
                if recs:
                    mean_steps = round(sum(r["steps"] for r in recs) / len(recs), 2)
                    success = sum(1 for r in recs if r["status"] == "success")
                    row.append(f"{mean_steps:>12.2f}")
                    succ_row.append(f"{success:>10d}")
                else:
                    row.append(f"{'N/A':>12}")
                    succ_row.append(f"{'N/A':>10}")
            print(f"  {model_key:<6} {'  '.join(row)}  {'  '.join(succ_row[:2])}")


# ─────────────────────────────────────────────────────────────
# Analysis 3: Compressed Thought content analysis
# ─────────────────────────────────────────────────────────────
THOUGHT_PATTERNS = {
    "format_anchor":    re.compile(r"(I (should|need|will|must)|Let me|To (find|answer|determine))", re.I),
    "entity_restate":   re.compile(r"(The (question|answer|director|film|person|year|country) is|based on|according to)", re.I),
    "shallow_reasoning":re.compile(r"(so|therefore|because|since|thus|means|implies|=\s*\d)", re.I),
    "numeric_fragment": re.compile(r"=\s*\d+|\d+\s*[+\-*/]\s*\d+"),
    "entity_bridge":    re.compile(r"(found|got|is)\s+[A-Z][a-z]+(\s+[A-Z][a-z]+)*"),
}


def classify_thought(thought: str) -> str:
    if not thought or thought.strip() == "":
        return "empty"
    t = thought.strip()
    if len(t.split()) > 15:
        return "too_long"
    if THOUGHT_PATTERNS["numeric_fragment"].search(t):
        return "numeric_fragment"
    if THOUGHT_PATTERNS["entity_bridge"].search(t):
        return "entity_bridge"
    if THOUGHT_PATTERNS["shallow_reasoning"].search(t):
        return "shallow_reasoning"
    if THOUGHT_PATTERNS["entity_restate"].search(t):
        return "entity_restate"
    if THOUGHT_PATTERNS["format_anchor"].search(t):
        return "format_anchor"
    return "other"


def extract_thought(response: str) -> str:
    """Extract Thought content from a response string."""
    m = re.match(r"Thought:\s*(.*?)(?:\n|Action:|$)", response, re.S | re.I)
    if m:
        return m.group(1).strip()
    return ""


def analysis_compressed_thought():
    print("\n" + "="*60)
    print("ANALYSIS 3: Compressed Thought content classification")
    print("="*60)

    configs = [
        ("WebQ V3 compressed (comp>full)", "V3", "webquestions"),
        ("WebQ 32B compressed (comp<none)", "32B", "webquestions"),
        ("GSM8K 32B compressed (comp<none)", "32B", "gsm8k"),
        ("HotpotQA 14B compressed", "14B", "hotpotqa"),
    ]

    for label, model_key, task in configs:
        model_slug = MODELS[model_key]
        comp_recs = load_log("compressed", model_slug, task)
        if not comp_recs:
            print(f"\n  {label}: no data")
            continue

        type_counts = defaultdict(int)
        correct_by_type = defaultdict(int)
        total_by_type = defaultdict(int)

        for rec in comp_recs:
            traj = rec.get("trajectory", [])
            # Extract first Thought from trajectory
            thought = ""
            for step in traj:
                t = extract_thought(step.get("response", ""))
                if t:
                    thought = t
                    break
            ttype = classify_thought(thought)
            type_counts[ttype] += 1
            total_by_type[ttype] += 1
            if rec["em"] == 1:
                correct_by_type[ttype] += 1

        total = len(comp_recs)
        print(f"\n  {label}  (n={total}, EM={em_avg(comp_recs)})")
        print(f"  {'Type':<22} {'Count':>6} {'%':>6} {'EM%':>6}")
        print(f"  {'-'*44}")
        for ttype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            pct = round(100 * count / total, 1)
            em_pct = round(100 * correct_by_type[ttype] / total_by_type[ttype], 1) if total_by_type[ttype] else 0
            print(f"  {ttype:<22} {count:>6} {pct:>5.1f}% {em_pct:>5.1f}%")

    # Bonus: WebQ-V3 compressed > full — show example thoughts
    print("\n  --- WebQ V3: sample compressed Thoughts (correct answers) ---")
    model_slug = MODELS["V3"]
    comp_recs = load_log("compressed", model_slug, "webquestions")
    shown = 0
    for rec in comp_recs:
        if rec["em"] == 1 and rec.get("trajectory"):
            thought = extract_thought(rec["trajectory"][0].get("response", ""))
            if thought:
                print(f"  Q: {rec['question'][:60]}")
                print(f"  Thought: {thought[:80]!r}")
                print(f"  Pred: {rec['pred']!r}  Gold: {rec['gold']!r}")
                print()
                shown += 1
            if shown >= 5:
                break


# ─────────────────────────────────────────────────────────────
# Summary table: full picture
# ─────────────────────────────────────────────────────────────
def summary_table():
    print("\n" + "="*60)
    print("SUMMARY: Thought-Gap across all tasks and models")
    print("="*60)

    task_n = {"hotpotqa": 100, "gsm8k": 100, "webquestions": 100}
    task_labels = {"hotpotqa": "HotpotQA", "gsm8k": "GSM8K", "webquestions": "WebQ"}

    for task in TASKS:
        print(f"\n  {task_labels[task]}")
        print(f"  {'Model':<6} {'full':>7} {'none':>7} {'comp':>7} {'Gap':>7}")
        print(f"  {'-'*38}")
        for model_key, model_slug in MODELS.items():
            ems = {}
            for v in VARIANTS:
                recs = load_log(v, model_slug, task, n=task_n[task])
                ems[v] = em_avg(recs)
            f, n, c = ems["full"], ems["none"], ems["compressed"]
            gap = round(f - n, 4) if f is not None and n is not None else None
            f_s = f"{f:.3f}" if f is not None else "N/A"
            n_s = f"{n:.3f}" if n is not None else "N/A"
            c_s = f"{c:.3f}" if c is not None else "N/A"
            g_s = f"{gap:+.3f}" if gap is not None else "N/A"
            print(f"  {model_key:<6} {f_s:>7} {n_s:>7} {c_s:>7} {g_s:>7}")


if __name__ == "__main__":
    summary_table()
    analysis_bridge_comparison()
    analysis_action_length()
    analysis_compressed_thought()
    print("\nDone.")
