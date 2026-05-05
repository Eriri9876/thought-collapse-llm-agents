"""
TCI v2: Thought-Causal-Influence, Content-Level Analysis

Three complementary experiments under one script:

  A. TCI-content
     Inject a mismatched Thought (from a different question), measure
     how much the search query changes via token-F1 similarity.
     High similarity → Thought didn't affect Action (question drove it)
     Low similarity  → Thought drove Action (model followed the Thought)

  B. Pollution gradient
     Test 4 levels of Thought corruption:
       original → mismatched → scrambled → empty placeholder
     Plot how query similarity to the expected answer drifts.
     Flat curve → Thought content is irrelevant to Action.

  C. Adversarial injection
     Inject a Thought that explicitly points to the WRONG search direction
     (using the mismatched question's original action_input as the target).
     Measure: does the model's Action follow the Question or the Thought?
     "follows_question" rate → proportion where model ignores the misdirection.

All results are saved to results/tci_v2_<model>_<task>.json (not lost like v1).
"""
import json
import random
import re
import sys
from pathlib import Path

from src.llm import chat
from src.react import SYSTEM_PROMPTS

sys.stdout.reconfigure(encoding="utf-8")

LOG_DIR  = Path("logs")
OUT_DIR  = Path("results")
OUT_DIR.mkdir(exist_ok=True)

MODELS = {
    "14B": ("Qwen/Qwen2.5-14B-Instruct", "Qwen_Qwen2.5-14B-Instruct"),
    "32B": ("Qwen/Qwen2.5-32B-Instruct", "Qwen_Qwen2.5-32B-Instruct"),
    "V3":  ("deepseek-chat",              "deepseek-chat"),
}
TASKS = ["gsm8k", "hotpotqa", "webquestions"]


# ── utilities ─────────────────────────────────────────────────────────────────

def token_sim(s1: str, s2: str) -> float:
    """Token-level F1 similarity between two strings."""
    t1 = set(s1.lower().split())
    t2 = set(s2.lower().split())
    if not t1 or not t2:
        return 0.0
    overlap = len(t1 & t2)
    p = overlap / len(t1)
    r = overlap / len(t2)
    return round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0


def scramble(thought: str) -> str:
    """Randomly shuffle words in thought, preserving length."""
    words = thought.split()
    random.shuffle(words)
    return " ".join(words)


# ── data loading ──────────────────────────────────────────────────────────────

def _load_log(model_slug: str, task: str, seed: int = 42, n: int = 100) -> list[dict]:
    path = LOG_DIR / f"pilot_full_{model_slug}_{task}_n{n}_seed{seed}.jsonl"
    if not path.exists() and task == "hotpotqa" and model_slug == "deepseek-chat":
        path = LOG_DIR / f"pilot_full_{model_slug}_n{n}_seed{seed}.jsonl"
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def _extract_first_step(record: dict) -> dict | None:
    """Return (thought, action_type, action_input) from first valid step."""
    for step in record.get("trajectory", []):
        resp = step.get("response", "")
        m_t = re.search(r"Thought:\s*(.*?)(?:\nAction:|$)", resp, re.S | re.I)
        m_a = re.search(r"Action:\s*(\w+)\[(.+?)\]", resp, re.DOTALL)
        if m_t and m_a:
            return {
                "thought":       m_t.group(1).strip(),
                "action_type":   m_a.group(1).lower(),
                "action_input":  m_a.group(2).strip(),
            }
    return None


# ── single-step inference with injected Thought ───────────────────────────────

def _get_action_input(question: str, thought: str, model: str) -> str | None:
    """Return action_input the model produces given injected thought."""
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPTS["full"]},
        {"role": "user",      "content": f"Question: {question}"},
        {"role": "assistant", "content": f"Thought: {thought}"},
        {"role": "user",      "content": "Write your Action now."},
    ]
    try:
        response = chat(messages, model=model)
        m = re.search(r"Action:\s*\w+\[(.+?)\]", response, re.DOTALL)
        return m.group(1).strip() if m else None
    except Exception as e:
        print(f"    [API error: {e}]")
        return None


# ── core experiment ───────────────────────────────────────────────────────────

def run_tci_cell(
    model_key: str,
    task: str,
    n: int = 50,
    seed: int = 42,
) -> dict | None:
    model_api, model_slug = MODELS[model_key]

    records = _load_log(model_slug, task, seed=seed)
    if not records:
        print(f"  [{model_key} × {task}] no log found, skipping")
        return None

    items = []
    for rec in records:
        step = _extract_first_step(rec)
        if step and step["action_type"] in ("search", "calculate"):
            items.append({
                "id":           rec["id"],
                "question":     rec["question"],
                "thought":      step["thought"],
                "action_input": step["action_input"],
            })

    if len(items) < 10:
        print(f"  [{model_key} × {task}] only {len(items)} usable steps, skipping")
        return None

    random.seed(seed)
    sample = random.sample(items, min(n, len(items)))
    pool   = {it["id"]: it for it in items}

    # per-sample results
    rows = []

    print(f"\n=== TCI v2: {model_key} × {task} (n={len(sample)}) ===")
    for i, item in enumerate(sample):
        # pick mismatched item
        candidates = [v for k, v in pool.items() if k != item["id"]]
        if not candidates:
            continue
        mismatch = random.choice(candidates)

        orig_input  = item["action_input"]
        wrong_input = mismatch["action_input"]   # what the adversarial Thought points to

        # ── Condition A: mismatched Thought (TCI-content) ──────────────
        a_mismatch = _get_action_input(item["question"], mismatch["thought"], model_api)

        # ── Condition B: scrambled Thought ─────────────────────────────
        a_scrambled = _get_action_input(item["question"], scramble(item["thought"]), model_api)

        # ── Condition C: empty placeholder Thought ─────────────────────
        a_empty = _get_action_input(item["question"], "I need to find the answer.", model_api)

        # ── Condition D: adversarial Thought (explicit misdirection) ───
        adversarial_thought = (
            f"Based on the question, I should search for '{wrong_input}' "
            f"to find the answer."
        )
        a_adversarial = _get_action_input(item["question"], adversarial_thought, model_api)

        if not all([a_mismatch, a_scrambled, a_empty, a_adversarial]):
            continue  # skip if any API call failed

        # ── similarity measurements ─────────────────────────────────────
        row = {
            "id":            item["id"],
            "orig_input":    orig_input,
            "wrong_input":   wrong_input,
            # similarity to original expected action (higher = model followed question)
            "sim_mismatch":    token_sim(a_mismatch,    orig_input),
            "sim_scrambled":   token_sim(a_scrambled,   orig_input),
            "sim_empty":       token_sim(a_empty,       orig_input),
            "sim_adversarial": token_sim(a_adversarial, orig_input),
            # adversarial: did model follow question or Thought misdirection?
            "adv_sim_to_question":    token_sim(a_adversarial, orig_input),
            "adv_sim_to_misdirection": token_sim(a_adversarial, wrong_input),
            "follows_question": token_sim(a_adversarial, orig_input) >=
                                token_sim(a_adversarial, wrong_input),
        }
        rows.append(row)

        if (i + 1) % 10 == 0 or i == 0:
            fq_rate = sum(r["follows_question"] for r in rows) / len(rows)
            avg_sim  = sum(r["sim_mismatch"] for r in rows) / len(rows)
            print(f"  [{i+1}/{len(sample)}] "
                  f"sim_mismatch={avg_sim:.3f}  "
                  f"follows_question={fq_rate:.2%}")

    if not rows:
        return None

    def avg(key): return round(sum(r[key] for r in rows) / len(rows), 4)

    summary = {
        "model": model_key,
        "task":  task,
        "n":     len(rows),
        # A. TCI-content: sim to original when Thought is mismatched
        #    (lower = Thought drove Action away from expected)
        "tci_content_sim":    avg("sim_mismatch"),
        # B. Pollution gradient
        "sim_original":   1.0,          # control: by definition
        "sim_mismatched": avg("sim_mismatch"),
        "sim_scrambled":  avg("sim_scrambled"),
        "sim_empty":      avg("sim_empty"),
        # C. Adversarial injection
        "adv_follows_question_rate": avg("follows_question"),
        "adv_sim_to_question":       avg("adv_sim_to_question"),
        "adv_sim_to_misdirection":   avg("adv_sim_to_misdirection"),
        "rows": rows,
    }

    print(f"\n  >> Results ({model_key} × {task}):")
    print(f"     Pollution gradient (sim to expected action):")
    print(f"       original=1.000  mismatched={summary['sim_mismatched']:.3f}  "
          f"scrambled={summary['sim_scrambled']:.3f}  empty={summary['sim_empty']:.3f}")
    print(f"     Adversarial injection:")
    print(f"       follows_question={summary['adv_follows_question_rate']:.2%}  "
          f"sim_to_Q={summary['adv_sim_to_question']:.3f}  "
          f"sim_to_wrong={summary['adv_sim_to_misdirection']:.3f}")

    # save to file
    out_path = OUT_DIR / f"tci_v2_{model_slug}_{task}_n{len(rows)}_seed{seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"     saved → {out_path}")

    return summary


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int,  default=50)
    parser.add_argument("--seed",   type=int,  default=42)
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    parser.add_argument("--tasks",  nargs="+", default=TASKS)
    args = parser.parse_args()

    all_results = []
    for model_key in args.models:
        for task in args.tasks:
            r = run_tci_cell(model_key, task, n=args.n, seed=args.seed)
            if r:
                all_results.append(r)

    # ── summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("TCI v2 SUMMARY")
    print("=" * 72)
    print(f"  {'Model':<6} {'Task':<14} {'Mismatch':>10} {'Scrambled':>10} "
          f"{'Empty':>8} {'Follows-Q%':>12}")
    print("  " + "-" * 62)
    for r in all_results:
        print(f"  {r['model']:<6} {r['task']:<14} "
              f"{r['sim_mismatched']:>10.3f} "
              f"{r['sim_scrambled']:>10.3f} "
              f"{r['sim_empty']:>8.3f} "
              f"{r['adv_follows_question_rate']:>11.1%}")

    print("\nInterpretation:")
    print("  Pollution gradient: if sim stays high across all conditions → Thought is irrelevant")
    print("  Adversarial: high follows_question% → model ignores misdirecting Thought")
    print("  Both point to Thought Collapse when consistent with low Gap tasks (WebQ).")


if __name__ == "__main__":
    main()
