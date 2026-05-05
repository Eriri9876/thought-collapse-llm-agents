"""
Logprob Probe — V3 only (DeepSeek API supports logprobs; SiliconFlow does not)

For each question, we compute P(original_action | context) under three conditions:
  - Q+T : Question + original Thought  (full context, baseline)
  - Q   : Question only, no Thought
  - T   : Thought only, no Question

If P(action | Q) ≈ P(action | Q+T)  → Question alone drives Action; Thought is redundant
If P(action | Q) << P(action | Q+T) → Thought was genuinely needed
If P(action | T) << P(action | Q+T) → Thought alone is insufficient; Question is essential

Mechanism A (early decision): P(action|Q) ≈ P(action|Q+T) >> P(action|T)
Mechanism B (attention bypass): P(action|Q+T) > P(action|Q), and P(action|T) also carries some signal
"""
import json
import os
import sys
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

LOG_DIR = Path("logs")
OUT_DIR = Path("results")
OUT_DIR.mkdir(exist_ok=True)

MODEL    = "deepseek-chat"
MODEL_SLUG = "deepseek-chat"

SYSTEM_FULL = """You are a question-answering agent. Solve the question using search and calculation.

Use this format strictly:
Thought: reason about what to do next
Action: search[your query]  OR  calculate[math expression]  OR  finish[your final answer]

An Observation will be provided after each action.
Always write a Thought before every Action.
If searches are not yielding useful results, use finish[your best guess] rather than searching indefinitely."""

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_log(task: str, seed: int = 42) -> list[dict]:
    path = LOG_DIR / f"pilot_full_{MODEL_SLUG}_{task}_n100_seed{seed}.jsonl"
    if not path.exists() and task == "hotpotqa":
        path = LOG_DIR / f"pilot_full_{MODEL_SLUG}_n100_seed{seed}.jsonl"
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


def _extract_first_step(record: dict) -> dict | None:
    for step in record.get("trajectory", []):
        resp = step.get("response", "")
        m_t = re.search(r"Thought:\s*(.*?)(?:\nAction:|$)", resp, re.S | re.I)
        m_a = re.search(r"Action:\s*(\w+)\[(.+?)\]", resp, re.DOTALL)
        if m_t and m_a:
            return {
                "thought":      m_t.group(1).strip(),
                "action_type":  m_a.group(1).lower(),
                "action_input": m_a.group(2).strip(),
            }
    return None


def _action_str(action_type: str, action_input: str) -> str:
    return f"Action: {action_type}[{action_input}]"


def _logprob_of_completion(messages: list[dict], target: str) -> float | None:
    """
    Compute sum of token logprobs for `target` appended as assistant continuation.
    Uses echo-style: append target as assistant prefix and get logprobs back.
    """
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=len(target.split()) + 20,
            temperature=0.0,
            logprobs=True,
        )
        lp_content = resp.choices[0].logprobs.content if resp.choices[0].logprobs else None
        if not lp_content:
            return None
        # sum all token logprobs (these are for the generated continuation)
        total = sum(tok.logprob for tok in lp_content)
        return round(total, 4)
    except Exception as e:
        print(f"    [API error: {e}]")
        return None


def _compute_action_logprobs(question: str, thought: str, action_type: str, action_input: str):
    """Return dict of logprobs under Q+T, Q-only, T-only conditions."""
    target = _action_str(action_type, action_input)
    prompt_suffix = [{"role": "user", "content": "Write your Action now."}]

    # Q+T: full context
    msgs_qt = [
        {"role": "system",    "content": SYSTEM_FULL},
        {"role": "user",      "content": f"Question: {question}"},
        {"role": "assistant", "content": f"Thought: {thought}"},
        *prompt_suffix,
    ]

    # Q only: question, no thought
    msgs_q = [
        {"role": "system",  "content": SYSTEM_FULL},
        {"role": "user",    "content": f"Question: {question}"},
        *prompt_suffix,
    ]

    # T only: thought, no question
    msgs_t = [
        {"role": "system",    "content": SYSTEM_FULL},
        {"role": "user",      "content": "Here is a reasoning step:"},
        {"role": "assistant", "content": f"Thought: {thought}"},
        *prompt_suffix,
    ]

    return {
        "lp_qt": _logprob_of_completion(msgs_qt, target),
        "lp_q":  _logprob_of_completion(msgs_q,  target),
        "lp_t":  _logprob_of_completion(msgs_t,  target),
        "action": target,
    }


# ── main experiment ───────────────────────────────────────────────────────────

def run_logprob_probe(task: str, n: int = 50, seed: int = 42):
    import random
    records = _load_log(task, seed=seed)
    if not records:
        print(f"  [{task}] no log found")
        return None

    items = []
    for rec in records:
        step = _extract_first_step(rec)
        if step and step["action_type"] in ("search", "calculate"):
            items.append({
                "id":           rec["id"],
                "question":     rec["question"],
                "thought":      step["thought"],
                "action_type":  step["action_type"],
                "action_input": step["action_input"],
            })

    random.seed(seed)
    sample = random.sample(items, min(n, len(items)))

    print(f"\n=== Logprob Probe: V3 × {task} (n={len(sample)}) ===")
    rows = []
    for i, item in enumerate(sample):
        result = _compute_action_logprobs(
            item["question"], item["thought"],
            item["action_type"], item["action_input"]
        )
        if None in result.values():
            continue

        # normalise: use Q+T as reference, express others as delta
        delta_q  = result["lp_q"]  - result["lp_qt"]   # positive = Q alone is better
        delta_t  = result["lp_t"]  - result["lp_qt"]   # negative expected (T alone is worse)

        row = {**item, **result, "delta_q": delta_q, "delta_t": delta_t}
        rows.append(row)

        if (i + 1) % 10 == 0 or i == 0:
            avg_dq = sum(r["delta_q"] for r in rows) / len(rows)
            avg_dt = sum(r["delta_t"] for r in rows) / len(rows)
            print(f"  [{i+1}/{len(sample)}] avg Δ(Q vs Q+T)={avg_dq:+.2f}  "
                  f"avg Δ(T vs Q+T)={avg_dt:+.2f}")

    if not rows:
        return None

    avg_dq = round(sum(r["delta_q"] for r in rows) / len(rows), 3)
    avg_dt = round(sum(r["delta_t"] for r in rows) / len(rows), 3)
    avg_qt = round(sum(r["lp_qt"]   for r in rows) / len(rows), 3)
    avg_q  = round(sum(r["lp_q"]    for r in rows) / len(rows), 3)
    avg_t  = round(sum(r["lp_t"]    for r in rows) / len(rows), 3)

    summary = {
        "task":   task,
        "model":  "V3",
        "n":      len(rows),
        "avg_lp_qt": avg_qt,
        "avg_lp_q":  avg_q,
        "avg_lp_t":  avg_t,
        "delta_q_vs_qt": avg_dq,   # Q vs Q+T: near 0 → Mechanism A
        "delta_t_vs_qt": avg_dt,   # T vs Q+T: very negative → T alone useless
        "rows": rows,
    }

    out_path = OUT_DIR / f"logprob_probe_V3_{task}_n{len(rows)}_seed{seed}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n  >> V3 × {task}:")
    print(f"     avg logP(action | Q+T) = {avg_qt}")
    print(f"     avg logP(action | Q  ) = {avg_q}   Δ={avg_dq:+.3f}")
    print(f"     avg logP(action | T  ) = {avg_t}   Δ={avg_dt:+.3f}")
    print(f"     Mechanism A signal: Δ(Q) near 0 means Question alone is sufficient")
    print(f"     saved → {out_path}")

    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=["gsm8k", "hotpotqa", "webquestions"])
    parser.add_argument("--n",    type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    all_results = []
    for task in args.tasks:
        r = run_logprob_probe(task, n=args.n, seed=args.seed)
        if r:
            all_results.append(r)

    print("\n" + "=" * 65)
    print("LOGPROB PROBE SUMMARY  (V3)")
    print("=" * 65)
    print(f"  {'Task':<14} {'lp(Q+T)':>10} {'lp(Q)':>8} {'Δ(Q)':>8} {'lp(T)':>8} {'Δ(T)':>8}")
    print("  " + "-" * 55)
    for r in all_results:
        print(f"  {r['task']:<14} {r['avg_lp_qt']:>10.2f} "
              f"{r['avg_lp_q']:>8.2f} {r['delta_q_vs_qt']:>+8.2f} "
              f"{r['avg_lp_t']:>8.2f} {r['delta_t_vs_qt']:>+8.2f}")

    print("\nInterpretation:")
    print("  Δ(Q) = lp(Q) - lp(Q+T): near 0 → Mechanism A (Question alone drives Action)")
    print("  Δ(T) = lp(T) - lp(Q+T): very negative → Thought alone is insufficient")


if __name__ == "__main__":
    main()
