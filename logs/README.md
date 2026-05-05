# `logs/` — Per-question experiment trajectories

This directory contains the raw experimental output of every model run reported in the paper. All files committed here are produced by `src.run` (ReAct experiments) or `src.probe` (Direct Probe). The reproduction scripts under `scripts/` read these files instead of calling any APIs, so the figures and tables can be regenerated **with no API key required**.

---

## File-name schema

```
pilot_<variant>_<model_slug>_<task>_n<N>_seed<SEED>.jsonl    ← ReAct runs
probe_direct_<model_slug>_<task>_n<N>_seed<SEED>.jsonl       ← Direct Probe runs
```

| Slot | Values |
|------|--------|
| `<variant>` | `full`, `none`, `compressed` |
| `<model_slug>` | `Qwen_Qwen2.5-7B-Instruct`, `Qwen_Qwen2.5-14B-Instruct`, `Qwen_Qwen2.5-32B-Instruct`, `deepseek-chat` |
| `<task>` | `gsm8k`, `hotpotqa`, `webquestions`, `triviaqa`, `math_hard` |
| `<N>` | `100` for most cells; `200` for the extended WebQ × {14B,32B,V3} and TriviaQA × 32B cells |
| `<SEED>` | `42` (primary), `7`, `123` (additional seeds for multi-seed cluster bootstrap) |

### Notes on file naming

**Legacy V3 × HotpotQA filenames.** A handful of files were saved before the `_<task>` slot was added to the schema. They remain on disk under their original names:

| Standard schema (used elsewhere)                         | Legacy file actually on disk                        |
|----------------------------------------------------------|-----------------------------------------------------|
| `pilot_full_deepseek-chat_hotpotqa_n100_seed42.jsonl`        | `pilot_full_deepseek-chat_n100_seed42.jsonl`        |
| `pilot_none_deepseek-chat_hotpotqa_n100_seed42.jsonl`        | `pilot_none_deepseek-chat_n100_seed42.jsonl`        |
| `pilot_compressed_deepseek-chat_hotpotqa_n100_seed42.jsonl`  | `pilot_compressed_deepseek-chat_n100_seed42.jsonl`  |

Only V3 × HotpotQA × seed=42 is affected; all other cells use the standard schema. `src.bootstrap_ci._log_path` and `src.routing.load_react` fall back to the legacy name automatically when the standard name is missing.

**HotpotQA bridge/comparison subset.** Files of the form `pilot_<variant>_<model_slug>_hotpotqa_comparison_n100_seed42.jsonl` hold the bridge-vs-comparison split used by `src.analyze`. These are not consumed by the main bootstrap (`src.bootstrap_ci`).

**MATH-hard Direct Probe alignment.** `src/data.py` uses Python's built-in `hash()` for MATH-hard question IDs, which is non-deterministic across processes (depends on `PYTHONHASHSEED`). As a result, Direct Probe and Full/None ReAct logs for `math_hard` cannot be aligned by question ID, and CAR analysis skips this task. Paper Table 4 reports CAR results on 4 tasks (gsm8k / hotpotqa / webquestions / triviaqa) only; `math_hard` is excluded by design. To fix in future runs, replace the hash function at `src/data.py:142` with a stable hash (e.g., `hashlib.md5`).

---

## Schema — `pilot_*.jsonl` (ReAct trajectories)

One JSON object per line. One file per (variant × model × task × seed) cell.

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Stable question id from the source dataset (HotpotQA `_id`, TriviaQA `question_id`, WebQ url, `gsm8k_<idx>`, `math_<hash>`) |
| `question` | str | Verbatim question text |
| `gold` | str | Reference answer (for WebQ/TriviaQA, the first alias) |
| `pred` | str \| null | Model's `finish[...]` argument; `null` if `status != "success"` |
| `status` | str | `"success"` if a `finish[...]` action was emitted; `"max_steps_reached"` otherwise |
| `steps` | int | Number of ReAct steps executed (≤ `max_steps=8`) |
| `elapsed` | float | Wall-clock seconds for the entire trajectory |
| `em` | int | Exact-match score, `0` or `1` (numeric tolerance for math tasks; alias-aware string match otherwise) |
| `f1` | float | Token-level F1 against the best alias, in `[0, 1]` |
| `trajectory` | list[step] | Step-by-step record (see below) |

Each step in `trajectory` is one of:

```jsonc
// Normal step
{
  "step": 0,
  "response": "Thought: I need to find ...\nAction: search[...]",
  "action": "search",       // "search" | "calculate" | "finish"
  "input": "...",            // verbatim argument inside the brackets
  "observation": "..."       // tool output; absent on the finish step
}

// Parse-failed step (terminal)
{ "step": 3, "response": "...", "error": "parse_failed" }
```

For the `none` variant, the model's `response` field omits the `Thought:` prefix by construction; for the `compressed` variant, the Thought is constrained to ≤ 10 words by the system prompt (see `src/react.py:SYSTEM_PROMPTS`).

---

## Schema — `probe_direct_*.jsonl` (Direct Probe)

One JSON object per line. The Direct Probe asks the same question with no ReAct loop and no tools, so each record is a single query/response pair (no `trajectory`).

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Same id as in the matching `pilot_*` file (used by `src.routing` to align records) |
| `question` | str | Verbatim question text |
| `gold` | str | Reference answer |
| `pred` | str \| null | Raw model response (no `finish[...]` wrapping) |
| `elapsed` | float | Wall-clock seconds |
| `em` | int | Exact-match against any alias |
| `f1` | float | Token-level F1 |

---

## Resume semantics

Both `src.run` and `src.probe` are resumable: on restart they read the existing `.jsonl`, collect the `id`s already done, and skip them. Re-running with the same arguments is therefore safe and incremental — useful when an API call fails partway through. This also means `.jsonl` files in this directory may have been written across multiple sessions.
