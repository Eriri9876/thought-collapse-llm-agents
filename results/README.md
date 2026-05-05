# `results/` — Aggregated analysis outputs

Each JSON file in this directory is the structured output of one analysis script. The reproduction scripts under `scripts/` regenerate these files in place from the raw `logs/*.jsonl` trajectories — **no API calls are involved**. Anything in this directory can be safely deleted and rebuilt.

---

## File map

| File | Produced by | Reproduce with |
|------|-------------|----------------|
| `routing_analysis.json` | `src.routing` | `bash scripts/reproduce_table4_car.sh` |
| `token_analysis.json` | `src.token_analysis` | `bash scripts/reproduce_token_savings.sh` |
| `tci_v2_<model_slug>_<task>_n<N>_seed42.json` (×15) | `src.tci` | `python -m src.tci` (requires API — not zero-cost) |
| `logprob_probe_V3_webquestions_n3_seed42.json` (×1) | `src.logprob_probe` | `python -m src.logprob_probe` (requires DeepSeek API — V3 only) |

---

## `routing_analysis.json`

Output of Coverage-Aware Routing (CAR). Top-level keys:

| Key | Description |
|-----|-------------|
| `results` | List of per-(model, task, θ) entries — the threshold-sensitivity sweep |
| `best` | Map `"(model, task)" → entry` choosing the θ with the highest routed EM per cell |
| `loto` | Map `model → list[entry]` for leave-one-task-out τ selection |

Each entry contains routing decision (`route_to_direct`), held-out EM under each policy (`em_routed`, `em_full`, `em_none`, `em_direct`, `em_oracle`), token usage (`tok_routed`, `tok_full`, `tok_savings_abs`, `tok_savings_pct`), and the calibration coverage estimate (`coverage_est`).

## `token_analysis.json`

Per-cell token cost breakdown. Top-level keys:

| Key | Description |
|-----|-------------|
| `cells` | Map `"(model, task, variant)" → {em, avg_steps, avg_total_tok, avg_thought_tok, n}` |
| `pareto` | List of per-(model, task) entries: token overhead vs. Thought-Gap, with a verdict label (`Thought worthwhile`, `Marginal gain`, `Break-even / noise`, `Thought harmful`) |

## `tci_v2_*.json` (15 files: 3 models × 5 tasks)

Output of the Thought Causal Influence v2 probe. One file per (model × task) cell at seed=42. Each contains four conditions (mismatched / scrambled / empty-placeholder / adversarial Thought) and the resulting action-input similarity to the original; the headline statistic is `adv_follows_question_rate` (proportion of cases where the model's action follows the question rather than the misdirecting Thought).

## `logprob_probe_V3_webquestions_n3_seed42.json`

Logprob-based variant of the Thought-causality test, run only on V3 × WebQuestions (DeepSeek API exposes logprobs; SiliconFlow does not). For each item, computes log P(action | Q+T), log P(action | Q), log P(action | T) to test whether question alone is sufficient to determine the action.
