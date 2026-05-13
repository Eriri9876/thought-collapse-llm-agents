# Thought Collapse in LLM Agents

*A Cross-Task Diagnosis of Capacity Domination in Agentic Reasoning*

> A formal paper link will be added once available.

ReAct's *Thought* step is widely assumed to improve agentic reasoning. On tasks well-covered by pre-training, this paper finds the opposite: **removing the Thought scaffold can improve EM by up to 4× (Qwen2.5-14B on WebQuestions) while saving 93–99% of generated tokens.** This repository contains:

- a ~200-line plain-Python ReAct stack (`src/react.py` + `src/tools.py` + `src/llm.py`, no LangChain),
- per-question trajectories for **5 tasks × 5 models × 3 Thought variants × three seeds** (Llama-3.1-8B kept at a single seed as a weak-end boundary check), fully logged in `logs/`,
- one-command reproduction of the paper's main figure and tables — **no API key required.**

```bash
pip install -r requirements.txt
bash scripts/reproduce_figure1.sh        # → figures/coverage_gap.pdf  (< 5 s)
```

---

## Repository layout

```
.
├── src/                            ~200-line ReAct + four probes + analysis
│   ├── react.py                       ReAct loop with three Thought variants
│   ├── tools.py                       Wikipedia REST API + sandboxed calculator
│   ├── data.py                        dataset loaders (via HuggingFace)
│   ├── llm.py                         OpenAI-compatible client (DeepSeek + SiliconFlow + OpenRouter, bf16-pinned)
│   ├── cache.py                       disk-backed LLM response cache (diskcache)
│   ├── cost_tracker.py                token → USD cumulative cost tracker (+ hard-cap budget guard)
│   ├── run.py                         ReAct experiment runner (Qwen / V3 family)
│   ├── run_exp2.py                    ReAct experiment runner (Llama-3.1-8B/70B via OpenRouter)
│   ├── verify_endpoints.py            Post-hoc bf16-endpoint-UUID compliance check on OpenRouter calls
│   ├── probe.py                       Stage 2 — Direct Probe (scaffold layer; 5 models supported)
│   ├── tci.py                         Stage 3 — Thought Causal Influence (mechanism layer; 24 multi-seed + 5 L8B boundary cells)
│   ├── routing.py                     Coverage-Aware Routing (CAR; cross-family validated on Llama-70B)
│   ├── token_analysis.py              token-cost / Pareto analysis
│   ├── bootstrap_ci.py                multi-seed cluster-bootstrap CI
│   ├── analyze.py                     post-hoc analyses (bridge/comparison split, etc.)
│   ├── plot_coverage_gap.py           Figure 1 plotter
│   └── coverage_signals/              §7 — per-query coverage probes
│       ├── ner.py                        spaCy NER + numeric/stop-word filter
│       ├── pageview.py                   Wikipedia REST pageview client (cached)
│       ├── infinigram.py                 InfiniGram count client over Dolma v1.7 (cached)
│       ├── extract_pageview{,_min}.py    head / multi-entity-min entity selection
│       ├── extract_infinigram{,_min}.py  head / multi-entity-min entity selection
│       └── correlation_*_min_vs_head.py  per-cell Pearson/Spearman + bootstrap CI
├── scripts/                        one-command reproduction shells
├── logs/                           raw per-question .jsonl trajectories  (see logs/README.md)
├── results/                        aggregated analysis JSONs              (see results/README.md)
├── experiments/coverage_signals/   §7 raw signals + correlation CSVs
└── figures/coverage_gap.{pdf,png}  Figure 1 from the paper (PDF for paper, 300 DPI PNG preview)
```

---

## Reproducing the paper

### Without API keys (default path)

`logs/` and `results/` are committed in full, so the analysis scripts need nothing else on disk.

| Output | Script | Runtime |
|--------|--------|---------|
| Figure 1 (`figures/coverage_gap.pdf`, 25 cells = 5 models × 5 tasks) | `bash scripts/reproduce_figure1.sh` | < 5 s |
| Table 1 — Thought-Gap with 95% CI (20 cells = 4 primary models × 5 tasks) | `bash scripts/reproduce_table1.sh` | ~30 s |
| Table 4 — Coverage-Aware Routing (5 tasks × 4 models incl. Llama-70B LOTO) | `bash scripts/reproduce_table4_car.sh` | < 10 s |
| Table 7 — Per-query correlation hits (pageview/InfiniGram × head/min) | `bash scripts/reproduce_section7.sh` | < 30 s |
| Token-savings analysis (the 93–99% figure) | `bash scripts/reproduce_token_savings.sh` | < 5 s |

### Re-running experiments end-to-end (requires API keys)

To regenerate the raw `logs/` from scratch, or to test on a new model, you need accounts at the three providers used by the paper.

```bash
cp .env.example .env
# fill in DEEPSEEK_API_KEY      (DeepSeek-V3)
#         SILICONFLOW_API_KEY   (Qwen2.5-*)
#         OPENROUTER_API_KEY    (Llama-3.1-8B / 70B; pinned to DeepInfra bf16)

# Qwen / V3 — one (model, dataset, seed) cell per invocation, resume-safe
python -m src.run --n 100 --model deepseek-chat              --seed 42 --dataset gsm8k
python -m src.run --n 100 --model Qwen/Qwen2.5-14B-Instruct  --seed 42 --dataset webquestions
# ...

# Llama-3.1 (cross-family validation, OpenRouter bf16-pinned, with cost cap)
python -m src.run_exp2 --stage 1 --n 100    # L8B  × 5 tasks × seed=42 (boundary)
python -m src.run_exp2 --stage 4 --n 100    # L70B × 5 tasks × seeds {7,123}
python -m src.verify_endpoints --n 20        # post-hoc bf16 UUID compliance check

# Direct Probe (Stage 2) for any subset of cells (5 model labels supported)
python -m src.probe --models 14B 32B V3 L8B L70B --tasks gsm8k webquestions --n 100 --seed 42
```

The full sweep (5 tasks × 5 models × 3 Thought variants × three seeds, plus Direct Probe and TCI) is feasible on a personal API budget; the paper-grade run cost under USD 6 total. SiliconFlow handles the Qwen2.5 family; DeepSeek handles `deepseek-chat` (V3); OpenRouter (DeepInfra bf16 endpoint, pinned at three layers — `order`, `quantizations`, and `ignore` — with hardcoded expected/forbidden endpoint UUIDs) handles Llama-3.1-8B/70B. All three expose OpenAI-compatible endpoints, dispatched by `src/llm.py` based on model name. Every OpenRouter call is logged to `logs/openrouter_call_audit.jsonl`; `src.verify_endpoints` samples that log post-hoc and asserts each generation hit the expected bf16 endpoint UUID.

#### Section 7 — Per-query coverage probes (no LLM API)

Section 7's external entity-frequency signals come from two public APIs (Wikipedia REST + InfiniGram public endpoint) and require no LLM keys, but do need a one-time spaCy model download:

```bash
python -m spacy download en_core_web_sm

# Wikipedia pageview signal (head + multi-entity min) — 5 tasks × n=50
for task in webquestions triviaqa hotpotqa gsm8k math_hard; do
  python -m src.coverage_signals.extract_pageview      --task "$task" --n 50 --seed 42
  python -m src.coverage_signals.extract_pageview_min  --task "$task" --n 50 --seed 42
done

# InfiniGram counts over Dolma v1.7 — same 5 × 50 questions
for task in webquestions triviaqa hotpotqa gsm8k math_hard; do
  python -m src.coverage_signals.extract_infinigram     --task "$task" --n 50 --seed 42
  python -m src.coverage_signals.extract_infinigram_min --task "$task" --n 50 --seed 42
done

# Correlate against per-question Direct EM and sign(Thought-Gap) — produces Table 7
bash scripts/reproduce_section7.sh
```

Both APIs are cached on disk under `.cache/` (via `src/cache.py`); reruns are instant.

> **Note on MATH-hard:** signal jsonl files for MATH-hard are committed (used to compute the cross-task median reported in the paper), but per-cell correlations against per-question Direct EM are not computable from the public artifacts: the committed `logs/probe_direct_*_math_hard_*.jsonl` predate a hash-stability fix and use a process-random ID space, whereas the §7 signal extraction uses a stable SHA-256 ID. This is the "three MATH-hard cells lack sufficient data" caveat in the paper.

---

## Method (one-paragraph overview)

The paper diagnoses ReAct in three stages of decreasing abstraction:

1. **Behavioural layer (Stage 1).** *Thought-Gap* = EM(full ReAct) − EM(no-Thought). A large positive Gap means Thought is causally helpful. Reported per (model × task) cell with cluster-bootstrap 95% CIs over question IDs across three seeds (seeds 42 / 7 / 123 for all four primary models — Qwen-14B, Qwen-32B, V3, Llama-70B; Llama-8B uses a single seed as a weak-end boundary check).
2. **Scaffold layer (Stage 2).** *Direct Probe* asks the same question with no ReAct loop and no tools, on the same three-seed grid (single seed for Llama-8B). If Direct EM ≈ Full EM, the scaffold is redundant; if Direct EM > Full EM, the scaffold is *net-harmful.*
3. **Mechanism layer (Stage 3).** *Thought Causal Influence* (TCI v2) injects mismatched, scrambled, empty, and adversarial Thoughts, then measures whether the model's next Action follows the Question or the (corrupted) Thought. 24 multi-seed mechanism-layer cells (four primary models × five tasks × three seeds) plus 5 Llama-8B single-seed cells; cross-seed standard errors on `follows_question` stay below 0.05 throughout.

The downstream artefact, **Coverage-Aware Routing (CAR)**, uses calibration-set Direct EM as a coverage estimate and routes between Direct and Full ReAct per task, recovering up to +0.225 EM while saving 93–99% Thought tokens.

---

## Key findings

- **Low-coverage tasks.** On GSM8K, Thought-Gap stays positive across all four primary models (14B: +0.587, 32B: +0.180, V3: +0.073, **Llama-3.1-70B: +0.443**) — Thought scaffolding remains load-bearing across both model scale and model family.
- **High-coverage tasks.** On WebQuestions and TriviaQA, Qwen2.5-32B's Thought-Gap is not statistically distinguishable from zero (the cleanest instance of Capacity Domination in the paper). DeepSeek-V3 retains a small but significant gap (+0.047 / +0.067), consistent with a two-stage collapse where scaffold degradation precedes thought-level collapse.
- **Cross-family validation: Llama-3.1-70B traverses the coverage spectrum strictly monotonically** — Thought-Gap decays +0.443 → +0.077 → +0.010 → −0.020 → −0.040 across GSM8K → MATH-hard → HotpotQA → WebQ → TriviaQA. 70B is the only single model in the matrix that crosses zero into negative-Gap territory, ruling out Qwen/DeepSeek-family idiosyncrasies. CAR's `τ=0.30` threshold reproduces on 70B (all five LOTO folds give `τ*∈{0.15, 0.25}`, well inside the Qwen/V3 range); the 70B × TriviaQA cell yields the largest scaffold reversal in the paper (Direct − Full = +0.143 across three seeds).
- **Direct > Full on high-coverage tasks.** Bypassing ReAct entirely outperforms full ReAct by up to 4× EM on WebQuestions, exposing scaffold degradation that goes beyond mere Thought redundancy.
- **TCI follow-question rates above 0.85** on high-coverage tasks across 24 multi-seed mechanism-layer cells (Qwen-14B/32B/V3 + Llama-70B × 5 tasks × 3 seeds): models effectively bypass the Thought content while preserving near-100% question-following. Cross-seed standard errors stay below 0.05 on the `follows_question` metric, ruling out the single-seed prompt-sensitivity concern.
- **Weak-end boundary: Llama-3.1-8B reproduces Lanham's CoT-fidelity collapse.** On GSM8K, 8B exhibits Gap=−0.08 (Thought *hurts*), TCI follow-question = 7.1%, and the action-similarity-to-misdirection (sim_to_wrong) = 0.875 — i.e. the model parrots the corrupted Thought almost verbatim. This is the mirror image of right-end Capacity Domination: weak-end *Thought hijacks Action*, right-end *Question hijacks Action*; both end-states make the Thought-Gap shrink, by opposite mechanisms.
- **Per-query routing via popularity proxies yields a symmetric null** (Section 7). Across 12 (model, task) cells, neither Wikipedia pageview nor InfiniGram-on-Dolma counts — under either head-entity or multi-entity-min selection — produce per-question correlations beyond chance: 3 of 48 nominal hits at α=0.05 (chance expectation ≈ 2.4). Coverage-Aware Routing's per-task estimator stands; lifting it to per-query needs a different signal class.

---

## License

Released under the [MIT License](LICENSE).
