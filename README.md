# Thought Collapse in LLM Agents

*A Cross-Task Diagnosis of Capacity Domination in Agentic Reasoning*

> A formal paper link will be added once available.

ReAct's *Thought* step is widely assumed to improve agentic reasoning. On tasks well-covered by pre-training, this paper finds the opposite: **removing the Thought scaffold can improve EM by up to 4× (Qwen2.5-14B on WebQuestions) while saving 93–99% of generated tokens.** This repository contains:

- a ~200-line plain-Python ReAct stack (`src/react.py` + `src/tools.py` + `src/llm.py`, no LangChain),
- per-question trajectories for **5 tasks × 3 models × 3 Thought variants × up to 3 seeds**, fully logged in `logs/`,
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
│   ├── llm.py                         OpenAI-compatible client (DeepSeek + SiliconFlow)
│   ├── run.py                         ReAct experiment runner
│   ├── probe.py                       Stage 2 — Direct Probe (scaffold layer)
│   ├── tci.py                         Stage 3 — Thought Causal Influence (mechanism layer)
│   ├── logprob_probe.py               logprob-based Thought-causality test (V3 only)
│   ├── routing.py                     Coverage-Aware Routing (CAR)
│   ├── token_analysis.py              token-cost / Pareto analysis
│   ├── bootstrap_ci.py                multi-seed cluster-bootstrap CI
│   ├── analyze.py                     post-hoc analyses (bridge/comparison split, etc.)
│   └── plot_coverage_gap.py           Figure 1 plotter
├── scripts/                        one-command reproduction shells
├── logs/                           raw per-question .jsonl trajectories  (see logs/README.md)
├── results/                        aggregated analysis JSONs              (see results/README.md)
└── figures/coverage_gap.pdf        Figure 1 from the paper
```

---

## Reproducing the paper

### Without API keys (default path)

`logs/` and `results/` are committed in full, so the analysis scripts need nothing else on disk.

| Output | Script | Runtime |
|--------|--------|---------|
| Figure 1 (`figures/coverage_gap.pdf`) | `bash scripts/reproduce_figure1.sh` | < 5 s |
| Table 1 — Thought-Gap with 95% CI (15 cells) | `bash scripts/reproduce_table1.sh` | ~30 s |
| Table 4 — Coverage-Aware Routing (5 tasks × 3 models) | `bash scripts/reproduce_table4_car.sh` | < 10 s |
| Token-savings analysis (the 93–99% figure) | `bash scripts/reproduce_token_savings.sh` | < 5 s |

### Re-running experiments end-to-end (requires API keys)

To regenerate the raw `logs/` from scratch, or to test on a new model, you need accounts at the two providers used by the paper.

```bash
cp .env.example .env
# fill in DEEPSEEK_API_KEY (DeepSeek-V3) and SILICONFLOW_API_KEY (Qwen2.5-*)

# one (model, dataset, seed) cell per invocation — resume-safe
python -m src.run --n 100 --model deepseek-chat              --seed 42 --dataset gsm8k
python -m src.run --n 100 --model Qwen/Qwen2.5-14B-Instruct  --seed 42 --dataset webquestions
# ...

# Direct Probe (Stage 2) for any subset of cells
python -m src.probe --models 14B 32B V3 --tasks gsm8k webquestions --n 100 --seed 42
```

The full sweep reported in the paper (5 tasks × 3 models × 3 Thought variants × up to 3 seeds, plus Direct Probe and TCI) is feasible on a personal API budget; SiliconFlow Qwen2.5-32B is the cost-dominant component. SiliconFlow handles the Qwen2.5 family; DeepSeek handles `deepseek-chat` (V3). Both expose OpenAI-compatible endpoints, dispatched by `src/llm.py` based on model name.

---

## Method (one-paragraph overview)

The paper diagnoses ReAct in three stages of decreasing abstraction:

1. **Behavioural layer (Stage 1).** *Thought-Gap* = EM(full ReAct) − EM(no-Thought). A large positive Gap means Thought is causally helpful. Reported per (model × task) cell with cluster-bootstrap 95% CIs over question IDs across three seeds.
2. **Scaffold layer (Stage 2).** *Direct Probe* asks the same question with no ReAct loop and no tools. If Direct EM ≈ Full EM, the scaffold is redundant; if Direct EM > Full EM, the scaffold is *net-harmful.*
3. **Mechanism layer (Stage 3).** *Thought Causal Influence* (TCI) injects mismatched, scrambled, and adversarial Thoughts, then measures whether the model's next Action follows the Question or the (corrupted) Thought.

The downstream artefact, **Coverage-Aware Routing (CAR)**, uses calibration-set Direct EM as a coverage estimate and routes between Direct and Full ReAct per task, recovering up to +0.225 EM while saving 93–99% Thought tokens.

---

## Key findings

- **Low-coverage tasks.** On GSM8K, Thought-Gap stays positive across all three model sizes (14B: +0.587, 32B: +0.180, V3: +0.073) — Thought scaffolding remains load-bearing even at the largest model.
- **High-coverage tasks.** On WebQuestions and TriviaQA, Qwen2.5-32B's Thought-Gap is not statistically distinguishable from zero (the cleanest instance of Capacity Domination in the paper). DeepSeek-V3 retains a small but significant gap (+0.047 / +0.067), consistent with a two-stage collapse where scaffold degradation precedes thought-level collapse.
- **Direct > Full on high-coverage tasks.** Bypassing ReAct entirely outperforms full ReAct by up to 4× EM on WebQuestions, exposing scaffold degradation that goes beyond mere Thought redundancy.
- **TCI follow-question rates above 0.85** on high-coverage tasks: models effectively bypass the Thought content while preserving near-100% question-following.

---

## License

Released under the [MIT License](LICENSE).
