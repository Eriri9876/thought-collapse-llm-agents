#!/usr/bin/env bash
# Reproduce Table 1 from the paper:
#   Per-cell EM (full / none / compressed) and Thought-Gap with 95% CI.
#   Multi-seed cells use cluster bootstrap on question IDs (resampling clusters
#   of [seed=42, seed=7, seed=123] observations together) to respect within-
#   question correlation.
#
# Inputs:  logs/pilot_*.jsonl  (already committed in this repo — no API calls)
# Output:  stdout — printed table covering HotpotQA / GSM8K / WebQ / TriviaQA / MATH-hard
# API:     none required
# Runtime: ~30 seconds  (2000 bootstrap iterations × 15 cells)

set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.bootstrap_ci
