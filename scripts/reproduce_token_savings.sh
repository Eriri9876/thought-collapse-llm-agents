#!/usr/bin/env bash
# Reproduce token-savings analysis from the paper:
#   per-cell EM / steps / generated tokens, Pareto analysis (token overhead
#   vs. Thought-Gap), and Thought token budget per task. The 93-99% token
#   savings figure cited in the abstract is computed here.
#
# Inputs:  logs/pilot_*.jsonl  (already committed — no API calls)
# Output:  stdout — three summary tables
#          results/token_analysis.json  (overwritten in place)
# API:     none required
# Runtime: < 5 seconds

set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.token_analysis

echo
echo "Done. Per-cell numbers saved to results/token_analysis.json"
