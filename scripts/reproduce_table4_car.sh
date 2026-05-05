#!/usr/bin/env bash
# Reproduce Table 4 from the paper: Coverage-Aware Routing (CAR).
#
# For each (model, task) cell, simulates routing between Direct mode and
# Full ReAct using calibration-set Direct EM as the coverage estimator.
# Reports per-cell routed EM vs. baselines, threshold sensitivity, token
# savings, oracle gap, and leave-one-task-out τ selection.
#
# Inputs:  logs/pilot_*.jsonl + logs/probe_direct_*.jsonl  (already committed)
# Output:  stdout — five summary tables
#          results/routing_analysis.json  (overwritten in place)
# API:     none required
# Runtime: < 10 seconds

set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.routing

echo
echo "Done. Detailed numbers saved to results/routing_analysis.json"
