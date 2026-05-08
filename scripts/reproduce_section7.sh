#!/usr/bin/env bash
# Reproduce Table 7 from the paper: Per-query coverage probes (symmetric null).
#
# Reads pre-extracted entity-frequency signals from
#   experiments/coverage_signals/outputs/{pageview,pageview_min,infinigram,infinigram_min}_<task>_n50_seed42.jsonl
# and computes per-cell Pearson + Spearman + bootstrap CI against
#   - per-question Direct EM
#   - sign(Thought-Gap)
# for the four (proxy, rule) combinations in Section 7.
#
# Inputs:  experiments/coverage_signals/outputs/*.jsonl  (committed)
#          logs/probe_direct_*.jsonl + logs/pilot_*.jsonl  (committed)
# Output:  stdout — two per-target tables
#          experiments/coverage_signals/outputs/correlation_min_vs_head.csv
#          experiments/coverage_signals/outputs/correlation_infinigram_min_vs_head.csv
#          (overwritten in place)
# API:     none required (raw signal extraction was done offline; see README §7)
# Runtime: < 30 seconds
#
# Note on MATH-hard: signal files exist (used for cross-task median quoted in
# the paper) but per-cell correlation reports "insufficient data" because the
# committed logs/probe_direct_*_math_hard_*.jsonl use the original
# process-random ID space, whereas signal extraction (run later) uses a stable
# SHA-256 ID. The two ID spaces don't join. This is reflected in the paper's
# "12 evaluable cells (three MATH-hard cells lack sufficient data)" wording.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "── Pageview channel (head + min) ─────────────────────────────────────────"
python -m src.coverage_signals.correlation_min_vs_head

echo
echo "── InfiniGram channel (head + min) ───────────────────────────────────────"
python -m src.coverage_signals.correlation_infinigram_min_vs_head

echo
echo "Done. Both correlation CSVs saved under experiments/coverage_signals/outputs/."
echo "Table 7 numbers: count rows where pearson_p < 0.05, grouped by (signal, target)."
