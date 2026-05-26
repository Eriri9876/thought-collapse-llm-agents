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
# Note on MATH-hard: the committed logs/probe_direct_*_math_hard_*.jsonl and
# logs/pilot_*_math_hard_*.jsonl predate a hash-stability fix and use the
# original process-random ID space; the §7 signal files (n=100 for math_hard)
# use stable SHA-256 IDs. correlation_*_min_vs_head.py re-keys every MATH-hard
# record by sha256(question)[:8] at load time (_canonical_id) so the two
# spaces join, yielding 15 evaluable cells (5 tasks × 3 models) and 120
# Pearson tests in total — all three MATH cells included.

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
