#!/usr/bin/env bash
# Reproduce Figure 1 (scaffold gap vs. coverage estimate) from the paper.
#
# Inputs:  paper-reported values embedded in src/plot_coverage_gap.py
# Output:  figures/coverage_gap.pdf
# API:     none required
# Runtime: < 5 seconds

set -euo pipefail
cd "$(dirname "$0")/.."

python -m src.plot_coverage_gap

echo
echo "Done. Figure saved to figures/coverage_gap.pdf"
