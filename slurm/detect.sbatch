#!/bin/bash
#SBATCH --job-name=ocsvm_detect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --partition=tropshalab
#SBATCH --output=logs/detect.%j.log
#SBATCH --error=logs/detect.%j.log
#
# Submit from the repo root, so --output resolves: sbatch slurm/detect.sbatch
# Another partition: sbatch -p general slurm/detect.sbatch  (the flag wins over the
# directive above; Slurm does not expand shell variables in #SBATCH lines).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
PRED="${1:-biolink:treats}"
NU="${2:-0.05}"
$PYTHON -s detect.py "$PRED" "$NU"
$PYTHON -s plot_explain.py "${PRED##*:}"
