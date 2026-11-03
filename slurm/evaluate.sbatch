#!/bin/bash
#SBATCH --job-name=ocsvm_evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --partition=tropshalab
#SBATCH --output=logs/evaluate.%j.log
#SBATCH --error=logs/evaluate.%j.log
#
# Submit from the repo root, so --output resolves: sbatch slurm/evaluate.sbatch
# Another partition: sbatch -p general slurm/evaluate.sbatch  (the flag wins over the
# directive above; Slurm does not expand shell variables in #SBATCH lines).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
$PYTHON -s evaluate.py "$@"
