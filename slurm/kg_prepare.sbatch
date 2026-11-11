#!/bin/bash
#SBATCH --job-name=kg_prepare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --partition=tropshalab
#SBATCH --output=logs/kg_prepare.%j.log
#SBATCH --error=logs/kg_prepare.%j.log
#
# Stage 5a. Submit from the repo root: sbatch slurm/kg_prepare.sbatch --tag kg
#
# 6.5 GB of jsonl parsed line by line, so this is minutes, not seconds, and the node table and
# the 12M-row dedup are what the memory is for. CPU only.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
$PYTHON -s kg_prepare.py "$@"
