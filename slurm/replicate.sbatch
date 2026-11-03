#!/bin/bash
#SBATCH --job-name=ocsvm_replicate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --partition=tropshalab
#SBATCH --output=logs/replicate.%j.log
#SBATCH --error=logs/replicate.%j.log
#
# Submit from the repo root, so --output resolves: sbatch slurm/replicate.sbatch
# Another partition: sbatch -p general slurm/replicate.sbatch  (the flag wins over the
# directive above; Slurm does not expand shell variables in #SBATCH lines).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
$PYTHON_UMAP -s replicate_original.py "$@"
