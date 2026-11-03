#!/bin/bash
#SBATCH --job-name=ocsvm_extract
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --partition=tropshalab
#SBATCH --output=logs/extract.%j.log
#SBATCH --error=logs/extract.%j.log
#
# Submit from the repo root, so --output resolves: sbatch slurm/extract.sbatch
# Another partition: sbatch -p general slurm/extract.sbatch  (the flag wins over the
# directive above; Slurm does not expand shell variables in #SBATCH lines).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
$PYTHON -s extract_edges.py
