#!/bin/bash
#SBATCH --job-name=ocsvm_plot_rbf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --partition=tropshalab
#SBATCH --output=logs/plot_rbf.%j.log
#SBATCH --error=logs/plot_rbf.%j.log
#
# Submit from the repo root, so --output resolves: sbatch slurm/plot_rbf.sbatch
# Another partition: sbatch -p general slurm/plot_rbf.sbatch  (the flag wins over the
# directive above; Slurm does not expand shell variables in #SBATCH lines).
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
$PYTHON -s plot_rbf.py "$@"
