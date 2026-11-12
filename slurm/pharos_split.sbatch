#!/bin/bash
#SBATCH --job-name=pharos_split
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=general,tropshalab
# CPU only, so it also takes `general`, which is usually idle. NOT l40-gpu: that partition
# requires --qos=gpu_access, and that QOS requires a GPU (QOSMinGRES), so a CPU-only job
# listing it is rejected outright with "Invalid qos specification".
#SBATCH --output=logs/pharos_split.%j.log
#SBATCH --error=logs/pharos_split.%j.log
#
# Stage 5b. Both tags must be prepared first:
#   sbatch slurm/kg_prepare.sbatch --tag kg
#   sbatch slurm/kg_prepare.sbatch --tag pharos
#   sbatch slurm/pharos_split.sbatch --predicate biolink:genetically_associated_with
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
$PYTHON -s pharos_split.py "$@"
