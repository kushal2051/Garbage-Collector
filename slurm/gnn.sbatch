#!/bin/bash
#SBATCH --job-name=kg_gnn
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --output=logs/gnn.%j.log
#SBATCH --error=logs/gnn.%j.log
#
# Stage 5c. Submit from the repo root, so --output resolves. Arguments pass straight through:
#
#   sbatch slurm/gnn.sbatch --graph kg --predicate biolink:genetically_associated_with \
#          --split out/split_genetically_associated_with_pairs.tsv --tag kg_gaw
#
# Run the SAME command with --graph pharos --tag pharos_gaw for the other side of the
# comparison. The two are only comparable if both use the same --split file.
#
# Another partition: sbatch -p volta-gpu slurm/gnn.sbatch ...  (the flag wins over the
# directive above; Slurm does not expand shell variables in #SBATCH lines).
# No GPU available: drop --gres and pass --device cpu; the sampler bounds the batch either way.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
nvidia-smi || true
$PYTHON_GNN -s -m gnn "$@"
