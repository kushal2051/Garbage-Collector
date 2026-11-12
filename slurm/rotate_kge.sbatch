#!/bin/bash
#SBATCH --job-name=kg_rotate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --partition=tropshalab,l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --output=logs/rotate_kge.%j.log
#SBATCH --error=logs/rotate_kge.%j.log
#
# Stage 5d. Submit from the repo root, so --output resolves. Arguments pass straight through:
#
#   sbatch slurm/rotate_kge.sbatch --graph pharos319 \
#          --predicate biolink:genetically_associated_with \
#          --split out/split_gawc319_pairs.tsv --tag pharos319_gawc319_rotate --seeds 1
#
# Run the SAME command with --graph kg --predicate biolink:gene_associated_with_condition
# --tag kg_gawc319_rotate for the other side. The two are comparable to each other, and to the
# stage 5c GraphSAGE runs, only because all of them use the same --split file and seeds.
#
# Budget: the merged graph is ~1,390 optimiser steps an epoch at --batch 8192 with 128
# negatives each, plus one filtered ranking of the validation pairs against all 1.2M entities
# per epoch. Pharos v319 is 16x smaller. Time the small graph first.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}/slurm/env.sh"
nvidia-smi || true
$PYTHON_ROTATE -s -m rotate_kge "$@"
