#!/bin/bash
#SBATCH --job-name=c1_orig_env
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/16_reproduce_original_env_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/16_reproduce_original_env_%j.err
set -euo pipefail

export NUMBA_CACHE_DIR="/tmp/c1_orig_env_numba_${SLURM_JOB_ID}"
mkdir -p "${NUMBA_CACHE_DIR}"

# Deliberately NOT `conda activate` -- invoke the frozen original
# execution environment's interpreter directly, read-only.
/home/programs/anaconda3/envs/gpt/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/16_reproduce_original_env.py
