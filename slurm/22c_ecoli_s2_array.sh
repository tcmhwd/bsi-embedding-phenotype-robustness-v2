#!/bin/bash
#SBATCH --job-name=t22c_s2
#SBATCH --partition=cpu_partition
#SBATCH --array=0-19%8
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/22c_s2_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/22c_s2_%A_%a.err
set -euo pipefail

export NUMBA_CACHE_DIR="/tmp/t22c_s2_numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${NUMBA_CACHE_DIR}"

# 20 shards x 10 reps = 200 boot reps (n_boot=200).
/home/programs/anaconda3/envs/gpt/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/22c_ecoli_s2_bootstrap.py \
  --shard-index "${SLURM_ARRAY_TASK_ID}" --n-shards 20
