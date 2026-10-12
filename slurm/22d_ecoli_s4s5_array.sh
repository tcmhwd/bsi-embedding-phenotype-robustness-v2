#!/bin/bash
#SBATCH --job-name=t22d_s4s5
#SBATCH --partition=cpu_partition
#SBATCH --array=0-103%8
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/22d_s4s5_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/22d_s4s5_%A_%a.err
set -euo pipefail

export NUMBA_CACHE_DIR="/tmp/t22d_s4s5_numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${NUMBA_CACHE_DIR}"

# 104 runs = 4 observed directions + 4*25 random-patient-split nulls.
/home/programs/anaconda3/envs/gpt/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/22d_ecoli_s4s5_transport.py \
  --run-index "${SLURM_ARRAY_TASK_ID}"
