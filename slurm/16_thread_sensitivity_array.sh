#!/bin/bash
#SBATCH --job-name=c1_thread_sens
#SBATCH --partition=cpu_partition
#SBATCH --array=0-3%2
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/16_thread_sens_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/16_thread_sens_%A_%a.err
set -euo pipefail

declare -a THREADS=(1 4 8 24)
T="${THREADS[$SLURM_ARRAY_TASK_ID]}"

export OMP_NUM_THREADS="${T}"
export NUMBA_NUM_THREADS="${T}"
export MKL_NUM_THREADS="${T}"
export OPENBLAS_NUM_THREADS="${T}"
export NUMBA_CACHE_DIR="/tmp/c1_thread_sens_numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${NUMBA_CACHE_DIR}"

/home/programs/anaconda3/envs/gpt/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/16_thread_sensitivity_original_env.py
