#!/bin/bash
#SBATCH --job-name=t17_s1
#SBATCH --partition=cpu_partition
#SBATCH --array=0-21%10
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/17a_s1_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/17a_s1_%A_%a.err
set -euo pipefail

# Index map: 0-19 -> seed=(i+1), threads=1 (S1 seed axis, seeds 1..20)
#            20   -> seed=42, threads=1  (S1 thread axis, threads=1 arm)
#            21   -> seed=42, threads=24 (S1 thread axis, threads=24 arm)
I="${SLURM_ARRAY_TASK_ID}"
if [ "$I" -le 19 ]; then
  SEED=$((I + 1))
  THREADS=1
elif [ "$I" -eq 20 ]; then
  SEED=42
  THREADS=1
else
  SEED=42
  THREADS=24
fi

export STRESS_THREADS="${THREADS}"
export NUMBA_CACHE_DIR="/tmp/t17_s1_numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${NUMBA_CACHE_DIR}"

echo "[wrapper] seed=${SEED} threads=${THREADS}"
/home/programs/anaconda3/envs/gpt/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/17a_s1_seed_thread.py --seed "${SEED}"
