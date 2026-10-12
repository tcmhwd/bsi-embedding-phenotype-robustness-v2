#!/bin/bash
#SBATCH --job-name=c1_rederive
#SBATCH --partition=cpu_partition
#SBATCH --array=0-23%8
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/12_codex_c1/logs/c1_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/12_codex_c1/logs/c1_%A_%a.err
set -euo pipefail

export NUMBA_CACHE_DIR="/tmp/c1_numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${NUMBA_CACHE_DIR}"

if (( SLURM_ARRAY_TASK_ID <= 20 )); then
  SOURCE=current
  if (( SLURM_ARRAY_TASK_ID == 20 )); then
    SEED=42
  else
    SEED=$((SLURM_ARRAY_TASK_ID + 1))
  fi
elif (( SLURM_ARRAY_TASK_ID >= 22 )); then
  SOURCE=contemporaneous
  SEED=42
else
  SOURCE=current
  SEED=42
fi

if (( SLURM_ARRAY_TASK_ID == 21 || SLURM_ARRAY_TASK_ID == 22 )); then
  PYTHON=/data/tcmhwd/miniconda3/envs/gpt/bin/python3
else
  PYTHON=/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
fi

"${PYTHON}" \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/out/12_codex_c1/c1_cluster.py \
  --source "${SOURCE}" --seed "${SEED}"
