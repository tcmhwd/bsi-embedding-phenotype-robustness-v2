#!/bin/bash
#SBATCH --job-name=c1_orig_env_arr
#SBATCH --partition=cpu_partition
#SBATCH --array=0-11%6
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/16_orig_env_arr_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/16_orig_env_arr_%A_%a.err
set -euo pipefail

export NUMBA_CACHE_DIR="/tmp/c1_orig_env_arr_numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${NUMBA_CACHE_DIR}"

# Task map:
#   0-4  : contemporaneous, seeds 1,2,3,4,5   (stability across seeds, published corpus)
#   5-9  : current,         seeds 1,2,3,4,5   (corpus-specificity, current corpus)
#   10   : current,         seed 42           (match the seed already used contemporaneously)
#   11   : contemporaneous, seed 7            (extra stability point)
declare -a SOURCES=(contemporaneous contemporaneous contemporaneous contemporaneous contemporaneous \
                     current current current current current \
                     current contemporaneous)
declare -a SEEDS=(1 2 3 4 5 1 2 3 4 5 42 7)

SOURCE="${SOURCES[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

/home/programs/anaconda3/envs/gpt/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/16_reproduce_original_env_multi.py \
  --source "${SOURCE}" --seed "${SEED}"
