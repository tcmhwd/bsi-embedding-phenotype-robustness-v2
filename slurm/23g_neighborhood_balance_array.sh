#!/bin/bash
#SBATCH --job-name=t23g_balance
#SBATCH --partition=cpu_partition
#SBATCH --array=0-3%2
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/23g_balance_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/23g_balance_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Task 23g -- neighbourhood confounding-by-indication diagnostic, one
# (representation, variant) combo per array task. Computes all of
# k in {10,25,50} x all 3 empiric-treatment splits within one invocation
# (reuses the same per-fold top-k neighbour computation across all of
# them). %2 caps concurrency at 2*8=16 cores.
declare -a REPRESENTATIONS=(openai-3large openai-3large gower gower)
declare -a VARIANTS=(unrestricted past_only unrestricted past_only)

I="${SLURM_ARRAY_TASK_ID}"
REPRESENTATION="${REPRESENTATIONS[$I]}"
VARIANT="${VARIANTS[$I]}"
echo "[wrapper] representation=${REPRESENTATION} variant=${VARIANT} cpus=${SLURM_CPUS_PER_TASK}"

/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/23g_run_neighborhood_balance.py \
  --representation "${REPRESENTATION}" --variant "${VARIANT}"
