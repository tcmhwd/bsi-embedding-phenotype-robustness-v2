#!/bin/bash
#SBATCH --job-name=t23b_retr
#SBATCH --partition=cpu_partition
#SBATCH --array=0-7%4
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/23b_retrieval_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/23b_retrieval_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Task 23b -- retrieval AUROC, one (representation, variant) combo per array
# task. All four k in {5,10,25,50} are computed within a single invocation
# (see 23b_run_retrieval.py docstring for why). %4 caps concurrency at
# 4*8=32 cores, well under this project's node2 budget headroom.
declare -a REPRESENTATIONS=(openai-3large openai-3large openai-3small openai-3small bge-m3 bge-m3 gower gower)
declare -a VARIANTS=(unrestricted past_only unrestricted past_only unrestricted past_only unrestricted past_only)

I="${SLURM_ARRAY_TASK_ID}"
REPRESENTATION="${REPRESENTATIONS[$I]}"
VARIANT="${VARIANTS[$I]}"
echo "[wrapper] representation=${REPRESENTATION} variant=${VARIANT} cpus=${SLURM_CPUS_PER_TASK}"

/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/23b_run_retrieval.py \
  --representation "${REPRESENTATION}" --variant "${VARIANT}"
