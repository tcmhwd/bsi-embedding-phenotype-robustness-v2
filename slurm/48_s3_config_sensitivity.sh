#!/bin/bash
#SBATCH --job-name=bsi_48_s3
#SBATCH --partition=cpu_partition
#SBATCH --array=0-21%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/48_s3_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/48_s3_%A_%a.err
set -euo pipefail
# The S3 axis (PRESPEC section 2), closing deviation D2.
#
# Array layout: tasks 0-20 are the 21 A1 fits (seed 42, then seeds 1-20);
# task 21 is the pair of published-2-D fits at min_cluster_size 100 and 200.
#
# Memory is 48G rather than the 32G the 2-D arms use because A1 reduces to
# n_components=20 with metric="cosine" on 32,329 x 3,072 -- a materially
# larger intermediate than the 2-D euclidean configuration of record -- and
# an OOM here would waste the whole array slot. Time is 8 h for the same
# reason: the 20-component cosine UMAP has no measured precedent in this
# repository, so the limit is set from the shape rather than from a prior run.
#
# Threads are pinned to 1 inside the script before any numeric import (this
# repo's standing gotcha: env vars set after numpy/numba import do nothing),
# so cpus-per-task=4 is driver headroom only and the script does not read
# SLURM_CPUS_PER_TASK. Throttled to 3 concurrent tasks because node2 is
# shared with other work.
#
# Every stage is resumable: a task whose result JSON already exists exits
# immediately, so a timeout or a partial array can simply be resubmitted.
#
# FROZEN environment: the only stack that reproduces this project's partitions.
PY=/home/programs/anaconda3/envs/gpt/bin/python3
S=/data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/48_s3_config_sensitivity.py
I="${SLURM_ARRAY_TASK_ID}"

declare -a A1_SEEDS=(42 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

if [ "${I}" -lt 21 ]; then
  SEED="${A1_SEEDS[$I]}"
  echo "[wrapper] stage=a1 seed=${SEED}"
  "${PY}" "${S}" --stage a1 --seed "${SEED}"
else
  echo "[wrapper] stage=mcs (published 2-D setting, mcs 100 and 200)"
  "${PY}" "${S}" --stage mcs
fi
