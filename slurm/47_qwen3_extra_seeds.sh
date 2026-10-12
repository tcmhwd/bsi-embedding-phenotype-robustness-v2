#!/bin/bash
#SBATCH --job-name=bsi_47_qwen3_seeds
#SBATCH --partition=cpu_partition
#SBATCH --array=0-4%3
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/47_qwen3_seeds_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/47_qwen3_seeds_%A_%a.err
set -euo pipefail
# One array task per Qwen3 arm (full / no_therapy / no_organism /
# mask_ab1_name_only / mask_ab_names_all), 11 UMAP+HDBSCAN fits each on
# 32,329 x 1024 -- the same shape as job 588163, which did 12 fits inside 3 h,
# so 6 h is roughly double headroom. Threads are pinned to 1 inside the script
# before any numeric import, so cpus-per-task=4 is driver headroom only; the
# script does not read SLURM_CPUS_PER_TASK and raising it would achieve
# nothing. Throttled to 3 concurrent tasks because node2 is shared.
# Each fit is written to disk before the next begins, so a timeout loses at
# most one fit and a resubmission resumes from the cache.
# FROZEN environment: the only stack that reproduces this project's partitions.
/home/programs/anaconda3/envs/gpt/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/47_qwen3_extra_seeds.py \
  --stage fit --arm "${SLURM_ARRAY_TASK_ID}"
