#!/bin/bash
#SBATCH --job-name=bsi_66b_B3_cluster
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/66b_B3_cluster_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/66b_B3_cluster_%A_%a.err
#SBATCH --array=0-10%4
set -euo pipefail
# Analysis B, arm B3 (Qwen3 no_department): 11 UMAP+HDBSCAN fits (seeds 42, 1..10) on 32,329 x 1024,
# scored against Qwen3's own whole-corpus reference partition (k=24, 851 noise).
#   array tasks 0-10 = one fit per seed, in the order of 47_qwen3_extra_seeds.SEEDS = [42,1,2,...,10]
#   array task  11   = aggregate (submit separately, see below)
#
# Resources: measured fits on this shape/stack take 20-50 s each (B1a per-run JSON, same frozen env, 1536-d);
# a task = one fit + imports + numba compile, ~2 min, so 30 min is >10x allowance. Threads are pinned to 1
# inside the script before any numeric import and the script does not read SLURM_CPUS_PER_TASK, so
# cpus-per-task=2 is driver headroom only. 16 GB: the data are 132 MB; 47/49 requested 32 GB for 11 SEQUENTIAL
# fits, here each task holds one. Throttle %4: the cpu node is shared.
# FROZEN environment: the only stack that reproduces this project's partitions (read-only, nothing installed).
#
# Aggregate is submitted AFTER the array with a larger allocation (field AMI over 35 fields x 11 fits, the
# 15-NN readout on two 32,329 x 1024 matrices at 1 thread):
#   sbatch --dependency=afterok:<ARRAY_JOBID> --array=11 --time=02:00:00 --mem=32G slurm/66b_B3_cluster.sh
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
PY="/home/programs/anaconda3/envs/gpt/bin/python3"
if [ "${SLURM_ARRAY_TASK_ID}" = "11" ]; then
  echo "[wrapper] task=aggregate"
  "${PY}" scripts/66b_B3_cluster.py --stage aggregate
else
  echo "[wrapper] task=cluster seed-index=${SLURM_ARRAY_TASK_ID}"
  "${PY}" scripts/66b_B3_cluster.py --stage cluster --seed-index "${SLURM_ARRAY_TASK_ID}"
fi
