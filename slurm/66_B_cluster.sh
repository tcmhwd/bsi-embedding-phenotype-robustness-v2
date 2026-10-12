#!/bin/bash
#SBATCH --job-name=bsi_66_B_cluster
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/66_B_cluster_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/66_B_cluster_%A_%a.err
#SBATCH --array=0-2%3
set -euo pipefail
# Analysis B: 11 UMAP+HDBSCAN fits (seeds 42, 1..10) on 32,329 x 1536 per arm,
# scored against 3-small's own whole-corpus reference partition (k=22, 277 noise).
#   task 0 = B1b  (needs the B1b embedding: submit with --dependency=aftercorr on 65_B_embed)
#   task 1 = B2   (needs the B2 embedding: same)
#   task 2 = B1a  (existing stored embedding: no dependency)
#   task 3 = aggregate (needs all three arms: submit with --array=3 and afterok on the others)
#
# Measured precedent: 49_3small_no_department fitted 22 such fits in ~10-15 min
# (20-64 s each), so 2 h per task is >8x allowance. Threads are pinned to 1 inside
# the script before any numeric import and the script does not read
# SLURM_CPUS_PER_TASK, so cpus-per-task=4 is driver headroom only (as in 47/49).
# Each fit's labels are written before the next begins; resubmission resumes.
# FROZEN environment: the only stack that reproduces this project's partitions.
# It is read-only; nothing is installed into it.
declare -a TASKS=(B1b B2 B1a aggregate)
T="${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "[wrapper] task=${T}"
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
PY="/home/programs/anaconda3/envs/gpt/bin/python3"
if [ "${T}" = "aggregate" ]; then
  "${PY}" scripts/66_B_cluster.py --stage aggregate
else
  "${PY}" scripts/66_B_cluster.py --stage cluster --arm "${T}"
fi
