#!/bin/bash
#SBATCH --job-name=bsi_49_3small_cl
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/49_3small_cluster_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/49_3small_cluster_%j.err
set -euo pipefail
# 22 UMAP+HDBSCAN fits (2 arms x 11 seeds) on 32,329 x 1536, scored against
# 3-small's own whole-corpus partition. Same shape and seed count as the other
# two encoders' ablation arms, so the three are directly comparable.
# Threads pinned to 1 inside the script before any numeric import; cpus=4 is
# driver headroom only. Resumable: each fit's labels are written before the
# next begins. FROZEN environment -- the only stack that reproduces this
# project's partitions.
cd /data/tcmhwd/AI/AST_ai/phenotype_revision
/home/programs/anaconda3/envs/gpt/bin/python3 \
  scripts/49_3small_no_department.py --stage cluster
