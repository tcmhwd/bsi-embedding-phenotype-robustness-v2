#!/bin/bash
#SBATCH --job-name=bsi_44_ablation
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/44_ablation_cluster_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/44_ablation_cluster_%j.err
set -euo pipefail
# 12 UMAP+HDBSCAN fits (3 variants x 4 seeds) on 32,329 x 1024. Threads are
# pinned to 1 inside the script before any numeric import, so cpus-per-task=4
# is headroom for the driver only. Runs in the FROZEN environment because
# that is the only stack that reproduces this project's published partitions.
/home/programs/anaconda3/envs/gpt/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/44_ablation_cluster.py
