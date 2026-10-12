#!/bin/bash
#SBATCH --job-name=bsi_a9_mask_cluster
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/audit9_cluster_mask_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/audit9_cluster_mask_%j.err
set -euo pipefail
# 8 UMAP+HDBSCAN fits (2 variants x 4 seeds) on 32,329 x 1024 in the FROZEN
# environment; threads pinned to 1 inside the script (mirrors slurm/44).
/home/programs/anaconda3/envs/gpt/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/audit9_cluster_mask.py
