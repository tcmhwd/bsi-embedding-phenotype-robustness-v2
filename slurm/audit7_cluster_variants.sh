#!/bin/bash
#SBATCH --job-name=audit7_cluster
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/audit7_cluster_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/audit7_cluster_%j.err
set -euo pipefail
# Frozen environment, executed only (never modified). Threads pinned to 1 in-script;
# 4 CPUs requested for I/O and headroom. 15 UMAP fits on 32,329 x 1024.
/home/programs/anaconda3/envs/gpt/bin/python3 /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/audit7_cluster_variants.py
