#!/bin/bash
#SBATCH --job-name=bsi_45_3large_seeds
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/45_3large_seeds_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/45_3large_seeds_%j.err
set -euo pipefail
# 16 UMAP+HDBSCAN fits (4 variants x 4 seeds) on 32,329 x 3,072 float32
# (~379 MB per array). Threads pinned to 1 inside the script before any
# numeric import; cpus-per-task=4 is driver headroom. Frozen environment,
# published configuration verbatim.
/home/programs/anaconda3/envs/gpt/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/45_3large_ablation_seeds.py
