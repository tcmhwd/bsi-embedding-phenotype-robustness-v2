#!/bin/bash
#SBATCH --job-name=bsi_27b_merge
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27b_mimic_merge_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27b_mimic_merge_%j.err
set -euo pipefail
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/27b_mimic_enrich.py --stage merge
