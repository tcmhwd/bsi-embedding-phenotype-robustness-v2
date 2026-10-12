#!/bin/bash
#SBATCH --job-name=bsi_27b_abx
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27b_mimic_abx_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27b_mimic_abx_%j.err
set -euo pipefail
# Chunked scan of prescriptions.csv (3.3G) filtered to cohort hadm_id set +
# an antibiotic-keyword regex.
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/27b_mimic_enrich.py --stage abx
