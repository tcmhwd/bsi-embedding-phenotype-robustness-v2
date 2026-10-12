#!/bin/bash
#SBATCH --job-name=bsi_27b_comorbid
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27b_mimic_comorbid_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27b_mimic_comorbid_%j.err
set -euo pipefail
# diagnoses_icd.csv (174M) read fully then filtered to cohort hadm_ids --
# small, single pass.
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/27b_mimic_enrich.py --stage comorbid
