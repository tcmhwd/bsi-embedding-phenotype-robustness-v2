#!/bin/bash
#SBATCH --job-name=bsi_27b_ast
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27b_mimic_ast_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27b_mimic_ast_%j.err
set -euo pipefail
# Re-reads microbiologyevents.csv (868M) full, filtered to cohort hadm_ids,
# for the ab_name/interpretation columns 27a's Step-1 pass did not need.
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/27b_mimic_enrich.py --stage ast
