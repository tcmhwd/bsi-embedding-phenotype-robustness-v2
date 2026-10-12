#!/bin/bash
#SBATCH --job-name=bsi_27b_labs
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27b_mimic_labs_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27b_mimic_labs_%j.err
set -euo pipefail
# Chunked scan of labevents.csv (18G) filtered to this cohort's hadm_id set
# + a small lab itemid whitelist. Single-process pandas (chunked read is
# I/O-bound, not BLAS-heavy) -- cpus-per-task=4 is headroom, not parallelism.
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/27b_mimic_enrich.py --stage labs
