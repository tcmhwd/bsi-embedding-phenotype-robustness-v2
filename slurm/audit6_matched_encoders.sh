#!/bin/bash
#SBATCH --job-name=bsi_audit6_match
#SBATCH --partition=cpu_partition
#SBATCH --array=1-10%10
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/audit6_matched_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/audit6_matched_%A_%a.err
set -euo pipefail
# Mirrors slurm/30b_structure_matched.sh (threads pinned to 1 inside the script);
# 4 encoders x 5 seeds = 20 UMAP+HDBSCAN fits per task at n=16,144, the 3072-d
# 3-large encoder being the slowest. Memory raised for the 3072-d matrix.
PY=/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
${PY} /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/audit6_matched_encoders.py --stage matched --draw "${SLURM_ARRAY_TASK_ID}" --seeds 1 2 3 4 5
