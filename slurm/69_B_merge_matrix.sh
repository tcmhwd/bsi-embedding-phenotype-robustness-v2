#!/bin/bash
#SBATCH --job-name=bsi_69_B_merge
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/69_B_merge_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/69_B_merge_%j.err
set -euo pipefail
# Analysis B supporting deliverable: merge ablation.csv (120 rows) + perturbation_matrix.csv (33) + perturbation_matrix_B3.csv (11)
# into perturbation_matrix_completed.csv (+ numbers-only summary + provenance json).  Pure table merge of ~164 rows; nothing is
# fitted or embedded, so 1 core / 4 GB / 15 min is a generous bound.
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[wrapper] host=$(hostname) job=${SLURM_JOB_ID}"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/69_B_merge_matrix.py
echo "[wrapper] done"
