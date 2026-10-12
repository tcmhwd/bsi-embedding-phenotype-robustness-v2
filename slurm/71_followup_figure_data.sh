#!/bin/bash
#SBATCH --job-name=bsi_71_figdata
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/71_figure_data_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/71_figure_data_%j.err
set -euo pipefail
# Follow-up figure candidates, step 1: reshape finished outputs of analyses A, B (completed matrix from script 69) and C into tidy CSVs under
# out/followup_analyses/tables/.  A few thousand rows of pandas work; nothing is fitted, embedded or re-scored, so 1 core / 4 GB / 15 min is a generous bound.
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[wrapper] host=$(hostname) job=${SLURM_JOB_ID}"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/71_followup_figure_data.py
echo "[wrapper] done"
