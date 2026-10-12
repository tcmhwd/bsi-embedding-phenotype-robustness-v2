#!/bin/bash
#SBATCH --job-name=bsi_74_figdata
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/74_v6_figure_data_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/74_v6_figure_data_%j.err
set -euo pipefail
# Manuscript package v6 figures, step 1: reshape finished outputs of analyses A, B (completed matrix), C and the Figure 3 kNN readouts into tidy CSVs under
# out/followup_analyses/tables_v6/ (asserted identical to out/followup_analyses/tables/ where the two overlap).  Modelled on slurm/71_followup_figure_data.sh:
# a few thousand rows of pandas work, nothing fitted or re-scored, single-threaded (the script pins every *_NUM_THREADS to 1), so 1 core / 4 GB / 15 min.
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[wrapper] host=$(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK:-unset}"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/74_v6_figure_data.py
echo "[wrapper] done"
