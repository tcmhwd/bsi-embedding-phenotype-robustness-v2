#!/bin/bash
#SBATCH --job-name=t26e_sofa_strat
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/26e_sofa_stratified_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/26e_sofa_stratified_%j.err
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/26e_sofa_stratified.py
