#!/bin/bash
#SBATCH --job-name=t28a_openpool
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/28a_openpool_query_mask_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/28a_openpool_query_mask_%j.err
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/28a_openpool_query_mask.py
