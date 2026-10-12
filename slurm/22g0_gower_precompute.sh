#!/bin/bash
#SBATCH --job-name=t22g0_gower
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/22g0_gower_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/22g0_gower_%j.err
set -euo pipefail
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/22g0_gower_precompute.py
