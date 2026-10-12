#!/bin/bash
#SBATCH --job-name=aud4_a_geom
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/25_audit4/logs/a_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/25_audit4/logs/a_%j.err
set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}" OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 /data/tcmhwd/AI/AST_ai/phenotype_revision/out/25_audit4/a_variant_geometry.py
