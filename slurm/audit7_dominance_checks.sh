#!/bin/bash
#SBATCH --job-name=audit7_dominance
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/audit7_dominance_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/audit7_dominance_%j.err
set -euo pipefail
# 16 CPUs: seven HistGradientBoosting fits x 5 folds on 31k rows plus ~1,500
# AMI evaluations; OMP threads follow the allocation.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/audit7_dominance_checks.py
