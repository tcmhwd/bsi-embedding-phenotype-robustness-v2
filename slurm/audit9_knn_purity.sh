#!/bin/bash
#SBATCH --job-name=bsi_a9_knn
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/audit9_knn_purity_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/audit9_knn_purity_%j.err
set -euo pipefail
# 5 arms x brute-force cosine k-NN on 32,329 x 1024 (chunked matmul); reads
# SLURM_CPUS_PER_TASK for BLAS threads.
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/audit9_knn_purity.py
