#!/bin/bash
#SBATCH --job-name=t18_s6_knn
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/18d_s6_knn_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/18d_s6_knn_%j.err
set -euo pipefail

# Task 18, Step 2 -- S6 recoverability: kNN (k=25, cosine,
# StratifiedGroupKFold(5, patient_ID)) predicting the published 16 labels
# from each encoder's embedding, plus the OpenAI-3-large ceiling. Not a
# clustering stage (no HDBSCAN/UMAP) so thread pinning to 1 is not required
# for reproducibility, but sklearn's kNN/BLAS calls are capped to the
# allocation via SLURM_CPUS_PER_TASK.

PY=/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
SCRIPT=/data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/18d_s6_knn.py
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

for ENC in bge-m3 qwen3-0.6b openai-3small openai-3large; do
  echo "=== ${ENC} ==="
  ${PY} ${SCRIPT} --encoder "${ENC}"
done
