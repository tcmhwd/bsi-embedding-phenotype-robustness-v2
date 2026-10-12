#!/bin/bash
#SBATCH --job-name=t20_recluster_onehot
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/20b_recluster_onehot_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/20b_recluster_onehot_%j.err
set -euo pipefail

# Task 1 (one-hot comparator) -- clusters both onehot variants (a: template
# fields only, b: a + rendered structural features) in the FROZEN
# /home/programs/anaconda3/envs/gpt interpreter (the only environment that
# reproduces the published partition) -- READ-ONLY, shared, never installed
# into or written inside. Threads pinned to 1 inside the script itself
# before any scientific import -- SLURM_CPUS_PER_TASK=4 here is headroom
# for numpy/BLAS incidental work, not a thread count fed to the clustering.

PY=/home/programs/anaconda3/envs/gpt/bin/python3
SCRIPT=/data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/20b_recluster_onehot.py

for VARIANT in a b; do
  echo "=== variant ${VARIANT} ==="
  ${PY} ${SCRIPT} --variant "${VARIANT}"
done
