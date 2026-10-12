#!/bin/bash
#SBATCH --job-name=t18_recluster_frozen
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/18e_recluster_frozen_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/18e_recluster_frozen_%j.err
set -euo pipefail

# Task 18, Step 3 -- re-discoverability (reported, not gated). Runs in the
# FROZEN /home/programs/anaconda3/envs/gpt interpreter (the only environment
# that reproduces the published partition) -- READ-ONLY, shared, never
# installed into or written inside. Threads pinned to 1 inside the script
# itself before any scientific import (HDBSCAN+UMAP are thread-count
# dependent even at a fixed seed) -- SLURM_CPUS_PER_TASK=4 here is headroom
# for numpy/BLAS incidental work, not a thread count fed to the clustering.

PY=/home/programs/anaconda3/envs/gpt/bin/python3
SCRIPT=/data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/18e_recluster_frozen.py

for ENC in bge-m3 qwen3-0.6b openai-3small; do
  echo "=== ${ENC} ==="
  ${PY} ${SCRIPT} --encoder "${ENC}"
done
