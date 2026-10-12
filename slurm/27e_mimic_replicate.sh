#!/bin/bash
#SBATCH --job-name=bsi_27e_replic
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27e_mimic_replicate_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27e_mimic_replicate_%j.err
set -euo pipefail
# All three replications, one job: 22 cluster fits (2 encoders x (seed 42 +
# seeds 1-10)) at the anchor config on ~16k rows -- cheap (this project's
# Korean-cohort precedent: ~42s/fit on 32,329 rows at the SAME anchor
# config, so a smaller n here is faster still) -- followed by
# cross_encoder, cross_seed (x2), token_recoverability. Threads pinned to 1
# inside 27e_mimic_replicate.py itself before any numpy/sklearn/umap/
# hdbscan import (same discipline as 02_recluster.py) -- cpus-per-task=2 is
# headroom for the driver process only.
PY=/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
SCRIPT=/data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/27e_mimic_replicate.py

for enc in bge-m3 qwen3-0.6b; do
  echo "[cluster] ${enc} seed=42 (anchor)"
  ${PY} ${SCRIPT} --stage cluster --encoder "${enc}" --seed 42
  for s in 1 2 3 4 5 6 7 8 9 10; do
    echo "[cluster] ${enc} seed=${s}"
    ${PY} ${SCRIPT} --stage cluster --encoder "${enc}" --seed "${s}"
  done
done

echo "[cross_encoder]"
${PY} ${SCRIPT} --stage cross_encoder

for enc in bge-m3 qwen3-0.6b; do
  echo "[cross_seed] ${enc}"
  ${PY} ${SCRIPT} --stage cross_seed --encoder "${enc}"
done

echo "[token_recoverability]"
${PY} ${SCRIPT} --stage token_recoverability
