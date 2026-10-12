#!/bin/bash
#SBATCH --job-name=bsi_43_abl_embed
#SBATCH --partition=gpu_partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/43_qwen3_ablation_embed_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/43_qwen3_ablation_embed_%j.err
set -euo pipefail
# Causal test of field dominance for Qwen3, mirroring the one that already
# exists for 3-large.  On disk: removing 3-large's dominant field (organism)
# collapses its partition -- no_organism gives k=2, ARI 0.007 against the
# published partition -- while removing a non-dominant field does not
# (no_department, ARI 0.881).  That is an ablation, not a correlation.
#
# The parallel for Qwen3 is the therapy block, which carries its dominant
# field (Antibiotic_1_name, AMI 0.947).  The variant already exists from the
# published-template pass; it has only ever been embedded with OpenAI
# 3-large.  Two controls are embedded alongside it so the comparison is not
# a single arm: no_organism (Qwen3's NON-dominant field, expected to leave
# the partition intact) and full (the unmodified template, to confirm the
# pipeline reproduces the stored Qwen3 partition on this path).
PY=/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
SCRIPT=/data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/audit7_embed_qwen3.py
NARR=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/21_published_variants/narratives

for v in no_therapy no_organism full; do
  echo "=== embedding ${v} ==="
  ${PY} ${SCRIPT} --variant "ablation_${v}" --texts-dir "${NARR}/${v}" --device cuda
done
