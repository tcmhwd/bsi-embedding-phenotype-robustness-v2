#!/bin/bash
#SBATCH --job-name=bsi_a9_mask_embed
#SBATCH --partition=gpu_partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/audit9_mask_embed_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/audit9_mask_embed_%j.err
set -euo pipefail
# Audit 9: isolate the antibiotic NAME from the rest of the therapy block.
# Mirrors slurm/43_qwen3_ablation_embed.sh (same env, script logic, device,
# batch size); only the variant texts and output directory differ.
PY=/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
REV=/data/tcmhwd/AI/AST_ai/phenotype_revision
${PY} ${REV}/scripts/audit9_make_mask_variants.py
for v in mask_ab_names_all mask_ab1_name_only; do
  echo "=== embedding ${v} ==="
  ${PY} ${REV}/scripts/audit9_embed_qwen3.py --variant "${v}" \
    --texts-dir "${REV}/out/45_audit9/narratives/${v}" --device cuda
done
