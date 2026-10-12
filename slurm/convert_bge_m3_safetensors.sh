#!/bin/bash
#SBATCH --job-name=bge_m3_convert
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/convert_bge_m3_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/convert_bge_m3_%j.err
set -euo pipefail

# One-time BGE-M3 .bin -> .safetensors conversion (design doc section 1.5).
# CPU-only, ~20 min estimated (download ~2.2GB + one clone/re-save pass over
# 568M params + a reload+diff verification pass). 4 cpu / 32G is generous
# headroom, not a measured requirement.

/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/convert_bge_m3_safetensors.py
