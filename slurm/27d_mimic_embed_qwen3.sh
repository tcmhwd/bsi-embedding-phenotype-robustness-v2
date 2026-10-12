#!/bin/bash
#SBATCH --job-name=bsi_27d_qwen3
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27d_mimic_embed_qwen3_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27d_mimic_embed_qwen3_%j.err
set -euo pipefail
# CPU-only. batch=8 was the measured peak throughput setting for
# qwen3-0.6b on this cluster (job 586647, 4.06 docs/sec at 64 threads/batch
# 8 -- larger batches were measured SLOWER for this model/hardware).
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/27d_mimic_embed.py \
  --model qwen3-0.6b --device cpu --threads "${SLURM_CPUS_PER_TASK}" --batch-size 8
