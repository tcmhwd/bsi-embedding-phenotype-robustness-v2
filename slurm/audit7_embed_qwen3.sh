#!/bin/bash
#SBATCH --job-name=audit7_embed_qwen3
#SBATCH --partition=gpu_partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:30:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/audit7_embed_qwen3_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/audit7_embed_qwen3_%j.err
set -euo pipefail
# Audit 7 question B: Qwen3 re-embedding of two Korean narrative variants
# (no_summary_tokens from out/21, no_cluster_ids from audit7_make_variant.py).
# GPU path as in slurm/18b_embed_open_gpu.sh (load on cpu, then .to("cuda");
# batch 16 = the batch size recorded in the original qwen3 manifest).
PY=/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
S=/data/tcmhwd/AI/AST_ai/phenotype_revision/scripts
O=/data/tcmhwd/AI/AST_ai/phenotype_revision/out
${PY} ${S}/audit7_embed_qwen3.py --variant no_cluster_ids --texts-dir ${O}/41_audit7/narratives/no_cluster_ids --device cuda --batch-size 16
${PY} ${S}/audit7_embed_qwen3.py --variant no_summary_tokens --texts-dir ${O}/21_published_variants/narratives/no_summary_tokens --device cuda --batch-size 16
echo "[audit7_embed] done"
