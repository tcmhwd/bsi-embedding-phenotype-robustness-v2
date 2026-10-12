#!/bin/bash
#SBATCH --job-name=bsi_A_62_embed
#SBATCH --partition=gpu_partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --array=1-6%1
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/62_A_embed_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/62_A_embed_%A_%a.err
set -euo pipefail
# Analysis A, PHASE 2: local-GPU embedding of one harmonized Korean arm with one open encoder.
# Array task N = (arm, encoder), arm-major: 1 primary/bge-m3, 2 primary/qwen3, 3 keep_therapy/bge-m3, 4 keep_therapy/qwen3,
# 5 random/bge-m3, 6 random/qwen3.  Local GPU, no external service, no API key.
#
# Resources (reasoned, not copied): the gpu_partition node has ONE GPU, so the six tasks are serialised (%1).
# Measured full-corpus rates on this GPU (out/18_encoder_dec/embeddings/*/.embed_manifest.json): BGE-M3 16.378 docs/s
# (1,974 s, batch 32), Qwen3 28.498 docs/s (1,134 s, batch 16).  The harmonized corpora hold 45-60% of the full corpus's
# tokens (median 469/600/458 vs 1,008 BGE-M3 tokens), so encoding is <= ~20 min (keep_therapy BGE-M3, the longest) and
# ~12 min for Qwen3.  Add the two per-document Python tokenizer passes (truncation guard + n_tokens, ~2-3 min), model
# load and the 64-document reproduction check: ~30 min worst case per task.  01:30:00 is 3x that.  8 CPUs / 32 GB / 1 GPU
# are the 18b_embed_open / 65b_B3_embed allocation; the script reads SLURM_CPUS_PER_TASK for its thread pins.
#
# Recipe = 18b_embed_open.py (batch size, cpu-load-then-.to("cuda"), normalisation, truncation guard); HF_HUB_OFFLINE=1
# forces the cached snapshots and forbids any contact with the HF hub (no numerics change).
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
PY="/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
echo "[wrapper] host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID}"
"${PY}" scripts/62_harmonized_embed.py --task "${SLURM_ARRAY_TASK_ID}" --device cuda
echo "[wrapper] done"
