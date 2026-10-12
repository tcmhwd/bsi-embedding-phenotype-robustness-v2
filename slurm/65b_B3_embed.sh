#!/bin/bash
#SBATCH --job-name=bsi_65b_B3_embed
#SBATCH --partition=gpu_partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/65b_B3_embed_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/65b_B3_embed_%j.err
set -euo pipefail
# Analysis B, arm B3 (FOLLOWUP-2026-09-21-B): Qwen3-Embedding-0.6B embedding of the Korean
# `no_department` narratives (32,329 docs). Local GPU; no external service, no API key.
#
# Resources (reasoned, not copied): the measured Qwen3 rate on this GPU node was 28.498 docs/s at
# batch 16 on the full corpus (1,134 s; out/18_encoder_dec/embeddings/qwen3-0.6b/.embed_manifest.json),
# and no_department is ~2% shorter, so encoding is ~19 min. Add the CPU-only verification stage
# (32,329 documents x apply_variant + cl100k count, a few minutes), the Qwen tokenizer pass for the
# truncation guard and the model load: ~30 min expected. 01:30:00 is ~3x. 8 CPUs / 32 GB / 1 GPU are the
# audit7_embed_qwen3.sh / 18b_embed_open_gpu.sh allocation; the script reads SLURM_CPUS_PER_TASK for
# its thread pins, so the 8 CPUs are used (tokenisation and verification).
#
# Recipe = audit7_embed_qwen3.sh: SentenceTransformer loaded on CPU then .to("cuda"), batch 16,
# normalised. HF_HUB_OFFLINE=1 forces the cached snapshot (models--Qwen--Qwen3-Embedding-0.6B is in
# ~/.cache/huggingface/hub) and forbids any contact with the HF hub; it changes no numerics.
# The stage `verify` writes the narrative sha256 manifest and must pass BEFORE `embed` runs (set -e).
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
PY="/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3"
export HF_HUB_OFFLINE=1
echo "[wrapper] host=$(hostname) job=${SLURM_JOB_ID}"
"${PY}" scripts/65b_B3_embed_qwen3.py --stage verify
"${PY}" scripts/65b_B3_embed_qwen3.py --stage embed --device cuda --batch-size 16
echo "[wrapper] done"
