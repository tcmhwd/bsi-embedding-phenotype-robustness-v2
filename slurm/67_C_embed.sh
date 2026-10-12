#!/bin/bash
#SBATCH --job-name=bsi_67_C_embed
#SBATCH --partition=gpu_partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-9%1
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/67_C_embed_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/67_C_embed_%A_%a.err
set -euo pipefail
# Analysis C, stage 2: 5 arms x 2 encoders = 10 from-scratch embeddings of 16,144 documents each.
# Serialised (%1): gpu_partition has one GPU (Gres=gpu:1); primary arms first so the reproduction
# check (unmodified) and the primary organism arm complete earliest.
# Precedents: slurm/audit7_embed_qwen3.sh (8 cpus / 32G / 1 GPU); Korean corpus (4x longer docs)
# ran 16.4 (BGE-M3) / 28.5 (Qwen3) docs/s on GPU, so 16k short MIMIC docs take a few minutes plus
# model load; 1 h per task is a large allowance. Load-on-CPU-then-.to("cuda") is inside 67b (27d:99-102).
# PREREGISTRATION_C_ADDENDUM_1: 27d's exact loading (Qwen3 bf16, BGE-M3 fp32), stored-run batch sizes, GPU, all arms.
# Tasks 0/1 (unmodified) write gross_error_check_<encoder>.json (min cosine vs stored vectors >= 0.99, else exit 3);
# every later task refuses to run unless both checks passed, so a failure halts the chain.
# HF_HUB_OFFLINE=1 is forced inside the script: no external service is contacted.
declare -a ARMS=(unmodified unmodified no_organism_no_resistance no_organism_no_resistance no_charlson no_charlson no_organism no_organism no_source no_source)
declare -a ENCS=(bge-m3 qwen3-0.6b bge-m3 qwen3-0.6b bge-m3 qwen3-0.6b bge-m3 qwen3-0.6b bge-m3 qwen3-0.6b)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
ENC="${ENCS[$SLURM_ARRAY_TASK_ID]}"
echo "[wrapper] task=${SLURM_ARRAY_TASK_ID} arm=${ARM} encoder=${ENC}"
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/67b_C_embed.py \
  --arm "${ARM}" --model "${ENC}" --device cuda --threads "${SLURM_CPUS_PER_TASK}"
