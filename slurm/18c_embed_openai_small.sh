#!/bin/bash
#SBATCH --job-name=t18_embed_3small
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/18c_embed_3small_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/18c_embed_3small_%j.err
set -euo pipefail

# Task 18, Step 1 -- embed reports_pilot_pre with OpenAI text-embedding-3-small.
# count_tokens already run on the login node (proceed=true, projected
# $0.7008, frac_truncated=0.0, out/18_encoder_dec/embeddings/openai-3small/
# token_report_3small.json) -- this job only does embed + merge. API-bound,
# no thread pinning needed.

PY=/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
SCRIPT=/data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/18c_embed_openai_small.py

echo "[1/2] embed"
${PY} ${SCRIPT} --stage embed

echo "[2/2] merge"
${PY} ${SCRIPT} --stage merge
