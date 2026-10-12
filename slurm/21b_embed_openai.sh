#!/bin/bash
#SBATCH --job-name=t21b_embed
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/21b_embed_openai_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/21b_embed_openai_%j.err
set -euo pipefail

# Task 21, Step 2 -- embed every generated variant (text-embedding-3-large)
# via the OpenAI API. API-bound, no thread pinning needed. Sequential
# single-writer run so the shared $80 cost ledger
# (out/21_published_variants/.cost_ledger.json) never race-writes across
# variants. Resumable (per-shard, per-variant) -- a re-run of this wrapper
# after a crash/cancel picks up exactly where it left off.

PY=/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
SCRIPT=/data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/21b_embed_openai.py

${PY} ${SCRIPT} --stage embed_all
