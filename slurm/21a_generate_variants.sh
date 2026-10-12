#!/bin/bash
#SBATCH --job-name=t21a_generate
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/21a_generate_variants_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/21a_generate_variants_%j.err
set -euo pipefail

# Task 21, Step 1 -- generate every narrative variant on the PUBLISHED
# template (reports_pilot_pre/, 32,329 files), verify (sample diff +
# corpus-wide line counts), count tokens, project cost, gate at $80.
# Pure CPU/tiktoken text processing (no GPU, no API calls). Measured on the
# shared cluster filesystem: the `full` variant alone (32,329 docs, includes
# per-doc tiktoken encoding + file write) took ~7 min under real cluster
# I/O contention -- 18 variants therefore budgeted at 4h, not the smaller
# login-node-timed estimate. Resumable: already-`status: ok` variants are
# skipped, so a re-run after a time-limit cancellation picks up cleanly.

PY=/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
SCRIPT=/data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/21a_generate_variants.py

${PY} ${SCRIPT}
