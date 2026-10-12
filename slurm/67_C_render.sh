#!/bin/bash
#SBATCH --job-name=bsi_67_C_render
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/67_C_render_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/67_C_render_%j.err
set -euo pipefail
# Analysis C, stage 1: line-level surgery on the 16,144 stored MIMIC narratives -> 5 arms + QA.
# Pure file I/O + regex + one structured-data AMI table; single-threaded. A 400-document smoke run
# took 4 s; the full run is ~40x that plus ~80k small file writes. The first submission (job 604462,
# 30 min, create+rename per file) wrote only ~38 files/s on this filesystem and was cancelled; the script
# now writes-if-different (idempotent re-run), and 2 h covers a cold 80k-file write at that rate.
# All QA conditions are hard assertions: a non-zero exit cancels the `afterok` embedding array.
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/67_C_mimic_variants.py --stage render
