#!/bin/bash
#SBATCH --job-name=bsi_65_B_embed
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/65_B_embed_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/65_B_embed_%A_%a.err
#SBATCH --array=0-1%1
set -euo pipefail
# Analysis B (FOLLOWUP-2026-09-21-B): fresh text-embedding-3-small embeddings for
#   task 0 = B1b  unmodified_reembedded  (narratives/full, 32,329 docs, ~35.04 M tokens, ~$0.70)
#   task 1 = B2   no_therapy             (narratives/no_therapy, 32,329 docs, ~27.59 M tokens, ~$0.55)
#
# API-bound, not CPU-bound: 2 cpus is driver headroom. Serialised (%1) so the two
# arms cannot race on the shared cost ledger (HARD_CAP_USD = 4.00, re-checked after
# every batch). Resumable by shard. Wall time: the 49 precedent embedded one full
# corpus in ~5 min, so 3 h is a >30x allowance for rate-limit backoff.
#
# NO credential handling here, by design: the Python process reads the 0600 key
# file itself. Never `export` the key and never enable `set -x` / `bash -x` in
# this wrapper.
#
# Preconditions (run once, interactively, before submitting):
#   --stage count_tokens   (no API call; also re-run below)
#   --stage auth_check     (one non-patient string; embed refuses to run without it)
declare -a ARMS=(B1b B2)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
echo "[wrapper] arm=${ARM}"
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
PY="/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3"
"${PY}" scripts/65_B_variants.py --stage count_tokens --arm "${ARM}"
"${PY}" scripts/65_B_variants.py --stage embed --arm "${ARM}"
"${PY}" scripts/65_B_variants.py --stage merge --arm "${ARM}"
