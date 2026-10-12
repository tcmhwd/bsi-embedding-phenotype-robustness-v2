#!/bin/bash
#SBATCH --job-name=bsi_49_3small_embed
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/49_3small_embed_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/49_3small_embed_%A_%a.err
#SBATCH --array=0-1%1
set -euo pipefail
# The missing third dominant-field ablation: text-embedding-3-small with its
# dominant field (treatment department) removed, plus its within-encoder
# control (no_organism, a NON-dominant field for this encoder).
#
# API-bound, not CPU-bound: 2 cpus is driver headroom. Serialised (%1) so the
# two variants cannot race on the shared $4.00 cost ledger. Resumable by shard,
# so a timeout or a rate-limit abort resumes where it stopped.
#
# Cost gate: --stage count_tokens must have written proceed=true first; the
# script refuses to call the API otherwise, and re-checks the running total
# against the cap after every batch.
declare -a VARIANTS=(no_department no_organism)
V="${VARIANTS[$SLURM_ARRAY_TASK_ID]}"
echo "[wrapper] variant=${V}"
cd /data/tcmhwd/AI/AST_ai/phenotype_revision
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  scripts/49_3small_no_department.py --stage count_tokens --variant "${V}"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  scripts/49_3small_no_department.py --stage embed --variant "${V}"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  scripts/49_3small_no_department.py --stage merge --variant "${V}"
