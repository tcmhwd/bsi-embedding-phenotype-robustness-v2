#!/bin/bash
#SBATCH --job-name=bsi_A_p2_render
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/A_p2_render_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/A_p2_render_%j.err
set -euo pipefail
# Analysis A, PHASE 2 (Addendum 1): manifest -> render (3 arms) -> QA (tokenizer files only) -> review package.
# NO embedding, NO clustering, NO API call, NO GPU. Resources are the Phase-1 numbers (job 604514 finished
# in ~25 min wall): the 32,329 small files are latency-bound on this filesystem (threaded reads/writes sized
# by SLURM_CPUS_PER_TASK), QA tokenizes 4 corpora x 2 tokenizers and re-reads 16,144 MIMIC narratives, and
# the random arm holds every document's segment list in RAM.  Optional first argument: an out-root override
# plus --limit for a smoke test (used only for the pre-submission dry run).
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-16}"
export TOKENIZERS_PARALLELISM=true
export CUDA_VISIBLE_DEVICES=""
export HF_HUB_OFFLINE=1
PY=/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
REV=/data/tcmhwd/AI/AST_ai/phenotype_revision
cd "${REV}"
EXTRA_M=""; EXTRA_R=""
if [ "${1:-}" = "smoke" ]; then
  SM="${REV}/out/followup_analyses/A_harmonized_template/_smoke_phase2"
  EXTRA_M="--limit 300 --out-dir ${SM}"; EXTRA_R="--limit 300 --out-root ${SM}"
fi
${PY} scripts/60_harmonized_manifest.py --stage manifest ${EXTRA_M}
${PY} scripts/61_harmonized_render.py --stage render ${EXTRA_R}
${PY} scripts/61_harmonized_render.py --stage qa ${EXTRA_R}
${PY} scripts/61_harmonized_render.py --stage package ${EXTRA_R#--limit 300 }
