#!/bin/bash
#SBATCH --job-name=bsi_76_mimicdocstats
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/76_mimic_doc_stats_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/76_mimic_doc_stats_%j.err
set -euo pipefail
# Recompute the MIMIC-IV row of Supplementary Table 15 (median lines / tokens per document; cited so far only from PREREGISTRATION_C.md:273).
# Modelled on slurm/74_v6_figure_data.sh (bsi_pheno python, cpu_partition, logs under out/followup_analyses/logs/).
# Resources (estimate, not copied): 2 corpora x 16,144 documents (~0.75 KB each, ~12 MB per corpus in memory) x 2 tokenizers = ~65k short tokenizer calls in
# 512-document batches (seconds each on the Rust fast tokenizer) plus reading ~32k small files, sha256 and the 40-document guard probe; model loading is tokenizer-only
# (no weights).  Expected wall time 2-4 min, dominated by tokenizer load and file I/O; 30 min is a >5x margin.  16 GB covers the texts (<100 MB) with a wide margin
# for the two tokenizers (Qwen3 vocab ~11 MB json, BGE-M3 ~17 MB json).
# Threads: unlike 74/75 this script DOES read SLURM_CPUS_PER_TASK -- only to set RAYON_NUM_THREADS for the Rust batch tokenizer (the one stage that scales with cores,
# as in 61_harmonized_render.py's n_workers()); every numeric-library pin stays 1.  4 CPUs is therefore honoured, not decorative.
# HF_HUB_OFFLINE=1: local tokenizers only, no contact with the HF hub.  Nothing is uploaded; the job writes aggregate files only (no document id / text / per-document value).
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[wrapper] host=$(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK:-unset}"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/76_mimic_doc_stats.py
echo "[wrapper] done"
