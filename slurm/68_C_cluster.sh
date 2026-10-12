#!/bin/bash
#SBATCH --job-name=bsi_68_C_cluster
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-9%5
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/68_C_cluster_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/68_C_cluster_%A_%a.err
set -euo pipefail
# Analysis C, stage 3: 11 UMAP+HDBSCAN fits (seeds 42, 1..10) of one arm x encoder per task.
# Task i needs the embedding of task i of 67_C_embed: submit with --dependency=aftercorr:<embed job>.
# Timing basis: 27e_mimic_replicate ran ~26 s/fit at this n and config (22 fits inside a 1 h
# allowance); re-measured 8-32 s/fit -> ~6 min per task; 45 min is a >7x allowance.
# Threads are pinned to 1 inside 68_C_cluster.py BEFORE any numeric import and the script does not
# read SLURM_CPUS_PER_TASK, so cpus-per-task=2 is driver headroom only (same as 27e / 47 / 49).
# Memory: 66 MB embedding + UMAP/HDBSCAN working set << 16 G (27e used 16 G for the same fits).
declare -a ARMS=(unmodified unmodified no_organism_no_resistance no_organism_no_resistance no_charlson no_charlson no_organism no_organism no_source no_source)
declare -a ENCS=(bge-m3 qwen3-0.6b bge-m3 qwen3-0.6b bge-m3 qwen3-0.6b bge-m3 qwen3-0.6b bge-m3 qwen3-0.6b)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
ENC="${ENCS[$SLURM_ARRAY_TASK_ID]}"
echo "[wrapper] task=${SLURM_ARRAY_TASK_ID} arm=${ARM} encoder=${ENC}"
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/68_C_cluster.py --stage cluster --arm "${ARM}" --encoder "${ENC}"
