#!/bin/bash
#SBATCH --job-name=bsi_75_runnerup
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/75_fix_runnerup_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/75_fix_runnerup_%j.err
set -euo pipefail
# Audit item 15 (ROUND-2): regenerate the runner-up columns of tables_v6/figA_dominance_summary.csv (scripts/71_followup_figure_data.py:155-156 takes the
# second INSERTED per-field item instead of the second-highest by value).  Modelled on slurm/74_v6_figure_data.sh: a dozen rows of pandas work over two JSON files
# and one CSV, nothing fitted or re-scored, single-threaded (the script pins every *_NUM_THREADS to 1), so 1 core / 4 GB / 10 min.
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[wrapper] host=$(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK:-unset}"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/75_fix_runnerup.py
echo "[wrapper] done"
