#!/bin/bash
#SBATCH --job-name=t22g1_kproto
#SBATCH --partition=cpu_partition
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/22g1_kproto_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/22g1_kproto_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

declare -a KS=(16 32)
K="${KS[$SLURM_ARRAY_TASK_ID]}"
echo "[wrapper] method=kprototypes k=${K} (measured full-scale k=16 single fit: 5829.7s)"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/22g1_comparator_point.py \
  --method kprototypes --k "${K}"
