#!/bin/bash
#SBATCH --job-name=t22g2_boot
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/22g2_boot_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/22g2_boot_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Generic patient-level bootstrap array for ONE (METHOD, K) combo.
# Submit with: sbatch --job-name=t22g2_${METHOD}_k${K} --array=0-$((NSHARDS-1))%THROTTLE \
#   --time=TIME_LIMIT --export=ALL,METHOD=...,K=...,NSHARDS=...,NBOOT=100 22g2_bootstrap_array.sh
: "${METHOD:?must export METHOD}"
: "${K:?must export K}"
: "${NSHARDS:?must export NSHARDS}"
: "${NBOOT:=100}"

echo "[wrapper] method=${METHOD} k=${K} shard=${SLURM_ARRAY_TASK_ID}/${NSHARDS} n_boot=${NBOOT}"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/22g2_comparator_bootstrap.py \
  --method "${METHOD}" --k "${K}" --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --n-shards "${NSHARDS}" --n-boot "${NBOOT}"
