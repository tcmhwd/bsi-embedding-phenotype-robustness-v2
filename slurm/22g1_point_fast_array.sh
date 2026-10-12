#!/bin/bash
#SBATCH --job-name=t22g1_fast
#SBATCH --partition=cpu_partition
#SBATCH --array=0-6%4
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/22g1_fast_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/22g1_fast_%A_%a.err
set -euo pipefail
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# Point-estimate combos for the four cheap methods (famd_kmeans/gmm_structured/
# lca/gower_pam) at both k -- excludes famd_kmeans k=16 (already computed
# interactively as a smoke test, resumable script will just skip it anyway).
declare -a METHODS=(famd_kmeans gmm_structured gmm_structured lca lca gower_pam gower_pam)
declare -a KS=(32 16 32 16 32 16 32)

I="${SLURM_ARRAY_TASK_ID}"
METHOD="${METHODS[$I]}"
K="${KS[$I]}"
echo "[wrapper] method=${METHOD} k=${K}"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/22g1_comparator_point.py \
  --method "${METHOD}" --k "${K}"
