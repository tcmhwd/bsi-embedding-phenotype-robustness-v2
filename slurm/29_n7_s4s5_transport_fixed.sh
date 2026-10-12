#!/bin/bash
#SBATCH --job-name=n7_s4s5fix
#SBATCH --partition=cpu_partition
#SBATCH --array=0-103%10
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/29_n7_s4s5_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/29_n7_s4s5_%A_%a.err
set -euo pipefail

# AUDIT_3 N7 rerun: 17c_s4s5_transport.py fixed so the target-side
# reference refit is scaled by the TARGET stratum's own min_cluster_size
# instead of reusing the source-derived one. Writes to out/29_fixes/ so the
# original (pre-fix) out/17_correct_env/s4s5_transport/ results stay on disk
# for the before/after ARI comparison. 104 runs = 4 observed directions +
# 4*25 random-patient-split nulls, same as the original 17c array.

export NUMBA_CACHE_DIR="/tmp/n7_s4s5fix_numba_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${NUMBA_CACHE_DIR}"

/home/programs/anaconda3/envs/gpt/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/17c_s4s5_transport.py \
  --run-index "${SLURM_ARRAY_TASK_ID}" \
  --out-dir /data/tcmhwd/AI/AST_ai/phenotype_revision/out/29_fixes/17c_s4s5_transport
