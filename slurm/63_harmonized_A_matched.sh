#!/bin/bash
#SBATCH --job-name=bsi_A_63_matched
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=12
#SBATCH --mem=36G
#SBATCH --time=03:00:00
#SBATCH --array=1-6
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/63_A_matched_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/63_A_matched_%A_%a.err
set -euo pipefail
# Analysis A, PHASE 2: the EXISTING n-matched clustering design (10 stored draws x seeds 1-5 = 50 fits at n=16,144, plus
# one whole-corpus seed-42 fit at n=32,329 for PREREG 4.3(1)) for one (arm, encoder) pair.  Same task numbering as 62;
# submitted with --dependency=aftercorr on the embedding array, so task N starts when embedding task N has succeeded.
#
# Env: bsi_pheno (the interpreter of the Figure-1 matched design, slurm/audit6_matched_encoders.sh:14) -- NOT the frozen env.
# Threads: the script pins OMP/NUMBA/MKL/OPENBLAS to 1 before any numeric import; parallelism is ACROSS fits (a process
# pool sized by SLURM_CPUS_PER_TASK), and a fit with a fixed random_state is deterministic regardless of the pool.
# Resources: audit6_matched_encoders did 20 fits/task (incl. 3072-d encoders) in <=4 h on 2 CPUs (slurm/audit6_matched_encoders.sh),
# i.e. <= ~12 min/fit; the 1024-d fits here are faster.  51 fits over 12 workers = ~5 rounds (whole-corpus fit ~2x a matched
# fit, scheduled first): ~1 h worst case; 03:00:00 is 3x.  Memory: ~2-3 GB per worker (UMAP graph + 66-132 MB matrix) x 12 = 36 GB.
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
PY="/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[wrapper] host=$(hostname) job=${SLURM_JOB_ID} task=${SLURM_ARRAY_TASK_ID} cpus=${SLURM_CPUS_PER_TASK}"
"${PY}" scripts/63_harmonized_matched.py --task "${SLURM_ARRAY_TASK_ID}"
echo "[wrapper] done"
