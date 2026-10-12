#!/bin/bash
#SBATCH --job-name=bsi_70_A_rescore
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/70_A_rescore_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/70_A_rescore_%j.err
set -euo pipefail
# Analysis A supporting deliverable: re-score the STORED Korea-full partitions (4 encoders x 50 matched fits + 4 whole-corpus partitions =
# 204 partitions) over the 39-field expanded set.  Analysis A's own attribution stage scored 408 partitions x 39 fields in 33 s on 16 workers
# (out/followup_analyses/A_harmonized_template/field_attribution.json "seconds": 33.1), so ~204 partitions on 8 workers should take about a minute plus the
# field-table build; 45 min is a large allowance.  The script reads SLURM_CPUS_PER_TASK for its fork pool (workers x ~1-2 GB copy-on-write field table
# fits comfortably in 32 GB).  Nothing is re-clustered.
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[wrapper] host=$(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK}"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/70_A_rescore_korea_full.py
echo "[wrapper] done"
