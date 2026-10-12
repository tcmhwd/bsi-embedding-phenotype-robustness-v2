#!/bin/bash
#SBATCH --job-name=bsi_A_64_agg
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/64_A_aggregate_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/64_A_aggregate_%j.err
set -euo pipefail
# Analysis A, PHASE 2: field attribution (AMI of every partition against every rendered field: 8 groups x 51 partitions
# x ~40 fields, high-cardinality fields such as antibiotic_set make single AMIs take seconds; process pool sized by
# SLURM_CPUS_PER_TASK), conditional agreement within organism strata, degeneracy, cross-encoder concordance, the mechanical
# preregistered rule outputs, and comparison_table.csv.  Everything is computed from stored partitions; nothing is re-fitted.
# Submitted with --dependency=afterok on the clustering array (all six tasks).
# Resources: 16 workers x ~1-2 GB (forked, copy-on-write field table) fits in 48 GB; 04:00:00 is a generous bound on
# ~400 partitions x 40 AMIs (the stored 4-encoder x 50-fit attribution in 40_field_dominance ran within one job).
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
PY="/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[wrapper] host=$(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK}"
"${PY}" scripts/64_harmonized_fields.py --stage all
echo "[wrapper] done"
