#!/bin/bash
#SBATCH --job-name=bsi_78_provchk
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/78_provenance_checks_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/78_provenance_checks_%j.err
set -euo pipefail
# Final audit part A, M-6: write out/followup_analyses/qc/provenance_checks_table3_0693_table2_0728.json (Table 3 footnote 0.693, Table 2 footnote 0.728) and
# verify the five half-way MIMIC E. coli cells of Table 1 (M-4).  Modelled on slurm/76_mimic_doc_stats.sh and 74_v6_figure_data.sh (bsi_pheno python, cpu_partition, logs under out/followup_analyses/logs/).
# Resources (estimate): one 32k-row parquet + one 32k-row npz, phenotype_ml_dataset.csv read with usecols (2 columns), one 16k-row parquet with 4 columns, one sklearn AMI on 32k rows
# (contingency 16 x 55 levels): seconds; peak RAM well under 1 GB (the CSV is parsed with usecols but pandas' C parser scans the whole file, a few hundred MB at most).  2 CPUs / 8 GB / 15 min is headroom.
# Aux checks in the same job (aggregate scalars only): M-4 Table 1 E. coli half-way quantiles; S-12 min/max norms of the ten stored Analysis-C embedding matrices (10 x 16,144 x 1024 float32, mmap, float64 norm per matrix = ~130 MB transient).
# Threads: the script is single-threaded and does not read SLURM_CPUS_PER_TASK; numeric libraries are pinned to 1.  Output is aggregate only (no per-patient value is printed or stored).
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
export OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[wrapper] host=$(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK:-unset}"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/78_provenance_checks.py
echo "[wrapper] done"
