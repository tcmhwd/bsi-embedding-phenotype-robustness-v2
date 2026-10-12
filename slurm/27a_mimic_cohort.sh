#!/bin/bash
#SBATCH --job-name=bsi_27a_cohort
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27a_mimic_cohort_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27a_mimic_cohort_%j.err
set -euo pipefail

# Task M0: MIMIC-IV multi-organism bacterial BSI cohort construction +
# QC (linkage sensitivity, reverse check, external plausibility).
#
# Cost basis: reads microbiologyevents.csv (868M), admissions.csv (90M),
# patients.csv, icustays.csv -- all read fully once each, single-threaded
# pandas I/O + groupby (no BLAS-heavy work here, so --cpus-per-task=4 is
# headroom for the pandas process + OS page cache, not real parallelism).
# --time=00:45:00 is generous margin over an expected few-minute runtime.
#
# Writes ONLY to out/27_mimic/cohort/ (episodes_raw.parquet,
# admissions.parquet, cohort_flow.json, COHORT_QC.md,
# linkage_sensitivity.json, unmapped_to_other_org_names.json,
# 27a_manifest.json). Does not touch any other out/* directory (read-only
# per the task's binding constraints: reads only
# /data/tcmhwd/AI/Fungemia/mimic4/ and
# manuscript/phenotype_ml_dataset.csv, both read-only sources here).

PY=/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
SCRIPT=/data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/27a_mimic_cohort.py

"$PY" "$SCRIPT" --stage cohort
"$PY" "$SCRIPT" --stage qc
