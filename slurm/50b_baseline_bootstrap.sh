#!/bin/bash
#SBATCH --job-name=bsi_50boot
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/50b_baseline_bootstrap_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/50b_baseline_bootstrap_%j.err
set -euo pipefail
# Companion to slurm/50_retrieval_bootstrap.sh: the same 10,000 patient-level
# resamples, applied to the three NON-retrieval baselines (published_16 cluster
# labels, SOFA, Charlson) against Gower k=25, so both halves of the retrieval
# claim carry intervals rather than one half carrying only point estimates.
#
# Resources: same shape as task 50 but with headroom for the extra work this
# script does before the bootstrap -- it builds the 44-variable structured
# matrix and refits three logistic baselines across five folds to retain their
# out-of-fold predictions (task 50 only loaded stored .npz predictions). Hence
# 24G rather than 16G and 06:00 rather than 04:00; three contrasts per variant
# instead of three total also doubles the AUROC evaluations per resample.
# Single-threaded numpy/sklearn; 4 cpus is driver headroom and the script does
# not read SLURM_CPUS_PER_TASK.
# Provenance: this ran once as job 590345 while the script was still numbered
# 52_baseline_bootstrap; its logs are logs/52_baseline_bootstrap_590345.{out,err}
# and its output out/23_retrieval/baseline_bootstrap.json is unchanged by the
# rename. Renamed to 50b so it sits beside its sibling 50_retrieval_bootstrap.py
# rather than among the 52_* package builders.
cd /data/tcmhwd/AI/AST_ai/phenotype_revision
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/50b_baseline_bootstrap.py
