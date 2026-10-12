#!/bin/bash
#SBATCH --job-name=bsi_50_boot
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/50_bootstrap_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/50_bootstrap_%j.err
set -euo pipefail
# Patient-level bootstrap of the retrieval AUROC differences: 10,000 resamples
# x 2 pool variants, each resample shared across the three embedding
# comparators so the contrasts stay mutually comparable.
#
# Submitted rather than run interactively: the first attempt was run on the
# login node and was killed for memory. That version also pre-allocated the
# (10,000 x 27,083) draw matrix; the resampler is now fully vectorised and
# generates draws one at a time, so 16G is ample headroom rather than a
# requirement. The script is single-threaded numpy/sklearn -- 4 cpus is driver
# headroom and it does not read SLURM_CPUS_PER_TASK.
cd /data/tcmhwd/AI/AST_ai/phenotype_revision
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/50_retrieval_bootstrap.py
