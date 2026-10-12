#!/bin/bash
#SBATCH --job-name=bsi_68_C_aggregate
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/68_C_aggregate_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/68_C_aggregate_%j.err
set -euo pipefail
# Analysis C, stage 4: score all 6 partitions-sets (5 arms + stored reference) x 2 encoders x 11 seeds.
# ~132 field-attribution sweeps (19 AMIs each), 132 x 2 agreement pairs, 66 cross-encoder + 132
# conditional-agreement calls: a 22-fit smoke test took 16 s, so the full run is a few minutes;
# 45 min is a large allowance. Single-threaded (pins set in the script); refuses to run if any fit is missing.
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/68_C_cluster.py --stage aggregate
