#!/bin/bash
#SBATCH --job-name=t26f_seed_sweep
#SBATCH --partition=cpu_partition
#SBATCH --array=0-139%20
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:50:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/26f_seed_sweep_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/26f_seed_sweep_%A_%a.err
set -euo pipefail

# Task 26 Part 2 -- multi-seed (1-10) ARI/Jaccard for the 14 Task-21
# variants `out/25_audit4/b_variant_seeds.py` has not yet covered (12 of
# the original 18 + the 2 new AUDIT_4-C1 controls). Reuses that
# script unchanged (variant/seed as argv), same frozen-env code-of-record
# config it already uses. Writes into its own existing, resumable
# location out/25_audit4/seeds/<variant>_seed<s>.json (skips if already
# present) -- the 6 variants x 10 seeds already computed there are left
# untouched.

VARIANTS=(index_window no_headers null_dropped no_organism no_resistance no_therapy \
          no_year host_source_severity_only no_organism_no_resistance no_resistance_strict \
          no_organism_no_resistance_strict no_summary_tokens fixed_permutation shuffle_order_seed2)
I=${SLURM_ARRAY_TASK_ID}
V=${VARIANTS[$((I / 10))]}
S=$((I % 10 + 1))
echo "variant=$V seed=$S"
/home/programs/anaconda3/envs/gpt/bin/python3 /data/tcmhwd/AI/AST_ai/phenotype_revision/out/25_audit4/b_variant_seeds.py --variant "$V" --seed "$S"
