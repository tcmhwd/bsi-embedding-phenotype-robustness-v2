#!/bin/bash
#SBATCH --job-name=aud4_b_seed
#SBATCH --partition=cpu_partition
#SBATCH --array=0-59%12
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/25_audit4/logs/b_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/25_audit4/logs/b_%A_%a.err
set -euo pipefail
VARIANTS=(full shuffle_order json_format no_definitions no_dates no_department)
I=${SLURM_ARRAY_TASK_ID}; V=${VARIANTS[$((I / 10))]}; S=$((I % 10 + 1))
echo "variant=$V seed=$S"
/home/programs/anaconda3/envs/gpt/bin/python3 /data/tcmhwd/AI/AST_ai/phenotype_revision/out/25_audit4/b_variant_seeds.py --variant "$V" --seed "$S"
