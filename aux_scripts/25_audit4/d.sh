#!/bin/bash
#SBATCH --job-name=aud4_d_gower
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:30:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/25_audit4/logs/d_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/25_audit4/logs/d_%j.err
set -euo pipefail
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 /data/tcmhwd/AI/AST_ai/phenotype_revision/out/25_audit4/d_gower_index_only.py
