#!/bin/bash
#SBATCH --job-name=bsi_27d_bgem3
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27d_mimic_embed_bge_m3_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/27d_mimic_embed_bge_m3_%j.err
set -euo pipefail
# CPU-only (GPU/gpu_partition is contended by other jobs -- binding
# constraint: never contend for GPU). 16,144 MIMIC narratives (smaller and
# shorter than the Korean 32,329-doc full template), batch=32 measured
# peak-ish setting from this project's own bge-m3 gate precedent
# (slurm/04_gate_bge_m3_cpu.sh, 128 cores -> ~2.8-4 docs/sec). Using 32
# cores here (node2 headroom-conscious -- see progress.md resource
# allocation principle) trades some wall time for staying well under the
# node budget while a qwen3 embed job runs alongside it.
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 \
  /data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/27d_mimic_embed.py \
  --model bge-m3 --device cpu --threads "${SLURM_CPUS_PER_TASK}" --batch-size 32
