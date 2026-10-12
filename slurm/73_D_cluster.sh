#!/bin/bash
#SBATCH --job-name=bsi_73_D
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=1-10%5
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/73_D_alt_%A_%a.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/73_D_alt_%A_%a.err
set -euo pipefail
# Analysis D: 20 UMAP+HDBSCAN fits per task (4 encoders x seeds 1-5) on the STORED draw of the
# same index. The header above holds the --stage alt resources (section 8.1). The other stages
# override resources, array and log names on the sbatch command line (A.5):
#   anchor_stored : sbatch --array=1 --mem=16G --time=00:30:00 --job-name=bsi_73_D_anchor_stored \
#                     --output=.../73_D_anchor_stored_%A_%a.out --error=.../73_D_anchor_stored_%A_%a.err  <this> anchor_stored
#   anchor_refit  : sbatch --array=1 --mem=32G --time=06:00:00 ... anchor_refit   (draw 01 only, 20 fits at n_components=2)
#   alt           : sbatch <this> alt                                              (array 1-10%5, n_components=20)
#   aggregate     : sbatch --array=1 --mem=32G --time=01:00:00 ... aggregate
# Threads are pinned to 1 inside 73_D_cluster.py before any numeric import (determinism:
# 18e_recluster_frozen.py:14-16) and the script does not read SLURM_CPUS_PER_TASK, so
# cpus-per-task=2 is driver headroom only, as in audit6 / 27e / 68.
# Time (alt): 27e measured 8-32 s/fit at n_components=2; HDBSCAN in 20-D degrades sharply, so a
# 10-20x per-fit slowdown is allowed for. Mem: the 3-large slice is 198 MB; 48G is ample.
# Environment: bsi_pheno (umap-learn 0.5.7, numpy 1.26.4) -- the interpreter that produced the
# stored Figure-1 labels; NOT the frozen gpt env (PREDECLARED_PIPELINE.md 2.5).
STAGE="${1:-alt}"
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
echo "[wrapper] stage=${STAGE} task=${SLURM_ARRAY_TASK_ID:-1} job=${SLURM_JOB_ID:-none}"
/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/73_D_cluster.py \
  --stage "${STAGE}" --draw "${SLURM_ARRAY_TASK_ID:-1}" --seeds 1 2 3 4 5
