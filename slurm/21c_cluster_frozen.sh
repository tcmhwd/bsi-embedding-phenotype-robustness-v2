#!/bin/bash
#SBATCH --job-name=t21c_cluster_frozen
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/21c_cluster_frozen_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/21c_cluster_frozen_%j.err
set -euo pipefail

# Task 21, Step 3 -- re-cluster every embedded variant in the FROZEN
# /home/programs/anaconda3/envs/gpt interpreter (READ-ONLY, shared, never
# installed into or written inside) -- the only environment that
# reproduces the published partition. Threads pinned to 1 inside the
# script itself before any scientific import. `full` MUST run first
# (every other variant's agreement-vs-unmodified-re-embedding depends on
# its cached labels) -- the explicit ordering below enforces that
# regardless of dict/glob ordering.

PY=/home/programs/anaconda3/envs/gpt/bin/python3
SCRIPT=/data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/21c_cluster_frozen.py
COMMON=/data/tcmhwd/AI/AST_ai/phenotype_revision/scripts/21_common.py

VARIANTS=$(${PY} - <<'PYEOF'
import sys
sys.path.insert(0, "/data/tcmhwd/AI/AST_ai/phenotype_revision/scripts")
import importlib
gen = importlib.import_module("21a_generate_variants")
print(" ".join(gen.ORDERED_VARIANTS))
PYEOF
)

for V in ${VARIANTS}; do
  echo "=== ${V} ==="
  ${PY} ${SCRIPT} --variant "${V}"
done
