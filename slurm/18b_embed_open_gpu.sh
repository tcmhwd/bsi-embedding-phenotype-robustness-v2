#!/bin/bash
#SBATCH --job-name=t18_embed_open_gpu
#SBATCH --partition=gpu_partition
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/18b_embed_open_gpu_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs/18b_embed_open_gpu_%j.err
set -euo pipefail

# Task 18, Step 1 -- embed reports_pilot_pre (32,329 docs) with BGE-M3 and
# Qwen3-Embedding-0.6B on the GPU (gpu_partition/node1, currently idle,
# confirmed via sinfo before submission).
#
# Gotcha (measured repeatedly in this repo, jobs 586270/586271/586273 vs
# 586275, reconfirmed by 04_embed_qwen3_gpu.sh job 586710): a GPU
# batch-size sweep is run fresh on THIS allocation before the real embed
# (CPU-optimal batch sizes do not transfer to GPU -- the throughput-vs-batch
# curve inverts), and SentenceTransformer(..., device="cuda") directly
# segfaults silently; 18b_embed_open.py already loads on cpu then .to("cuda").

PY=/data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
SCRIPT_DIR=/data/tcmhwd/AI/AST_ai/phenotype_revision/scripts
LOG_DIR=/data/tcmhwd/AI/AST_ai/phenotype_revision/logs

for MODEL in bge-m3 qwen3-0.6b; do
  SWEEP_LOG="${LOG_DIR}/18b_sweep_${MODEL}_${SLURM_JOB_ID}.log"
  echo "=== ${MODEL}: GPU batch-size sweep ==="
  ${PY} - "${MODEL}" <<'PYEOF' | tee "${SWEEP_LOG}"
import glob
import os
import random
import sys
import time

model_slug = sys.argv[1]
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"

sys.path.insert(0, "/data/tcmhwd/AI/AST_ai/phenotype_revision/scripts")
import importlib
oe = importlib.import_module("04_open_embedding")
from sentence_transformers import SentenceTransformer

cfg = oe.MODEL_CONFIGS[model_slug]
random.seed(42)
files = sorted(glob.glob("/data/tcmhwd/AI/AST_ai/reports_pilot_pre/*.report.md"))
texts = [open(f, encoding="utf-8", errors="replace").read() for f in random.sample(files, 200)]

t0 = time.time()
model = SentenceTransformer(cfg["load_path"], device="cpu")
model.max_seq_length = cfg["max_seq_length"]
model = model.to("cuda")
print(f"RESULT load_sec={time.time() - t0:.1f}", flush=True)

candidates = (32, 64, 128, 256) if model_slug == "bge-m3" else (16, 32, 64, 128)
results = {}
for bs in candidates:
    t0 = time.time()
    v = model.encode(texts, batch_size=bs, convert_to_numpy=True,
                      normalize_embeddings=True, show_progress_bar=False)
    sec = time.time() - t0
    dps = len(texts) / sec
    print(f"RESULT batch={bs:4d} docs_per_sec={dps:7.3f} sec={sec:7.1f}", flush=True)
    results[bs] = dps

best_bs = max(results, key=results.get)
print(f"RESULT n_docs={len(texts)} dim={v.shape[1]}", flush=True)
print(f"BEST_BATCH={best_bs}", flush=True)
PYEOF

  BEST_BS=$(grep -oP 'BEST_BATCH=\K[0-9]+' "${SWEEP_LOG}" | tail -1)
  if [ -z "${BEST_BS}" ]; then
    echo "ERROR: could not parse BEST_BATCH for ${MODEL} from ${SWEEP_LOG}" >&2
    exit 1
  fi
  echo "[sweep] ${MODEL}: selected batch=${BEST_BS} (GPU-measured, this allocation)"

  echo "=== ${MODEL}: full-corpus embed (cuda, batch=${BEST_BS}) ==="
  ${PY} ${SCRIPT_DIR}/18b_embed_open.py --model "${MODEL}" --device cuda --batch-size "${BEST_BS}"
done

echo "[18b_embed_open_gpu] both models done"
