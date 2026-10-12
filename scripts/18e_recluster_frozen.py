#!/usr/bin/env python3
"""Task 18, Step 3 -- re-discoverability (reported, NOT gated per PRESPEC
S6). Clusters each encoder's Step-1 embedding with the code-of-record
UMAP+HDBSCAN configuration (n_components=2, n_neighbors=15, min_dist=0.1,
metric=euclidean, random_state=42; HDBSCAN min_cluster_size=max(10,n//200),
min_samples=None, eom) in the FROZEN original execution environment, then
reports k, noise, ARI/AMI vs the published 16 labels (both all-row and
noise-excluded, each explicitly labelled), and per-published-cluster
best-match Jaccard.

MUST be invoked with:
    /home/programs/anaconda3/envs/gpt/bin/python3 18e_recluster_frozen.py --encoder <slug>

Thread count pinned to 1 before any scientific import (this repo's
established gotcha: thread count changes HDBSCAN's cluster count even at a
fixed seed).

Writes only under `out/18_encoder_dec/recluster/<encoder>/`:
  labels.npz, result.json. Resumable: skips if result.json already exists.
"""
import os
import sys

for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import importlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import hdbscan
import numba
import numpy as np
import pandas as pd
import sklearn
import umap
from threadpoolctl import threadpool_info

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c18 = importlib.import_module("18_common")

OUT_DIR = c18.OUT / "recluster"

ENCODER_PATHS = {
    "bge-m3": (c18.OUT / "embeddings" / "bge-m3" / "merged.npy",
               c18.OUT / "embeddings" / "bge-m3" / "merged_ids.json"),
    "qwen3-0.6b": (c18.OUT / "embeddings" / "qwen3-0.6b" / "merged.npy",
                   c18.OUT / "embeddings" / "qwen3-0.6b" / "merged_ids.json"),
    "openai-3small": (c18.OUT / "embeddings" / "openai-3small" / "merged.npy",
                       c18.OUT / "embeddings" / "openai-3small" / "merged_ids.json"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True, choices=list(ENCODER_PATHS))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = OUT_DIR / args.encoder
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.exists():
        print(f"[18e/{args.encoder}] already done, skipping (resume)", flush=True)
        return

    print(f"[18e/{args.encoder}] python executable: {sys.executable}", flush=True)

    vec_path, id_path = ENCODER_PATHS[args.encoder]
    if not vec_path.exists():
        raise FileNotFoundError(f"{vec_path} missing -- run Step 1 embedding for {args.encoder!r} first")
    emb = np.load(vec_path)
    ids = json.loads(id_path.read_text())

    pub_frame = c18.load_published_frame()
    admission_ids = pub_frame["admission_ID"].to_numpy()
    reference = pub_frame[c18.PUBLISHED_CLUSTER_COL].to_numpy(dtype=np.int64)
    X = c18.align_embeddings_to_admission_ids(emb, ids, admission_ids)
    if X.shape[0] != len(reference):
        raise ValueError(f"row mismatch: labels={len(reference)} emb={X.shape[0]}")

    n = X.shape[0]
    min_cluster_size = c18.min_cluster_size_for(n)
    print(f"[18e/{args.encoder}] config actually used: UMAP={dict(**c18.UMAP_KW, random_state=42)} "
          f"HDBSCAN={dict(min_cluster_size=min_cluster_size, **c18.HDBSCAN_KW)}", flush=True)

    t0 = time.time()
    reducer = umap.UMAP(random_state=42, **c18.UMAP_KW)
    space = reducer.fit_transform(X)
    t_umap = time.time() - t0

    t1 = time.time()
    derived = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, core_dist_n_jobs=1, **c18.HDBSCAN_KW,
    ).fit_predict(np.ascontiguousarray(space, dtype=np.float64))
    t_hdbscan = time.time() - t1

    k = int(len(set(derived.tolist()) - {-1}))
    noise_frac = float((derived == -1).mean())
    print(f"[18e/{args.encoder}] k={k} noise_frac={noise_frac:.4f}", flush=True)

    per_cluster_jaccard = {}
    for c in sorted(set(reference.tolist()) - {-1}):
        per_cluster_jaccard[str(int(c))] = c18.jaccard_for_cluster(reference, derived, c)

    result = {
        "encoder": args.encoder,
        "n_rows": int(n),
        "python_executable": sys.executable,
        "n_clusters": k,
        "noise_frac": noise_frac,
        "n_noise": int((derived == -1).sum()),
        "agreement_vs_published": c18.agreement(reference, derived),
        "best_match_jaccard_vs_published": per_cluster_jaccard,
        "effective_numba_threads": int(numba.get_num_threads()),
        "effective_native_threadpools": threadpool_info(),
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "umap-learn": umap.__version__, "hdbscan": importlib.metadata.version("hdbscan"),
            "numba": numba.__version__,
        },
        "wall_time_umap_seconds": t_umap,
        "wall_time_hdbscan_seconds": t_hdbscan,
    }
    np.savez_compressed(out_dir / "labels.npz", labels=derived.astype(np.int32),
                         admission_ID=admission_ids)
    c18.atomic_write_json(result, result_path)
    print(json.dumps({k_: v for k_, v in result.items() if k_ != "effective_native_threadpools"},
                      indent=2, ensure_ascii=False), flush=True)
    print(f"[18e/{args.encoder}] wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
