#!/usr/bin/env python3
"""Task 17, S1 (seed/thread) -- refits of the code-of-record UMAP+HDBSCAN
configuration on the contemporaneous `out_pre` embeddings, in the frozen
original execution environment.

MUST be invoked with:
    /home/programs/anaconda3/envs/gpt/bin/python3 17a_s1_seed_thread.py --seed N

Thread count is set via STRESS_THREADS (default "1"), which the SLURM
wrapper exports BEFORE this process starts. Pinned here again, before any
scientific import, per this repo's established gotcha (env vars set after
numpy/numba import silently do nothing).

Reference partition: the PUBLISHED labels themselves
(`out_pre/phenotype_clusters_labels.csv`, column `phenotype_cluster`) --
NOT a refit -- since this environment reproduces them bit-for-bit (task
brief; verified independently in `out/16_c1_sweep/result_original_env_seed42.json`).

Writes only under `phenotype_revision/out/17_correct_env/s1_seed/`:
  - `labels_seed{N}_threads{T}.npz` (labels only; the fitted UMAP space is
    not needed downstream and is dropped to save disk)
  - `result_seed{N}_threads{T}.json` (n_clusters, noise_frac, full agreement,
    per-published-cluster best-match Jaccard, versions, wall time)
Resumable: skips a (seed, threads) pair whose result JSON already exists.
"""
import os
import sys

_THREADS = os.environ.get("STRESS_THREADS", "1")
for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = _THREADS

import argparse
import importlib
import importlib.metadata
import importlib.util
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
c17 = importlib.import_module("17_common")

OUT_DIR = c17.OUT / "s1_seed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"seed{args.seed}_threads{_THREADS}"
    result_path = OUT_DIR / f"result_{tag}.json"
    if result_path.exists():
        print(f"[{tag}] already done, skipping (resume)", flush=True)
        return

    print(f"[{tag}] python executable: {sys.executable}", flush=True)
    print(f"[{tag}] requested threads: {_THREADS}", flush=True)

    ref = c17.load_reference_frame()
    reference = ref[c17.PUBLISHED_CLUSTER_COL].to_numpy(dtype=np.int64)
    emb = c17.load_embeddings()
    if emb.shape[0] != len(reference):
        raise ValueError(f"row mismatch: labels={len(reference)} emb={emb.shape[0]}")

    n = emb.shape[0]
    min_cluster_size = c17.min_cluster_size_for(n)
    print(f"[{tag}] config actually used: UMAP={dict(**c17.UMAP_KW, random_state=args.seed)} "
          f"HDBSCAN={dict(min_cluster_size=min_cluster_size, **c17.HDBSCAN_KW)}", flush=True)

    t0 = time.time()
    reducer = umap.UMAP(random_state=args.seed, **c17.UMAP_KW)
    space = reducer.fit_transform(emb)
    t_umap = time.time() - t0

    t1 = time.time()
    derived = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, core_dist_n_jobs=1, **c17.HDBSCAN_KW,
    ).fit_predict(np.ascontiguousarray(space, dtype=np.float64))
    t_hdbscan = time.time() - t1

    k = int(len(set(derived.tolist()) - {-1}))
    noise_frac = float((derived == -1).mean())
    print(f"[{tag}] k={k} noise_frac={noise_frac:.4f}", flush=True)

    per_cluster = {}
    for c in sorted(set(reference.tolist()) - {-1}):
        per_cluster[str(int(c))] = c17.jaccard_for_cluster(reference, derived, c)

    result = {
        "seed": args.seed,
        "threads_requested": _THREADS,
        "python_executable": sys.executable,
        "n_clusters": k,
        "noise_frac": noise_frac,
        "n_noise": int((derived == -1).sum()),
        "agreement_vs_published": c17.agreement(reference, derived),
        "jaccard_vs_published": per_cluster,
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
    np.savez_compressed(OUT_DIR / f"labels_{tag}.npz", labels=derived.astype(np.int32))
    c17.atomic_write_json(result, result_path)
    print(json.dumps({k_: v for k_, v in result.items() if k_ != "effective_native_threadpools"},
                      indent=2, ensure_ascii=False), flush=True)
    print(f"[{tag}] wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
