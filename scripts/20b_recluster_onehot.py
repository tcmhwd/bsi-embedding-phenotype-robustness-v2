#!/usr/bin/env python3
"""Task 1 -- Step 2: cluster the one-hot template-field matrices (variant
a: template fields only; variant b: a + rendered structural features)
with the code-of-record UMAP+HDBSCAN configuration
(`pipeline_new/phenotype_umap_hdbscan.py` / `17_common.UMAP_KW`/
`HDBSCAN_KW`: n_components=2, n_neighbors=15, min_dist=0.1,
metric=euclidean, random_state=42; HDBSCAN min_cluster_size=max(10,n//200),
min_samples=None, eom) in the FROZEN original execution environment (the
only one that reproduces the published partition).

Mirrors `18e_recluster_frozen.py` almost exactly (read and reused
structurally) -- the only difference is the input matrix (a dense one-hot/
standardized feature matrix here, instead of a text embedding).

MUST be invoked with:
    /home/programs/anaconda3/envs/gpt/bin/python3 20b_recluster_onehot.py --variant {a,b}

Thread count pinned to 1 before any scientific import (this repo's
established gotcha: thread count changes HDBSCAN's cluster count even at a
fixed seed).

Writes only under `out/20_onehot/recluster/<variant>/`: labels.npz,
result.json. Resumable: skips if result.json already exists.

Outcome-blind (Global Constraints): no mortality/discharge/death column is
read anywhere in this module -- inputs are the pre-built `onehot_<variant>.npy`
feature matrices from `20a_parse_template_fields.py`.
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
c17 = importlib.import_module("17_common")

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
OUT = REV / "out" / "20_onehot"
FEAT_DIR = OUT / "features"
OUT_DIR = OUT / "recluster"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["a", "b"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_dir = OUT_DIR / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.exists():
        print(f"[20b/{args.variant}] already done, skipping (resume)", flush=True)
        return

    print(f"[20b/{args.variant}] python executable: {sys.executable}", flush=True)

    vec_path = FEAT_DIR / f"onehot_{args.variant}.npy"
    id_path = FEAT_DIR / f"onehot_{args.variant}_ids.json"
    if not vec_path.exists():
        raise FileNotFoundError(f"{vec_path} missing -- run 20a_parse_template_fields.py --stage vectorize first")
    X_raw = np.load(vec_path)
    ids = np.asarray(json.loads(id_path.read_text()))

    pub_frame = c17.load_reference_frame()
    admission_ids = pub_frame["admission_ID"].to_numpy()
    reference = pub_frame[c17.PUBLISHED_CLUSTER_COL].to_numpy(dtype=np.int64)

    pos = pd.Series(np.arange(len(ids)), index=ids)
    missing = [int(a) for a in admission_ids if a not in pos.index]
    if missing:
        raise RuntimeError(f"{len(missing)} admission_ID(s) missing from onehot features, e.g. {missing[:5]}")
    X = X_raw[pos.loc[admission_ids].to_numpy()]
    if X.shape[0] != len(reference):
        raise ValueError(f"row mismatch: labels={len(reference)} X={X.shape[0]}")

    n = X.shape[0]
    min_cluster_size = c17.min_cluster_size_for(n)
    print(f"[20b/{args.variant}] X shape={X.shape}; config actually used: "
          f"UMAP={dict(**c17.UMAP_KW, random_state=42)} "
          f"HDBSCAN={dict(min_cluster_size=min_cluster_size, **c17.HDBSCAN_KW)}", flush=True)

    t0 = time.time()
    reducer = umap.UMAP(random_state=42, **c17.UMAP_KW)
    space = reducer.fit_transform(X)
    t_umap = time.time() - t0

    t1 = time.time()
    derived = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, core_dist_n_jobs=1, **c17.HDBSCAN_KW,
    ).fit_predict(np.ascontiguousarray(space, dtype=np.float64))
    t_hdbscan = time.time() - t1

    k = int(len(set(derived.tolist()) - {-1}))
    noise_frac = float((derived == -1).mean())
    print(f"[20b/{args.variant}] k={k} noise_frac={noise_frac:.4f}", flush=True)

    per_cluster_jaccard = {}
    for c in sorted(set(reference.tolist()) - {-1}):
        per_cluster_jaccard[str(int(c))] = c17.jaccard_for_cluster(reference, derived, c)

    result = {
        "variant": args.variant,
        "n_rows": int(n),
        "n_features": int(X.shape[1]),
        "python_executable": sys.executable,
        "n_clusters": k,
        "noise_frac": noise_frac,
        "n_noise": int((derived == -1).sum()),
        "agreement_vs_published": c17.agreement(reference, derived),
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
    c17.atomic_write_json(result, result_path)
    print(json.dumps({k_: v for k_, v in result.items() if k_ != "effective_native_threadpools"},
                      indent=2, ensure_ascii=False), flush=True)
    print(f"[20b/{args.variant}] wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
