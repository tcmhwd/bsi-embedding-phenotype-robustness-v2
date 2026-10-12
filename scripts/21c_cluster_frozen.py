#!/usr/bin/env python3
"""Task 21, Step 3 -- cluster every embedded variant with the code-of-record
UMAP+HDBSCAN configuration (n_components=2, n_neighbors=15, min_dist=0.1,
metric=euclidean, random_state=42; HDBSCAN min_cluster_size=max(10,n//200),
min_samples=None, eom) in the FROZEN `/home/programs/anaconda3/envs/gpt`
interpreter -- the only environment that reproduces the published partition
(already verified bit-for-bit in Task 17/18: ARI=1.0, k=16, n_noise=1792).

MUST be invoked with:
    /home/programs/anaconda3/envs/gpt/bin/python3 21c_cluster_frozen.py --variant <name>

Reports, per variant:
  - k, noise_frac, n_noise
  - agreement vs the PUBLISHED 16 labels (ARI/AMI, all-row AND
    noise-excluded, each explicitly labelled -- `18_common.agreement`)
  - agreement vs the UNMODIFIED RE-EMBEDDING's own derived labels (the
    `full` variant, clustered first and cached to disk so every other
    variant's run -- independent SLURM invocations -- can load it without
    recomputing)
  - per-published-cluster best-match Jaccard (all 16 clusters; the task's
    decisive index-window comparison highlights clusters 0/7/11/12/14)

Thread count pinned to 1 before any scientific import (project gotcha:
HDBSCAN+UMAP are thread-count-dependent even at a fixed seed). Writes only
under `out/21_published_variants/cluster/<variant>/`. Resumable: skips if
`result.json` already exists.
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
c21 = importlib.import_module("21_common")

CLUSTER_OUT = c21.OUT / "cluster"
EMB_ROOT = c21.OUT / "embeddings"


def _emb_paths(variant: str):
    d = EMB_ROOT / variant
    return d / "merged.npy", d / "merged_ids.json"


def _versions() -> dict:
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "sklearn": sklearn.__version__,
        "umap-learn": umap.__version__, "hdbscan": importlib.metadata.version("hdbscan"),
        "numba": numba.__version__,
    }


def cluster_one(variant: str) -> dict:
    out_dir = CLUSTER_OUT / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "result.json"
    if result_path.exists():
        print(f"[21c/{variant}] already done, skipping (resume)", flush=True)
        return json.loads(result_path.read_text())

    print(f"[21c/{variant}] python executable: {sys.executable}", flush=True)

    vec_path, id_path = _emb_paths(variant)
    if not vec_path.exists():
        raise FileNotFoundError(f"{vec_path} missing -- run 21b_embed_openai.py --stage merge --variant {variant}")
    emb = np.load(vec_path)
    ids = json.loads(id_path.read_text())

    pub_frame = c21.load_published_frame()
    admission_ids = pub_frame["admission_ID"].to_numpy()
    reference = pub_frame[c21.PUBLISHED_CLUSTER_COL].to_numpy(dtype=np.int64)
    X = c21.align_embeddings_to_admission_ids(emb, ids, admission_ids)
    if X.shape[0] != len(reference):
        raise ValueError(f"row mismatch: labels={len(reference)} emb={X.shape[0]}")

    n = X.shape[0]
    min_cluster_size = c21.min_cluster_size_for(n)
    print(f"[21c/{variant}] config actually used: UMAP={dict(**c21.UMAP_KW, random_state=42)} "
          f"HDBSCAN={dict(min_cluster_size=min_cluster_size, **c21.HDBSCAN_KW)}", flush=True)

    t0 = time.time()
    reducer = umap.UMAP(random_state=42, **c21.UMAP_KW)
    space = reducer.fit_transform(X)
    t_umap = time.time() - t0

    t1 = time.time()
    derived = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, core_dist_n_jobs=1, **c21.HDBSCAN_KW,
    ).fit_predict(np.ascontiguousarray(space, dtype=np.float64))
    t_hdbscan = time.time() - t1

    k = int(len(set(derived.tolist()) - {-1}))
    noise_frac = float((derived == -1).mean())
    print(f"[21c/{variant}] k={k} noise_frac={noise_frac:.4f}", flush=True)

    per_cluster_jaccard = {
        str(int(cl)): c21.jaccard_for_cluster(reference, derived, cl)
        for cl in sorted(set(reference.tolist()) - {-1})
    }

    agreement_vs_published = c21.agreement(reference, derived)

    agreement_vs_full_reembed = None
    full_labels_path = CLUSTER_OUT / "full" / "labels.npz"
    if variant != "full":
        if not full_labels_path.exists():
            raise RuntimeError(
                f"{full_labels_path} missing -- cluster the 'full' variant FIRST "
                "(every other variant's agreement-vs-unmodified-re-embedding depends on it)."
            )
        full_npz = np.load(full_labels_path)
        full_admission_ids = full_npz["admission_ID"]
        full_labels = full_npz["labels"]
        # Align full's labels onto THIS variant's admission_ID order (both
        # should already equal `admission_ids`, but never assume row order).
        pos = pd.Series(np.arange(len(full_admission_ids)), index=full_admission_ids)
        full_aligned = full_labels[pos.loc[admission_ids].to_numpy()]
        agreement_vs_full_reembed = c21.agreement(full_aligned, derived)
    else:
        agreement_vs_full_reembed = {"note": "variant is 'full' itself -- agreement vs itself is trivially ARI=AMI=1.0"}

    result = {
        "variant": variant,
        "n_rows": int(n),
        "python_executable": sys.executable,
        "n_clusters": k,
        "noise_frac": noise_frac,
        "n_noise": int((derived == -1).sum()),
        "agreement_vs_published": agreement_vs_published,
        "agreement_vs_full_reembedding": agreement_vs_full_reembed,
        "best_match_jaccard_vs_published": per_cluster_jaccard,
        "effective_numba_threads": int(numba.get_num_threads()),
        "effective_native_threadpools": threadpool_info(),
        "versions": _versions(),
        "wall_time_umap_seconds": t_umap,
        "wall_time_hdbscan_seconds": t_hdbscan,
    }
    np.savez_compressed(out_dir / "labels.npz", labels=derived.astype(np.int32),
                         admission_ID=admission_ids)
    c21.atomic_write_json(result, result_path)
    print(json.dumps({k_: v for k_, v in result.items() if k_ != "effective_native_threadpools"},
                      indent=2, ensure_ascii=False), flush=True)
    print(f"[21c/{variant}] wrote {result_path}", flush=True)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    args = ap.parse_args()
    cluster_one(args.variant)


if __name__ == "__main__":
    main()
