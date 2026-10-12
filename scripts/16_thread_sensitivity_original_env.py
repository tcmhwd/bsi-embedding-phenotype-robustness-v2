#!/usr/bin/env python3
"""Thread-count sensitivity of the code-of-record UMAP+HDBSCAN pipeline,
run in the frozen original execution environment (/home/programs/anaconda3/envs/gpt).

This repo has an established measurement that thread count changes cluster
counts for this data. This script checks whether the published k=16,
noise=5.54% signature is reachable at some thread count other than 1,
using the ACTUAL original env (not bsi_pheno/miniconda gpt).

Thread count is set via the *_NUM_THREADS env vars, which must be set
before this process starts (i.e. by the SLURM wrapper / caller), since
numba/BLAS libraries read them at import time. This script reports what
it inherited and runs one UMAP+HDBSCAN fit under that setting.

Writes only under phenotype_revision/out/16_c1_sweep/.
"""
import os

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
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from threadpoolctl import threadpool_info

ROOT = Path("/data/tcmhwd/AI/AST_ai")
OUT_DIR = ROOT / "phenotype_revision" / "out" / "16_c1_sweep"
EMB_PATH = ROOT / "pipeline_new/out_pre/case_report_embeddings.npy"
LABELS_PATH = ROOT / "pipeline_new/out_pre/phenotype_clusters_labels.csv"


def agreement(reference: np.ndarray, derived: np.ndarray) -> dict:
    both_nonnoise = (reference != -1) & (derived != -1)
    published_nonnoise = reference != -1
    return {
        "ari_all_rows_noise_as_label": float(adjusted_rand_score(reference, derived)),
        "ari_both_nonnoise": float(
            adjusted_rand_score(reference[both_nonnoise], derived[both_nonnoise])
        ),
        "n_both_nonnoise": int(both_nonnoise.sum()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    threads_requested = os.environ.get("OMP_NUM_THREADS", "unset")
    print(f"[INFO] python executable: {os.sys.executable}")
    print(f"[INFO] requested thread count env: OMP={os.environ.get('OMP_NUM_THREADS')} "
          f"NUMBA={os.environ.get('NUMBA_NUM_THREADS')} "
          f"MKL={os.environ.get('MKL_NUM_THREADS')} "
          f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}")

    emb = np.load(EMB_PATH)
    frame = pd.read_csv(LABELS_PATH, usecols=["phenotype_cluster"])
    reference = frame["phenotype_cluster"].to_numpy(dtype=np.int64)

    n = emb.shape[0]
    min_cluster_size = max(10, n // 200)

    t0 = time.time()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    emb_umap = reducer.fit_transform(emb)
    t_umap = time.time() - t0

    t1 = time.time()
    derived = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit_predict(emb_umap)
    t_hdbscan = time.time() - t1

    k = int(len(set(derived.tolist()) - {-1}))
    noise_fraction = float((derived == -1).mean())

    result = {
        "threads_requested": threads_requested,
        "requested_threads_all": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            )
        },
        "effective_numba_threads": int(numba.get_num_threads()),
        "effective_native_threadpools": threadpool_info(),
        "n_clusters_excluding_noise": k,
        "n_noise": int((derived == -1).sum()),
        "noise_fraction": noise_fraction,
        "published_k": int(len(set(reference.tolist()) - {-1})),
        "published_noise_fraction": float((reference == -1).mean()),
        "agreement": agreement(reference, derived),
        "wall_time_umap_seconds": t_umap,
        "wall_time_hdbscan_seconds": t_hdbscan,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "umap-learn": umap.__version__,
            "hdbscan": importlib.metadata.version("hdbscan"),
            "numba": numba.__version__,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    dest = OUT_DIR / f"result_thread_sensitivity_threads{threads_requested}.json"
    dest.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[INFO] Wrote {dest}")


if __name__ == "__main__":
    main()
