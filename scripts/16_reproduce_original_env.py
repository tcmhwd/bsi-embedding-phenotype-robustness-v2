#!/usr/bin/env python3
"""Reproduce the published phenotype clusters using the ACTUAL original
execution environment (/home/programs/anaconda3/envs/gpt), frozen since
2025-12-02 (confirmed via conda-meta/history — no operations after the
umap-learn/hdbscan install at 14:43:31 that same day, and no conda-meta
file modified after 2025-12-02 20:00).

This environment differs from every environment used in prior reproduction
attempts (bsi_pheno, and the /data/tcmhwd/miniconda3 "gpt" env, which is a
DIFFERENT, later-created env with the same name -- see pheno_umap.err,
which shows the original run actually used
/home/programs/anaconda3/envs/gpt/lib/python3.10/site-packages/...):

    package        original (this env)    all prior repro attempts
    numpy          2.2.6                   1.26.4
    scipy          1.15.2                  1.11.4
    umap-learn     0.5.9.post2             0.5.7
    scikit-learn   1.7.2                   1.4.2 / 1.7.2 (already ruled out)
    hdbscan        0.8.39                  0.8.39 (same)
    numba          0.62.1                  0.62.1 (same)
    llvmlite       0.45.1                  0.45.1 (same)
    pynndescent    0.5.13                  0.5.13 (same)

Must be invoked with:
    /home/programs/anaconda3/envs/gpt/bin/python3 16_reproduce_original_env.py

Reuses exactly the UMAP+HDBSCAN calls in
pipeline_new/phenotype_umap_hdbscan.py (read, not modified) applied to the
contemporaneous embedding (pipeline_new/out_pre/case_report_embeddings.npy),
and compares against the published labels
(pipeline_new/out_pre/phenotype_clusters_labels.csv).

Writes only under phenotype_revision/out/16_c1_sweep/.
"""
import os

for _key in (
    "OMP_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ[_key] = "1"

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
        "ami_all_rows_noise_as_label": float(adjusted_mutual_info_score(reference, derived)),
        "ari_both_nonnoise": float(
            adjusted_rand_score(reference[both_nonnoise], derived[both_nonnoise])
        ),
        "ami_both_nonnoise": float(
            adjusted_mutual_info_score(reference[both_nonnoise], derived[both_nonnoise])
        ),
        "n_both_nonnoise": int(both_nonnoise.sum()),
        "ari_published_nonnoise": float(
            adjusted_rand_score(reference[published_nonnoise], derived[published_nonnoise])
        ),
        "ami_published_nonnoise": float(
            adjusted_mutual_info_score(reference[published_nonnoise], derived[published_nonnoise])
        ),
        "n_published_nonnoise": int(published_nonnoise.sum()),
    }


def best_jaccards(reference: np.ndarray, derived: np.ndarray) -> dict:
    answer = {}
    candidates = sorted(set(derived.tolist()) - {-1})
    for published_cluster in sorted(set(reference.tolist()) - {-1}):
        a = reference == published_cluster
        scores = []
        for candidate in candidates:
            b = derived == candidate
            union = np.count_nonzero(a | b)
            scores.append((np.count_nonzero(a & b) / union if union else 0.0, candidate))
        score, candidate = max(scores, default=(0.0, None))
        answer[str(int(published_cluster))] = {
            "best_derived_cluster": None if candidate is None else int(candidate),
            "jaccard": float(score),
            "published_size": int(a.sum()),
        }
    return answer


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] python executable: {os.sys.executable}")
    print(f"[INFO] Loading embeddings: {EMB_PATH}")
    emb = np.load(EMB_PATH)
    print(f"[INFO] Embeddings shape: {emb.shape}")

    print(f"[INFO] Loading published labels: {LABELS_PATH}")
    frame = pd.read_csv(LABELS_PATH, usecols=["phenotype_cluster"])
    reference = frame["phenotype_cluster"].to_numpy(dtype=np.int64)
    if len(reference) != emb.shape[0]:
        raise ValueError(f"Row mismatch: labels={len(reference)} emb={emb.shape[0]}")

    n = emb.shape[0]
    min_cluster_size = max(10, n // 200)

    t0 = time.time()
    print("[INFO] Running UMAP (code-of-record params: n_components=2, "
          "n_neighbors=15, min_dist=0.1, metric=euclidean, random_state=42)...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=42,
    )
    emb_umap = reducer.fit_transform(emb)
    t_umap = time.time() - t0

    print(f"[INFO] Running HDBSCAN (min_cluster_size={min_cluster_size}, "
          "min_samples=None, eom, euclidean)...")
    t1 = time.time()
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    derived = clusterer.fit_predict(emb_umap)
    t_hdbscan = time.time() - t1

    k = int(len(set(derived.tolist()) - {-1}))
    noise_fraction = float((derived == -1).mean())
    print(f"[INFO] k={k} noise_fraction={noise_fraction:.4f} "
          f"n_noise={int((derived == -1).sum())}")

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "umap-learn": umap.__version__,
        "hdbscan": importlib.metadata.version("hdbscan"),
        "numba": numba.__version__,
        "llvmlite": __import__("llvmlite").__version__,
        "pynndescent": __import__("pynndescent").__version__,
    }

    result = {
        "description": "Reproduction using the frozen original execution "
                        "environment /home/programs/anaconda3/envs/gpt "
                        "(confirmed via conda-meta/history to be unmodified "
                        "since 2025-12-02, the day of the published run).",
        "python_executable": os.sys.executable,
        "shape": list(emb.shape),
        "configuration": {
            "umap": {
                "n_components": 2,
                "n_neighbors": 15,
                "min_dist": 0.1,
                "metric": "euclidean",
                "random_state": 42,
            },
            "hdbscan": {
                "min_cluster_size": min_cluster_size,
                "min_samples": None,
                "metric": "euclidean",
                "cluster_selection_method": "eom",
            },
        },
        "n_clusters_excluding_noise": k,
        "n_noise": int((derived == -1).sum()),
        "noise_fraction": noise_fraction,
        "published_k": int(len(set(reference.tolist()) - {-1})),
        "published_n_noise": int((reference == -1).sum()),
        "published_noise_fraction": float((reference == -1).mean()),
        "agreement": agreement(reference, derived),
        "best_match_jaccard": best_jaccards(reference, derived),
        "versions": versions,
        "requested_threads": {
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
        "wall_time_umap_seconds": t_umap,
        "wall_time_hdbscan_seconds": t_hdbscan,
    }

    np.savez_compressed(
        OUT_DIR / "labels_original_env_seed42.npz",
        labels=derived.astype(np.int32),
        umap=emb_umap.astype(np.float32),
    )
    out_path = OUT_DIR / "result_original_env_seed42.json"
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[INFO] Wrote {out_path}")


if __name__ == "__main__":
    main()
