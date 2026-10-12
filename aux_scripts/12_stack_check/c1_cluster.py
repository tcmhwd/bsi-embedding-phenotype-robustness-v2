#!/usr/bin/env python3
"""Independent UMAP/HDBSCAN reproduction for the C1 phenotype audit.

This file intentionally does not import any revision-project loader or
clustering helper.  It reconstructs both candidate embedding matrices and
computes agreement directly against the code-of-record labels.
"""
import os

# These must be fixed before importing numpy/numba/scikit-learn/UMAP.
for _key in (
    "OMP_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ[_key] = "1"

import argparse
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
HERE = ROOT / "phenotype_revision/out/12_codex_c1"
LABELS = ROOT / "pipeline_new/out_pre/phenotype_clusters_labels.csv"
OLD_EMBEDDINGS = ROOT / "pipeline_new/out_pre/case_report_embeddings.npy"
CURRENT_SHARDS = ROOT / "pipeline_new/out/embeddings_shards"


def load_embeddings(source: str) -> np.ndarray:
    if source == "contemporaneous":
        return np.load(OLD_EMBEDDINGS)
    shards = sorted(CURRENT_SHARDS.glob("shard_*.npy"))
    if not shards:
        raise FileNotFoundError(f"No shards in {CURRENT_SHARDS}")
    return np.concatenate([np.load(p) for p in shards], axis=0)


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
            scores.append((np.count_nonzero(a & b) / union, candidate))
        score, candidate = max(scores, default=(0.0, None))
        answer[str(int(published_cluster))] = {
            "best_derived_cluster": None if candidate is None else int(candidate),
            "jaccard": float(score),
            "published_size": int(a.sum()),
        }
    return answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["current", "contemporaneous"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    HERE.mkdir(parents=True, exist_ok=True)
    data = load_embeddings(args.source)
    frame = pd.read_csv(LABELS, usecols=["phenotype_cluster"])
    reference = frame["phenotype_cluster"].to_numpy(dtype=np.int64)
    if data.shape != (len(reference), 3072):
        raise ValueError(f"Unexpected matrix shape {data.shape}; labels={len(reference)}")

    started = time.time()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=args.seed,
    )
    coordinates = reducer.fit_transform(data)
    derived = hdbscan.HDBSCAN(
        min_cluster_size=max(10, len(data) // 200),
        min_samples=None,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit_predict(coordinates)

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "umap-learn": umap.__version__,
        "hdbscan": importlib.metadata.version("hdbscan"),
        "numba": numba.__version__,
    }
    result = {
        "source": args.source,
        "seed": args.seed,
        "shape": list(data.shape),
        "configuration": {
            "umap": {
                "n_components": 2,
                "n_neighbors": 15,
                "min_dist": 0.1,
                "metric": "euclidean",
                "random_state": args.seed,
            },
            "hdbscan": {
                "min_cluster_size": max(10, len(data) // 200),
                "min_samples": None,
                "metric": "euclidean",
                "cluster_selection_method": "eom",
            },
        },
        "n_clusters_excluding_noise": int(len(set(derived.tolist()) - {-1})),
        "n_noise": int((derived == -1).sum()),
        "noise_fraction": float((derived == -1).mean()),
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
        "wall_time_seconds": time.time() - started,
    }

    stem = f"{args.source}_seed{args.seed}_{versions['sklearn'].replace('.', '_')}"
    np.savez_compressed(
        HERE / f"labels_{stem}.npz",
        labels=derived.astype(np.int32),
        umap=coordinates.astype(np.float32),
    )
    destination = HERE / f"result_{stem}.json"
    destination.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
