#!/usr/bin/env python3
"""Multi-seed / multi-corpus reproduction using the frozen original
execution environment (/home/programs/anaconda3/envs/gpt), for stability
and corpus-specificity checks (C1 follow-up, Step 3).

Usage (must be invoked with that env's interpreter):
    /home/programs/anaconda3/envs/gpt/bin/python3 \
        16_reproduce_original_env_multi.py --source contemporaneous --seed 1
    /home/programs/anaconda3/envs/gpt/bin/python3 \
        16_reproduce_original_env_multi.py --source current --seed 42

--source contemporaneous uses pipeline_new/out_pre/case_report_embeddings.npy
  (the embedding matrix contemporaneous with the published run).
--source current concatenates pipeline_new/out/embeddings_shards/shard_*.npy
  (the embeddings this revision has been using).

Always compares against the published labels
(pipeline_new/out_pre/phenotype_clusters_labels.csv), which is the fixed
reference regardless of source.

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
OUT_DIR = ROOT / "phenotype_revision" / "out" / "16_c1_sweep"
LABELS_PATH = ROOT / "pipeline_new/out_pre/phenotype_clusters_labels.csv"
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
            scores.append((np.count_nonzero(a & b) / union if union else 0.0, candidate))
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] python executable: {os.sys.executable}")
    data = load_embeddings(args.source)
    frame = pd.read_csv(LABELS_PATH, usecols=["phenotype_cluster"])
    reference = frame["phenotype_cluster"].to_numpy(dtype=np.int64)
    if data.shape[0] != len(reference):
        raise ValueError(f"Row mismatch: labels={len(reference)} data={data.shape[0]}")

    n = data.shape[0]
    min_cluster_size = max(10, n // 200)

    t0 = time.time()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="euclidean",
        random_state=args.seed,
    )
    coordinates = reducer.fit_transform(data)
    t_umap = time.time() - t0

    t1 = time.time()
    derived = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,
        metric="euclidean",
        cluster_selection_method="eom",
    ).fit_predict(coordinates)
    t_hdbscan = time.time() - t1

    k = int(len(set(derived.tolist()) - {-1}))
    noise_fraction = float((derived == -1).mean())

    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "sklearn": sklearn.__version__,
        "umap-learn": umap.__version__,
        "hdbscan": importlib.metadata.version("hdbscan"),
        "numba": numba.__version__,
        "llvmlite": __import__("llvmlite").__version__,
        "pynndescent": __import__("pynndescent").__version__,
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

    stem = f"{args.source}_seed{args.seed}_origenv"
    np.savez_compressed(
        OUT_DIR / f"labels_{stem}.npz",
        labels=derived.astype(np.int32),
        umap=coordinates.astype(np.float32),
    )
    dest = OUT_DIR / f"result_{stem}.json"
    dest.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"[INFO] Wrote {dest}")


if __name__ == "__main__":
    main()
