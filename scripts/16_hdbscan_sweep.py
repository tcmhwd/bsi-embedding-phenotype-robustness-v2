#!/usr/bin/env python3
"""HDBSCAN parameter sweep on cached seed-42 UMAP coordinates (C1 follow-up).

UMAP is not refit here -- it reuses the 2-D coordinates already produced by
out/12_codex_c1/c1_cluster.py for source=contemporaneous, seed=42
(labels_contemporaneous_seed42_1_7_2.npz), which were fit from the
contemporaneous embedding (pipeline_new/out_pre/case_report_embeddings.npy)
with the code-of-record UMAP settings (n_components=2, n_neighbors=15,
min_dist=0.1, euclidean, random_state=42).

Only HDBSCAN parameters are swept: min_cluster_size, min_samples,
cluster_selection_method. For every combination we record k (excluding
noise), noise fraction, ARI/AMI vs the published labels (all-rows and
noise-excluded variants), and the per-published-cluster best-match Jaccard.

Resumable: each completed combination is appended as one JSON line to
out/16_c1_sweep/sweep_results.jsonl; combinations already present are
skipped on rerun.
"""
import os

for _key in (
    "OMP_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
):
    os.environ[_key] = "1"

import itertools
import json
import time
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

ROOT = Path("/data/tcmhwd/AI/AST_ai")
REV = ROOT / "phenotype_revision"
LABELS_CSV = ROOT / "pipeline_new/out_pre/phenotype_clusters_labels.csv"
CACHED_NPZ = REV / "out/12_codex_c1/labels_contemporaneous_seed42_1_7_2.npz"
OUT_DIR = REV / "out/16_c1_sweep"
RESULTS_JSONL = OUT_DIR / "sweep_results.jsonl"

MIN_CLUSTER_SIZE_GRID = list(range(20, 251, 5))
MIN_SAMPLES_GRID = [None, 1, 5, 10, 15, 20, 30, 50]
CLUSTER_SELECTION_METHODS = ["eom", "leaf"]


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


def load_done_keys() -> set:
    done = set()
    if RESULTS_JSONL.exists():
        with RESULTS_JSONL.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((row["min_cluster_size"], row["min_samples"], row["cluster_selection_method"]))
    return done


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading cached UMAP coordinates: {CACHED_NPZ}")
    cached = np.load(CACHED_NPZ)
    coords = cached["umap"].astype(np.float64)
    print(f"[INFO] coords shape={coords.shape} dtype={coords.dtype}")

    print(f"[INFO] Loading published labels: {LABELS_CSV}")
    frame = pd.read_csv(LABELS_CSV, usecols=["phenotype_cluster"])
    reference = frame["phenotype_cluster"].to_numpy(dtype=np.int64)
    if len(reference) != coords.shape[0]:
        raise ValueError(f"Row mismatch: labels={len(reference)} coords={coords.shape[0]}")

    published_noise_frac = float((reference == -1).mean())
    published_k = len(set(reference.tolist()) - {-1})
    print(f"[INFO] Published: k={published_k}, noise_fraction={published_noise_frac:.4f}")

    done = load_done_keys()
    print(f"[INFO] {len(done)} combinations already completed; resuming.")

    grid = list(itertools.product(MIN_CLUSTER_SIZE_GRID, MIN_SAMPLES_GRID, CLUSTER_SELECTION_METHODS))
    print(f"[INFO] Grid size: {len(grid)} combinations")

    n_run = 0
    t_start = time.time()
    with RESULTS_JSONL.open("a") as out_fh:
        for min_cluster_size, min_samples, method in grid:
            key = (min_cluster_size, min_samples, method)
            if key in done:
                continue
            t0 = time.time()
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method=method,
            )
            derived = clusterer.fit_predict(coords)
            elapsed = time.time() - t0

            k = int(len(set(derived.tolist()) - {-1}))
            noise_fraction = float((derived == -1).mean())

            row = {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "cluster_selection_method": method,
                "k": k,
                "n_noise": int((derived == -1).sum()),
                "noise_fraction": noise_fraction,
                "agreement": agreement(reference, derived),
                "best_match_jaccard": best_jaccards(reference, derived),
                "fit_seconds": elapsed,
            }
            out_fh.write(json.dumps(row) + "\n")
            out_fh.flush()
            n_run += 1
            if n_run % 20 == 0:
                print(
                    f"[INFO] {n_run} new combos done "
                    f"(elapsed {time.time() - t_start:.1f}s); last: "
                    f"mcs={min_cluster_size} ms={min_samples} method={method} "
                    f"k={k} noise={noise_fraction:.4f} "
                    f"ari_all={row['agreement']['ari_all_rows_noise_as_label']:.4f}"
                )

    print(f"[INFO] Done. {n_run} new combinations run in {time.time() - t_start:.1f}s")
    print(f"[INFO] Results: {RESULTS_JSONL}")


if __name__ == "__main__":
    main()
