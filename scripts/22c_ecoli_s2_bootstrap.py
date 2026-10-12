#!/usr/bin/env python3
"""Task 22 Task A, S2 (bootstrap/subsample) -- patient-level subsampling
(f=0.632, resample distinct `patient_ID`, keep all their admissions),
n_boot=200, clustering seed FIXED at 42 (only the subsample draw seed
varies, 42+b) -- PRESPEC section 2 (S2 row) / DESIGN section 2.4-2.5.

MUST be invoked with:
    /home/programs/anaconda3/envs/gpt/bin/python3 22c_ecoli_s2_bootstrap.py \
        --shard-index I --n-shards N

Reference partition: `ecoli_umap_hdbscan_labels.csv[ecoli_cluster]`.

`min_cluster_size`/`min_samples` are scaled by f=0.632 from this object's own
Table A.2 values (300/50), via `scale_threshold` (round, floor 2) --
identical rule to `03_stability.scale_threshold` / `17_common.scale_threshold`.

Writes only under
`phenotype_revision/out/22_published_objects/task_a_ecoli/s2_boot/`:
  - `shard_{I:03d}.csv`, one row per (boot rep, reference cluster) with the
    DESIGN section 2.5 `classify_outcome` fields. Appended incrementally,
    resumable (skips boot reps already present in the shard file).
"""
import os
import sys

for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"  # S2 is not a thread-sensitivity axis; always 1 (Design S convention)

import argparse
import csv
import importlib
import time
from pathlib import Path

import hdbscan
import numpy as np
import umap

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c22 = importlib.import_module("22a_ecoli_common")

OUT_DIR = c22.OUT / "s2_boot"
FIELDNAMES = ["boot", "cluster", "retained_size", "jaccard", "coassign",
              "noise_frac", "best_recall", "outcome", "wall_time_sec"]


def fit(emb, seed, min_cluster_size, min_samples):
    reducer = umap.UMAP(random_state=seed, n_jobs=1, **c22.UMAP_KW)
    space = reducer.fit_transform(emb)
    cl = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples,
        metric=c22.HDBSCAN_KW["metric"], cluster_selection_method=c22.HDBSCAN_KW["cluster_selection_method"],
        core_dist_n_jobs=1,
    ).fit(np.ascontiguousarray(space, dtype=np.float64))
    return cl.labels_


def load_completed_reps(path: Path) -> set:
    if not path.exists():
        return set()
    import pandas as pd
    df = pd.read_csv(path, usecols=["boot"])
    return set(df["boot"].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shard_path = OUT_DIR / f"shard_{args.shard_index:03d}.csv"
    my_reps = [b for b in range(c22.N_BOOT) if b % args.n_shards == args.shard_index]
    done = load_completed_reps(shard_path)
    todo = [b for b in my_reps if b not in done]
    print(f"[shard {args.shard_index}/{args.n_shards}] {len(my_reps)} reps assigned, "
          f"{len(done)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return

    ref = c22.load_reference_frame()
    base_labels = ref[c22.REFERENCE_CLUSTER_COL].to_numpy(dtype=np.int64)
    patient_ids = ref["patient_ID"].to_numpy()
    emb = c22.load_embeddings()

    mcs_sub = c22.scale_threshold(c22.MIN_CLUSTER_SIZE, c22.F_SUBSAMPLE)
    ms_sub = c22.scale_threshold(c22.MIN_SAMPLES, c22.F_SUBSAMPLE)
    print(f"[shard {args.shard_index}] mcs_full={c22.MIN_CLUSTER_SIZE} ms_full={c22.MIN_SAMPLES} "
          f"mcs_sub={mcs_sub} ms_sub={ms_sub} (config actually used per rep: UMAP={c22.UMAP_KW} "
          f"HDBSCAN min_cluster_size={mcs_sub} min_samples={ms_sub} {c22.HDBSCAN_KW['metric']} "
          f"{c22.HDBSCAN_KW['cluster_selection_method']})", flush=True)

    clusters = sorted(set(base_labels.tolist()) - {-1})
    file_exists = shard_path.exists()
    with open(shard_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for b in todo:
            draw_seed = c22.CLUSTER_SEED + b
            idx = c22.patient_subsample_indices(patient_ids, c22.F_SUBSAMPLE, seed=draw_seed)
            base_sub = base_labels[idx]
            t0 = time.time()
            new_labels = fit(emb[idx], c22.CLUSTER_SEED, mcs_sub, ms_sub)
            wall = time.time() - t0
            n_new_clusters = len(set(new_labels.tolist()) - {-1})
            for k in clusters:
                row = c22.classify_outcome(base_sub, new_labels, k)
                if row is not None:
                    row["boot"] = b
                    row["wall_time_sec"] = round(wall, 2)
                    writer.writerow(row)
            fh.flush()
            print(f"[shard {args.shard_index}] boot {b} done, n_sub={len(idx)}, "
                  f"n_clusters={n_new_clusters}, wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
