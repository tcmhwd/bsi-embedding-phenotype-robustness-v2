#!/usr/bin/env python3
"""Task 17, S2 (bootstrap/subsample) -- patient-level subsampling (f=0.632,
`03_stability.patient_subsample_indices` semantics: resample distinct
`patient_ID`, keep every admission of a sampled patient; a patient is never
split across in/out), n_boot=200, clustering seed FIXED at 42 (only the
subsample draw seed varies, 42+b) -- DESIGN §2.4. Runs in the frozen
original execution environment on the contemporaneous `out_pre` embeddings.

MUST be invoked with:
    /home/programs/anaconda3/envs/gpt/bin/python3 17b_s2_bootstrap.py \
        --shard-index I --n-shards N

Reference partition: the published labels (`out_pre/phenotype_clusters_labels.csv`).

`min_cluster_size`/`min_samples` are scaled by f=0.632 from the full-cohort
published values (161/161, since `min_samples=None` means "= min_cluster_size"
to HDBSCAN) via `scale_threshold` (round, floor 2) -- exactly
`03_stability.scale_threshold`'s rule.

Writes only under `phenotype_revision/out/17_correct_env/s2_boot/`:
  - `shard_{I:03d}.csv`, one row per (boot rep, published cluster) with the
    DESIGN §2.5 `classify_outcome` fields (jaccard, coassign, noise_frac,
    outcome). Appended incrementally, resumable (skips boot reps already
    present in the shard file).
"""
import os
import sys

for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"  # S2 is not a thread-sensitivity axis; always 1 (matches Design S convention)

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
c17 = importlib.import_module("17_common")

OUT_DIR = c17.OUT / "s2_boot"
FIELDNAMES = ["boot", "cluster", "retained_size", "jaccard", "coassign",
              "noise_frac", "best_recall", "outcome", "wall_time_sec"]


def fit(emb, seed, min_cluster_size, min_samples):
    reducer = umap.UMAP(random_state=seed, n_jobs=1, **c17.UMAP_KW)
    space = reducer.fit_transform(emb)
    cl = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples,
        metric=c17.HDBSCAN_KW["metric"], cluster_selection_method=c17.HDBSCAN_KW["cluster_selection_method"],
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
    my_reps = [b for b in range(c17.N_BOOT) if b % args.n_shards == args.shard_index]
    done = load_completed_reps(shard_path)
    todo = [b for b in my_reps if b not in done]
    print(f"[shard {args.shard_index}/{args.n_shards}] {len(my_reps)} reps assigned, "
          f"{len(done)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return

    ref = c17.load_reference_frame()
    base_labels = ref[c17.PUBLISHED_CLUSTER_COL].to_numpy(dtype=np.int64)
    meta = c17.load_metadata_aligned(ref["admission_ID"].to_numpy())
    patient_ids = meta["patient_ID"].to_numpy()
    emb = c17.load_embeddings()

    mcs_full = c17.min_cluster_size_for(len(base_labels))
    ms_full = mcs_full  # min_samples=None in the published config == min_cluster_size
    mcs_sub = c17.scale_threshold(mcs_full, c17.F_SUBSAMPLE)
    ms_sub = c17.scale_threshold(ms_full, c17.F_SUBSAMPLE)
    print(f"[shard {args.shard_index}] mcs_full={mcs_full} mcs_sub={mcs_sub} ms_sub={ms_sub} "
          f"(config actually used per rep: UMAP={c17.UMAP_KW} HDBSCAN min_cluster_size={mcs_sub} "
          f"min_samples={ms_sub} {c17.HDBSCAN_KW['metric']} {c17.HDBSCAN_KW['cluster_selection_method']})",
          flush=True)

    clusters = sorted(set(base_labels.tolist()) - {-1})
    file_exists = shard_path.exists()
    with open(shard_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for b in todo:
            draw_seed = c17.CLUSTER_SEED + b
            idx = c17.patient_subsample_indices(patient_ids, c17.F_SUBSAMPLE, seed=draw_seed)
            base_sub = base_labels[idx]
            t0 = time.time()
            new_labels = fit(emb[idx], c17.CLUSTER_SEED, mcs_sub, ms_sub)
            wall = time.time() - t0
            n_new_clusters = len(set(new_labels.tolist()) - {-1})
            for k in clusters:
                row = c17.classify_outcome(base_sub, new_labels, k)
                if row is not None:
                    row["boot"] = b
                    row["wall_time_sec"] = round(wall, 2)
                    writer.writerow(row)
            fh.flush()
            print(f"[shard {args.shard_index}] boot {b} done, n_sub={len(idx)}, "
                  f"n_clusters={n_new_clusters}, wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
