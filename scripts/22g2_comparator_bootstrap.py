#!/usr/bin/env python3
"""Task 22 Task C, step 2 -- patient-level bootstrap ARI for ONE (method, k)
combo, sharded so replicates can run in parallel (n_boot target=100, spread
across `--n-shards` array tasks of `--n-shards` each).

MUST be invoked with:
    <bsi_pheno python3> 22g2_comparator_bootstrap.py \
        --method NAME --k K --shard-index I --n-shards N [--n-boot 100]

Requires the point estimate for this (method, k) to already exist
(`point_{method}_k{k}.json` / `labels_{method}_k{k}.npz`) -- its labels are
the `base_labels` each bootstrap replicate's ARI is measured against (same
convention as `03_stability.bootstrap_stability_generic`).

For `gower_pam`, reuses the precomputed full-cohort Gower matrix
(`22g0_gower_precompute.py`) via `np.ix_` submatrix extraction -- see
`22g_comparator_common.patient_bootstrap_ari` docstring ("Fix 2").

Writes only under
`phenotype_revision/out/22_published_objects/task_c_comparators/bootstrap/`:
  - `{method}_k{k}_shard{I:03d}.csv`, one row per boot rep (ari, wall_time).
Appended incrementally, resumable (skips boot reps already present).
"""
import os

os.environ["OMP_NUM_THREADS"] = os.environ["NUMBA_NUM_THREADS"] = \
    os.environ["MKL_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import csv
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c22g = importlib.import_module("22g_comparator_common")

BOOT_DIR = c22g.OUT / "bootstrap"
FIELDNAMES = ["boot", "ari", "wall_time_sec"]


def load_completed(path: Path) -> set:
    if not path.exists():
        return set()
    import pandas as pd
    return set(pd.read_csv(path, usecols=["boot"])["boot"].tolist())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(c22g.METHOD_FIT_FUNCS))
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--n-boot", type=int, default=c22g.N_BOOT_TARGET)
    ap.add_argument("--seed", type=int, default=c22g.SEED)
    args = ap.parse_args()

    BOOT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.method}_k{args.k}"
    shard_path = BOOT_DIR / f"{tag}_shard{args.shard_index:03d}.csv"

    point_path = c22g.OUT / f"point_{tag}.json"
    if not point_path.exists():
        raise SystemExit(f"{point_path} missing -- run 22g1_comparator_point.py --method {args.method} --k {args.k} first")
    point = json.loads(point_path.read_text())
    labels_npz = np.load(c22g.OUT / point["labels_path"])
    base_labels = labels_npz["labels"]

    my_reps = [b for b in range(args.n_boot) if b % args.n_shards == args.shard_index]
    done = load_completed(shard_path)
    todo = [b for b in my_reps if b not in done]
    print(f"[{tag} shard {args.shard_index}/{args.n_shards}] {len(my_reps)} reps assigned, "
          f"{len(done)} already done, {len(todo)} to run", flush=True)
    if not todo:
        return

    print(f"[{tag} shard {args.shard_index}] loading frame + structured matrix", flush=True)
    frame, matrix, categorical_cols, imputed, admission_ids, patient_ids = c22g.load_common()
    if not np.array_equal(admission_ids, labels_npz["admission_ids"]):
        raise RuntimeError("admission_ID row order mismatch between point-estimate labels and this shard's frame load")

    full_gower = None
    if args.method == "gower_pam":
        c22g.assert_gower_row_order_matches(admission_ids)
        full_gower = c22g.load_gower_matrix()

    fit_func = c22g.METHOD_FIT_FUNCS[args.method]
    file_exists = shard_path.exists()
    with open(shard_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for b in todo:
            t0 = time.time()
            ari_list, _ = c22g.patient_bootstrap_ari(
                imputed, categorical_cols, fit_func, args.method, args.k,
                base_labels, patient_ids, n_boot=1, seed=args.seed + b,
                full_gower_matrix=full_gower,
            )
            wall = time.time() - t0
            writer.writerow({"boot": b, "ari": ari_list[0], "wall_time_sec": round(wall, 2)})
            fh.flush()
            print(f"[{tag} shard {args.shard_index}] boot {b} done ari={ari_list[0]:.4f} wall={wall:.1f}s", flush=True)


if __name__ == "__main__":
    main()
