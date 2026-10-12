#!/usr/bin/env python3
"""Task 22 Task C, step 0 -- compute the full-cohort Gower distance matrix
ONCE and persist it to disk (`compute_gower_matrix` from
`05_baseline_comparators.py` is reused unmodified; this script only adds
persistence, which the reviewed 05b/05c code deliberately does not do).

Rationale: `05c_comparison_table.py`'s own docstring documents the O(n^2)
Gower matrix build at 1640.5s (~27.3 min) for n=32,329, and explicitly
recomputes it once per script invocation (never persisted). Task C here
needs the matrix from THREE separate processes (point-estimate fit,
patient-level bootstrap over 100+ replicates, and any later resumed run) --
recomputing it in each would waste ~27 min every time. Persisting it once
lets every downstream step `np.load(..., mmap_mode="r")` it directly.

Also persists `admission_ID` in the exact row order of the matrix, so
downstream scripts can align without re-deriving the frame.

Writes only under
`phenotype_revision/out/22_published_objects/task_c_comparators/`:
  - `gower_distance_matrix.npy` (float32, n x n, ~4.18 GB)
  - `gower_admission_ids.csv` (row order)
  - `gower_precompute_manifest.json`

Resumable: skips entirely if the matrix file already exists.
"""
import os

os.environ["OMP_NUM_THREADS"] = os.environ["NUMBA_NUM_THREADS"] = \
    os.environ["MKL_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
frame_mod = importlib.import_module("01_build_analysis_frame")
comparators = importlib.import_module("05_baseline_comparators")

OUT = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision/out/22_published_objects/task_c_comparators")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    matrix_path = OUT / "gower_distance_matrix.npy"
    if matrix_path.exists():
        print(f"[gower_precompute] {matrix_path} already exists, skipping (resume)", flush=True)
        return

    print("[gower_precompute] loading frame + structured matrix", flush=True)
    frame = frame_mod.load_frame(verify_reports=False)
    matrix, categorical_cols = comparators.build_structured_matrix(frame)
    admission_ids = frame["admission_ID"].to_numpy()
    numeric_cols = [c for c in matrix.columns if c not in categorical_cols]
    print(f"[gower_precompute] numeric_cols actually used ({len(numeric_cols)}): {numeric_cols}", flush=True)
    print(f"[gower_precompute] categorical_cols actually used ({len(categorical_cols)}): {categorical_cols}", flush=True)

    t0 = time.time()
    distance = comparators.compute_gower_matrix(matrix, categorical_cols)
    wall = time.time() - t0
    print(f"[gower_precompute] computed {distance.shape} matrix in {wall:.1f}s, "
          f"dtype={distance.dtype}, bytes={distance.nbytes}", flush=True)

    # `np.save` auto-appends ".npy" to any filename that doesn't already end
    # in ".npy" -- so a tmp name like "*.npy.tmp" silently becomes
    # "*.npy.tmp.npy" on disk, and the atomic os.replace(tmp, matrix_path)
    # below then fails with FileNotFoundError (caught the hard way: job
    # 587526 spent 1791s computing the matrix correctly, then hit exactly
    # this on the save step). Use a tmp name that itself ends in ".npy" so
    # numpy does not rewrite it out from under us.
    tmp = matrix_path.with_name(matrix_path.stem + ".tmp.npy")
    np.save(tmp, distance)
    os.replace(tmp, matrix_path)

    pd.DataFrame({"admission_ID": admission_ids}).to_csv(OUT / "gower_admission_ids.csv", index=False)

    manifest = dict(
        n_rows=int(distance.shape[0]), dtype=str(distance.dtype), bytes=int(distance.nbytes),
        wall_time_sec=round(wall, 1), numeric_cols=numeric_cols, categorical_cols=categorical_cols,
        dense_variant_fix={
            "applied": True,
            "sparse_to_dense_map": comparators.DENSE_VARIANT_FIX_MAP,
            "superseded_sparse_numeric_cols": comparators.SUPERSEDED_SPARSE_NUMERIC_COLS,
            "superseded_sparse_categorical_cols": comparators.SUPERSEDED_SPARSE_CATEGORICAL_COLS,
        },
        python_executable=sys.executable, python_prefix=sys.prefix,
    )
    (OUT / "gower_precompute_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[gower_precompute] wrote {matrix_path}", flush=True)


if __name__ == "__main__":
    main()
