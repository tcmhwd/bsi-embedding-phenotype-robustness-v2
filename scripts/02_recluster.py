#!/usr/bin/env python3
"""High-dimensional UMAP + HDBSCAN grid. 2D is kept for visualization only.

The originally derived [manuscript] ran HDBSCAN on 2D UMAP coordinates. This
script re-runs HDBSCAN on that same clustering space (2D) as well as higher
dimensions (10/20/50) and the raw space (3072 dimensions, no projection), to
build the evidence base for determining whether the originally derived
cluster count (16) and noise fraction (5.5%) are an artifact of the 2D
projection itself.

grid = {n_components in 2,10,20,50} x {n_neighbors in 15,30,50} x
       {min_cluster_size in 50,100,200}                              = 36
     + {n_components = 0 (raw space, no UMAP)} x {min_cluster_size in 50,100,200} = 3
     = 39 configs (exploratory grid; `min_dist=0.0`/`metric=cosine`/`min_samples=10`)
     + config_index 39 = PUBLISHED_CONFIG (the configuration the originally
       derived [manuscript] actually used -- 2D, nn15, min_dist=0.1,
       euclidean, mcs=161, min_samples=None -- audit finding M1 fix: the
       earlier grid never once included this configuration)
     = 40 configs total.

Each config is run independently as a single SLURM array task selected via
`--config-index`, writing only its own labels_<tag>.parquet /
summary_<tag>.json (not appending to a shared CSV -- to avoid concurrent-
write conflicts). Aggregation across configs is performed separately by
`02_aggregate.py`.

Outcome (mortality) variables are not loaded (Global Constraints).
"""
import os

# Force single-threaded BLAS/OpenMP/numba execution BEFORE numpy/sklearn/
# umap/hdbscan/numba are imported. These libraries read these env vars at
# import time, so setting them later (even at the top of main()) silently
# does nothing. This is required for reproducibility, not just performance:
# with these left at their defaults, clustering is non-deterministic across
# thread counts even with random_state fixed (empirically confirmed: 8 vs 24
# threads gave 18 vs 33 clusters for the same config/seed). Pinning every
# thread pool to 1 trades wall-clock speed for a clustering that is
# identical regardless of how many CPUs the job was scheduled with.
#
# NOTE on umap.UMAP's own n_jobs kwarg (passed below in run_config): it is
# NOT the mechanism behind this fix and never was. Installed
# umap-learn==0.5.7 (umap_.py ~L1950-1953) unconditionally forces its
# internal n_jobs to 1 whenever random_state is not None (emitting a
# UserWarning), and this module always passes random_state=seed. So the
# explicit n_jobs=1 kwarg below is inert -- pynndescent was already running
# single-threaded regardless of that argument. Determinism instead comes
# entirely from the BLAS/numba thread-pool pinning above (OMP/NUMBA/MKL/
# OPENBLAS_NUM_THREADS=1), which controls the thread-count-dependent
# floating-point accumulation order inside UMAP's and HDBSCAN's native
# numerical routines. n_jobs=1 is kept as harmless, self-documenting intent,
# not as the fix.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import importlib
import importlib.metadata
import itertools
import json
import sys
import time
from pathlib import Path

import hdbscan
import numba
import numpy as np
import pandas as pd
import sklearn
import umap

try:
    import threadpoolctl
except ImportError:  # ships with scikit-learn, but guard anyway
    threadpoolctl = None

sys.path.insert(0, os.path.dirname(__file__))
frame_mod = importlib.import_module("01_build_analysis_frame")

OUT = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision/out/02_recluster")

N_COMPONENTS_GRID = [2, 10, 20, 50]
N_NEIGHBORS_GRID = [15, 30, 50]
MIN_CLUSTER_SIZE_GRID = [50, 100, 200]

# The ORIGINAL published configuration, i.e. the exact UMAP+HDBSCAN settings
# used by `pipeline_new/phenotype_umap_hdbscan.py` to produce the
# `phenotype_cluster` column the manuscript reports on. This is NOT one of
# the 39 grid points above -- the grid's `min_dist=0.0`/`metric="cosine"`/
# `min_samples=10` never actually reproduced the published setting (audit
# finding M1: "the re-clustering grid never contains the published
# configuration"), so the grid's noise/cluster-count range and the
# thread-count nondeterminism demonstration were both measured on settings
# the manuscript never uses.
#
# Values confirmed directly against two independent sources (both read-only,
# neither modified):
#   1. `pipeline_new/phenotype_umap_hdbscan.py` (the actual code that
#      produced `phenotype_cluster`): `umap.UMAP(n_components=2,
#      n_neighbors=15, min_dist=0.1, metric="euclidean", random_state=42)`
#      then `hdbscan.HDBSCAN(min_cluster_size=max(10, n // 200),
#      min_samples=None, metric="euclidean",
#      cluster_selection_method="eom")`.
#   2. The authors' earlier supplementary material,
#      Table S2 ("Analytical parameters of UMAP and HDBSCAN"), "Full BSI
#      cohort" row: "n_neighbors=15, min_dist=0.1, n_components=2,
#      metric='euclidean', random_state=42" / "min_cluster_size=max(10,
#      floor(N/200)), min_samples=None, cluster_selection_method='eom',
#      metric='euclidean'".
# Both sources agree exactly -- every field below is CONFIRMED, not
# assumed. `min_cluster_size=161` is `n // 200` for this project's fixed
# full cohort (n=32,329, established by Task 1 and unchanged since);
# hardcoded here (rather than recomputed from a live frame at grid-build
# time, before any data is loaded) because this project's full-cohort n is
# fixed and this matches the audit's own stated corrected value.
PUBLISHED_CONFIG = {
    "n_components": 2,
    "n_neighbors": 15,
    "min_dist": 0.1,
    "metric": "euclidean",
    "min_cluster_size": 161,  # 32_329 // 200, confirmed against Table S2's formula
    "min_samples": None,
    "cluster_selection_method": "eom",
    "published": True,  # marks this as the published setting, not an anonymous grid point
}


def build_grid() -> list[dict]:
    """Return the 40 configs in deterministic order.

    The first 36 are UMAP (n_components in {2,10,20,50}) x n_neighbors x
    min_cluster_size, and the next 3 are direct clustering in the raw space
    (n_components=0, n_neighbors=None) x min_cluster_size (these 39 are the
    exploratory grid with `min_dist=0.0`/`metric="cosine"`/`min_samples=10`
    -- not the originally derived configuration). The final, 40th entry
    (`config_index=39`) is `PUBLISHED_CONFIG` -- the configuration the
    originally derived [manuscript] actually used -- and is explicitly
    marked `"published": True` so it does not get mixed up with the other
    39, and also receives a separately identifiable tag in `tag_for()`
    (fix for audit finding M1: "the grid never once included the
    originally derived configuration").
    """
    grid = []
    for nc, nn, mcs in itertools.product(
        N_COMPONENTS_GRID, N_NEIGHBORS_GRID, MIN_CLUSTER_SIZE_GRID
    ):
        grid.append({"n_components": nc, "n_neighbors": nn, "min_cluster_size": mcs})
    for mcs in MIN_CLUSTER_SIZE_GRID:
        grid.append({"n_components": 0, "n_neighbors": None, "min_cluster_size": mcs})
    grid.append(dict(PUBLISHED_CONFIG))
    return grid


def tag_for(cfg: dict) -> str:
    if cfg.get("published"):
        # Deliberately verbose/unambiguous -- must be identifiable as "the
        # published setting" in every output (labels_*.parquet,
        # summary_*.json, grid_summary.csv) without cross-referencing this
        # module, per the audit's M1 correction.
        return "published_full_cohort_nc2_nn15_mindist0.1_euclidean_mcs161_msNone"
    if cfg["n_components"] == 0:
        return f"nc0_mcs{cfg['min_cluster_size']}"
    return f"nc{cfg['n_components']}_nn{cfg['n_neighbors']}_mcs{cfg['min_cluster_size']}"


def run_config(
    emb, n_components, n_neighbors, min_cluster_size, seed=42, return_models=False,
    min_dist=0.0, metric="cosine", min_samples=10, cluster_selection_method="eom",
):
    """Fit UMAP (optional) + HDBSCAN once.

    The defaults for `min_dist`/`metric`/`min_samples`/`cluster_selection_method`
    (0.0/cosine/10/eom) are exactly the values Task 2's 39-config
    exploratory grid has always used -- the existing 39-config grid path
    that calls this function, `03_stability.py`, `05c_comparison_table.py`,
    and `08_transport.py` do not pass these arguments, so behavior is
    completely unchanged (backward compatible, no regression). These
    values are explicitly overridden only when running `PUBLISHED_CONFIG`
    (the originally derived configuration: `min_dist=0.1`,
    `metric="euclidean"`, `min_samples=None`).

    `return_models=False` (the default) is completely identical to the
    prior behavior (backward compatible). If `return_models=True`, the
    returned dict additionally carries the fitted `reducer` (the UMAP
    object, or None if `n_components=0`) and `clusterer` (the HDBSCAN
    object, fit with `prediction_data=True` so it supports
    `hdbscan.approximate_predict()`) -- Task 8 (transport validation) needs
    these to evaluate new data in the same space via `reducer.transform()`
    + `approximate_predict()` (without refitting). `prediction_data=True`
    only builds extra auxiliary structures for prediction after fitting
    (a KD/Ball tree); it does not change `labels_` or the clustering result
    itself.
    """
    if n_components == 0:  # cluster directly in the raw space (no projection)
        space = emb
        reducer = None
    else:
        reducer = umap.UMAP(
            n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
            metric=metric, random_state=seed, n_jobs=1,
        )
        space = reducer.fit_transform(emb)
    cl = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size, min_samples=min_samples,
        metric="euclidean", gen_min_span_tree=True,
        cluster_selection_method=cluster_selection_method,
        # core_dist_n_jobs: hdbscan's own default is 4 (joblib-based), which
        # is NOT covered by the OMP/NUMBA/MKL/OPENBLAS pinning above and
        # would otherwise silently reintroduce a parallel/thread-count-
        # dependent code path. Pinned to 1 for explicitness. Empirically
        # tested and found NOT to be a source of nondeterminism by itself:
        # config index 5 run with core_dist_n_jobs=1 vs 4 under otherwise
        # identical pinned-thread conditions produced bit-identical labels
        # (agreement 100.0000%, ARI 1.000000, k=26 and noise 5.57% both
        # times). Pinned anyway as defensive/self-documenting practice, not
        # because it was found to fix anything.
        core_dist_n_jobs=1,
        prediction_data=return_models,
    ).fit(np.ascontiguousarray(space, dtype=np.float64))
    labels = cl.labels_
    result = {
        "labels": labels,
        "n_clusters": int(len(set(labels) - {-1})),
        "noise_frac": float((labels == -1).mean()),
        "dbcv": float(cl.relative_validity_),
        "space": space,
    }
    if return_models:
        result["reducer"] = reducer
        result["clusterer"] = cl
    return result


def _thread_config() -> dict:
    """Requested (env var) vs. empirically effective thread-pool settings.

    The env vars we set are only a request; recording them back verbatim
    (as this function used to do) is circular -- it proves an assignment
    executed, not that any library honored it. "effective" instead
    introspects what the loaded native libraries actually report at
    runtime: numba.get_num_threads() for numba, and
    threadpoolctl.threadpool_info() (ships with scikit-learn) for the
    BLAS/OpenMP libraries actually loaded into the process (numpy's,
    scikit-learn's, etc.), each with its own reported num_threads.
    """
    keys = ["OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]
    requested = {k: os.environ.get(k) for k in keys}

    effective: dict = {"numba_get_num_threads": numba.get_num_threads()}
    if threadpoolctl is None:
        effective["threadpoolctl"] = "unavailable (import failed)"
    else:
        effective["threadpoolctl"] = threadpoolctl.threadpool_info()

    return {"requested_env": requested, "effective": effective}


def _library_versions() -> dict:
    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "umap-learn": umap.__version__,
        "hdbscan": importlib.metadata.version("hdbscan"),
    }


def _atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)


def _atomic_write_json(obj: dict, dest: Path) -> None:
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, dest)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--config-index", type=int, default=None,
        help="Config index into build_grid() (0-39; 39=PUBLISHED_CONFIG). Runs exactly one config.",
    )
    ap.add_argument(
        "--list-configs", action="store_true",
        help="Print the index -> config tag mapping and exit (for array-size validation).",
    )
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    grid = build_grid()

    if args.list_configs:
        print(f"n_configs={len(grid)}")
        for idx, cfg in enumerate(grid):
            print(f"{idx}\t{tag_for(cfg)}\t{cfg}")
        return

    if args.config_index is None:
        raise SystemExit("--config-index is required unless --list-configs is given")
    if not (0 <= args.config_index < len(grid)):
        raise SystemExit(
            f"--config-index {args.config_index} out of range [0, {len(grid) - 1}]"
        )

    cfg = grid[args.config_index]
    tag = tag_for(cfg)
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"[config {args.config_index}] {tag}: {cfg}", flush=True)

    # verify_reports=False: Task 1 already verified every report path exists;
    # re-stat 32,329 files inside every array task is pointless work.
    frame = frame_mod.load_frame(verify_reports=False)
    emb = frame_mod.load_embeddings(frame)

    # cfg["min_dist"]/["metric"]/["min_samples"]/["cluster_selection_method"]
    # are only present for PUBLISHED_CONFIG (config_index 39) -- the other 39
    # grid configs fall back to run_config()'s own defaults (min_dist=0.0,
    # cosine, min_samples=10, eom), identical to before this change.
    t0 = time.time()
    r = run_config(
        emb,
        n_components=cfg["n_components"],
        n_neighbors=cfg["n_neighbors"],
        min_cluster_size=cfg["min_cluster_size"],
        seed=args.seed,
        min_dist=cfg.get("min_dist", 0.0),
        metric=cfg.get("metric", "cosine"),
        min_samples=cfg.get("min_samples", 10),
        cluster_selection_method=cfg.get("cluster_selection_method", "eom"),
    )
    wall_time_sec = time.time() - t0

    # No partial/fake outputs: any exception above aborts before either file
    # below is written (and each write itself is atomic via tmp + os.replace).
    _atomic_write_parquet(
        pd.DataFrame({"admission_ID": frame["admission_ID"], "label": r["labels"]}),
        OUT / f"labels_{tag}.parquet",
    )

    summary = {
        "config": tag,
        "config_index": args.config_index,
        "is_published_config": bool(cfg.get("published", False)),
        "n_components": cfg["n_components"],
        "n_neighbors": cfg["n_neighbors"],
        "min_cluster_size": cfg["min_cluster_size"],
        "min_dist": cfg.get("min_dist", 0.0),
        "metric": cfg.get("metric", "cosine"),
        "min_samples": cfg.get("min_samples", 10),
        "cluster_selection_method": cfg.get("cluster_selection_method", "eom"),
        "seed": args.seed,
        "n_rows": int(len(frame)),
        "n_clusters": r["n_clusters"],
        "noise_frac": round(r["noise_frac"], 6),
        "dbcv": round(r["dbcv"], 6),
        "wall_time_sec": round(wall_time_sec, 2),
        "libraries": _library_versions(),
        "threads": _thread_config(),
    }
    _atomic_write_json(summary, OUT / f"summary_{tag}.json")

    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
