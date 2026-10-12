#!/usr/bin/env python3
"""Cross-validate HDBSCAN's collapse using partition-based (k-means) and
distribution-based (GMM) clustering.

Task 2 diagnosis: HDBSCAN in the raw space (3,072 dimensions) collapsed
(mcs=50 -> 2 clusters, mcs=100/200 -> 42-47% noise). Task 2's results alone
cannot tell whether this is because (a) the embedding itself has weak
robust cluster structure, or (b) HDBSCAN (density-based) is hitting a
general limitation in high dimensions (density-estimate sparsification from
the curse of dimensionality). This script applies k-means and GMM to the
same embedding and the same dimensions (the raw 3,072D space and UMAP
10/20/50D -- 2D is excluded because Task 2 already showed a tendency toward
fragmentation there) to contrast what two algorithm families that do not
share the density-sparsification failure mode find in the same space.

On reusing UMAP coordinates: `out/02_recluster/` only stores
labels_*.parquet / summary_*.json -- the raw UMAP coordinates were not
saved (confirmed directly). Therefore the 10/20/50-dimensional spaces are
refit here with the same UMAP parameters that 02_recluster.py's
`run_config()` uses (min_dist=0.0, metric="cosine", random_state=seed,
n_jobs=1). `run_config()` itself is not reused, because that function is a
single unit that runs HDBSCAN right after the UMAP fit, which would pay the
HDBSCAN fitting cost every time (unneeded for this task) -- a single
HDBSCAN fit in the raw space takes ~3,100s (see grid_summary.csv). Instead
`get_space()` duplicates just that one UMAP call line, with the same
parameters. n_neighbors is fixed at 30, the median of Task 2's grid
{15,30,50} -- since dbcv in grid_summary.csv does not move consistently
with n_neighbors (nc10's best is nn15, nc20/50's best is nn30), this is a
single compromise value chosen to represent the grid as a whole, not a case
where all three were actually run.

On the GMM covariance type: `covariance_type="full"` is computationally
intractable in the raw space (D=3,072) -- a full covariance matrix requires
a D x D matrix per component, with per-iteration cost O(n * k * D^2), which
comes out to ~10^13 operations for a single E-step at raw3072 x k=32 alone.
This script therefore uses `covariance_type="diag"` across all four spaces
(applied identically to the UMAP spaces too, for cross-space
comparability). This introduces the constraint that GMM can only represent
axis-aligned cluster shapes (per-axis variance differs, but
cross-component covariance is zero), so BIC/silhouette values must be read
as "the best *computationally tractable* GMM", not "the best possible GMM"
-- this is stated in DIAGNOSIS.md.

On GMM `reg_covar` (fix for a defect actually reproduced in production): with
sklearn's default `reg_covar=1e-6`, the real SLURM job (586187) died at
umap50 space, k=16, with `ValueError: ... ill-defined empirical covariance
(singleton or collapsed samples) ...` -- because spaces reduced by UMAP to
50 dimensions can have regions packed tightly enough that a component
assigned there has a variance estimate that goes to (near) zero and becomes
singular. To prevent this, `fit_gmm` explicitly sets `reg_covar=1e-3` -- a
practical value commonly used in this situation, which forces a minimum
variance floor onto each axis of the diagonal covariance to prevent
singularity. This carries a real tradeoff: a larger `reg_covar` can flatten
out even "genuinely small" variance into that floor, risking underestimating
the true variance structure of tightly-packed (dense) clusters in
particular -- since this script's entire purpose is diagnosing cluster
structure, this is not a minor implementation detail but a caveat that
directly affects interpretation of the results. The same point is stated in
DIAGNOSIS.md's caveats section. Also, if `ValueError` (covariance/convergence
related) still occurs even with `reg_covar=1e-3`, `fit_gmm` catches it and
returns a record explicitly carrying `failed=True` and the error message --
the fact that GMM cannot be fit for that (space, k) combination is itself a
genuinely reportable result (e.g., that clusters are very tightly packed at
that granularity), not something to hide or let abort the whole run.

Outcome (mortality) variables are not loaded (Global Constraints):
30d_mortality, 90d_mortality, in_hospital_mortality, death_date,
discharge_date are not referenced on any code path in this task.
"""
import os

# Force single-threaded BLAS/OpenMP/numba execution BEFORE numpy/sklearn/umap
# are imported (identical pattern to 02_recluster.py -- see that module's
# top-of-file comment for the full empirical justification: 1/8/24 threads
# gave 26/18/33 HDBSCAN clusters for the same config/seed on the real data).
# scikit-learn's KMeans and GaussianMixture also route their heavy linear
# algebra (distance computations, E-step matrix products) through BLAS, so
# the same thread-count-dependent floating-point accumulation order risk
# applies to them, not just to HDBSCAN/UMAP. This MUST be set before any
# scientific import below -- these libraries read the env vars at import
# time, so setting them later (even at the top of main()) is a silent no-op.
os.environ["OMP_NUM_THREADS"] = os.environ["NUMBA_NUM_THREADS"] = \
    os.environ["MKL_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import importlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

sys.path.insert(0, os.path.dirname(__file__))
frame_mod = importlib.import_module("01_build_analysis_frame")
recl = importlib.import_module("02_recluster")  # _thread_config, _library_versions, atomic writers

OUT = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision/out/02b_partition")

# n_components=None means "cluster the raw embedding directly, no UMAP" --
# same convention 02_recluster.py uses (there: n_components=0).
SPACE_N_COMPONENTS = {"raw3072": None, "umap10": 10, "umap20": 20, "umap50": 50}
SPACE_TAGS = list(SPACE_N_COMPONENTS)

UMAP_N_NEIGHBORS = 30  # see module docstring: middle of Task 2's {15,30,50} grid
FIXED_K = [8, 16, 32]  # 16 = published paper's cluster count, for direct comparison
K_SCAN_GRID = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]  # candidate range for data-driven k
SCAN_N_INIT = 3  # cheap proxy restarts for the k-scan only, not the final fit
N_INIT = 10  # explicit n_init for the final kmeans/gmm fits (avoid any implicit default)
GMM_COVARIANCE_TYPE = "diag"  # see module docstring: "full" is intractable at D=3072
# sklearn default (1e-6) is insufficient: job 586187 crashed in fit_gmm at
# umap50/k=16 with "ill-defined empirical covariance (singleton or collapsed
# samples)". 1e-3 is a common practical floor for diagonal-covariance GMM in
# tightly-packed/low-variance embedding spaces. Tradeoff (see module
# docstring and DIAGNOSIS.md caveats): a larger reg_covar can mask genuinely
# small per-axis variance, which matters here because this script's entire
# purpose is measuring cluster structure.
GMM_REG_COVAR = 1e-3
SILHOUETTE_SAMPLE_SIZE = 5000  # O(n^2) on the full 32,329 rows is prohibitive


def get_space(emb: np.ndarray, space_tag: str, seed: int = 42) -> np.ndarray:
    """Return the raw space (raw3072) or a UMAP-reduced space (umap10/20/50).

    The UMAP parameters match 02_recluster.run_config()'s UMAP call exactly
    (min_dist=0.0, metric="cosine", random_state=seed, n_jobs=1); refit here
    because the coordinates were not saved in Task 2 (see module docstring).
    """
    nc = SPACE_N_COMPONENTS[space_tag]
    if nc is None:
        return np.ascontiguousarray(emb, dtype=np.float64)
    return umap.UMAP(
        n_components=nc, n_neighbors=UMAP_N_NEIGHBORS, min_dist=0.0,
        metric="cosine", random_state=seed, n_jobs=1,
    ).fit_transform(emb)


def fit_kmeans(space: np.ndarray, k: int, seed: int = 42) -> dict:
    """Fit k-means and return labels/inertia/(subsampled) silhouette."""
    km = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(space)
    labels = km.labels_
    sil = silhouette_score(
        space, labels, sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=seed
    )
    return {"labels": labels, "inertia": float(km.inertia_), "silhouette": float(sil)}


def fit_gmm(space: np.ndarray, k: int, seed: int = 42) -> dict:
    """Fit a GMM (diagonal covariance) and return labels/BIC/(subsampled)
    silhouette.

    Explicitly sets `reg_covar=GMM_REG_COVAR` (see module docstring: the
    sklearn default 1e-6 actually caused a singular-covariance crash at
    umap50/k=16). If a covariance/convergence-related ValueError still
    occurs, this catches it and returns a record carrying `failed=True`
    and the error message instead of raising the exception -- the fact
    that GMM cannot be fit for that (space, k) combination is itself a
    result worth reporting, not a reason to abort the whole run.
    """
    try:
        gmm = GaussianMixture(
            n_components=k, covariance_type=GMM_COVARIANCE_TYPE,
            n_init=N_INIT, random_state=seed, reg_covar=GMM_REG_COVAR,
        ).fit(space)
    except ValueError as exc:
        return {
            "labels": None, "bic": None, "silhouette": None,
            "failed": True, "error": str(exc),
        }
    labels = gmm.predict(space)
    sil = silhouette_score(
        space, labels, sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=seed
    )
    return {
        "labels": labels, "bic": float(gmm.bic(space)), "silhouette": float(sil),
        "failed": False, "error": None,
    }


def select_k_by_silhouette(
    space: np.ndarray, k_grid: list[int] = K_SCAN_GRID, seed: int = 42
) -> dict:
    """Run a cheap k-means scan (n_init=SCAN_N_INIT) over the whole k_grid
    and pick the k with the highest silhouette. The k that is ultimately
    adopted is then refit formally with n_init=N_INIT via
    fit_kmeans/fit_gmm (the labels/model from this scan itself are
    discarded -- the standard elbow/silhouette practice of separating a
    cheap search from the formal fit).
    """
    rows = []
    for k in k_grid:
        km = KMeans(n_clusters=k, n_init=SCAN_N_INIT, random_state=seed).fit(space)
        sil = silhouette_score(
            space, km.labels_, sample_size=SILHOUETTE_SAMPLE_SIZE, random_state=seed
        )
        rows.append({"k": k, "silhouette": float(sil), "inertia": float(km.inertia_)})
    best = max(rows, key=lambda r: r["silhouette"])
    return {"scan": rows, "best_k": best["k"], "best_silhouette": best["silhouette"]}


def output_tag(algo: str, space_tag: str, k: int) -> str:
    return f"{algo}_{space_tag}_k{k}"


def _merge_and_write_csv(new_rows: pd.DataFrame, dest: Path, key_cols: list[str]) -> pd.DataFrame:
    """If dest already exists, replace existing rows whose key_cols match
    with the new rows, merge, and write atomically (so that re-running just
    a single combo still preserves the whole grid's CSV -- unlike
    02_recluster's array-per-config pattern, this script runs the whole
    grid sequentially in one process so there is no concurrent-write
    conflict, but this preserves existing results on a partial rerun /
    re-validation).
    """
    if dest.exists():
        old = pd.read_csv(dest)
        keep = old.merge(
            new_rows[key_cols].drop_duplicates(), on=key_cols, how="left", indicator=True
        )["_merge"].eq("left_only")
        combined = pd.concat([old[keep.values], new_rows], ignore_index=True)
    else:
        combined = new_rows
    combined = combined.sort_values(key_cols).reset_index(drop=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    combined.to_csv(tmp, index=False)
    os.replace(tmp, dest)
    return combined


def _write_diagnosis_md(summary_df: pd.DataFrame, dest: Path) -> None:
    """Re-render DIAGNOSIS.md from scratch each time, from the full
    summary_df (accumulated on disk so far), and write it atomically.
    Since `main()` calls this after every space, even if the process dies
    partway through, the latest version reflecting every already-finished
    space's results always remains on disk (the same partial-run
    durability guarantee as elbow_silhouette.csv/partition_summary.csv).
    """
    lines = [
        "# Task 2b: partition/distribution clustering cross-check",
        "",
        "Cross-checks HDBSCAN's raw-embedding-space collapse (Task 2) against",
        "k-means and GMM on the same embedding spaces (raw3072, umap10/20/50).",
        "",
        "## Caveats",
        "",
        "1. **GMM covariance_type=\"diag\".** `covariance_type=\"full\"` is",
        "   computationally intractable at raw3072 (D=3,072): a full covariance",
        "   matrix is D x D per component, making the E-step O(n * k * D^2) --",
        "   ~10^13 operations for a single E-step at k=32. All four spaces use",
        "   `covariance_type=\"diag\"` for cross-space comparability, which can",
        "   only represent axis-aligned cluster shapes (per-axis variance, zero",
        "   cross-component covariance). BIC/silhouette values here should be",
        "   read as \"the best *computationally tractable* GMM\", not \"the best",
        "   possible GMM\".",
        "2. **GMM reg_covar=1e-3 (not the sklearn default 1e-6).** The sklearn",
        "   default was insufficient in production: SLURM job 586187 crashed",
        "   inside `GaussianMixture.fit()` at umap50/k=16 with \"ill-defined",
        "   empirical covariance (singleton or collapsed samples)\", because",
        "   UMAP-reduced spaces can pack points tightly enough that a component's",
        "   diagonal covariance estimate becomes singular. `reg_covar=1e-3` adds",
        "   a variance floor per axis to prevent that. Tradeoff: a larger",
        "   `reg_covar` can mask genuinely small per-axis variance, which",
        "   matters directly here because this script's entire purpose is",
        "   measuring cluster structure -- tightly-packed clusters may look",
        "   artificially more diffuse than they really are.",
        "3. **A (space, k) combination can still fail.** If `reg_covar=1e-3` is",
        "   not enough, `fit_gmm` records `failed=True` plus the sklearn error",
        "   message instead of crashing the run. A genuine GMM-fit failure at a",
        "   given granularity is itself a reportable finding (very tight",
        "   clusters at that k), not something to hide or silently skip.",
        "",
    ]

    failed = summary_df[summary_df.get("failed", False) == True] if len(summary_df) else summary_df  # noqa: E712
    if len(failed):
        lines += ["## Failed fits", ""]
        for _, r in failed.iterrows():
            lines.append(f"- `{r['algorithm']}_{r['space']}_k{r['k']}`: {r.get('error')}")
        lines.append("")

    lines += ["## Results so far", ""]
    if len(summary_df):
        cols = ["space", "algorithm", "k", "k_source", "silhouette", "fit_quality_metric", "fit_quality_value", "failed"]
        cols = [c for c in cols if c in summary_df.columns]
        table = summary_df[cols].sort_values(["space", "algorithm", "k"])
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in table.iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    else:
        lines.append("(no fits completed yet)")
    lines.append("")

    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text("\n".join(lines))
    os.replace(tmp, dest)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--spaces", default=",".join(SPACE_TAGS),
        help=f"Comma-separated list of space tags (for running a subset). choices from {SPACE_TAGS}",
    )
    ap.add_argument(
        "--algos", default="kmeans,gmm",
        help="Comma-separated list of algorithms: a subset of kmeans,gmm",
    )
    ap.add_argument(
        "--skip-k-scan", action="store_true",
        help="Skip the data-driven k scan and use only FIXED_K (for quick single-combo validation)",
    )
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    spaces = args.spaces.split(",")
    algos = args.algos.split(",")
    for s in spaces:
        if s not in SPACE_N_COMPONENTS:
            raise SystemExit(f"unknown space tag {s!r}; choices are {SPACE_TAGS}")
    for a in algos:
        if a not in ("kmeans", "gmm"):
            raise SystemExit(f"unknown algo {a!r}; choices are kmeans,gmm")

    OUT.mkdir(parents=True, exist_ok=True)

    # verify_reports=False: Task 1 already verified every report path exists;
    # re-stat 32,329 files on every run is wasted I/O.
    frame = frame_mod.load_frame(verify_reports=False)
    emb = frame_mod.load_embeddings(frame)
    admission_id = frame["admission_ID"]

    total_detail_rows, total_summary_rows = 0, 0

    for space_tag in spaces:
        detail_rows, summary_rows = [], []
        print(f"[space] {space_tag}: building space", flush=True)
        t0 = time.time()
        space = get_space(emb, space_tag, seed=args.seed)
        space_build_sec = time.time() - t0
        print(f"[space] {space_tag}: built in {space_build_sec:.1f}s, shape={space.shape}", flush=True)

        k_list = list(FIXED_K)
        scan_info = None
        if not args.skip_k_scan:
            t0 = time.time()
            scan_info = select_k_by_silhouette(space, seed=args.seed)
            scan_sec = time.time() - t0
            print(
                f"[space] {space_tag}: k-scan best_k={scan_info['best_k']} "
                f"silhouette={scan_info['best_silhouette']:.4f} ({scan_sec:.1f}s)",
                flush=True,
            )
            if scan_info["best_k"] not in k_list:
                k_list.append(scan_info["best_k"])

        for algo, k in itertools.product(algos, k_list):
            tag = output_tag(algo, space_tag, k)
            print(f"[fit] {tag}", flush=True)
            t0 = time.time()
            failed, error = False, None
            if algo == "kmeans":
                r = fit_kmeans(space, k, seed=args.seed)
                quality_col, quality_val = "inertia", r["inertia"]
            else:
                r = fit_gmm(space, k, seed=args.seed)
                failed, error = r["failed"], r["error"]
                quality_col, quality_val = "bic", r.get("bic")
            wall_time_sec = time.time() - t0

            if failed:
                # GMM could not fit this (space, k) even with reg_covar --
                # record it explicitly and move on (Fix 1); no parquet to write.
                print(f"[fit] {tag}: FAILED -- {error}", flush=True)
            else:
                dest = OUT / f"{tag}.parquet"
                tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
                pd.DataFrame({"admission_ID": admission_id, "label": r["labels"]}).to_parquet(
                    tmp, index=False
                )
                os.replace(tmp, dest)

            row = {
                "space": space_tag,
                "algorithm": algo,
                "k": k,
                "k_source": "data_driven" if (scan_info and k == scan_info["best_k"] and k not in FIXED_K) else "fixed",
                "inertia": r.get("inertia"),
                "bic": r.get("bic"),
                "silhouette": r.get("silhouette"),
                "silhouette_sample_size": SILHOUETTE_SAMPLE_SIZE,
                "n_init": N_INIT,
                "gmm_covariance_type": GMM_COVARIANCE_TYPE if algo == "gmm" else None,
                "gmm_reg_covar": GMM_REG_COVAR if algo == "gmm" else None,
                "seed": args.seed,
                "wall_time_sec": round(wall_time_sec, 2),
                "space_build_sec": round(space_build_sec, 2),
                "n_rows": int(len(frame)),
                "failed": failed,
                "error": error,
                "libraries": recl._library_versions(),
                "threads": recl._thread_config(),
            }
            detail_rows.append(row)
            summary_rows.append({
                "space": space_tag,
                "algorithm": algo,
                "k": k,
                "k_source": row["k_source"],
                "silhouette": row["silhouette"],
                "fit_quality_metric": quality_col,
                "fit_quality_value": quality_val,
                "failed": failed,
                "error": error,
            })
            print(json.dumps({k_: v for k_, v in row.items() if k_ not in ("libraries", "threads")}), flush=True)

        if scan_info is not None:
            scan_df = pd.DataFrame(scan_info["scan"])
            scan_df["space"] = space_tag
            scan_dest = OUT / f"k_scan_{space_tag}.csv"
            tmp = scan_dest.with_name(f".tmp_{os.getpid()}_{scan_dest.name}")
            scan_df.to_csv(tmp, index=False)
            os.replace(tmp, scan_dest)

        # Incremental persistence (Fix 2): merge-and-write after EVERY space,
        # not only once at the very end of main(). If the process crashes or
        # is killed partway through a later space, every space that already
        # finished is durably on disk here, not only in per-fit parquet files
        # and stdout JSON lines (see job 586187's crash on umap50, which lost
        # umap10/umap20's already-completed results from these two CSVs).
        detail_df = pd.DataFrame(detail_rows)
        # libraries/threads are dicts -- stringify for CSV, matching grid_summary.csv's convention
        for col in ("libraries", "threads"):
            detail_df[col] = detail_df[col].map(str)
        _merge_and_write_csv(detail_df, OUT / "elbow_silhouette.csv", ["space", "algorithm", "k"])

        summary_df = pd.DataFrame(summary_rows)
        merged_summary = _merge_and_write_csv(
            summary_df, OUT / "partition_summary.csv", ["space", "algorithm", "k"]
        )
        _write_diagnosis_md(merged_summary, OUT / "DIAGNOSIS.md")

        total_detail_rows += len(detail_rows)
        total_summary_rows += len(summary_rows)
        print(
            f"[space] {space_tag}: wrote {len(detail_rows)} rows -> elbow_silhouette.csv, "
            f"partition_summary.csv, DIAGNOSIS.md updated",
            flush=True,
        )

    print(f"wrote {total_detail_rows} rows -> {OUT / 'elbow_silhouette.csv'}", flush=True)
    print(f"wrote {total_summary_rows} rows -> {OUT / 'partition_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
