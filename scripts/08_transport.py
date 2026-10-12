#!/usr/bin/env python3
"""Task 8: transport validation -- across hospitals and across calendar periods.

Responds to (prior review comments): a stability-across-time point ("examine
phenotype stability across calendar periods and across the two hospitals ...
report whether major phenotype families are preserved") and a
generalizability point. The originally derived paper performed no such check.

## Core methodology: the definition of "transport"

The scientific question is not "can each subset be clustered independently"
but "does a phenotype found in one subset (source) reappear in another
subset (target)". To do this:

1. `02_recluster.run_config(..., return_models=True)` fits UMAP+HDBSCAN
   **only on the source stratum** (yielding a fitted `reducer`/`clusterer`).
2. **The target is never refit.** Instead, `reducer.transform()` projects
   the target into that SAME learned space, and
   `hdbscan.approximate_predict()` assigns the target to that learned
   clusterer.

Refitting the target would produce a different UMAP space, making cluster-ID
comparison itself meaningless -- this is the key pitfall the brief
specifies.

## Directions (4, both ways per axis)

Hospital: 10 (n=24,660) <-> 20 (n=7,669). Period: early (2010-2017,
registered-count n=15,682) <-> late (2018-2024, n=16,647), cutoff 2017/2018.
Reporting only one direction would hide an asymmetry where the larger
group's phenotypes transport into the smaller group but not vice versa.

## Link to the stability gate (reuses Task 3, no recomputation)

Task 3 found that, in the whole cohort (n=32,329, config nc20_nn30_mcs100),
only 19 of 40 clusters survived bootstrap resampling (Jaccard>=0.75). This
task's source clustering is newly fit on the source-stratum subset (not the
whole cohort), so its cluster IDs differ from Task 3's. To link the two
clusterings, each source cluster is mapped to the "whole-cohort cluster with
the largest overlap when the whole-cohort clustering is restricted to this
source subset's rows" (max Jaccard, `best_whole_cohort_cluster`), and that
whole-cohort cluster's stability (from Task 3's
`out/03_stability/per_cluster_jaccard.csv`) is inherited, splitting the
transport results into "stable vs. unstable source clusters" reported
separately. An unstable cluster failing to transport is not informative.

## Organism-composition comparison

Using `BSI_episode_short_clean` (the 20-category organism encoding
established by Task 5a), composition-proportion vectors (using the full set
of categories observed across the whole cohort in a fixed order, so they are
comparable across directions) are compared by cosine similarity between
"the source cluster vs. its transported target counterpart (target rows
assigned the same label number)".

## Reference clustering -- never to be confused with the transport result

For each direction, an **independently** refit UMAP+HDBSCAN on the target is
also reported (n_clusters, noise_frac) -- this is not a transport result but
only a reference point for "what does the target look like on its own",
and is clearly labeled "reference" in FINDINGS.md and never mixed with the
transport results.

## Thread pinning (same reason as 02_recluster.py/03_stability.py)

It has already been confirmed that UMAP+HDBSCAN results themselves vary with
thread count (1/8/24 threads -> 26/18/33 clusters, same seed/config,
measured). The 4 environment variables below must be pinned before
importing any scientific computing library.

## Outcome (mortality) variable not used (Global Constraints)

This task is a structural question, not an outcome question. `30d_mortality`
/`90d_mortality`/`in_hospital_mortality`/`death_date`/`discharge_date` are
not loaded/referenced anywhere in this script's code paths.

## Task 12 leakage-channel warning (cannot be removed, only disclosed)

The Task 12 audit found that `treatment_department_eng` labels differ by
hospital (hospital 10: 51 labels, hospital 20: 38 labels, 34 shared), and
that report narratives contain the literal string `Admission year`. Both
channels are already baked into the fixed (original) OpenAI embeddings this
task uses and cannot be removed here -- the FINDINGS.md below discloses that
what looks like a hospital/period split may be due to these tokens rather
than clinical differences. Task 6's `no_department`/`no_year` ablation
variants are the pending evidence that would directly test this (not yet
generated at the time this task ran), and this script cross-references them
without speculating about their results.
"""
import os

os.environ["OMP_NUM_THREADS"] = os.environ["NUMBA_NUM_THREADS"] = \
    os.environ["MKL_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
frame_mod = importlib.import_module("01_build_analysis_frame")
recl = importlib.import_module("02_recluster")  # run_config(return_models=True), atomic writers, _thread_config, _library_versions

ROOT = Path("/data/tcmhwd/AI/AST_ai")
DATASET = ROOT / "manuscript/phenotype_ml_dataset.csv"
OUT = ROOT / "phenotype_revision/out/08_transport"

# Task 3's completed stability artifacts (read as-is, not recomputed).
WHOLE_COHORT_LABELS_PATH = ROOT / "phenotype_revision/out/02_recluster/labels_nc20_nn30_mcs100.parquet"
WHOLE_COHORT_JACCARD_PATH = ROOT / "phenotype_revision/out/03_stability/per_cluster_jaccard.csv"

SEED = 42
ORGANISM_COL = "BSI_episode_short_clean"

# Same as the primary configuration adopted by Task 3 -- makes it directly
# comparable with the whole-cohort clustering.
N_COMPONENTS = 20
N_NEIGHBORS = 30
MIN_CLUSTER_SIZE = 100

STABILITY_THRESHOLD = 0.75  # same criterion as Task 3

# [Added 2026-09-04, Task 10 expansion Part D] minimum sample threshold,
# --organism-filter only. A stratum pre-filtered to a single organism (e.g.
# E. coli x hospital 20 = 1,868 cases) is much smaller than the whole-cohort
# strata (tens of thousands of cases) -- forcing MIN_CLUSTER_SIZE=100 as-is
# would make it a relatively much more suppressive threshold exceeding 5% of
# n_subset (the same pitfall as Task 10 Part A). When --organism-filter is
# on, this script re-derives min_cluster_size for each direction's SOURCE
# stratum size using the same rule as
# `10_within_pathogen.scaled_min_cluster_sizes` (n_subset // 200, floor=50)
# (explicitly overridable via --min-cluster-size). A stratum smaller than
# this threshold is not forced through clustering and is explicitly skipped
# (brief requirement).
MIN_VIABLE_STRATUM_N = 500

# Period-split cutoff: admit_year <= 2017 -> "early", >= 2018 -> "late".
# Measured counts (01_build_analysis_frame.load_frame()): early=15,682,
# late=16,647 (a reasonably balanced split, near the midpoint of the
# 2010-2024 span).
EARLY_YEAR_CUTOFF = 2017

DIRECTIONS = [
    {"name": "hospital_10_to_20", "axis": "hospital", "source": 10, "target": 20},
    {"name": "hospital_20_to_10", "axis": "hospital", "source": 20, "target": 10},
    {"name": "period_early_to_late", "axis": "period", "source": "early", "target": "late"},
    {"name": "period_late_to_early", "axis": "period", "source": "late", "target": "early"},
]
DIRECTION_NAMES = [d["name"] for d in DIRECTIONS]


# =========================================================================
# Data loading
# =========================================================================

def load_organism_frame(dataset_path: Path = DATASET, verify_reports: bool = False) -> pd.DataFrame:
    """Joins only `BSI_episode_short_clean` onto Task 1's outcome-blind frame
    (the same kind of exception as Task 10's `load_organism_frame` -- this
    join logic is not moved to a shared module. Task 8 doesn't need the
    outcome/treatment columns, so it's much lighter than Task 10).
    """
    frame = frame_mod.load_frame(verify_reports=verify_reports)
    extra = pd.read_csv(dataset_path, usecols=["admission_ID", ORGANISM_COL], low_memory=False)
    merged = frame.merge(extra, on="admission_ID", how="left", validate="one_to_one")
    assert len(merged) == len(frame), "join changed row count -- dataset_path has duplicate admission_ID"
    assert merged[ORGANISM_COL].isna().sum() == 0, "unexpected missingness in organism column"
    return merged


def load_whole_cohort_stability(
    labels_path: Path = WHOLE_COHORT_LABELS_PATH,
    jaccard_path: Path = WHOLE_COHORT_JACCARD_PATH,
    threshold: float = STABILITY_THRESHOLD,
):
    """Reads Task 3's completed artifacts (labels_nc20_nn30_mcs100.parquet,
    per_cluster_jaccard.csv) and returns (whole-cohort label df, set of
    stable cluster IDs). Does not recompute -- fails explicitly if absent."""
    if not labels_path.exists() or not jaccard_path.exists():
        raise FileNotFoundError(
            f"Task 8 requires Task 3's completed whole-cohort run: "
            f"{labels_path} and {jaccard_path} must both exist first."
        )
    labels_df = pd.read_parquet(labels_path)  # admission_ID, label
    jac_df = pd.read_csv(jaccard_path)        # cluster, jaccard_mean, jaccard_p05
    stable_clusters = set(jac_df.loc[jac_df["jaccard_mean"] >= threshold, "cluster"].astype(int))
    return labels_df, stable_clusters


# =========================================================================
# Subset selection (hospital / period axes)
# =========================================================================

def _stratum_mask(frame: pd.DataFrame, axis: str, stratum) -> pd.Series:
    """Just the mask computation split out of `select_stratum` -- Part D
    (main()) needs to check/report the stratum size before actually
    clustering, and this avoids the cost of copying/reset_index-ing a new
    DataFrame via `select_stratum` each time."""
    if axis == "hospital":
        return frame["hospital_code"] == stratum
    elif axis == "period":
        if stratum == "early":
            return frame["admit_year"] <= EARLY_YEAR_CUTOFF
        elif stratum == "late":
            return frame["admit_year"] > EARLY_YEAR_CUTOFF
        else:
            raise ValueError(f"unknown period stratum: {stratum!r}")
    else:
        raise ValueError(f"unknown axis: {axis!r}")


def select_stratum(frame: pd.DataFrame, axis: str, stratum) -> pd.DataFrame:
    """Selects rows according to `axis` ("hospital"|"period") and `stratum`
    value, and resets the index. The `embedding_index` column is preserved,
    so the return value can be passed directly to `load_embeddings`."""
    return frame.loc[_stratum_mask(frame, axis, stratum)].reset_index(drop=True)


# =========================================================================
# Clustering (source fit) + transport (target transform+predict, not refit)
# =========================================================================

def fit_source_clustering(
    emb: np.ndarray, min_cluster_size: int = MIN_CLUSTER_SIZE, seed: int = SEED,
    n_components: int = N_COMPONENTS, n_neighbors: int = N_NEIGHBORS,
) -> dict:
    """A thin wrapper around `02_recluster.run_config`, called with
    `return_models=True` to also get back the fitted reducer/clusterer
    needed for transport. Does not create new clustering logic.
    `n_components`/`n_neighbors` are overridden only in tests to fit small
    synthetic data; the real run (main) always uses the module constants
    (20/30)."""
    return recl.run_config(
        emb, n_components=n_components, n_neighbors=n_neighbors,
        min_cluster_size=min_cluster_size, seed=seed, return_models=True,
    )


def transport_predict(fit: dict, target_emb: np.ndarray):
    """Projects `target_emb` into the SAME space using `fit`'s
    (fit_source_clustering's return value) fitted UMAP `reducer`
    (`reducer.transform`, not a refit), and assigns labels/confidence
    (strength) via the fitted HDBSCAN `clusterer`'s
    `hdbscan.approximate_predict`. Does not refit UMAP/HDBSCAN on the
    target -- refitting would produce a different space, making cluster-ID
    comparison itself meaningless (the brief's key pitfall)."""
    reducer = fit.get("reducer")
    clusterer = fit.get("clusterer")
    if reducer is None or clusterer is None:
        raise ValueError(
            "transport_predict requires fit_source_clustering(..., n_components>0) "
            "(a fitted UMAP reducer + prediction_data=True HDBSCAN clusterer)"
        )
    target_space = reducer.transform(target_emb)
    target_space = np.ascontiguousarray(target_space, dtype=np.float64)
    labels, strengths = hdbscan.approximate_predict(clusterer, target_space)
    return labels, strengths, target_space


def unassigned_fraction(labels) -> float:
    labels = np.asarray(labels)
    return float((labels == -1).mean())


# =========================================================================
# Mapping to whole-cohort clusters (inheriting Task 3 stability)
# =========================================================================

def best_whole_cohort_cluster(source_labels: np.ndarray, whole_labels: np.ndarray, k: int):
    """Returns the cluster ID and Jaccard coefficient of the whole_labels
    cluster (the whole-cohort clustering restricted to this source subset's
    rows) that overlaps the most with cluster `k` (by positional index) of
    source_labels. Uses the same intersection/union computation as
    `03_stability.jaccard_for_cluster`, but that function only returns the
    max value (score), whereas this function must also return which
    cluster produced that max (argmax) -- so that Task 3's stability label
    can be inherited onto this cluster -- hence it's kept separate."""
    source_labels = np.asarray(source_labels)
    whole_labels = np.asarray(whole_labels)
    source_set = np.where(source_labels == k)[0]
    if len(source_set) == 0:
        return None, float("nan")
    best_j, best_score = None, 0.0
    for j in set(whole_labels.tolist()) - {-1}:
        whole_set = np.where(whole_labels == j)[0]
        inter = len(np.intersect1d(source_set, whole_set))
        union = len(np.union1d(source_set, whole_set))
        score = inter / union if union else 0.0
        if score > best_score:
            best_score, best_j = score, j
    return best_j, best_score


# =========================================================================
# Organism-composition comparison
# =========================================================================

def organism_proportions(series: pd.Series, categories: list) -> np.ndarray:
    """Builds a composition-proportion vector in the fixed order of
    `categories` (an empty series yields an all-zero vector)."""
    n = len(series)
    if n == 0:
        return np.zeros(len(categories), dtype=float)
    counts = series.value_counts()
    return np.array([counts.get(c, 0) for c in categories], dtype=float) / n


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


# =========================================================================
# Per-cluster transport metrics for a single direction
# =========================================================================

def build_metrics_table(
    source_frame: pd.DataFrame, source_labels: np.ndarray,
    target_frame: pd.DataFrame, target_labels: np.ndarray,
    whole_labels_df: pd.DataFrame, stable_clusters: set, categories: list,
    organism_col: str = ORGANISM_COL,
) -> pd.DataFrame:
    """Builds one row of transport metrics per source cluster (excluding
    noise -1): source/target size and fraction, whether it's empty in the
    target, the best-matching whole-cohort cluster/Jaccard/stability,
    organism-composition cosine similarity, and the size-concordance
    ratio."""
    n_source, n_target = len(source_labels), len(target_labels)
    source_labels = np.asarray(source_labels)
    target_labels = np.asarray(target_labels)

    whole_map = whole_labels_df.set_index("admission_ID")["label"]
    source_whole_labels = source_frame["admission_ID"].map(whole_map).to_numpy()
    assert not pd.isna(source_whole_labels).any(), (
        "some source admission_IDs are missing from Task 3's whole-cohort labels -- "
        "source stratum must be a subset of the same Task 1 frame Task 3 clustered"
    )
    source_whole_labels = source_whole_labels.astype(int)

    rows = []
    for k in sorted(set(source_labels.tolist()) - {-1}):
        s_mask = source_labels == k
        t_mask = target_labels == k
        source_n = int(s_mask.sum())
        target_n = int(t_mask.sum())

        best_j, best_jac = best_whole_cohort_cluster(source_labels, source_whole_labels, k)
        stable = bool(best_j in stable_clusters) if best_j is not None else False

        source_vec = organism_proportions(source_frame.loc[s_mask, organism_col], categories)
        if target_n > 0:
            target_vec = organism_proportions(target_frame.loc[t_mask, organism_col], categories)
            organism_cos_sim = cosine_similarity(source_vec, target_vec)
        else:
            organism_cos_sim = float("nan")

        source_frac = source_n / n_source if n_source else float("nan")
        target_frac = target_n / n_target if n_target else float("nan")
        size_ratio = (target_frac / source_frac) if source_frac else float("nan")

        rows.append({
            "source_cluster": int(k),
            "source_n": source_n,
            "source_frac": source_frac,
            "target_n": target_n,
            "target_frac": target_frac,
            "empty_in_target": target_n == 0,
            "whole_cohort_best_match": best_j,
            "whole_cohort_best_jaccard": best_jac,
            "whole_cohort_stable": stable,
            "organism_cosine_similarity": organism_cos_sim,
            "size_concordance_ratio": size_ratio,
        })
    return pd.DataFrame(rows)


def _nanmedian(values) -> float | None:
    s = pd.Series(values, dtype=float).dropna()
    return float(s.median()) if len(s) else None


# =========================================================================
# Atomic writes (Task 2b's lesson: writing only at the end loses data on timeout)
# =========================================================================

def _atomic_write_csv(df: pd.DataFrame, dest: Path) -> None:
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, dest)


def _atomic_write_text(text: str, dest: Path) -> None:
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(text)
    os.replace(tmp, dest)


# =========================================================================
# The full pipeline for a single direction
# =========================================================================

def process_direction(
    frame: pd.DataFrame, direction: dict, out_dir: Path,
    whole_labels_df: pd.DataFrame, stable_clusters: set, categories: list,
    seed: int = SEED, n_components: int = N_COMPONENTS, n_neighbors: int = N_NEIGHBORS,
    min_cluster_size: int = MIN_CLUSTER_SIZE, organism_filter: str | None = None,
) -> dict:
    name = direction["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    source_frame = select_stratum(frame, direction["axis"], direction["source"])
    target_frame = select_stratum(frame, direction["axis"], direction["target"])
    n_source, n_target = len(source_frame), len(target_frame)
    print(f"[{name}] n_source={n_source} n_target={n_target}", flush=True)

    source_emb = frame_mod.load_embeddings(source_frame)
    target_emb = frame_mod.load_embeddings(target_frame)

    t0 = time.time()
    fit = fit_source_clustering(
        source_emb, min_cluster_size=min_cluster_size, seed=seed,
        n_components=n_components, n_neighbors=n_neighbors,
    )
    fit_wall = time.time() - t0
    print(f"[{name}] source fit: n_clusters={fit['n_clusters']} "
          f"noise_frac={fit['noise_frac']:.4f} wall={fit_wall:.2f}s", flush=True)

    t1 = time.time()
    target_labels, target_strengths, _ = transport_predict(fit, target_emb)
    transport_wall = time.time() - t1
    target_unassigned = unassigned_fraction(target_labels)
    print(f"[{name}] transport: target_unassigned_frac={target_unassigned:.4f} "
          f"wall={transport_wall:.2f}s", flush=True)

    # Reference only: independently refit the target. Not a transport result.
    t2 = time.time()
    ref = recl.run_config(
        target_emb, n_components=n_components, n_neighbors=n_neighbors,
        min_cluster_size=min_cluster_size, seed=seed,
    )
    ref_wall = time.time() - t2
    print(f"[{name}] REFERENCE ONLY (independent target refit, not transport): "
          f"n_clusters={ref['n_clusters']} noise_frac={ref['noise_frac']:.4f} "
          f"wall={ref_wall:.2f}s", flush=True)

    metrics = build_metrics_table(
        source_frame, fit["labels"], target_frame, target_labels,
        whole_labels_df, stable_clusters, categories,
    )

    n_receiving = int((metrics["target_n"] > 0).sum())
    n_empty = int((metrics["target_n"] == 0).sum())
    is_stable = metrics["whole_cohort_stable"]
    n_stable = int(is_stable.sum())
    n_unstable = int((~is_stable).sum())
    n_stable_receiving = int((metrics.loc[is_stable, "target_n"] > 0).sum())
    n_unstable_receiving = int((metrics.loc[~is_stable, "target_n"] > 0).sum())

    summary = {
        "direction": name,
        "axis": direction["axis"],
        "source_stratum": direction["source"],
        "target_stratum": direction["target"],
        "n_source": n_source,
        "n_target": n_target,
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_cluster_size": min_cluster_size,
        "seed": seed,
        "source_n_clusters": fit["n_clusters"],
        "source_noise_frac": round(fit["noise_frac"], 6),
        "source_dbcv": round(fit["dbcv"], 6),
        "n_source_clusters_receiving_target_members": n_receiving,
        "n_source_clusters_empty_in_target": n_empty,
        "target_unassigned_frac": round(target_unassigned, 6),
        "reference_target_n_clusters": ref["n_clusters"],
        "reference_target_noise_frac": round(ref["noise_frac"], 6),
        "reference_target_dbcv": round(ref["dbcv"], 6),
        "n_stable_source_clusters": n_stable,
        "n_unstable_source_clusters": n_unstable,
        "n_stable_clusters_receiving_target": n_stable_receiving,
        "n_unstable_clusters_receiving_target": n_unstable_receiving,
        "median_organism_cosine_similarity_all": _nanmedian(metrics["organism_cosine_similarity"]),
        "median_organism_cosine_similarity_stable": _nanmedian(
            metrics.loc[is_stable, "organism_cosine_similarity"]),
        "median_organism_cosine_similarity_unstable": _nanmedian(
            metrics.loc[~is_stable, "organism_cosine_similarity"]),
        "fit_wall_time_sec": round(fit_wall, 2),
        "transport_predict_wall_time_sec": round(transport_wall, 2),
        "reference_fit_wall_time_sec": round(ref_wall, 2),
        "organism_filter": organism_filter,
        "libraries": recl._library_versions(),
        "threads": recl._thread_config(),
    }

    # Atomic per-direction output (a completed direction is preserved even on timeout).
    recl._atomic_write_parquet(
        pd.DataFrame({
            "admission_ID": target_frame["admission_ID"],
            "transported_label": target_labels,
            "transport_strength": target_strengths,
        }),
        out_dir / f"assignment_{name}.parquet",
    )
    _atomic_write_csv(metrics, out_dir / f"metrics_{name}.csv")
    recl._atomic_write_json(summary, out_dir / f"summary_{name}.json")
    _atomic_write_text(build_direction_fragment(summary, metrics), out_dir / f"FRAGMENT_{name}.md")

    return summary


def build_direction_fragment(summary: dict, metrics: pd.DataFrame) -> str:
    """Assembles the markdown only from actually-computed `summary`/`metrics`
    values -- not a fixed template that could describe only favorable
    results (not repeating the mistake Task 5c/Task 9 were flagged for)."""
    name = summary["direction"]
    lines = [
        f"## {name} (source={summary['source_stratum']} n={summary['n_source']} "
        f"-> target={summary['target_stratum']} n={summary['n_target']})",
        "",
        f"- Source clustering: n_clusters={summary['source_n_clusters']}, "
        f"noise_frac={summary['source_noise_frac']:.4f}, dbcv={summary['source_dbcv']:.4f}.",
        f"- Transport: {summary['n_source_clusters_receiving_target_members']}/"
        f"{summary['source_n_clusters']} source clusters received >=1 target member "
        f"({summary['n_source_clusters_empty_in_target']} empty in target). "
        f"Target unassigned (noise under approximate_predict) fraction = "
        f"{summary['target_unassigned_frac']:.4f}.",
        f"- REFERENCE ONLY (independently-fit target clustering -- NOT the transport "
        f"result, never conflate the two): n_clusters={summary['reference_target_n_clusters']}, "
        f"noise_frac={summary['reference_target_noise_frac']:.4f}.",
        f"- Stability split (Task 3, whole-cohort nc20_nn30_mcs100, Jaccard>=0.75, "
        f"inherited via best-overlap mapping): "
        f"{summary['n_stable_clusters_receiving_target']}/{summary['n_stable_source_clusters']} "
        f"stable source clusters received target members vs. "
        f"{summary['n_unstable_clusters_receiving_target']}/{summary['n_unstable_source_clusters']} "
        f"unstable source clusters.",
        f"- Organism-composition cosine similarity (source cluster vs. its transported "
        f"target counterpart, {ORGANISM_COL}), median: "
        f"all={summary['median_organism_cosine_similarity_all']}, "
        f"stable-source={summary['median_organism_cosine_similarity_stable']}, "
        f"unstable-source={summary['median_organism_cosine_similarity_unstable']}.",
        f"- Wall time: source fit={summary['fit_wall_time_sec']}s, "
        f"transform+predict={summary['transport_predict_wall_time_sec']}s, "
        f"reference target fit={summary['reference_fit_wall_time_sec']}s.",
        "",
        "Per-cluster detail:",
        "",
        "```",
        metrics.sort_values("source_cluster").to_string(index=False),
        "```",
        "",
    ]
    return "\n".join(lines)


# =========================================================================
# Aggregation (run once after all direction array tasks complete)
# =========================================================================

def aggregate(out_dir: Path = OUT, organism_filter: str | None = None) -> dict:
    """Collects the `summary_<direction>.json`/`FRAGMENT_<direction>.md`/
    `metrics_<direction>.csv` written by each direction's array task into a
    single `FINDINGS.md`/`manifest_all.json`/`metrics_summary.csv`. Same
    convention as `10_within_pathogen.aggregate`: if any direction is
    missing, reports it explicitly rather than staying silent. If
    `organism_filter` is given (Part D), states it in the preamble, and
    reports directions skipped for insufficient size (only a
    `summary_*.json` with `skipped: true`, no FRAGMENT/metrics) in a
    separate "skipped (too small, skipped)" list rather than lumping them
    into "missing (failed)"."""
    preamble = [
        "# Task 8: Transport validation across hospitals and calendar periods"
        + (f" -- organism_filter={organism_filter}" if organism_filter else ""),
        "",
        "(prior review comments). The published paper performed no such check.",
        "",]
    if organism_filter:
        preamble += [
            f"**[Task 10 expansion Part D] organism_filter={organism_filter}**: the "
            "frame was first restricted to this single organism, and the 4 "
            "directions below were then applied. min_cluster_size is (unless "
            "explicitly overridden) not the fixed whole-cohort value of 100, but is "
            "auto-derived from each direction's SOURCE stratum size "
            "(`derive_min_cluster_size_for_stratum`, n_subset//200, floor=50 -- the "
            "same rule as Task 10 Part A) -- the `min_cluster_size` field in each "
            "direction's summary below is the actual value used.",
            "",
        ]
    preamble += [
        "**Method**: for each direction, UMAP(n_components=20, n_neighbors=30) + "
        "HDBSCAN(min_cluster_size=100) is fit on the SOURCE stratum only "
        "(`02_recluster.run_config(..., return_models=True)`). The TARGET stratum is "
        "then projected into that SAME learned space via `umap.UMAP.transform()` and "
        "assigned via `hdbscan.approximate_predict()` -- the target is never refit. "
        "Four directions are run (both ways per axis): hospital 10<->20, "
        f"period early(<=({EARLY_YEAR_CUTOFF}))<->late.",
        "",
        "**Leakage caveat (Task 12)**: `treatment_department_eng` differs by hospital "
        "(51 labels at hospital 10, 38 at hospital 20, only 34 shared) and report "
        "narratives embed the literal string `Admission year`. Both are baked into the "
        "fixed (original) OpenAI embeddings used here and cannot be removed in this "
        "task. Any apparent hospital- or era-specific separation reported below may "
        "partly reflect these tokens rather than genuine clinical structure. Task 6's "
        "`no_department`/`no_year` ablation embeddings are the pending direct test of "
        "this (if transport improves under those variants, that would be direct "
        "evidence the tokens acted as leakage) -- that comparison is NOT run in this "
        "task (those ablation embeddings had not been generated at the time of this "
        "run) and is left as a cross-reference, not resolved by speculation here.",
        "",
        "**Stability caveat (Task 3)**: only 19/40 whole-cohort clusters "
        "(nc20_nn30_mcs100, Jaccard>=0.75) survived bootstrap resampling. Each source "
        "cluster below is mapped to its best-overlapping whole-cohort cluster to "
        "inherit that stability label; an unstable source cluster failing to transport "
        "is uninformative and is always reported separately from stable-cluster "
        "transport, never pooled with it.",
        "",
        "Outcome-blind: no mortality/discharge/death column is read anywhere in this "
        "script.",
        "",
    ]

    summaries = {}
    missing = []
    skipped = []
    body = []
    metrics_frames = []
    for d in DIRECTIONS:
        name = d["name"]
        fpath = out_dir / f"FRAGMENT_{name}.md"
        spath = out_dir / f"summary_{name}.json"
        mpath = out_dir / f"metrics_{name}.csv"
        if spath.exists() and not fpath.exists() and not mpath.exists():
            # written only by main()'s too-small-stratum guard (Part D) --
            # not a failure, an explicit refusal to force the configuration.
            s = json.loads(spath.read_text())
            if s.get("skipped"):
                skipped.append(name)
                summaries[name] = s
                continue
        if not (fpath.exists() and spath.exists() and mpath.exists()):
            missing.append(name)
            continue
        summaries[name] = json.loads(spath.read_text())
        body.append(fpath.read_text())
        m = pd.read_csv(mpath)
        m.insert(0, "direction", name)
        metrics_frames.append(m)

    text = "\n".join(preamble) + "\n".join(body)
    if skipped:
        text += (
            f"\n\n**{len(skipped)} direction(s) skipped ({', '.join(skipped)})** -- "
            "source/target stratum too small for this configuration "
            f"(< MIN_VIABLE_STRATUM_N={MIN_VIABLE_STRATUM_N}); not forced. "
            "See each direction's summary_*.json skip_reason for the measured counts.\n"
        )
    if missing:
        text += (
            f"\n\n**WARNING: {len(missing)} direction(s) missing "
            f"({', '.join(missing)})** -- not yet run or failed; the above is "
            "incomplete.\n"
        )
    _atomic_write_text(text, out_dir / "FINDINGS.md")

    if metrics_frames:
        _atomic_write_csv(pd.concat(metrics_frames, ignore_index=True), out_dir / "metrics_summary.csv")

    combined = {
        "directions": summaries,
        "missing_directions": missing,
        "skipped_directions": skipped,
        "organism_filter": organism_filter,
        "stability_threshold": STABILITY_THRESHOLD,
        "early_year_cutoff": EARLY_YEAR_CUTOFF,
    }
    recl._atomic_write_json(combined, out_dir / "manifest_all.json")

    print(f"aggregated {len(summaries)}/{len(DIRECTIONS)} directions -> {out_dir / 'FINDINGS.md'}")
    if missing:
        print(f"MISSING: {missing}")
    return combined


# =========================================================================
# main
# =========================================================================

def derive_min_cluster_size_for_stratum(n_stratum: int, floor: int = 50) -> int:
    """[Part D, --organism-filter only] Repeats exactly the primary-value
    rule of `10_within_pathogen.scaled_min_cluster_sizes` (re-applying the
    originally derived 161=32,329//200) -- duplicated as a single line
    rather than moved to a shared module (this repo's established
    convention, see the `load_organism_frame` docstring -- join/derivation
    logic is not shared across tasks). Forcing the whole-cohort-scale
    `MIN_CLUSTER_SIZE=100` as-is onto an organism-filtered stratum
    (thousands of cases) is relatively much more suppressive -- a pitfall
    already confirmed in Task 10 Part A.
    """
    return max(floor, n_stratum // 200)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--direction-index", type=int, default=None,
        help="Index into DIRECTIONS (0-3). Runs exactly one direction.",
    )
    ap.add_argument(
        "--list-directions", action="store_true",
        help="Prints the index -> direction name mapping and exits (for array-size verification).",
    )
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument(
        "--aggregate", action="store_true",
        help="After all directions have been processed, aggregates the results into FINDINGS.md/manifest_all.json.",
    )
    ap.add_argument(
        "--organism-filter", default=None,
        help="[Task 10 expansion Part D] Pre-restricts the frame to a single "
             "BSI_episode_short_clean value (e.g. 'Ecoli'). When given, results "
             "are written to out/08_transport_<slug>/ (never overwriting the "
             "whole-cohort out/08_transport/); min_cluster_size is, unless "
             "explicitly set via --min-cluster-size, auto-derived for each "
             "direction's SOURCE stratum size via "
             "derive_min_cluster_size_for_stratum() (does not force the "
             "whole-cohort fixed value of 100). Omitting it (default) behaves "
             "exactly as before (no regression).",
    )
    ap.add_argument(
        "--min-cluster-size", type=int, default=None,
        help="Explicitly pins min_cluster_size. If omitted together with "
             "--organism-filter, it is auto-derived from the stratum size; if "
             "omitted without --organism-filter, the whole-cohort default of "
             "100 is used (existing behavior).",
    )
    return ap.parse_args(argv)


def _out_dir_for(organism_filter: str | None) -> Path:
    if organism_filter is None:
        return OUT
    slug = organism_filter.lower()
    return OUT.parent / f"08_transport_{slug}"


def main() -> None:
    args = parse_args()
    out_dir = _out_dir_for(args.organism_filter)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.list_directions:
        print(f"n_directions={len(DIRECTIONS)}")
        for idx, d in enumerate(DIRECTIONS):
            print(f"{idx}\t{d['name']}\t{d}")
        return

    if args.aggregate:
        aggregate(out_dir, organism_filter=args.organism_filter)
        return

    if args.direction_index is None:
        raise SystemExit("--direction-index is required unless --list-directions/--aggregate is given")
    if not (0 <= args.direction_index < len(DIRECTIONS)):
        raise SystemExit(
            f"--direction-index {args.direction_index} out of range [0, {len(DIRECTIONS) - 1}]"
        )

    direction = DIRECTIONS[args.direction_index]
    print(f"[main] direction={direction['name']} {direction} organism_filter={args.organism_filter}",
          flush=True)

    print("[main] loading organism-joined frame", flush=True)
    frame = load_organism_frame()
    if args.organism_filter is not None:
        frame = frame.loc[frame[ORGANISM_COL] == args.organism_filter].reset_index(drop=True)
        if frame.empty:
            raise SystemExit(f"--organism-filter {args.organism_filter!r} matched 0 rows")
    categories = sorted(frame[ORGANISM_COL].unique().tolist())

    # [Part D] verify + report stratum counts BEFORE clustering, and refuse to
    # force a stratum too small for this configuration (brief requirement) --
    # cheap (a boolean mask + len()), so do it even though process_direction
    # will recompute the same masks internally.
    source_n = int((_stratum_mask(frame, direction["axis"], direction["source"])).sum())
    target_n = int((_stratum_mask(frame, direction["axis"], direction["target"])).sum())
    print(f"[main] stratum counts (organism_filter={args.organism_filter}): "
          f"source({direction['axis']}={direction['source']})={source_n}, "
          f"target({direction['axis']}={direction['target']})={target_n}", flush=True)
    if source_n < MIN_VIABLE_STRATUM_N or target_n < MIN_VIABLE_STRATUM_N:
        skip_reason = (
            f"source_n={source_n} or target_n={target_n} < MIN_VIABLE_STRATUM_N="
            f"{MIN_VIABLE_STRATUM_N}; too small to support this clustering "
            "configuration -- not forced, not clustered."
        )
        print(f"[main] SKIPPED: {skip_reason}", flush=True)
        recl._atomic_write_json(
            {"direction": direction["name"], "skipped": True, "skip_reason": skip_reason,
             "source_n": source_n, "target_n": target_n, "organism_filter": args.organism_filter},
            out_dir / f"summary_{direction['name']}.json",
        )
        return

    min_cluster_size = args.min_cluster_size
    if min_cluster_size is None:
        min_cluster_size = (
            derive_min_cluster_size_for_stratum(source_n) if args.organism_filter is not None
            else MIN_CLUSTER_SIZE
        )
    print(f"[main] min_cluster_size={min_cluster_size} "
          f"({'auto-derived from source stratum n' if args.organism_filter and args.min_cluster_size is None else 'fixed'})",
          flush=True)

    print("[main] loading Task 3 whole-cohort stability", flush=True)
    whole_labels_df, stable_clusters = load_whole_cohort_stability()
    print(f"[main] {len(stable_clusters)} stable whole-cohort clusters", flush=True)

    summary = process_direction(
        frame, direction, out_dir, whole_labels_df, stable_clusters, categories,
        seed=args.seed, min_cluster_size=min_cluster_size, organism_filter=args.organism_filter,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str), flush=True)
    print("[main] done", flush=True)


if __name__ == "__main__":
    main()
