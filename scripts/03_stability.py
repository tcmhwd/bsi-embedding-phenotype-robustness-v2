#!/usr/bin/env python3
"""Evaluate cluster stability via bootstrap resampling (ARI + per-cluster Jaccard).

(Responds to prior review comments): the originally derived paper has no
cluster-stability metric at all. This script builds a base clustering once,
then repeats n_boot times the process of resampling rows with replacement
(bootstrap) and refitting UMAP+HDBSCAN from scratch, recording the Adjusted
Rand Index (ARI) against the base labels and the per-cluster Jaccard
coefficient.

Decision criterion (brief Step 5): only clusters with mean Jaccard >= 0.75
are named "stable phenotypes"; those below that are downgraded to
exploratory findings (recorded in out/03_stability/FINDINGS.md based on the
final run results — this script itself does not render that verdict).

Outcome (mortality) variables are not loaded (Global Constraints):
30d_mortality, 90d_mortality, in_hospital_mortality, death_date,
discharge_date are not referenced anywhere in this task's code paths.
"""
import os

# Force single-threaded BLAS/OpenMP/numba execution BEFORE numpy/sklearn/
# umap/hdbscan/numba are imported (identical pattern to 02_recluster.py --
# see that module's top-of-file comment for the full empirical justification:
# 1/8/24 threads gave 26/18/33 clusters for the same config/seed on
# nc2_nn30_mcs200 even with random_state fixed). A stability analysis whose
# own clustering calls are themselves thread-count-dependent would be
# self-defeating, so this MUST be set before any scientific import below.
os.environ["OMP_NUM_THREADS"] = os.environ["NUMBA_NUM_THREADS"] = \
    os.environ["MKL_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import importlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(__file__))
frame_mod = importlib.import_module("01_build_analysis_frame")
recl = importlib.import_module("02_recluster")  # run_config, _thread_config, _library_versions

OUT = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision/out/03_stability")

CONFIG_TAG_RE = re.compile(r"^nc(?P<nc>\d+)_nn(?P<nn>\d+)_mcs(?P<mcs>\d+)$")


def parse_config_tag(tag: str) -> dict:
    """"ncX_nnY_mcsZ" -> {"n_components": X, "n_neighbors": Y, "min_cluster_size": Z}.

    Only UMAP configs (n_components > 0) are supported by this tag format,
    matching 02_recluster.tag_for()'s convention for that branch; the
    n_components=0 (no-UMAP) branch is exercised directly via
    bootstrap_stability()'s config dict in tests, not through this parser.
    """
    m = CONFIG_TAG_RE.match(tag)
    if not m:
        raise ValueError(f"unrecognized config tag: {tag!r} (expected ncX_nnY_mcsZ)")
    return {
        "n_components": int(m["nc"]),
        "n_neighbors": int(m["nn"]),
        "min_cluster_size": int(m["mcs"]),
    }


def jaccard_for_cluster(base_labels, new_labels, k):
    """Jaccard coefficient between base's cluster k and the new cluster that overlaps it best."""
    base_set = np.where(base_labels == k)[0]
    if len(base_set) == 0:
        return float("nan")
    best = 0.0
    for j in set(new_labels) - {-1}:
        new_set = np.where(new_labels == j)[0]
        inter = len(np.intersect1d(base_set, new_set))
        union = len(np.union1d(base_set, new_set))
        best = max(best, inter / union if union else 0.0)
    return float(best)


def _subset_rows(data, idx):
    """The positional equivalent of `data[idx]`, working identically whether
    `data` is a numpy array (Task 3's embeddings) or a pandas DataFrame
    (Task 5c's structured matrix) -- this is the single branch point that
    lets `bootstrap_stability_generic` support both data types.
    """
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.iloc[idx]
    return data[idx]


def bootstrap_stability_generic(data, base_labels, fit_fn, n_boot=50, seed=42):
    """Generalized version that computes a list of ARIs and a per-cluster
    Jaccard summary over n_boot bootstrap resamples (added for Task 5c).

    Keeps `bootstrap_stability`'s (below) original bootstrap-resampling logic
    (resample with replacement -> refit only on the unique indices -> compare
    against the base labels via ARI/Jaccard) exactly as-is, but factors out
    "what to fit and how" into an arbitrary callable of the form
    `fit_fn(data_subset, seed) -> labels`. Task 3's UMAP+HDBSCAN
    (`02_recluster.run_config`, with signature
    `(emb, n_components, n_neighbors, min_cluster_size, seed)`) and Task 5c's
    five structured-data methods (`05_baseline_comparators.fit_*`, with
    signature `(matrix, categorical_cols, k, seed)`) have different
    signatures and can't be called directly from a single function, so the
    caller wraps its actual signature as `fit_fn` and passes it to this
    function -- the bootstrap-resampling logic itself exists only once, in
    this function, with no duplication.

    `data` supports both a numpy array (embeddings) and a pandas DataFrame
    (structured matrix) (see `_subset_rows`).
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    ari_list, jac_rows = [], []
    clusters = sorted(set(base_labels) - {-1})
    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        uniq = np.unique(idx)
        new_labels = np.asarray(fit_fn(_subset_rows(data, uniq), seed + b))
        ari_list.append(adjusted_rand_score(base_labels[uniq], new_labels))
        for k in clusters:
            jac_rows.append({"boot": b, "cluster": k,
                              "jaccard": jaccard_for_cluster(base_labels[uniq], new_labels, k)})
    jac = (pd.DataFrame(jac_rows).groupby("cluster")["jaccard"]
           .agg(jaccard_mean="mean", jaccard_p05=lambda s: s.quantile(0.05)).reset_index())
    return ari_list, jac


def bootstrap_stability(emb, base_labels, config, n_boot=50, seed=42):
    """UMAP+HDBSCAN-only wrapper -- functionally identical to passing a
    `fit_fn` that wraps `run_config` to `bootstrap_stability_generic`
    (backward-compatible: both the signature and behavior are unchanged from
    before, so existing callers, including `main()`, keep working as-is
    with no change). The resampling logic itself no longer lives here --
    see `bootstrap_stability_generic`.
    """
    def fit_fn(sub_emb, s):
        return recl.run_config(sub_emb, config["n_components"], config["n_neighbors"],
                                config["min_cluster_size"], seed=s)["labels"]
    return bootstrap_stability_generic(emb, base_labels, fit_fn, n_boot=n_boot, seed=seed)


##############################################################################
# Design S -- subsample stability (DESIGN_embedding_and_stability.md §2.2-2.6)
#
# Replaces `bootstrap_stability_generic` for NEW callers. The function above
# is left completely unchanged (same name, same signature, same defective
# bootstrap-with-replacement behaviour) -- Tasks 5c and 10 keep calling it
# and keep reproducing exactly the numbers they have already reported.
# Migrating those two callers onto this design is Wave 2.3/2.4 of the
# design doc's own re-run order (after the clustering anchor settles) and
# is out of scope for this change; see task-3-redesign-report.md
# "backward-compatibility decision" for the full rationale. Nothing here
# is imported by any existing caller, so this addition changes no
# previously-reported number.
#
# What this fixes (AUDIT C4): bootstrap-with-replacement + np.unique
# retains ~63.2% of rows on average while `min_cluster_size` stays fixed at
# its full-data value, so any base cluster with size < min_cluster_size/0.632
# cannot re-form by construction -- the bootstrap was measuring cluster size,
# not stability (Spearman rho(mean Jaccard, cluster size) = 0.698,
# p = 5.7e-7, measured against out/03_stability/per_cluster_jaccard.csv).
##############################################################################


def scale_threshold(value_full: int, f: float) -> int:
    """Scales HDBSCAN's absolute-count thresholds (`min_cluster_size`/
    `min_samples`) by the subsampling fraction `f` (DESIGN §2.2.2):
    `max(2, round(f * value_full))`. Without this scaling, the threshold
    stays at the full-sample size while resampled clusters shrink by a
    factor of f on average, so small clusters become structurally unable to
    re-form (this is exactly the C4 defect). When `min_samples=None` (in the
    full config, = min_cluster_size), the caller also passes
    `value_full=min_cluster_size_full` to scale it the same way (§2.2.2
    "scale it identically").
    """
    return max(2, round(f * value_full))


def patient_subsample_indices(patient_ids, f: float, seed: int) -> np.ndarray:
    """Patient-level sampling without replacement (DESIGN §2.2.1): draws
    patients one at a time in a seeded random order, continuing until the
    running total of admission (row) counts for the drawn patients first
    reaches `f * n` or more. All of a given patient's admission rows are
    always included together or excluded together (no patient is ever split
    across the in/out partition -- prevents Mo7; this cohort has 32,329
    admissions / 27,083 patients, so multiple admissions per patient are not
    rare).

    The return value is an array of positional integer indices into
    `data`/`base_labels`, sorted in ascending order (preserves original row
    order -- this has no effect on the logic itself, it's purely for
    reproducibility/debugging convenience).
    """
    patient_ids = np.asarray(patient_ids)
    n = len(patient_ids)
    target = f * n
    rng = np.random.default_rng(seed)
    unique_patients, inverse, counts = np.unique(
        patient_ids, return_inverse=True, return_counts=True
    )
    order = rng.permutation(len(unique_patients))
    selected_mask = np.zeros(len(unique_patients), dtype=bool)
    total = 0
    for pos in order:
        selected_mask[pos] = True
        total += counts[pos]
        if total >= target:
            break
    row_mask = selected_mask[inverse]
    return np.where(row_mask)[0]


def _pairwise_coassignment(replicate_labels_of_retained_members) -> float:
    """For the resample-retained members of base cluster C, given the label
    array in that replicate (`replicate_labels_of_retained_members`, with
    noise as -1), computes "the fraction of pairs co-assigned to the same
    replicate cluster" (DESIGN §2.2.4). If either point in a pair is noise
    (-1), that pair is counted as "not co-assigned" -- noise-noise pairs are
    included in the total pair count but excluded from the co-assigned pair
    count.
    """
    labels = np.asarray(replicate_labels_of_retained_members)
    n = len(labels)
    if n < 2:
        return float("nan")
    total_pairs = n * (n - 1) // 2
    non_noise = labels[labels != -1]
    if len(non_noise) < 2:
        return 0.0
    _, counts = np.unique(non_noise, return_counts=True)
    coassigned_pairs = int(np.sum(counts * (counts - 1) // 2))
    return coassigned_pairs / total_pairs


def _normalized_entropy(replicate_labels_of_retained_members) -> float:
    """Normalized entropy of the label distribution that base cluster C's
    resample-retained members scatter into within this replicate (0 = all
    collapse onto a single label, 1 = maximally and evenly scattered across
    as many labels as were observed). Noise (-1) is treated as just another
    distinct label.
    """
    labels = np.asarray(replicate_labels_of_retained_members)
    vals, counts = np.unique(labels, return_counts=True)
    if len(vals) <= 1:
        return 0.0
    p = counts / counts.sum()
    ent = float(-(p * np.log(p)).sum())
    return ent / np.log(len(vals))


def ari_excluding_noise(labels_a, labels_b) -> float:
    """ARI computed keeping only rows that are non-noise (-1) in both
    labelings (Mo3): noise points count each other as "the same cluster"
    and so inflate ARI, so this metric must be reported alongside the
    noise-inclusive ARI to avoid an artificial overestimate at 5-7% noise.
    """
    labels_a = np.asarray(labels_a)
    labels_b = np.asarray(labels_b)
    mask = (labels_a != -1) & (labels_b != -1)
    if mask.sum() < 2:
        return float("nan")
    return float(adjusted_rand_score(labels_a[mask], labels_b[mask]))


OUTCOME_CLASSES = ("recovered", "perturbed", "split", "merged", "dissolved")


def classify_outcome(base_labels_sub, new_labels, k) -> dict:
    """Five-way outcome classification for base cluster `k` in one replicate
    (DESIGN §2.5). `base_labels_sub`/`new_labels` are position-aligned
    arrays over the same resampled subset (kept separate from the
    resampling logic itself so that just this function is unit-testable).

    Classification rules (fixed, taken verbatim from the table):
      recovered: best-match Jaccard J >= 0.75
      perturbed: 0.50 <= J < 0.75, and not split
      split:     two or more replicate clusters each account for >=25% of
                 the retained members
      merged:    the best-match replicate cluster accounts for >=80% of the
                 retained members, but J < 0.50
      dissolved: >=50% of the retained members are noise (-1) in the
                 replicate

    Application order: the split condition is a structural condition
    independent of the J range, so it is checked first (it is in fact
    mutually exclusive with the others -- if two clusters each account for
    25% or more, the best-match J cannot arithmetically reach 0.75). Then
    recovered -> perturbed -> merged -> dissolved, in that order.

    Residual cases the spec does not explicitly cover (J<0.50, best-match
    coverage<0.80, noise<0.50, not split -- i.e. scattered into minor
    fragments none of which reaches 25% and where noise is not a majority
    either) are conservatively classified as `dissolved`: since none of the
    five categories supports calling it "re-formed," the more honest default
    is "this cluster does not exist as a recognizable density peak in this
    replicate." This choice is recorded in task-3-redesign-report.md.
    """
    base_labels_sub = np.asarray(base_labels_sub)
    new_labels = np.asarray(new_labels)
    mask = base_labels_sub == k
    retained_size = int(mask.sum())
    if retained_size == 0:
        return None

    sub_labels = new_labels[mask]
    noise_frac = float(np.mean(sub_labels == -1))
    non_noise = sub_labels[sub_labels != -1]
    if len(non_noise) > 0:
        _, counts = np.unique(non_noise, return_counts=True)
        fracs = counts / retained_size
        best_recall = float(fracs.max())
        n_major = int((fracs >= 0.25).sum())
    else:
        best_recall = 0.0
        n_major = 0
    split = n_major >= 2
    jaccard = jaccard_for_cluster(base_labels_sub, new_labels, k)

    if split:
        outcome = "split"
    elif jaccard >= 0.75:
        outcome = "recovered"
    elif jaccard >= 0.50:
        outcome = "perturbed"
    elif best_recall >= 0.80:
        outcome = "merged"
    else:  # noise_frac >= 0.50 lands here too; residual/ambiguous cases default here as well
        outcome = "dissolved"

    return {
        "cluster": k,
        "retained_size": retained_size,
        "jaccard": jaccard,
        "coassign": _pairwise_coassignment(sub_labels),
        "noise_frac": noise_frac,
        "entropy": _normalized_entropy(sub_labels),
        "best_recall": best_recall,
        "outcome": outcome,
    }


def subsample_stability_generic(
    data, base_labels, fit_fn, patient_ids,
    min_cluster_size_full, min_samples_full=None,
    f: float = 0.632, n_boot: int = 200, seed: int = 42,
):
    """Design S's resampling loop (DESIGN §2.2, §2.4). Occupies the same
    slot as `bootstrap_stability_generic` but differs in data type:
      - subsamples **patients**, not rows, without replacement
        (`patient_subsample_indices`);
      - scales `min_cluster_size`/`min_samples` by `f` before passing them
        to `fit_fn`;
      - the clustering's own random_state is fixed per replicate (`seed`) --
        only the subsampling rng varies, as `seed + b` (DESIGN §2.4 seed
        row: the key device for separating "resampling variance" from "seed
        variance." Seed variance is measured separately by
        `seed_only_floor_generic`).

    Requires the signature
    `fit_fn(data_subset, seed, min_cluster_size_sub, min_samples_sub) -> labels`
    (whether the threshold scaling is actually used is up to the `fit_fn`
    implementation -- HDBSCAN-based methods pass it through, fixed-k methods
    may ignore it).

    Returns: `(per_cluster_df, per_boot_df, extra)` -- `per_cluster_df` is
    the resampling half of the DESIGN §2.5 columns (cohort size / resampling
    metrics), `per_boot_df` is raw (cluster, boot)-level data (for
    diagnostics/debugging/plotting), and `extra` is a dict holding the ARI
    summary and the scaled thresholds.
    """
    base_labels = np.asarray(base_labels)
    patient_ids = np.asarray(patient_ids)
    n = len(base_labels)
    if len(patient_ids) != n:
        raise ValueError("patient_ids must be aligned 1:1 with data/base_labels")

    clusters = sorted(set(base_labels) - {-1})
    sizes = {k: int(np.sum(base_labels == k)) for k in clusters}

    if min_samples_full is None:
        min_samples_full = min_cluster_size_full
    mcs_sub = scale_threshold(min_cluster_size_full, f)
    ms_sub = scale_threshold(min_samples_full, f)

    per_boot_rows = []
    ari_all_list, ari_excl_list = [], []
    for b in range(n_boot):
        idx = patient_subsample_indices(patient_ids, f, seed=seed + b)
        base_sub = base_labels[idx]
        new_labels = np.asarray(fit_fn(_subset_rows(data, idx), seed, mcs_sub, ms_sub))
        ari_all_list.append(float(adjusted_rand_score(base_sub, new_labels)))
        ari_excl_list.append(ari_excluding_noise(base_sub, new_labels))
        for k in clusters:
            row = classify_outcome(base_sub, new_labels, k)
            if row is not None:
                row["boot"] = b
                per_boot_rows.append(row)

    per_boot = pd.DataFrame(per_boot_rows)
    per_cluster = _summarize_resample_per_cluster(per_boot, sizes, f, mcs_sub)
    extra = {
        "f": f, "n_boot": n_boot, "seed": seed,
        "min_cluster_size_full": int(min_cluster_size_full),
        "min_samples_full": int(min_samples_full),
        "min_cluster_size_sub": mcs_sub, "min_samples_sub": ms_sub,
        "ari_all_mean": float(np.mean(ari_all_list)) if ari_all_list else None,
        "ari_all_p05": float(np.quantile(ari_all_list, 0.05)) if ari_all_list else None,
        "ari_excl_noise_mean": (float(np.nanmean(ari_excl_list)) if ari_excl_list else None),
    }
    return per_cluster, per_boot, extra


def _percentile_ci(values, lo=2.5, hi=97.5):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(values, lo)), float(np.percentile(values, hi))


def _summarize_resample_per_cluster(per_boot: pd.DataFrame, sizes: dict, f: float, mcs_sub: int) -> pd.DataFrame:
    rows = []
    for k, size in sizes.items():
        sub = per_boot[per_boot["cluster"] == k] if not per_boot.empty else per_boot
        n_rep = len(sub)
        expected_retained = f * size
        if n_rep == 0:
            rows.append({
                "cluster": k, "size": size, "expected_retained_size": expected_retained,
                "jaccard_mean": float("nan"), "jaccard_p05": float("nan"),
                "jaccard_ci_lo": float("nan"), "jaccard_ci_hi": float("nan"),
                "coassign_mean": float("nan"), "coassign_ci_lo": float("nan"), "coassign_ci_hi": float("nan"),
                "frac_recovered": float("nan"), "frac_perturbed": float("nan"), "frac_split": float("nan"),
                "frac_merged": float("nan"), "frac_dissolved": float("nan"),
                "noise_absorption_mean": float("nan"), "label_entropy_mean": float("nan"),
                "structurally_resolvable": bool(expected_retained >= mcs_sub),
            })
            continue
        jac = sub["jaccard"].to_numpy(dtype=float)
        coa = sub["coassign"].to_numpy(dtype=float)
        jac_lo, jac_hi = _percentile_ci(jac)
        coa_lo, coa_hi = _percentile_ci(coa)
        outcome_counts = sub["outcome"].value_counts()
        rows.append({
            "cluster": k, "size": size, "expected_retained_size": expected_retained,
            "jaccard_mean": float(np.nanmean(jac)),
            "jaccard_p05": float(np.nanpercentile(jac, 5)),
            "jaccard_ci_lo": jac_lo, "jaccard_ci_hi": jac_hi,
            "coassign_mean": float(np.nanmean(coa)),
            "coassign_ci_lo": coa_lo, "coassign_ci_hi": coa_hi,
            "frac_recovered": outcome_counts.get("recovered", 0) / n_rep,
            "frac_perturbed": outcome_counts.get("perturbed", 0) / n_rep,
            "frac_split": outcome_counts.get("split", 0) / n_rep,
            "frac_merged": outcome_counts.get("merged", 0) / n_rep,
            "frac_dissolved": outcome_counts.get("dissolved", 0) / n_rep,
            "noise_absorption_mean": float(np.nanmean(sub["noise_frac"].to_numpy(dtype=float))),
            "label_entropy_mean": float(np.nanmean(sub["entropy"].to_numpy(dtype=float))),
            "structurally_resolvable": bool(expected_retained >= mcs_sub),
        })
    return pd.DataFrame(rows)


def seed_only_floor_generic(
    data, base_labels, fit_fn, min_cluster_size_full, min_samples_full=None,
    seeds=range(1, 21),
):
    """Seed-only floor (DESIGN §2.2.3, §2.4 "seed-only arm inverts this"):
    holds the data fixed in full (no resampling), and refits by varying only
    the clustering's `random_state` from 1..20. The Jaccard/coassign values
    observed even with no resampling at all are exactly "the agreement when
    nothing has actually changed" --- only a resampling result lower than
    this floor can be called "unstable" on that basis (this redesign's
    essential requirement is that any stability number must be read against
    this floor, not against 1.0).

    The `fit_fn(data, seed, min_cluster_size_full, min_samples_full) -> labels`
    signature is identical to `subsample_stability_generic`'s, so the same
    `fit_fn` can be reused as-is (with no scaling, i.e. with thresholds
    equivalent to f=1).
    """
    base_labels = np.asarray(base_labels)
    n = len(base_labels)
    clusters = sorted(set(base_labels) - {-1})
    sizes = {k: int(np.sum(base_labels == k)) for k in clusters}
    if min_samples_full is None:
        min_samples_full = min_cluster_size_full

    per_boot_rows = []
    for b, s in enumerate(seeds):
        new_labels = np.asarray(fit_fn(data, s, min_cluster_size_full, min_samples_full))
        for k in clusters:
            row = classify_outcome(base_labels, new_labels, k)
            if row is not None:
                row["boot"] = b
                row["seed"] = s
                per_boot_rows.append(row)

    per_boot = pd.DataFrame(per_boot_rows)
    per_cluster = _summarize_resample_per_cluster(per_boot, sizes, f=1.0, mcs_sub=min_cluster_size_full)
    return per_cluster, per_boot


def merge_stability_report(resample_summary: pd.DataFrame, seed_only_summary: pd.DataFrame) -> pd.DataFrame:
    """Merges the resampling summary with the seed-only floor per cluster to
    complete DESIGN §2.5's `per_cluster_stability.csv` schema. The
    seed-only-side columns always carry a `seed_only_` prefix so they are
    never mixed under the same column name as the resampling metrics
    (whether a Jaccard of 0.55 means "fragile under resampling" or simply
    reflects that that cluster's own seed-only floor is 0.6 and thus within
    the normal range must always be read side by side).
    """
    floor = seed_only_summary[["cluster", "jaccard_mean", "coassign_mean"]].rename(
        columns={"jaccard_mean": "seed_only_jaccard_mean", "coassign_mean": "seed_only_coassign_mean"}
    )
    return resample_summary.merge(floor, on="cluster", how="left")


def stability_size_spearman(per_cluster_df: pd.DataFrame, size_col="size", stability_col="jaccard_mean") -> dict:
    """Mandatory diagnostic (DESIGN §2.5): the Spearman rho between cluster
    size and stability. The naive design had rho=0.698, p=5.7e-7 (evidence
    of the design defect) -- the new design's rho must be reported
    alongside this value, and the requirement is that the design itself
    must be reconsidered if it has not dropped noticeably. This function
    only measures; it adjusts nothing to make the result look better.
    """
    from scipy.stats import spearmanr
    valid = per_cluster_df[[size_col, stability_col]].dropna()
    if len(valid) < 3:
        return {"rho": float("nan"), "p_value": float("nan"), "n": int(len(valid))}
    rho, p = spearmanr(valid[size_col], valid[stability_col])
    return {"rho": float(rho), "p_value": float(p), "n": int(len(valid)),
            "naive_design_rho": 0.698, "naive_design_p": 5.7e-7}


def held_out_membership_stability(reducer, clusterer, held_out_emb, held_out_base_labels) -> pd.DataFrame:
    """Secondary metric (DESIGN §2.2.5): with the base model held fixed (no
    refitting), assigns the (1-f) points excluded from the subsample using
    `hdbscan.approximate_predict`, and checks what fraction are reassigned
    to the same label as their original base cluster label. This asks a
    different question from "re-derivation" -- "will a new observation be
    assigned to this phenotype?" (directly relevant to Task 8
    transfer/clinical application) -- so it is kept as an independent
    column (`membership_stability`) and never merged with the resampling
    metrics (coassign/jaccard) (DESIGN §2.3 rejected-alternatives table:
    excluded from the primary metrics on the grounds that "what it
    measures is the assignment stability of a fixed model").
    """
    import hdbscan as _hdbscan
    space = reducer.transform(held_out_emb) if reducer is not None else held_out_emb
    pred_labels, _strengths = _hdbscan.approximate_predict(clusterer, space)
    df = pd.DataFrame({"cluster": np.asarray(held_out_base_labels), "predicted": np.asarray(pred_labels)})
    df = df[df["cluster"] != -1]
    if df.empty:
        return pd.DataFrame(columns=["cluster", "membership_stability"])
    out = (df.assign(match=df["cluster"] == df["predicted"])
             .groupby("cluster")["match"].mean()
             .reset_index().rename(columns={"match": "membership_stability"}))
    return out


def subsample_stability(emb, base_labels, patient_ids, config, f: float = 0.632,
                         n_boot: int = 200, seed: int = 42, min_samples_full=None):
    """UMAP+HDBSCAN-only wrapper (Task 3) -- passes a `fit_fn` wrapping
    `recl.run_config` to `subsample_stability_generic`. `config` has the
    same dict shape as `bootstrap_stability`'s (`n_components`/`n_neighbors`/
    `min_cluster_size`, optionally `min_dist`/`metric`/`min_samples`/
    `cluster_selection_method`); if `min_samples_full` is not specified,
    `config.get("min_samples", 10)` is used (the default from the original
    39-config grid; for a `min_samples=None` config such as the originally
    derived one, the caller must explicitly pass
    `min_samples_full=config["min_cluster_size"]`
    -- DESIGN §2.2.2 "scale the min_samples=None case the same way").
    """
    mcs_full = config["min_cluster_size"]
    if min_samples_full is None:
        min_samples_full = config.get("min_samples", 10)

    def fit_fn(sub_emb, s, mcs_sub, ms_sub):
        return recl.run_config(
            sub_emb, config["n_components"], config["n_neighbors"], mcs_sub, seed=s,
            min_dist=config.get("min_dist", 0.0), metric=config.get("metric", "cosine"),
            min_samples=ms_sub, cluster_selection_method=config.get("cluster_selection_method", "eom"),
        )["labels"]

    return subsample_stability_generic(
        emb, base_labels, fit_fn, patient_ids,
        min_cluster_size_full=mcs_full, min_samples_full=min_samples_full,
        f=f, n_boot=n_boot, seed=seed,
    )


def seed_only_floor(emb, base_labels, config, seeds=range(1, 21), min_samples_full=None):
    """UMAP+HDBSCAN-only seed-only floor wrapper (Task 3)."""
    mcs_full = config["min_cluster_size"]
    if min_samples_full is None:
        min_samples_full = config.get("min_samples", 10)

    def fit_fn(full_emb, s, mcs, ms):
        return recl.run_config(
            full_emb, config["n_components"], config["n_neighbors"], mcs, seed=s,
            min_dist=config.get("min_dist", 0.0), metric=config.get("metric", "cosine"),
            min_samples=ms, cluster_selection_method=config.get("cluster_selection_method", "eom"),
        )["labels"]

    return seed_only_floor_generic(
        emb, base_labels, fit_fn, min_cluster_size_full=mcs_full,
        min_samples_full=min_samples_full, seeds=seeds,
    )


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="nc20_nn30_mcs100",
                     help="02_recluster grid config tag, format ncX_nnY_mcsZ")
    ap.add_argument("--n-boot", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--design", choices=["naive", "subsample"], default="naive",
                     help="'naive' (default) = unchanged bootstrap-with-replacement design "
                          "(AUDIT C4, size-confounded, kept for reproducibility of "
                          "already-reported numbers). 'subsample' = DESIGN_embedding_and_stability.md "
                          "Decision 2 (patient-level subsampling at f, scaled thresholds, "
                          "seed-only floor, co-assignment, five-way outcome classification).")
    ap.add_argument("--f", type=float, default=0.632, help="[subsample only] retained fraction")
    ap.add_argument("--n-seed-reps", type=int, default=20,
                     help="[subsample only] number of seed-only floor replicates (DESIGN §2.4: 20)")
    ap.add_argument("--min-samples-full", type=int, default=None,
                     help="[subsample only] override min_samples for the full-data fit; "
                          "defaults to the grid's min_samples=10, or pass min_cluster_size "
                          "explicitly for a min_samples=None (published) config (DESIGN §2.2.2)")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = parse_config_tag(args.config)

    print(f"[base] config={args.config} {cfg}", flush=True)

    # verify_reports=False: Task 1 already verified every report path exists.
    frame = frame_mod.load_frame(verify_reports=False)
    emb = frame_mod.load_embeddings(frame)

    t0 = time.time()
    base = recl.run_config(emb, cfg["n_components"], cfg["n_neighbors"],
                            cfg["min_cluster_size"], seed=args.seed)
    base_wall_time_sec = time.time() - t0
    base_labels = base["labels"]
    print(f"[base] n_clusters={base['n_clusters']} noise_frac={base['noise_frac']:.4f} "
          f"dbcv={base['dbcv']:.4f} wall_time_sec={base_wall_time_sec:.2f}", flush=True)

    if args.design == "naive":
        _run_naive_design(args, cfg, frame, emb, base, base_labels, base_wall_time_sec)
    else:
        _run_subsample_design(args, cfg, frame, emb, base, base_labels, base_wall_time_sec)


def _run_naive_design(args, cfg, frame, emb, base, base_labels, base_wall_time_sec) -> None:
    """Unchanged since before this redesign -- AUDIT C4 "naive" design,
    kept byte-for-byte so `--design naive` (the default) continues to
    reproduce exactly what is already reported in out/03_stability/.
    """
    t1 = time.time()
    ari, jac = bootstrap_stability(emb, base_labels, cfg, n_boot=args.n_boot, seed=args.seed)
    bootstrap_wall_time_sec = time.time() - t1

    pd.DataFrame({"boot": range(len(ari)), "ari": ari}).to_csv(OUT / "bootstrap_ari.csv", index=False)
    jac.to_csv(OUT / "per_cluster_jaccard.csv", index=False)

    manifest = {
        "config": args.config,
        "config_params": cfg,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_rows": int(len(frame)),
        "base_n_clusters": base["n_clusters"],
        "base_noise_frac": round(base["noise_frac"], 6),
        "base_dbcv": round(base["dbcv"], 6),
        "base_wall_time_sec": round(base_wall_time_sec, 2),
        "bootstrap_wall_time_sec": round(bootstrap_wall_time_sec, 2),
        "bootstrap_mean_sec_per_boot": round(bootstrap_wall_time_sec / args.n_boot, 2) if args.n_boot else None,
        "ari_mean": float(np.mean(ari)) if ari else None,
        "ari_p05": float(np.quantile(ari, 0.05)) if ari else None,
        "stability_threshold_jaccard_mean": 0.75,
        "n_clusters_stable": int((jac["jaccard_mean"] >= 0.75).sum()),
        "n_clusters_unstable": int((jac["jaccard_mean"] < 0.75).sum()),
        "libraries": recl._library_versions(),
        "threads": recl._thread_config(),
    }
    (OUT / "03_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"ARI mean={manifest['ari_mean']:.3f} p05={manifest['ari_p05']:.3f}", flush=True)
    print(jac.to_string(), flush=True)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


def _run_subsample_design(args, cfg, frame, emb, base, base_labels, base_wall_time_sec) -> None:
    """DESIGN_embedding_and_stability.md Decision 2 end-to-end driver: base fit
    (already done by caller) -> mandatory seed-only floor -> patient-level
    subsampling at `args.f` -> per-cluster report with the five-way outcome
    classification, co-assignment, and the size/stability Spearman rho
    diagnostic surfaced prominently (never buried) in both the console
    output and the manifest.

    NOTE (scope): this is the validation/production driver for a *single*
    SLURM task. At n_boot=200 (DESIGN §2.4) this must be run as an array,
    not this serial loop -- see task-3-redesign-report.md "what remains to
    be re-run" for the sharding wrapper still to be written before the
    real Task 3/10 re-runs (deferred until the clustering anchor settles,
    per the task's explicit scope boundary).
    """
    patient_ids = frame["patient_ID"].to_numpy()

    t_seed = time.time()
    seed_only_summary, seed_only_boot = seed_only_floor(
        emb, base_labels, cfg, seeds=range(1, args.n_seed_reps + 1),
        min_samples_full=args.min_samples_full,
    )
    seed_only_wall_time_sec = time.time() - t_seed
    print(f"[seed-only floor] {args.n_seed_reps} reps, wall_time_sec={seed_only_wall_time_sec:.2f}", flush=True)

    t_sub = time.time()
    resample_summary, resample_boot, extra = subsample_stability(
        emb, base_labels, patient_ids, cfg, f=args.f, n_boot=args.n_boot, seed=args.seed,
        min_samples_full=args.min_samples_full,
    )
    subsample_wall_time_sec = time.time() - t_sub

    report = merge_stability_report(resample_summary, seed_only_summary)
    rho_diag = stability_size_spearman(report)

    report.to_csv(OUT / "per_cluster_stability.csv", index=False)
    resample_boot.to_csv(OUT / "per_cluster_stability_boot_raw.csv", index=False)
    seed_only_boot.to_csv(OUT / "seed_only_boot_raw.csv", index=False)

    n_named = int(((report["jaccard_mean"] >= 0.75)
                    & (report["frac_dissolved"] <= 0.20)
                    & (report["coassign_mean"] >= 0.75)).sum())

    manifest = {
        "design": "subsample",
        "config": args.config,
        "config_params": cfg,
        "f": args.f,
        "n_boot": args.n_boot,
        "n_seed_reps": args.n_seed_reps,
        "seed": args.seed,
        "n_rows": int(len(frame)),
        "base_n_clusters": base["n_clusters"],
        "base_noise_frac": round(base["noise_frac"], 6),
        "base_dbcv": round(base["dbcv"], 6),
        "base_wall_time_sec": round(base_wall_time_sec, 2),
        "seed_only_wall_time_sec": round(seed_only_wall_time_sec, 2),
        "subsample_wall_time_sec": round(subsample_wall_time_sec, 2),
        "subsample_mean_sec_per_boot": (round(subsample_wall_time_sec / args.n_boot, 2) if args.n_boot else None),
        **extra,
        "n_clusters_named_eligible": n_named,
        "STABILITY_VS_SIZE_SPEARMAN_RHO": rho_diag,  # DESIGN §2.5 mandatory diagnostic -- surfaced first-class
        "libraries": recl._library_versions(),
        "threads": recl._thread_config(),
    }
    (OUT / "03_manifest_subsample.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))

    print("=" * 72, flush=True)
    print(f"STABILITY vs SIZE Spearman rho = {rho_diag['rho']:.4f} (p={rho_diag['p_value']:.3g}, "
          f"n={rho_diag['n']})  ||  naive design rho = {rho_diag.get('naive_design_rho')} "
          f"(p={rho_diag.get('naive_design_p')})", flush=True)
    print("=" * 72, flush=True)
    print(report.to_string(), flush=True)
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
