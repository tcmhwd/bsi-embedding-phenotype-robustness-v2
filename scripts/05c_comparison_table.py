#!/usr/bin/env python3
"""Task 5c: assembling the 5-axis comparison table across the five structured-data
clustering comparators.

This calls the five already-reviewed `fit_*` functions built by 5a
(`fit_kprototypes`/`fit_famd_kmeans`/`fit_gmm_structured`) and 5b
(`fit_gower_pam`/`fit_lca`) as-is -- no new clustering method is implemented
here. What this script newly builds is only the five-axis comparison a
prior review comment requested:

  (1) Number of clusters (actual number of distinct labels produced vs. the
      requested k)
  (2) Mean bootstrap ARI (reusing a generalised version of Task 3
      `03_stability.py`'s bootstrap logic)
  (3) ARI against the embedding-derived labels (Task 2 grid) and against the
      originally derived labels (`phenotype_cluster`)
  (4) Prognostic separation for 90-day mortality (5-fold stratified CV AUROC
      using only the cluster label as a feature) -- the only point in all of
      Task 5 that reads the outcome variable
  (6) Top 3 characteristic variables per cluster (clinical interpretability)

## Gower distance matrix reuse strategy (CRITICAL performance constraint)

Every call to `fit_gower_pam` honestly recomputes the O(n^2) Gower distance
matrix from scratch (measured by 5b at the full 32,329-row scale: 1640.5s,
~27.3 minutes -- see the `05_baseline_comparators.py` module docstring /
`task-5b-report.md`). This dispatcher needs to compare several `k` values
(`K_VALUES`), so naively calling `fit_gower_pam` once per `k` would repeat
that 27-minute computation once per `k` value.

**Approach adopted: (a) add a backward-compatible `precomputed_distance`
parameter to `05_baseline_comparators.fit_gower_pam`.** Alternative (b)
(writing a separate helper in this script that calls `gower.gower_matrix` +
`KMedoids(metric="precomputed")` directly) was also possible, but (a) was
chosen because it keeps the `GOWER_PAM_METHOD`/`GOWER_PAM_INIT` clustering
configuration managed from the single source in `05_baseline_comparators.py`,
eliminating the risk that the "compute once, reuse" path and the "recompute
every time" path would end up using different clustering configurations
(the same DRY principle this project also preferred for the bootstrap
generalisation). A new helper, `compute_gower_matrix(matrix, categorical_cols)`,
was added, and `fit_gower_pam` was refactored to call it internally too
(behaviour unchanged, re-confirmed by 22/22 passing tests). With
`precomputed_distance=None` (the default) behaviour is completely identical
to before, so existing 5a/5b callers and tests are entirely unaffected.

For `gower_pam`, this script calls `compute_gower_matrix` **exactly once,
before entering the `K_VALUES` loop**, and reuses that result across the
`fit_gower_pam(..., precomputed_distance=D)` calls for each `k` -- the
additional per-`k` cost is then only the `KMedoids` fit itself (a few
seconds). **The one exception is the bootstrap (axis 2):** each bootstrap
replicate resamples a different row subset, so the Gower distance matrix for
that subset cannot be reused from the full matrix (the values themselves
differ per subset) -- during bootstrapping, `fit_gower_pam` is called
without `precomputed_distance`, honestly recomputing it fresh on every
replicate.

## Bootstrap generalisation (modifying `03_stability.py` -- approach (a), the
preferred approach, was adopted)

`03_stability.bootstrap_stability(emb, base_labels, config, n_boot, seed)`
was written specifically for Task 2's
`run_config(emb, n_components, n_neighbors, min_cluster_size, seed)`, so the
five `fit_*` functions of this task, which have a different signature
(`fit_*(matrix, categorical_cols, k, seed)`), cannot be passed into it
as-is. Approach (a), the one the brief explicitly preferred, was adopted:
a new `bootstrap_stability_generic(data, base_labels, fit_fn, n_boot, seed)`
was added to `03_stability.py`, keeping the resampling logic (sample with
replacement -> refit on the unique indices -> ARI/Jaccard) in one place
only, and the existing `bootstrap_stability` was reimplemented as a thin
wrapper that passes a `fit_fn` wrapping `run_config` (backward-compatible --
identical signature/return value). `tests/test_03_stability.py` (Task 3's
existing 3 tests) was rerun to confirm no regression (3 passed). The
`_subset_rows` helper supports both numpy arrays (embeddings) and pandas
DataFrames (structured matrices).

## Design decisions driven by computational cost (see task-5c-report.md for
full figures)

Measured on the login node, `fit_kprototypes` already took 167.8s at
n=2,000/k=16, revealing that one of the four other methods the brief had
assumed would be "fast" was actually a cost problem comparable to
`fit_gower_pam` (FAMD/GMM were genuinely fast, at 0.5s and 0.6s respectively;
LCA (stepmix) was intermediate at 5.5s). SLURM job 586192 measured all four
methods (k-prototypes/FAMD+k-means/GMM/LCA) once each at the full
32,329-row scale to validate this estimate (see `task-5c-report.md`). Based
on these measurements:

- `K_VALUES = [16, 32]`: 16 is required to exactly match the originally
  derived cluster count (`phenotype_cluster` cardinality, confirmed by
  measurement excluding -1 noise); 32 is one context value drawn from the
  fixed k-grid ({8,16,32}) shared by 5a/5b/Task 2b (satisfying the brief's
  instruction for "one or two additional context values") -- k=8 was
  deliberately excluded to control total cost (2 k values, not 3).
- The bootstrap (axis 2) is only computed at `BOOTSTRAP_K = 16` (the
  matching value) -- it is not computed at k=32, which is left as NaN in
  `comparison_table.csv`. Repeating the bootstrap for every (method, k)
  combination would, given the measured costs of `fit_kprototypes`/
  `fit_gower_pam`, inflate total runtime to an unmanageable level (an
  estimated 10+ hours).
- The number of bootstrap replicates per method (`N_BOOT`) was scaled
  inversely to measured cost: the three fast methods (FAMD+k-means/GMM/LCA)
  get 20/20/15 replicates, while the two expensive methods
  (k-prototypes/Gower+PAM) were reduced to 3 -- the same rationale as for
  Gower+PAM (each bootstrap resample must pay the O(n^2) cost again from
  scratch) was applied to k-prototypes as well. This is an **explicit
  compromise forced by computational limits, and bootstrap ARI based on
  fewer than 5 replicates should be read as indicative only, not as a
  stable estimate** -- this is stated as-is in `comparison_table.csv` /
  DIAGNOSIS.md.

## Comparator embedding grid configuration (axis 3)

From `out/02_recluster/grid_summary.csv`, among the configurations whose
`n_clusters` is closest (15) to the originally derived 16 (on a
non-noise basis), the one with the lowest noise fraction and whose
`n_components` matches 5a's `FAMD_N_COMPONENTS=10`, `nc10_nn15_mcs200`
(n_clusters=15, noise_frac=0.018), was selected -- see `task-5c-report.md`
for the detailed comparison.

## Outcome-blind constraint (exact scope)

`30d_mortality`, `in_hospital_mortality`, `death_date` and `discharge_date`
are not read anywhere in this script's code paths. `90d_mortality` is read
only inside the `evaluate_prognostic_separation` function (the one
exception point in all of Task 5) -- no other function (the code for axes
1/2/3/5, including the rest of `build_comparison_row`) reads this column.
"""
import os

# Force single-threaded BLAS/OpenMP/numba execution BEFORE numpy/pandas/
# sklearn (or the imported 01/02/03/05 modules, which import those
# themselves) are imported -- identical pattern/rationale to every prior
# script in this project (02_recluster.py, 02b_partition_clustering.py,
# 03_stability.py, 05_baseline_comparators.py). This line MUST come before
# any of those imports below, including the `importlib.import_module` calls
# (Python executes a module's own top-of-file os.environ pinning only the
# first time it is imported, but this script imports numpy/pandas/sklearn
# directly itself below too, so its own pinning must come first regardless
# of import order among the sibling modules).
os.environ["OMP_NUM_THREADS"] = os.environ["NUMBA_NUM_THREADS"] = \
    os.environ["MKL_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))
frame_mod = importlib.import_module("01_build_analysis_frame")
recl = importlib.import_module("02_recluster")  # _thread_config, _library_versions
stability_mod = importlib.import_module("03_stability")  # bootstrap_stability_generic
comparators = importlib.import_module("05_baseline_comparators")  # build_structured_matrix, fit_*

OUT = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision/out/05_comparators")
DATASET = comparators.DATASET  # manuscript/phenotype_ml_dataset.csv

# Axis 3 reference: embedding-derived labels from Task 2's grid. Chosen
# config nc10_nn15_mcs200 (n_clusters=15, noise_frac=0.018) -- see module
# docstring "Comparator embedding grid configuration" for the selection rationale.
EMBEDDING_CONFIG_TAG = "nc10_nn15_mcs200"
EMBEDDING_LABELS_PATH = Path(
    "/data/tcmhwd/AI/AST_ai/phenotype_revision/out/02_recluster/labels_nc10_nn15_mcs200.parquet"
)

# Outcome column read ONLY inside evaluate_prognostic_separation (Global
# Constraints exception for Task 5c's axis 4; see module docstring).
MORTALITY_COL = "90d_mortality"

METHOD_FIT_FUNCS = {
    "kprototypes": comparators.fit_kprototypes,
    "famd_kmeans": comparators.fit_famd_kmeans,
    "gmm_structured": comparators.fit_gmm_structured,
    "gower_pam": comparators.fit_gower_pam,
    "lca": comparators.fit_lca,
}

# See module docstring "Design decisions driven by computational cost" for the full justification
# (measured costs, SLURM job 586192, task-5c-report.md).
K_VALUES = [16, 32]
BOOTSTRAP_K = 16
N_BOOT = {
    "kprototypes": 3,
    "famd_kmeans": 20,
    "gmm_structured": 20,
    "gower_pam": 3,
    "lca": 15,
}
DEFAULT_N_BOOT = 10  # fallback for any method not explicitly listed above
SEED = 42


def evaluate_prognostic_separation(labels, admission_ids, dataset_path=DATASET, seed=42):
    """Axis 4: 5-fold stratified CV AUROC (90-day mortality prediction) using
    only the cluster label as a feature. The one point in all of Task 5
    that reads the outcome variable -- no outcome/mortality-related column
    other than `MORTALITY_COL` ("90d_mortality") is read (explicitly
    restricted via `usecols`).

    A logistic regression is fit using a one-hot encoding of the cluster
    label as the sole feature set -- logistic regression was chosen over a
    random forest (per the task instructions) to ask "can knowing cluster
    membership alone discriminate mortality?" in the most directly
    interpretable way. The one-hot encoding is done once over the full
    label vector before splitting, so there is no risk of the column set
    differing between folds.
    """
    outcome = pd.read_csv(dataset_path, usecols=["admission_ID", MORTALITY_COL], low_memory=False)
    joined = pd.DataFrame({"admission_ID": admission_ids, "label": labels}).merge(
        outcome, on="admission_ID", how="left", validate="one_to_one",
    )
    assert joined[MORTALITY_COL].notna().all(), (
        f"{MORTALITY_COL} missing for some admission_ID after join -- dataset_path/admission_ids mismatch"
    )

    X = pd.get_dummies(joined["label"].astype(str)).to_numpy(dtype=float)
    y = joined[MORTALITY_COL].to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))
    return float(np.mean(aucs)), float(np.std(aucs))


def ari_against_reference(admission_ids, labels, reference_df, reference_col):
    """Axis 3: ARI against the reference label (`reference_col`) joined on
    `admission_ID`. The join is done safely, keyed on the order of
    `admission_ids`/`labels`, regardless of the reference frame's row order
    or duplication -- `-1` (e.g. HDBSCAN noise, the originally derived
    paper's unassigned code) is itself treated as just another label (this
    project's existing convention -- identical to the ARI computation in
    `03_stability.bootstrap_stability`).
    """
    ref = reference_df[["admission_ID", reference_col]].rename(columns={reference_col: "_ref_label"})
    joined = pd.DataFrame({"admission_ID": admission_ids, "_label": labels}).merge(
        ref, on="admission_ID", how="left", validate="one_to_one",
    )
    assert joined["_ref_label"].notna().all(), "reference label missing for some admission_ID after join"
    return float(adjusted_rand_score(joined["_ref_label"].to_numpy(), joined["_label"].to_numpy()))


def characteristic_features_per_cluster(imputed_matrix, categorical_cols, labels, top_n=3):
    """Axis 5: the top `top_n` characteristic variables per cluster.

    A unified principle (do not use different scales for numeric vs.
    categorical variables -- per the task instruction "principled,
    consistent method across both types"): every variable is scored by a
    standardised mean difference (SMD = (cluster mean - rest-of-sample
    mean) / overall-sample standard deviation). For categorical variables,
    the same SMD formula is applied to each one-hot-encoded level, and
    whichever level has the largest |SMD| is taken as that variable's
    representative score and representative level. This way, continuous
    variables (Cohen's-d-like) and categorical variables (proportion-
    difference-like) are not computed separately in different units and
    then mixed when ranking -- the top `top_n` per cluster can be drawn
    from one consistent scale.

    `imputed_matrix` must be the output of `impute_structured_matrix`
    (no missing values assumed).
    """
    numeric_cols = [c for c in imputed_matrix.columns if c not in categorical_cols]

    feat_cols = {}
    level_lookup = {}  # one-hot column name -> (original variable, level)
    for c in numeric_cols:
        feat_cols[c] = imputed_matrix[c].astype(float).to_numpy()
    for c in categorical_cols:
        onehot = pd.get_dummies(imputed_matrix[c].astype(str), prefix=c)
        for col in onehot.columns:
            level_lookup[col] = (c, col[len(c) + 1:])
            feat_cols[col] = onehot[col].to_numpy(dtype=float)

    feat_df = pd.DataFrame(feat_cols)
    overall_std = feat_df.std(ddof=0)

    labels = np.asarray(labels)
    rows = []
    for cl in sorted(set(labels.tolist())):
        mask = labels == cl
        cluster_mean = feat_df.loc[mask].mean()
        rest_mean = feat_df.loc[~mask].mean()
        with np.errstate(invalid="ignore", divide="ignore"):
            smd = (cluster_mean - rest_mean) / overall_std
        smd = smd.replace([np.inf, -np.inf], np.nan).dropna()

        # Collapse one-hot levels back to their parent categorical variable,
        # keeping only the most distinctive (largest |SMD|) level per variable.
        best_per_variable = {}
        for col, val in smd.items():
            var, level = level_lookup.get(col, (col, None))
            if var not in best_per_variable:
                best_per_variable[var] = (level, val)
                continue
            cur_level, cur_val = best_per_variable[var]
            # A tied |SMD| happens exactly for a 2-level categorical (its two
            # one-hot indicators are complements, so their SMDs are exact
            # negatives) -- prefer the positive-direction level ("this
            # cluster is enriched for category X") over the negative one
            # ("depleted of the complementary category") as the more
            # directly interpretable characterization.
            if abs(val) > abs(cur_val) or (abs(val) == abs(cur_val) and val > cur_val):
                best_per_variable[var] = (level, val)

        ranked = sorted(best_per_variable.items(), key=lambda kv: abs(kv[1][1]), reverse=True)[:top_n]
        for rank, (var, (level, val)) in enumerate(ranked, start=1):
            rows.append({"cluster": cl, "rank": rank, "variable": var, "level": level, "smd": float(val)})
    return pd.DataFrame(rows)


def bootstrap_ari_mean(matrix, categorical_cols, fit_func, k, base_labels, n_boot, seed=SEED):
    """Axis 2: mean bootstrap-resampling ARI. Reuses
    `03_stability.bootstrap_stability_generic` (the generalised version this
    dispatcher relies on) -- the resampling logic is not reimplemented in
    this script. Even when `fit_func` is `fit_gower_pam`, `precomputed_distance`
    is not passed here -- each bootstrap replicate is a different row
    subset, so the full distance matrix cannot be reused, and it must be
    honestly recomputed fresh on every replicate (see the module docstring).
    """
    if n_boot <= 0:
        return float("nan")

    def fit_fn(sub_matrix, s):
        return fit_func(sub_matrix, categorical_cols, k, seed=s)

    ari_list, _jac = stability_mod.bootstrap_stability_generic(
        matrix, base_labels, fit_fn, n_boot=n_boot, seed=seed,
    )
    return float(np.mean(ari_list)) if ari_list else float("nan")


def build_comparison_row(
    method_name, fit_func, matrix, categorical_cols, k, admission_ids,
    embedding_labels_df, published_labels_df, imputed_matrix, out_dir,
    dataset_path=DATASET, seed=SEED, n_boot=0, precomputed_distance=None,
):
    """Computes all five axes for a single (method, k) combination, writes
    the top-3-feature CSV to `out_dir`, and builds one row of
    `comparison_table.csv`. Both real use (`main()`) and the tests share
    this function, so the assembly logic is not duplicated in two places.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fit_kwargs = {"seed": seed}
    if method_name == "gower_pam" and precomputed_distance is not None:
        fit_kwargs["precomputed_distance"] = precomputed_distance

    t0 = time.time()
    labels = np.asarray(fit_func(matrix, categorical_cols, k, **fit_kwargs))
    fit_sec = time.time() - t0

    n_clusters_actual = int(len(set(labels.tolist())))

    ari_vs_embedding = ari_against_reference(admission_ids, labels, embedding_labels_df, "label")
    ari_vs_published = (
        ari_against_reference(admission_ids, labels, published_labels_df, "phenotype_cluster")
        if published_labels_df is not None else float("nan")
    )

    auroc_mean, auroc_std = evaluate_prognostic_separation(
        labels, admission_ids, dataset_path=dataset_path, seed=seed,
    )

    top3 = characteristic_features_per_cluster(imputed_matrix, categorical_cols, labels)
    top_features_filename = f"top_features_{method_name}_k{k}.csv"
    top3.to_csv(out_dir / top_features_filename, index=False)

    t1 = time.time()
    bootstrap_ari = bootstrap_ari_mean(matrix, categorical_cols, fit_func, k, labels, n_boot, seed=seed)
    bootstrap_sec = time.time() - t1

    return {
        "method": method_name,
        "k": k,
        "n_clusters_actual": n_clusters_actual,
        "bootstrap_ari_mean": bootstrap_ari,
        "n_boot_used": n_boot,
        "ari_vs_embedding": ari_vs_embedding,
        "ari_vs_published_paper": ari_vs_published,
        "auroc_90d_mortality_mean": auroc_mean,
        "auroc_90d_mortality_std": auroc_std,
        "top_features_path": top_features_filename,
        "fit_wall_time_sec": round(fit_sec, 3),
        "bootstrap_wall_time_sec": round(bootstrap_sec, 3),
    }


def _df_to_markdown_table(df: pd.DataFrame) -> str:
    """Builds a markdown pipe table using plain pandas, without
    `pandas.DataFrame.to_markdown` (the `tabulate` package is not installed
    in this environment).
    """
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = "\n".join(
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    )
    return "\n".join([header, sep, body])


def build_diagnosis_markdown(rows_df: pd.DataFrame, embedding_config_tag: str,
                              k_values: list, bootstrap_k: int, n_boot: dict) -> str:
    """Generates the body of DIAGNOSIS.md from the actual results
    (`rows_df`) -- the values themselves are always read from `rows_df` and
    never hardcoded here. The "per-axis max/min" section is purely
    descriptive, and per a prior review comment's guidance does not
    interpret any method/pipeline as "superior".
    """
    lines = [
        "# Task 5c: 5-axis comparison table across the five structured-data clustering methods",
        "",
        f"Compared methods: k-prototypes, FAMD+k-means, GMM (structured variables), "
        f"Gower distance+PAM, latent class analysis. k in {k_values} "
        f"(the value matching the originally derived cluster count, plus one "
        f"context value from the k-grid shared by 5a/5b/Task 2b). Embedding-derived "
        f"reference labels: "
        f"`{embedding_config_tag}` (out/02_recluster). The bootstrap (axis 2) was "
        f"computed only at k={bootstrap_k} (`N_BOOT`={n_boot} -- scaled per method "
        f"by measured cost; see task-5c-report.md for the detailed rationale).",
        "",
        "## Important -- evaluation criterion (prior review comment)",
        "",
        '"difference alone does not imply better clinical validity." '
        "Nowhere in this document is a difference in ARI (axis 2/3) used as "
        "grounds to claim that any clustering method or pipeline is "
        "'superior'. What follows is a factual, side-by-side presentation of "
        "the five axes; if there is an axis on which the embedding-based "
        "pipeline does not come out ahead, that is not written as a win.",
        "",
        "## Results",
        "",
        _df_to_markdown_table(rows_df),
        "",
        "## Per-axis factual summary (not a superiority judgement)",
        "",
    ]

    def _fact_line(col, label):
        sub = rows_df.dropna(subset=[col])
        if sub.empty:
            return f"- **{label}**: no computed value."
        top = sub.loc[sub[col].idxmax()]
        bot = sub.loc[sub[col].idxmin()]
        return (
            f"- **{label}**: max {top[col]:.4f} ({top['method']}, k={top['k']}); "
            f"min {bot[col]:.4f} ({bot['method']}, k={bot['k']}). "
            "(The numeric difference itself is not interpreted as grounds for superiority.)"
        )

    lines.append(_fact_line("bootstrap_ari_mean", "Mean bootstrap ARI (stability, axis 2)"))
    lines.append(_fact_line("ari_vs_embedding", "ARI vs. embedding-derived labels (axis 3)"))
    if rows_df["ari_vs_published_paper"].notna().any():
        lines.append(_fact_line("ari_vs_published_paper", "ARI vs. originally derived labels (axis 3)"))
    lines.append(_fact_line("auroc_90d_mortality_mean", "90-day mortality prognostic-separation AUROC (axis 4)"))

    lines += [
        "",
        "## Limitations and caveats",
        "",
        "- **The low bootstrap replicate count (3) for k-prototypes/Gower+PAM is "
        "an explicit compromise forced by computational limits.** Both methods "
        "must pay an O(n^2)-scale cost again from scratch on every bootstrap "
        "replicate (k-prototypes: n_init=10 repeated fits; Gower+PAM: distance "
        "matrix recomputation), so using the same number of replicates as the "
        "other three methods (15-20) would inflate total runtime to an "
        "unmanageable level (see task-5c-report.md). The mean based on 3 "
        "replicates is indicative only and should not be read as a stable "
        "estimate.",
        "- **The bootstrap was computed only at k=" + str(bootstrap_k) + ".** "
        "`bootstrap_ari_mean` for other k value(s) is NaN -- this means it was "
        "not computed, not that it is zero or unstable.",
        "- GMM (structured) handles categorical variables only via one-hot "
        "encoding, which treats higher-cardinality categorical variables more "
        "crudely than k-prototypes (matching distance) / FAMD (indicator-matrix "
        "structure) / Gower (equal per-column weighting) (see the "
        "`05_baseline_comparators.fit_gmm_structured` docstring).",
        "- All methods share the same `impute_structured_matrix` missing-value "
        "imputation policy (numeric = per-column median, categorical = a "
        "dedicated `__missing__` category) -- this is so that differences "
        "between methods are not confounded with differences in imputation "
        "policy, but it also inherits 5a's caveat that, for columns with heavy "
        "missingness (SOFA/GCS sub-scores, 87-95% missing), the imputed value "
        "itself can artificially influence cluster structure / axis-5 feature "
        "rankings.",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--methods", nargs="+", default=list(METHOD_FIT_FUNCS),
                     choices=list(METHOD_FIT_FUNCS))
    ap.add_argument("--k-values", nargs="+", type=int, default=K_VALUES)
    ap.add_argument("--bootstrap-k", type=int, default=BOOTSTRAP_K)
    ap.add_argument("--skip-bootstrap", action="store_true",
                     help="skip the axis-2 bootstrap (for quick checks only; not for real use)")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[main] loading frame + structured matrix", flush=True)
    frame = frame_mod.load_frame(verify_reports=False)
    matrix, categorical_cols = comparators.build_structured_matrix(frame)
    imputed = comparators.impute_structured_matrix(matrix, categorical_cols)
    admission_ids = frame["admission_ID"].to_numpy()

    embedding_labels_df = pd.read_parquet(EMBEDDING_LABELS_PATH)
    published_labels_df = frame[["admission_ID", "phenotype_cluster"]]

    rows = []
    for method_name in args.methods:
        fit_func = METHOD_FIT_FUNCS[method_name]

        precomputed_distance = None
        if method_name == "gower_pam":
            t0 = time.time()
            precomputed_distance = comparators.compute_gower_matrix(matrix, categorical_cols)
            print(f"[gower_pam] distance matrix computed once in {time.time() - t0:.1f}s, "
                  f"reused across k={args.k_values}", flush=True)

        for k in args.k_values:
            do_bootstrap = (not args.skip_bootstrap) and (k == args.bootstrap_k)
            n_boot = N_BOOT.get(method_name, DEFAULT_N_BOOT) if do_bootstrap else 0
            print(f"[{method_name}] k={k} starting (n_boot={n_boot})", flush=True)

            t0 = time.time()
            row = build_comparison_row(
                method_name, fit_func, matrix, categorical_cols, k, admission_ids,
                embedding_labels_df, published_labels_df, imputed, out_dir=OUT,
                dataset_path=DATASET, seed=args.seed, n_boot=n_boot,
                precomputed_distance=precomputed_distance,
            )
            print(f"[{method_name}] k={k} done in {time.time() - t0:.1f}s: "
                  f"{json.dumps(row, default=str)}", flush=True)
            rows.append(row)
            # incremental persistence (Task 2b's established convention) so a
            # later method/k failure does not lose already-computed rows.
            pd.DataFrame(rows).to_csv(OUT / "comparison_table.csv", index=False)

    rows_df = pd.DataFrame(rows)
    diagnosis = build_diagnosis_markdown(
        rows_df, EMBEDDING_CONFIG_TAG, args.k_values, args.bootstrap_k, N_BOOT,
    )
    (OUT / "DIAGNOSIS.md").write_text(diagnosis)

    manifest = {
        "k_values": args.k_values,
        "bootstrap_k": args.bootstrap_k,
        "n_boot": N_BOOT,
        "embedding_config_tag": EMBEDDING_CONFIG_TAG,
        "seed": args.seed,
        "methods": args.methods,
        "n_rows": int(len(frame)),
        "libraries": recl._library_versions(),
        "threads": recl._thread_config(),
    }
    (OUT / "05c_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
