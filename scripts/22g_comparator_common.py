#!/usr/bin/env python3
"""Task 22 Task C -- shared utilities for completing the five-method
structured-variable comparator table (`out/05_comparators/comparison_table.csv`,
which has only 2 rows, both `gower_pam`, with `bootstrap_ari_mean` computed at
n_boot=3 and 0).

Reuses, via importlib, UNMODIFIED: `01_build_analysis_frame.load_frame`,
`05_baseline_comparators.{build_structured_matrix, impute_structured_matrix,
fit_kprototypes, fit_famd_kmeans, fit_gmm_structured, fit_gower_pam,
compute_gower_matrix}`, and `05c_comparison_table.{ari_against_reference,
characteristic_features_per_cluster}` (those two are corpus/method-agnostic
pure functions, unaffected by the two fixes made here). Does NOT import or
call `05c_comparison_table.evaluate_prognostic_separation` (plain
`StratifiedKFold`) or `03_stability.bootstrap_stability_generic`/
`05c_comparison_table.bootstrap_ari_mean` (row-level `rng.choice` resampling)
-- both are replaced below because the task requires patient-grouped CV and
patient-level bootstrap, and this repo's own convention is "add a new
function under a new name, never mutate a reviewed one that other completed
tasks still depend on byte-identically."

## Fix 1 -- AUROC: StratifiedKFold -> StratifiedGroupKFold(5, patient_ID)

`05c_comparison_table.evaluate_prognostic_separation` used a plain
`StratifiedKFold`, which lets a patient's several admissions land in both
the train and test fold of the SAME cluster-label-only classifier -- a
patient-overlap leak. `evaluate_prognostic_separation_grouped` below is
otherwise identical (same feature construction: one-hot of the label,
logistic regression, same "read only `90d_mortality`" outcome-blind scope)
but groups splits by `patient_ID`.

## Fix 2 -- bootstrap: row-level rng.choice -> patient-level resample

`bootstrap_stability_generic`/`bootstrap_ari_mean` resample INDIVIDUAL ROWS
with replacement (`rng.choice(n, n, replace=True)`), which is wrong for data
with repeated admissions per patient (this project's own established
gotcha: "resample distinct patient_ID, keep all their admissions -- row-
level is wrong and audits check for it"). `patient_bootstrap_ari` below
mirrors the SAME algorithm shape (resample-with-replacement at the
resampling unit, then fit only on the resulting UNIQUE set -- exactly
`bootstrap_stability_generic`'s convention, generalised from rows to
patients) so the "~0.632 retained" behaviour and ARI-vs-base-labels
convention are unchanged, only the resampling unit is corrected.

For `gower_pam`, `patient_bootstrap_ari` exploits that Gower distance is a
fixed pairwise function of the ORIGINAL rows: a bootstrap resample is just a
row-index array into those same fixed points, so the resampled distance
matrix is `D_full[np.ix_(idx, idx)]` -- no O(n^2) recomputation per
replicate. This is the same `precomputed_distance` mechanism 05c already
uses for the point-estimate k-grid, extended here to the bootstrap loop
(05c's own docstring explicitly calls this out as NOT done for its
bootstrap, "each bootstrap replicate is a different row subset, so the full
matrix can't be reused" -- true for a raw row-level bootstrap where the
subsample is arbitrary, but NOT true here: `idx` always indexes into the
SAME fixed 32,329 original points, so the submatrix is a valid extraction,
not a recomputation).

Outcome-blind (Global Constraints): `30d_mortality`, `in_hospital_mortality`,
`death_date`, `discharge_date` are never read anywhere in this module.
`90d_mortality` is read only inside `evaluate_prognostic_separation_grouped`.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
frame_mod = importlib.import_module("01_build_analysis_frame")
comparators = importlib.import_module("05_baseline_comparators")
comp05c = importlib.import_module("05c_comparison_table")  # ari_against_reference,
                                                              # characteristic_features_per_cluster (reused as-is)

ROOT = Path("/data/tcmhwd/AI/AST_ai")
REV = ROOT / "phenotype_revision"
OUT = REV / "out/22_published_objects/task_c_comparators"
DATASET = comparators.DATASET
GOWER_MATRIX_PATH = OUT / "gower_distance_matrix.npy"
GOWER_ADMISSION_IDS_PATH = OUT / "gower_admission_ids.csv"

EMBEDDING_CONFIG_TAG = comp05c.EMBEDDING_CONFIG_TAG
EMBEDDING_LABELS_PATH = comp05c.EMBEDDING_LABELS_PATH
MORTALITY_COL = comp05c.MORTALITY_COL
K_VALUES = [16, 32]
SEED = 42
N_BOOT_TARGET = 100

METHOD_FIT_FUNCS = {
    "kprototypes": comparators.fit_kprototypes,
    "famd_kmeans": comparators.fit_famd_kmeans,
    "gmm_structured": comparators.fit_gmm_structured,
    "gower_pam": comparators.fit_gower_pam,
    "lca": comparators.fit_lca,
}


def load_common():
    """Frame + structured matrix + imputed matrix + admission_ID/patient_ID
    arrays, shared by every stage (point estimate, bootstrap, AUROC)."""
    frame = frame_mod.load_frame(verify_reports=False)
    matrix, categorical_cols = comparators.build_structured_matrix(frame)
    imputed = comparators.impute_structured_matrix(matrix, categorical_cols)
    admission_ids = frame["admission_ID"].to_numpy()
    patient_ids = frame["patient_ID"].to_numpy()
    return frame, matrix, categorical_cols, imputed, admission_ids, patient_ids


def load_gower_matrix() -> np.ndarray:
    """Memory-mapped load of the once-computed full-cohort Gower distance
    matrix (`22g0_gower_precompute.py`). Raises with a clear message if the
    precompute step has not been run yet."""
    if not GOWER_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"{GOWER_MATRIX_PATH} not found -- run 22g0_gower_precompute.py first"
        )
    return np.load(GOWER_MATRIX_PATH, mmap_mode="r")


def assert_gower_row_order_matches(admission_ids: np.ndarray) -> None:
    saved = pd.read_csv(GOWER_ADMISSION_IDS_PATH)["admission_ID"].to_numpy()
    if not np.array_equal(saved, admission_ids):
        raise RuntimeError(
            "admission_ID row order of the precomputed Gower matrix does not match "
            "load_common()'s frame order -- do not reuse the matrix"
        )


def evaluate_prognostic_separation_grouped(labels, admission_ids, patient_ids,
                                            dataset_path=DATASET, seed=SEED):
    """Axis 4, FIXED: `StratifiedGroupKFold(5, groups=patient_ID)` instead of
    plain `StratifiedKFold` (see module docstring "Fix 1"). Otherwise
    identical to `05c_comparison_table.evaluate_prognostic_separation`:
    one-hot of the cluster label as the sole feature set, logistic
    regression, mean/std AUROC across folds. Reads `90d_mortality` only.
    """
    outcome = pd.read_csv(dataset_path, usecols=["admission_ID", MORTALITY_COL], low_memory=False)
    joined = pd.DataFrame({"admission_ID": admission_ids, "label": labels, "patient_ID": patient_ids}).merge(
        outcome, on="admission_ID", how="left", validate="one_to_one",
    )
    assert joined[MORTALITY_COL].notna().all(), (
        f"{MORTALITY_COL} missing for some admission_ID after join -- dataset_path/admission_ids mismatch"
    )

    X = pd.get_dummies(joined["label"].astype(str)).to_numpy(dtype=float)
    y = joined[MORTALITY_COL].to_numpy()
    groups = joined["patient_ID"].to_numpy()

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for train_idx, test_idx in sgkf.split(X, y, groups):
        assert len(set(groups[train_idx].tolist()) & set(groups[test_idx].tolist())) == 0, (
            "StratifiedGroupKFold leaked a patient_ID across train/test"
        )
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))
    return float(np.mean(aucs)), float(np.std(aucs))


def patient_bootstrap_indices(patient_ids: np.ndarray, seed: int) -> np.ndarray:
    """Resample-with-replacement AT THE PATIENT LEVEL, then keep every row of
    every patient actually drawn (each drawn patient contributes ALL of
    their admissions, once each -- no row-level duplication), mirroring
    `03_stability.bootstrap_stability_generic`'s own "resample -> dedupe ->
    fit on the unique set" shape, generalised from rows to patients (see
    module docstring "Fix 2"). Retains ~63.2% of patients in expectation,
    matching this project's established f=0.632 convention elsewhere."""
    uniq_patients = np.unique(patient_ids)
    n_p = len(uniq_patients)
    rng = np.random.default_rng(seed)
    draw = rng.choice(n_p, size=n_p, replace=True)
    selected = uniq_patients[np.unique(draw)]
    row_mask = np.isin(patient_ids, selected)
    return np.where(row_mask)[0]


def patient_bootstrap_ari(imputed, categorical_cols, fit_func, method_name, k,
                           base_labels, patient_ids, n_boot, seed=SEED,
                           full_gower_matrix=None):
    """Axis 2, FIXED: patient-level bootstrap (see module docstring "Fix 2"),
    generalising `bootstrap_stability_generic`'s algorithm shape. For
    `gower_pam` with `full_gower_matrix` given, reuses the precomputed
    full-cohort distance matrix via `np.ix_` submatrix extraction instead of
    recomputing Gower distance per replicate.

    Returns (list of per-replicate ARI, n_boot actually computed).
    """
    if n_boot <= 0:
        return [], 0
    ari_list = []
    for b in range(n_boot):
        idx = patient_bootstrap_indices(patient_ids, seed=seed + b)
        if method_name == "gower_pam" and full_gower_matrix is not None:
            sub_distance = np.asarray(full_gower_matrix[np.ix_(idx, idx)])
            new_labels = fit_func(imputed.iloc[idx], categorical_cols, k, seed=seed + b,
                                   precomputed_distance=sub_distance)
        else:
            new_labels = fit_func(imputed.iloc[idx], categorical_cols, k, seed=seed + b)
        new_labels = np.asarray(new_labels)
        ari_list.append(float(adjusted_rand_score(base_labels[idx], new_labels)))
    return ari_list, n_boot


def ari_against_reference(admission_ids, labels, reference_df, reference_col):
    return comp05c.ari_against_reference(admission_ids, labels, reference_df, reference_col)


def characteristic_features_per_cluster(imputed_matrix, categorical_cols, labels, top_n=3):
    return comp05c.characteristic_features_per_cluster(imputed_matrix, categorical_cols, labels, top_n=top_n)
