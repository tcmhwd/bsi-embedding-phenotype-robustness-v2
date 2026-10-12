#!/usr/bin/env python3
"""Task 23 -- shared utilities for the constructive retrieval result: does
the SAME embedding that fails at phenotyping partitioning (Task 22 Task C:
`published_16` AUROC 0.6206 vs five structured comparators 0.678-0.779, all
on the identical N=32,329 population/CV) instead work for CASE RETRIEVAL?

Retrieval definition (identical shape for every representation): for an
index admission, take its k nearest neighbours in a representation-specific
distance/similarity space, restricted to a candidate pool that is (a) the
TRAINING FOLD of a StratifiedGroupKFold(5, groups=patient_ID) split only --
so it never contains the index admission itself and never contains another
admission of the same patient (both guaranteed structurally by the disjoint
train/test index sets and the group-fold guarantee, and additionally
verified at runtime in `23b_run_retrieval.py`, not just assumed) -- and,
for the `past_only` variant, (b) further restricted to admissions whose
`test_date` (index-culture date) is STRICTLY EARLIER than the query's own
`test_date` (ties on the same date are excluded, not treated as "past").
The mean `90d_mortality` of the retrieved neighbours is the SOLE predictor
of the index admission's own `90d_mortality`; AUROC is computed per test
fold then averaged (mean/std across folds).

**Where outcome data enters (stated once, applies everywhere in this
module and its callers):** `90d_mortality` is read ONLY (a) as the label
being predicted (`y`) and (b) as the values being AVERAGED across a
query's neighbours to form that query's own risk score -- never an
admission's own outcome scoring itself, since the neighbour pool for a
test-fold query is always the disjoint training fold. No other
outcome/discharge column (`30d_mortality`, `in_hospital_mortality`,
`death_date`, `discharge_date`) is read anywhere in this module. The
representations themselves (embeddings, Gower distance) are outcome-blind
by construction (built upstream of this task, from case-report text or
structured clinical variables only).

**CV scheme, reused not reimplemented:** `build_folds` calls
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` on
(y=90d_mortality, groups=patient_ID) -- the IDENTICAL construction
`22g_comparator_common.evaluate_prognostic_separation_grouped` uses
internally to produce the stored `point_published_16.json` AUROC
(0.6205749882834398). `StratifiedGroupKFold.split()` only uses its `X`
argument for `len(X)` (verified against sklearn 1.7.2 source -- the split
algorithm depends only on `y`/`groups`/`n_splits`/`shuffle`/`random_state`),
so as long as (y, groups) share the same admission_ID order as that
function's internal join -- guaranteed here because `build_master_frame`
uses the SAME `01_build_analysis_frame.load_frame(verify_reports=False)`
call that `22g_comparator_common.load_common()` and `22g0_gower_precompute.py`
both use -- the folds this module builds are BYTE-IDENTICAL to the ones
that produced 0.6206. `sanity_check_published_16_reproduces_0_6206` proves
this at runtime (this repo's own documented gotcha: "a patch here once
silently failed to match and ran with a different config without
erroring") rather than assuming it; every point-estimate script below
calls it and prints/asserts the result before doing any retrieval work.

**Master population/order:** `build_master_frame()` returns
`01_build_analysis_frame.load_frame(verify_reports=False)`'s admission_ID/
patient_ID/phenotype_cluster (N=32,329, canonical Task-22 order), plus
`test_date` and `90d_mortality` joined on by `admission_ID` (never by row
position). Every representation loaded below (three embeddings, one
precomputed Gower matrix) is reindexed onto this ONE order before any
comparison -- `load_representation_vectors` raises if any admission_ID is
missing, and `load_gower_matrix_checked` asserts the stored matrix's own
row order (`22g_comparator_common.assert_gower_row_order_matches`) matches
it exactly rather than trusting the two happen to agree.

Writes nothing itself -- only under `phenotype_revision/out/23_retrieval/`,
and only by callers (`23b_run_retrieval.py`, `23c_run_anchors.py`,
`23e_assemble_table.py`).
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path("/data/tcmhwd/AI/AST_ai")
REV = ROOT / "phenotype_revision"
OUT = REV / "out/23_retrieval"
DATASET = ROOT / "manuscript/phenotype_ml_dataset.csv"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
frame_mod = importlib.import_module("01_build_analysis_frame")
comparators = importlib.import_module("05_baseline_comparators")
c22g = importlib.import_module("22g_comparator_common")
c18 = importlib.import_module("18_common")

MORTALITY_COL = c22g.MORTALITY_COL  # "90d_mortality"
SEED = c22g.SEED  # 42
CV_SPLITS = 5
K_VALUES = [5, 10, 25, 50]
K_MAX = max(K_VALUES)
VARIANTS = ["unrestricted", "past_only"]
REPRESENTATIONS = ["openai-3large", "openai-3small", "bge-m3", "gower"]

PUBLISHED_16_POINT_PATH = c22g.OUT / "point_published_16.json"

# Vector-form representations (everything except "gower", which is a
# precomputed pairwise distance matrix, not a vector space). openai-3large
# reads the STORED out_pre embedding directly via `18_common` (same model
# that produced the published clusters); the other two read Task 18's own
# Step-1 embedding output.
ENCODER_EMBED_PATHS = {
    "bge-m3": REV / "out/18_encoder_dec/embeddings/bge-m3/merged.npy",
    "openai-3small": REV / "out/18_encoder_dec/embeddings/openai-3small/merged.npy",
}
ENCODER_IDS_PATHS = {
    "bge-m3": REV / "out/18_encoder_dec/embeddings/bge-m3/merged_ids.json",
    "openai-3small": REV / "out/18_encoder_dec/embeddings/openai-3small/merged_ids.json",
}


def env_manifest() -> dict:
    return {
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


def build_master_frame() -> pd.DataFrame:
    """admission_ID, patient_ID, phenotype_cluster (canonical Task-22 order,
    via `load_frame`) + test_date + 90d_mortality, the last two joined onto
    that order by admission_ID (never by row position). This is the ONE
    population/order every representation in this module is reindexed to."""
    frame = frame_mod.load_frame(verify_reports=False)[
        ["admission_ID", "patient_ID", "phenotype_cluster"]
    ].copy()
    extra = pd.read_csv(DATASET, usecols=["admission_ID", "test_date", MORTALITY_COL], low_memory=False)
    merged = frame.merge(extra, on="admission_ID", how="left", validate="one_to_one")
    assert len(merged) == len(frame), "join changed row count -- DATASET has duplicate admission_ID?"
    assert merged[MORTALITY_COL].notna().all(), f"{MORTALITY_COL} missing for some admission_ID after join"
    assert merged["test_date"].notna().all(), "test_date missing for some admission_ID after join"
    merged["test_date"] = pd.to_datetime(merged["test_date"])
    merged["test_date_ordinal"] = merged["test_date"].to_numpy("datetime64[D]").astype(np.int64)
    return merged


def build_folds(y: np.ndarray, groups: np.ndarray, seed: int = SEED, n_splits: int = CV_SPLITS):
    """IDENTICAL `StratifiedGroupKFold` call (same n_splits/shuffle/seed) to
    `22g_comparator_common.evaluate_prognostic_separation_grouped` -- see
    module docstring for why this reproduces byte-identical folds. Asserts
    the no-patient-leak / disjoint-index guarantees explicitly rather than
    trusting sklearn's contract silently."""
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(skf.split(np.zeros(len(y)), y, groups))
    for train_idx, test_idx in folds:
        assert len(set(groups[train_idx].tolist()) & set(groups[test_idx].tolist())) == 0, (
            "StratifiedGroupKFold leaked a patient_ID across train/test"
        )
        assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0, (
            "train/test index sets are not disjoint"
        )
    return folds


def sanity_check_published_16_reproduces_0_6206(master: pd.DataFrame, tol: float = 1e-9) -> dict:
    """Runtime proof (not an assumption) that this module's admission_ID/
    patient_ID/order matches the population that produced the stored
    `point_published_16.json` AUROC exactly: recomputes that AUROC via the
    SAME reused function (`22g_comparator_common.evaluate_prognostic_separation_grouped`)
    called on THIS module's own (phenotype_cluster, admission_ID, patient_ID),
    and asserts equality to the stored value within floating-point
    tolerance. This is the runtime check for the repo's documented gotcha
    ("a patch here once silently failed to match and ran with a different
    config without erroring") -- every point-estimate script calls this
    before doing any retrieval work and refuses to proceed if it fails.
    """
    labels = master["phenotype_cluster"].to_numpy()
    admission_ids = master["admission_ID"].to_numpy()
    patient_ids = master["patient_ID"].to_numpy()
    auroc_mean, auroc_std = c22g.evaluate_prognostic_separation_grouped(
        labels, admission_ids, patient_ids, dataset_path=c22g.DATASET, seed=SEED,
    )
    stored = json.loads(PUBLISHED_16_POINT_PATH.read_text())
    ok_mean = abs(auroc_mean - stored["auroc_90d_mortality_mean"]) < tol
    ok_std = abs(auroc_std - stored["auroc_90d_mortality_std"]) < tol
    result = {
        "recomputed_auroc_mean": auroc_mean,
        "recomputed_auroc_std": auroc_std,
        "stored_auroc_mean": stored["auroc_90d_mortality_mean"],
        "stored_auroc_std": stored["auroc_90d_mortality_std"],
        "matches_stored_0_6206": bool(ok_mean and ok_std),
    }
    assert result["matches_stored_0_6206"], (
        f"Fold-construction/population mismatch: recomputed published_16 AUROC "
        f"{auroc_mean} (std {auroc_std}) != stored {stored['auroc_90d_mortality_mean']} "
        f"(std {stored['auroc_90d_mortality_std']}) -- do NOT trust downstream retrieval "
        "numbers until this is fixed"
    )
    return result


def load_representation_vectors(representation: str, admission_ids: np.ndarray) -> np.ndarray:
    """Returns (n, dim) float32 array, rows aligned EXACTLY to
    `admission_ids` order (never by row position). Raises if any requested
    admission_ID is missing from the stored embedding."""
    if representation == "openai-3large":
        meta = c18.load_metadata()
        stored = np.asarray(np.load(c18.OUT_PRE_EMB, mmap_mode="r"))
        return c18.align_embeddings_to_admission_ids(stored, meta["admission_ID"].to_numpy(), admission_ids)
    if representation in ENCODER_EMBED_PATHS:
        vecs = np.load(ENCODER_EMBED_PATHS[representation])
        ids = json.loads(ENCODER_IDS_PATHS[representation].read_text())
        return c18.align_embeddings_to_admission_ids(vecs, ids, admission_ids)
    raise ValueError(
        f"load_representation_vectors: {representation!r} is not a vector representation "
        "(gower has no vector form -- use load_gower_matrix_checked)"
    )


def load_gower_matrix_checked(admission_ids: np.ndarray) -> np.ndarray:
    """Memory-mapped precomputed full-cohort Gower distance matrix, with its
    stored row order asserted (not assumed) to match `admission_ids` --
    never recomputed here (task instruction: reuse the existing matrix)."""
    mat = c22g.load_gower_matrix()
    c22g.assert_gower_row_order_matches(admission_ids)
    return mat


def normalize_rows(X: np.ndarray) -> np.ndarray:
    """L2-row-normalize so a plain dot product equals cosine similarity."""
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def topk_ordered(score_matrix: np.ndarray, valid_mask: np.ndarray, k_max: int):
    """score_matrix, valid_mask: (n_test, n_train). HIGHER score = MORE
    similar (caller negates a distance first, e.g. Gower). Computes the
    top-`k_max` largest-score column (train-local) index per row ONCE via
    `argpartition` + a small per-row sort of just those k_max candidates --
    so slicing the first k<=k_max columns of the result gives the EXACT
    top-k for every k in one pass, no repeated full sort/partition per k.

    Returns (top_idx_sorted, valid_counts):
      - top_idx_sorted: (n_test, k_max) int array of TRAIN-LOCAL column
        indices, descending by score among valid (mask=True) candidates.
        Entries at position >= valid_counts[i] (when valid_counts[i] <
        k_max) are NOT valid neighbours (some are masked-out padding) --
        callers must NEVER read past min(k, valid_counts[i]).
      - valid_counts: (n_test,) count of valid (mask=True) candidates per
        row (may exceed k_max; only capped implicitly when a caller slices
        by k).
    """
    n_train = score_matrix.shape[1]
    k_eff = min(k_max, n_train)
    masked = np.where(valid_mask, score_matrix, -np.inf).astype(np.float32)
    top_unsorted = np.argpartition(masked, n_train - k_eff, axis=1)[:, n_train - k_eff:]
    top_scores = np.take_along_axis(masked, top_unsorted, axis=1)
    order = np.argsort(-top_scores, axis=1)
    top_idx_sorted = np.take_along_axis(top_unsorted, order, axis=1)
    if k_eff < k_max:
        pad = k_max - k_eff
        top_idx_sorted = np.pad(top_idx_sorted, ((0, 0), (0, pad)), constant_values=0)
    valid_counts = valid_mask.sum(axis=1)
    return top_idx_sorted, valid_counts


def neighbour_selection_mask(top_idx_sorted: np.ndarray, valid_counts: np.ndarray, k: int):
    """Generic form of the masking logic inside `neighbour_risk_scores`,
    exposed so callers (e.g. the neighbourhood confounding-by-indication
    diagnostic, `23f_neighborhood_balance_common.py`) can index ANY
    train-side array -- not just `90d_mortality` -- by the SAME
    actual-neighbours-used mask, so "which neighbours were actually
    retrieved for k" is defined in exactly one place.

    Returns (idx_safe, use_mask): `idx_safe` is (n_test, min(k, k_max))
    train-local column indices, safe to index any (n_train,)-shaped array
    (a dummy 0 is substituted at masked-out positions -- caller must gate
    every read through `use_mask`); `use_mask` marks which positions are
    real, retrieved neighbours (a row with fewer than k valid candidates
    has `False` in the trailing positions; a row with zero valid
    candidates is all-`False`)."""
    n_test, k_max = top_idx_sorted.shape
    kk = min(k, k_max)
    cols = np.arange(kk)[None, :]
    actual_k = np.minimum(kk, valid_counts)
    use_mask = cols < actual_k[:, None]
    idx_safe = np.where(use_mask, top_idx_sorted[:, :kk], 0)
    return idx_safe, use_mask


def neighbour_risk_scores(top_idx_sorted: np.ndarray, valid_counts: np.ndarray, y_train: np.ndarray, k: int):
    """Mean 90d_mortality of the (up to) k valid nearest neighbours per
    query row. A row with fewer than k valid candidates uses all it has;
    a row with ZERO valid candidates (only possible in the `past_only`
    variant, for an admission with no earlier same-fold candidate at all)
    gets risk=NaN and is EXCLUDED from that k's AUROC (counted, reported,
    never silently imputed to 0.5 or dropped without a count)."""
    n_test, k_max = top_idx_sorted.shape
    kk = min(k, k_max)
    cols = np.arange(kk)[None, :]
    actual_k = np.minimum(kk, valid_counts)
    use_mask = cols < actual_k[:, None]
    idx_safe = np.where(use_mask, top_idx_sorted[:, :kk], 0)
    y_neighbours = y_train[idx_safe]
    summed = np.where(use_mask, y_neighbours, 0).sum(axis=1)
    counts = use_mask.sum(axis=1)
    risk = np.full(n_test, np.nan, dtype=np.float64)
    has_neighbours = counts > 0
    risk[has_neighbours] = summed[has_neighbours] / counts[has_neighbours]
    return risk, counts


def evaluate_continuous_predictor_grouped(feature_values: np.ndarray, admission_ids: np.ndarray,
                                           patient_ids: np.ndarray, y: np.ndarray, seed: int = SEED):
    """Single continuous clinical predictor (SOFA or Charlson) -> AUROC via
    patient-grouped 5-fold CV. Deliberately a NEW function, not a call to
    `22g_comparator_common.evaluate_prognostic_separation_grouped`: that
    function one-hot-encodes its `labels` argument (`pd.get_dummies`),
    correct for a categorical cluster label but WRONG for a continuous
    score (one-hot-encoding a continuous variable would treat every
    distinct numeric value as its own unrelated category). Otherwise
    identical CV construction (SAME `build_folds` -- same
    `StratifiedGroupKFold(5, groups=patient_ID, seed=42)`), a plain
    LogisticRegression on the single feature column, mean/std AUROC across
    folds."""
    folds = build_folds(y, patient_ids, seed=seed)
    X = np.asarray(feature_values, dtype=float).reshape(-1, 1)
    aucs = []
    for train_idx, test_idx in folds:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))
    return float(np.mean(aucs)), float(np.std(aucs))


def evaluate_feature_matrix_grouped(feature_matrix: np.ndarray, admission_ids: np.ndarray,
                                     patient_ids: np.ndarray, y: np.ndarray, seed: int = SEED):
    """Arm 3 (coordinator-directed, 2026-09-06): generalises
    `evaluate_continuous_predictor_grouped` to an ARBITRARY (n, d) feature
    matrix instead of a single column -- e.g. [sofa_score_2,
    sofa_score_was_measured] together -- via a plain multi-feature
    LogisticRegression. Deliberately a NEW function (never mutates
    `evaluate_continuous_predictor_grouped`, which reshapes its input to a
    single column and would silently corrupt a 2D input). SAME CV
    construction (`build_folds`, same `StratifiedGroupKFold(5,
    groups=patient_ID, seed=42)`) as every other AUROC number in this
    module."""
    folds = build_folds(y, patient_ids, seed=seed)
    X = np.asarray(feature_matrix, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    aucs = []
    for train_idx, test_idx in folds:
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], proba))
    return float(np.mean(aucs)), float(np.std(aucs))
