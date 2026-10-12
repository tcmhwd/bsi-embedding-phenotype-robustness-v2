#!/usr/bin/env python3
"""Task 23f -- shared utilities for the neighbourhood confounding-by-
indication DIAGNOSTIC (added mid-task, by explicit request): if retrieval
is useful, the natural clinician-facing display is "of this patient's 25
most similar past cases, N received drug X and had mortality m1, M did not
and had mortality m0" -- and that display is misleading by INDICATION
confounding (sicker patients get broader empiric therapy, so a mortality
gap inside a neighbourhood reflects severity imbalance, not drug effect).
This module quantifies that risk; it does NOT estimate a treatment effect.

**This is a DESCRIPTIVE balance diagnostic, not a causal analysis.**
Nothing computed here should be read as evidence for or against any
treatment's effect -- only as "how imbalanced, on measured severity, are
the two treatment-defined subgroups inside a retrieved neighbourhood."

**Where the outcome enters (stated once):** `90d_mortality` is read ONLY
to report the same-neighbourhood mortality-rate DIFFERENCE between the two
treatment-defined subgroups, purely descriptively, alongside the balance
number -- never to select, weight, or exclude neighbours (neighbour
retrieval itself is the UNMODIFIED mechanism from `23a_retrieval_common.py`
/ `23b_run_retrieval.py`: same fold construction, same "train fold only,
never self, never same patient" pool, same `unrestricted`/`past_only`
variants).

**Treatment-split variables** (from `pipeline_new/out_pre/admission_level_base.csv`,
read-only reference data per this repo's README; all three exist, complete,
no missingness, verified below): `antipseudoab_empiric` (empiric
anti-pseudomonal coverage), `carbaab_empiric` (empiric carbapenem),
`antiMRSAab_empiric` (empiric anti-MRSA coverage). Each is its own,
separate binary split -- never combined into one "any broad-spectrum"
composite.

**Balance covariates**: `sofa_score`/`charlson_score` (imputed via
`05_baseline_comparators.impute_structured_matrix` -- the SAME median
imputation already applied to these columns inside the 44-variable
structured comparator matrix and the SOFA/Charlson anchor rows, not
re-derived here), `icu`, `ventilator`, `Vasopressin_any` (all complete,
no imputation needed), `age_at_test` (complete).

**[dense-variant fix]** The `sofa_score` covariate reported here is sourced
from `impute_structured_matrix`'s `sofa_score_2` column (dense twin, 0.2%
missing, 100% agreement with the sparse `sofa_score` wherever both are
present -- see `05_baseline_comparators.py` docstring point 2 and
`scripts/24_verify_dense_pairs.py`), not the sparse `sofa_score` (88.3%
missing) this module used before. The output column is still NAMED
`sofa_score` for report continuity; only its source changed.

**SMD**: standardized mean difference, `(mean(treated) - mean(control)) /
sqrt((var(treated) + var(control)) / 2)`, population variance (ddof=0) so
a neighbourhood subgroup of size 1 still has a defined (zero) variance
rather than raising -- appropriate here because subgroups are often small
(k in {10, 25, 50}). The SAME formula applied to a 0/1 covariate (icu,
ventilator, Vasopressin_any) reduces algebraically to the standard
proportion-difference SMD (var of a Bernoulli(p) is p(1-p)), so one
function handles both continuous and binary covariates without a special
case. Reported as |SMD|, per the field's convention that only the
magnitude of imbalance matters for a balance check (0.1 = conventional
"acceptable balance" threshold, 0.25 = a commonly used "seriously
imbalanced" threshold -- both reported, not just one).
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c23 = importlib.import_module("23a_retrieval_common")

ROOT = c23.ROOT
ADMISSION_LEVEL_BASE_PATH = ROOT / "pipeline_new/out_pre/admission_level_base.csv"

TREATMENT_SPLITS = ["antipseudoab_empiric", "carbaab_empiric", "antiMRSAab_empiric"]
BALANCE_COVARIATES = ["sofa_score", "charlson_score", "icu", "ventilator", "Vasopressin_any", "age_at_test"]
BALANCE_K_VALUES = [10, 25, 50]  # 25 is the primary/main analysis, per request
BALANCE_REPRESENTATIONS = ["openai-3large", "gower"]
SMD_THRESHOLDS = [0.1, 0.25]

OUT = c23.OUT


def load_balance_covariates(master: pd.DataFrame) -> pd.DataFrame:
    """One row per `master["admission_ID"]` (same order), columns:
    `sofa_score`, `charlson_score` (imputed), `icu`, `ventilator`,
    `Vasopressin_any`, `age_at_test` (raw, complete), and the three
    treatment-split flags (raw, complete). Joined by admission_ID only,
    never row position; asserts completeness after every join rather than
    assuming it."""
    matrix, categorical_cols = c23.comparators.build_structured_matrix(master[["admission_ID"]])
    imputed = c23.comparators.impute_structured_matrix(matrix, categorical_cols)

    out = pd.DataFrame({"admission_ID": master["admission_ID"].to_numpy()})
    # dense-variant fix: imputed["sofa_score_2"] (dense), NOT imputed["sofa_score"]
    # (sparse, no longer a column of `imputed` after the NUMERIC_COLS swap in
    # 05_baseline_comparators.py) -- kept named "sofa_score" here for report
    # continuity, see module docstring "[dense-variant fix]" above.
    out["sofa_score"] = imputed["sofa_score_2"].to_numpy()
    out["charlson_score"] = imputed["charlson_score"].to_numpy()

    raw = pd.read_csv(
        c23.DATASET, usecols=["admission_ID", "icu", "ventilator", "Vasopressin_any", "age_at_test"],
        low_memory=False,
    )
    out = out.merge(raw, on="admission_ID", how="left", validate="one_to_one")

    tx = pd.read_csv(
        ADMISSION_LEVEL_BASE_PATH, usecols=["admission_ID"] + TREATMENT_SPLITS, low_memory=False,
    )
    out = out.merge(tx, on="admission_ID", how="left", validate="one_to_one")

    assert len(out) == len(master), "covariate join changed row count"
    for c in out.columns:
        if c == "admission_ID":
            continue
        assert out[c].notna().all(), f"{c} missing after join -- covariate/master population mismatch"
    for c in TREATMENT_SPLITS + ["icu", "ventilator", "Vasopressin_any"]:
        vals = set(out[c].unique().tolist())
        assert vals <= {0, 1}, f"{c} is not a clean 0/1 flag: {vals}"
    return out


def standardized_mean_diff_vec(mean1: np.ndarray, var1: np.ndarray, mean0: np.ndarray, var0: np.ndarray) -> np.ndarray:
    """Vectorized SMD: (mean1-mean0) / sqrt((var1+var0)/2), population
    variance. Where the pooled SD is ~0 (a covariate that happens to be
    constant within BOTH subgroups of a tiny neighbourhood), returns 0.0
    if the means also match (both fully degenerate, no imbalance) and NaN
    otherwise (imbalance exists but is not expressible as a standardized
    ratio -- reported as missing, never silently coerced to +-inf)."""
    denom = np.sqrt((var1 + var0) / 2.0)
    diff = mean1 - mean0
    near_zero = denom < 1e-12
    equal_means = np.abs(diff) < 1e-12
    with np.errstate(invalid="ignore", divide="ignore"):
        smd = diff / denom
    smd = np.where(near_zero, np.where(equal_means, 0.0, np.nan), smd)
    return smd


def group_mean_var(values: np.ndarray, group_mask: np.ndarray):
    """values, group_mask: (n_test, k). Returns (mean, var, n) per row,
    over only the `group_mask`-True entries of each row (population
    variance, ddof=0). Rows with n=0 get mean=var=nan."""
    n = group_mask.sum(axis=1)
    n_safe = np.maximum(n, 1)
    s = np.where(group_mask, values, 0.0).sum(axis=1)
    mean = s / n_safe
    ssq = np.where(group_mask, values ** 2, 0.0).sum(axis=1)
    var = ssq / n_safe - mean ** 2
    var = np.maximum(var, 0.0)  # guard tiny negative floating-point residue
    mean = np.where(n > 0, mean, np.nan)
    var = np.where(n > 0, var, np.nan)
    return mean, var, n
