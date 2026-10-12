#!/usr/bin/env python3
"""Task 5a+5b: a "proper" embedding-independent structured-data clustering comparator.

This module implements all of the methods that (prior review comment) named
specifically by name (Gower distance/PAM, k-prototypes, FAMD, latent class
analysis, mixture models): k-prototypes, FAMD+k-means, and GMM were added in
the 5a dispatch; Gower distance+PAM (`fit_gower_pam`) and latent class
analysis (`fit_lca`) were added in the 5b dispatch. Assembling the five-axis
comparison table (comparison_table.csv) is split off into yet another
dispatch (5c) -- this module only provides the `build_structured_matrix` /
`impute_structured_matrix` shared functions that 5c depends on, plus the
five `fit_*` functions.

**Scalability of `fit_gower_pam` (5b, empirically measured via SLURM at the
full 32,329-row scale -- job 586191):** The Gower distance matrix is O(n^2)
memory, and the `gower` package empirically returns float32 (version 0.1.2,
`gower_get`'s `out = np.zeros(..., dtype=np.float32)`). Measured on the
actual full 32,329 rows: distance-matrix computation took 1640.5s (~27.3
min), size 4,180,656,964 bytes (~4.18GB, exactly matching `32329^2*4`),
`KMedoids(method="alternate", k=8)` fit took 6.98s, peak RSS ~4.96GB
(comfortable margin against the requested 16GB). **Conclusion: feasible at
full scale** -- this is also consistent with the extrapolated estimate
(~28-40 min) from subsamples (n=2,000/5,000/8,000: 5.5s/35.6s/102.3s).
Because the distance matrix only needs to be computed once regardless of k
and can then be reused, if 5c compares multiple k values the additional
cost per extra k is just the KMedoids fit (a few seconds) -- however, that
is 5c's design decision, and this function itself always honestly computes
the full distance matrix fresh every time it is called (no caching).
`KMedoids(method="alternate")` was a negligible, sub-one-second cost at
subsample scale (`method="pam"` already rose sharply from 0.554s to 6.58s
between n=2,000 and n=5,000 -- distinctly steeper than O(n^2) -- and was
judged unable to cope at 32,329 rows, hence `"alternate"` was adopted). See
`task-5b-report.md` for the full measurement process/rationale.

**Key point (the decisive difference from Task 2/2b):** Task 2/2b clustered
LLM embeddings (3,072 dimensions, including the UMAP-reduced space). This
module does not touch that embedding at all, and clusters using only the
structured clinical variables from `manuscript/phenotype_ml_dataset.csv` --
this separation is exactly the point of the "clinically-selected-variable-
based" comparator that a reviewer requested, so no function in this module
reads the embedding or any column derived from it (phenotype_cluster,
umap_x/y, embedding_index, report_path). The outcome (mortality) variable is
likewise never read anywhere in this entire dispatch (see "Global
Constraints" below) -- the sole exception is Task 5c, which reads it only
when computing prognostic separability (axis 4).

## Variable Selection Criteria (`NUMERIC_COLS` / `CATEGORICAL_COLS`)

Task 12's `out/12_audit/variable_timing_table.csv` (124 variables,
classified into index_window/whole_admission/unknown along with source
citations) is used as the primary criterion for the candidate variable
pool. To avoid picking variables by name alone, the rules below were
applied explicitly to every actual candidate column (see
`task-5a-report.md` for the full cross-check):

1. **Primary candidates = columns that Task 12 classified as `index_window`
   and that literally exist in the real CSV.** Vital signs/lab values (SBP,
   RR, PR, Body_temperature, WBC_Count, CRP, Creatinine, T_Bilirubin,
   PLT_Count, Albumin, Hemoglobin, FiO2, AST, ALT, pO2_Arterial, pao2/fio2,
   GCS), SOFA/qSOFA sub-scores, the sepsis-determination flag, and the
   index BSI episode's organism/resistance phenotype belong here.
2. **[Re-correction, missingness-policy correction] The `_2`-suffixed
   variants are adopted -- but because of comparability, not because they
   are a "more complete version".** This item was wrong twice before being
   corrected a second time. The three judgments are recorded below in
   order.

   **First judgment (wrong):** "`_2` is a different computed version
   (`sofa_score`/`sofa_score_2` agreement rate 11.7%)." -- this 11.7%
   turned out to be a dtype string-comparison artifact (see the second
   judgment below).

   **Second judgment (incomplete -- mis-classified as a "bug"):** after
   normalizing dtype and re-comparing, the two columns agreed completely on
   every row where neither was missing (`sofa_score`/`sofa_score_2`:
   100.000% of n=3,793, r=1.0; `score_GCS`, `score_plt`, `score_bili`,
   `score_creat`, `score_hypotension`, `GCS` were all likewise identical --
   `scripts/24_verify_dense_pairs.py`). From this it was concluded that
   "`_2` is the more complete (dense) version of the same computation, the
   original (no suffix) is its sparse subset, and adopting the original up
   to now was a bug." **This conclusion was wrong** -- 100% agreement on
   the overlapping rows is exactly what would be observed even if the two
   policies genuinely differ, so this verification itself had no power to
   distinguish between the two hypotheses (as the author pointed out).

   **Third judgment (current, confirmed with information the author
   provided):** `sofa_score` (original) and `sofa_score_2` are not
   sparse/dense versions of the same value -- they are **two different
   missingness-handling policies**:
   - `sofa_score` (no suffix) = **complete-case**: if any of the 6
     component scores (`score_GCS`, `score_plt`, `score_bili`,
     `score_creat`, `score_hypotension`, `score_pao2/fio2`) is missing, the
     total is not computed and is left NULL. Empirically confirmed: the
     28,536 rows where `sofa_score` is NULL agree 100% with the rows where
     "at least one of the 6 components is NULL," and in the 3,793 rows
     where all 6 are present, `sofa_score` == sum(6 components) holds 100%
     of the time.
   - `sofa_score_2` = **assume-normal-if-missing**: if a component is
     missing, that component is treated as a score of 0 (normal) and
     summed anyway. Empirically confirmed: in the 32,272 rows where
     `sofa_score_2` is defined, `sofa_score_2` == sum(6 `_2` components)
     holds 100% of the time, and in rows where "the original component is
     NULL but the `_2` component has a value," that `_2` value is
     overwhelmingly 0 (`score_plt_2`/`score_bili_2`/`score_creat_2`/
     `score_hypotension_2`/`score_pao2/fio2_2`: 100.0% are 0;
     `score_GCS_2`: 98.2% are 0, with the remainder being normal-side
     values such as 3.0/1.0). According to the author, this same policy
     difference applies identically to most SOFA/GCS/sepsis-family
     variables in the original data. The 27.4-27.6% disagreement between
     `match_sepsis`/`match_sepsis_2` is explained by the same reason
     (sepsis determination depends on SOFA) -- this column is NOT swapped
     for its dense counterpart (kept as the original in the column
     selection below).

   | Complete-case (original) | Missing rate | Assume-normal (`_2`) | Missing rate |
   |---|---|---|---|
   | `sofa_score` | 88.3% | `sofa_score_2` | 0.2% |
   | `score_GCS` / `GCS` | 87.4% | `score_GCS_2` / `GCS_2` | 0.2% |
   | `sepsis` | 88.3% | `sepsis_2` | 0.2% |
   | `score_plt` | 0.9% | `score_plt_2` | 0.2% |
   | `score_bili` | 5.2% | `score_bili_2` | 0.2% |
   | `score_creat` | 3.4% | `score_creat_2` | 0.2% |
   | `score_hypotension` | 1.0% | `score_hypotension_2` | 0.2% |

   **Conclusion -- this is not a bug fix but an explicit analytical
   choice.** `NUMERIC_COLS`/`CATEGORICAL_COLS` still adopt `_2`
   (assume-normal-if-missing), but the reason is **comparability**: the
   originally derived embedding-based narrative text (case report text)
   rendered the `_2`-family values, so matching the severity information
   the embedding "saw" to the severity information the structured
   comparator "sees" requires using `_2` -- not because `_2` is more
   clinically valid or "less buggy." Adopting the complete-case original
   pays the price that 88% of the SOFA-based sample gets median-imputed;
   adopting `_2` pays the price that the severity signal carried by the
   missingness itself (sicker patients get tested more often -- missingness
   is informative) is erased, mixing in the clinically optimistic
   assumption that "not tested = normal." To handle this trade-off
   transparently, all three arms are computed:
   - **Arm 1 (primary analysis, this module's
     `NUMERIC_COLS`/`CATEGORICAL_COLS`):** `_2` (assume-normal-if-missing)
     -- comparability rationale, as described above.
   - **Arm 2 (complete-case sensitivity):** the original, unsuffixed
     columns exactly as-is, the very columns listed in
     `SUPERSEDED_SPARSE_*` -- this honestly reports that valid SOFA-based
     information exists for only 11.7% (3,793/32,329) of the sample.
     Already computed in `out/24_dense_fix/before_snapshot/` (the
     pre-correction code happened to be exactly this configuration).
   - **Arm 3 (missingness indicators added):** on top of Arm 1's `_2`
     values as-is, adds 8 binary indicator columns that flag whether each
     value was "an actual measurement (complete-case)" or "an
     assume-normal imputation" (see
     `build_structured_matrix_with_missingness_indicators`,
     `MISSINGNESS_INDICATOR_SOURCE_COLS`) -- restoring as an explicit
     feature the informative-missingness signal that Arm 1 erases.
   The names `SUPERSEDED_SPARSE_NUMERIC_COLS`/
   `SUPERSEDED_SPARSE_CATEGORICAL_COLS` are left unchanged (for
   audit-trail continuity), but their meaning should now be read not as
   "the sparse variant that was a bug" but as **"Arm 2: the complete-case
   variant, the arm that applies no assume-normal-if-missing imputation at
   all."**
3. **Low-variance (effectively constant) columns are statically excluded.**
   `quick_sofa_score`, `sepsis_quicksofa`, `score_q_GCS`, `score_q_RR`,
   `score_q_SBP` were confirmed, on the real data, to have exactly one
   single value across every row where they are non-missing (e.g., every
   non-missing value of `quick_sofa_score` is 3.0). They have zero
   discriminative power for clustering and are effectively nothing but a
   "was this missing" indicator, so they are statically excluded (instead
   of filtering them out data-dependently at runtime, the rationale is
   instead stated explicitly in this docstring and the report -- for
   reproducibility and testability).
4. **Of redundant/over-granular encodings of the same fact, only one is
   kept.** The index BSI organism identity is quadruply encoded as
   `BSI_episode_species` (1,491 species), `BSI_episode_short` (413
   species), `BSI_episode_short_clean` (20 categories), and
   `BSI_episode_microbe_id` (effectively a per-admission unique join
   string, 32,329 unique values). Only `BSI_episode_short_clean` is
   adopted -- coarse enough to use as a categorical feature while still
   clinically meaningful -- and the other three are excluded (to prevent
   multicollinearity/excessive cardinality).
5. **Pre-computed cluster/group labels are excluded.** `empiric_cluster`
   and `admission_cluster` already had their provenance settled by Task
   12's audit -- they are cluster labels produced by KMeans in
   `pipeline_new/unsupervised_clustering_empirical_regimen.py` from
   antibiotic-class flags (`admission_cluster`) and empiric-regimen-class
   flags (`empiric_cluster`) (see task-12-report.md). They are not
   embedding-derived, but regardless of provenance, "using an
   already-computed cluster label as an input feature for a new
   clustering" is itself circular reasoning, so it is excluded -- the same
   concern that applies to excluding `phenotype_cluster` applies here
   unchanged.
6. **Raw timestamps are excluded.** `test_date`, `first_test_date` are not
   meaningful clustering features without further derivation (day of
   week, season, etc.) -- that derivation is out of scope for this
   dispatch, so they are excluded.
7. **A small number of standard baseline covariates from
   `whole_admission` are explicitly added** (consistent with a reviewer
   requesting a "selected clinical variables" comparator, and with
   published clinical phenotype studies commonly including such
   variables): `charlson_score` (comorbidity index),
   `icu`/`ventilator`/`central_orarterial_line`/`urinary_orrenal_catheter`
   (organ-support flags -- Task 12 classified these as whole_admission but
   explicitly noted it could not rule out that they might be
   "index-anchored persistence" variables, i.e. ones that become true at
   the index time point and never revert; this caveat is inherited as-is
   here). `sex`, `age_at_test`, `hospital_code` are all listed as
   `whole_admission` in the Task 12 table, but they are self-evidently
   static demographic/institutional identifiers for which the very reason
   they were flagged for audit (timing ambiguity) does not apply in the
   first place -- they do not change during the admission, so the
   index-window/whole-admission distinction is meaningless for them.
8. **whole_admission candidates deliberately excluded:**
   `treatment_department_eng` (possible department transfer during
   admission -- risk of mixing in post-index-time information),
   `VRE_colonization`/`CRE_colonization` (not standard severity variables
   but separate screening-test variables, excluded to keep the scope
   minimal), `n_bsi_episodes`/`multiple_bsi_episodes` (the total number
   of BSI episodes across the whole admission was judged to be
   effectively an outcome-adjacent derived value, so conservatively
   excluded).

## Missing-Value Handling (`impute_structured_matrix`)

Numeric: per-column median imputation (an all-missing column, where the
median itself is NaN, is imputed with 0.0 -- this does not occur in the
real data but is a defensive branch for synthetic test data). Categorical:
filled by adding a dedicated "__missing__" category (k-prototypes'
matching_dissim always treats NaN as disagreeing with every other value,
which can distort results, and FAMD/one-hot encoding either fail or
implicitly drop rows if NaN is left as-is, so missingness is converted
into the explicit signal that "missing is itself a category"). This policy
must be reused identically by every method in Task 5 (5a/5b/5c), so it was
split out into a separate shared function -- if each method used a
different imputation policy, the comparison between methods would become
confounded, measuring "difference in missing-value handling" rather than
"difference in clustering algorithm."

**Interpretive caveat (disclosure of missing-value rates) [updated by the
dense-variant fix]:** before the correction above, the SOFA/GCS-family
sub-scores (and `sepsis`) went in as the sparse variant, which is 87-95%
missing. After swapping in the dense (`_2`) variant, the missing rate for
these 7 columns all drop to around 0.2% -- for these variables there is no
longer a problem of "median/dedicated-missing-category imputation
dominating the cluster structure." Missingness in the remaining columns
(vital signs/lab values, etc.) is unaffected by this correction and
remains uneven column by column (roughly 0.9-44.5%; as with the
`score_pao2/fio2` family, there are also columns that have a dense
counterpart but were not adopted by `NUMERIC_COLS` -- see the column
selection below). For the missingness that remains, the original caveat
still applies as-is: this comparator clusters not a "complete clinical
profile" but a "profile imputed with medians/a dedicated missing
category."

Global Constraints (applies to all of Task 5): 30d_mortality,
90d_mortality, in_hospital_mortality, death_date, discharge_date are not
referenced by any code path in this dispatch (Task 5c is the sole
exception).
"""
import os

# Force single-threaded BLAS/OpenMP/numba execution BEFORE numpy/pandas/
# sklearn/kmodes/prince are imported -- identical pattern and rationale to
# 02_recluster.py / 02b_partition_clustering.py (see those modules' top-of-
# file comments for the full empirical justification: 1/8/24 threads gave
# 26/18/33 different HDBSCAN cluster counts for the same seeded config on
# this exact codebase). k-prototypes, FAMD's underlying SVD, k-means and
# GaussianMixture all route their heavy linear algebra through BLAS, so the
# same thread-count-dependent floating-point accumulation order risk applies
# here. These libraries read the env vars at import time, so setting them
# later (even at the top of main()) is a silent no-op.
os.environ["OMP_NUM_THREADS"] = os.environ["NUMBA_NUM_THREADS"] = \
    os.environ["MKL_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import importlib
import importlib.metadata
import json
import sys
import time
from pathlib import Path

import gower
import numpy as np
import pandas as pd
import prince
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn_extra.cluster import KMedoids
from stepmix import StepMix
from stepmix.utils import get_mixed_descriptor

sys.path.insert(0, os.path.dirname(__file__))
frame_mod = importlib.import_module("01_build_analysis_frame")
recl = importlib.import_module("02_recluster")  # _thread_config, _library_versions
partition_mod = importlib.import_module("02b_partition_clustering")  # GMM_COVARIANCE_TYPE/GMM_REG_COVAR/N_INIT reuse

DATASET = Path("/data/tcmhwd/AI/AST_ai/manuscript/phenotype_ml_dataset.csv")
OUT = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision/out/05_comparators")

# See module docstring "Variable Selection Criteria" point 2 (dense-variant fix) for the
# full re-verification. Below: DENSE ("_2") variants of the four severity
# concepts (SOFA total, GCS, four SOFA sub-scores, sepsis flag) -- each
# agrees 100% (n_overlap, Pearson r=1.0) with its sparse twin wherever both
# are present (scripts/24_verify_dense_pairs.py), but the dense twin is
# ~0.2% missing vs. 87-95% missing for the sparse original.
NUMERIC_COLS = [
    # index_window vitals/labs
    "SBP", "RR", "PR", "Body_temperature", "WBC_Count", "CRP", "Creatinine",
    "T_Bilirubin", "PLT_Count", "Albumin", "Hemoglobin", "FiO2", "AST", "ALT",
    "pO2_Arterial", "pao2/fio2", "GCS_2",
    # index_window SOFA/qSOFA sub-scores (DENSE "_2" variant -- see fix note above)
    "sofa_score_2", "score_GCS_2", "score_plt_2", "score_bili_2", "score_creat_2",
    "score_hypotension_2",
    # index_window episode timing
    "days_from_admission_to_index_bsi",
    # whole_admission additions, documented exceptions (see docstring point 7)
    "age_at_test", "charlson_score",
]

CATEGORICAL_COLS = [
    # index_window binary/categorical clinical flags
    # NOTE: "sepsis" -> "sepsis_2" (dense-variant fix); "match_sepsis" is
    # DELIBERATELY UNCHANGED -- it is the one pair in the full X/X_2 sweep
    # that does NOT agree with its "_2" twin (27.4-27.6% agreement, a
    # genuinely different measurement, not a sparse/dense duplicate -- see
    # docstring point 2 and 24_verify_dense_pairs.py), so swapping it would
    # be the exact mistake this fix corrects for the other seven variables.
    "sepsis_2", "match_sepsis", "Vasopressin_any",
    # index_window index-BSI-episode organism identity + resistance phenotype
    "BSI_episode_short_clean", "BSI_episode_polymicrobial", "BSI_episode_3GCR",
    "BSI_episode_CarbaR", "BSI_episode_VRE", "BSI_episode_MRSA",
    "BSI_episode_CRAB", "BSI_episode_CeftriaxoneR", "BSI_episode_microbe_R",
    # whole_admission organ-support flags, documented exception (see docstring point 7)
    "icu", "ventilator", "central_orarterial_line", "urinary_orrenal_catheter",
    # demographic/site, documented exceptions (see docstring point 7)
    "sex", "hospital_code",
]

# SUPERSEDED_SPARSE_* names are kept UNCHANGED for audit-trail continuity
# (they are what earlier manifests/scripts already reference), but their
# MEANING was corrected (coordinator, 2026-09-06): these are NOT a "buggy
# sparse variant" of the same measurement -- they are the COMPLETE-CASE
# variant (no missing-as-normal imputation): SOFA/GCS/sepsis total is NULL
# if ANY of its component measurements is missing, rather than treating a
# missing component as normal/0. Every one of these is 87-95% missing
# (except "sepsis"/"match_sepsis"-style flags, ~88% missing) BECAUSE most
# admissions are missing at least one component, not because of any
# extraction bug. Never read by NUMERIC_COLS/CATEGORICAL_COLS (Arm 1, the
# primary analysis) -- read only by the Arm 2 complete-case sensitivity
# analysis (see module docstring point 2 and scripts/24_verify_dense_pairs.py
# for the verification, and out/24_dense_fix/before_snapshot/ for Arm 2's
# results, computed before this correction under what was -- coincidentally
# -- exactly this column selection).
SUPERSEDED_SPARSE_NUMERIC_COLS = [
    "GCS", "sofa_score", "score_GCS", "score_plt", "score_bili", "score_creat",
    "score_hypotension",
]
SUPERSEDED_SPARSE_CATEGORICAL_COLS = ["sepsis"]
# Column-name mapping actually applied (complete-case -> assume-normal-if-
# missing) by NUMERIC_COLS/CATEGORICAL_COLS (Arm 1), for manifests to record
# verbatim. Kept name (DENSE_VARIANT_FIX_MAP) unchanged for continuity;
# "dense" here means "assume-normal-if-missing", not "more completely
# measured" -- see module docstring point 2.
DENSE_VARIANT_FIX_MAP = {
    "GCS": "GCS_2", "sofa_score": "sofa_score_2", "score_GCS": "score_GCS_2",
    "score_plt": "score_plt_2", "score_bili": "score_bili_2",
    "score_creat": "score_creat_2", "score_hypotension": "score_hypotension_2",
    "sepsis": "sepsis_2",
}

# Arm 3 (coordinator-directed, 2026-09-06): for each Arm-1 ("_2",
# assume-normal-if-missing) severity variable, the complete-case column
# whose non-null/null status defines a "was this admission's value an
# actual measurement, or an assume-normal imputation" indicator. Missingness
# in the complete-case column is NOT random (sicker patients get tested
# more), so it is itself a severity signal that Arm 1 silently discards --
# Arm 3 restores it as an explicit feature. See
# build_structured_matrix_with_missingness_indicators below and module
# docstring point 2, "Arm 3".
MISSINGNESS_INDICATOR_SOURCE_COLS = {
    "GCS_2": "GCS", "sofa_score_2": "sofa_score", "score_GCS_2": "score_GCS",
    "score_plt_2": "score_plt", "score_bili_2": "score_bili",
    "score_creat_2": "score_creat", "score_hypotension_2": "score_hypotension",
    "sepsis_2": "sepsis",
}

# Explicitly banned regardless of NUMERIC_COLS/CATEGORICAL_COLS content --
# defense in depth against a future edit accidentally re-adding one of these
# (Global Constraints; see module docstring).
BANNED_OUTCOME_COLS = {
    "30d_mortality", "90d_mortality", "in_hospital_mortality",
    "death_date", "discharge_date",
}
BANNED_EMBEDDING_COLS = {
    "phenotype_cluster", "embedding_index", "umap_x", "umap_y", "report_path",
}
# Pre-existing cluster/group assignments -- excluded for the same circularity
# concern as phenotype_cluster even though their provenance (embedding-derived
# or not) could not be confirmed in this repository (see docstring point 5).
EXCLUDED_DERIVED_COLS = {"empiric_cluster", "admission_cluster"}
BANNED_COLS = BANNED_OUTCOME_COLS | BANNED_EMBEDDING_COLS | EXCLUDED_DERIVED_COLS

MISSING_CATEGORY = "__missing__"

# FAMD components: fixed at 10 to match the smallest UMAP dimensionality
# Task 2b used for its embedding-space comparator grid (umap10), so the
# reduced-space dimensionality is directly comparable across Task 2b's
# embedding comparators and this structured-data comparator rather than
# picking an unrelated number. See task-5a-report.md for the cumulative
# variance explained by 10 components on the real 32,329-row matrix.
FAMD_N_COMPONENTS = 10

KPROTO_N_INIT = 10  # matches partition_mod.N_INIT's convention for final (non-scan) fits
KMEANS_N_INIT = 10  # ditto, for the FAMD+k-means comparator's final k-means step

# fit_gower_pam (5b): sklearn_extra's exhaustive-swap "pam" method does not
# scale to n=32,329 (measured; see module docstring and task-5b-report.md),
# so "alternate" (Voronoi-iteration, roughly k-means-like cost) is used
# instead. init="k-medoids++" is NOT sklearn_extra's default (that default is
# "heuristic") -- it was picked empirically after "heuristic" was found, on a
# small clean 2-cluster synthetic case with a genuinely separable Gower
# distance structure (within-cluster mean distance ~0.44, between-cluster
# ~0.95), to collapse everything into a single cluster (see
# test_fit_gower_pam_recovers_categorical_structure_invisible_to_naive_euclidean
# and task-5b-report.md for the reproducible measurement). "k-medoids++"
# recovered the true clusters (ARI ~0.87) on the same data at negligible
# extra cost (<0.03s at n=2,000 vs 0.025s for "heuristic" -- the failure was
# a bad local optimum, not a speed/cost tradeoff).
GOWER_PAM_METHOD = "alternate"
GOWER_PAM_INIT = "k-medoids++"


def build_structured_matrix(
    frame: pd.DataFrame, dataset_path: Path = DATASET
) -> tuple[pd.DataFrame, list[str]]:
    """For the admission_IDs in the Task 1 frame, joins and returns only the
    structured clinical variables from `dataset_path` (default:
    `manuscript/phenotype_ml_dataset.csv`). `load_frame()` deliberately
    excludes most structured columns, so rather than reusing it, this
    function only reuses `frame`'s `admission_ID` column as the join key.

    See the module docstring's "Variable Selection Criteria" section for
    the variable selection criteria. The return value is a (matrix, list
    of categorical column names) tuple; numeric columns are float dtype
    and categorical columns are category dtype -- Gower distance
    (5b)/k-prototypes/FAMD all require this distinction.

    `dataset_path` is a parameter left in place so tests can inject a
    small synthetic CSV (real usage always uses the default as-is).
    """
    needed_cols = ["admission_ID"] + NUMERIC_COLS + CATEGORICAL_COLS
    raw = pd.read_csv(dataset_path, usecols=lambda c: c in needed_cols, low_memory=False)

    numeric_present = [c for c in NUMERIC_COLS if c in raw.columns]
    categorical_present = [c for c in CATEGORICAL_COLS if c in raw.columns]

    merged = frame[["admission_ID"]].merge(
        raw[["admission_ID"] + numeric_present + categorical_present],
        on="admission_ID", how="left", validate="one_to_one",
    )
    assert len(merged) == len(frame), (
        f"join changed row count ({len(frame)} -> {len(merged)}); "
        "dataset_path likely has duplicate admission_ID values"
    )

    matrix = pd.DataFrame(index=merged.index)
    for c in numeric_present:
        matrix[c] = merged[c].astype(float)
    for c in categorical_present:
        matrix[c] = merged[c].astype("category")

    leaked = BANNED_COLS & set(matrix.columns)
    assert not leaked, f"banned/derived column(s) leaked into structured matrix: {leaked}"

    return matrix, categorical_present


def build_structured_matrix_with_missingness_indicators(
    frame: pd.DataFrame, dataset_path: Path = DATASET
) -> tuple[pd.DataFrame, list[str]]:
    """Arm 3 (coordinator-directed, 2026-09-06). SAME matrix as
    `build_structured_matrix` (Arm 1: `_2`/assume-normal-if-missing severity
    variables), PLUS one binary indicator column per
    `MISSINGNESS_INDICATOR_SOURCE_COLS` entry marking whether that
    admission's severity value came from an ACTUAL measurement (its
    complete-case column is non-null) or from the assume-normal-if-missing
    policy baked into the "_2" variant (complete-case column is null).
    `1` = actually measured, `0` = assume-normal imputation.

    Deliberately a NEW function (never mutates `build_structured_matrix`,
    per this repo's convention) -- reuses it unmodified for the base matrix,
    then joins the complete-case columns ONLY to derive the 8 indicators
    (their own values are never added as features, only their null/non-null
    status).

    Rationale: missingness in the complete-case columns is informative
    (sicker patients get tested more), and Arm 1 alone silently erases that
    signal by assuming "not tested" = "normal". This arm restores it as an
    explicit feature so a comparator method can use it if it is predictive.
    See module docstring point 2, "Arm 3".
    """
    matrix, categorical_cols = build_structured_matrix(frame, dataset_path=dataset_path)

    sparse_cols = list(MISSINGNESS_INDICATOR_SOURCE_COLS.values())
    needed_cols = ["admission_ID"] + sparse_cols
    raw = pd.read_csv(dataset_path, usecols=lambda c: c in needed_cols, low_memory=False)
    sparse_present = [c for c in sparse_cols if c in raw.columns]
    missing_sources = set(sparse_cols) - set(sparse_present)
    assert not missing_sources, f"missingness-indicator source column(s) not found in {dataset_path}: {missing_sources}"

    merged = frame[["admission_ID"]].merge(
        raw[["admission_ID"] + sparse_present], on="admission_ID", how="left", validate="one_to_one",
    )
    assert len(merged) == len(frame), (
        f"join changed row count ({len(frame)} -> {len(merged)}); "
        "dataset_path likely has duplicate admission_ID values"
    )

    out = matrix.copy()
    new_categorical_cols = list(categorical_cols)
    for dense_col, sparse_col in MISSINGNESS_INDICATOR_SOURCE_COLS.items():
        indicator_name = f"{sparse_col}_was_measured"
        out[indicator_name] = merged[sparse_col].notna().astype(int).astype("category")
        new_categorical_cols.append(indicator_name)

    leaked = BANNED_COLS & set(out.columns)
    assert not leaked, f"banned/derived column(s) leaked into structured matrix: {leaked}"

    return out, new_categorical_cols


def impute_structured_matrix(matrix: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """Fills missing values: numeric columns with the per-column median,
    categorical columns with a dedicated `__missing__` category -- a
    helper split out so that every method in Task 5 (5a/5b/5c) shares the
    same policy (see the module docstring's "Missing-Value Handling"
    section). The returned DataFrame has no NaN left in it at all.
    """
    out = matrix.copy()
    numeric_cols = [c for c in out.columns if c not in categorical_cols]

    for c in numeric_cols:
        median = out[c].median()
        fill_value = median if pd.notna(median) else 0.0  # all-missing column (defensive; not expected on real data)
        out[c] = out[c].fillna(fill_value)

    for c in categorical_cols:
        if not isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].astype("category")
        if out[c].isna().any():
            if MISSING_CATEGORY not in out[c].cat.categories:
                out[c] = out[c].cat.add_categories([MISSING_CATEGORY])
            out[c] = out[c].fillna(MISSING_CATEGORY)

    assert not out.isna().any().any(), "impute_structured_matrix left NaN behind"
    return out


def fit_kprototypes(
    matrix: pd.DataFrame, categorical_cols: list[str], k: int, seed: int = 42
) -> np.ndarray:
    """k-prototypes (Huang 1998) -- computes Euclidean distance for numeric
    features and matching (mismatch) distance for categorical features,
    combines them with a gamma weight, then iteratively updates in
    k-means style.

    Numeric features are standardized with `StandardScaler` before
    fitting: k-prototypes' Euclidean part adds each feature's raw
    unit/scale directly into the distance (e.g. WBC_Count is on the order
    of thousands while FiO2 is in the 0-1 range), so without
    standardization a large-scale variable would dominate the distance
    regardless of its actual clinical importance. Categorical features
    are handled by kmodes itself, so they are only converted to strings
    without separate encoding (so that comparison is equivalent
    regardless of differences in internal representation/category codes).

    kmodes cannot accept NaN, so `impute_structured_matrix` fills missing
    values first.
    """
    imputed = impute_structured_matrix(matrix, categorical_cols)
    numeric_cols = [c for c in imputed.columns if c not in categorical_cols]

    scaled = imputed.copy()
    if numeric_cols:
        scaler = StandardScaler()
        scaled[numeric_cols] = scaler.fit_transform(imputed[numeric_cols].astype(float))
    for c in categorical_cols:
        scaled[c] = scaled[c].astype(str)

    ordered_cols = numeric_cols + categorical_cols
    X = scaled[ordered_cols].to_numpy(dtype=object)
    categorical_positions = list(range(len(numeric_cols), len(ordered_cols)))

    kp = KPrototypes(
        n_clusters=k, init="Huang", n_init=KPROTO_N_INIT, random_state=seed, n_jobs=1,
    )
    labels = kp.fit_predict(X, categorical=categorical_positions)
    return np.asarray(labels)


def fit_famd_kmeans(
    matrix: pd.DataFrame, categorical_cols: list[str], k: int, seed: int = 42
) -> np.ndarray:
    """Reduces the mixed-type variables to a `FAMD_N_COMPONENTS`-dimensional
    continuous space with FAMD (Factor Analysis of Mixed Data), then fits
    k-means on those coordinates.

    Confirmed scikit-learn-style API for prince==0.20.2: build a
    `prince.FAMD(...)` instance, call `.fit(df)`, then obtain the per-row
    component coordinates (a DataFrame) with `.row_coordinates(df)`
    (older-tutorial `famd.fit_transform(df)`, which returns the
    coordinates directly, gives an equivalent result, but here `fit` and
    `row_coordinates` are called separately to make the
    coordinate-extraction step explicit). `.fit()` auto-detects numeric
    vs. categorical from each input DataFrame column's dtype (numeric
    dtype -> continuous, category/object/bool dtype -> categorical), so
    it directly uses the float/category dtype distinction that
    `impute_structured_matrix` produces -- no separate encoding is
    needed.

    FAMD cannot accept missing values, so `impute_structured_matrix`
    fills missing values first.
    """
    imputed = impute_structured_matrix(matrix, categorical_cols)
    famd = prince.FAMD(n_components=FAMD_N_COMPONENTS, random_state=seed)
    famd = famd.fit(imputed)
    coords = famd.row_coordinates(imputed).to_numpy()

    km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=seed).fit(coords)
    return km.labels_


def fit_gmm_structured(
    matrix: pd.DataFrame, categorical_cols: list[str], k: int, seed: int = 42
) -> np.ndarray:
    """Fits GMM directly on the structured variables -- included to form a
    pair with Task 2b's `fit_gmm` (GMM on the embedding space) that is
    algorithmically identical and differs only in representation
    (enabling a "same algorithm, different representation" comparison).
    `covariance_type`/`reg_covar`/`n_init` are reused as-is from Task 2b
    (`02b_partition_clustering.py`) -- not re-derived: `reg_covar=1e-3` is
    the value Task 2b introduced to stop an actual SLURM job (586187) from
    crashing on singular covariance with sklearn's default (1e-6) in the
    UMAP50 space, and since standardized continuous features + one-hot
    categorical features can also pack densely in this structured-variable
    space, the same failure mode could in principle apply here too, so the
    value is inherited unchanged (there is no evidence that this space is
    actually safer than UMAP50 -- it is not lowered to a smaller value
    without evidence).

    **Limitation (unlike k-prototypes/FAMD, cannot handle mixed types
    directly):** GMM only accepts numeric input, so categorical columns
    are one-hot encoded and numeric columns are standardized, then
    concatenated. This is a more primitive way of handling categorical
    variables than k-prototypes (matching distance for categorical) or
    FAMD (which treats categorical variables via its own
    indicator-matrix structure) -- each category becomes an independent
    binary axis, so a high-cardinality categorical variable (e.g.
    `BSI_episode_microbe_R`, 55 categories) ends up occupying far more
    dimensions than its actual information content warrants. This is a
    separate, substantive limitation distinct from the very reason this
    comparator is included (not an algorithm comparison but a "same
    algorithm, different representation" comparison), and must be stated
    explicitly when interpreting the results.
    """
    imputed = impute_structured_matrix(matrix, categorical_cols)
    numeric_cols = [c for c in imputed.columns if c not in categorical_cols]

    parts = []
    if numeric_cols:
        scaler = StandardScaler()
        parts.append(scaler.fit_transform(imputed[numeric_cols].astype(float)))
    if categorical_cols:
        onehot = pd.get_dummies(imputed[categorical_cols].astype(str))
        parts.append(onehot.to_numpy(dtype=float))
    X = np.hstack(parts) if len(parts) > 1 else parts[0]

    gmm = GaussianMixture(
        n_components=k,
        covariance_type=partition_mod.GMM_COVARIANCE_TYPE,
        n_init=partition_mod.N_INIT,
        random_state=seed,
        reg_covar=partition_mod.GMM_REG_COVAR,
    ).fit(X)
    return gmm.predict(X)


def compute_gower_matrix(matrix: pd.DataFrame, categorical_cols: list[str]) -> np.ndarray:
    """A helper that splits out just the O(n^2) Gower distance matrix
    computation that `fit_gower_pam` otherwise recomputes every time
    (added for Task 5c) -- paired with `fit_gower_pam`'s
    `precomputed_distance` parameter, this lets a caller (5c) who needs
    to compare multiple `k` values against the same `matrix` do this
    ~27-minute computation only once and reuse it. `fit_gower_pam` also
    calls this function internally as-is (see below), so there is no
    risk of the two paths ("compute once and reuse" vs. "recompute every
    time") ending up with different computation logic.

    `gower.gower_matrix` does not accept NaN, so `impute_structured_matrix`
    fills missing values first.
    """
    imputed = impute_structured_matrix(matrix, categorical_cols)
    cat_mask = [c in categorical_cols for c in imputed.columns]
    return gower.gower_matrix(imputed, cat_features=cat_mask)


def fit_gower_pam(
    matrix: pd.DataFrame, categorical_cols: list[str], k: int, seed: int = 42,
    init: str = GOWER_PAM_INIT, precomputed_distance: np.ndarray | None = None,
) -> np.ndarray:
    """A practical approximation of Gower distance (Gower 1971) + PAM
    (partitioning around medoids) -- computes the pairwise Gower distance
    matrix between rows with the `gower` package (0.1.2), then clusters
    with `sklearn_extra.cluster.KMedoids(metric="precomputed")`.

    `init` defaults to the module constant `GOWER_PAM_INIT`
    ("k-medoids++"), and real usage (`main()`) always uses this default
    as-is. The only reason it is exposed as a parameter is so tests can
    directly compare `init="heuristic"` against `init="k-medoids++"`
    without duplicating the clustering-call code (see the "init="
    rationale paragraph below and
    `test_fit_gower_pam_init_default_recovers_clusters_while_heuristic_collapses`).

    Gower distance computes, for each numeric column, the
    (range-normalized) absolute difference, and for each categorical
    column, agreement/disagreement (0/1), then averages these with equal
    weight per column (confirmed from the actual implementation of
    `gower.gower_dist.gower_get`: `sij_cat = (xi_cat != xj_cat)`, so the
    maximum contribution of any single column is always fixed at 1
    regardless of its number of categories/cardinality). Unlike
    k-prototypes (categorical = matching, numeric = Euclidean,
    gamma-weighted sum) or GMM's one-hot encoding (where higher
    cardinality means a variable occupies more dimensions), this is
    inherently free of distortion from per-column scale/cardinality
    differences -- the core reason this comparator is included.

    **Why `method="alternate"` is used (based on measured computational
    cost, not assumed without evidence):** `sklearn_extra`'s two base
    algorithms are `method="pam"` (classic exhaustive-swap search) and
    `method="alternate"` (Voronoi iteration, k-means-like cost).
    Measuring on the login node with subsamples of the real 32,329-row
    structured matrix (see `task-5b-report.md`), `"pam"` already rose
    from 0.554s to 6.58s between n=2,000 and n=5,000 (a 11.9x ratio,
    distinctly steeper than `(5000/2000)^2` = 6.25x -- suggesting that,
    being an exhaustive-swap search, its per-iteration cost grows
    super-linearly), and was judged unable to cope at 32,329 rows.
    `"alternate"` was nearly negligible over the same range, 0.015s to
    0.201s (the O(n^2) cost of the distance matrix itself overwhelmingly
    dominates, and that cost is independent of the `method` choice).
    `GOWER_PAM_METHOD = "alternate"` was therefore adopted.

    **Why `init="k-medoids++"` is used (not `sklearn_extra`'s default of
    `"heuristic"`; this reason is likewise measured, not assumed without
    evidence):** on small synthetic data (two categorical columns of
    differing cardinality forming 2 clusters that are clearly separated
    in Gower distance -- within-cluster mean distance ~0.44,
    between-cluster ~0.95), `init="heuristic"` (the default) was actually
    observed to converge to a bad local optimum that collapses the two
    clusters into one (every point assigned to a single cluster, with the
    remaining medoid left empty) (`sklearn_extra` issues a "Cluster N is
    empty" warning). `init="k-medoids++"` recovered the true cluster
    structure on the same data (ARI ~0.87), at negligible extra cost
    (n=2,000: 0.015s for heuristic vs. 0.025s for k-medoids++) -- i.e.,
    this was purely an initialization-quality problem, not a
    speed-accuracy trade-off. The reproducible measurements are in
    `test_fit_gower_pam_recovers_categorical_structure_invisible_to_naive_euclidean`
    and
    `test_fit_gower_pam_init_default_recovers_clusters_while_heuristic_collapses`
    (see the "init=" parameter description below), and
    `task-5b-report.md`.

    **Known weakness of `k-medoids++` (disclosed, not measured):**
    `k-medoids++` applies a k-means++-style seeding scheme --
    preferentially selecting the farthest point with weighted probability
    -- to medoids, and this scheme is itself known to be fragile when the
    input has many duplicate/near-duplicate points: seeding can become
    biased toward regions where nearly identical points are clustered
    together, or performance can degrade when the data has many exact or
    very-close-distance ties. This is not a purely theoretical concern
    for this function specifically: Task 2/2b has already extensively
    documented that, in this cohort's LLM-generated clinical narrative
    embeddings, near-duplicate embedding rows genuinely exist between
    patients with very similar clinical profiles (see
    task-2-report.md/task-2b-report.md). It is reasonable to think the
    same concern extends in principle to the structured feature matrix
    (demographics/lab values, etc.) that this function clusters --
    because patients with very similar demographics/lab values can
    become near-ties, literally close together, in Gower distance.
    **However, this is only a disclosure of a known algorithmic property
    and its plausible applicability to this dataset, not a claim that
    its actual impact has been measured on this structured matrix** --
    such a measurement would require a separate analysis outside the
    scope of this dispatch.

    **Computational cost (full 32,329-row scale, empirically measured via
    SLURM job 586191):** the Gower distance matrix is O(n^2) memory.
    `gower.gower_matrix` empirically returns float32 (see the module
    docstring), so the full 32,329-row distance matrix was exactly
    4,180,656,964 bytes (~4.18GB), the computation took 1640.5s (~27.3
    min), the `KMedoids(method="alternate", k=8)` fit took 6.98s, and
    peak RSS was ~4.96GB -- **feasible**, also consistent with the
    subsample extrapolation (roughly 30-45 min). This cost is for one
    full call of this function (one `k`, one full distance-matrix
    computation); if 5c compares multiple `k` values and calls this
    function each time, the distance matrix gets recomputed every time,
    multiplying the cost accordingly -- this function always honestly
    computes the full distance matrix for whatever `matrix` it is given
    and never caches it, so whether to reuse the distance matrix across
    multiple `k` values is 5c's design decision (if reused, the
    additional cost per extra `k` is just the KMedoids fit, a few
    seconds).

    `gower.gower_matrix` does not accept NaN, so `impute_structured_matrix`
    fills missing values first.

    **`precomputed_distance` (added for Task 5c):** with the default
    `None`, this function computes the full O(n^2) distance matrix
    honestly from scratch every time via `compute_gower_matrix`, exactly
    as before -- backward compatible, so existing callers/tests are
    completely unaffected even if they are unaware this parameter exists.
    When, as in 5c, multiple `k` values must be compared against the same
    `matrix`, the caller can call `compute_gower_matrix` only once and
    reuse its result across multiple `fit_gower_pam(...,
    precomputed_distance=D)` calls, so the ~27-minute distance-matrix
    computation does not have to be repeated for every `k` value (see the
    module docstring/`task-5c-report.md`). Even when reusing it, the
    `matrix`/`categorical_cols` arguments are still required to preserve
    the call signature, but they are not used for the distance
    computation -- the caller must itself guarantee that
    `precomputed_distance` was actually computed from that `matrix` (this
    function does not verify the two are consistent).
    """
    if precomputed_distance is None:
        distance = compute_gower_matrix(matrix, categorical_cols)
    else:
        distance = precomputed_distance

    km = KMedoids(
        n_clusters=k,
        metric="precomputed",
        method=GOWER_PAM_METHOD,
        init=init,
        random_state=seed,
    ).fit(distance)
    return np.asarray(km.labels_)


def fit_lca(
    matrix: pd.DataFrame, categorical_cols: list[str], k: int, seed: int = 42
) -> np.ndarray:
    """Latent class analysis -- fit with `StepMix` from `stepmix`
    (`stepmix==3.0.0`).

    Older `stepmix` tutorials assume a single global
    `measurement="categorical"` string interface, but since this matrix
    is a mixed type with 26 continuous + 18 categorical columns together,
    checking the actually-installed 3.0.0 API (`help(StepMix)`, the
    `stepmix.utils.get_mixed_descriptor` source) confirmed that the
    correct current API is to build a "nested" descriptor dict specifying
    the model type per column group with the
    `stepmix.utils.get_mixed_descriptor(dataframe, **kwargs)` helper, and
    pass it to `StepMix(measurement=descriptor)`: the numeric column
    group is given `"continuous"` (= `GaussianDiag`, confirmed from
    `EMISSION_DICT`), and the categorical column group is given
    `"categorical"` (= `Multinoulli`). `structural`/`y` are not specified
    -- this comparator is classic LCA, fitting only the pure measurement
    model with no structural model (e.g. latent class -> covariates/
    outcome), and it was empirically confirmed that calling
    `model.fit(data)` without `Y`/`y` indeed works correctly (only the
    unsupervised latent-class assignment is needed).

    The `Multinoulli` emission model expects integer codes (zero-based
    category indices) (the internal implementation of
    `stepmix.utils.max_one_hot` casts values to `int`), so categorical
    columns are integer-encoded via pandas `category` dtype's
    `.cat.codes` (since `impute_structured_matrix` has already filled
    missing values with the dedicated `__missing__` category, the code
    value -1 (NaN) never occurs).

    `progress_bar=0` turns off stepmix's tqdm progress-bar output (to
    make the output style consistent with the other fit_* functions, and
    to avoid needlessly filling up SLURM logs).
    """
    imputed = impute_structured_matrix(matrix, categorical_cols)
    numeric_cols = [c for c in imputed.columns if c not in categorical_cols]

    coded = imputed.copy()
    for c in categorical_cols:
        coded[c] = coded[c].cat.codes.astype(float)

    descriptor_kwargs = {}
    if numeric_cols:
        descriptor_kwargs["continuous"] = numeric_cols
    if categorical_cols:
        descriptor_kwargs["categorical"] = categorical_cols
    data, descriptor = get_mixed_descriptor(coded, **descriptor_kwargs)

    model = StepMix(
        n_components=k, measurement=descriptor, random_state=seed, progress_bar=0,
    )
    model.fit(data)
    return np.asarray(model.predict(data))


def _atomic_write_json(obj: dict, dest: Path) -> None:
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, dest)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--sanity-check", action="store_true",
        help=(
            "Run build_structured_matrix + impute_structured_matrix once on the "
            "real 32,329-row data and report column counts/types (does not run "
            "fit_* -- running the full grid is Task 5c's job)."
        ),
    )
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    if not args.sanity_check:
        raise SystemExit(
            "This script currently supports only --sanity-check (5a scope: "
            "provides the shared functions + three fit_* functions. Running the "
            "full grid/SLURM submission is Task 5c)."
        )

    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    frame = frame_mod.load_frame(verify_reports=False)
    matrix, categorical_cols = build_structured_matrix(frame)
    build_sec = time.time() - t0

    t0 = time.time()
    imputed = impute_structured_matrix(matrix, categorical_cols)
    impute_sec = time.time() - t0

    numeric_cols = [c for c in matrix.columns if c not in categorical_cols]
    n_missing_before = int(matrix.isna().sum().sum())
    n_missing_after = int(imputed.isna().sum().sum())

    summary = {
        "n_rows": int(len(matrix)),
        "n_numeric_cols": len(numeric_cols),
        "n_categorical_cols": len(categorical_cols),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "n_missing_cells_before_impute": n_missing_before,
        "n_missing_cells_after_impute": n_missing_after,
        "build_sec": round(build_sec, 3),
        "impute_sec": round(impute_sec, 3),
        "seed": args.seed,
        "dense_variant_fix": {
            "applied": True,
            "sparse_to_dense_map": DENSE_VARIANT_FIX_MAP,
            "superseded_sparse_numeric_cols": SUPERSEDED_SPARSE_NUMERIC_COLS,
            "superseded_sparse_categorical_cols": SUPERSEDED_SPARSE_CATEGORICAL_COLS,
            "excluded_from_swap": {
                "match_sepsis": "27.4-27.6% agreement with match_sepsis_2 -- genuinely "
                                 "different measurement, not a sparse/dense duplicate; "
                                 "left unswapped (see scripts/24_verify_dense_pairs.py)",
            },
        },
        "libraries": {
            **recl._library_versions(),
            "kmodes": importlib.metadata.version("kmodes"),
            "prince": prince.__version__,
        },
        "threads": recl._thread_config(),
    }
    _atomic_write_json(summary, OUT / "sanity_check.json")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("libraries", "threads")},
                      indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
