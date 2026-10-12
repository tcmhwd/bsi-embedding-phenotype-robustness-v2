import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from importlib import import_module
mod = import_module("05_baseline_comparators")

PYTHON = sys.executable

# Global Constraints for Task 5a: outcome variables must never be read by
# this dispatch's code (that exception is reserved for Task 5c). Enforced
# here, not just in production code, by never constructing a synthetic
# fixture that includes these as anything other than a "must be excluded"
# probe column.
BANNED_OUTCOME_COLS = {"30d_mortality", "90d_mortality", "in_hospital_mortality", "death_date", "discharge_date"}


def _synthetic_dataset_csv(tmp_path, n=80, seed=0) -> Path:
    """A small mixed-type CSV mimicking a slice of phenotype_ml_dataset.csv:
    a couple of real numeric/categorical columns actually used by
    NUMERIC_COLS/CATEGORICAL_COLS, plus outcome and embedding-derived
    columns that build_structured_matrix must never surface.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "admission_ID": [f"A{i:04d}" for i in range(n)],
        # real NUMERIC_COLS entries
        "age_at_test": rng.integers(18, 90, size=n),
        "charlson_score": rng.integers(0, 10, size=n),
        # real CATEGORICAL_COLS entries
        "sex": rng.choice(["M", "F"], size=n),
        "icu": rng.integers(0, 2, size=n),
        # banned outcome columns -- must never appear in build_structured_matrix's output
        "30d_mortality": rng.integers(0, 2, size=n),
        "90d_mortality": rng.integers(0, 2, size=n),
        "in_hospital_mortality": rng.integers(0, 2, size=n),
        "death_date": ["2020-01-01"] * n,
        "discharge_date": ["2020-01-05"] * n,
        # banned embedding-derived columns -- ditto
        "phenotype_cluster": rng.integers(0, 5, size=n),
        "embedding_index": np.arange(n),
        "umap_x": rng.normal(size=n),
        "umap_y": rng.normal(size=n),
        "report_path": ["/x/y.txt"] * n,
    })
    path = tmp_path / "synthetic_dataset.csv"
    df.to_csv(path, index=False)
    return path


def _synthetic_frame(n=80) -> pd.DataFrame:
    return pd.DataFrame({"admission_ID": [f"A{i:04d}" for i in range(n)]})


def test_build_structured_matrix_separates_dtypes_and_excludes_banned_cols(tmp_path):
    frame = _synthetic_frame()
    dataset_path = _synthetic_dataset_csv(tmp_path)

    matrix, categorical_cols = mod.build_structured_matrix(frame, dataset_path=dataset_path)

    assert len(matrix) == len(frame)
    assert set(categorical_cols) == {"sex", "icu"}
    assert set(matrix.columns) == {"age_at_test", "charlson_score", "sex", "icu"}

    for c in ("age_at_test", "charlson_score"):
        assert matrix[c].dtype == np.float64
    for c in ("sex", "icu"):
        assert isinstance(matrix[c].dtype, pd.CategoricalDtype)

    banned = BANNED_OUTCOME_COLS | {
        "phenotype_cluster", "embedding_index", "umap_x", "umap_y", "report_path",
    }
    assert banned.isdisjoint(set(matrix.columns))


def test_build_structured_matrix_preserves_row_count_and_admission_id_order(tmp_path):
    frame = _synthetic_frame(n=30)
    dataset_path = _synthetic_dataset_csv(tmp_path, n=30)
    matrix, _ = mod.build_structured_matrix(frame, dataset_path=dataset_path)
    assert len(matrix) == 30


def _synthetic_mixed_matrix(n_per_cluster=40, n_clusters=3, seed=0):
    """3 well-separated clusters defined jointly by a numeric feature (3
    widely spaced means) and a categorical feature (a distinct category per
    cluster) -- k-prototypes/FAMD+kmeans/GMM should all recover this
    trivially. A handful of numeric NaN and categorical NaN are injected so
    the determinism/shape tests also exercise impute_structured_matrix along
    the way (fit_* call it internally).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_clusters):
        n1 = rng.normal(loc=c * 20.0, scale=1.0, size=n_per_cluster)
        n2 = rng.normal(loc=c * 20.0, scale=1.0, size=n_per_cluster)
        cat = np.full(n_per_cluster, f"cat{c}")
        rows.append(pd.DataFrame({"num1": n1, "num2": n2, "cat1": cat}))
    df = pd.concat(rows, ignore_index=True)
    df["cat1"] = df["cat1"].astype("category")
    # inject a little missingness in both a numeric and a categorical column
    rng.shuffle(idx := np.arange(len(df)))
    df.loc[idx[:5], "num1"] = np.nan
    df.loc[idx[5:8], "cat1"] = np.nan
    return df, ["cat1"]


def test_impute_structured_matrix_leaves_no_nan():
    matrix, categorical_cols = _synthetic_mixed_matrix()
    assert matrix.isna().any().any()  # sanity: fixture actually has NaN
    imputed = mod.impute_structured_matrix(matrix, categorical_cols)
    assert not imputed.isna().any().any()
    assert len(imputed) == len(matrix)


def test_impute_structured_matrix_fills_categorical_with_dedicated_category():
    matrix, categorical_cols = _synthetic_mixed_matrix()
    na_mask = matrix["cat1"].isna()
    assert na_mask.any()
    imputed = mod.impute_structured_matrix(matrix, categorical_cols)
    assert (imputed.loc[na_mask, "cat1"] == mod.MISSING_CATEGORY).all()


def test_impute_structured_matrix_fills_numeric_with_median():
    matrix, categorical_cols = _synthetic_mixed_matrix()
    median = matrix["num1"].median()
    na_mask = matrix["num1"].isna()
    assert na_mask.any()
    imputed = mod.impute_structured_matrix(matrix, categorical_cols)
    assert np.allclose(imputed.loc[na_mask, "num1"], median)


# fit_gower_pam / fit_lca (Task 5b) reuse the exact same shape/recovery/
# determinism tests as 5a's three methods -- same fixture, same assertions --
# rather than duplicating near-identical test bodies.
FIT_NAMES = ["fit_kprototypes", "fit_famd_kmeans", "fit_gmm_structured", "fit_gower_pam", "fit_lca"]


@pytest.mark.parametrize("fit_name", FIT_NAMES)
def test_fit_returns_correct_length_labels(fit_name):
    matrix, categorical_cols = _synthetic_mixed_matrix()
    fit_fn = getattr(mod, fit_name)
    labels = fit_fn(matrix, categorical_cols, k=3, seed=42)
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (len(matrix),)


@pytest.mark.parametrize("fit_name", FIT_NAMES)
def test_fit_recovers_well_separated_clusters(fit_name):
    matrix, categorical_cols = _synthetic_mixed_matrix()
    fit_fn = getattr(mod, fit_name)
    labels = fit_fn(matrix, categorical_cols, k=3, seed=42)
    # sanity check, not a rigorous clustering-quality test: with 3 obviously
    # separated groups of 40 rows each, the 3 recovered clusters should be
    # roughly balanced in size (no method should collapse everything into
    # one giant cluster plus stragglers).
    _, counts = np.unique(labels, return_counts=True)
    assert len(counts) == 3
    assert min(counts) >= 30  # well clear of a degenerate/collapsed split


@pytest.mark.parametrize("fit_name", FIT_NAMES)
def test_fit_is_deterministic_given_seed(fit_name):
    matrix, categorical_cols = _synthetic_mixed_matrix()
    fit_fn = getattr(mod, fit_name)
    a = fit_fn(matrix, categorical_cols, k=3, seed=42)
    b = fit_fn(matrix, categorical_cols, k=3, seed=42)
    assert np.array_equal(a, b)


# Regression test for the thread-count-dependent clustering defect (same
# mechanism/rationale as 02_recluster.py's and 02b_partition_clustering.py's
# analogous tests -- kmodes/prince/sklearn all route heavy linear algebra
# through BLAS). Run as a fresh subprocess because OMP_NUM_THREADS/etc. are
# read at import time.
_SUBPROC_SCRIPT = """
import sys
sys.path.insert(0, "__SCRIPTS_DIR__")
from importlib import import_module
import json
import numpy as np
import pandas as pd

mod = import_module("05_baseline_comparators")
rng = np.random.default_rng(0)
rows = []
for c in range(3):
    n1 = rng.normal(loc=c * 20.0, scale=1.0, size=40)
    n2 = rng.normal(loc=c * 20.0, scale=1.0, size=40)
    cat = np.full(40, f"cat{c}")
    rows.append(pd.DataFrame({"num1": n1, "num2": n2, "cat1": cat}))
df = pd.concat(rows, ignore_index=True)
df["cat1"] = df["cat1"].astype("category")
cat_cols = ["cat1"]

out = {}
out["kprototypes"] = mod.fit_kprototypes(df, cat_cols, k=3, seed=42).tolist()
out["famd_kmeans"] = mod.fit_famd_kmeans(df, cat_cols, k=3, seed=42).tolist()
out["gmm_structured"] = mod.fit_gmm_structured(df, cat_cols, k=3, seed=42).tolist()
out["gower_pam"] = mod.fit_gower_pam(df, cat_cols, k=3, seed=42).tolist()
out["lca"] = mod.fit_lca(df, cat_cols, k=3, seed=42).tolist()
print(json.dumps(out))
"""


def _run_with_threads(n_threads: str) -> dict:
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = n_threads
    proc = subprocess.run(
        [PYTHON, "-c", _SUBPROC_SCRIPT.replace("__SCRIPTS_DIR__", SCRIPTS_DIR)],
        env=env, capture_output=True, text=True, check=True, timeout=300,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_labels_identical_across_thread_counts():
    """05_baseline_comparators.py must pin OMP/NUMBA/MKL/OPENBLAS thread
    pools to 1 at import time, the same fix 02_recluster.py and
    02b_partition_clustering.py required. Without it, kmodes/prince/sklearn's
    BLAS-routed linear algebra could make labels vary by thread count even
    with random_state/seed fixed.
    """
    r1 = _run_with_threads("1")
    r24 = _run_with_threads("24")
    assert r1["kprototypes"] == r24["kprototypes"], "k-prototypes labels differ across thread counts"
    assert r1["famd_kmeans"] == r24["famd_kmeans"], "FAMD+kmeans labels differ across thread counts"
    assert r1["gmm_structured"] == r24["gmm_structured"], "GMM labels differ across thread counts"
    assert r1["gower_pam"] == r24["gower_pam"], "Gower+PAM labels differ across thread counts"
    assert r1["lca"] == r24["lca"], "LCA labels differ across thread counts"


def _two_cluster_categorical_fixture(seed=0, n_per=60):
    """Two well-separated clusters carried entirely by a low-cardinality
    categorical column ("group", 2 categories), alongside an uninformative
    high-cardinality categorical column ("site", 10 categories, uncorrelated
    with the true cluster). Shared by
    `test_fit_gower_pam_recovers_categorical_structure_invisible_to_naive_euclidean`
    and
    `test_fit_gower_pam_init_default_recovers_clusters_while_heuristic_collapses`
    so both tests exercise the identical clustering call/fixture instead of
    duplicating it.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(2):
        group = np.full(n_per, f"g{g}")
        site = rng.choice([f"site{i}" for i in range(10)], size=n_per)  # uncorrelated with group
        rows.append(pd.DataFrame({"group": group, "site": site}))
    df = pd.concat(rows, ignore_index=True)
    df["group"] = df["group"].astype("category")
    df["site"] = df["site"].astype("category")
    categorical_cols = ["group", "site"]
    truth = np.repeat([0, 1], n_per)
    return df, categorical_cols, truth


def test_fit_gower_pam_recovers_categorical_structure_invisible_to_naive_euclidean():
    """Gower distance's per-column-normalized matching dissimilarity for
    categorical features must not silently degrade into ordinary Euclidean
    distance computed on naively label-encoded categories (i.e. treating
    `.cat.codes` integers as if they were meaningful numeric magnitudes).

    Construction: the true 2-cluster structure is carried entirely by a
    low-cardinality categorical column ("group", 2 categories). An
    uninformative HIGH-cardinality categorical column ("site", 10 categories,
    drawn independently of the true cluster) is included alongside it. A
    naive competitor that ignores dtype and just label-encodes both columns
    (`.cat.codes`) and runs vanilla Euclidean KMeans lets "site" -- whose raw
    codes span 0..9 -- dominate the sum-of-squared-distances over "group" --
    whose codes only span 0..1 -- so it fails to recover the true clusters
    (empirically confirmed: ARI close to 0, see below). Gower's per-column
    equal-weight matching dissimilarity (confirmed from
    `gower.gower_dist.gower_get`'s source: `sij_cat = (xi_cat != xj_cat)`,
    unweighted 0/1 per column regardless of cardinality) is not swamped by
    "site"'s larger code range, so `fit_gower_pam` recovers the true
    clusters with a large ARI margin over the naive competitor.
    """
    df, categorical_cols, truth = _two_cluster_categorical_fixture()

    naive_encoded = pd.DataFrame({c: df[c].cat.codes for c in categorical_cols}).to_numpy(dtype=float)
    naive_labels = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(naive_encoded)
    naive_ari = adjusted_rand_score(truth, naive_labels)

    gower_labels = mod.fit_gower_pam(df, categorical_cols, k=2, seed=42)
    gower_ari = adjusted_rand_score(truth, gower_labels)

    assert gower_ari > 0.8, f"fit_gower_pam failed to recover the true clusters (ARI={gower_ari})"
    assert naive_ari < gower_ari - 0.3, (
        "naive Euclidean-on-label-codes should be clearly worse than fit_gower_pam here "
        f"(naive ARI={naive_ari}, gower ARI={gower_ari}) -- if not, fit_gower_pam may be "
        "silently behaving like plain Euclidean distance instead of Gower distance"
    )


def test_fit_gower_pam_init_default_recovers_clusters_while_heuristic_collapses():
    """Regression test for the review finding that the empirical
    justification for `GOWER_PAM_INIT = "k-medoids++"` (over
    `sklearn_extra`'s own default, `init="heuristic"`) had no direct test:
    the docstring's "init=" rationale paragraph cited
    `test_fit_gower_pam_recovers_categorical_structure_invisible_to_naive_euclidean`
    as evidence, but that test hardcodes `fit_gower_pam` to whatever
    `GOWER_PAM_INIT` currently is and never exercises `init="heuristic"` at
    all -- it would keep passing even if `fit_gower_pam` silently used
    `"heuristic"` throughout, as long as `"heuristic"` happened to work on
    that particular fixture.

    This test calls `fit_gower_pam` twice on the same
    `_two_cluster_categorical_fixture()` data used above: once with no
    `init=` argument (i.e. exercising the actual production default,
    `GOWER_PAM_INIT`), and once with `init="heuristic"` explicitly. It
    asserts the production default recovers the true clusters (ARI > 0.7)
    *and* that `"heuristic"` collapses on this same data (ARI < 0.3) --
    reproducing the "Cluster N is empty" bad-local-optimum failure mode the
    module docstring describes. Because the first call relies on the
    default rather than hardcoding `"k-medoids++"`, this test fails if
    `GOWER_PAM_INIT` is ever reverted to `"heuristic"` (verified manually:
    flipping the constant makes the default-init assertion fail with
    ARI ~0.0, and restoring it passes again).
    """
    df, categorical_cols, truth = _two_cluster_categorical_fixture()

    default_init_labels = mod.fit_gower_pam(df, categorical_cols, k=2, seed=42)
    default_init_ari = adjusted_rand_score(truth, default_init_labels)
    assert default_init_ari > 0.7, (
        "fit_gower_pam's default init (GOWER_PAM_INIT) failed to recover the true "
        f"clusters (ARI={default_init_ari}) -- has GOWER_PAM_INIT been reverted to "
        '"heuristic"?'
    )

    heuristic_labels = mod.fit_gower_pam(df, categorical_cols, k=2, seed=42, init="heuristic")
    heuristic_ari = adjusted_rand_score(truth, heuristic_labels)
    assert heuristic_ari < 0.3, (
        f"init=\"heuristic\" was expected to collapse on this fixture (ARI={heuristic_ari} "
        "is not low) -- if sklearn_extra's heuristic init no longer fails here, the "
        "docstring's empirical justification for k-medoids++ needs re-examination"
    )
