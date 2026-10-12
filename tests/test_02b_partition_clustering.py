import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from importlib import import_module
mod = import_module("02b_partition_clustering")

PYTHON = sys.executable


def _synthetic_blobs(n_per_cluster=150, n_clusters=4, dim=32, seed=0):
    """4 well-separated isotropic Gaussian blobs -- a minimal validation
    structure that both k-means and GMM should recover easily (following
    the Task2/3 convention of testing the function's contract on small
    synthetic data rather than the real 32,329x3,072 data)."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=8.0, size=(n_clusters, dim))
    rows = [centers[c] + rng.normal(scale=0.5, size=(n_per_cluster, dim)) for c in range(n_clusters)]
    return np.vstack(rows).astype(np.float64)


def test_get_space_raw3072_returns_embedding_unchanged_shape():
    emb = np.random.default_rng(0).normal(size=(200, 3072)).astype(np.float32)
    space = mod.get_space(emb, "raw3072", seed=42)
    assert space.shape == (200, 3072)
    assert space.dtype == np.float64


def test_get_space_umap_reduces_dimension():
    emb = np.random.default_rng(1).normal(size=(300, 128)).astype(np.float32)
    space = mod.get_space(emb, "umap10", seed=42)
    assert space.shape == (300, 10)


def test_fit_kmeans_returns_expected_keys_and_recovers_clusters():
    X = _synthetic_blobs()
    r = mod.fit_kmeans(X, k=4, seed=42)
    assert set(r) == {"labels", "inertia", "silhouette"}
    assert r["labels"].shape == (X.shape[0],)
    assert r["inertia"] > 0
    # well-separated blobs -> k-means should find them cleanly
    assert r["silhouette"] > 0.8


def test_fit_gmm_returns_expected_keys_and_recovers_clusters():
    X = _synthetic_blobs()
    r = mod.fit_gmm(X, k=4, seed=42)
    assert set(r) == {"labels", "bic", "silhouette", "failed", "error"}
    assert r["failed"] is False
    assert r["error"] is None
    assert r["labels"].shape == (X.shape[0],)
    assert isinstance(r["bic"], float)
    assert r["silhouette"] > 0.8


def test_fit_kmeans_is_deterministic_given_seed():
    X = _synthetic_blobs(seed=1)
    a = mod.fit_kmeans(X, k=4, seed=42)
    b = mod.fit_kmeans(X, k=4, seed=42)
    assert np.array_equal(a["labels"], b["labels"])


def test_fit_gmm_is_deterministic_given_seed():
    X = _synthetic_blobs(seed=1)
    a = mod.fit_gmm(X, k=4, seed=42)
    b = mod.fit_gmm(X, k=4, seed=42)
    assert np.array_equal(a["labels"], b["labels"])


# Regression test for the real production crash (SLURM job 586187, umap50
# space, k=16): GaussianMixture.fit() raised
#   ValueError: ... ill-defined empirical covariance (singleton or collapsed
#   samples) ... Try to decrease the number of components, or increase reg_covar.
# This is a genuine sklearn floating-point failure mode, not a hypothetical:
# a component's diag covariance estimate can come out <= 0 due to
# catastrophic cancellation in `avg_X2 - 2*avg_X_means + avg_means2` when a
# component collapses onto a handful of near-duplicate points whose
# coordinates are large in magnitude. reg_covar adds a per-axis floor to
# that estimate, so it must be large enough to survive the cancellation
# error, which scales with the coordinates' magnitude squared.
def _near_singular_blobs_with_collapsed_cluster():
    """3 well-separated blobs plus a 4th, tiny cluster of 10 near-duplicate
    points offset by 1e6 -- empirically confirmed (see task-2b-report.md) to
    make GaussianMixture(reg_covar=1e-6, sklearn's default) raise the exact
    ValueError above at k=4/n_init=10/seed=42, while reg_covar=1e-3 (this
    module's GMM_REG_COVAR) succeeds and correctly isolates the 10 collapsed
    points as their own component.
    """
    dim = 5
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=8.0, size=(3, dim))
    rows = [centers[c] + rng.normal(scale=0.5, size=(30, dim)) for c in range(3)]
    X = np.vstack(rows)
    dup_point = rng.normal(scale=1.0, size=dim) + 1e6
    dup = dup_point + rng.normal(scale=1e-9, size=(10, dim))
    return np.vstack([X, dup]).astype(np.float64)


def test_old_default_reg_covar_actually_reproduces_the_production_crash():
    """Sanity-check the synthetic repro against sklearn's own (pre-fix)
    default reg_covar=1e-6, so the next test's "success" outcome is known to
    be due to GMM_REG_COVAR and not a dataset that was never actually
    at-risk. This does not exercise this module's code -- it exercises
    sklearn directly with the old default, to validate the fixture."""
    from sklearn.mixture import GaussianMixture

    X = _near_singular_blobs_with_collapsed_cluster()
    with pytest.raises(ValueError, match="ill-defined empirical covariance"):
        GaussianMixture(
            n_components=4, covariance_type=mod.GMM_COVARIANCE_TYPE,
            n_init=mod.N_INIT, random_state=42, reg_covar=1e-6,
        ).fit(X)


def test_fit_gmm_handles_near_singular_covariance_without_crashing():
    """Fix 1: fit_gmm must not raise on a (space, k) that makes a component
    collapse onto very few near-duplicate points -- it must either succeed
    (thanks to reg_covar=GMM_REG_COVAR) or return the explicit failed=True /
    error=<message> record, never propagate the ValueError.

    Observed outcome on this dataset/environment: SUCCESS -- GMM_REG_COVAR
    (1e-3) is large enough to survive the cancellation error that makes
    sklearn's own default (1e-6) crash (see the previous test), and the
    10-point collapsed cluster is correctly recovered as its own component.
    """
    X = _near_singular_blobs_with_collapsed_cluster()
    r = mod.fit_gmm(X, k=4, seed=42)

    assert r["failed"] in (True, False)  # never raises -- that's the actual contract
    if r["failed"]:
        assert isinstance(r["error"], str) and r["error"]
        assert r["labels"] is None
    else:
        assert r["error"] is None
        _, counts = np.unique(r["labels"], return_counts=True)
        # the collapsed 10-point micro-cluster should come back as its own component
        assert sorted(counts.tolist()) == [10, 30, 30, 30]


# Coverage gap fix: the tests above only ever exercise fit_gmm's success
# branch (GMM_REG_COVAR=1e-3 is large enough to survive the earlier fixture's
# cancellation error), so the `except ValueError` / failed=True branch itself
# had zero coverage. Pushing the collapsed-cluster offset from 1e6 to 1e9
# (confirmed empirically -- see task-2b-report.md -- to still raise even at
# reg_covar=1e-3, unlike 1e6 which fit_gmm's own reg_covar already survives)
# reproduces a genuine sklearn ValueError inside fit_gmm itself, not a mock.
def _collapsed_cluster_beyond_reg_covar(dim=5, dup_n=10, offset=1e9, dup_scale=1e-9, seed=0):
    """Same construction as `_near_singular_blobs_with_collapsed_cluster` (3
    well-separated blobs + a tiny near-duplicate cluster), but the duplicate
    cluster is offset by 1e9 instead of 1e6. At that magnitude, the
    catastrophic-cancellation error in sklearn's covariance estimate
    (`avg_X2 - 2*avg_X_means + avg_means2`) scales with offset^2 and exceeds
    even GMM_REG_COVAR=1e-3's variance floor, so GaussianMixture.fit() raises
    ValueError('... ill-defined empirical covariance ...') even at this
    module's production reg_covar -- unlike the 1e6 fixture, which
    GMM_REG_COVAR successfully survives (see
    test_fit_gmm_handles_near_singular_covariance_without_crashing).
    """
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=8.0, size=(3, dim))
    rows = [centers[c] + rng.normal(scale=0.5, size=(30, dim)) for c in range(3)]
    X = np.vstack(rows)
    dup_point = rng.normal(scale=1.0, size=dim) + offset
    dup = dup_point + rng.normal(scale=dup_scale, size=(dup_n, dim))
    return np.vstack([X, dup]).astype(np.float64)


def test_fit_gmm_failed_branch_returns_exact_failure_record_shape():
    """Genuinely exercises the `except ValueError` branch in fit_gmm (not a
    mock): the 1e9-offset collapsed cluster above raises ValueError inside
    GaussianMixture.fit() even at this module's production
    reg_covar=GMM_REG_COVAR (1e-3), confirmed directly against sklearn
    (see test_old_default_reg_covar_actually_reproduces_the_production_crash
    for the analogous sanity-check pattern on the 1e6 fixture). Asserts the
    exact dict shape the except block constructs -- not just "no exception".
    """
    X = _collapsed_cluster_beyond_reg_covar()
    r = mod.fit_gmm(X, k=4, seed=42)

    assert r["failed"] is True
    assert r["labels"] is None
    assert r["bic"] is None
    assert r["silhouette"] is None
    assert isinstance(r["error"], str)
    assert "ill-defined empirical covariance" in r["error"]
    assert set(r) == {"labels", "bic", "silhouette", "failed", "error"}


def test_select_k_by_silhouette_picks_true_k_for_well_separated_blobs():
    X = _synthetic_blobs(n_clusters=4)
    result = mod.select_k_by_silhouette(X, k_grid=[2, 3, 4, 6, 8], seed=42)
    assert result["best_k"] == 4
    assert len(result["scan"]) == 5
    assert all({"k", "silhouette", "inertia"} == set(row) for row in result["scan"])


def test_output_tag_is_unambiguous():
    tags = [
        mod.output_tag(algo, space, k)
        for algo in ("kmeans", "gmm")
        for space in mod.SPACE_TAGS
        for k in (8, 16, 32)
    ]
    assert len(tags) == len(set(tags))


# Regression test for the thread-count-dependent clustering defect (same
# mechanism as 02_recluster.py's test_labels_identical_across_thread_counts,
# applied here to KMeans/GaussianMixture instead of UMAP+HDBSCAN -- both
# route their heavy linear algebra through BLAS). Run as a fresh subprocess
# because OMP_NUM_THREADS/etc. are read at import time, so changing them
# after this test module has already imported 02b_partition_clustering would
# have no effect.
_SUBPROC_SCRIPT = """
import sys
sys.path.insert(0, "__SCRIPTS_DIR__")
from importlib import import_module
import json
import numpy as np

mod = import_module("02b_partition_clustering")
rng = np.random.default_rng(0)
centers = rng.normal(scale=8.0, size=(6, 64))
rows = [centers[c] + rng.normal(scale=1.5, size=(300, 64)) for c in range(6)]
X = np.vstack(rows).astype(np.float64)
km = mod.fit_kmeans(X, k=6, seed=42)
gmm = mod.fit_gmm(X, k=6, seed=42)
print(json.dumps({"kmeans": km["labels"].tolist(), "gmm": gmm["labels"].tolist()}))
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
    """02b_partition_clustering.py must pin OMP/NUMBA/MKL/OPENBLAS thread
    pools to 1 at import time (before numpy/sklearn/umap are imported), the
    same fix 02_recluster.py required for HDBSCAN. scikit-learn's KMeans and
    GaussianMixture also route distance/E-step computations through BLAS, so
    without the fix their labels could likewise vary by thread count even
    with random_state fixed.
    """
    r1 = _run_with_threads("1")
    r24 = _run_with_threads("24")
    assert r1["kmeans"] == r24["kmeans"], "kmeans labels differ across thread counts"
    assert r1["gmm"] == r24["gmm"], "gmm labels differ across thread counts"
