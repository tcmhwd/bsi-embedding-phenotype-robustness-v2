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
mod = import_module("02_recluster")

PYTHON = sys.executable

# Runs run_config() on a small synthetic matrix and prints the resulting
# labels as JSON. Executed as a fresh subprocess (not imported into this
# test process) because OMP_NUM_THREADS/NUMBA_NUM_THREADS/MKL_NUM_THREADS
# are read by numpy/numba/sklearn at import time -- changing them after
# this test module has already imported "02_recluster" would have no
# effect, so each thread-count case needs its own process from a clean
# environment.
#
# The synthetic matrix mimics the real data's shape and near-duplicate
# structure (3072-dim, cosine metric, many rows that are noisy copies of a
# small set of base vectors -- like repeated boilerplate phrasing in real
# AST reports), because a plain low-dimensional Gaussian matrix does NOT
# trigger the thread-count sensitivity (verified: matched at 32/64 dims).
# At this shape it reliably differs pre-fix; see class docstring below.
_SUBPROC_SCRIPT = """
import sys
sys.path.insert(0, "__SCRIPTS_DIR__")
from importlib import import_module
import json
import numpy as np

mod = import_module("02_recluster")
rng = np.random.default_rng(0)
base = rng.normal(size=(80, 3072)).astype(np.float32)
idx = rng.integers(0, 80, size=1500)
noise = rng.normal(scale=0.05, size=(1500, 3072)).astype(np.float32)
emb = base[idx] + noise
r = mod.run_config(emb, n_components=10, n_neighbors=30, min_cluster_size=20, seed=42)
print(json.dumps(r["labels"].tolist()))
"""


def _run_with_threads(n_threads: str) -> list:
    env = dict(os.environ)
    for key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = n_threads
    proc = subprocess.run(
        [PYTHON, "-c", _SUBPROC_SCRIPT.replace("__SCRIPTS_DIR__", SCRIPTS_DIR)],
        env=env, capture_output=True, text=True, check=True, timeout=300,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_labels_identical_across_thread_counts():
    """Regression test for the thread-count-dependent clustering defect.

    02_recluster.py must pin OMP/NUMBA/MKL/OPENBLAS thread pools to 1 at
    import time (before numpy/umap/hdbscan/numba are imported) and pass
    n_jobs=1 to umap.UMAP. Without that fix, umap.UMAP's pynndescent
    nearest-neighbour search is parallel and non-deterministic across
    thread counts even with random_state fixed, so labels_1 != labels_24
    for the same input/seed. This was confirmed empirically against the
    real 32,329-row/3072-dim dataset (18 vs 33 clusters at 8 vs 24
    threads); this test reproduces the same failure mode on a small
    synthetic matrix that shares the real data's dimensionality and
    near-duplicate-row structure (see module comment above) so it runs in
    well under a minute rather than requiring the full dataset.
    """
    labels_1 = _run_with_threads("1")
    labels_24 = _run_with_threads("24")
    assert labels_1 == labels_24, (
        "clustering labels differ across thread counts -- reproducibility "
        "fix (thread pinning at import time + n_jobs=1) is not effective"
    )


def test_run_config_is_deterministic():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(2000, 64)).astype(np.float32)
    a = mod.run_config(emb, n_components=10, n_neighbors=15, min_cluster_size=50, seed=42)
    b = mod.run_config(emb, n_components=10, n_neighbors=15, min_cluster_size=50, seed=42)
    assert np.array_equal(a["labels"], b["labels"])


def test_run_config_reports_noise_fraction():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(2000, 64)).astype(np.float32)
    r = mod.run_config(emb, n_components=10, n_neighbors=15, min_cluster_size=50, seed=42)
    assert 0.0 <= r["noise_frac"] <= 1.0
    assert r["n_clusters"] == len(set(r["labels"]) - {-1})


def test_run_config_original_space_n_components_zero():
    # n_components=0 means "cluster the raw embedding directly, no UMAP"
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(500, 32)).astype(np.float32)
    r = mod.run_config(emb, n_components=0, n_neighbors=None, min_cluster_size=50, seed=42)
    assert r["labels"].shape == (500,)
    assert 0.0 <= r["noise_frac"] <= 1.0


def test_build_grid_has_40_configs_with_unambiguous_tags():
    """39 exploratory-grid configs + 1 explicitly-named published config
    (audit finding M1: the grid never contained the published setting)."""
    grid = mod.build_grid()
    assert len(grid) == 40
    tags = [mod.tag_for(c) for c in grid]
    assert len(set(tags)) == 40, "config tags must be unique/unambiguous"

    exploratory = [c for c in grid if not c.get("published")]
    assert len(exploratory) == 39

    n_components_zero = [c for c in exploratory if c["n_components"] == 0]
    assert len(n_components_zero) == 3
    for c in n_components_zero:
        assert c["n_neighbors"] is None
        tag = mod.tag_for(c)
        assert tag.startswith("nc0_mcs")
        assert "nn" not in tag

    n_components_umap = [c for c in exploratory if c["n_components"] != 0]
    assert len(n_components_umap) == 36
    for c in n_components_umap:
        assert c["n_neighbors"] is not None


def test_published_config_is_explicitly_named_and_matches_confirmed_source():
    """Regression test for audit finding M1: the published UMAP+HDBSCAN
    configuration (confirmed against `pipeline_new/phenotype_umap_hdbscan.py`
    and the accepted manuscript's supplementary Table S2, both of which
    agree exactly) must be present in the grid as an explicitly-flagged,
    unambiguously-tagged entry -- not silently folded in as an anonymous
    40th grid point indistinguishable from the exploratory configs.
    """
    grid = mod.build_grid()
    published = [c for c in grid if c.get("published")]
    assert len(published) == 1, "exactly one config must be flagged as published"
    cfg = published[0]
    assert cfg["n_components"] == 2
    assert cfg["n_neighbors"] == 15
    assert cfg["min_dist"] == pytest.approx(0.1)
    assert cfg["metric"] == "euclidean"
    assert cfg["min_cluster_size"] == 161  # 32_329 // 200
    assert cfg["min_samples"] is None
    assert cfg["cluster_selection_method"] == "eom"

    tag = mod.tag_for(cfg)
    assert "published" in tag
    # must not collide with (or be confusable with) any exploratory tag
    other_tags = {mod.tag_for(c) for c in grid if not c.get("published")}
    assert tag not in other_tags


def test_run_config_accepts_published_hyperparameters_without_regressing_grid_defaults():
    """`run_config`'s new min_dist/metric/min_samples/cluster_selection_method
    kwargs must default to the original grid's values (0.0/cosine/10/eom) so
    the 39 exploratory configs are byte-for-byte unaffected, while still
    accepting the published overrides explicitly."""
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(500, 32)).astype(np.float32)
    # Grid-default call (no new kwargs) vs. the same call with the defaults
    # spelled out explicitly must be identical -- proves the new kwargs
    # don't change existing behavior when omitted.
    a = mod.run_config(emb, n_components=10, n_neighbors=15, min_cluster_size=50, seed=42)
    b = mod.run_config(
        emb, n_components=10, n_neighbors=15, min_cluster_size=50, seed=42,
        min_dist=0.0, metric="cosine", min_samples=10, cluster_selection_method="eom",
    )
    assert np.array_equal(a["labels"], b["labels"])
    # Published-style overrides must be accepted without error and produce a
    # valid labeling (min_samples=None is HDBSCAN's own "use min_cluster_size"
    # default, not a crash case).
    r = mod.run_config(
        emb, n_components=2, n_neighbors=15, min_cluster_size=50, seed=42,
        min_dist=0.1, metric="euclidean", min_samples=None, cluster_selection_method="eom",
    )
    assert r["labels"].shape == (500,)
