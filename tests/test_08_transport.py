import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from importlib import import_module

mod = import_module("08_transport")


# =========================================================================
# select_stratum: whether the source/target split picks the correct rows
# =========================================================================

def test_select_stratum_hospital():
    frame = pd.DataFrame({
        "admission_ID": range(6),
        "hospital_code": [10, 10, 20, 20, 10, 20],
        "admit_year": [2015, 2019, 2016, 2020, 2012, 2022],
    })
    sub10 = mod.select_stratum(frame, "hospital", 10)
    sub20 = mod.select_stratum(frame, "hospital", 20)
    assert list(sub10["admission_ID"]) == [0, 1, 4]
    assert list(sub20["admission_ID"]) == [2, 3, 5]
    assert (sub10["hospital_code"] == 10).all()
    # index reset (0..n-1), required for downstream positional embedding alignment
    assert list(sub10.index) == list(range(len(sub10)))


def test_select_stratum_period():
    frame = pd.DataFrame({
        "admission_ID": range(4),
        "admit_year": [2015, mod.EARLY_YEAR_CUTOFF, mod.EARLY_YEAR_CUTOFF + 1, 2024],
    })
    early = mod.select_stratum(frame, "period", "early")
    late = mod.select_stratum(frame, "period", "late")
    assert list(early["admission_ID"]) == [0, 1]
    assert list(late["admission_ID"]) == [2, 3]
    assert len(early) + len(late) == len(frame)


def test_select_stratum_unknown_axis_or_stratum_raises():
    frame = pd.DataFrame({"admission_ID": [0], "hospital_code": [10], "admit_year": [2020]})
    with pytest.raises(ValueError):
        mod.select_stratum(frame, "bogus_axis", 10)
    with pytest.raises(ValueError):
        mod.select_stratum(frame, "period", "bogus_stratum")


# =========================================================================
# unassigned_fraction: whether the accounting is correct
# =========================================================================

def test_unassigned_fraction():
    labels = np.array([-1, -1, 0, 1, 2])
    assert mod.unassigned_fraction(labels) == pytest.approx(0.4)
    assert mod.unassigned_fraction(np.array([0, 1, 2])) == 0.0
    assert mod.unassigned_fraction(np.array([-1, -1])) == 1.0


# =========================================================================
# Whether transport actually uses transform()+approximate_predict()
# (not a refit) -- verified by monkeypatching to forbid refit calls.
# =========================================================================

def _make_source_target_emb(seed=0, n_source=400, n_target=200, n_base=40, dim=32):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n_base, dim)).astype(np.float32)
    idx_s = rng.integers(0, n_base, size=n_source)
    source_emb = base[idx_s] + rng.normal(scale=0.05, size=(n_source, dim)).astype(np.float32)
    idx_t = rng.integers(0, n_base, size=n_target)
    target_emb = base[idx_t] + rng.normal(scale=0.05, size=(n_target, dim)).astype(np.float32)
    return source_emb, target_emb


def test_fit_source_clustering_returns_fitted_models():
    source_emb, _ = _make_source_target_emb()
    fit = mod.fit_source_clustering(
        source_emb, min_cluster_size=20, seed=42, n_components=5, n_neighbors=15,
    )
    assert fit["reducer"] is not None
    assert fit["clusterer"] is not None
    assert hasattr(fit["reducer"], "transform")
    # HDBSCAN must have been fit with prediction_data=True so approximate_predict works
    assert hasattr(fit["clusterer"], "prediction_data_")


def test_transport_predict_uses_transform_not_refit(monkeypatch):
    source_emb, target_emb = _make_source_target_emb()
    fit = mod.fit_source_clustering(
        source_emb, min_cluster_size=20, seed=42, n_components=5, n_neighbors=15,
    )

    import umap
    import hdbscan

    def _boom(*a, **k):
        raise AssertionError(
            "must not refit -- transport_predict should call reducer.transform() + "
            "hdbscan.approximate_predict() only, never UMAP.fit()/fit_transform() or "
            "HDBSCAN.fit() again"
        )

    monkeypatch.setattr(umap.UMAP, "fit", _boom)
    monkeypatch.setattr(umap.UMAP, "fit_transform", _boom)
    monkeypatch.setattr(hdbscan.HDBSCAN, "fit", _boom)

    labels, strengths, space = mod.transport_predict(fit, target_emb)
    assert labels.shape == (target_emb.shape[0],)
    assert strengths.shape == (target_emb.shape[0],)
    assert space.shape == (target_emb.shape[0], 5)


def test_transport_predict_requires_fitted_models():
    with pytest.raises(ValueError):
        mod.transport_predict({"reducer": None, "clusterer": None}, np.zeros((5, 8)))


def test_transport_predict_is_deterministic():
    source_emb, target_emb = _make_source_target_emb(seed=3)
    fit = mod.fit_source_clustering(
        source_emb, min_cluster_size=20, seed=42, n_components=5, n_neighbors=15,
    )
    labels1, strengths1, _ = mod.transport_predict(fit, target_emb)
    labels2, strengths2, _ = mod.transport_predict(fit, target_emb)
    assert np.array_equal(labels1, labels2)
    assert np.allclose(strengths1, strengths2)


# =========================================================================
# Organism composition comparison
# =========================================================================

def test_organism_proportions_and_cosine_similarity():
    cats = ["Ecoli", "Klebsiella", "Other"]
    s = pd.Series(["Ecoli", "Ecoli", "Klebsiella", "Other"])
    v = mod.organism_proportions(s, cats)
    assert np.allclose(v, [0.5, 0.25, 0.25])
    assert mod.cosine_similarity(v, v) == pytest.approx(1.0)
    assert mod.cosine_similarity(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])) == pytest.approx(0.0)


def test_organism_proportions_empty_series_is_zero_vector():
    v = mod.organism_proportions(pd.Series([], dtype=object), ["a", "b"])
    assert np.array_equal(v, np.zeros(2))


def test_cosine_similarity_zero_vector_is_nan():
    assert np.isnan(mod.cosine_similarity(np.zeros(3), np.array([1.0, 0.0, 0.0])))


# =========================================================================
# Whole-cohort cluster mapping (for inheriting Task 3 stability)
# =========================================================================

def test_best_whole_cohort_cluster_matches_overlap():
    source_labels = np.array([0, 0, 0, 1, 1, -1])
    whole_labels = np.array([5, 5, 7, 9, 9, 9])
    j, score = mod.best_whole_cohort_cluster(source_labels, whole_labels, 0)
    assert j == 5
    assert score == pytest.approx(2 / 3)


def test_best_whole_cohort_cluster_empty_source_cluster_returns_none():
    source_labels = np.array([0, 0, -1])
    whole_labels = np.array([1, 1, 1])
    j, score = mod.best_whole_cohort_cluster(source_labels, whole_labels, 5)
    assert j is None
    assert np.isnan(score)


# =========================================================================
# build_metrics_table: integration-level -- whether columns/values are consistent
# =========================================================================

def test_build_metrics_table_shape_and_values():
    source_frame = pd.DataFrame({
        "admission_ID": [1, 2, 3, 4, 5, 6],
        "BSI_episode_short_clean": ["Ecoli", "Ecoli", "Klebsiella", "Ecoli", "Klebsiella", "Other"],
    })
    source_labels = np.array([0, 0, 0, 1, 1, -1])

    target_frame = pd.DataFrame({
        "admission_ID": [7, 8, 9],
        "BSI_episode_short_clean": ["Ecoli", "Klebsiella", "Ecoli"],
    })
    target_labels = np.array([0, 1, -1])

    whole_labels_df = pd.DataFrame({
        "admission_ID": [1, 2, 3, 4, 5, 6],
        "label": [10, 10, 10, 20, 20, 20],
    })
    stable_clusters = {10}
    categories = ["Ecoli", "Klebsiella", "Other"]

    metrics = mod.build_metrics_table(
        source_frame, source_labels, target_frame, target_labels,
        whole_labels_df, stable_clusters, categories,
    )
    assert set(metrics["source_cluster"]) == {0, 1}  # noise (-1) excluded

    row0 = metrics.loc[metrics["source_cluster"] == 0].iloc[0]
    assert row0["source_n"] == 3
    assert row0["target_n"] == 1  # one target row assigned to label 0
    assert row0["whole_cohort_stable"] == True  # noqa: E712 -- best match is cluster 10
    assert not row0["empty_in_target"]

    row1 = metrics.loc[metrics["source_cluster"] == 1].iloc[0]
    assert row1["source_n"] == 2
    assert row1["target_n"] == 1
    assert row1["whole_cohort_stable"] == False  # noqa: E712 -- best match is cluster 20 (unstable)


# =========================================================================
# Part D (Task 10 expansion) -- organism-filtered transport helpers
# =========================================================================

def test_derive_min_cluster_size_for_stratum_matches_task10_rule():
    # Same rule as 10_within_pathogen.scaled_min_cluster_sizes's primary value
    # (n // 200, floored at 50) -- deliberately duplicated locally rather than
    # cross-imported (this repo's established convention), so pin the values.
    assert mod.derive_min_cluster_size_for_stratum(6400) == 50   # 6400//200=32 -> floor
    assert mod.derive_min_cluster_size_for_stratum(1868) == 50   # 1868//200=9 -> floor
    assert mod.derive_min_cluster_size_for_stratum(20000) == 100  # 20000//200=100, above floor


def test_out_dir_for_no_filter_is_unchanged():
    assert mod._out_dir_for(None) == mod.OUT


def test_out_dir_for_organism_filter_is_a_sibling_directory():
    out = mod._out_dir_for("Ecoli")
    assert out != mod.OUT
    assert out.parent == mod.OUT.parent
    assert out.name == "08_transport_ecoli"


def test_stratum_mask_matches_select_stratum():
    frame = pd.DataFrame({
        "admission_ID": range(6),
        "hospital_code": [10, 10, 20, 20, 10, 20],
        "admit_year": [2015, 2019, 2016, 2020, 2012, 2022],
    })
    mask = mod._stratum_mask(frame, "hospital", 10)
    expected = mod.select_stratum(frame, "hospital", 10)
    assert list(frame.loc[mask, "admission_ID"]) == list(expected["admission_ID"])


def test_aggregate_separates_skipped_from_missing(tmp_path):
    import json as _json

    # direction 0 has a full set of outputs (not skipped, not missing)
    d0 = mod.DIRECTIONS[0]["name"]
    (tmp_path / f"summary_{d0}.json").write_text(_json.dumps({"direction": d0, "organism_filter": "Ecoli"}))
    (tmp_path / f"FRAGMENT_{d0}.md").write_text(f"## {d0}\n")
    pd.DataFrame({"source_cluster": [0]}).to_csv(tmp_path / f"metrics_{d0}.csv", index=False)

    # direction 1 was skipped by main()'s too-small-stratum guard: only a
    # minimal summary_*.json exists (no FRAGMENT/metrics).
    d1 = mod.DIRECTIONS[1]["name"]
    (tmp_path / f"summary_{d1}.json").write_text(_json.dumps({
        "direction": d1, "skipped": True, "skip_reason": "too small", "source_n": 10, "target_n": 5000,
    }))
    # directions 2/3 never ran at all -> genuinely missing

    combined = mod.aggregate(tmp_path, organism_filter="Ecoli")
    assert combined["skipped_directions"] == [d1]
    assert set(combined["missing_directions"]) == {mod.DIRECTIONS[2]["name"], mod.DIRECTIONS[3]["name"]}
    text = (tmp_path / "FINDINGS.md").read_text()
    assert "skipped" in text
    assert d1 in text
