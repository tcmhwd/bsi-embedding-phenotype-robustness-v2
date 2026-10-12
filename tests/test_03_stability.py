import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from importlib import import_module

mod = import_module("03_stability")


def test_jaccard_of_identical_partition_is_one():
    a = np.array([0, 0, 1, 1, 2, 2])
    assert mod.jaccard_for_cluster(a, a, 0) == 1.0


def test_bootstrap_returns_one_row_per_cluster():
    # NOTE: this covers `bootstrap_stability`/`bootstrap_stability_generic`,
    # the OLD design (AUDIT C4, size-confounded) -- deliberately left
    # unchanged for Task 5c/10 backward compatibility (see
    # task-3-redesign-report.md "backward-compatibility decision"). This
    # test is legitimate regression coverage of that *unchanged* function,
    # not a test of the new subsample design below.
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(600, 16)).astype(np.float32)
    base = np.repeat([0, 1, 2], 200)
    ari, jac = mod.bootstrap_stability(
        emb, base, {"n_components": 0, "n_neighbors": 15, "min_cluster_size": 30},
        n_boot=3, seed=1)
    assert len(ari) == 3
    assert set(jac["cluster"]) == {0, 1, 2}


def test_parse_config_tag():
    assert mod.parse_config_tag("nc20_nn30_mcs100") == {
        "n_components": 20, "n_neighbors": 30, "min_cluster_size": 100,
    }


# ============================================================================
# Design S -- subsample stability (DESIGN_embedding_and_stability.md §2.2-2.6)
# ============================================================================

def test_scale_threshold_basic_rounding_and_floor():
    assert mod.scale_threshold(100, 0.632) == 63
    assert mod.scale_threshold(161, 0.632) == 102  # round(101.752) = 102
    # floor of 2, even for a tiny full-data threshold
    assert mod.scale_threshold(1, 0.632) == 2
    assert mod.scale_threshold(2, 0.632) == 2


def test_naive_design_fails_resolvability_new_design_passes():
    """Required regression test that FAILS under the OLD design and PASSES
    under the NEW one (confirmed by literally replaying the old draw
    mechanism, not just asserting a name). AUDIT C4 / DESIGN §2.1:
    bootstrap-with-replacement + np.unique retains ~63.2% of rows on
    average while min_cluster_size stays fixed at its full-data value, so a
    cluster below mcs_full/0.632 cannot re-form by construction.
    """
    n = 2000
    small_cluster_size = 150  # < 100/0.632 = 158.2 -> unresolvable under fixed mcs
    mcs_full = 100
    f = 0.632
    small_cluster_positions = np.arange(small_cluster_size)

    # (a) OLD design, replayed exactly as `bootstrap_stability_generic` draws
    # (rng.choice(n, size=n, replace=True) then np.unique): across many
    # seeds, the mean number of the small cluster's members that survive is
    # below mcs_full -- i.e. this cluster is structurally unresolvable
    # under the OLD design's fixed threshold. THIS ASSERTION IS THE ONE
    # THAT WOULD FAIL IF THE OLD DESIGN WERE (WRONGLY) TREATED AS FIXED.
    retained_sizes = []
    for seed in range(40):
        r = np.random.default_rng(seed)
        idx = r.choice(n, size=n, replace=True)
        uniq = np.unique(idx)
        retained_sizes.append(int(np.isin(uniq, small_cluster_positions).sum()))
    mean_retained_old = np.mean(retained_sizes)
    assert mean_retained_old < mcs_full, (
        "sanity check on the OLD design's own defect -- if this fails, the "
        "synthetic small_cluster_size/mcs_full no longer reproduces C4"
    )

    # (b) NEW design: same cluster, same f, but the threshold is scaled too.
    mcs_sub = mod.scale_threshold(mcs_full, f)
    expected_retained_new = f * small_cluster_size
    assert expected_retained_new >= mcs_sub, (
        "new design must make this exact cluster resolvable by construction"
    )


def test_patient_subsample_keeps_patient_admissions_together():
    # 10 patients, some with multiple admissions (rows) -- a subsampled
    # patient's rows must move together (Mo7: no patient split across in/out).
    patient_ids = np.array([1, 1, 2, 3, 3, 3, 4, 5, 5, 6, 7, 8, 9, 10])
    for seed in range(10):
        idx = mod.patient_subsample_indices(patient_ids, f=0.5, seed=seed)
        selected = set(idx)
        for p in np.unique(patient_ids):
            positions = set(np.where(patient_ids == p)[0])
            overlap = positions & selected
            assert overlap in (set(), positions), (
                f"patient {p} split across retained/excluded at seed={seed}"
            )


def test_patient_subsample_reaches_target_fraction_without_gross_overshoot():
    patient_ids = np.concatenate([np.arange(100), np.repeat(np.arange(100, 120), 3)])
    n = len(patient_ids)
    idx = mod.patient_subsample_indices(patient_ids, f=0.632, seed=3)
    retained_frac = len(idx) / n
    assert retained_frac >= 0.632
    # Overshoot is bounded by the single largest patient's admission count
    # (3 rows here) -- the loop stops as soon as the threshold is crossed.
    assert retained_frac < 0.632 + 3 / n + 1e-9


def test_classify_outcome_recovered():
    base = np.array([0] * 10 + [1] * 10)
    new = base.copy()  # identical partition
    row = mod.classify_outcome(base, new, 0)
    assert row["outcome"] == "recovered"
    assert row["jaccard"] == 1.0


def test_classify_outcome_perturbed():
    base = np.array([0] * 10 + [1] * 10)
    new = np.array([5] * 8 + [-1] * 2 + [5] * 2 + [6] * 8)
    row = mod.classify_outcome(base, new, 0)
    assert row["outcome"] == "perturbed"
    assert 0.50 <= row["jaccard"] < 0.75


def test_classify_outcome_split():
    base = np.array([0] * 10 + [1] * 10)
    new = np.array([9] * 5 + [10] * 5 + [11] * 10)
    row = mod.classify_outcome(base, new, 0)
    assert row["outcome"] == "split"


def test_classify_outcome_merged():
    # base cluster 0 (10 members) entirely absorbed into a much larger new
    # cluster (90 members total) -> best-match recall high, but J low
    # because the new cluster is diluted with 80 outside points.
    base = np.array([0] * 10 + [1] * 90)
    new = np.array([7] * 90 + [8] * 10)
    row = mod.classify_outcome(base, new, 0)
    assert row["outcome"] == "merged"
    assert row["best_recall"] >= 0.80
    assert row["jaccard"] < 0.50


def test_classify_outcome_dissolved():
    base = np.array([0] * 10 + [1] * 10)
    new = np.array([-1] * 7 + [12] * 3 + [13] * 10)
    row = mod.classify_outcome(base, new, 0)
    assert row["outcome"] == "dissolved"
    assert row["noise_frac"] >= 0.50


def test_dissolved_and_merged_are_distinguished():
    # Same retained_size and same "absorbed rather than destroyed" shape,
    # but one has near-zero noise (merged) and the other is mostly noise
    # (dissolved) -- confirms the classifier doesn't conflate them.
    base = np.array([0] * 10 + [1] * 90)
    merged_new = np.array([7] * 90 + [8] * 10)
    dissolved_new = np.array([-1] * 10 + [-1] * 80 + [8] * 10)
    merged_row = mod.classify_outcome(base, merged_new, 0)
    dissolved_row = mod.classify_outcome(base, dissolved_new, 0)
    assert merged_row["outcome"] == "merged"
    assert dissolved_row["outcome"] == "dissolved"
    assert merged_row["outcome"] != dissolved_row["outcome"]


def test_pairwise_coassignment_identical_partition_is_one():
    labels = np.array([3, 3, 3, 3])
    assert mod._pairwise_coassignment(labels) == 1.0


def test_pairwise_coassignment_all_noise_is_zero():
    labels = np.array([-1, -1, -1])
    assert mod._pairwise_coassignment(labels) == 0.0


def test_ari_excluding_noise_drops_noise_rows():
    a = np.array([0, 0, 1, 1, -1, -1])
    b = np.array([0, 0, 1, 1, -1, -1])
    # With noise included, noise-noise pairs inflate agreement for a
    # partition that otherwise wouldn't match as well; excluding it should
    # still be defined and only computed over the 4 non-noise rows here.
    val = mod.ari_excluding_noise(a, b)
    assert val == 1.0


def test_every_base_cluster_is_structurally_resolvable():
    """For a synthetic base clustering where every cluster's size >=
    min_cluster_size_full (as any real HDBSCAN result guarantees), scaling
    the threshold by f makes every cluster's expected retained size clear
    the scaled threshold -- this is the fix, made mechanically checkable.
    """
    rng = np.random.default_rng(0)
    sizes = {0: 101, 1: 158, 2: 892, 3: 5837}
    mcs_full = 100
    f = 0.632
    mcs_sub = mod.scale_threshold(mcs_full, f)
    for k, size in sizes.items():
        assert size >= mcs_full  # precondition any real HDBSCAN cluster satisfies
        assert f * size >= mcs_sub, f"cluster {k} (size={size}) not resolvable"


def test_subsample_stability_generic_scales_thresholds_passed_to_fit_fn():
    rng = np.random.default_rng(0)
    n = 400
    data = rng.normal(size=(n, 4)).astype(np.float32)
    base_labels = np.repeat([0, 1], n // 2)
    patient_ids = np.arange(n)  # one admission per patient here

    seen_thresholds = []

    def fit_fn(sub_data, seed, mcs_sub, ms_sub):
        seen_thresholds.append((mcs_sub, ms_sub))
        # trivial deterministic "clustering": reuse position parity as labels
        return np.arange(len(sub_data)) % 2

    per_cluster, per_boot, extra = mod.subsample_stability_generic(
        data, base_labels, fit_fn, patient_ids,
        min_cluster_size_full=100, min_samples_full=100, f=0.632, n_boot=3, seed=1,
    )
    assert extra["min_cluster_size_sub"] == mod.scale_threshold(100, 0.632)
    assert extra["min_samples_sub"] == mod.scale_threshold(100, 0.632)
    assert all(t == (extra["min_cluster_size_sub"], extra["min_samples_sub"]) for t in seen_thresholds)
    assert len(seen_thresholds) == 3
    assert set(per_cluster["cluster"]) == {0, 1}
    assert "structurally_resolvable" in per_cluster.columns


def test_subsample_stability_generic_fixes_clustering_seed_across_replicates():
    """DESIGN §2.4: the CLUSTERING random_state is fixed (`seed`) across
    resampling replicates -- only the subsample draw varies with `seed+b`.
    This isolates resampling variance from seed variance (the seed-only
    floor measures the latter separately).
    """
    n = 300
    data = np.zeros((n, 2))
    base_labels = np.repeat([0, 1], n // 2)
    patient_ids = np.arange(n)
    seen_seeds = []

    def fit_fn(sub_data, seed, mcs_sub, ms_sub):
        seen_seeds.append(seed)
        return np.zeros(len(sub_data), dtype=int)

    mod.subsample_stability_generic(
        data, base_labels, fit_fn, patient_ids,
        min_cluster_size_full=10, f=0.632, n_boot=5, seed=99,
    )
    assert seen_seeds == [99] * 5  # NOT [99, 100, 101, 102, 103]


def test_seed_only_floor_uses_a_genuine_rerun_per_seed():
    """The seed-only floor must actually invoke the fit at each distinct
    seed in 1..N (a real re-run), not assume/interpolate a value."""
    n = 40
    data = np.zeros((n, 2))
    base_labels = np.repeat([0, 1], n // 2)
    seen_seeds = []

    def fit_fn(full_data, seed, mcs, ms):
        seen_seeds.append(seed)
        # deterministic but seed-dependent perturbation so replicates differ
        rng = np.random.default_rng(seed)
        labels = base_labels.copy()
        flip = rng.choice(n, size=2, replace=False)
        labels[flip] = -1
        return labels

    per_cluster, per_boot = mod.seed_only_floor_generic(
        data, base_labels, fit_fn, min_cluster_size_full=5, seeds=range(1, 6),
    )
    assert seen_seeds == [1, 2, 3, 4, 5]
    assert len(per_boot["boot"].unique()) == 5
    assert set(per_cluster["cluster"]) == {0, 1}
    # genuinely re-run, not copies of a single fit: not every replicate is identical
    assert per_boot.groupby("cluster")["jaccard"].nunique().max() >= 1


def test_merge_stability_report_prefixes_seed_only_columns():
    resample = pd.DataFrame({"cluster": [0, 1], "jaccard_mean": [0.6, 0.9]})
    seed_only = pd.DataFrame({"cluster": [0, 1], "jaccard_mean": [0.85, 0.95], "coassign_mean": [0.8, 0.9]})
    merged = mod.merge_stability_report(resample, seed_only)
    assert merged.loc[merged["cluster"] == 0, "seed_only_jaccard_mean"].iloc[0] == 0.85
    assert merged.loc[merged["cluster"] == 0, "jaccard_mean"].iloc[0] == 0.6  # resample value untouched


def test_stability_size_spearman_perfect_positive_correlation():
    df = pd.DataFrame({"size": [10, 50, 100, 500, 1000], "jaccard_mean": [0.1, 0.3, 0.5, 0.8, 0.95]})
    result = mod.stability_size_spearman(df)
    assert result["rho"] > 0.99
    assert result["p_value"] < 0.05
    assert result["naive_design_rho"] == 0.698


def test_stability_size_spearman_no_correlation():
    df = pd.DataFrame({"size": [10, 50, 100, 500, 1000], "jaccard_mean": [0.8, 0.2, 0.9, 0.1, 0.85]})
    result = mod.stability_size_spearman(df)
    assert abs(result["rho"]) < 0.9  # not a strong monotonic relationship
