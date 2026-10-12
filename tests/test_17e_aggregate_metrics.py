"""Tests for `17e_aggregate_metrics.py` -- the aggregation step that turns
the raw S1/S2/S4/S5 per-run outputs into the survival matrix Task 17's
headline pass counts are read from.

These are not "does the whole 104-run pipeline execute" tests (that would
need real S4/S5 transport output on disk) -- they build small synthetic
fixture directories, monkeypatch the module's I/O path constants onto them
(the same technique used to verify the N9 threshold-extraction fix produced
byte-identical output), and assert on the aggregation LOGIC: thresholds
match the pre-specification verbatim (a silent threshold change is caught),
no cluster is silently dropped from the output, the S4/S5 "must pass BOTH
directions" and "relative-to-null-median" rules are actually enforced, and
row count survives a full `main()` run end to end.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from importlib import import_module

mod = import_module("17e_aggregate_metrics")


# =========================================================================
# Canary: thresholds must match PRESPEC_stress_criteria.md section 3
# verbatim -- catches a silent threshold edit before it ever reaches a
# survival-matrix number.
# =========================================================================

def test_thresholds_match_prespec_verbatim():
    assert mod.S1_THRESHOLDS == {
        "J_seed_mean_min": 0.75, "J_seed_p05_min": 0.50, "J_thread24_min": 0.75,
    }
    assert mod.S2_THRESHOLDS == {
        "jaccard_mean_min": 0.75, "frac_dissolved_max": 0.20, "coassign_mean_min": 0.75,
    }
    assert mod.S4_S5_THRESHOLDS == {"t_abs_min": 0.50, "t_rel_null_frac": 0.80}


# =========================================================================
# aggregate_s1
# =========================================================================

def _write_seed_json(path, cluster_jaccards: dict, n_clusters=2, noise_frac=0.01):
    path.write_text(json.dumps({
        "n_clusters": n_clusters, "noise_frac": noise_frac,
        "jaccard_vs_published": {str(k): v for k, v in cluster_jaccards.items()},
        "agreement_vs_published": {"ari_all_row": 1.0, "ari_noise_excl": 1.0},
    }))


def test_aggregate_s1_computes_mean_p05_and_pass_boundary(tmp_path, monkeypatch):
    s1_dir = tmp_path / "s1_seed"
    s1_dir.mkdir()
    monkeypatch.setattr(mod, "S1_DIR", s1_dir)
    monkeypatch.setattr(mod, "N_SEED_REPS", 20)

    # cluster 0: perfect Jaccard at every seed and both thread counts -> PASS.
    # cluster 1: Jaccard 0.0 at 2/20 seeds (the rest 1.0) -> mean is still a
    # comfortable 0.90, but the bimodality this project's own seed-refit data
    # actually shows (PRESPEC section 7: some published clusters disappear
    # entirely at some seeds) must be caught by the p05 term even though the
    # mean alone would pass -- exactly what p05 exists to prevent going unnoticed.
    for s in range(1, 21):
        jac1 = 0.0 if s in (1, 2) else 1.0
        _write_seed_json(s1_dir / f"result_seed{s}_threads1.json", {0: 1.0, 1: jac1})
    _write_seed_json(s1_dir / "result_seed42_threads1.json", {0: 1.0, 1: 1.0})
    _write_seed_json(s1_dir / "result_seed42_threads24.json", {0: 1.0, 1: 1.0})

    s1_df, per_seed_k, r1, r24 = mod.aggregate_s1([0, 1])
    row0 = s1_df.set_index("cluster").loc[0]
    row1 = s1_df.set_index("cluster").loc[1]

    assert row0["J_seed_mean"] == pytest.approx(1.0)
    assert bool(row0["S1_pass"]) is True

    assert row1["J_seed_mean"] == pytest.approx(0.90)
    assert row1["J_seed_mean"] >= mod.S1_THRESHOLDS["J_seed_mean_min"], (
        "test setup sanity: the mean alone must clear 0.75 here, so a pass "
        "requires the p05 term to be the thing that actually catches this cluster"
    )
    assert row1["J_seed_p05"] < mod.S1_THRESHOLDS["J_seed_p05_min"]
    assert bool(row1["S1_pass"]) is False, (
        "mean clears 0.75 but p05 does not -- S1_pass must still be False "
        "(guards against the p05 conjunct being silently dropped)"
    )
    assert len(per_seed_k) == 20


def test_aggregate_s1_missing_seed_file_raises_not_silently_skips(tmp_path, monkeypatch):
    s1_dir = tmp_path / "s1_seed"
    s1_dir.mkdir()
    monkeypatch.setattr(mod, "S1_DIR", s1_dir)
    monkeypatch.setattr(mod, "N_SEED_REPS", 20)
    # Only write 19 of the 20 required seed files.
    for s in range(1, 20):
        _write_seed_json(s1_dir / f"result_seed{s}_threads1.json", {0: 1.0})
    _write_seed_json(s1_dir / "result_seed42_threads1.json", {0: 1.0})
    _write_seed_json(s1_dir / "result_seed42_threads24.json", {0: 1.0})

    with pytest.raises(FileNotFoundError):
        mod.aggregate_s1([0])


# =========================================================================
# aggregate_s2 -- must not silently drop a cluster with zero bootstrap reps
# =========================================================================

def test_aggregate_s2_keeps_a_row_for_a_cluster_with_zero_bootstrap_reps(tmp_path, monkeypatch):
    s2_dir = tmp_path / "s2_boot"
    s2_dir.mkdir()
    monkeypatch.setattr(mod, "S2_DIR", s2_dir)

    # cluster 0 has bootstrap replicates; cluster 1 (present in `clusters`
    # but never sampled -- e.g. a tiny cluster that dissolved every rep and
    # was recorded under a different label) has none in the shard CSV.
    pd.DataFrame({
        "cluster": [0] * 10, "boot": list(range(10)),
        "jaccard": [0.9] * 10, "coassign": [0.9] * 10, "outcome": ["stable"] * 10,
    }).to_csv(s2_dir / "shard_0.csv", index=False)

    s2_df, n_boot_present = mod.aggregate_s2([0, 1])
    assert len(s2_df) == 2, "cluster with zero bootstrap reps must still get a row, not be dropped"
    row1 = s2_df.set_index("cluster").loc[1]
    assert row1["n_boot_rep"] == 0
    assert bool(row1["S2_pass"]) is False
    assert np.isnan(row1["jaccard_mean"])


def test_aggregate_s2_pass_requires_all_three_conditions(tmp_path, monkeypatch):
    s2_dir = tmp_path / "s2_boot"
    s2_dir.mkdir()
    monkeypatch.setattr(mod, "S2_DIR", s2_dir)

    # cluster 0: passes all three. cluster 1: jaccard/coassign fine but
    # frac_dissolved over the max -- must fail despite the other two passing
    # (guards against the AND being silently weakened to an OR / majority).
    rows = []
    for b in range(10):
        rows.append({"cluster": 0, "boot": b, "jaccard": 0.9, "coassign": 0.9, "outcome": "stable"})
    for b in range(10):
        dissolved = b < 3  # 30% dissolved > 20% max
        rows.append({
            "cluster": 1, "boot": b, "jaccard": 0.9, "coassign": 0.9,
            "outcome": "dissolved" if dissolved else "stable",
        })
    pd.DataFrame(rows).to_csv(s2_dir / "shard_0.csv", index=False)

    s2_df, _ = mod.aggregate_s2([0, 1])
    s2_df = s2_df.set_index("cluster")
    assert bool(s2_df.loc[0, "S2_pass"]) is True
    assert s2_df.loc[1, "frac_dissolved"] == pytest.approx(0.3)
    assert bool(s2_df.loc[1, "S2_pass"]) is False


# =========================================================================
# _s4_s5_pass_table -- both directions required; relative-to-null-median
# term must actually gate (not just the absolute floor)
# =========================================================================

def _write_percluster_csv(path, cluster_T: dict):
    pd.DataFrame({
        "published_cluster": list(cluster_T.keys()), "T": list(cluster_T.values()),
    }).to_csv(path, index=False)


def test_s4_s5_requires_both_directions_to_pass(tmp_path, monkeypatch):
    s4s5_dir = tmp_path / "s4s5_transport"
    s4s5_dir.mkdir()
    monkeypatch.setattr(mod, "S4S5_DIR", s4s5_dir)
    monkeypatch.setattr(mod, "N_NULL_PER_DIRECTION", 5)

    directions = ["dir_a", "dir_b"]
    # cluster 0 clears the bar in BOTH directions -> overall PASS.
    # cluster 1 clears it in dir_a only, fails (T below null-relative bar) in
    # dir_b -> overall must FAIL (a single failing direction must veto).
    _write_percluster_csv(s4s5_dir / "observed_dir_a_percluster.csv", {0: 0.9, 1: 0.9})
    _write_percluster_csv(s4s5_dir / "observed_dir_b_percluster.csv", {0: 0.9, 1: 0.55})
    for rep in range(5):
        _write_percluster_csv(s4s5_dir / f"null_dir_a_seed{rep}_percluster.csv", {0: 0.1, 1: 0.1})
        _write_percluster_csv(s4s5_dir / f"null_dir_b_seed{rep}_percluster.csv", {0: 0.1, 1: 0.7})

    out = mod._s4_s5_pass_table(directions, "test_axis", [0, 1]).set_index("cluster")
    assert bool(out.loc[0, "PASS"]) is True
    assert bool(out.loc[1, "PASS"]) is False, (
        "cluster clears the absolute floor in both directions but the "
        "relative-to-null-median term fails in dir_b -- must still fail overall"
    )


def test_s4_s5_relative_null_term_is_enforced_not_just_absolute_floor(tmp_path, monkeypatch):
    s4s5_dir = tmp_path / "s4s5_transport"
    s4s5_dir.mkdir()
    monkeypatch.setattr(mod, "S4S5_DIR", s4s5_dir)
    monkeypatch.setattr(mod, "N_NULL_PER_DIRECTION", 5)

    # T_obs=0.55 clears the absolute floor (0.50) but the null median is 0.60,
    # so 0.80*null_median=0.48 -- 0.55 >= 0.48 so this one actually should
    # PASS; construct a null high enough that 0.80*median exceeds T_obs.
    _write_percluster_csv(s4s5_dir / "observed_only_percluster.csv", {0: 0.55})
    for rep in range(5):
        _write_percluster_csv(s4s5_dir / f"null_only_seed{rep}_percluster.csv", {0: 0.80})
    # 0.80 * null_median(0.80) = 0.64 > T_obs(0.55) -> must fail despite
    # clearing the absolute 0.50 floor.
    out = mod._s4_s5_pass_table(["only"], "test_axis", [0]).set_index("cluster")
    assert bool(out.loc[0, "PASS"]) is False


def test_s4_s5_missing_cluster_in_a_direction_fails_closed(tmp_path, monkeypatch):
    s4s5_dir = tmp_path / "s4s5_transport"
    s4s5_dir.mkdir()
    monkeypatch.setattr(mod, "S4S5_DIR", s4s5_dir)
    monkeypatch.setattr(mod, "N_NULL_PER_DIRECTION", 5)

    _write_percluster_csv(s4s5_dir / "observed_only_percluster.csv", {0: 0.9})  # cluster 1 absent
    for rep in range(5):
        _write_percluster_csv(s4s5_dir / f"null_only_seed{rep}_percluster.csv", {0: 0.1})

    out = mod._s4_s5_pass_table(["only"], "test_axis", [0, 1]).set_index("cluster")
    assert len(out) == 2, "cluster absent from a direction's output must still get a row"
    assert bool(out.loc[1, "PASS"]) is False


# =========================================================================
# main() end-to-end: row count preserved, ALL_AXES_pass is the AND of the
# four individual axis passes (not silently weakened to an OR/majority).
# =========================================================================

def test_main_end_to_end_preserves_row_count_and_all_axes_is_conjunction(tmp_path, monkeypatch):
    clusters = [0, 1]
    ref = pd.DataFrame({
        "phenotype_cluster": [0] * 5 + [1] * 5 + [-1] * 2,
    })
    monkeypatch.setattr(mod.c17, "load_reference_frame", lambda: ref)

    s1_dir, s2_dir, s4s5_dir = tmp_path / "s1", tmp_path / "s2", tmp_path / "s4s5"
    s1_dir.mkdir(); s2_dir.mkdir(); s4s5_dir.mkdir()
    monkeypatch.setattr(mod, "S1_DIR", s1_dir)
    monkeypatch.setattr(mod, "S2_DIR", s2_dir)
    monkeypatch.setattr(mod, "S4S5_DIR", s4s5_dir)
    monkeypatch.setattr(mod, "TOKEN_PATH", tmp_path / "does_not_exist.csv")
    monkeypatch.setattr(mod, "OUT", tmp_path)
    monkeypatch.setattr(mod, "N_NULL_PER_DIRECTION", 5)

    # cluster 0: passes S1/S2/S4/S5 (all thresholds comfortably cleared).
    # cluster 1: passes S1/S2/S4 but fails S5 -> ALL_AXES_pass must be False.
    for s in range(1, 21):
        _write_seed_json(s1_dir / f"result_seed{s}_threads1.json", {0: 1.0, 1: 1.0})
    _write_seed_json(s1_dir / "result_seed42_threads1.json", {0: 1.0, 1: 1.0})
    _write_seed_json(s1_dir / "result_seed42_threads24.json", {0: 1.0, 1: 1.0})

    rows = [{"cluster": c, "boot": b, "jaccard": 0.9, "coassign": 0.9, "outcome": "stable"}
            for c in (0, 1) for b in range(10)]
    pd.DataFrame(rows).to_csv(s2_dir / "shard_0.csv", index=False)

    for name in ("hospital_10_to_20", "hospital_20_to_10"):
        _write_percluster_csv(s4s5_dir / f"observed_{name}_percluster.csv", {0: 0.9, 1: 0.9})
        for rep in range(5):
            _write_percluster_csv(s4s5_dir / f"null_{name}_seed{rep}_percluster.csv", {0: 0.1, 1: 0.1})
    for name in ("period_early_to_late", "period_late_to_early"):
        # cluster 1 fails S5 here (below absolute floor).
        _write_percluster_csv(s4s5_dir / f"observed_{name}_percluster.csv", {0: 0.9, 1: 0.2})
        for rep in range(5):
            _write_percluster_csv(s4s5_dir / f"null_{name}_seed{rep}_percluster.csv", {0: 0.1, 1: 0.1})

    mod.main()

    matrix = pd.read_csv(tmp_path / "survival_matrix.csv").set_index("cluster")
    assert len(matrix) == len(clusters), "main() must emit exactly one row per published cluster, no drops"
    assert bool(matrix.loc[0, "ALL_AXES_pass"]) is True
    assert bool(matrix.loc[1, "S5_pass"]) is False
    assert bool(matrix.loc[1, "ALL_AXES_pass"]) is False, (
        "ALL_AXES_pass must be the AND of all four axes -- cluster 1 fails S5 alone "
        "and must not pass overall"
    )
