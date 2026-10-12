"""Tests for `18f_findings.py` -- assembles the S6 (encoder) axis table and
verdict from per-encoder kNN/reclustering JSON plus the (reused, not
recomputed) `best_lookup_f1` column of Task 17's survival matrix.

`main()` holds all the aggregation logic inline (no separate pure
functions), so these build small synthetic fixture directories, monkeypatch
`18_common.OUT`/`18_common.SURVIVAL_MATRIX` onto them, run `main()` end to
end, and assert on the written `s6_matrix.csv`: the PRESPEC S6 rule (>=2 of
3 encoders with F1>=0.60 AND F1>=lookup+0.10) is actually enforced, a
cluster missing one encoder's result is reported as UNDETERMINED (None)
rather than silently mislabeled PASS/FAIL, and no cluster row is dropped.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from importlib import import_module

mod = import_module("18f_findings")
c18 = import_module("18_common")


def _write_recluster(path, jaccards: dict, n_clusters=16, noise_frac=0.05, n_rows=32329, n_noise=1792):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "n_clusters": n_clusters, "noise_frac": noise_frac, "n_rows": n_rows, "n_noise": n_noise,
        "agreement_vs_published": {
            "ari_all_row": 0.5, "ari_noise_excl": 0.5, "ami_all_row": 0.5, "ami_noise_excl": 0.5,
        },
        "best_match_jaccard_vs_published": {str(k): v for k, v in jaccards.items()},
    }))


def _write_knn(path, per_cluster_f1: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"per_cluster_f1": {str(k): v for k, v in per_cluster_f1.items()}}))


@pytest.fixture
def s6_fixture(tmp_path, monkeypatch):
    out = tmp_path / "18_encoder_dec"
    out.mkdir()
    monkeypatch.setattr(c18, "OUT", out)
    monkeypatch.setattr(mod, "c18", c18)

    survival = pd.DataFrame({
        "cluster": [0, 1, 2],
        "size": [471, 1780, 300],
        "best_lookup_f1": [0.03, 0.95, 0.50],
        "ALL_AXES_pass": [True, False, True],
    })
    survival_path = tmp_path / "survival_matrix.csv"
    survival.to_csv(survival_path, index=False)
    monkeypatch.setattr(c18, "SURVIVAL_MATRIX", survival_path)

    return out


def test_s6_pass_requires_two_of_three_encoders_over_threshold_and_margin(s6_fixture):
    out = s6_fixture
    # cluster 0: lookup=0.03 (near-invisible to tokens). Two encoders clear
    # F1>=0.60 AND F1>=lookup+0.10=0.13 -> S6_pass True.
    # cluster 1: lookup=0.95 (token-defined). Encoders score high in absolute
    # terms but never clear lookup+0.10=1.05 -- impossible -- so S6_pass False
    # despite every encoder individually exceeding 0.60.
    # cluster 2: lookup=0.50. Only ONE encoder clears both conditions ->
    # S6_pass False (fails the ">=2 of 3" count, not the per-encoder rule).
    _write_knn(out / "s6_knn" / "bge-m3.json", {0: 0.70, 1: 0.90, 2: 0.65})
    _write_knn(out / "s6_knn" / "qwen3-0.6b.json", {0: 0.65, 1: 0.85, 2: 0.40})
    _write_knn(out / "s6_knn" / "openai-3small.json", {0: 0.20, 1: 0.92, 2: 0.30})
    _write_knn(out / "s6_knn" / "openai-3large.json", {0: 0.95, 1: 0.98, 2: 0.90})
    for enc in mod.ENCODERS:
        _write_recluster(out / "recluster" / enc / "result.json", {0: 0.5, 1: 0.5, 2: 0.5})

    mod.main()

    df = pd.read_csv(out / "s6_matrix.csv").set_index("cluster")
    assert len(df) == 3, "no cluster row may be dropped"

    assert int(df.loc[0, "n_encoders_passing"]) == 2
    assert bool(df.loc[0, "S6_pass"]) is True

    assert int(df.loc[1, "n_encoders_passing"]) == 0, (
        "every encoder clears the absolute F1 floor for cluster 1, but none "
        "clears lookup+margin -- the margin-over-lookup conjunct must still veto"
    )
    assert bool(df.loc[1, "S6_pass"]) is False

    assert int(df.loc[2, "n_encoders_passing"]) == 1
    assert bool(df.loc[2, "S6_pass"]) is False, (
        "exactly 1 of 3 encoders passing must fail the >=2-of-3 rule, not be "
        "silently rounded up to a pass"
    )

    findings = (out / "FINDINGS.md").read_text()
    assert "1/3 measured published clusters PASS S6" in findings


def test_s6_pass_is_none_not_false_when_an_encoder_result_is_missing(s6_fixture):
    out = s6_fixture
    # Only 2 of 3 encoders have results for this run (e.g. the third gate
    # job hasn't finished) -- S6_pass must be reported as UNDETERMINED
    # (None), never silently treated as a fail (or a pass).
    _write_knn(out / "s6_knn" / "bge-m3.json", {0: 0.90, 1: 0.10, 2: 0.90})
    _write_knn(out / "s6_knn" / "qwen3-0.6b.json", {0: 0.90, 1: 0.10, 2: 0.90})
    # openai-3small.json intentionally not written.
    _write_knn(out / "s6_knn" / "openai-3large.json", {0: 0.95, 1: 0.95, 2: 0.95})
    for enc in mod.ENCODERS:
        _write_recluster(out / "recluster" / enc / "result.json", {0: 0.5, 1: 0.5, 2: 0.5})

    mod.main()

    df = pd.read_csv(out / "s6_matrix.csv").set_index("cluster")
    assert int(df.loc[0, "n_encoders_measured"]) == 2
    assert pd.isna(df.loc[0, "S6_pass"]), (
        "an incompletely-measured cluster must report S6_pass as missing/NaN, "
        "not be coerced into a determinate True/False"
    )


def test_s6_matrix_preserves_every_survival_matrix_cluster(s6_fixture):
    out = s6_fixture
    for enc_file in ("bge-m3", "qwen3-0.6b", "openai-3small", "openai-3large"):
        _write_knn(out / "s6_knn" / f"{enc_file}.json", {0: 0.8, 1: 0.8, 2: 0.8})
    for enc in mod.ENCODERS:
        _write_recluster(out / "recluster" / enc / "result.json", {0: 0.5, 1: 0.5, 2: 0.5})

    mod.main()

    df = pd.read_csv(out / "s6_matrix.csv")
    assert sorted(df["cluster"].tolist()) == [0, 1, 2]
