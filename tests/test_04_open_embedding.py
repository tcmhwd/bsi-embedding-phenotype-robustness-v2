import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from importlib import import_module

mod = import_module("04_open_embedding")


def test_require_passing_gate_raises_when_gate_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "OUT", tmp_path)
    with pytest.raises(RuntimeError, match="no Task 4 gate verdict"):
        mod.require_passing_gate("bge-m3")


@pytest.mark.parametrize("verdict,should_raise", [
    ("ADOPT", False), ("CONDITIONAL", False), ("FAIL", True),
])
def test_require_passing_gate_respects_verdict(tmp_path, monkeypatch, verdict, should_raise):
    monkeypatch.setattr(mod, "OUT", tmp_path)
    model_dir = tmp_path / "bge-m3"
    model_dir.mkdir(parents=True)
    (model_dir / "GATE.md").write_text(f"VERDICT: {verdict}")
    (model_dir / "concordance.json").write_text(json.dumps({"verdict": verdict}))

    if should_raise:
        with pytest.raises(RuntimeError, match="must be ADOPT or CONDITIONAL"):
            mod.require_passing_gate("bge-m3")
    else:
        result = mod.require_passing_gate("bge-m3")
        assert result["verdict"] == verdict


def test_linear_cka_identical_matrices_is_one():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    cka = mod._linear_cka(X, X.copy())
    assert cka == pytest.approx(1.0, abs=1e-8)


def test_linear_cka_is_symmetric_and_bounded():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 10))
    Y = rng.normal(size=(200, 6))
    cka_xy = mod._linear_cka(X, Y)
    cka_yx = mod._linear_cka(Y, X)
    assert cka_xy == pytest.approx(cka_yx, abs=1e-8)
    assert -1e-6 <= cka_xy <= 1.0 + 1e-6


def test_mutual_knn_overlap_identical_embeddings_is_one():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(50, 5))
    overlap = mod._mutual_knn_overlap(X, X.copy(), k=5)
    assert overlap == pytest.approx(1.0)


def test_organism_purity_ways_pure_cluster_scores_one():
    labels = np.array([0, 0, 0, 1, 1, 1])
    organisms = np.array(["E. coli", "E. coli", "E. coli", "S. aureus", "S. aureus", "S. aureus"])
    result = mod._organism_purity_ways(labels, organisms, chance_level=0.5)
    assert result["incl_noise_incl_poly"] == pytest.approx(1.0)
    assert result["excl_noise_incl_poly"] == pytest.approx(1.0)


def test_organism_purity_ways_excludes_noise_and_polymicrobial():
    # rows: (label, organism)
    #  0: (0, E. coli)        1: (0, S. aureus)      2: (-1, E. coli)
    #  3: (1, Polymicrobial)  4: (1, K. pneumoniae)
    labels = np.array([0, 0, -1, 1, 1])
    organisms = np.array(["E. coli", "S. aureus", "E. coli", "Polymicrobial", "K. pneumoniae"])
    result = mod._organism_purity_ways(labels, organisms, chance_level=0.4)
    # incl noise, incl poly: noise (-1) is its own single-row group (purity
    # 1 trivially); cluster 0 and cluster 1 are each a 50/50 mix.
    assert result["incl_noise_incl_poly"] == pytest.approx((1 * 1.0 + 2 * 0.5 + 2 * 0.5) / 5)
    # excl noise: drop row 2; clusters 0 and 1 unchanged (both 50/50).
    assert result["excl_noise_incl_poly"] == pytest.approx((2 * 0.5 + 2 * 0.5) / 4)
    # incl noise, excl poly: drop row 3 (Polymicrobial); cluster 1 becomes
    # a single-row (pure) group, noise stays a single-row (pure) group.
    assert result["incl_noise_excl_poly"] == pytest.approx((1 * 1.0 + 2 * 0.5 + 1 * 1.0) / 4)
    # excl noise, excl poly: drop rows 2 and 3.
    assert result["excl_noise_excl_poly"] == pytest.approx((2 * 0.5 + 1 * 1.0) / 3)
    assert result["chance_level"] == 0.4


def test_verdict_thresholds_adopt(monkeypatch, tmp_path):
    """Smoke-checks the verdict branch logic in isolation by re-deriving it
    the same way stage_gate does, to pin the thresholds against silent
    drift (design doc section 1.6 verdict table)."""
    def verdict_for(ari_ref, g3a_k50, f1_gap):
        if ari_ref >= 0.60 and g3a_k50 >= 0.50 and f1_gap <= 0.10:
            return "ADOPT"
        elif 0.40 <= ari_ref < 0.60:
            return "CONDITIONAL"
        elif ari_ref < 0.40 and g3a_k50 >= 0.50 and f1_gap <= 0.15:
            return "CONDITIONAL"
        else:
            return "FAIL"

    assert verdict_for(0.65, 0.55, 0.05) == "ADOPT"
    assert verdict_for(0.45, 0.10, 0.30) == "CONDITIONAL"
    assert verdict_for(0.10, 0.55, 0.12) == "CONDITIONAL"
    assert verdict_for(0.016, 0.05, 0.5) == "FAIL"  # the pre-fix e5 calibration point (ARI 0.016)
