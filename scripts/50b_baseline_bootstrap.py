#!/usr/bin/env python3
"""Task 50b -- patient-level bootstrap intervals for the retrieval comparison
against the NON-retrieval baselines (cluster labels, SOFA, Charlson).

WHY THIS EXISTS
---------------
`50_retrieval_bootstrap.py` closed the embedding-vs-structured contrast
(Gower minus each embedding retrieval arm) with a patient-level bootstrap.
It did not touch the other half of the constructive claim: that structured
retrieval also discriminates 90-day mortality better than the phenotype
labels themselves (published_16, AUROC 0.621) and better than SOFA (0.727).
Both were point estimates with a per-fold SD and no interval, so the
Discussion sentence "far better than the cluster labels" rested on a
difference of two point estimates. The third audit round flagged exactly
that (MN-5).

This script supplies the missing intervals, on the SAME resamples and with
the SAME machinery as task 50, so the two halves of the claim are stated on
one footing.

METHOD
------
Each baseline is a logistic regression on a fixed feature set, fitted inside
the identical `StratifiedGroupKFold(5, groups=patient_ID, seed=42)` folds
that every AUROC in this project uses. We refit them here only to RETAIN the
out-of-fold predicted probabilities, which the original point-estimate
scripts computed and discarded; before any bootstrap runs, the recomputed
mean/SD AUROC is asserted against the stored `point_*.json` value, so a
fold-construction or population drift fails loudly instead of producing a
plausible interval for the wrong quantity.

Baselines:
  published_16 : one-hot of the 16 originally derived cluster labels
                 (`22g_comparator_common.evaluate_prognostic_separation_grouped`
                 feature construction, reproduced to keep the predictions)
  sofa         : `sofa_score_2` alone, median-imputed as in point_sofa.json
  charlson     : `charlson_score` alone, likewise

The comparator is Gower k = 25 retrieval, read from the stored
`predictions_gower_{variant}.npz` written by `23b_run_retrieval.py`, so the
Gower side of this contrast is byte-identical to the one task 50 used.

Bootstrap: 10,000 patient-level resamples, seed 2026 -- the same seed and the
same resampler as task 50, so a reader can compare intervals across the two
files without wondering whether the draws differed.

WHAT THIS DOES NOT FIX
----------------------
As in task 50, the bootstrap resamples the evaluation and not the fitting:
folds, imputation and neighbour pools stay at their observed values. And the
baselines are not matched in information content -- the cluster labels are a
16-level categorical, SOFA one integer, Gower a 44-variable distance -- so
the interval quantifies the comparison actually made, not a like-for-like one.
Both limits belong in the sentence that cites this row.

Runs in `bsi_pheno` (numpy/sklearn/pandas only). Minutes, single core.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c23 = importlib.import_module("23a_retrieval_common")
comparators = c23.comparators

OUT = c23.OUT
SEED = 2026
N_BOOT = 10_000
K_REPORT = 25
VARIANTS = ["unrestricted", "past_only"]
TOL = 1e-9


def oof_predictions(X: np.ndarray, y: np.ndarray, patient_ids: np.ndarray):
    """Refit the baseline in the project's standard folds and keep the
    out-of-fold probabilities as well as the per-fold AUROCs."""
    folds = c23.build_folds(y, patient_ids, seed=c23.SEED)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    oof = np.full(len(y), np.nan)
    aucs = []
    for train_idx, test_idx in folds:
        assert not (set(patient_ids[train_idx].tolist()) & set(patient_ids[test_idx].tolist())), \
            "StratifiedGroupKFold leaked a patient_ID across train/test"
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        oof[test_idx] = proba
        aucs.append(roc_auc_score(y[test_idx], proba))
    assert not np.isnan(oof).any(), "some admission never appeared in a test fold"
    return oof, float(np.mean(aucs)), float(np.std(aucs))


def check_against_stored(name: str, mean: float, std: float, stored_path: Path) -> dict:
    stored = json.loads(stored_path.read_text())
    sm = stored["auroc_90d_mortality_mean"]
    ss = stored["auroc_90d_mortality_std"]
    ok = abs(mean - sm) < TOL and abs(std - ss) < TOL
    assert ok, (f"{name}: recomputed AUROC {mean} (sd {std}) != stored {sm} (sd {ss}) -- "
                "fold construction or population drifted; do not trust the interval below")
    return {"recomputed_mean": mean, "recomputed_std": std,
            "stored_mean": sm, "stored_std": ss, "matches_stored": True,
            "stored_path": str(stored_path)}


def main() -> None:
    master = c23.build_master_frame()
    sanity = c23.sanity_check_published_16_reproduces_0_6206(master)
    print(f"[50b] published_16 sanity: {sanity['matches_stored_0_6206']}", flush=True)

    admission_ids = master["admission_ID"].to_numpy()
    patient_ids = master["patient_ID"].to_numpy()
    y = master[c23.MORTALITY_COL].to_numpy().astype(int)

    matrix, categorical_cols = comparators.build_structured_matrix_with_missingness_indicators(
        master[["admission_ID"]]
    )
    imputed = comparators.impute_structured_matrix(matrix, categorical_cols)

    baselines = {
        "published_16": pd.get_dummies(
            pd.Series(master["phenotype_cluster"].to_numpy()).astype(str)
        ).to_numpy(dtype=float),
        "sofa": imputed["sofa_score_2"].to_numpy(),
        "charlson": imputed["charlson_score"].to_numpy(),
    }
    stored_paths = {
        "published_16": c23.PUBLISHED_16_POINT_PATH,
        "sofa": OUT / "point_sofa.json",
        "charlson": OUT / "point_charlson.json",
    }

    base_pred, checks = {}, {}
    for name, X in baselines.items():
        oof, m, s = oof_predictions(X, y, patient_ids)
        checks[name] = check_against_stored(name, m, s, stored_paths[name])
        base_pred[name] = oof
        print(f"[50b] {name:13s} AUROC {m:.4f} +- {s:.4f} (matches stored)", flush=True)

    # Patient-grouped row index, identical construction to task 50.
    uniq, inv = np.unique(patient_ids, return_inverse=True)
    n_pat = len(uniq)
    order = np.argsort(inv, kind="stable").astype(np.int32)
    counts = np.bincount(inv, minlength=n_pat).astype(np.int32)
    starts = np.zeros(n_pat, dtype=np.int32)
    np.cumsum(counts[:-1], out=starts[1:])

    def resample_rows(pat_idx: np.ndarray) -> np.ndarray:
        sel = counts[pat_idx]
        total = int(sel.sum())
        ends = np.cumsum(sel)
        begins = ends - sel
        pos = np.arange(total, dtype=np.int64) - np.repeat(begins, sel) \
            + np.repeat(starts[pat_idx], sel)
        return order[pos]

    result = {
        "k": K_REPORT, "n_boot": N_BOOT, "seed": SEED,
        "comparator": "gower retrieval k=25",
        "baselines": list(baselines),
        "sampling_unit": "patient_ID (with replacement); an admission enters as often as its patient is drawn",
        "method": ("out-of-fold predictions refit in the project-standard "
                   "StratifiedGroupKFold(5, groups=patient_ID, seed=42) folds and checked "
                   "against the stored point estimates; patient-level bootstrap of the "
                   "paired AUROC difference, same resampler and seed as task 50"),
        "point_estimate_checks": checks,
        "published_16_sanity": sanity,
        "contrasts": {},
    }

    key = f"risk_k{K_REPORT}"
    for variant in VARIANTS:
        g = np.load(OUT / f"predictions_gower_{variant}.npz", allow_pickle=True)
        assert np.array_equal(g["admission_ID"], admission_ids), \
            f"gower/{variant}: admission order differs from the master frame"
        assert np.array_equal(g["y"].astype(int), y), f"gower/{variant}: outcome vector differs"
        gower = g[key]

        usable = ~np.isnan(gower)
        for v in base_pred.values():
            usable &= ~np.isnan(v)
        obs = {n: (roc_auc_score(y[usable], gower[usable])
                   - roc_auc_score(y[usable], base_pred[n][usable])) for n in baselines}

        rng = np.random.default_rng(SEED)
        diffs = {n: [] for n in baselines}
        skipped = 0
        for _ in range(N_BOOT):
            rows = resample_rows(rng.integers(0, n_pat, size=n_pat))
            rows = rows[usable[rows]]
            yy = y[rows]
            if yy.min() == yy.max():
                skipped += 1
                continue
            a_g = roc_auc_score(yy, gower[rows])
            for n in baselines:
                diffs[n].append(a_g - roc_auc_score(yy, base_pred[n][rows]))

        for n in baselines:
            d = np.asarray(diffs[n])
            result["contrasts"][f"{variant}|gower-{n}"] = {
                "n_usable_admissions": int(usable.sum()),
                "n_patients": int(n_pat),
                "observed_delta_auroc": float(obs[n]),
                "boot_mean": float(d.mean()),
                "ci95_low": float(np.percentile(d, 2.5)),
                "ci95_high": float(np.percentile(d, 97.5)),
                "share_delta_le_0": float((d <= 0).mean()),
                "n_resamples_used": int(len(d)),
                "n_resamples_skipped_constant_outcome": int(skipped),
            }
            c = result["contrasts"][f"{variant}|gower-{n}"]
            print(f"[50b] {variant:12s} gower - {n:13s} "
                  f"delta {c['observed_delta_auroc']:+.4f} "
                  f"95% CI [{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}] "
                  f"P(delta<=0) {c['share_delta_le_0']:.4f}", flush=True)

    result["env"] = c23.env_manifest()
    dest = OUT / "baseline_bootstrap.json"
    dest.write_text(json.dumps(result, indent=1))
    print(f"[50b] wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
