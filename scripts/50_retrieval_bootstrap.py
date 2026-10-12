#!/usr/bin/env python3
"""Task 50 -- patient-level bootstrap confidence intervals for the retrieval
AUROC differences (RT-004 / RT-011).

WHY THIS EXISTS
---------------
The constructive result of the manuscript is that nearest-neighbour retrieval
over structured variables discriminates 90-day mortality better than retrieval
over text embeddings. Until now its only inferential statement was an exact
one-sided sign test over the five cross-validation folds. Internal review of
the Phase-2B draft rejected that as the sole support, for two reasons:

  * five folds are complementary partitions of ONE dataset, not five
    independent studies, so a fold-level test overstates its own evidence; and
  * an exact sign test at n = 5 cannot produce a one-sided value below
    1/32 = 0.03125 whatever the size of the effect, so "p = 0.031" carries no
    information beyond "all five folds agreed" -- and the same contrast was
    reported against three embedding representations without any adjustment.

This script replaces that with a **patient-level bootstrap** over the
out-of-fold predictions, which respects the actual unit of independence
(3,773 patients contribute more than one admission, up to 23; N-002).

METHOD
------
`23b_run_retrieval.py` now writes, for each (representation, pool variant), the
out-of-fold risk score of every admission (`predictions_*.npz`, added
2026-09-08). Every representation is scored on the IDENTICAL folds, so the
predictions are aligned row-for-row and the difference is paired per admission.

For B resamples:
  1. draw `n_patients` patients WITH replacement (`patient_ID` is the sampling
     unit, so an admission enters as many times as its patient is drawn);
  2. assemble the admission rows of the drawn patients;
  3. compute AUROC for each representation on that resample;
  4. record the difference.

Reported: the observed difference, the 2.5th/97.5th percentiles of the
bootstrap difference distribution, and the share of resamples in which the
difference is <= 0 (a bootstrap one-sided level that, unlike the sign test, is
not floored by the number of folds). Resamples in which the outcome is constant
are skipped and counted, not silently dropped.

WHAT THIS DOES NOT FIX
----------------------
The bootstrap resamples the evaluation, not the model fitting: folds and
neighbour pools are fixed at their observed values. It therefore quantifies
sampling uncertainty in the comparison, not the full uncertainty of the whole
pipeline, and it cannot repair the fact that the two representations do not
carry identical feature sets. Both limits belong in the Results sentence that
cites this row.

Deterministic: seed 2026, matching the size-stability analysis (SS-001).
Runs in `bsi_pheno` (numpy/sklearn only; no clustering, no frozen env needed).
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c23 = importlib.import_module("23a_retrieval_common")

OUT = c23.OUT
SEED = 2026
N_BOOT = 10_000
K_REPORT = 25
BASELINE = "gower"
COMPARATORS = ["openai-3large", "openai-3small", "bge-m3"]
VARIANTS = ["unrestricted", "past_only"]


def load(rep: str, variant: str):
    p = OUT / f"predictions_{rep}_{variant}.npz"
    if not p.exists():
        raise SystemExit(f"{p} missing -- rerun 23b_run_retrieval.py "
                         f"(--representation {rep} --variant {variant}) after the "
                         "2026-09-08 patch that writes per-query predictions")
    return np.load(p, allow_pickle=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=K_REPORT)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    key = f"risk_k{args.k}"

    result = {
        "k": args.k, "n_boot": args.n_boot, "seed": SEED,
        "baseline": BASELINE, "comparators": COMPARATORS,
        "sampling_unit": "patient_ID (with replacement); an admission enters as often as its patient is drawn",
        "method": ("patient-level bootstrap over out-of-fold risk scores; folds are identical "
                   "across representations so each resample is a paired comparison"),
        "contrasts": {},
    }

    for variant in VARIANTS:
        base = load(BASELINE, variant)
        admissions = base["admission_ID"]
        patients = base["patient_ID"]
        y = base["y"].astype(int)
        base_risk = base[key]

        uniq, inv = np.unique(patients, return_inverse=True)
        n_pat = len(uniq)
        # Flat row index grouped by patient, plus per-patient offsets/counts.
        # A resample is then assembled by pure integer arithmetic instead of a
        # 27,083-element Python list comprehension per draw, and the draws are
        # generated one at a time rather than pre-allocated as a
        # (n_boot x n_patients) matrix -- the first version did the latter and
        # was killed for memory (2.2 GB for the draw matrix alone).
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

        comps = {}
        for comp in COMPARATORS:
            other = load(comp, variant)
            if not np.array_equal(other["admission_ID"], admissions):
                raise SystemExit(f"{comp}/{variant}: admission order differs from {BASELINE}")
            comps[comp] = other[key]

        usable = ~np.isnan(base_risk)
        for r in comps.values():
            usable &= ~np.isnan(r)
        obs = {c: (roc_auc_score(y[usable], base_risk[usable])
                   - roc_auc_score(y[usable], comps[c][usable])) for c in COMPARATORS}

        # One resample serves every comparator: the draws are shared, so the
        # three contrasts are computed on identical patient sets and remain
        # mutually comparable.
        rng = np.random.default_rng(SEED)
        diffs = {c: [] for c in COMPARATORS}
        skipped = 0
        for _ in range(args.n_boot):
            rows = resample_rows(rng.integers(0, n_pat, size=n_pat))
            rows = rows[usable[rows]]
            yy = y[rows]
            if yy.min() == yy.max():
                skipped += 1
                continue
            a_base = roc_auc_score(yy, base_risk[rows])
            for c in COMPARATORS:
                diffs[c].append(a_base - roc_auc_score(yy, comps[c][rows]))

        for comp in COMPARATORS:
            d = np.asarray(diffs[comp])
            result["contrasts"][f"{variant}|{BASELINE}-{comp}"] = {
                "n_usable_admissions": int(usable.sum()),
                "n_patients": int(n_pat),
                "observed_delta_auroc": float(obs[comp]),
                "boot_mean": float(d.mean()),
                "ci95_low": float(np.percentile(d, 2.5)),
                "ci95_high": float(np.percentile(d, 97.5)),
                "share_delta_le_0": float((d <= 0).mean()),
                "n_resamples_used": int(len(d)),
                "n_resamples_skipped_constant_outcome": int(skipped),
            }
            c = result["contrasts"][f"{variant}|{BASELINE}-{comp}"]
            print(f"[50] {variant:12s} {BASELINE} - {comp:14s} "
                  f"delta {c['observed_delta_auroc']:+.4f} "
                  f"95% CI [{c['ci95_low']:+.4f}, {c['ci95_high']:+.4f}] "
                  f"P(delta<=0) {c['share_delta_le_0']:.4f}", flush=True)

    dest = OUT / "retrieval_bootstrap.json"
    dest.write_text(json.dumps(result, indent=1))
    print(f"[50] wrote {dest}", flush=True)


if __name__ == "__main__":
    main()
