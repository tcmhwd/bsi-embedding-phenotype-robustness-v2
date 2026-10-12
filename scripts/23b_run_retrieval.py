#!/usr/bin/env python3
"""Task 23b -- per-representation, per-variant retrieval AUROC. Computes ALL
FOUR k in {5, 10, 25, 50} in one invocation, since the expensive part (the
per-fold cross-similarity/distance block) is representation-and-fold
specific, not k-specific: computed ONCE per fold via `c23.topk_ordered` and
sliced for every k (see that function's docstring).

--representation in {openai-3large, openai-3small, bge-m3, gower}
--variant       in {unrestricted, past_only}

**Sizing (stated explicitly, per task instructions -- never the full
32,329 x 32,329 matrix for any representation):** each
StratifiedGroupKFold(5) fold has ~25,863 train / ~6,466 test rows. For an
embedding representation, one fold's cross-similarity block is a single
BLAS matmul, (n_test x dim) @ (dim x n_train) -> (n_test, n_train) float32
(~668MB for n_test=6466, n_train=25863); for openai-3large (dim=3072) that
is ~1.03e12 FLOPs per fold, ~5.1e12 FLOPs for all 5 folds of one
(representation, variant) job -- seconds to low minutes on the wrapper's
CPU allocation via threaded BLAS, not the ~1e14-1e15 FLOPs a full 32,329^2
similarity would need. For `gower`, the per-fold block is instead a
`np.ix_(test_idx, train_idx)` fancy-index READ off the precomputed 4.18GB
mmap matrix (same extraction pattern `22g_comparator_common.patient_bootstrap_ari`
already uses for gower_pam bootstrap replicates) -- never recomputed, and
never materialized beyond one fold's (n_test, n_train) block at a time.

Outcome-blind scope: see `23a_retrieval_common.py` module docstring
("Where outcome data enters").
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c23 = importlib.import_module("23a_retrieval_common")


def build_valid_mask(variant: str, train_idx: np.ndarray, test_idx: np.ndarray, dates_ordinal: np.ndarray) -> np.ndarray:
    n_test, n_train = len(test_idx), len(train_idx)
    if variant == "unrestricted":
        return np.ones((n_test, n_train), dtype=bool)
    if variant == "past_only":
        train_dates = dates_ordinal[train_idx]
        test_dates = dates_ordinal[test_idx]
        # strictly earlier only -- same-day admissions are NOT "past" (a
        # deployed tool cannot yet know a same-day case's own outcome
        # either), so ties are excluded, not treated as available.
        return train_dates[None, :] < test_dates[:, None]
    raise ValueError(f"unknown variant {variant!r}")


def compute_score_matrix(representation: str, source, train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    """Returns (n_test, n_train) float array, HIGHER = MORE similar."""
    if representation == "gower":
        sub = np.asarray(source[np.ix_(test_idx, train_idx)], dtype=np.float32)
        return -sub  # Gower distance -> similarity by negation
    Xn = source  # already L2-row-normalized, full master-order array
    return Xn[test_idx] @ Xn[train_idx].T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--representation", required=True, choices=c23.REPRESENTATIONS)
    ap.add_argument("--variant", required=True, choices=c23.VARIANTS)
    args = ap.parse_args()

    c23.OUT.mkdir(parents=True, exist_ok=True)
    out_path = c23.OUT / f"point_{args.representation}_{args.variant}.json"

    master = c23.build_master_frame()
    print(f"[23b/{args.representation}/{args.variant}] n_rows={len(master)}", flush=True)

    sanity = c23.sanity_check_published_16_reproduces_0_6206(master)
    print(f"[23b/{args.representation}/{args.variant}] sanity check (published_16 vs stored "
          f"0.6205749882834398): {json.dumps(sanity)}", flush=True)

    admission_ids = master["admission_ID"].to_numpy()
    patient_ids = master["patient_ID"].to_numpy()
    y = master[c23.MORTALITY_COL].to_numpy()
    dates_ordinal = master["test_date_ordinal"].to_numpy()

    folds = c23.build_folds(y, patient_ids)
    print(f"[23b/{args.representation}/{args.variant}] {len(folds)} folds built, sizes: "
          f"{[(len(tr), len(te)) for tr, te in folds]}", flush=True)

    if args.representation == "gower":
        source = c23.load_gower_matrix_checked(admission_ids)
        print(f"[23b/{args.representation}/{args.variant}] Gower matrix row order verified "
              f"against admission_ids, shape={source.shape}", flush=True)
    else:
        vecs = c23.load_representation_vectors(args.representation, admission_ids)
        source = c23.normalize_rows(vecs)
        print(f"[23b/{args.representation}/{args.variant}] embedding aligned+normalized: "
              f"shape={vecs.shape}", flush=True)

    per_k_fold_aucs = {k: [] for k in c23.K_VALUES}
    per_k_n_excluded = {k: 0 for k in c23.K_VALUES}
    # Per-QUERY risk scores, added 2026-09-08. Fold-level AUROCs cannot support
    # a paired interval (five folds; an exact sign test bottoms out at 0.031),
    # so the out-of-fold prediction for every admission is retained and a
    # patient-level bootstrap is computed downstream by 50_retrieval_bootstrap.py.
    # Stored as parallel arrays in the master row order, NaN where a query had
    # too few eligible neighbours.
    per_k_risk = {k: np.full(len(admission_ids), np.nan, dtype=np.float64)
                  for k in c23.K_VALUES}
    total_same_patient = 0
    total_self_retrieval = 0

    for fi, (train_idx, test_idx) in enumerate(folds):
        t0 = time.time()
        assert len(set(train_idx.tolist()) & set(test_idx.tolist())) == 0, "train/test overlap"

        valid_mask = build_valid_mask(args.variant, train_idx, test_idx, dates_ordinal)
        score_matrix = compute_score_matrix(args.representation, source, train_idx, test_idx)

        y_train = y[train_idx]
        y_test = y[test_idx]
        patient_train = patient_ids[train_idx]
        patient_test = patient_ids[test_idx]
        admission_train = admission_ids[train_idx]
        admission_test = admission_ids[test_idx]

        top_idx_sorted, valid_counts = c23.topk_ordered(score_matrix, valid_mask, c23.K_MAX)

        # --- explicit runtime verification: never same-patient, never self ---
        idx_safe_max, use_mask_max = c23.neighbour_selection_mask(top_idx_sorted, valid_counts, c23.K_MAX)
        neighbour_admission = admission_train[idx_safe_max]
        neighbour_patient = patient_train[idx_safe_max]
        self_hits = use_mask_max & (neighbour_admission == admission_test[:, None])
        same_patient_hits = use_mask_max & (neighbour_patient == patient_test[:, None])
        total_self_retrieval += int(self_hits.sum())
        total_same_patient += int(same_patient_hits.sum())

        for k in c23.K_VALUES:
            risk, counts = c23.neighbour_risk_scores(top_idx_sorted, valid_counts, y_train, k)
            valid_rows = ~np.isnan(risk)
            n_excl = int((~valid_rows).sum())
            per_k_n_excluded[k] += n_excl
            n_classes = len(set(y_test[valid_rows].tolist()))
            if valid_rows.sum() == 0 or n_classes < 2:
                print(f"[23b/{args.representation}/{args.variant}] fold {fi + 1} k={k}: "
                      f"AUROC undefined (n_valid={int(valid_rows.sum())}, n_classes={n_classes}) "
                      "-- fold excluded from this k's mean/std", flush=True)
                continue
            auc = roc_auc_score(y_test[valid_rows], risk[valid_rows])
            per_k_fold_aucs[k].append(auc)
            per_k_risk[k][test_idx] = risk

        print(f"[23b/{args.representation}/{args.variant}] fold {fi + 1}/{len(folds)} done in "
              f"{time.time() - t0:.1f}s (train={len(train_idx)}, test={len(test_idx)})", flush=True)

    assert total_self_retrieval == 0, "index admission retrieved as its own neighbour -- pool construction bug"
    assert total_same_patient == 0, "same-patient neighbour used -- pool construction bug"
    print(f"[23b/{args.representation}/{args.variant}] VERIFIED across all {len(folds)} folds, up to "
          f"k={c23.K_MAX}: 0 self-retrievals, 0 same-patient neighbours "
          f"(checked {int(use_mask_max.sum())} total retrieved-neighbour slots in the last fold alone; "
          "identical checks ran and passed for every fold)", flush=True)

    result = {
        "representation": args.representation,
        "variant": args.variant,
        "k_values": c23.K_VALUES,
        "n_rows": int(len(master)),
        "n_folds": len(folds),
        "cv_scheme": f"StratifiedGroupKFold({c23.CV_SPLITS}, groups=patient_ID), seed={c23.SEED}, shuffle=True -- "
                     "same construction as 22g_comparator_common.evaluate_prognostic_separation_grouped, "
                     "verified to reproduce the stored published_16 AUROC exactly (see sanity_check_published_16 below)",
        "sanity_check_published_16": sanity,
        "n_same_patient_neighbours_found": total_same_patient,
        "n_self_retrievals_found": total_self_retrieval,
        "neighbour_pool": ("train fold only (never the index admission, never another admission of the "
                            "same patient -- both structurally guaranteed by the disjoint group-fold split "
                            "and verified above)") if args.variant == "unrestricted" else
                          ("train fold AND test_date strictly earlier than the query's own test_date "
                           "(same-day admissions excluded, not treated as available)"),
        "per_k": {
            str(k): {
                "auroc_mean": (float(np.mean(per_k_fold_aucs[k])) if per_k_fold_aucs[k] else None),
                "auroc_std": (float(np.std(per_k_fold_aucs[k])) if per_k_fold_aucs[k] else None),
                "n_folds_scored": len(per_k_fold_aucs[k]),
                # Per-fold AUROCs, added 2026-09-08. Every representation is
                # scored on the IDENTICAL folds (build_folds, verified against
                # the stored published_16 AUROC at the top of this run), so
                # differences between representations are PAIRED by fold and a
                # paired interval can be computed downstream without refitting.
                # Previously only the mean and SD were written, which made the
                # constructive comparison (RT-004) unreportable as a paired
                # contrast -- the one inferential statement journal reviewers
                # ask for. Fold order is the order build_folds yields.
                "auroc_per_fold": [float(a) for a in per_k_fold_aucs[k]],
                "n_queries_excluded_insufficient_neighbours": per_k_n_excluded[k],
            }
            for k in c23.K_VALUES
        },
        "outcome_note": f"{c23.MORTALITY_COL} read only as (a) the label being predicted and (b) the "
                        "values averaged across a query's neighbours to form that query's own risk score "
                        "-- never an admission's own outcome scoring itself (neighbour pool is always the "
                        "disjoint training fold).",
        "env": c23.env_manifest(),
    }
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    # Out-of-fold per-query risk scores alongside the JSON (2026-09-08). Kept in
    # a separate .npz because they are 32,329 x |K| floats, far too large for the
    # result JSON, and because only the bootstrap consumes them. admission_ID and
    # patient_ID travel with the scores so the downstream resampler never has to
    # re-derive the row order.
    pred_path = c23.OUT / f"predictions_{args.representation}_{args.variant}.npz"
    np.savez_compressed(
        pred_path,
        admission_ID=admission_ids,
        patient_ID=patient_ids,
        y=y,
        **{f"risk_k{k}": per_k_risk[k] for k in c23.K_VALUES},
    )
    print(json.dumps({k: v for k, v in result.items() if k != "env"}, indent=2, ensure_ascii=False), flush=True)
    print(f"[23b/{args.representation}/{args.variant}] wrote {out_path}", flush=True)
    print(f"[23b/{args.representation}/{args.variant}] wrote {pred_path}", flush=True)


if __name__ == "__main__":
    main()
