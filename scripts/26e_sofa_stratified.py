#!/usr/bin/env python3
"""Task 26 Part 3 -- SOFA-availability stratification.

Complete-case SOFA (`sofa_score`, no suffix) exists for 3,793/32,329
(11.7%) admissions; `sofa_score_2` imputes unmeasured components as
normal and is defined for all 32,329. Question: does case retrieval
discriminate 90-day mortality where SOFA CANNOT actually be computed?

Splits the full cohort (`23a_retrieval_common.build_master_frame`, the
SAME canonical N=32,329/admission_ID order the published_16 AUROC
(0.6206) and the Gower matrix are built on -- verified, not assumed, via
the master/gower row-order check below) into two row masks:
  - sofa_computable:      sofa_score.notna()   (n ~= 3,793)
  - sofa_not_computable:  sofa_score.isna()    (n ~= 28,536)

For each stratum, reuses UNCHANGED:
  - `22g_comparator_common.evaluate_prognostic_separation_grouped` for the
    published-16 AUROC (called on the stratum's own admission_ID/
    patient_ID/label arrays -- its own internal StratifiedGroupKFold(5,
    patient_ID, seed=42) then runs within that stratum).
  - `23a_retrieval_common.evaluate_continuous_predictor_grouped` for
    SOFA-alone (`sofa_score_2`, the only version defined in both strata)
    and Charlson-alone (`charlson_score`), same CV construction.
  - `23a_retrieval_common.{build_folds, topk_ordered, neighbour_risk_scores,
    normalize_rows, load_representation_vectors}` for retrieval AUROC at
    k=25, Gower and openai-3large, unrestricted and past_only -- same
    per-fold procedure as `23b_run_retrieval.py`, just built on the
    stratum's own admission_ID/patient_ID/test_date/y arrays (the
    candidate pool for a stratum's retrieval is therefore CLOSED to that
    stratum -- see the note in the written-out JSON/printed summary).
  - the precomputed full-cohort Gower matrix (`22g_comparator_common.
    load_gower_matrix`), fancy-indexed by this stratum's GLOBAL row
    positions in the canonical order (same `np.ix_` extraction pattern as
    `22g_comparator_common.patient_bootstrap_ari`) -- never recomputed.

Nothing in `22g_comparator_common.py` / `23a_retrieval_common.py` is
modified; the stratification is applied entirely as row masks on this
module's own copies of the master frame before calling those functions.

Writes out/26_controls/sofa_stratified.json. bsi_pheno interpreter,
cpu_partition (Gower fancy-indexing + a few embedding matmuls per fold --
no GPU, no frozen-env clustering needed here).
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c23 = importlib.import_module("23a_retrieval_common")
c22g = importlib.import_module("22g_comparator_common")
rr = importlib.import_module("23b_run_retrieval")  # build_valid_mask, compute_score_matrix

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
OUT_DIR = REV / "out/26_controls"
DEST = OUT_DIR / "sofa_stratified.json"

K = 25
VARIANTS = ["unrestricted", "past_only"]
REPRESENTATIONS = ["gower", "openai-3large"]


def retrieval_auroc_for_stratum(sub: pd.DataFrame, representation: str, variant: str,
                                 global_positions: np.ndarray, full_gower) -> dict:
    admission_ids = sub["admission_ID"].to_numpy()
    patient_ids = sub["patient_ID"].to_numpy()
    y = sub[c23.MORTALITY_COL].to_numpy()
    dates_ordinal = sub["test_date_ordinal"].to_numpy()

    folds = c23.build_folds(y, patient_ids)

    if representation == "openai-3large":
        vecs = c23.load_representation_vectors("openai-3large", admission_ids)
        source = c23.normalize_rows(vecs)
    else:
        source = None  # gower handled per-fold via full_gower + global_positions

    fold_aucs = []
    n_excluded_total = 0
    total_self, total_same_patient = 0, 0
    for train_idx, test_idx in folds:
        valid_mask = rr.build_valid_mask(variant, train_idx, test_idx, dates_ordinal)
        if representation == "gower":
            g_test = global_positions[test_idx]
            g_train = global_positions[train_idx]
            block = np.asarray(full_gower[np.ix_(g_test, g_train)], dtype=np.float32)
            score_matrix = -block
        else:
            score_matrix = source[test_idx] @ source[train_idx].T

        top_idx_sorted, valid_counts = c23.topk_ordered(score_matrix, valid_mask, K)

        idx_safe, use_mask = c23.neighbour_selection_mask(top_idx_sorted, valid_counts, K)
        admission_train, admission_test = admission_ids[train_idx], admission_ids[test_idx]
        patient_train, patient_test = patient_ids[train_idx], patient_ids[test_idx]
        neigh_adm = admission_train[idx_safe]
        neigh_pat = patient_train[idx_safe]
        total_self += int((use_mask & (neigh_adm == admission_test[:, None])).sum())
        total_same_patient += int((use_mask & (neigh_pat == patient_test[:, None])).sum())

        y_train, y_test = y[train_idx], y[test_idx]
        risk, counts = c23.neighbour_risk_scores(top_idx_sorted, valid_counts, y_train, K)
        valid_rows = ~np.isnan(risk)
        n_excluded_total += int((~valid_rows).sum())
        n_classes = len(set(y_test[valid_rows].tolist()))
        if valid_rows.sum() == 0 or n_classes < 2:
            continue
        fold_aucs.append(roc_auc_score(y_test[valid_rows], risk[valid_rows]))

    assert total_self == 0, f"{representation}/{variant}: self-retrieval found in stratum"
    assert total_same_patient == 0, f"{representation}/{variant}: same-patient neighbour found in stratum"

    return {
        "representation": representation, "variant": variant, "k": K,
        "auroc_mean": float(np.mean(fold_aucs)) if fold_aucs else None,
        "auroc_std": float(np.std(fold_aucs)) if fold_aucs else None,
        "n_folds_scored": len(fold_aucs),
        "n_folds_total": len(folds),
        "n_queries_excluded_insufficient_neighbours": n_excluded_total,
    }


def run_stratum(name: str, sub: pd.DataFrame, global_positions: np.ndarray, full_gower) -> dict:
    t0 = time.time()
    n = len(sub)
    n_events = int(sub[c23.MORTALITY_COL].sum())
    admission_ids = sub["admission_ID"].to_numpy()
    patient_ids = sub["patient_ID"].to_numpy()
    y = sub[c23.MORTALITY_COL].to_numpy()

    # Published-16 clusters (22g function, unchanged, called on this stratum's own arrays).
    pub_auc_mean, pub_auc_std = c22g.evaluate_prognostic_separation_grouped(
        sub["phenotype_cluster"].to_numpy(), admission_ids, patient_ids,
        dataset_path=c22g.DATASET, seed=c22g.SEED,
    )

    # SOFA-alone (sofa_score_2, the only version defined in both strata) and
    # Charlson-alone -- same imputation call (23c's own pattern), restricted
    # to this stratum's admission_ID.
    matrix, categorical_cols = c23.comparators.build_structured_matrix(sub[["admission_ID"]])
    imputed = c23.comparators.impute_structured_matrix(matrix, categorical_cols)
    anchors = {}
    for feat_name, col in [("sofa", "sofa_score_2"), ("charlson", "charlson_score")]:
        values = imputed[col].to_numpy()
        auc_mean, auc_std = c23.evaluate_continuous_predictor_grouped(values, admission_ids, patient_ids, y)
        anchors[feat_name] = {
            "source_column": col, "auroc_mean": auc_mean, "auroc_std": auc_std,
            "n_missing_before_impute": int(matrix[col].isna().sum()),
        }

    retrieval = {}
    for rep in REPRESENTATIONS:
        for variant in VARIANTS:
            r = retrieval_auroc_for_stratum(sub, rep, variant, global_positions, full_gower)
            retrieval[f"{rep}__{variant}"] = r
            print(f"[26e/{name}] {rep}/{variant}: AUROC={r['auroc_mean']}", flush=True)

    result = {
        "stratum": name, "n": n, "n_events_90d_mortality": n_events,
        "event_rate": n_events / n if n else None,
        "published_16": {"auroc_mean": pub_auc_mean, "auroc_std": pub_auc_std},
        "anchors": anchors,
        "retrieval_k25": retrieval,
        "wall_sec": time.time() - t0,
        "note": "CV (StratifiedGroupKFold(5, patient_ID, seed=42)) and, for retrieval, the "
                "candidate pool are built FRESH within this stratum only (closed cohort) -- "
                "reusing 22g/23a functions unchanged on stratum-masked input arrays, per task "
                "instruction. Retrieval neighbours therefore come only from other admissions "
                "in the SAME stratum, not from the whole 32,329-row cohort.",
    }
    print(f"[26e] stratum={name} n={n} events={n_events} published16={pub_auc_mean:.4f} "
          f"sofa={anchors['sofa']['auroc_mean']:.4f} charlson={anchors['charlson']['auroc_mean']:.4f} "
          f"({time.time() - t0:.1f}s)", flush=True)
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    master = c23.build_master_frame()

    sofa_sparse = pd.read_csv(c23.DATASET, usecols=["admission_ID", "sofa_score"], low_memory=False)
    master = master.merge(sofa_sparse, on="admission_ID", how="left", validate="one_to_one")
    assert len(master) == 32329

    gower_ids = pd.read_csv(c22g.GOWER_ADMISSION_IDS_PATH)["admission_ID"].to_numpy()
    assert np.array_equal(gower_ids, master["admission_ID"].to_numpy()), (
        "master frame row order does not match the stored Gower matrix's row order -- "
        "global_positions below would be wrong"
    )
    full_gower = c22g.load_gower_matrix()

    mask_computable = master["sofa_score"].notna().to_numpy()
    print(f"[26e] sofa_computable n={mask_computable.sum()} "
          f"({100 * mask_computable.mean():.2f}%), not_computable n={(~mask_computable).sum()}",
          flush=True)

    results = {}
    for name, mask in [("sofa_computable", mask_computable), ("sofa_not_computable", ~mask_computable)]:
        global_positions = np.where(mask)[0]
        sub = master.iloc[global_positions].reset_index(drop=True)
        results[name] = run_stratum(name, sub, global_positions, full_gower)
        DEST.write_text(json.dumps(results, indent=2))

    DEST.write_text(json.dumps(results, indent=2))
    print(f"[26e] wrote {DEST}", flush=True)


if __name__ == "__main__":
    main()
