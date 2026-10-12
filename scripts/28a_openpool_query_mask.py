#!/usr/bin/env python3
"""Task 26 Part 3, OPEN-POOL variant -- SOFA-availability stratification with
QUERY-ONLY masking (deployability design).

`26e_sofa_stratified.py` (out/26_controls/sofa_stratified.json) split the
cohort by whether a complete-case SOFA is computable (n=3,793) or not
(n=28,536) and, for retrieval, rebuilt BOTH the CV folds AND the candidate
pool from scratch within each stratum (a CLOSED cohort: neighbours for a
sofa_not_computable query could only ever be other sofa_not_computable
admissions). A deployed retrieval system searches ALL past cases regardless
of which stratum the query falls in, so this script re-runs the SAME
retrieval definition with the candidate pool held OPEN to the full 32,329
admissions -- only the QUERIES being scored are restricted to a stratum.

Design (the only thing that differs from `26e_sofa_stratified.py`):
  - Folds are built ONCE via `23a_retrieval_common.build_folds` on the
    FULL master frame (admission_ID/patient_ID/90d_mortality for all
    32,329 rows) -- BYTE-IDENTICAL to the folds `23b_run_retrieval.py` and
    `22g_comparator_common.evaluate_prognostic_separation_grouped` use for
    the published_16 AUROC (0.6206), reproduced and asserted below via
    `sanity_check_published_16_reproduces_0_6206` before any retrieval
    work, exactly as every other Task-23 point-estimate script does.
  - For a given fold (train_idx, test_idx), test_idx is NARROWED to
    `test_idx[stratum_mask[test_idx]]` -- so only queries belonging to the
    target stratum are scored -- while train_idx (the candidate pool) is
    left UNCHANGED: the full training fold across the ENTIRE cohort, not
    restricted to the stratum. A sofa_not_computable query can therefore
    retrieve a sofa_computable neighbour and vice versa.
  - Every other function is reused UNCHANGED from `23a_retrieval_common.py`
    / `23b_run_retrieval.py`: `topk_ordered`, `neighbour_selection_mask`,
    `neighbour_risk_scores`, `build_valid_mask` (variant logic),
    `load_gower_matrix_checked`/`load_representation_vectors`,
    `normalize_rows`. k=25 only (task instruction). Same exclusions as
    every other Task-23/26 retrieval run: never the query itself, never
    another admission of the same patient -- both structurally guaranteed
    here too, since train_idx and (stratum-narrowed) test_idx remain
    disjoint subsets of the same group-fold split, and explicitly verified
    at runtime below (assert, not just documented).

Anchors (SOFA-alone, Charlson-alone, published-16) are NOT recomputed here
-- per task instruction, only Gower and openai-3large / unrestricted and
past_only are re-run under the open-pool design. The report layer reuses
the already-computed closed-cohort anchor values from
`out/26_controls/sofa_stratified.json` as the fixed reference points for
both designs.

Writes out/28_openpool/sofa_stratified_openpool.json. bsi_pheno
interpreter, cpu_partition (Gower fancy-indexing off the full-cohort
matrix + embedding matmuls per fold -- no GPU needed, same shape of work
as 26e, just against the full-size train folds instead of stratum-sized
ones, so somewhat more expensive for the openai-3large matmuls).
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c23 = importlib.import_module("23a_retrieval_common")
c22g = importlib.import_module("22g_comparator_common")
rr = importlib.import_module("23b_run_retrieval")  # build_valid_mask

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
OUT_DIR = REV / "out/28_openpool"
DEST = OUT_DIR / "sofa_stratified_openpool.json"
CLOSED_COHORT_PATH = REV / "out/26_controls/sofa_stratified.json"

K = 25
VARIANTS = ["unrestricted", "past_only"]
REPRESENTATIONS = ["gower", "openai-3large"]


def env_manifest() -> dict:
    return {
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
    }


def retrieval_auroc_open_pool(master: pd.DataFrame, stratum_mask: np.ndarray, representation: str,
                               variant: str, folds, full_gower=None, source=None) -> dict:
    """Candidate pool (train_idx) = FULL fold, unrestricted to the stratum.
    Queries (test_idx) narrowed to stratum_mask BEFORE any retrieval work,
    so the expensive score-matrix block is only ever (n_test_in_stratum,
    n_train_full) -- never the full (n_test_fold, n_train_full)."""
    admission_ids = master["admission_ID"].to_numpy()
    patient_ids = master["patient_ID"].to_numpy()
    y = master[c23.MORTALITY_COL].to_numpy()
    dates_ordinal = master["test_date_ordinal"].to_numpy()

    fold_aucs = []
    n_excluded_total = 0
    total_self, total_same_patient = 0, 0
    n_folds_scored = 0
    n_test_per_fold = []

    for train_idx, test_idx_full in folds:
        test_idx = test_idx_full[stratum_mask[test_idx_full]]
        n_test_per_fold.append(int(len(test_idx)))
        if len(test_idx) == 0:
            continue

        valid_mask = rr.build_valid_mask(variant, train_idx, test_idx, dates_ordinal)
        if representation == "gower":
            block = np.asarray(full_gower[np.ix_(test_idx, train_idx)], dtype=np.float32)
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
        n_folds_scored += 1

    assert total_self == 0, f"{representation}/{variant}: self-retrieval found (open-pool)"
    assert total_same_patient == 0, f"{representation}/{variant}: same-patient neighbour found (open-pool)"

    return {
        "representation": representation, "variant": variant, "k": K,
        "candidate_pool": "OPEN -- full training fold across all 32,329 admissions, "
                          "not restricted to the query's stratum",
        "auroc_mean": float(np.mean(fold_aucs)) if fold_aucs else None,
        "auroc_std": float(np.std(fold_aucs)) if fold_aucs else None,
        "n_folds_scored": n_folds_scored,
        "n_folds_total": len(folds),
        "n_test_queries_per_fold": n_test_per_fold,
        "n_queries_excluded_insufficient_neighbours": n_excluded_total,
    }


def run_stratum(name: str, master: pd.DataFrame, stratum_mask: np.ndarray, folds,
                 full_gower, source_openai: np.ndarray) -> dict:
    t0 = time.time()
    n = int(stratum_mask.sum())
    y_full = master[c23.MORTALITY_COL].to_numpy()
    n_events = int(y_full[stratum_mask].sum())

    retrieval = {}
    for rep in REPRESENTATIONS:
        source = source_openai if rep == "openai-3large" else None
        for variant in VARIANTS:
            r = retrieval_auroc_open_pool(master, stratum_mask, rep, variant, folds,
                                           full_gower=full_gower, source=source)
            retrieval[f"{rep}__{variant}"] = r
            print(f"[28a/{name}] {rep}/{variant}: AUROC={r['auroc_mean']} "
                  f"(n_folds_scored={r['n_folds_scored']}/{r['n_folds_total']}, "
                  f"n_excluded={r['n_queries_excluded_insufficient_neighbours']})", flush=True)

    result = {
        "stratum": name, "n": n, "n_events_90d_mortality": n_events,
        "event_rate": n_events / n if n else None,
        "retrieval_k25_openpool": retrieval,
        "wall_sec": time.time() - t0,
        "note": "OPEN-POOL design: folds built ONCE on the FULL 32,329-row master frame "
                "(byte-identical to 23a_retrieval_common/23b_run_retrieval's folds -- verified "
                "via sanity_check_published_16_reproduces_0_6206 below). Only the QUERY set "
                "(test_idx) is narrowed to this stratum per fold; the candidate/training pool "
                "(train_idx) is the FULL cross-cohort training fold, unrestricted to the "
                "stratum. Contrast with 26e_sofa_stratified.py's CLOSED-cohort design, where "
                "both folds AND candidate pool were rebuilt within the stratum alone.",
    }
    print(f"[28a] stratum={name} n={n} events={n_events} ({time.time() - t0:.1f}s)", flush=True)
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    master = c23.build_master_frame()
    print(f"[28a] master frame n_rows={len(master)}", flush=True)

    sanity = c23.sanity_check_published_16_reproduces_0_6206(master)
    print(f"[28a] sanity check (published_16 vs stored 0.6205749882834398): "
          f"{json.dumps(sanity)}", flush=True)

    sofa_sparse = pd.read_csv(c23.DATASET, usecols=["admission_ID", "sofa_score"], low_memory=False)
    merged = master.merge(sofa_sparse, on="admission_ID", how="left", validate="one_to_one")
    assert len(merged) == 32329
    master = merged

    gower_ids = pd.read_csv(c22g.GOWER_ADMISSION_IDS_PATH)["admission_ID"].to_numpy()
    assert np.array_equal(gower_ids, master["admission_ID"].to_numpy()), (
        "master frame row order does not match the stored Gower matrix's row order"
    )
    full_gower = c22g.load_gower_matrix()

    admission_ids = master["admission_ID"].to_numpy()
    patient_ids = master["patient_ID"].to_numpy()
    y = master[c23.MORTALITY_COL].to_numpy()

    # SAME folds used everywhere else in Task 23/26 -- built ONCE on the full
    # cohort, never rebuilt per stratum (that is precisely the closed-cohort
    # behaviour this script exists to avoid).
    folds = c23.build_folds(y, patient_ids)
    print(f"[28a] {len(folds)} folds built on FULL cohort, sizes: "
          f"{[(len(tr), len(te)) for tr, te in folds]}", flush=True)

    vecs = c23.load_representation_vectors("openai-3large", admission_ids)
    source_openai = c23.normalize_rows(vecs)
    print(f"[28a] openai-3large embedding aligned+normalized: shape={vecs.shape}", flush=True)

    mask_computable = master["sofa_score"].notna().to_numpy()
    print(f"[28a] sofa_computable n={mask_computable.sum()} "
          f"({100 * mask_computable.mean():.2f}%), not_computable n={(~mask_computable).sum()}",
          flush=True)

    results = {
        "design": "open_pool_query_only_masking",
        "sanity_check_published_16": sanity,
        "strata": {},
    }
    for name, mask in [("sofa_computable", mask_computable), ("sofa_not_computable", ~mask_computable)]:
        results["strata"][name] = run_stratum(name, master, mask, folds, full_gower, source_openai)
        results["env"] = env_manifest()
        DEST.write_text(json.dumps(results, indent=2))

    results["env"] = env_manifest()
    results["closed_cohort_reference_file"] = str(CLOSED_COHORT_PATH)
    DEST.write_text(json.dumps(results, indent=2))
    print(f"[28a] wrote {DEST}", flush=True)


if __name__ == "__main__":
    main()
