#!/usr/bin/env python3
"""Task 23g -- neighbourhood confounding-by-indication DIAGNOSTIC (see
`23f_neighborhood_balance_common.py` module docstring for full scope/
rationale -- this is a DESCRIPTIVE balance check, NOT a causal estimate).

For every index admission, using the EXACT SAME retrieved neighbourhood as
the main retrieval analysis (`23b_run_retrieval.py`'s `build_valid_mask`/
`compute_score_matrix`, reused here unchanged -- never self, never same
patient, train-fold only, and for `past_only` additionally restricted to
strictly-earlier `test_date`), split the neighbours by one of three
empiric-treatment flags and compute the |standardized mean difference| of
each balance covariate between the two treatment-defined subgroups, plus
(purely descriptively) the same-neighbourhood 90-day mortality-rate
difference between them.

--representation in {openai-3large, gower}
--variant       in {unrestricted, past_only}
Computes ALL of k in {10, 25, 50} (25 primary) x all 3 treatment splits in
one invocation -- reuses the SAME per-fold top-k neighbour computation
(`c23.topk_ordered`) across every (k, split) combination, so the only
per-combination cost is a handful of vectorized numpy reductions over an
(n_test, k<=50) array, not a new similarity/distance computation.

Outcome-blind scope: `90d_mortality` is read ONLY for the descriptive
mortality-rate-difference number, never for neighbour selection (identical
neighbour-retrieval code path as the main analysis, which never reads
`90d_mortality` before forming the neighbour pool).
"""
import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c23 = importlib.import_module("23a_retrieval_common")
c23b = importlib.import_module("23b_run_retrieval")
c23f = importlib.import_module("23f_neighborhood_balance_common")


def summarize(arr: np.ndarray) -> dict:
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return {"n_index_cases_valid_split": 0, "median": None, "q1": None, "q3": None, "mean": None}
    return {
        "n_index_cases_valid_split": int(len(arr)),
        "median": float(np.median(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "mean": float(np.mean(arr)),
    }


def summarize_abs_smd(arr: np.ndarray) -> dict:
    base = summarize(arr)
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) > 0:
        base["pct_gt_0.1"] = float((arr > 0.1).mean())
        base["pct_gt_0.25"] = float((arr > 0.25).mean())
    else:
        base["pct_gt_0.1"] = None
        base["pct_gt_0.25"] = None
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--representation", required=True, choices=c23f.BALANCE_REPRESENTATIONS)
    ap.add_argument("--variant", required=True, choices=c23.VARIANTS)
    args = ap.parse_args()

    c23f.OUT.mkdir(parents=True, exist_ok=True)
    out_path = c23f.OUT / f"neighborhood_balance_{args.representation}_{args.variant}.json"

    master = c23.build_master_frame()
    print(f"[23g/{args.representation}/{args.variant}] n_rows={len(master)}", flush=True)

    sanity = c23.sanity_check_published_16_reproduces_0_6206(master)
    print(f"[23g/{args.representation}/{args.variant}] sanity check: {json.dumps(sanity)}", flush=True)

    covariates = c23f.load_balance_covariates(master)
    assert (covariates["admission_ID"].to_numpy() == master["admission_ID"].to_numpy()).all(), (
        "covariate frame order does not match master frame order"
    )
    print(f"[23g/{args.representation}/{args.variant}] covariates loaded: "
          f"{c23f.BALANCE_COVARIATES}, splits: {c23f.TREATMENT_SPLITS}", flush=True)

    admission_ids = master["admission_ID"].to_numpy()
    patient_ids = master["patient_ID"].to_numpy()
    y = master[c23.MORTALITY_COL].to_numpy()
    dates_ordinal = master["test_date_ordinal"].to_numpy()

    folds = c23.build_folds(y, patient_ids)

    if args.representation == "gower":
        source = c23.load_gower_matrix_checked(admission_ids)
    else:
        vecs = c23.load_representation_vectors(args.representation, admission_ids)
        source = c23.normalize_rows(vecs)
        print(f"[23g/{args.representation}/{args.variant}] embedding aligned+normalized: shape={vecs.shape}", flush=True)

    k_max = max(c23f.BALANCE_K_VALUES)

    smd_accum = {(k, s, c): [] for k in c23f.BALANCE_K_VALUES for s in c23f.TREATMENT_SPLITS for c in c23f.BALANCE_COVARIATES}
    max_smd_accum = {(k, s): [] for k in c23f.BALANCE_K_VALUES for s in c23f.TREATMENT_SPLITS}
    mort_diff_accum = {(k, s): [] for k in c23f.BALANCE_K_VALUES for s in c23f.TREATMENT_SPLITS}

    for fi, (train_idx, test_idx) in enumerate(folds):
        t0 = time.time()
        valid_mask = c23b.build_valid_mask(args.variant, train_idx, test_idx, dates_ordinal)
        score_matrix = c23b.compute_score_matrix(args.representation, source, train_idx, test_idx)
        top_idx_sorted, valid_counts = c23.topk_ordered(score_matrix, valid_mask, k_max)

        y_train = y[train_idx]
        cov_train = {c: covariates[c].to_numpy()[train_idx] for c in c23f.BALANCE_COVARIATES}
        split_train = {s: covariates[s].to_numpy()[train_idx] for s in c23f.TREATMENT_SPLITS}

        for k in c23f.BALANCE_K_VALUES:
            idx_safe, use_mask = c23.neighbour_selection_mask(top_idx_sorted, valid_counts, k)

            for split in c23f.TREATMENT_SPLITS:
                treat_vals = split_train[split][idx_safe]
                treated_mask = use_mask & (treat_vals == 1)
                control_mask = use_mask & (treat_vals == 0)
                n_treated = treated_mask.sum(axis=1)
                n_control = control_mask.sum(axis=1)
                valid_case = (n_treated >= 1) & (n_control >= 1)

                per_cov_abs_smd = []
                for cov in c23f.BALANCE_COVARIATES:
                    cov_vals = cov_train[cov][idx_safe].astype(float)
                    m1, v1, _ = c23f.group_mean_var(cov_vals, treated_mask)
                    m0, v0, _ = c23f.group_mean_var(cov_vals, control_mask)
                    smd = c23f.standardized_mean_diff_vec(m1, v1, m0, v0)
                    abs_smd = np.where(valid_case, np.abs(smd), np.nan)
                    smd_accum[(k, split, cov)].append(abs_smd[valid_case])
                    per_cov_abs_smd.append(abs_smd)

                stacked = np.vstack(per_cov_abs_smd)  # (n_covariates, n_test)
                with np.errstate(invalid="ignore"):
                    case_max = np.nanmax(np.where(np.isnan(stacked), -np.inf, stacked), axis=0)
                case_max = np.where(np.isneginf(case_max), np.nan, case_max)
                max_smd_accum[(k, split)].append(case_max[valid_case])

                mort_vals = y_train[idx_safe].astype(float)
                mort1, _, _ = c23f.group_mean_var(mort_vals, treated_mask)
                mort0, _, _ = c23f.group_mean_var(mort_vals, control_mask)
                mort_diff = np.where(valid_case, mort1 - mort0, np.nan)
                mort_diff_accum[(k, split)].append(mort_diff[valid_case])

        print(f"[23g/{args.representation}/{args.variant}] fold {fi + 1}/{len(folds)} done in "
              f"{time.time() - t0:.1f}s", flush=True)

    result = {
        "representation": args.representation,
        "variant": args.variant,
        "k_values": c23f.BALANCE_K_VALUES,
        "splits": c23f.TREATMENT_SPLITS,
        "covariates": c23f.BALANCE_COVARIATES,
        "n_index_cases_total": int(len(master)),
        "scope_note": "DESCRIPTIVE balance diagnostic quantifying confounding-by-indication risk for a "
                      "'similar past cases, treated vs untreated' clinician-facing display -- NOT a "
                      "causal or treatment-effect estimate. 90d_mortality is read ONLY for the "
                      "same-neighbourhood mortality-rate difference reported alongside |SMD|, never for "
                      "neighbour selection (identical, unmodified neighbour-retrieval code path as the "
                      "main analysis in 23b_run_retrieval.py).",
        "neighbour_pool": ("train fold only, never the index admission, never another admission of the "
                            "same patient" if args.variant == "unrestricted" else
                            "train fold AND test_date strictly earlier than the query's own test_date"),
        "smd_by_covariate": {},
        "smd_max_across_covariates": {},
        "mortality_rate_diff": {},
        "dense_variant_fix": {
            "applied": True,
            "note": "the 'sofa_score' covariate is sourced from the dense 'sofa_score_2' column "
                    "(0.2% missing), not the sparse 'sofa_score' (88.3% missing) used before this "
                    "fix -- see 23f_neighborhood_balance_common.py module docstring and "
                    "05_baseline_comparators.py docstring point 2.",
        },
        "env": c23.env_manifest(),
    }

    for k in c23f.BALANCE_K_VALUES:
        for split in c23f.TREATMENT_SPLITS:
            key = f"k{k}__{split}"
            for cov in c23f.BALANCE_COVARIATES:
                arr = np.concatenate(smd_accum[(k, split, cov)]) if smd_accum[(k, split, cov)] else np.array([])
                result["smd_by_covariate"].setdefault(key, {})[cov] = summarize_abs_smd(arr)

            arr_max = np.concatenate(max_smd_accum[(k, split)]) if max_smd_accum[(k, split)] else np.array([])
            result["smd_max_across_covariates"][key] = summarize_abs_smd(arr_max)

            arr_mort = np.concatenate(mort_diff_accum[(k, split)]) if mort_diff_accum[(k, split)] else np.array([])
            mort_summary = summarize(arr_mort)
            arr_mort_clean = arr_mort[~np.isnan(arr_mort)]
            mort_summary["pct_cases_treated_group_higher_mortality"] = (
                float((arr_mort_clean > 0).mean()) if len(arr_mort_clean) > 0 else None
            )
            result["mortality_rate_diff"][key] = mort_summary

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "env"}, indent=2, ensure_ascii=False), flush=True)
    print(f"[23g/{args.representation}/{args.variant}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
