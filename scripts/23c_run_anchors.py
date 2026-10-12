#!/usr/bin/env python3
"""Task 23c -- SOFA-alone and Charlson-alone clinical anchors, same CV/
outcome as every retrieval row (`23b_run_retrieval.py`) and the stored
`published_16` comparator (0.6206): `StratifiedGroupKFold(5,
groups=patient_ID, seed=42)`, `90d_mortality` outcome. Not a retrieval
method -- no neighbour pool, no k, no unrestricted/past_only variant (a
single continuous predictor from the index admission's OWN row is not
affected by a "which admissions can this admission see" restriction the
way a neighbour-lookup is) -- these are fixed reference rows reported
alongside both the unrestricted and past-only retrieval tables.

Reuses `05_baseline_comparators.{build_structured_matrix,impute_structured_matrix}`
UNMODIFIED for the SOFA/Charlson values themselves: median imputation,
same policy already applied to these two columns inside the 44-variable
structured comparator matrix -- not re-derived here, so the anchor numbers
use the SAME imputed values the structured comparators were built from.

Runs the same runtime sanity check as `23b_run_retrieval.py`
(`sanity_check_published_16_reproduces_0_6206`) before computing anything,
for the same reason: proves the population/order this script uses matches
the one that produced 0.6206, rather than assuming it.

Outcome-blind scope: see `23a_retrieval_common.py` module docstring.
"""
import importlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c23 = importlib.import_module("23a_retrieval_common")


def main() -> None:
    c23.OUT.mkdir(parents=True, exist_ok=True)

    master = c23.build_master_frame()
    print(f"[23c] n_rows={len(master)}", flush=True)

    sanity = c23.sanity_check_published_16_reproduces_0_6206(master)
    print(f"[23c] sanity check (published_16 vs stored 0.6205749882834398): {json.dumps(sanity)}", flush=True)

    admission_ids = master["admission_ID"].to_numpy()
    patient_ids = master["patient_ID"].to_numpy()
    y = master[c23.MORTALITY_COL].to_numpy()

    matrix, categorical_cols = c23.comparators.build_structured_matrix(master[["admission_ID"]])
    imputed = c23.comparators.impute_structured_matrix(matrix, categorical_cols)

    # NOTE (dense-variant fix): "sofa_score" here means the DENSE "sofa_score_2"
    # column -- 05_baseline_comparators.NUMERIC_COLS was switched from the
    # sparse "sofa_score" (88.3% missing) to its dense twin "sofa_score_2"
    # (0.2% missing, 100% agreement with "sofa_score" wherever both are
    # present; see that module's docstring point 2 and
    # scripts/24_verify_dense_pairs.py). The published SOFA-alone anchor
    # (0.552) was computed on the 88%-missing sparse column and is superseded
    # by this run.
    for feat_name, col in [("sofa", "sofa_score_2"), ("charlson", "charlson_score")]:
        values = imputed[col].to_numpy()
        auc_mean, auc_std = c23.evaluate_continuous_predictor_grouped(values, admission_ids, patient_ids, y)
        result = {
            "anchor": feat_name,
            "source_column": col,
            "dense_variant_fix": {
                "applied": feat_name == "sofa",
                "superseded_source_column": "sofa_score" if feat_name == "sofa" else None,
                "note": ("dense-variant fix: source_column switched from the sparse 'sofa_score' "
                          "(88.3% missing) to its dense twin 'sofa_score_2' (0.2% missing) -- see "
                          "05_baseline_comparators.py docstring point 2") if feat_name == "sofa" else None,
            },
            "imputation": "05_baseline_comparators.impute_structured_matrix (column median) -- IDENTICAL "
                          "imputation already applied to this column inside the 44-variable structured "
                          "comparator matrix, not re-derived here.",
            "n_missing_before_impute": int(matrix[col].isna().sum()),
            "auroc_90d_mortality_mean": auc_mean,
            "auroc_90d_mortality_std": auc_std,
            "auroc_cv_scheme": f"StratifiedGroupKFold({c23.CV_SPLITS}, groups=patient_ID), seed={c23.SEED} -- "
                               "same fold construction as every retrieval row and the published_16 row "
                               "(verified via sanity_check_published_16 above).",
            "n_rows": int(len(master)),
            "sanity_check_published_16": sanity,
            "env": c23.env_manifest(),
        }
        out_path = c23.OUT / f"point_{feat_name}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(json.dumps({k: v for k, v in result.items() if k != "env"}, indent=2, ensure_ascii=False), flush=True)
        print(f"[23c] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
