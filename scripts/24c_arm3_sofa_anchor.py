#!/usr/bin/env python3
"""Arm 3 SOFA anchor (coordinator-directed, 2026-09-06) -- companion to
`23c_run_anchors.py`'s Arm 1 SOFA-alone anchor (`sofa_score_2` alone,
`out/23_retrieval/point_sofa.json`).

Computes AUROC(90d_mortality) via the IDENTICAL StratifiedGroupKFold(5,
groups=patient_ID, seed=42) CV as every other anchor/retrieval/comparator
number in this project, for THREE feature sets built from
`05_baseline_comparators.build_structured_matrix_with_missingness_indicators`:
  - "sofa_2_alone"            : [sofa_score_2]                      (== Arm 1, reproduced here for the side-by-side table)
  - "sofa_was_measured_alone" : [sofa_score_was_measured]            (bonus, not requested -- shows how much signal the indicator alone carries)
  - "sofa_2_plus_indicator"   : [sofa_score_2, sofa_score_was_measured]  (Arm 3 proper)

Writes only under `phenotype_revision/out/24_dense_fix/arm3_dense_plus_missingness_indicator/`:
  - `point_sofa_arm3.json`

Outcome-blind scope: `90d_mortality` read only as the label being predicted,
same as `23c_run_anchors.py` and `23a_retrieval_common.py`.
"""
import importlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c23 = importlib.import_module("23a_retrieval_common")
comparators = c23.comparators  # 05_baseline_comparators

OUT = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision/out/24_dense_fix/arm3_dense_plus_missingness_indicator")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    master = c23.build_master_frame()
    print(f"[24c/arm3_sofa_anchor] n_rows={len(master)}", flush=True)

    sanity = c23.sanity_check_published_16_reproduces_0_6206(master)
    print(f"[24c/arm3_sofa_anchor] sanity check (published_16 vs stored 0.6205749882834398): "
          f"{json.dumps(sanity)}", flush=True)

    admission_ids = master["admission_ID"].to_numpy()
    patient_ids = master["patient_ID"].to_numpy()
    y = master[c23.MORTALITY_COL].to_numpy()

    matrix, categorical_cols = comparators.build_structured_matrix_with_missingness_indicators(
        master[["admission_ID"]]
    )
    imputed = comparators.impute_structured_matrix(matrix, categorical_cols)

    sofa2 = imputed["sofa_score_2"].to_numpy()
    indicator = imputed["sofa_score_was_measured"].to_numpy()
    print(f"[24c/arm3_sofa_anchor] sofa_score_was_measured distribution: "
          f"measured={int(indicator.sum())}, assume-normal-imputed={int((1 - indicator).sum())}", flush=True)

    feature_sets = {
        "sofa_2_alone": sofa2.reshape(-1, 1),
        "sofa_was_measured_alone": indicator.reshape(-1, 1),
        "sofa_2_plus_indicator": __import__("numpy").column_stack([sofa2, indicator]),
    }

    rows = {}
    for name, X in feature_sets.items():
        auc_mean, auc_std = c23.evaluate_feature_matrix_grouped(X, admission_ids, patient_ids, y)
        print(f"[24c/arm3_sofa_anchor] {name}: AUROC={auc_mean:.4f}+-{auc_std:.4f}", flush=True)
        rows[name] = {"auroc_90d_mortality_mean": auc_mean, "auroc_90d_mortality_std": auc_std}

    result = {
        "arm": "3 (dense/_2 severity value + explicit was-measured indicator, coordinator-directed 2026-09-06)",
        "n_missing_before_impute_sofa_score_2": int(matrix["sofa_score_2"].isna().sum()),
        "n_measured_sofa_score": int(indicator.sum()),
        "n_assume_normal_imputed_sofa_score_2": int((1 - indicator).sum()),
        "feature_sets": rows,
        "requested_row": "sofa_2_plus_indicator",
        "bonus_rows_not_requested": ["sofa_was_measured_alone"],
        "reference_rows": {
            "arm1_sofa_2_alone_from_23c": "out/23_retrieval/point_sofa.json (should match sofa_2_alone above)",
            "arm2_complete_case_sofa_from_before_snapshot": "out/24_dense_fix/before_snapshot/retrieval/point_sofa.json",
            "published_16_sanity": sanity,
        },
        "auroc_cv_scheme": f"StratifiedGroupKFold({c23.CV_SPLITS}, groups=patient_ID), seed={c23.SEED} -- "
                           "same fold construction as every retrieval/anchor/comparator row.",
        "n_rows": int(len(master)),
        "env": c23.env_manifest(),
    }
    out_path = OUT / "point_sofa_arm3.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in result.items() if k != "env"}, indent=2, ensure_ascii=False), flush=True)
    print(f"[24c/arm3_sofa_anchor] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
