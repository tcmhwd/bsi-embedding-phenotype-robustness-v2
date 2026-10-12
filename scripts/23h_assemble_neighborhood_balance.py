#!/usr/bin/env python3
"""Task 23h -- assemble the four `neighborhood_balance_<representation>_
<variant>.json` files (`23g_run_neighborhood_balance.py`) into two flat
CSVs under `out/23_retrieval/`: `neighborhood_balance_smd.csv` (one row per
representation x variant x k x split x covariate, plus a
"MAX_ACROSS_COVARIATES" row per representation x variant x k x split) and
`neighborhood_balance_mortality_diff.csv` (one row per representation x
variant x k x split). Reads only already-written JSON -- login-node safe.
"""
import importlib
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c23f = importlib.import_module("23f_neighborhood_balance_common")

OUT = c23f.OUT


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run the matching 23g job first")
    return json.loads(path.read_text())


def main() -> None:
    smd_rows = []
    mort_rows = []

    for representation in c23f.BALANCE_REPRESENTATIONS:
        for variant in ["unrestricted", "past_only"]:
            p = load_json(OUT / f"neighborhood_balance_{representation}_{variant}.json")
            for k in p["k_values"]:
                for split in p["splits"]:
                    key = f"k{k}__{split}"
                    for cov in p["covariates"]:
                        s = p["smd_by_covariate"][key][cov]
                        smd_rows.append({
                            "representation": representation, "variant": variant, "k": k, "split": split,
                            "covariate": cov,
                            "n_index_cases_total": p["n_index_cases_total"],
                            "n_index_cases_valid_split": s["n_index_cases_valid_split"],
                            "median_abs_smd": s["median"], "q1_abs_smd": s["q1"], "q3_abs_smd": s["q3"],
                            "mean_abs_smd": s["mean"],
                            "pct_abs_smd_gt_0.1": s["pct_gt_0.1"], "pct_abs_smd_gt_0.25": s["pct_gt_0.25"],
                        })
                    s = p["smd_max_across_covariates"][key]
                    smd_rows.append({
                        "representation": representation, "variant": variant, "k": k, "split": split,
                        "covariate": "MAX_ACROSS_COVARIATES",
                        "n_index_cases_total": p["n_index_cases_total"],
                        "n_index_cases_valid_split": s["n_index_cases_valid_split"],
                        "median_abs_smd": s["median"], "q1_abs_smd": s["q1"], "q3_abs_smd": s["q3"],
                        "mean_abs_smd": s["mean"],
                        "pct_abs_smd_gt_0.1": s["pct_gt_0.1"], "pct_abs_smd_gt_0.25": s["pct_gt_0.25"],
                    })

                    m = p["mortality_rate_diff"][key]
                    mort_rows.append({
                        "representation": representation, "variant": variant, "k": k, "split": split,
                        "n_index_cases_total": p["n_index_cases_total"],
                        "n_index_cases_valid_split": m["n_index_cases_valid_split"],
                        "median_mortality_rate_diff_treated_minus_control": m["median"],
                        "q1_mortality_rate_diff": m["q1"], "q3_mortality_rate_diff": m["q3"],
                        "mean_mortality_rate_diff_treated_minus_control": m["mean"],
                        "pct_cases_treated_group_higher_mortality": m["pct_cases_treated_group_higher_mortality"],
                    })

    smd_df = pd.DataFrame(smd_rows)
    mort_df = pd.DataFrame(mort_rows)
    smd_path = OUT / "neighborhood_balance_smd.csv"
    mort_path = OUT / "neighborhood_balance_mortality_diff.csv"
    smd_df.to_csv(smd_path, index=False)
    mort_df.to_csv(mort_path, index=False)
    print(smd_df.to_string(index=False))
    print(mort_df.to_string(index=False))
    print(f"[23h] wrote {smd_path} ({len(smd_df)} rows)")
    print(f"[23h] wrote {mort_path} ({len(mort_df)} rows)")


if __name__ == "__main__":
    main()
