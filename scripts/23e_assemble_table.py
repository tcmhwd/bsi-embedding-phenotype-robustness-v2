#!/usr/bin/env python3
"""Task 23e -- assemble the final representation x k x variant retrieval
AUROC table, alongside the published_16 cluster-label baseline (0.6206)
and the SOFA/Charlson clinical anchors, into one CSV/MD under
`out/23_retrieval/`. Reads only already-written point_*.json files (no new
computation, no new imports of heavy libraries) -- safe to run on the login
node per the project's interactive-work policy ("reading logs and result
tables").
"""
import importlib
import json
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c23 = importlib.import_module("23a_retrieval_common")

OUT = c23.OUT


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- run the matching 23b/23c job first")
    return json.loads(path.read_text())


def main() -> None:
    rows = []

    published = load_json(c23.PUBLISHED_16_POINT_PATH)
    rows.append({
        "representation": "published_16 (cluster labels, one-hot)",
        "variant": "n/a (fixed cluster assignment, not a per-query retrieval)",
        "k": None,
        "auroc_mean": published["auroc_90d_mortality_mean"],
        "auroc_std": published["auroc_90d_mortality_std"],
        "n_folds_scored": 5,
        "n_queries_excluded": 0,
        "note": "Task 22 Task C reference row, reused unchanged (not recomputed here).",
    })

    for feat_name, label in [("sofa", "SOFA (continuous, alone)"), ("charlson", "Charlson (continuous, alone)")]:
        p = load_json(OUT / f"point_{feat_name}.json")
        rows.append({
            "representation": label,
            "variant": "n/a (single continuous predictor, not a retrieval)",
            "k": None,
            "auroc_mean": p["auroc_90d_mortality_mean"],
            "auroc_std": p["auroc_90d_mortality_std"],
            "n_folds_scored": 5,
            "n_queries_excluded": 0,
            "note": f"logistic regression on {p['source_column']} alone, {p['imputation'][:60]}...",
        })

    for representation in c23.REPRESENTATIONS:
        for variant in c23.VARIANTS:
            p = load_json(OUT / f"point_{representation}_{variant}.json")
            for k in c23.K_VALUES:
                pk = p["per_k"][str(k)]
                rows.append({
                    "representation": representation,
                    "variant": variant,
                    "k": k,
                    "auroc_mean": pk["auroc_mean"],
                    "auroc_std": pk["auroc_std"],
                    "n_folds_scored": pk["n_folds_scored"],
                    "n_queries_excluded": pk["n_queries_excluded_insufficient_neighbours"],
                    "note": f"n_same_patient_neighbours_found={p['n_same_patient_neighbours_found']}, "
                            f"n_self_retrievals_found={p['n_self_retrievals_found']}",
                })

    df = pd.DataFrame(rows)
    csv_path = OUT / "comparison_table.csv"
    md_path = OUT / "comparison_table.md"
    df.to_csv(csv_path, index=False)
    # `tabulate` (df.to_markdown()'s dependency) is not installed in this
    # env -- build the markdown table manually instead of adding a new
    # dependency for one cosmetic output.
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = "\n".join(
        "| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    )
    md_path.write_text("\n".join([header, sep, body]) + "\n")
    print(df.to_string(index=False))
    print(f"[23e] wrote {csv_path}")
    print(f"[23e] wrote {md_path}")


if __name__ == "__main__":
    main()
