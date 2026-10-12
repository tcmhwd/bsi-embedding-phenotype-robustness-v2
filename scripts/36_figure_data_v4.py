#!/usr/bin/env python3
"""V4 additive tidy-data step for Figure 4 (see QC/V4_PLAN.md, section 4.1).

This script is ADDITIVE and read-only with respect to everything already on
disk. It does NOT modify or overwrite `scripts/33_figure_data.py` or any file
it wrote (in particular `out/33_figures/retrieval_ladder.csv`, which is left
untouched with its original 7 rows / 5 rungs). It reads only stored outputs
--

    out/23_retrieval/comparison_table.csv
    out/23_retrieval/retrieval_bootstrap.json
    out/23_retrieval/baseline_bootstrap.json

-- and writes exactly two NEW files, used by scripts/36_figures_v4.R:

    out/33_figures/retrieval_ladder_v4.csv
    out/33_figures/retrieval_bootstrap_contrasts.csv

No new computation, no re-estimation: every number here is copied verbatim
from the three source files above.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
RETRIEVAL_DIR = REV / "out" / "23_retrieval"
FIGDATA_DIR = REV / "out" / "33_figures"

COMPARISON_TABLE = RETRIEVAL_DIR / "comparison_table.csv"
RETRIEVAL_BOOTSTRAP = RETRIEVAL_DIR / "retrieval_bootstrap.json"
BASELINE_BOOTSTRAP = RETRIEVAL_DIR / "baseline_bootstrap.json"

OUT_LADDER = FIGDATA_DIR / "retrieval_ladder_v4.csv"
OUT_CONTRASTS = FIGDATA_DIR / "retrieval_bootstrap_contrasts.csv"


def pick(df: pd.DataFrame, repr_prefix: str, variant_prefix: str, k=None) -> pd.Series:
    """Same selection guard as scripts/33_figure_data.py::fig4a_retrieval_ladder.pick()."""
    sub = df[df["representation"].str.startswith(repr_prefix) &
              df["variant"].str.startswith(variant_prefix)]
    if k is not None:
        sub = sub[sub["k"] == k]
    assert len(sub) == 1, f"expected 1 row for {repr_prefix}/{variant_prefix}/{k}, got {len(sub)}"
    return sub.iloc[0]


def build_ladder(df: pd.DataFrame) -> pd.DataFrame:
    """Seven-rung ladder (V4_PLAN.md section 4.1 table). Rungs 1-4 (Gower and
    all three embedding representations) get both `unrestricted` and
    `past_only` rows at k=25; rungs 5-7 (the non-retrieval comparators) get a
    single `unrestricted`/`n/a...` row each."""
    rungs = [
        dict(rung=1, label="Structured retrieval (Gower, k = 25)",
             representation="gower", group="Structured retrieval",
             is_original_representation=False,
             row=pick(df, "gower", "unrestricted", 25.0),
             past_row=pick(df, "gower", "past_only", 25.0)),
        dict(rung=2, label="Embedding retrieval (3-small, k = 25)",
             representation="openai-3small", group="Embedding retrieval",
             is_original_representation=False,
             row=pick(df, "openai-3small", "unrestricted", 25.0),
             past_row=pick(df, "openai-3small", "past_only", 25.0)),
        dict(rung=3, label="Embedding retrieval (BGE-M3, k = 25)",
             representation="bge-m3", group="Embedding retrieval",
             is_original_representation=False,
             row=pick(df, "bge-m3", "unrestricted", 25.0),
             past_row=pick(df, "bge-m3", "past_only", 25.0)),
        dict(rung=4, label="Embedding retrieval (3-large, k = 25)",
             representation="openai-3large", group="Embedding retrieval",
             is_original_representation=True,
             row=pick(df, "openai-3large", "unrestricted", 25.0),
             past_row=pick(df, "openai-3large", "past_only", 25.0)),
        dict(rung=5, label="SOFA", representation="SOFA (continuous, alone)",
             group="Comparator", is_original_representation=False,
             row=pick(df, "SOFA", "n/a")),
        dict(rung=6, label="Charlson", representation="Charlson (continuous, alone)",
             group="Comparator", is_original_representation=False,
             row=pick(df, "Charlson", "n/a")),
        dict(rung=7, label="Originally derived cluster labels",
             representation="published_16 (cluster labels, one-hot)",
             group="Comparator", is_original_representation=False,
             row=pick(df, "published_16", "n/a")),
    ]

    rows = []
    for r in rungs:
        row = r["row"]
        rows.append(dict(
            rung=r["rung"], label=r["label"], representation=r["representation"],
            variant="unrestricted", estimate=float(row["auroc_mean"]),
            sd=float(row["auroc_std"]), n_folds=row["n_folds_scored"],
            group=r["group"], is_original_representation=r["is_original_representation"],
        ))
        if "past_row" in r:
            pr = r["past_row"]
            rows.append(dict(
                rung=r["rung"], label=r["label"], representation=r["representation"],
                variant="past_only", estimate=float(pr["auroc_mean"]),
                sd=float(pr["auroc_std"]), n_folds=pr["n_folds_scored"],
                group=r["group"], is_original_representation=r["is_original_representation"],
            ))
    out = pd.DataFrame(rows, columns=[
        "rung", "label", "representation", "variant", "estimate", "sd",
        "n_folds", "group", "is_original_representation",
    ])
    return out


def load_bootstrap(path: Path) -> dict:
    with open(path) as fh:
        d = json.load(fh)
    assert d["n_boot"] == 10000, f"{path}: expected n_boot=10000, got {d['n_boot']}"
    assert d["seed"] == 2026, f"{path}: expected seed=2026, got {d['seed']}"
    return d


def build_contrasts(retrieval_boot: dict, baseline_boot: dict) -> pd.DataFrame:
    """Six paired patient-level bootstrap contrasts for Figure 4 panel b
    (V4_PLAN.md section 4.1, second table), all `unrestricted|...` keys, in
    smallest-delta-first display order."""
    spec = [
        dict(order=1, comparator_label="Embedding retrieval (3-small)",
             comparator_group="Embedding retrieval",
             contrast_key="unrestricted|gower-openai-3small",
             source_json="retrieval_bootstrap.json", boot=retrieval_boot),
        dict(order=2, comparator_label="Embedding retrieval (BGE-M3)",
             comparator_group="Embedding retrieval",
             contrast_key="unrestricted|gower-bge-m3",
             source_json="retrieval_bootstrap.json", boot=retrieval_boot),
        dict(order=3, comparator_label="Embedding retrieval (3-large)",
             comparator_group="Embedding retrieval",
             contrast_key="unrestricted|gower-openai-3large",
             source_json="retrieval_bootstrap.json", boot=retrieval_boot),
        dict(order=4, comparator_label="SOFA", comparator_group="Comparator",
             contrast_key="unrestricted|gower-sofa",
             source_json="baseline_bootstrap.json", boot=baseline_boot),
        dict(order=5, comparator_label="Charlson", comparator_group="Comparator",
             contrast_key="unrestricted|gower-charlson",
             source_json="baseline_bootstrap.json", boot=baseline_boot),
        dict(order=6, comparator_label="Originally derived cluster labels",
             comparator_group="Comparator",
             contrast_key="unrestricted|gower-published_16",
             source_json="baseline_bootstrap.json", boot=baseline_boot),
    ]

    rows = []
    for s in spec:
        boot = s["boot"]
        c = boot["contrasts"][s["contrast_key"]]
        rows.append(dict(
            order=s["order"], contrast_key=s["contrast_key"],
            comparator_label=s["comparator_label"], comparator_group=s["comparator_group"],
            delta=float(c["observed_delta_auroc"]),
            ci_low=float(c["ci95_low"]), ci_high=float(c["ci95_high"]),
            share_delta_le_0=float(c["share_delta_le_0"]),
            n_boot=boot["n_boot"], seed=boot["seed"],
            n_patients=c["n_patients"], source_json=s["source_json"],
        ))
    out = pd.DataFrame(rows, columns=[
        "order", "contrast_key", "comparator_label", "comparator_group",
        "delta", "ci_low", "ci_high", "share_delta_le_0", "n_boot", "seed",
        "n_patients", "source_json",
    ])
    return out


def main() -> None:
    df = pd.read_csv(COMPARISON_TABLE)
    retrieval_boot = load_bootstrap(RETRIEVAL_BOOTSTRAP)
    baseline_boot = load_bootstrap(BASELINE_BOOTSTRAP)

    ladder = build_ladder(df)
    assert len(ladder) == 11, f"expected 11 rows (4 rungs x 2 variants + 3 rungs x 1), got {len(ladder)}"
    ladder.to_csv(OUT_LADDER, index=False)
    print(f"[36_figure_data_v4] wrote {OUT_LADDER} ({len(ladder)} rows)")

    contrasts = build_contrasts(retrieval_boot, baseline_boot)
    assert len(contrasts) == 6, f"expected 6 contrast rows, got {len(contrasts)}"
    assert (contrasts["share_delta_le_0"] == 0.0).all()
    contrasts.to_csv(OUT_CONTRASTS, index=False)
    print(f"[36_figure_data_v4] wrote {OUT_CONTRASTS} ({len(contrasts)} rows)")


if __name__ == "__main__":
    main()
