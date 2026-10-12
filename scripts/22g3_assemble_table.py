#!/usr/bin/env python3
"""Task 22 Task C, step 3 -- assemble the completed five-method comparator
table from the per-(method,k) point-estimate JSONs and bootstrap shard CSVs.

Writes only under
`phenotype_revision/out/22_published_objects/task_c_comparators/`:
  - `comparison_table.csv` -- same column set as
    `out/05_comparators/comparison_table.csv` (read-only original, never
    written here) plus `auroc_cv_scheme` and `bootstrap_resampling_unit`
    columns making the two methodology fixes explicit in the table itself.
  - `comparison_table.md` -- one markdown table, run whenever invoked (safe
    to re-run as more (method,k,shard) pieces complete -- it reports
    whatever is on disk, with `n_boot_used` showing partial progress
    honestly rather than padding to 100).

Run this any time; it never blocks on missing pieces -- it reports NaN /
partial `n_boot_used` for anything not yet computed, so it is safe to check
progress mid-run.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/tcmhwd/AI/AST_ai")
OUT = ROOT / "phenotype_revision/out/22_published_objects/task_c_comparators"
BOOT_DIR = OUT / "bootstrap"
METHODS = ["kprototypes", "famd_kmeans", "gmm_structured", "gower_pam", "lca"]
K_VALUES = [16, 32]

# k-prototypes bootstrap: deliberately NOT attempted (author/coordinator
# decision, 2026-09-06) -- single point-estimate fit measured at 6,126.3s
# (k=16) / 11,735.9s (k=32) on the full N=32,329 (this run's own
# `fit_wall_time_sec`, in the same ballpark as the earlier
# `fourmethod_timing.json` measurement of 5,829.7s for k=16). Extrapolating
# to a ~63%-patient bootstrap subsample at n_boot=100 for both k gives an
# estimated 300+ CPU-hours, with no equivalent to the Gower distance-matrix
# reuse shortcut (k-prototypes recomputes its mixed distance from raw data
# on every fit, not from a cacheable pairwise matrix). Four of five methods
# already show the same qualitative pattern (own-bootstrap ARI well above
# ARI-vs-embedding/published), so this does not change the table's
# conclusion. Reported as NOT ATTEMPTED, not as 0 or NaN presented silently.
KPROTOTYPES_BOOTSTRAP_NOTE = (
    "NOT ATTEMPTED -- resource cost (est. 300+ CPU-hours for n_boot=100 x k in {16,32}; "
    "single-fit costs measured at 6126.3s/11735.9s full-N; no distance-matrix-reuse shortcut "
    "exists for this method). Decision: author/coordinator, 2026-09-06 -- four of five methods "
    "already show the same qualitative pattern, so this is not expected to change the table's "
    "conclusion."
)


def bootstrap_summary(method: str, k: int) -> dict:
    shards = sorted(BOOT_DIR.glob(f"{method}_k{k}_shard*.csv"))
    if not shards:
        return dict(bootstrap_ari_mean=float("nan"), n_boot_used=0)
    dfs = []
    for p in shards:
        try:
            if p.stat().st_size > 0:
                dfs.append(pd.read_csv(p))
        except pd.errors.EmptyDataError:
            continue  # shard file just created by a concurrent SLURM task, header not yet flushed
    if not dfs:
        return dict(bootstrap_ari_mean=float("nan"), n_boot_used=0)
    df = pd.concat(dfs, ignore_index=True).drop_duplicates(subset="boot")
    if df.empty:
        return dict(bootstrap_ari_mean=float("nan"), n_boot_used=0)
    return dict(
        bootstrap_ari_mean=float(df["ari"].mean()),
        bootstrap_ari_std=float(df["ari"].std()),
        n_boot_used=int(len(df)),
        bootstrap_wall_time_sec=float(df["wall_time_sec"].sum()),
    )


def main() -> None:
    rows = []
    for method in METHODS:
        for k in K_VALUES:
            point_path = OUT / f"point_{method}_k{k}.json"
            if not point_path.exists():
                rows.append(dict(method=method, k=k, status="POINT ESTIMATE NOT YET COMPUTED"))
                continue
            point = json.loads(point_path.read_text())
            boot = bootstrap_summary(method, k)
            rows.append(dict(
                method=method, k=k,
                n_clusters_actual=point["n_clusters_actual"],
                bootstrap_ari_mean=boot.get("bootstrap_ari_mean"),
                bootstrap_ari_std=boot.get("bootstrap_ari_std"),
                n_boot_used=boot.get("n_boot_used", 0),
                n_boot_target=100,
                bootstrap_resampling_unit="patient_ID (resample-with-replacement, dedupe, "
                                          "keep all admissions of each drawn patient)",
                ari_vs_embedding=point["ari_vs_embedding"],
                ari_vs_published_paper=point["ari_vs_published_paper"],
                auroc_90d_mortality_mean=point["auroc_90d_mortality_mean"],
                auroc_90d_mortality_std=point["auroc_90d_mortality_std"],
                auroc_cv_scheme=point["auroc_cv_scheme"],
                top_features_path=point["top_features_path"],
                fit_wall_time_sec=point["fit_wall_time_sec"],
                bootstrap_wall_time_sec=boot.get("bootstrap_wall_time_sec"),
                status=(
                    "COMPLETE" if boot.get("n_boot_used", 0) >= 100 else
                    KPROTOTYPES_BOOTSTRAP_NOTE if (method == "kprototypes" and boot.get("n_boot_used", 0) == 0) else
                    f"PARTIAL ({boot.get('n_boot_used', 0)}/100 boot reps)"
                ),
            ))
    # Comparison baseline rows: the published embedding-derived phenotypes
    # themselves, scored under the IDENTICAL AUROC protocol
    # (`22h_published_baseline_rows.py`) -- without these, the five
    # structured comparators' AUROC has nothing to be compared against.
    # No bootstrap concept applies (these are fixed published labels, not a
    # refit procedure) -- bootstrap columns are N/A, not 0/NaN-as-if-untried.
    for method in ["published_16", "published_ecoli_10"]:
        point_path = OUT / f"point_{method}.json"
        if not point_path.exists():
            rows.append(dict(method=method, status="NOT YET COMPUTED (run 22h_published_baseline_rows.py)"))
            continue
        point = json.loads(point_path.read_text())
        rows.append(dict(
            method=method, k=point["k"],
            n_clusters_actual=point["n_clusters_actual"],
            bootstrap_ari_mean="N/A (fixed published labels, no refit/bootstrap procedure applies)",
            n_boot_used="N/A", n_boot_target="N/A",
            bootstrap_resampling_unit="N/A",
            ari_vs_embedding=point["ari_vs_embedding"],
            ari_vs_published_paper=point["ari_vs_published_paper"],
            auroc_90d_mortality_mean=point["auroc_90d_mortality_mean"],
            auroc_90d_mortality_std=point["auroc_90d_mortality_std"],
            auroc_cv_scheme=point["auroc_cv_scheme"],
            fit_wall_time_sec=None,
            status="COMPLETE (comparison baseline)",
            population_note=point.get("population_note"),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "comparison_table.csv", index=False)

    def _md(df):
        header = "| " + " | ".join(df.columns) + " |"
        sep = "| " + " | ".join("---" for _ in df.columns) + " |"
        body = "\n".join("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |"
                          for r in df.itertuples(index=False, name=None))
        return "\n".join([header, sep, body])

    (OUT / "comparison_table.md").write_text(_md(df) + "\n")
    print(df.to_string(index=False))
    print(f"\n[assemble] wrote {OUT}/comparison_table.csv and comparison_table.md")


if __name__ == "__main__":
    main()
