#!/usr/bin/env python3
"""Task 22 Task A, aggregation -- combine S1/S2/S4/S5 raw outputs for the
published E. coli subphenotype object into one per-cluster PASS/FAIL matrix,
applying PRESPEC section 3 thresholds VERBATIM (copied into
`22a_ecoli_common.THRESHOLDS`, never re-typed here).

Runs in bsi_pheno (pure pandas aggregation over already-written JSON/CSV) --
no SLURM needed.

Writes only under `phenotype_revision/out/22_published_objects/task_a_ecoli/`:
  - `survival_matrix_s1s2s4s5.csv` -- one row per reference cluster
  - `SUMMARY.json` -- per-axis pass counts + overall reproduction summary
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c22 = importlib.import_module("22a_ecoli_common")

S1_DIR = c22.OUT / "s1_seed"
S2_DIR = c22.OUT / "s2_boot"
S4S5_DIR = c22.OUT / "s4s5_transport"
THRESH = c22.THRESHOLDS


def load_reproduction_summary() -> dict:
    r = json.loads((S1_DIR / "result_seed42_threads1.json").read_text())
    return dict(
        n_rows=r["n_rows"], n_clusters=r["n_clusters"], noise_frac=r["noise_frac"],
        n_noise=r["n_noise"], agreement_vs_reference=r["agreement_vs_reference"],
        jaccard_vs_reference=r["jaccard_vs_reference"],
        versions=r["versions"], python_executable=r["python_executable"],
    )


def aggregate_s1(clusters) -> pd.DataFrame:
    seed_jaccards = {c: [] for c in clusters}
    for s in range(1, 21):
        p = S1_DIR / f"result_seed{s}_threads1.json"
        r = json.loads(p.read_text())
        for c in clusters:
            seed_jaccards[c].append(r["jaccard_vs_reference"][str(c)])
    thread24 = json.loads((S1_DIR / "result_seed42_threads24.json").read_text())
    thread8 = json.loads((S1_DIR / "result_seed42_threads8.json").read_text())

    rows = []
    for c in clusters:
        js = np.array(seed_jaccards[c])
        j_mean, j_p05 = float(js.mean()), float(np.quantile(js, 0.05))
        j_thread24 = thread24["jaccard_vs_reference"][str(c)]
        j_thread8 = thread8["jaccard_vs_reference"][str(c)]
        s1_pass = (j_mean >= THRESH["S1"]["j_seed_mean_min"]
                   and j_p05 >= THRESH["S1"]["j_seed_p05_min"]
                   and j_thread24 >= THRESH["S1"]["j_thread24_min"])
        rows.append(dict(cluster=c, s1_j_seed_mean=round(j_mean, 4), s1_j_seed_p05=round(j_p05, 4),
                          s1_j_thread8=round(j_thread8, 4), s1_j_thread24=round(j_thread24, 4),
                          S1_PASS=s1_pass))
    return pd.DataFrame(rows)


def aggregate_s2(clusters) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in sorted(S2_DIR.glob("shard_*.csv"))]
    df = pd.concat(dfs, ignore_index=True)
    n_boot_seen = df["boot"].nunique()
    rows = []
    for c in clusters:
        sub = df[df["cluster"] == c]
        jaccard_mean = float(sub["jaccard"].mean())
        coassign_mean = float(sub["coassign"].mean())
        frac_dissolved = float((sub["outcome"] == "dissolved").mean())
        s2_pass = (jaccard_mean >= THRESH["S2"]["jaccard_mean_min"]
                   and frac_dissolved <= THRESH["S2"]["frac_dissolved_max"]
                   and coassign_mean >= THRESH["S2"]["coassign_mean_min"])
        rows.append(dict(cluster=c, s2_n_boot=len(sub), s2_jaccard_mean=round(jaccard_mean, 4),
                          s2_coassign_mean=round(coassign_mean, 4),
                          s2_frac_dissolved=round(frac_dissolved, 4), S2_PASS=s2_pass))
    return pd.DataFrame(rows), n_boot_seen


def aggregate_transport(clusters, axis_name: str, directions: list, thresh_key: str) -> pd.DataFrame:
    rows = []
    per_cluster_by_direction = {}
    for d in directions:
        obs = json.loads((S4S5_DIR / f"observed_{d}.json").read_text())
        obs_pc = pd.read_csv(S4S5_DIR / f"observed_{d}_percluster.csv").set_index("reference_cluster")
        null_T = {c: [] for c in clusters}
        for rep in range(c22.N_NULL_PER_DIRECTION):
            null_pc = pd.read_csv(S4S5_DIR / f"null_{d}_seed{rep}_percluster.csv").set_index("reference_cluster")
            for c in clusters:
                null_T[c].append(null_pc.loc[c, "T"] if c in null_pc.index else float("nan"))
        per_cluster_by_direction[d] = dict(obs=obs_pc, null_T=null_T, n_source=obs["n_source"], n_target=obs["n_target"])

    for c in clusters:
        direction_pass = {}
        detail = {}
        for d in directions:
            info = per_cluster_by_direction[d]
            t_obs = info["obs"].loc[c, "T"] if c in info["obs"].index else float("nan")
            null_vals = np.array(info["null_T"][c], dtype=float)
            null_median = float(np.nanmedian(null_vals)) if np.isfinite(null_vals).any() else float("nan")
            abs_ok = (not np.isnan(t_obs)) and t_obs >= THRESH[thresh_key]["t_abs_min"]
            rel_ok = (not np.isnan(t_obs)) and (not np.isnan(null_median)) and (
                t_obs >= THRESH[thresh_key]["t_rel_frac_of_null_median_min"] * null_median
            )
            direction_pass[d] = bool(abs_ok and rel_ok)
            detail[f"{axis_name}_{d}_T"] = round(float(t_obs), 4) if not np.isnan(t_obs) else None
            detail[f"{axis_name}_{d}_null_median"] = round(null_median, 4) if not np.isnan(null_median) else None
        overall_pass = all(direction_pass.values())
        row = dict(cluster=c, **detail, **{f"{axis_name}_PASS": overall_pass})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ref = c22.load_reference_frame()
    clusters = sorted(c for c in ref[c22.REFERENCE_CLUSTER_COL].unique() if c != -1)

    repro = load_reproduction_summary()
    s1_df = aggregate_s1(clusters)
    s2_df, s2_n_boot_seen = aggregate_s2(clusters)
    hosp_directions = ["hospital_10_to_20", "hospital_20_to_10"]
    era_directions = ["period_early_to_late", "period_late_to_early"]
    s4_df = aggregate_transport(clusters, "S4", hosp_directions, "S4")
    s5_df = aggregate_transport(clusters, "S5", era_directions, "S5")

    matrix = (s1_df.merge(s2_df, on="cluster").merge(s4_df, on="cluster").merge(s5_df, on="cluster"))
    n_ref = ref[c22.REFERENCE_CLUSTER_COL].value_counts()
    matrix.insert(1, "n", matrix["cluster"].map(n_ref))
    matrix = matrix.sort_values("cluster").reset_index(drop=True)
    c22.atomic_write_csv(matrix, c22.OUT / "survival_matrix_s1s2s4s5.csv")

    summary = dict(
        reproduction=repro,
        s2_n_boot_seen=int(s2_n_boot_seen),
        n_clusters=len(clusters),
        pass_counts={
            "S1": int(matrix["S1_PASS"].sum()),
            "S2": int(matrix["S2_PASS"].sum()),
            "S4": int(matrix["S4_PASS"].sum()),
            "S5": int(matrix["S5_PASS"].sum()),
            "S1_and_S2 (Core, S3 not in Task A scope)": int((matrix["S1_PASS"] & matrix["S2_PASS"]).sum()),
            "S4_and_S5 (Transportable)": int((matrix["S4_PASS"] & matrix["S5_PASS"]).sum()),
            "S4_or_S5 (partially transportable)": int((matrix["S4_PASS"] | matrix["S5_PASS"]).sum()),
            "S1_S2_S4_S5_all_pass": int((matrix["S1_PASS"] & matrix["S2_PASS"] & matrix["S4_PASS"] & matrix["S5_PASS"]).sum()),
        },
        note="S3 (clustering-config sensitivity) and S6 (encoder recoverability) are NOT in Task A's "
             "scope per the task brief (only S1/S2/S4/S5 requested) -- 'Core'/'Survivor'/'Clinical "
             "phenotype candidate' per PRESPEC section 5 cannot be computed from this task alone.",
    )
    (c22.OUT / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print()
    print(matrix[["cluster", "n", "S1_PASS", "S2_PASS", "S4_PASS", "S5_PASS"]].to_string(index=False))
    print(f"\n[aggregate] wrote {c22.OUT}/survival_matrix_s1s2s4s5.csv and SUMMARY.json")


if __name__ == "__main__":
    main()
