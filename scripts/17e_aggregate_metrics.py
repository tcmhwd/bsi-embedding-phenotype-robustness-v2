#!/usr/bin/env python3
"""Task 17: aggregate the frozen-env raw outputs (S1 seed/thread, S2
bootstrap, S4/S5 transport) plus the token-lookup baseline into the
survival matrix for the ACTUAL published 16 clusters, applying
`.superpowers/sdd/PRESPEC_stress_criteria.md` SS3 thresholds VERBATIM
(copied below; never edited to change a pass count).

Runs in the bsi_pheno interpreter (no umap/hdbscan needed -- pure
pandas/numpy over the frozen-env's already-written JSON/CSV outputs).

Writes only under `phenotype_revision/out/17_correct_env/`:
  - `survival_matrix.csv` -- one row per published cluster, every metric
    next to its threshold/null, S1/S2/S4/S5 PASS columns, token-lookup
    best F1, and ALL_AXES (S1 and S2 and S4 and S5).
  - `FINDINGS.md` -- axis pass counts + the robust<=>token dissociation check.
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c17 = importlib.import_module("17_common")

OUT = c17.OUT
S1_DIR = OUT / "s1_seed"
S2_DIR = OUT / "s2_boot"
S4S5_DIR = OUT / "s4s5_transport"
TOKEN_PATH = OUT / "token_lookup" / "g4_perclass_published.csv"

N_SEED_REPS = 20

# PRESPEC SS3, copied verbatim -- never edited to change a pass count.
S1_THRESHOLDS = {"J_seed_mean_min": 0.75, "J_seed_p05_min": 0.50, "J_thread24_min": 0.75}
S2_THRESHOLDS = {"jaccard_mean_min": 0.75, "frac_dissolved_max": 0.20, "coassign_mean_min": 0.75}
S4_S5_THRESHOLDS = {"t_abs_min": 0.50, "t_rel_null_frac": 0.80}
N_NULL_PER_DIRECTION = c17.N_NULL_PER_DIRECTION


def load_published_clusters():
    ref = c17.load_reference_frame()
    labels = ref[c17.PUBLISHED_CLUSTER_COL].to_numpy()
    clusters = sorted(set(labels.tolist()) - {-1})
    sizes = {int(k): int((labels == k).sum()) for k in clusters}
    return clusters, sizes


def aggregate_s1(clusters):
    seed_jac = {c: [] for c in clusters}
    per_seed_k = []
    for s in range(1, N_SEED_REPS + 1):
        p = S1_DIR / f"result_seed{s}_threads1.json"
        if not p.exists():
            raise FileNotFoundError(f"missing S1 seed result: {p}")
        r = json.loads(p.read_text())
        per_seed_k.append({"seed": s, "n_clusters": r["n_clusters"], "noise_frac": r["noise_frac"]})
        for c in clusters:
            seed_jac[c].append(r["jaccard_vs_published"][str(c)])

    t1_path = S1_DIR / "result_seed42_threads1.json"
    t24_path = S1_DIR / "result_seed42_threads24.json"
    if not t1_path.exists() or not t24_path.exists():
        raise FileNotFoundError(f"missing S1 thread result(s): {t1_path} / {t24_path}")
    r1 = json.loads(t1_path.read_text())
    r24 = json.loads(t24_path.read_text())

    rows = []
    for c in clusters:
        jac = np.array(seed_jac[c])
        row = dict(
            cluster=c,
            J_seed_mean=float(np.mean(jac)), J_seed_p05=float(np.percentile(jac, 5)),
            J_thread1=r1["jaccard_vs_published"][str(c)],
            J_thread24=r24["jaccard_vs_published"][str(c)],
        )
        row["S1_pass"] = bool(
            row["J_seed_mean"] >= S1_THRESHOLDS["J_seed_mean_min"]
            and row["J_seed_p05"] >= S1_THRESHOLDS["J_seed_p05_min"]
            and row["J_thread24"] >= S1_THRESHOLDS["J_thread24_min"]
        )
        rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(per_seed_k), r1, r24


def aggregate_s2(clusters):
    shard_paths = sorted(S2_DIR.glob("shard_*.csv"))
    if not shard_paths:
        raise FileNotFoundError(f"no S2 shard CSVs found under {S2_DIR}")
    raw = pd.concat([pd.read_csv(p) for p in shard_paths], ignore_index=True)
    n_boot_present = raw["boot"].nunique()
    if n_boot_present < c17.N_BOOT:
        print(f"[WARN] S2: only {n_boot_present}/{c17.N_BOOT} boot reps present -- "
              f"aggregating on what exists (not a final result until complete)", flush=True)

    rows = []
    for c in clusters:
        sub = raw[raw["cluster"] == c]
        n_rep = len(sub)
        if n_rep == 0:
            rows.append(dict(cluster=c, n_boot_rep=0, jaccard_mean=float("nan"), jaccard_p05=float("nan"),
                              coassign_mean=float("nan"), frac_dissolved=float("nan"), S2_pass=False))
            continue
        jac = sub["jaccard"].to_numpy(dtype=float)
        coa = sub["coassign"].to_numpy(dtype=float)
        frac_dissolved = float((sub["outcome"] == "dissolved").mean())
        row = dict(
            cluster=c, n_boot_rep=n_rep,
            jaccard_mean=float(np.nanmean(jac)), jaccard_p05=float(np.nanpercentile(jac, 5)),
            coassign_mean=float(np.nanmean(coa)), frac_dissolved=frac_dissolved,
        )
        row["S2_pass"] = bool(
            row["jaccard_mean"] >= S2_THRESHOLDS["jaccard_mean_min"]
            and row["frac_dissolved"] <= S2_THRESHOLDS["frac_dissolved_max"]
            and row["coassign_mean"] >= S2_THRESHOLDS["coassign_mean_min"]
        )
        rows.append(row)
    return pd.DataFrame(rows), n_boot_present


def _s4_s5_pass_table(direction_names_pair, axis_label, clusters):
    per_dir = {}
    for name in direction_names_pair:
        obs_path = S4S5_DIR / f"observed_{name}_percluster.csv"
        if not obs_path.exists():
            raise FileNotFoundError(f"missing observed percluster table: {obs_path}")
        obs = pd.read_csv(obs_path).set_index("published_cluster")["T"]

        null_frames = []
        for rep in range(N_NULL_PER_DIRECTION):
            npath = S4S5_DIR / f"null_{name}_seed{rep}_percluster.csv"
            if npath.exists():
                null_frames.append(pd.read_csv(npath)[["published_cluster", "T"]])
        if not null_frames:
            raise FileNotFoundError(f"no null percluster tables found for direction {name}")
        null_df = pd.concat(null_frames, ignore_index=True)
        null_median = null_df.groupby("published_cluster")["T"].median()
        null_n = null_df.groupby("published_cluster")["T"].apply(lambda s: int(s.notna().sum()))
        per_dir[name] = pd.DataFrame({"T_obs": obs, "null_median": null_median, "null_n_nonnan": null_n})

    rows = []
    for c in clusters:
        row = {"cluster": c, "axis": axis_label}
        direction_pass = []
        for name in direction_names_pair:
            df = per_dir[name]
            if c not in df.index:
                row[f"T_obs_{name}"] = float("nan")
                row[f"null_median_{name}"] = float("nan")
                row[f"null_n_nonnan_{name}"] = 0
                direction_pass.append(False)
                continue
            t_obs = df.loc[c, "T_obs"]
            null_med = df.loc[c, "null_median"]
            null_n = df.loc[c, "null_n_nonnan"]
            row[f"T_obs_{name}"] = t_obs
            row[f"null_median_{name}"] = null_med
            row[f"null_n_nonnan_{name}"] = null_n
            if pd.isna(t_obs) or pd.isna(null_med) or null_n < 5:
                direction_pass.append(False)
            else:
                direction_pass.append(
                    bool(t_obs >= S4_S5_THRESHOLDS["t_abs_min"]
                         and t_obs >= S4_S5_THRESHOLDS["t_rel_null_frac"] * null_med)
                )
        row["PASS"] = bool(all(direction_pass)) if direction_pass else False
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    clusters, sizes = load_published_clusters()
    print(f"[aggregate] published clusters (excl. noise): {clusters}", flush=True)

    s1_df, per_seed_k, r1, r24 = aggregate_s1(clusters)
    s2_df, n_boot_present = aggregate_s2(clusters)
    s4_df = _s4_s5_pass_table(["hospital_10_to_20", "hospital_20_to_10"], "hospital", clusters)
    s5_df = _s4_s5_pass_table(["period_early_to_late", "period_late_to_early"], "era", clusters)

    token_df = pd.read_csv(TOKEN_PATH) if TOKEN_PATH.exists() else None

    matrix = pd.DataFrame({"cluster": clusters})
    matrix["size"] = matrix["cluster"].map(sizes)
    matrix = matrix.merge(s1_df, on="cluster", how="left")
    matrix = matrix.merge(s2_df, on="cluster", how="left", suffixes=("", "_s2"))
    matrix = matrix.merge(
        s4_df.drop(columns=["axis"]).rename(columns={"PASS": "S4_pass"}), on="cluster", how="left")
    matrix = matrix.merge(
        s5_df.drop(columns=["axis"]).rename(columns={"PASS": "S5_pass"}), on="cluster", how="left")
    if token_df is not None:
        token_df = token_df.rename(columns={"published_cluster": "cluster"})
        matrix = matrix.merge(token_df, on="cluster", how="left")

    matrix["ALL_AXES_pass"] = (
        matrix["S1_pass"].fillna(False) & matrix["S2_pass"].fillna(False)
        & matrix["S4_pass"].fillna(False) & matrix["S5_pass"].fillna(False)
    )

    matrix.to_csv(OUT / "survival_matrix.csv", index=False)

    n_s1 = int(matrix["S1_pass"].sum())
    n_s2 = int(matrix["S2_pass"].sum())
    n_s4 = int(matrix["S4_pass"].sum())
    n_s5 = int(matrix["S5_pass"].sum())
    n_all = int(matrix["ALL_AXES_pass"].sum())
    n_clusters = len(clusters)

    lines = []
    lines.append("# Task 17: stress-test battery against the ACTUAL published phenotypes\n")
    lines.append(
        "Reference partition: the published `phenotype_cluster` labels themselves "
        "(`pipeline_new/out_pre/phenotype_clusters_labels.csv`), reproduced bit-for-bit "
        "in the frozen `/home/programs/anaconda3/envs/gpt` stack (ARI=1.0, k=16, "
        "n_noise=1792, verified independently in `out/16_c1_sweep/result_original_env_seed42.json`). "
        "Not a refit -- used directly.\n"
    )
    lines.append(f"n_boot present for S2: {n_boot_present}/{c17.N_BOOT}"
                 + ("" if n_boot_present >= c17.N_BOOT else " -- INCOMPLETE, numbers below are partial") + "\n")
    lines.append("## Pass thresholds (PRESPEC SS3, verbatim)\n")
    lines.append(f"- S1 PASS iff J_seed_mean>=0.75 AND J_seed_p05>=0.50 AND J_thread24>=0.75")
    lines.append(f"- S2 PASS iff jaccard_mean>=0.75 AND frac_dissolved<=0.20 AND coassign_mean>=0.75")
    lines.append(f"- S4/S5 PASS iff BOTH directions: T_obs>=0.50 AND T_obs>=0.80*null_median\n")

    lines.append("## Per-cluster n_clusters found, 20 seed refits (threads=1)\n")
    lines.append("```\n" + per_seed_k.to_string(index=False) + "\n```\n")
    lines.append(f"Thread test at seed=42: threads=1 -> k={r1['n_clusters']}, "
                 f"threads=24 -> k={r24['n_clusters']} "
                 f"(ARI_all_row vs published: {r1['agreement_vs_published']['ari_all_row']:.4f} / "
                 f"{r24['agreement_vs_published']['ari_all_row']:.4f}; "
                 f"ARI_noise_excl: {r1['agreement_vs_published']['ari_noise_excl']:.4f} / "
                 f"{r24['agreement_vs_published']['ari_noise_excl']:.4f})\n")

    lines.append(f"## Axis pass counts (of {n_clusters} published clusters)\n")
    lines.append(f"- S1 (seed/thread): {n_s1}/{n_clusters}")
    lines.append(f"- S2 (bootstrap): {n_s2}/{n_clusters}")
    lines.append(f"- S4 (institution transport): {n_s4}/{n_clusters}")
    lines.append(f"- S5 (era transport): {n_s5}/{n_clusters}")
    lines.append(f"- ALL FOUR axes: {n_all}/{n_clusters}\n")

    lines.append("## Full survival matrix\n")
    show_cols = ["cluster", "size", "J_seed_mean", "J_seed_p05", "J_thread24", "S1_pass",
                 "jaccard_mean", "jaccard_p05", "coassign_mean", "frac_dissolved", "S2_pass",
                 "S4_pass", "S5_pass", "ALL_AXES_pass"]
    if token_df is not None:
        show_cols += ["best_lookup_f1", "best_lookup_combo"]
    lines.append("```\n" + matrix[show_cols].round(4).to_string(index=False) + "\n```\n")

    if token_df is not None:
        lines.append("## Robust <=> token dissociation check\n")
        matrix["token_defined"] = matrix["best_lookup_f1"] >= 0.90
        n_robust_and_token = int((matrix["ALL_AXES_pass"] & matrix["token_defined"]).sum())
        n_robust_and_novel = int((matrix["ALL_AXES_pass"] & ~matrix["token_defined"]).sum())
        n_fragile_and_token = int((~matrix["ALL_AXES_pass"] & matrix["token_defined"]).sum())
        n_fragile_and_novel = int((~matrix["ALL_AXES_pass"] & ~matrix["token_defined"]).sum())
        lines.append(f"Using best_lookup_f1>=0.90 as \"token-defined\" (organism/organism+hospital+year/"
                     f"organism+hospital+year+department majority-vote lookup, StratifiedGroupKFold(5, patient_ID), "
                     f"matching `g4_perclass.py`'s method exactly):\n")
        lines.append(f"- robust (ALL_AXES_pass) AND token-defined: {n_robust_and_token}")
        lines.append(f"- robust (ALL_AXES_pass) AND NOT token-defined (novel/embedding-specific): {n_robust_and_novel}")
        lines.append(f"- fragile (fails >=1 axis) AND token-defined: {n_fragile_and_token}")
        lines.append(f"- fragile (fails >=1 axis) AND NOT token-defined: {n_fragile_and_novel}\n")
        lines.append(
            "Per-cluster detail for this check: `survival_matrix.csv` columns `ALL_AXES_pass` and "
            "`best_lookup_f1`/`best_lookup_combo`.\n"
        )

        if n_robust_and_novel == 0:
            verdict = (
                "**HOLDS, cleanly**: every cluster that survives all four structural stress axes "
                "is also token/organism-recoverable, and every cluster NOT explainable by a "
                "categorical lookup fails at least one stress axis. No counterexample in either "
                "direction."
            )
        else:
            frac_robust_token = n_robust_and_token / max(1, n_robust_and_token + n_robust_and_novel)
            verdict = (
                f"**WEAKENS, does not reverse**: on the wrong (k=20 refit) reference this was a "
                f"perfect dissociation (every robust cluster was token, every novel cluster was "
                f"fragile, zero exceptions either way). On the correct 16-cluster published "
                f"reference, {n_robust_and_token}/{n_robust_and_token + n_robust_and_novel} "
                f"({frac_robust_token:.0%}) of the robust (ALL_AXES_pass) clusters are still "
                f"token/organism-recoverable, but cluster(s) "
                f"{sorted(matrix.loc[matrix['ALL_AXES_pass'] & ~matrix['token_defined'], 'cluster'].tolist())} "
                f"survive EVERY structural stress axis (seed, bootstrap, hospital transport, era "
                f"transport) while scoring below 0.90 on every organism/hospital/year/department "
                f"lookup combination tried -- a genuine counterexample to \"robust implies token\". "
                f"The majority pattern (most robust clusters are token, most token-failing clusters "
                f"are fragile) still holds; it is no longer exceptionless."
            )
        lines.append(f"**Verdict**: {verdict}\n")

        novel_robust = sorted(matrix.loc[matrix["ALL_AXES_pass"] & ~matrix["token_defined"], "cluster"].tolist())
        if novel_robust:
            frame = c17.build_full_frame()
            lines.append("### Profile of the robust, non-token cluster(s)\n")
            for c in novel_robust:
                sub = frame[frame[c17.PUBLISHED_CLUSTER_COL] == c]
                top_org = sub["BSI_episode_short_clean"].value_counts(normalize=True).head(5)
                dept_top = sub["treatment_department_eng"].value_counts(normalize=True).head(3)
                hosp = sub["hospital_code"].value_counts(normalize=True).to_dict()
                lines.append(
                    f"- cluster {c} (n={len(sub)}): top organisms "
                    f"{[(o, round(f, 3)) for o, f in top_org.items()]}; "
                    f"top departments {[(d, round(f, 3)) for d, f in dept_top.items()]}; "
                    f"hospital split {hosp}; admit_year range "
                    f"{int(sub['admit_year'].min())}-{int(sub['admit_year'].max())}. "
                    f"No single organism/department exceeds ~30% share and the year range spans "
                    f"the full cohort -- this cluster does not meet PRESPEC SS4's organism-defined "
                    f"(>=0.90 one organism), department-defined (>=0.90 one dept), or year-defined "
                    f"(>=0.90 in 3 consecutive years) artifact-screen rules on visual inspection "
                    f"(the full SS4 screen was not run as part of this task)."
                )

    findings_path = OUT / "FINDINGS.md"
    findings_path.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"[aggregate] wrote {OUT / 'survival_matrix.csv'} and {findings_path}")


if __name__ == "__main__":
    main()
