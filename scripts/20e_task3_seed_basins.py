#!/usr/bin/env python3
"""Task 3 -- seed-basin characterization.

Pure aggregation of labels already on disk from Task 17's S1 seed/thread
sweep (`out/17_correct_env/s1_seed/labels_seed{1..20}_threads1.npz`, 20
independent frozen-env refits at threads=1, code-of-record UMAP+HDBSCAN
config) -- no new clustering is run; the stored labels are sufficient
(each npz's `labels` array is row-order-aligned to
`17_common.load_reference_frame()`'s `out_pre` order, the same convention
used throughout Task 17/18). `out/16_c1_sweep/result_original_env_seed42.json`
is used only as the reference-reproduction anchor (ARI=1.0 vs published at
seed 42) already established by Task 16/17 -- not as an extra seed in this
sweep.

For each of the 20 seed labelings: n_clusters, noise_frac, ARI/AMI vs
published (all-row and noise-excluded, explicitly labelled per project
convention). Basin assignment: k==8 -> "k8_basin"; k>=15 -> "k15_basin";
else -> "intermediate" (seeds 8, 13, 19 in this sweep -- reported
separately, not forced into either basin).

Basin composition: for k8_basin seeds, which derived cluster each
published cluster's majority mass lands in (cross-tab), to show WHICH
published clusters merge together in that basin. Pairwise ARI (all-row)
between every pair of the 20 seed labelings, summarized as
within-k8-basin / within-k15-basin / between-basin / involving-intermediate
means.

Spearman rho between published-cluster size and each stability metric
already in `out/17_correct_env/survival_matrix.csv` (J_seed_mean,
J_seed_p05, J_thread24, jaccard_mean, jaccard_p05, coassign_mean,
frac_dissolved, best_lookup_f1).

Outcome-blind (Global Constraints): reads only cluster-label arrays and
`patient_ID`-free stability metrics; no mortality/discharge/death column.

Login-node safe: reads only small npz/csv files already on disk, no
embeddings, no SLURM needed.
"""
import importlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c17 = importlib.import_module("17_common")

OUT_DIR = c17.REV / "out" / "20_onehot"
S1_DIR = c17.OUT / "s1_seed"
SURVIVAL_MATRIX = c17.OUT / "survival_matrix.csv"

SEEDS = list(range(1, 21))


def load_seed_labels() -> dict:
    out = {}
    for s in SEEDS:
        p = S1_DIR / f"labels_seed{s}_threads1.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing -- expected all 20 S1 seed refits on disk")
        out[s] = np.load(p)["labels"]
    return out


def classify_basin(k: int) -> str:
    if k == 8:
        return "k8_basin"
    if k >= 15:
        return "k15_basin"
    return "intermediate"


def main() -> None:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref_frame = c17.load_reference_frame()
    reference = ref_frame[c17.PUBLISHED_CLUSTER_COL].to_numpy()

    seed_labels = load_seed_labels()
    per_seed_rows = []
    for s in SEEDS:
        lab = seed_labels[s]
        if len(lab) != len(reference):
            raise RuntimeError(f"seed {s}: label length {len(lab)} != reference length {len(reference)}")
        k = int(len(set(lab.tolist()) - {-1}))
        noise_frac = float((lab == -1).mean())
        ag = c17.agreement(reference, lab)
        basin = classify_basin(k)
        per_seed_rows.append({"seed": s, "n_clusters": k, "noise_frac": noise_frac, "basin": basin, **ag})
    per_seed = pd.DataFrame(per_seed_rows)
    c17.atomic_write_csv(per_seed, OUT_DIR / "task3_per_seed.csv")
    print(per_seed.round(4).to_string(index=False))

    basin_counts = per_seed["basin"].value_counts().to_dict()
    print(f"\nBasin composition: {basin_counts}")

    k8_seeds = per_seed.loc[per_seed["basin"] == "k8_basin", "seed"].tolist()
    k15_seeds = per_seed.loc[per_seed["basin"] == "k15_basin", "seed"].tolist()
    intermediate_seeds = per_seed.loc[per_seed["basin"] == "intermediate", "seed"].tolist()

    # -------------------------------------------------------------
    # k8-basin composition: for each published cluster, which k8-basin
    # derived cluster(s) it lands in (majority-fraction crosstab), pooled
    # across all k8-basin seeds -- shows WHICH published clusters merge.
    # -------------------------------------------------------------
    pub_clusters = sorted(set(reference.tolist()) - {-1})
    merge_rows = []
    for s in k8_seeds:
        lab = seed_labels[s]
        for c in pub_clusters:
            mask = reference == c
            sub = lab[mask]
            non_noise = sub[sub != -1]
            if len(non_noise) == 0:
                continue
            vals, counts = np.unique(non_noise, return_counts=True)
            top = vals[np.argmax(counts)]
            frac = counts.max() / len(sub)
            merge_rows.append({"seed": s, "published_cluster": int(c), "majority_k8_cluster": int(top),
                                "majority_fraction": float(frac)})
    merge_df = pd.DataFrame(merge_rows)
    c17.atomic_write_csv(merge_df, OUT_DIR / "task3_k8_basin_merge_detail.csv")

    # Pivot: published_cluster x seed -> majority_k8_cluster, to visualize
    # which published clusters consistently co-land in the same derived
    # cluster across k8-basin seeds.
    pivot = merge_df.pivot(index="published_cluster", columns="seed", values="majority_k8_cluster")
    c17.atomic_write_csv(pivot.reset_index(), OUT_DIR / "task3_k8_basin_merge_pivot.csv")
    print("\nk8-basin: published_cluster -> majority derived-cluster id, per k8-basin seed")
    print(pivot.to_string())

    # Grouping summary at a representative k8-basin seed (the first one):
    # which published clusters share a majority derived-cluster id.
    rep_seed = k8_seeds[0]
    rep = merge_df[merge_df["seed"] == rep_seed].set_index("published_cluster")["majority_k8_cluster"]
    groups = {}
    for c, g in rep.items():
        groups.setdefault(int(g), []).append(int(c))
    print(f"\nRepresentative k8-basin seed ({rep_seed}) grouping of published clusters "
          f"by shared majority derived-cluster id:")
    for g, members in sorted(groups.items()):
        print(f"  derived cluster {g}: published clusters {sorted(members)}")

    # -------------------------------------------------------------
    # Pairwise ARI (all-row) among all 20 seed labelings.
    # -------------------------------------------------------------
    pair_rows = []
    for s1, s2 in itertools.combinations(SEEDS, 2):
        l1, l2 = seed_labels[s1], seed_labels[s2]
        ari = float(adjusted_rand_score(l1, l2))
        ami = float(adjusted_mutual_info_score(l1, l2))
        b1 = per_seed.loc[per_seed["seed"] == s1, "basin"].iloc[0]
        b2 = per_seed.loc[per_seed["seed"] == s2, "basin"].iloc[0]
        if b1 == b2 == "k8_basin":
            pair_type = "within_k8_basin"
        elif b1 == b2 == "k15_basin":
            pair_type = "within_k15_basin"
        elif {b1, b2} == {"k8_basin", "k15_basin"}:
            pair_type = "between_basins"
        else:
            pair_type = "involving_intermediate"
        pair_rows.append({"seed_a": s1, "seed_b": s2, "ari": ari, "ami": ami, "pair_type": pair_type})
    pair_df = pd.DataFrame(pair_rows)
    c17.atomic_write_csv(pair_df, OUT_DIR / "task3_pairwise_ari.csv")

    pair_summary = pair_df.groupby("pair_type")["ari"].agg(["mean", "std", "min", "max", "count"])
    print("\nPairwise ARI by pair type:")
    print(pair_summary.round(4).to_string())
    c17.atomic_write_csv(pair_summary.reset_index(), OUT_DIR / "task3_pairwise_ari_summary.csv")

    # -------------------------------------------------------------
    # Is any published cluster stable ACROSS basins? -- i.e. does it have
    # high best-match Jaccard in BOTH a k8-basin seed and a k15-basin seed?
    # Use jaccard_for_cluster per seed per published cluster.
    # -------------------------------------------------------------
    cross_basin_rows = []
    for c in pub_clusters:
        j_k8 = [c17.jaccard_for_cluster(reference, seed_labels[s], c) for s in k8_seeds]
        j_k15 = [c17.jaccard_for_cluster(reference, seed_labels[s], c) for s in k15_seeds]
        cross_basin_rows.append({
            "published_cluster": int(c),
            "jaccard_k8_basin_mean": float(np.mean(j_k8)), "jaccard_k8_basin_min": float(np.min(j_k8)),
            "jaccard_k15_basin_mean": float(np.mean(j_k15)), "jaccard_k15_basin_min": float(np.min(j_k15)),
            "stable_across_both_basins": bool(np.mean(j_k8) >= 0.75 and np.mean(j_k15) >= 0.75),
        })
    cross_basin_df = pd.DataFrame(cross_basin_rows)
    c17.atomic_write_csv(cross_basin_df, OUT_DIR / "task3_cross_basin_stability.csv")
    print("\nPer-published-cluster stability within each basin (mean best-match Jaccard):")
    print(cross_basin_df.round(4).to_string(index=False))
    n_stable_both = int(cross_basin_df["stable_across_both_basins"].sum())
    print(f"\nPublished clusters stable (Jaccard mean >= 0.75) in BOTH basins: {n_stable_both}/16"
          f" -> {sorted(cross_basin_df.loc[cross_basin_df['stable_across_both_basins'], 'published_cluster'].tolist())}")

    # -------------------------------------------------------------
    # Spearman rho: cluster size vs each stability metric in survival_matrix.csv
    # -------------------------------------------------------------
    survival = pd.read_csv(SURVIVAL_MATRIX)
    metric_cols = ["J_seed_mean", "J_seed_p05", "J_thread24", "jaccard_mean", "jaccard_p05",
                    "coassign_mean", "frac_dissolved", "best_lookup_f1"]
    metric_cols = [c for c in metric_cols if c in survival.columns]
    rho_rows = []
    for m in metric_cols:
        sub = survival[["size", m]].dropna()
        rho, p = spearmanr(sub["size"], sub[m])
        rho_rows.append({"metric": m, "spearman_rho": float(rho), "p_value": float(p), "n": len(sub)})
    rho_df = pd.DataFrame(rho_rows)
    c17.atomic_write_csv(rho_df, OUT_DIR / "task3_size_vs_stability_spearman.csv")
    print("\nSpearman rho: published-cluster size vs stability metrics (n=16 clusters):")
    print(rho_df.round(4).to_string(index=False))

    summary = {
        "basin_counts": basin_counts,
        "k8_seeds": k8_seeds, "k15_seeds": k15_seeds, "intermediate_seeds": intermediate_seeds,
        "pairwise_ari_by_type": pair_summary["mean"].to_dict(),
        "n_published_clusters_stable_across_both_basins": n_stable_both,
        "stable_across_both_basins_clusters":
            sorted(cross_basin_df.loc[cross_basin_df["stable_across_both_basins"], "published_cluster"].tolist()),
        "size_vs_stability_spearman": {r["metric"]: r["spearman_rho"] for r in rho_rows},
    }
    c17.atomic_write_json(summary, OUT_DIR / "task3_summary.json")
    print(f"\n[20e] wrote task3_* files under {OUT_DIR}")


if __name__ == "__main__":
    main()
