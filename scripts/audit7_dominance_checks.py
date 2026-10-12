#!/usr/bin/env python3
"""Audit 7 (questions A and B) -- is 'field dominance' the right reading of
the AMI table, and does admission_cluster carry unique information about the
Korean Qwen3 partition?

Reads only stored partitions (out/39_audit6/matched/*.npz, out/27_mimic/
replicate/labels/, out/18_encoder_dec/recluster/) -- nothing is re-clustered.

Blocks:
  1. permutation null of AMI(partition, dominant field), 100 permutations.
  2. directional decomposition: U(C|F) = fraction of partition entropy
     explained by the field (field -> cluster), U(F|C) (cluster -> field),
     and cluster purity in the field.  AMI is symmetric and cannot say which
     way 'dominance' runs.
  3. conditional cross-encoder agreement WITHIN organism strata (n-weighted
     mean of per-stratum AMI/ARI), for every Korean encoder pair on all 50
     seed-matched n-matched fits, and for MIMIC BGE-M3 vs Qwen3 on 10 seeds.
     If the mechanism is 'shared dominant field', agreement should vanish
     inside the field.
  4. antibiotic-block alignment of the Korean Qwen3 partition: fields the
     26-field set in 40_field_dominance.py did not include (admission
     antibiotic name/category, number of antibiotic lines, total duration),
     next to admission_cluster; and AMI(admission_cluster, antibiotic
     category) to show what admission_cluster is a function of.
  5. classifier ablation on the whole-corpus Korean Qwen3 partition: the
     MIMIC-equivalent set (audit-6 kappa 0.325) plus (a) the two cluster IDs
     only, (b) sepsis/colonisation only, (c) the antibiotic block only.
"""
import os

for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

import glob
import importlib.util
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
ROOT = REV.parent
OUT = REV / "out/41_audit7"
MATCHED = REV / "out/39_audit6/matched"
MIMIC_LABELS = REV / "out/27_mimic/replicate/labels"
MIMIC_COHORT = REV / "out/27_mimic/cohort"
DATASET = ROOT / "manuscript/phenotype_ml_dataset.csv"
PUBLISHED = ROOT / "pipeline_new/out_pre/phenotype_clusters_labels.csv"
RAW_FIELDS = REV / "out/20_onehot/features/raw_fields.parquet"
IDS = REV / "out/18_encoder_dec/embeddings/bge-m3/merged_ids.json"
ENCODERS = ("bge-m3", "qwen3-0.6b", "openai-3small", "openai-3large")
RNG = np.random.default_rng(41)


def ami(a, b):
    return float(adjusted_mutual_info_score(a, b))


def entropy(x):
    p = np.asarray(list(Counter(x).values()), float); p /= p.sum()
    return float(-(p * np.log(p)).sum())


def directional(cluster, field):
    """U(C|F), U(F|C), purity of clusters in the field, purity of field levels in clusters."""
    df = pd.DataFrame({"c": cluster, "f": field})
    hc, hf = entropy(df["c"]), entropy(df["f"])
    joint = df.groupby(["c", "f"]).size()
    p = joint / joint.sum()
    hcf = float(-(p * np.log(p)).sum())
    mi = hc + hf - hcf
    purity_c = float((df.groupby("c")["f"].agg(lambda s: s.value_counts().iloc[0])).sum() / len(df))
    purity_f = float((df.groupby("f")["c"].agg(lambda s: s.value_counts().iloc[0])).sum() / len(df))
    return {"U_cluster_given_field": mi / hc if hc else None, "U_field_given_cluster": mi / hf if hf else None,
            "purity_of_clusters_in_field": purity_c, "purity_of_field_levels_in_clusters": purity_f,
            "k": int(df["c"].nunique()), "field_levels": int(df["f"].nunique())}


def perm_null(cluster, field, n=100):
    vals = [ami(cluster, RNG.permutation(field)) for _ in range(n)]
    return {"n_perm": n, "mean": float(np.mean(vals)), "sd": float(np.std(vals)), "max": float(np.max(vals))}


def conditional_agreement(a, b, strata, min_n=50):
    both = (a != -1) & (b != -1)
    a, b, s = a[both], b[both], strata[both]
    rows = []
    for level in pd.unique(s):
        m = s == level
        if m.sum() < min_n:
            continue
        rows.append((int(m.sum()), ami(a[m], b[m]), float(adjusted_rand_score(a[m], b[m]))))
    n = np.array([r[0] for r in rows], float)
    return {"n_strata": len(rows), "n_rows": int(n.sum()),
            "ami_weighted": float((n * np.array([r[1] for r in rows])).sum() / n.sum()),
            "ari_weighted": float((n * np.array([r[2] for r in rows])).sum() / n.sum()),
            "ami_unconditional": ami(a, b), "ari_unconditional": float(adjusted_rand_score(a, b))}


def combine(a, b):
    return np.array([f"{x}#{y}" for x, y in zip(a, b)], dtype=object)


def band(x, q=10):
    v = pd.to_numeric(x, errors="coerce")
    out = pd.qcut(v, q=q, duplicates="drop").astype(str)
    return out.where(v.notna(), "NA").to_numpy()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {}
    ids = [int(i) for i in json.load(open(IDS))]
    ds = pd.read_csv(DATASET, low_memory=False, usecols=[
        "admission_ID", "patient_ID", "admission_cluster", "empiric_cluster", "BSI_episode_short_clean",
        "treatment_department_eng", "ab_admission_name_clean", "ab_admission_category_clean",
        "ab_empiric24_name_clean", "ab_empiric24_category_clean", "n_bsi_episodes", "test_date",
        "hospital_code", "icu", "sex", "sepsis_2", "VRE_colonization", "CRE_colonization",
        "charlson_score", "age_at_test", "BSI_episode_3GCR", "BSI_episode_CarbaR", "BSI_episode_VRE",
        "BSI_episode_MRSA", "BSI_episode_CRAB", "BSI_episode_CeftriaxoneR", "BSI_episode_polymicrobial",
        "antiMRSAab_empiric", "antiVREab_empiric", "antipseudoab_empiric", "carbaab_empiric",
        "outpatient_inpatient_ER", "ventilator"]).drop_duplicates("admission_ID")
    ds = ds.set_index("admission_ID").loc[ids].reset_index()
    raw = pd.read_parquet(RAW_FIELDS).set_index("admission_ID").loc[ids].reset_index()
    ab_lists = raw["tx__admission_antibiotics"].apply(
        lambda v: json.loads(v) if isinstance(v, str) and v.startswith("[") else ([] if v in (None, "NA") or pd.isna(v) else [v]))
    ds["n_antibiotic_lines"] = ab_lists.apply(len).astype(str)
    ds["first_antibiotic"] = ab_lists.apply(lambda l: str(l[0]) if l else "NA")
    ds["antibiotic_set"] = ab_lists.apply(lambda l: "|".join(sorted(set(map(str, l)))) if l else "NA")
    ds["total_ab_duration_band"] = band(raw["tx__total_antibiotic_duration_days"])
    ds["admit_year"] = pd.to_datetime(ds["test_date"], errors="coerce").dt.year.astype(str)
    ds["charlson_band"] = pd.cut(ds["charlson_score"], bins=[-0.5, 0.5, 2.5, 4.5, 100], labels=["0", "1-2", "3-4", "5+"]).astype(str)
    ds["age_band"] = pd.cut(ds["age_at_test"], bins=[0, 40, 50, 60, 70, 80, 200], labels=["<40", "40s", "50s", "60s", "70s", "80+"], right=False).astype(str)
    F = {c: ds[c].astype(str).to_numpy() for c in ds.columns if c not in ("admission_ID", "patient_ID")}
    organism = F["BSI_episode_short_clean"]

    # ------------------------------------------------------------ Korea matched fits
    draws = [np.load(p, allow_pickle=True) for p in sorted(glob.glob(str(MATCHED / "*.npz")))]
    seeds = sorted({int(k.split("|")[1]) for k in draws[0].files if "|" in k})
    dominant = {"openai-3large": "BSI_episode_short_clean", "openai-3small": "treatment_department_eng",
                "qwen3-0.6b": "admission_cluster", "bge-m3": "admission_cluster"}

    # 1+2: null and direction on the first fit of each encoder, plus medians over all 50.
    result["korea_dominance"] = {}
    for enc in ENCODERS:
        fld = dominant[enc]
        per_fit = {"ami": [], "U_cluster_given_field": [], "U_field_given_cluster": [], "purity_c": [], "purity_f": [], "k": []}
        for z in draws:
            sub = z["subset"]
            for s in seeds:
                lab = z[f"{enc}|{s}"]; keep = lab != -1
                c, f = lab[keep], F[fld][sub][keep]
                d = directional(c, f)
                per_fit["ami"].append(ami(c, f)); per_fit["k"].append(d["k"])
                per_fit["U_cluster_given_field"].append(d["U_cluster_given_field"])
                per_fit["U_field_given_cluster"].append(d["U_field_given_cluster"])
                per_fit["purity_c"].append(d["purity_of_clusters_in_field"]); per_fit["purity_f"].append(d["purity_of_field_levels_in_clusters"])
        z = draws[0]; lab = z[f"{enc}|{seeds[0]}"]; keep = lab != -1
        result["korea_dominance"][enc] = {
            "dominant_field": fld, "field_levels": int(len(set(F[fld]))),
            "perm_null_first_fit": perm_null(lab[keep], F[fld][z["subset"]][keep]),
            "median_over_50_fits": {k: float(np.median(v)) for k, v in per_fit.items()},
            "range_over_50_fits": {k: [float(np.min(v)), float(np.max(v))] for k, v in per_fit.items()},
        }
        print(f"[dom] korea/{enc} {fld}: AMI med {np.median(per_fit['ami']):.3f} "
              f"U(C|F) {np.median(per_fit['U_cluster_given_field']):.3f} U(F|C) {np.median(per_fit['U_field_given_cluster']):.3f} "
              f"purity_c {np.median(per_fit['purity_c']):.3f} null max {result['korea_dominance'][enc]['perm_null_first_fit']['max']:.4f}", flush=True)

    # 3: conditional agreement within organism (and within department), Korea pairs, all seed-matched fits
    result["korea_conditional_agreement"] = {}
    for a, b in itertools.combinations(ENCODERS, 2):
        rows = {"organism": [], "department": [], "organism_x_department": []}
        for z in draws:
            sub = z["subset"]
            strata = {"organism": organism[sub], "department": F["treatment_department_eng"][sub],
                      "organism_x_department": combine(organism[sub], F["treatment_department_eng"][sub])}
            for s in seeds:
                la, lb = z[f"{a}|{s}"], z[f"{b}|{s}"]
                for name, st in strata.items():
                    rows[name].append(conditional_agreement(la, lb, st))
        result["korea_conditional_agreement"][f"{a} vs {b}"] = {
            name: {"n_fits": len(v), "ami_within_median": float(np.median([r["ami_weighted"] for r in v])),
                   "ari_within_median": float(np.median([r["ari_weighted"] for r in v])),
                   "ami_unconditional_median": float(np.median([r["ami_unconditional"] for r in v])),
                   "n_strata_median": float(np.median([r["n_strata"] for r in v]))}
            for name, v in rows.items()}
        o = result["korea_conditional_agreement"][f"{a} vs {b}"]["organism"]
        print(f"[cond] korea {a} vs {b}: AMI uncond {o['ami_unconditional_median']:.3f} -> within organism {o['ami_within_median']:.3f} "
              f"(within org x dept {result['korea_conditional_agreement'][f'{a} vs {b}']['organism_x_department']['ami_within_median']:.3f})", flush=True)

    # ------------------------------------------------------------ MIMIC
    mimic = pd.read_parquet(MIMIC_COHORT / "narrative_manifest.parquet")
    com = pd.read_parquet(MIMIC_COHORT / "comorbidities.parquet")[["subject_id", "hadm_id", "source_category"]]
    mimic = mimic.merge(com, on=["subject_id", "hadm_id"], how="left", validate="one_to_one")
    result["mimic_dominance"], result["mimic_conditional_agreement"] = {}, {}
    for enc in ("bge-m3", "qwen3-0.6b"):
        per = {"ami": [], "U_cluster_given_field": [], "U_field_given_cluster": [], "purity_c": [], "purity_f": [], "k": []}
        for seed in range(1, 11):
            t = pd.read_parquet(MIMIC_LABELS / f"{enc}_seed{seed}.parquet")
            m = mimic.merge(t, on="admission_ID", validate="one_to_one")
            m = m[m["cluster"] != -1]
            c, f = m["cluster"].to_numpy(), m["organism_category"].astype(str).to_numpy()
            d = directional(c, f)
            per["ami"].append(ami(c, f)); per["k"].append(d["k"])
            per["U_cluster_given_field"].append(d["U_cluster_given_field"]); per["U_field_given_cluster"].append(d["U_field_given_cluster"])
            per["purity_c"].append(d["purity_of_clusters_in_field"]); per["purity_f"].append(d["purity_of_field_levels_in_clusters"])
            if seed == 1:
                null = perm_null(c, f)
        result["mimic_dominance"][enc] = {"dominant_field": "organism_category", "field_levels": int(mimic["organism_category"].nunique()),
                                          "perm_null_seed1": null,
                                          "median_over_10_seeds": {k: float(np.median(v)) for k, v in per.items()},
                                          "range_over_10_seeds": {k: [float(np.min(v)), float(np.max(v))] for k, v in per.items()}}
        print(f"[dom] mimic/{enc} organism: AMI med {np.median(per['ami']):.3f} U(C|F) {np.median(per['U_cluster_given_field']):.3f} "
              f"U(F|C) {np.median(per['U_field_given_cluster']):.3f} purity_c {np.median(per['purity_c']):.3f}", flush=True)
    rows = {"organism": [], "organism_x_source": []}
    for seed in range(1, 11):
        ta = pd.read_parquet(MIMIC_LABELS / f"bge-m3_seed{seed}.parquet").rename(columns={"cluster": "a"})
        tb = pd.read_parquet(MIMIC_LABELS / f"qwen3-0.6b_seed{seed}.parquet").rename(columns={"cluster": "b"})
        m = mimic.merge(ta, on="admission_ID").merge(tb, on="admission_ID")
        org = m["organism_category"].astype(str).to_numpy()
        src = m["source_category"].astype(str).to_numpy()
        rows["organism"].append(conditional_agreement(m["a"].to_numpy(), m["b"].to_numpy(), org))
        rows["organism_x_source"].append(conditional_agreement(m["a"].to_numpy(), m["b"].to_numpy(), combine(org, src)))
    result["mimic_conditional_agreement"]["bge-m3 vs qwen3-0.6b"] = {
        name: {"n_fits": len(v), "ami_within_median": float(np.median([r["ami_weighted"] for r in v])),
               "ami_within_range": [float(np.min([r["ami_weighted"] for r in v])), float(np.max([r["ami_weighted"] for r in v]))],
               "ari_within_median": float(np.median([r["ari_weighted"] for r in v])),
               "ami_unconditional_median": float(np.median([r["ami_unconditional"] for r in v])),
               "n_strata_median": float(np.median([r["n_strata"] for r in v]))} for name, v in rows.items()}
    o = result["mimic_conditional_agreement"]["bge-m3 vs qwen3-0.6b"]["organism"]
    print(f"[cond] mimic bge vs qwen: AMI uncond {o['ami_unconditional_median']:.3f} -> within organism {o['ami_within_median']:.3f}", flush=True)

    # ------------------------------------------------------------ 4: antibiotic block vs Korean Qwen3
    z18 = np.load(REV / "out/18_encoder_dec/recluster/qwen3-0.6b/labels.npz")
    assert [int(i) for i in z18["admission_ID"]] == ids
    qwen_full = z18["labels"].astype(int)
    keep = qwen_full != -1
    cand = ["admission_cluster", "empiric_cluster", "ab_admission_category_clean", "ab_admission_name_clean",
            "ab_empiric24_category_clean", "ab_empiric24_name_clean", "n_antibiotic_lines", "first_antibiotic",
            "antibiotic_set", "total_ab_duration_band", "treatment_department_eng", "BSI_episode_short_clean"]
    whole = {c: ami(qwen_full[keep], F[c][keep]) for c in cand}
    matched = {c: [] for c in cand}
    matched_all = {e: {c: [] for c in cand} for e in ENCODERS}
    for z in draws:
        sub = z["subset"]
        for s in seeds:
            for e in ENCODERS:
                lab = z[f"{e}|{s}"]; kp = lab != -1
                for c in cand:
                    matched_all[e][c].append(ami(lab[kp], F[c][sub][kp]))
    matched = matched_all["qwen3-0.6b"]
    result["korea_qwen3_antibiotic_block"] = {
        "whole_corpus_seed42_k24": dict(sorted(whole.items(), key=lambda kv: -kv[1])),
        "n_matched_50_fits_median": dict(sorted({c: float(np.median(v)) for c, v in matched.items()}.items(), key=lambda kv: -kv[1])),
        "field_levels": {c: int(len(set(F[c]))) for c in cand},
        "n_matched_50_fits_median_all_encoders": {e: dict(sorted({c: float(np.median(v)) for c, v in d.items()}.items(), key=lambda kv: -kv[1])) for e, d in matched_all.items()},
        "first_antibiotic_purity_whole_corpus": directional(qwen_full[keep], F["first_antibiotic"][keep]),
        "perm_null_whole_corpus_first_antibiotic": perm_null(qwen_full[keep], F["first_antibiotic"][keep], 200),
        "ami_admission_cluster_vs_ab_admission_category": ami(F["admission_cluster"], F["ab_admission_category_clean"]),
        "ami_admission_cluster_vs_antibiotic_set": ami(F["admission_cluster"], F["antibiotic_set"]),
        "ami_empiric_cluster_vs_ab_empiric24_category": ami(F["empiric_cluster"], F["ab_empiric24_category_clean"]),
        "directional_whole_corpus_admission_cluster": directional(qwen_full[keep], F["admission_cluster"][keep]),
        "perm_null_whole_corpus_admission_cluster": perm_null(qwen_full[keep], F["admission_cluster"][keep], 200),
    }
    print("[ab] qwen3 whole corpus:", {k: round(v, 3) for k, v in result["korea_qwen3_antibiotic_block"]["whole_corpus_seed42_k24"].items()}, flush=True)

    (OUT / "dominance_checks.json").write_text(json.dumps(result, indent=2, default=float))
    # ------------------------------------------------------------ 5: classifier ablation (audit-6 machinery)
    spec = importlib.util.spec_from_file_location("a6", REV / "scripts/audit6_token_checks.py")
    a6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a6)
    frame = ds.copy()
    for c in frame.columns:
        if c not in ("admission_ID", "patient_ID"):
            frame[c] = frame[c].astype(str)
    frame["enc_cluster"] = qwen_full
    frame = frame[frame["enc_cluster"] != -1].reset_index(drop=True)
    y = frame["enc_cluster"].to_numpy(); g = frame["patient_ID"].to_numpy()
    P = ["BSI_episode_short_clean", "hospital_code", "admit_year", "treatment_department_eng"]
    MIMIC_EQUIV = P + ["n_bsi_episodes", "icu", "charlson_band", "sex", "age_band", "BSI_episode_3GCR", "BSI_episode_CarbaR",
                       "BSI_episode_VRE", "BSI_episode_MRSA", "BSI_episode_CRAB", "BSI_episode_CeftriaxoneR",
                       "BSI_episode_polymicrobial", "antiMRSAab_empiric", "antiVREab_empiric", "antipseudoab_empiric",
                       "carbaab_empiric", "outpatient_inpatient_ER", "ventilator"]
    AB = ["first_antibiotic", "n_antibiotic_lines", "total_ab_duration_band"]  # the multi-drug strings have >255 levels (HGB limit) and are scored by AMI only
    sets = {
        "mimic_equivalent (audit6 baseline)": MIMIC_EQUIV,
        "+ cluster IDs only": MIMIC_EQUIV + ["admission_cluster", "empiric_cluster"],
        "+ sepsis/colonisation only": MIMIC_EQUIV + ["sepsis_2", "VRE_colonization", "CRE_colonization"],
        "+ antibiotic block only": MIMIC_EQUIV + AB,
        "+ cluster IDs + antibiotic block": MIMIC_EQUIV + ["admission_cluster", "empiric_cluster"] + AB,
        "cluster IDs alone": ["admission_cluster", "empiric_cluster"],
        "antibiotic block alone": AB,
    }
    result["korea_qwen3_classifier_ablation"] = {}
    for name, cols in sets.items():
        blk = a6.classifier_full(frame, cols, y, g, f"qwen3/{name}")
        result["korea_qwen3_classifier_ablation"][name] = {k: blk[k] for k in ("kappa", "macro_f1", "accuracy", "ami_pred_vs_true", "n", "k")}
        (OUT / "dominance_checks.json").write_text(json.dumps(result, indent=2, default=float))

    (OUT / "dominance_checks.json").write_text(json.dumps(result, indent=2, default=float))
    print(f"[audit7] wrote {OUT}/dominance_checks.json")


if __name__ == "__main__":
    main()
