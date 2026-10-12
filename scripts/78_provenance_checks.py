#!/usr/bin/env python3
"""Provenance checks for two manuscript footnote values that had no stored source file (final audit part A, correction M-6).

  Table 3 footnote 0.693  = AMI between the stored text-embedding-3-small whole-corpus partition
                            (out/18_encoder_dec/recluster/openai-3small/labels.npz) and the rendered field `admin__department`
                            (out/20_onehot/features/raw_fields.parquet), rows non-noise in the partition (label != -1).
                            Same AMI call as scripts/40_field_dominance.py::field_ami (sklearn adjusted_mutual_info_score, default averaging).
  Table 2 footnote 0.728  = share of cluster 0 (published labels, n = 471) with days_from_admission_to_index_bsi >= 3
                            (labels: pipeline_new/out_pre/phenotype_clusters_labels.csv; dataset: manuscript/phenotype_ml_dataset.csv;
                            join, drop_duplicates and the >= comparison exactly as scripts/22f_artifact_screen.py::load_frame).
                            The >= 2 share (0.737, the neighbouring value of the same footnote) is recorded alongside as a cross-check.
  Auxiliary (correction M-4 verification) = median/Q1/Q3 of CRP, Albumin, PLT_Count in the MIMIC-IV E. coli subset
                            (out/27_mimic/cohort/admissions_enriched.parquet, organism_category == 'Ecoli'), the five half-way cells of Table 1,
                            with repr() of the raw float so the rounding direction can be checked.

  Auxiliary (correction S-12 verification) = min/max L2 norm of the stored Analysis-C MIMIC-IV embedding matrices
                            (out/followup_analyses/C_mimic_organism_ablation/embed/<arm>/<encoder>/merged.npy, five arms x two encoders), per encoder and
                            per arm; only min/max scalars are written.

AGGREGATE ONLY: nothing per-patient is printed or written; the JSON holds scalars and counts.
Output: out/followup_analyses/qc/provenance_checks_table3_0693_table2_0728.json
Env: /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 (login node: py_compile only; the run is sbatch, slurm/78_provenance_checks.sh)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

ROOT = Path("/data/tcmhwd/AI/AST_ai")
REV = ROOT / "phenotype_revision"
LABELS_3S = REV / "out/18_encoder_dec/recluster/openai-3small/labels.npz"
RAW_FIELDS = REV / "out/20_onehot/features/raw_fields.parquet"
PUB_LABELS = ROOT / "pipeline_new/out_pre/phenotype_clusters_labels.csv"
DATASET = ROOT / "manuscript/phenotype_ml_dataset.csv"
MIMIC = REV / "out/27_mimic/cohort/admissions_enriched.parquet"
OUT = REV / "out/followup_analyses/qc/provenance_checks_table3_0693_table2_0728.json"


def check_table3():
    z = np.load(LABELS_3S, allow_pickle=False)
    lab = pd.DataFrame({"admission_ID": z["admission_ID"], "label": z["labels"]})
    raw = pd.read_parquet(RAW_FIELDS, columns=["admission_ID", "admin__department"])
    assert lab["admission_ID"].is_unique and raw["admission_ID"].is_unique
    fr = lab.merge(raw, on="admission_ID", how="inner", validate="one_to_one")
    assert len(fr) == len(lab) == len(raw) == 32329, (len(fr), len(lab), len(raw))
    keep = fr["label"].to_numpy() != -1
    ami = float(adjusted_mutual_info_score(fr["label"].to_numpy()[keep], fr["admin__department"].astype(str).to_numpy()[keep]))
    return {
        "value": ami, "rounded_3dp": round(ami, 3), "manuscript_value": 0.693,
        "n_total": int(len(fr)), "n_non_noise": int(keep.sum()), "noise_fraction": float(1 - keep.mean()),
        "n_partition_clusters": int(len(np.unique(fr["label"].to_numpy()[keep]))),
        "n_department_levels": int(fr["admin__department"].nunique(dropna=False)),
    }


def check_table2():
    pub = pd.read_csv(PUB_LABELS, usecols=["admission_ID", "phenotype_cluster"])
    ds = pd.read_csv(DATASET, usecols=["admission_ID", "days_from_admission_to_index_bsi"], low_memory=False).drop_duplicates(subset="admission_ID")
    fr = pub.merge(ds, on="admission_ID", how="left", validate="one_to_one")
    c0 = fr[fr["phenotype_cluster"] == 0]
    d = c0["days_from_admission_to_index_bsi"]
    res = {"n_cluster0": int(len(c0)), "n_days_missing": int(d.isna().sum()), "manuscript_n_cluster0": 471}
    for k in (2, 3):
        share = float((d >= k).mean())
        res[f"share_onset_day_ge_{k}"] = share
        res[f"share_onset_day_ge_{k}_rounded_3dp"] = round(share, 3)
        res[f"n_onset_day_ge_{k}"] = int((d >= k).sum())
    res["manuscript_value_ge3"] = 0.728
    res["manuscript_value_ge2"] = 0.737
    return res


def check_m4_quantiles():
    df = pd.read_parquet(MIMIC, columns=["organism_category", "CRP", "Albumin", "PLT_Count"])
    ec = df[df["organism_category"] == "Ecoli"]
    out = {"n_ecoli": int(len(ec))}
    for col in ("CRP", "Albumin", "PLT_Count"):
        s = pd.to_numeric(ec[col], errors="coerce").dropna()
        q1, med, q3 = (float(x) for x in s.quantile([0.25, 0.5, 0.75]))
        dec = 2 if col == "Albumin" else (0 if col == "PLT_Count" else 1)
        out[col] = {"n_valid": int(len(s)), "q1": repr(q1), "median": repr(med), "q3": repr(q3),
                    "formatted_as_in_table1": f"{med:,.{dec if col != 'Albumin' else 1}f} [{q1:,.{dec if col != 'Albumin' else 1}f}-{q3:,.{dec if col != 'Albumin' else 1}f}]"}
    return out


def check_s12_norms():
    base = REV / "out/followup_analyses/C_mimic_organism_ablation/embed"
    per_arm = {}
    for enc in ("qwen3-0.6b", "bge-m3"):
        for arm_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            f = arm_dir / enc / "merged.npy"
            if not f.exists():
                continue
            x = np.load(f, mmap_mode="r")
            n = np.linalg.norm(np.asarray(x, dtype=np.float64), axis=1)
            per_arm[f"{enc}|{arm_dir.name}"] = {"n_rows": int(x.shape[0]), "dtype": str(x.dtype), "norm_min": float(n.min()), "norm_max": float(n.max())}
    out = {"per_arm_encoder": per_arm}
    for enc in ("qwen3-0.6b", "bge-m3"):
        v = [d for k, d in per_arm.items() if k.startswith(enc + "|")]
        out[f"{enc}_all_arms"] = {"n_arms": len(v), "norm_min": min(d["norm_min"] for d in v), "norm_max": max(d["norm_max"] for d in v)}
    return out


def main():
    t3 = check_table3()
    t2 = check_table2()
    m4 = check_m4_quantiles()
    s12 = check_s12_norms()
    result = {
        "purpose": "provenance of two footnote values (final audit part A, M-6) plus M-4 verification; aggregates only",
        "table3_footnote_0693": t3,
        "table2_footnote_0728": t2,
        "aux_M4_table1_ecoli_halfway_cells": m4,
        "aux_S12_analysisC_embedding_norms": s12,
        "pass": {"table3_0693": abs(t3["value"] - 0.69252) < 5e-5 and t3["rounded_3dp"] == 0.693,
                 "table2_0728": abs(t2["share_onset_day_ge_3"] - 0.7282) < 5e-5 and t2["share_onset_day_ge_3_rounded_3dp"] == 0.728},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")
    print(json.dumps(result, indent=2))
    assert all(result["pass"].values()), result["pass"]


if __name__ == "__main__":
    main()
