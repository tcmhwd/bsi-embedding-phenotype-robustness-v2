#!/usr/bin/env python3
"""Which rendered template field dominates each encoder's geometry, and is
that stable across clustering seeds?

This is the paper's mechanism.  Korean 3-large tracks organism (AMI 0.685),
3-small tracks treatment department (0.693), BGE-M3 tracks episode count
(0.646) and Qwen3 tracks the first antibiotic (0.954); MIMIC's two encoders
both track organism (0.696-0.759) and therefore agree with each other.  Four
Korean encoders, four different fields.  A claim like that rests on the
dominant field being the same in every fit rather than a coincidence of the
seed that happened to be run, so every stored fit is scored here.

Three questions, in order of what they decide:

  A. DOMINANCE STABILITY.  For every stored partition (Korea: 10 subsample
     draws x 5 seeds x 4 encoders, from the audit-6 matched refits; MIMIC:
     10 seeds x 2 encoders), compute AMI against every rendered field and
     record which field wins.  The claim needs the winner to be the same
     field almost every time.

  B. THE FULL FIELD SET, INCLUDING THE PRE-COMPUTED CLUSTER IDS.  The
     published narrative's "[Summary tokens]" block renders
     `admission_cluster_<k>` and `empiric_cluster_<k>` -- cluster
     assignments computed BEFORE this analysis and embedded in the text the
     analysis clusters.  Restricting the field set to organism / department
     / year / episode count cannot see them.  If an encoder's partition
     tracks one of those IDs, that is circularity, and it has to be found
     here rather than by a reviewer.

  C. WHAT THE KOREAN QWEN3 PARTITION ENCODES.  It is the most
     seed-reproducible object in the study (cross-seed ARI 0.991).  The
     first version of this script concluded it matched no single field
     (max AMI 0.171, margin 0.008) -- that conclusion was WRONG, and wrong
     for an instructive reason: the field list was built by matching
     dataset column names, so it never contained `first_antibiotic`, which
     the narrative renders as `Antibiotic_1_name` out of a JSON-encoded raw
     field.  That field has 94 levels and reaches AMI 0.954.  Severity and
     laboratory fields are still banded and included, because "tracks
     severity" remained a live alternative until the antibiotic was found.

Nothing is re-clustered: every partition this reads is already on disk.
"""
import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import glob
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

ROOT = Path("/data/tcmhwd/AI/AST_ai")
REV = ROOT / "phenotype_revision"
MATCHED = REV / "out/39_audit6/matched"
MIMIC_LABELS = REV / "out/27_mimic/replicate/labels"
MIMIC_COHORT = REV / "out/27_mimic/cohort"
PUBLISHED = ROOT / "pipeline_new/out_pre/phenotype_clusters_labels.csv"
RAW_FIELDS = REV / "out/20_onehot/features/raw_fields.parquet"
DATASET = ROOT / "manuscript/phenotype_ml_dataset.csv"
OUT = REV / "out/40_field_dominance"

SEEDS_MIMIC = list(range(1, 11))

# Categorical fields the published Korean narrative renders.  The two
# cluster IDs are listed first because they are the circularity test, not an
# afterthought.
KOREA_CATEGORICAL = [
    # first_antibiotic is derived below from the JSON-encoded raw field, not
    # from a dataset column.  The first version of this script built its
    # field list from dataset column names and therefore MISSED it -- and it
    # turns out to be the field that dominates the Qwen3 partition (AMI
    # 0.954).  A field list assembled by column-name matching will silently
    # omit anything the narrative renders from a nested or encoded source.
    "first_antibiotic", "antibiotic_set", "n_antibiotic_lines",
    "admission_cluster", "empiric_cluster",
    "BSI_episode_short_clean", "treatment_department_eng", "admit_year",
    "n_bsi_episodes", "sex", "entry_mode", "hospital_code", "icu",
    "charlson_category_withcancermerged",
    "index_bsi_3gcr", "index_bsi_mrsa", "index_bsi_vre", "index_bsi_carbar",
    "index_bsi_polymicrobial",
    "antiMRSAab_empiric", "antiVREab_empiric", "antipseudoab_empiric",
    "carbaab_empiric",
    "ventilator", "central_orarterial_line", "urinary_orrenal_catheter",
]

# Continuous fields, banded.  Raw values would make a "field" a near-unique
# identifier and score high for reasons that have nothing to do with the
# geometry; deciles are coarse enough to behave like a rendered category.
KOREA_CONTINUOUS = [
    "sofa_score_2", "quick_sofa_score_2", "age_at_test",
    "WBC_Count", "CRP", "T_Bilirubin", "Creatinine", "Hemoglobin", "Albumin",
]

MIMIC_CATEGORICAL = [
    "organism_category", "care_unit", "admit_period", "n_bsi_episodes",
    "sex", "icu", "charlson_category", "source_category",
    "BSI_3gcr", "BSI_mrsa", "BSI_vre", "BSI_carbar",
    "polymicrobial", "fungemia",
    "antiMRSAab_empiric", "antiVREab_empiric", "antipseudoab_empiric",
    "carbaab_empiric",
]
MIMIC_CONTINUOUS = ["age_at_admission"]


def band(series: pd.Series, n_bands: int = 10) -> pd.Series:
    """Deciles, with NA kept as its own level.

    NA is a level rather than a dropped row because 'this field is missing'
    is itself rendered in the narrative (as `NA`) and can therefore be part
    of what the encoder keys on.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(["NA"] * len(series), index=series.index)
    banded = pd.qcut(numeric, q=n_bands, duplicates="drop").astype(str)
    return banded.where(numeric.notna(), "NA")


def field_ami(labels: np.ndarray, fields: dict) -> dict:
    """AMI between one partition and each field, noise rows dropped.

    Noise is dropped rather than treated as a cluster because HDBSCAN's -1
    is 'unassigned', not a group, and letting it act as a 19th category
    inflates agreement with any field that has a large modal level.
    """
    keep = labels != -1
    y = labels[keep]
    return {
        name: float(adjusted_mutual_info_score(y, values.to_numpy()[keep]))
        for name, values in fields.items()
    }


def summarise_dominance(per_fit: list) -> dict:
    """Across fits: which field wins, how often, and by how much."""
    winners = Counter()
    margins, top_amis = [], []
    for scores in per_fit:
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        winners[ordered[0][0]] += 1
        top_amis.append(ordered[0][1])
        margins.append(ordered[0][1] - (ordered[1][1] if len(ordered) > 1 else 0.0))
    total = len(per_fit)
    field_medians = {
        name: float(np.median([s[name] for s in per_fit])) for name in per_fit[0]
    }
    modal_field, modal_count = winners.most_common(1)[0]
    return {
        "n_fits": total,
        "modal_dominant_field": modal_field,
        "modal_share": float(modal_count / total),
        "winner_counts": dict(winners.most_common()),
        "top_ami_median": float(np.median(top_amis)),
        "top_ami_range": [float(np.min(top_amis)), float(np.max(top_amis))],
        "margin_over_runner_up_median": float(np.median(margins)),
        "per_field_ami_median": dict(
            sorted(field_medians.items(), key=lambda kv: -kv[1])
        ),
    }


def load_korea_fields() -> pd.DataFrame:
    columns = ["admission_ID", "test_date"] + KOREA_CATEGORICAL + KOREA_CONTINUOUS
    columns = [c for c in dict.fromkeys(columns) if c != "admit_year"]
    available = set(pd.read_csv(DATASET, nrows=0, low_memory=False).columns)
    missing = [c for c in columns if c not in available]
    if missing:
        print(f"[warn] absent from dataset, dropped: {missing}", flush=True)
    frame = pd.read_csv(
        DATASET, low_memory=False,
        usecols=[c for c in columns if c in available],
    ).drop_duplicates("admission_ID")
    frame["admit_year"] = pd.to_datetime(frame["test_date"], errors="coerce").dt.year.astype(str)

    # The antibiotic fields are NOT dataset columns: the narrative's
    # "[Treatments during admission]" block renders `Antibiotic_1_name` from
    # a JSON-encoded raw field.  Building the field list from dataset column
    # names alone is exactly how the first version of this analysis missed
    # the field that dominates the Qwen3 partition.
    raw = pd.read_parquet(RAW_FIELDS)[["admission_ID", "tx__admission_antibiotics"]]

    def antibiotic_list(value):
        if isinstance(value, str) and value.startswith("["):
            return json.loads(value)
        if value is None or value == "NA" or (isinstance(value, float) and pd.isna(value)):
            return []
        return [value]

    parsed = raw["tx__admission_antibiotics"].apply(antibiotic_list)
    raw["first_antibiotic"] = parsed.apply(lambda l: str(l[0]) if l else "NA")
    raw["n_antibiotic_lines"] = parsed.apply(len).astype(str)
    # Kept as a contrast rather than a candidate: ~15,000 levels makes it a
    # near-unique identifier, so a high score there would mean nothing.
    # first_antibiotic has 94.
    raw["antibiotic_set"] = parsed.apply(
        lambda l: "|".join(sorted(set(map(str, l)))) if l else "NA")

    frame = frame.merge(
        raw[["admission_ID", "first_antibiotic", "n_antibiotic_lines", "antibiotic_set"]],
        on="admission_ID", how="left", validate="one_to_one",
    )
    for column in ("first_antibiotic", "n_antibiotic_lines", "antibiotic_set"):
        if frame[column].isna().any():
            raise RuntimeError(f"{column}: {frame[column].isna().sum()} rows unmatched")
    return frame


def build_fields(frame: pd.DataFrame, categorical: list, continuous: list) -> dict:
    fields = {}
    for name in categorical:
        if name in frame.columns:
            fields[name] = frame[name].astype(str)
    for name in continuous:
        if name in frame.columns:
            fields[f"{name} (decile)"] = band(frame[name])
    return fields


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"design": __doc__.split("\n")[0], "cohorts": {}}

    # ---------------------------------------------------------------- Korea
    published = pd.read_csv(PUBLISHED, usecols=["admission_ID", "phenotype_cluster"])
    korea = published.merge(load_korea_fields(), on="admission_ID", validate="one_to_one")
    korea_fields = build_fields(korea, KOREA_CATEGORICAL, KOREA_CONTINUOUS)
    print(f"[korea] {len(korea)} rows, {len(korea_fields)} fields", flush=True)

    # The published partition, whole corpus.
    result["cohorts"]["korea_published_wholecorpus"] = {
        "n": int(len(korea)),
        "ami_by_field": dict(sorted(
            field_ami(korea["phenotype_cluster"].to_numpy(), korea_fields).items(),
            key=lambda kv: -kv[1])),
    }

    # The n-matched refits: 10 draws x 5 seeds x 4 encoders, already on disk.
    per_encoder: dict = {}
    for path in sorted(glob.glob(str(MATCHED / "*.npz"))):
        payload = np.load(path, allow_pickle=True)
        subset = payload["subset"]
        sub_fields = {name: values.iloc[subset].reset_index(drop=True)
                      for name, values in korea_fields.items()}
        for key in payload.files:
            if "|" not in key:
                continue
            encoder, seed = key.split("|")
            per_encoder.setdefault(encoder, []).append(
                field_ami(payload[key], sub_fields)
            )

    result["cohorts"]["korea_matched_refits"] = {}
    for encoder, per_fit in sorted(per_encoder.items()):
        block = summarise_dominance(per_fit)
        result["cohorts"]["korea_matched_refits"][encoder] = block
        print(f"[korea/{encoder}] dominant={block['modal_dominant_field']} "
              f"({block['modal_share']:.0%} of {block['n_fits']} fits), "
              f"top AMI {block['top_ami_median']:.3f}", flush=True)

    # ---------------------------------------------------------------- MIMIC
    keys = ["subject_id", "hadm_id"]
    mimic = pd.read_parquet(MIMIC_COHORT / "narrative_manifest.parquet")
    for filename, wanted in [
        ("admissions_enriched.parquet",
         ["age_at_admission", "sex", "polymicrobial", "fungemia"]),
        ("comorbidities.parquet", ["charlson_category", "source_category"]),
        ("ast_detail.parquet", ["BSI_3gcr", "BSI_mrsa", "BSI_vre", "BSI_carbar"]),
        ("empiric_antibiotics.parquet",
         ["antiMRSAab_empiric", "antiVREab_empiric",
          "antipseudoab_empiric", "carbaab_empiric"]),
    ]:
        table = pd.read_parquet(MIMIC_COHORT / filename)
        mimic = mimic.merge(table[keys + wanted], on=keys, how="left",
                            validate="one_to_one")
    mimic_fields = build_fields(mimic, MIMIC_CATEGORICAL, MIMIC_CONTINUOUS)
    print(f"[mimic] {len(mimic)} rows, {len(mimic_fields)} fields", flush=True)

    result["cohorts"]["mimic"] = {}
    for encoder in ("bge-m3", "qwen3-0.6b"):
        per_fit = []
        for seed in SEEDS_MIMIC:
            table = pd.read_parquet(MIMIC_LABELS / f"{encoder}_seed{seed}.parquet")
            merged = mimic.merge(table, on="admission_ID", how="inner",
                                 validate="one_to_one")
            fields = {name: values.loc[merged.index] if len(values) == len(merged)
                      else build_fields(merged, MIMIC_CATEGORICAL, MIMIC_CONTINUOUS)[name]
                      for name, values in
                      build_fields(merged, MIMIC_CATEGORICAL, MIMIC_CONTINUOUS).items()}
            per_fit.append(field_ami(merged["cluster"].to_numpy(), fields))
        block = summarise_dominance(per_fit)
        result["cohorts"]["mimic"][encoder] = block
        print(f"[mimic/{encoder}] dominant={block['modal_dominant_field']} "
              f"({block['modal_share']:.0%} of {block['n_fits']} fits), "
              f"top AMI {block['top_ami_median']:.3f}", flush=True)

    (OUT / "field_dominance.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\n[dominance] wrote {OUT}/field_dominance.json")


if __name__ == "__main__":
    main()
