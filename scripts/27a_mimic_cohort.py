#!/usr/bin/env python3
"""Task M (MIMIC-IV replication), Step 1 -- cohort construction.

Builds the multi-organism bacterial BSI cohort from MIMIC-IV v3.1
(`/data/tcmhwd/AI/Fungemia/mimic4`), one row per ADMISSION (matching the
Korean unit of analysis, Task 1: repeat admissions retained, deduped to one
row per admission, never per patient).

Reuses the logic of `/data/tcmhwd/AI/Fungemia/script/13_3_mimic_bacterial_cohort.py`
(organism/antibiotic name mapping onto a curated ontology, approximate-year
recovery via `patients.csv` anchor_year/anchor_year_group) but does NOT
import it directly -- that script emits an AST-result-level long table for a
different unit of analysis (organism-episode, with an `amb<subject>`
pseudo-episode for ED-only cases). This script needs admission-level
covariates (age, sex, ICU, LOS-adjacent timing), so the ED/null-hadm_id
problem is solved differently: by *recovering* the admission via a charttime
inside the admission's [edregtime|admittime, dischtime] interval (see
`recover_hadm_id`), never by inventing a pseudo-episode.

Cohort definition (multi-organism, PRIMARY object per the task brief --
"a prior probe found 2,192 distinct subjects / 3,945 culture events for E.
coli alone; the multi-organism cohort is the primary object here"):

1. `spec_type_desc == 'BLOOD CULTURE'` EXACTLY (not `.str.contains`, which
   would also pull in 'FLUID RECEIVED IN BLOOD CULTURE BOTTLES', 'BLOOD
   CULTURE ( MYCO/F LYTIC BOTTLE)', 'BLOOD CULTURE (POST-MORTEM)', 'Stem
   Cell - Blood Culture', 'BLOOD CULTURE - NEONATE').
2. `org_name` non-null and not 'CANCELLED' (excludes cancelled-only
   specimens, the bug that inflated an earlier multi-organism count to
   23,119 -- corrected value is measured fresh here, not assumed).
3. Organism classified into the Korean 19-category taxonomy
   (`manuscript/phenotype_ml_dataset.csv`'s `BSI_episode_short_clean` value
   set, read and matched exactly) via `classify_organism()`, OR into a
   separate FUNGAL bucket (Candida/Cryptococcus/other yeasts and molds).
   FUNGAL-only admissions (no bacterial-category isolate anywhere in the
   admission) are EXCLUDED from this bacterial-BSI cohort -- the Korean
   `BSI_episode_short_clean` vocabulary has no fungal category (Fungemia is
   a separate binary flag section in the Korean template, never the primary
   organism label), so an admission whose only positive blood culture is
   fungal has no bacterial index organism to assign and is out of scope
   here. This is a stated cohort-definition choice, not an oversight --
   recorded in COHORT_QC.md.
4. Linked to an admission: already has `hadm_id`, OR recovered via
   `recover_hadm_id` (see below). Specimens that cannot be linked to any
   admission (true ED-discharge / outpatient) are excluded and counted.
5. Age >= 18 at admission is ASSERTED (hosp module is already adult-only)
   -- any violation is logged, not silently dropped.

Unit of analysis / index episode:
- Unit = (subject_id, resolved_hadm_id), one row per admission.
- All positive bacterial specimens in an admission are grouped by exact
  timestamp (`ts`, charttime falling back to chartdate at midnight) into
  "episodes". The INDEX episode is the earliest such group. Ties (same
  timestamp, multiple specimens/organisms) merge: the index organism set is
  the union of mapped categories across the tied specimens.
- `organism_category` = that union's single category if |union| == 1, else
  'Polymicrobial' (matches the Korean convention -- Polymicrobial is its
  own top-level category, not organism-specific).
- `n_bsi_episodes` = number of distinct positive-bacterial-specimen
  timestamp groups in the WHOLE admission (not just the index) -- carried
  through as a covariate because the task brief names it decisive in the
  Korean cohort's token-recoverability step.
- `fungemia` = any fungal-positive specimen in the same admission at or
  before the index timestamp (Korean template convention, DESIGN doc
  section B.1).

ED/null-`hadm_id` recovery (measured, not assumed -- see `COHORT_QC.md`
after running `--stage qc`):
    coalesce(admissions.edregtime, admissions.admittime) <= ts <= admissions.dischtime
`edregtime` (not `admittime`) is the correct left edge: a culture drawn in
the ED precedes the inpatient clock. Falls back to `admittime` when
`edregtime` is null. Multiple-match rows take the earliest `admittime`
(logged, expected rare).

No mortality/discharge/death/LOS column is ever read (Global Constraints,
outcome-blind). `hospital_expire_flag`, `deathtime`, `dischtime` are used
ONLY for interval/window arithmetic (linkage, lab/antibiotic windowing in
`27b_mimic_enrich.py`), never surfaced as a modeled or narrative field, and
`dischtime`/`admittime` never appear as literal dates in the rendered
narrative (Step 2 renders only hospital-day-relative timing).

Run (SBATCH only -- HPC policy: cohort building is not login-node work):
    /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/27a_mimic_cohort.py --stage cohort
    /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/27a_mimic_cohort.py --stage qc
"""
import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

MIMIC = Path("/data/tcmhwd/AI/Fungemia/mimic4")
HOSP = MIMIC / "hosp"
ICU = MIMIC / "icu"
KOREAN_DATASET = Path("/data/tcmhwd/AI/AST_ai/manuscript/phenotype_ml_dataset.csv")
OUT = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision/out/27_mimic/cohort")

# ---------------------------------------------------------------------------
# Organism taxonomy -- matched against the Korean corpus's own
# BSI_episode_short_clean value set (verified live in main(), not hardcoded
# blind): Ecoli, Klebsiella, Staph_coagulasenegative, Polymicrobial,
# Staph_aureus, Streptococcus, Efaecium, Other, Acinetobacter, Enterobacter,
# Pseudomonas_aeruginosa, Efaecalis, Citrobacter, Proteus, Other_enterococci,
# Serratia, Stenotrophomonas, Strepto_pneumoniae, Morganella, Providencia.
# ---------------------------------------------------------------------------
FUNGAL_KEYWORDS = (
    "CANDIDA", "CRYPTOCOCC", "ASPERGILL", "YEAST", "SACCHAROMYC",
    "RHODOTORULA", "RHODATORULA", "TRICHOSPORON", "TRICHODERMA", "FUSARIUM",
    "PAECILOMYCES", "AUREOBASIDIUM", "EXOPHIALA", "SAPROCHAETE", "USTILAGO",
    "SCEDOSPORIUM", "GEOTRICHUM", "MOLD",
)


def classify_organism(name: str) -> str:
    """Map a raw MIMIC `org_name` to a Korean BSI_episode_short_clean
    category, or 'FUNGAL' (excluded from the bacterial index organism,
    tracked only for the Fungemia flag)."""
    n = name.upper()
    if any(k in n for k in FUNGAL_KEYWORDS):
        return "FUNGAL"
    if "STAPH AUREUS COAG +" in n or "STAPHYLOCOCCUS AUREUS" in n:
        return "Staph_aureus"
    if n.startswith("STAPHYLOCOCCUS") or n == "STAPH COAG NEGATIVE" or "COAGULASE NEGATIVE" in n:
        return "Staph_coagulasenegative"
    if n == "ESCHERICHIA COLI":
        return "Ecoli"
    if "KLEBSIELLA" in n:
        return "Klebsiella"
    if "ENTEROBACTER" in n or "PANTOEA" in n or "CRONOBACTER" in n:
        return "Enterobacter"
    if "ENTEROCOCCUS FAECIUM" in n:
        return "Efaecium"
    if "ENTEROCOCCUS FAECALIS" in n:
        return "Efaecalis"
    if "ENTEROCOCCUS" in n:
        return "Other_enterococci"
    if "PSEUDOMONAS AERUGINOSA" in n:
        return "Pseudomonas_aeruginosa"
    if "ACINETOBACTER" in n:
        return "Acinetobacter"
    if "STREPTOCOCCUS PNEUMONIAE" in n:
        return "Strepto_pneumoniae"
    if "STREPTOCOC" in n or "STREPTOCCUS" in n or "STREPTOCOCCI" in n or re.search(r"\bSTREP\b", n):
        return "Streptococcus"
    if "CITROBACTER" in n:
        return "Citrobacter"
    if "PROTEUS" in n:
        return "Proteus"
    if "SERRATIA" in n:
        return "Serratia"
    if "STENOTROPHOMONAS" in n:
        return "Stenotrophomonas"
    if "MORGANELLA" in n:
        return "Morganella"
    if "PROVIDENCIA" in n:
        return "Providencia"
    return "Other"


PERIODS = [(2008, 2013, "2008-2013"), (2014, 2019, "2014-2019"), (2020, 2022, "2020-2022")]


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _atomic_write_json(obj, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, dest)


def _atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)


def load_korean_organism_vocab() -> set:
    df = pd.read_csv(KOREAN_DATASET, usecols=["BSI_episode_short_clean"], low_memory=False)
    return set(df["BSI_episode_short_clean"].dropna().unique().tolist())


def load_admissions() -> pd.DataFrame:
    cols = ["subject_id", "hadm_id", "admittime", "dischtime", "edregtime",
            "admission_type", "admission_location"]
    df = pd.read_csv(HOSP / "admissions.csv", usecols=cols, low_memory=False)
    for c in ("admittime", "dischtime", "edregtime"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def load_patients() -> pd.DataFrame:
    cols = ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"]
    df = pd.read_csv(HOSP / "patients.csv", usecols=cols, low_memory=False)
    df["group_start"] = df["anchor_year_group"].str.slice(0, 4).astype(int)
    return df


def load_positive_blood_cultures() -> tuple:
    """Returns (df, flow) where df has one row per (specimen, organism) with
    columns subject_id, hadm_id (raw, possibly NaN), ts, org_name,
    org_category. `flow` is a dict of the drop counts for the flow diagram."""
    usecols = ["subject_id", "hadm_id", "micro_specimen_id", "chartdate",
               "charttime", "spec_type_desc", "org_name"]
    df = pd.read_csv(HOSP / "microbiologyevents.csv", usecols=usecols, low_memory=False)
    flow = {"microbiologyevents_rows": len(df)}

    exact_bc = df[df["spec_type_desc"] == "BLOOD CULTURE"]
    flow["blood_culture_exact_rows"] = len(exact_bc)
    flow["blood_culture_exact_specimens"] = int(exact_bc["micro_specimen_id"].nunique())

    pos = exact_bc[exact_bc["org_name"].notna() & (exact_bc["org_name"] != "CANCELLED")].copy()
    flow["positive_rows"] = len(pos)
    flow["positive_specimens"] = int(pos["micro_specimen_id"].nunique())

    pos["ts"] = pd.to_datetime(pos["charttime"], errors="coerce")
    fallback = pd.to_datetime(pos["chartdate"], errors="coerce")
    n_charttime_missing = int(pos["ts"].isna().sum())
    pos["ts"] = pos["ts"].fillna(fallback)
    flow["charttime_missing_used_chartdate_fallback"] = n_charttime_missing
    n_ts_missing = int(pos["ts"].isna().sum())
    pos = pos.dropna(subset=["ts"])
    flow["dropped_no_usable_timestamp"] = n_ts_missing

    pos["org_category"] = pos["org_name"].map(classify_organism)
    flow["fungal_only_rows_before_admission_grouping"] = int((pos["org_category"] == "FUNGAL").sum())

    return pos.reset_index(drop=True), flow


def recover_hadm_id(pos: pd.DataFrame, adm: pd.DataFrame, left_edge: str = "edregtime") -> tuple:
    """For rows with a null `hadm_id`, find the admission of the same
    subject_id whose [left, dischtime] interval contains `ts`.

    `left_edge` in {'edregtime', 'admittime', 'admittime_minus_24h'} selects
    the left-edge rule (A.4 linkage sensitivity requires all three).
    Returns (resolved_hadm_id array aligned to `pos.index`, n_multi_match).
    """
    has_hadm = pos["hadm_id"].notna()
    resolved = pos["hadm_id"].copy()

    null_rows = pos.loc[~has_hadm, ["subject_id", "ts"]].reset_index()
    if null_rows.empty:
        return resolved, 0

    cand = adm[["subject_id", "hadm_id", "admittime", "dischtime", "edregtime"]].copy()
    if left_edge == "edregtime":
        cand["left"] = cand["edregtime"].fillna(cand["admittime"])
    elif left_edge == "admittime":
        cand["left"] = cand["admittime"]
    elif left_edge == "admittime_minus_24h":
        cand["left"] = cand["admittime"] - pd.Timedelta(hours=24)
    else:
        raise ValueError(left_edge)

    merged = null_rows.merge(cand, on="subject_id", how="left")
    inside = merged[(merged["ts"] >= merged["left"]) & (merged["ts"] <= merged["dischtime"])]
    match_counts = inside.groupby("index").size()
    n_multi_match = int((match_counts > 1).sum())

    inside = inside.sort_values(["index", "admittime"])
    best = inside.drop_duplicates(subset="index", keep="first").set_index("index")["hadm_id"]

    resolved.loc[best.index] = best.values
    return resolved, n_multi_match


def stage_cohort() -> None:
    t0 = time.time()
    korean_vocab = load_korean_organism_vocab()
    print(f"[27a] Korean organism vocab ({len(korean_vocab)}): {sorted(korean_vocab)}", flush=True)

    pos, flow = load_positive_blood_cultures()
    print(f"[27a] positive bacterial+fungal rows after spec/org filter: {len(pos)}", flush=True)

    unmapped_bacterial = sorted(
        set(pos.loc[pos["org_category"] == "Other", "org_name"].str.upper().unique())
    )
    (OUT / "unmapped_to_other_org_names.json").write_text(json.dumps(unmapped_bacterial, indent=2))

    adm = load_admissions()
    pats = load_patients()

    has_hadm = pos["hadm_id"].notna()
    flow["rows_with_hadm_id_present"] = int(has_hadm.sum())
    flow["rows_with_hadm_id_null"] = int((~has_hadm).sum())

    resolved, n_multi = recover_hadm_id(pos, adm, left_edge="edregtime")
    pos["resolved_hadm_id"] = resolved
    n_recovered = int((~has_hadm & pos["resolved_hadm_id"].notna()).sum())
    n_unmatched = int(pos["resolved_hadm_id"].isna().sum())
    flow["null_hadm_recovered_via_edregtime_window"] = n_recovered
    flow["null_hadm_unmatched_excluded"] = n_unmatched
    flow["null_hadm_multi_admission_matches"] = n_multi

    pos = pos.dropna(subset=["resolved_hadm_id"]).copy()
    pos["resolved_hadm_id"] = pos["resolved_hadm_id"].astype("int64")
    flow["rows_linked_to_an_admission"] = len(pos)

    # Reverse check (A.4 item 3): rows that already had hadm_id must fall
    # inside their OWN admission's interval.
    adm_idx = adm.set_index(["subject_id", "hadm_id"])
    check = pos.loc[has_hadm.reindex(pos.index, fill_value=False)].merge(
        adm[["subject_id", "hadm_id", "admittime", "dischtime"]],
        left_on=["subject_id", "resolved_hadm_id"], right_on=["subject_id", "hadm_id"], how="left",
    )
    outside = check[(check["ts"] < check["admittime"]) | (check["ts"] > check["dischtime"])]
    flow["reverse_check_rows_checked"] = len(check)
    flow["reverse_check_failures_outside_own_admission_interval"] = len(outside)

    # ------------------------------------------------------------------
    # Build episodes table: group positive rows by (subject_id,
    # resolved_hadm_id, ts) -- one row per episode, organism set = union of
    # org_category at that ts.
    # ------------------------------------------------------------------
    ep = (
        pos.groupby(["subject_id", "resolved_hadm_id", "ts"])["org_category"]
        .agg(lambda s: sorted(set(s)))
        .reset_index()
    )
    ep = ep.rename(columns={"resolved_hadm_id": "hadm_id"})
    ep["n_distinct_categories"] = ep["org_category"].apply(len)
    ep["is_fungal_only"] = ep["org_category"].apply(lambda cats: set(cats) == {"FUNGAL"})
    ep["bacterial_categories"] = ep["org_category"].apply(lambda cats: [c for c in cats if c != "FUNGAL"])
    ep["has_bacterial"] = ep["bacterial_categories"].apply(len) > 0

    _atomic_write_parquet(ep, OUT / "episodes_raw.parquet")

    # Cohort = admissions with >=1 bacterial-positive episode.
    bact_ep = ep[ep["has_bacterial"]].copy()
    n_admissions_bacterial = bact_ep.drop_duplicates(["subject_id", "hadm_id"]).shape[0]
    n_admissions_fungal_only_excluded = (
        ep.groupby(["subject_id", "hadm_id"])["has_bacterial"].any()
        .pipe(lambda s: int((~s).sum()))
    )
    flow["admissions_with_bacterial_positive_episode"] = int(n_admissions_bacterial)
    flow["admissions_fungal_only_excluded"] = int(n_admissions_fungal_only_excluded)

    bact_ep = bact_ep.sort_values(["subject_id", "hadm_id", "ts"])
    idx_ep = bact_ep.groupby(["subject_id", "hadm_id"], as_index=False).first()
    idx_ep = idx_ep.rename(columns={"ts": "index_ts", "bacterial_categories": "index_categories"})

    n_episodes_per_adm = bact_ep.groupby(["subject_id", "hadm_id"]).size().rename("n_bsi_episodes")

    # Fungemia: any fungal episode at or before the index ts, same admission.
    fungal_ep = ep[ep["org_category"].apply(lambda cats: "FUNGAL" in cats)]
    fungal_ep = fungal_ep.merge(
        idx_ep[["subject_id", "hadm_id", "index_ts"]], on=["subject_id", "hadm_id"], how="inner",
    )
    fungemia_adm = set(
        map(tuple, fungal_ep.loc[fungal_ep["ts"] <= fungal_ep["index_ts"], ["subject_id", "hadm_id"]].to_numpy())
    )

    cohort = idx_ep.merge(n_episodes_per_adm, on=["subject_id", "hadm_id"], how="left")
    cohort["organism_category"] = cohort["index_categories"].apply(
        lambda cats: "Polymicrobial" if len(cats) > 1 else cats[0]
    )
    cohort["polymicrobial"] = cohort["index_categories"].apply(lambda cats: "yes" if len(cats) > 1 else "no")
    cohort["fungemia"] = cohort.apply(
        lambda r: "yes" if (r["subject_id"], r["hadm_id"]) in fungemia_adm else "no", axis=1
    )

    unknown_cats = set(cohort["organism_category"].unique()) - korean_vocab - {"Polymicrobial"}
    if unknown_cats:
        print(f"[27a] WARNING: organism categories not in Korean vocab: {unknown_cats}", flush=True)

    # ------------------------------------------------------------------
    # Admission-level covariates.
    # ------------------------------------------------------------------
    cohort = cohort.merge(
        adm[["subject_id", "hadm_id", "admittime", "dischtime", "edregtime", "admission_type"]],
        on=["subject_id", "hadm_id"], how="left",
    )
    cohort = cohort.merge(pats, on="subject_id", how="left")

    n_no_patient = int(cohort["anchor_age"].isna().sum())
    flow["admissions_dropped_no_patient_record"] = n_no_patient
    cohort = cohort.dropna(subset=["anchor_age"])

    cohort["age_at_admission"] = (
        cohort["anchor_age"] + (cohort["admittime"].dt.year - cohort["anchor_year"])
    ).round().astype("Int64")
    n_under18 = int((cohort["age_at_admission"] < 18).sum())
    flow["age_under_18_at_admission"] = n_under18  # asserted-expected-zero, logged not silently dropped
    if n_under18 > 0:
        print(f"[27a] WARNING: {n_under18} admissions with age_at_admission < 18 (hosp is adult-only; investigate)", flush=True)

    cohort["sex"] = cohort["gender"]
    cohort["entry_mode"] = np.where(cohort["edregtime"].notna(), "ED", "Direct")

    cohort["admit_year_approx"] = (
        cohort["group_start"] + (cohort["admittime"].dt.year - cohort["anchor_year"])
    ).astype("Int64")
    cohort["admit_period"] = None
    for lo, hi, label in PERIODS:
        m = cohort["admit_year_approx"].between(lo, hi)
        cohort.loc[m, "admit_period"] = label

    # ICU: any icustay overlapping the admission.
    icu = pd.read_csv(ICU / "icustays.csv", usecols=["subject_id", "hadm_id", "stay_id", "first_careunit", "intime"])
    icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
    icu_first = icu.sort_values("intime").drop_duplicates(["subject_id", "hadm_id"], keep="first")
    cohort = cohort.merge(
        icu_first[["subject_id", "hadm_id", "first_careunit"]].rename(columns={"first_careunit": "care_unit"}),
        on=["subject_id", "hadm_id"], how="left",
    )
    cohort["icu"] = np.where(cohort["care_unit"].notna(), "yes", "no")

    keep_cols = [
        "subject_id", "hadm_id", "index_ts", "admittime", "dischtime",
        "age_at_admission", "sex", "entry_mode", "admission_type",
        "icu", "care_unit", "admit_year_approx", "admit_period",
        "organism_category", "index_categories", "polymicrobial", "fungemia",
        "n_bsi_episodes",
    ]
    cohort_out = cohort[keep_cols].reset_index(drop=True)
    _atomic_write_parquet(cohort_out, OUT / "admissions.parquet")

    flow["final_admissions_n"] = len(cohort_out)
    flow["final_subjects_n"] = int(cohort_out["subject_id"].nunique())
    flow["organism_distribution"] = cohort_out["organism_category"].value_counts().to_dict()
    flow["fungemia_yes_n"] = int((cohort_out["fungemia"] == "yes").sum())
    flow["icu_yes_n"] = int((cohort_out["icu"] == "yes").sum())
    flow["icu_yes_frac"] = float((cohort_out["icu"] == "yes").mean())
    flow["n_bsi_episodes_distribution"] = cohort_out["n_bsi_episodes"].value_counts().sort_index().to_dict()
    flow["elapsed_sec"] = round(time.time() - t0, 1)

    _atomic_write_json(flow, OUT / "cohort_flow.json")

    manifest = {
        "stage": "cohort", "python": sys.executable, "sys_prefix": sys.prefix,
        "pandas_version": pd.__version__, "numpy_version": np.__version__,
        "elapsed_sec": flow["elapsed_sec"],
    }
    _atomic_write_json(manifest, OUT / "27a_manifest.json")

    print(json.dumps({k: v for k, v in flow.items() if not isinstance(v, dict)}, indent=2), flush=True)
    print("organism_distribution:", flow["organism_distribution"], flush=True)
    print(f"[27a] wrote {OUT}", flush=True)


def stage_qc() -> None:
    """A.4 verification: linkage sensitivity under 3 left-edge rules +
    duplicate-match audit + external plausibility. Reuses stage_cohort's
    loaders but does not rebuild the full cohort three times -- only the
    linkage step, which is what the sensitivity check is about."""
    pos, flow0 = load_positive_blood_cultures()
    adm = load_admissions()

    results = {}
    for rule in ("edregtime", "admittime", "admittime_minus_24h"):
        resolved, n_multi = recover_hadm_id(pos, adm, left_edge=rule)
        has_hadm = pos["hadm_id"].notna()
        n_recovered = int((~has_hadm & resolved.notna()).sum())
        n_total_linked = int(resolved.notna().sum())
        results[rule] = {
            "n_recovered_from_null": n_recovered,
            "n_total_linked_specimens": n_total_linked,
            "n_multi_match": n_multi,
        }
        print(f"[27a/qc] left_edge={rule}: {results[rule]}", flush=True)

    base = results["edregtime"]["n_total_linked_specimens"]
    max_delta_frac = max(
        abs(results[r]["n_total_linked_specimens"] - base) / base for r in results
    )
    results["max_relative_delta_across_rules"] = max_delta_frac
    results["sensitivity_flag_5pct"] = bool(max_delta_frac > 0.05)

    qc_md = [
        "# MIMIC cohort QC (Task M0, section A.4)\n",
        "## Linkage sensitivity (3 left-edge rules)\n",
        "| rule | recovered from null hadm_id | total linked specimens | multi-match |",
        "|---|---|---|---|",
    ]
    for r in ("edregtime", "admittime", "admittime_minus_24h"):
        v = results[r]
        qc_md.append(f"| {r} | {v['n_recovered_from_null']} | {v['n_total_linked_specimens']} | {v['n_multi_match']} |")
    qc_md.append(f"\nMax relative delta across rules: {max_delta_frac:.4%} "
                 f"({'FLAG >5%' if results['sensitivity_flag_5pct'] else 'OK, <=5%'})\n")

    admissions_path = OUT / "admissions.parquet"
    if admissions_path.exists():
        cohort_out = pd.read_parquet(admissions_path)
        n_subj = cohort_out["subject_id"].nunique()
        qc_md.append("## External plausibility\n")
        qc_md.append(f"{n_subj} distinct subjects with a bacterial BSI admission "
                     f"(flag if outside [1500, 20000] as an order-of-magnitude sanity check).\n")
        if not (1500 <= n_subj <= 20000):
            print(f"[27a/qc] WARNING: n_subjects={n_subj} outside plausible range", flush=True)

    flow_path = OUT / "cohort_flow.json"
    if flow_path.exists():
        flow = json.loads(flow_path.read_text())
        qc_md.append("## Reverse check\n")
        qc_md.append(f"Checked {flow.get('reverse_check_rows_checked')} rows with pre-existing hadm_id; "
                     f"{flow.get('reverse_check_failures_outside_own_admission_interval')} fell outside "
                     f"their own admission's [admittime, dischtime] interval.\n")

    (OUT / "COHORT_QC.md").write_text("\n".join(qc_md))
    _atomic_write_json(results, OUT / "linkage_sensitivity.json")
    print(f"[27a/qc] wrote {OUT / 'COHORT_QC.md'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cohort", "qc"], required=True)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "cohort":
        stage_cohort()
    else:
        stage_qc()


if __name__ == "__main__":
    main()
