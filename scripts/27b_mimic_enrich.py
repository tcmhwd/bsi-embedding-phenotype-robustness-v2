#!/usr/bin/env python3
"""Task M, Step 1b -- structured enrichment of the MIMIC cohort built by
`27a_mimic_cohort.py`: labs, comorbidities (Charlson + source-of-infection),
empiric antibiotics/spectrum flags, and AST detail over the shared panel.

Reads the large hosp-module tables (`labevents.csv` 18G, `prescriptions.csv`
3.3G) via chunked iteration filtered to this cohort's `hadm_id` set, so peak
memory stays bounded regardless of file size. `diagnoses_icd.csv` (174M) and
`microbiologyevents.csv` (868M, re-read here for the AST/interpretation
columns this cohort script's Step 1 pass did not need) are read in full.

Every derivation here is a documented, coarse, MIMIC-side-only heuristic
(never validated against or matched to the Korean derivation's exact
label set) -- consistent with `DESIGN_mimic_external_validation.md` section
B.1's guidance ("do not attempt to reproduce the exact Korean label set;
keep the score/coarse category"). Field-for-field equivalence with Korea is
not attempted; every field that could not be harmonized is recorded, not
silently invented (see `ENRICH_FIELD_NOTES` at the bottom).

Stages (run in order; each writes its own artifact under
out/27_mimic/cohort/ and is independently resumable/re-runnable):
    --stage labs      -> labs.parquet
    --stage comorbid  -> comorbidities.parquet
    --stage abx       -> empiric_antibiotics.parquet
    --stage ast       -> ast_detail.parquet
    --stage merge     -> admissions_enriched.parquet (joins all of the above
                          onto 27a's admissions.parquet by (subject_id, hadm_id))

Outcome-blind (Global Constraints): no mortality/discharge/death/LOS column
is read anywhere in this module.
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
COHORT_DIR = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision/out/27_mimic/cohort")

CHUNK = 3_000_000

# ---------------------------------------------------------------------------
# Labs: itemid whitelist per Korean-template lab field (primary + fallback
# itemids across the Hematology/Chemistry/Blood-Gas fluids MIMIC splits
# these tests across). Window: nearest within +-24h of index_ts; else
# nearest within the admission; else NA (DESIGN doc convention).
# ---------------------------------------------------------------------------
LAB_ITEMIDS = {
    "WBC_Count": [51301, 51755, 51756],
    "PLT_Count": [51265, 53189],
    "Hemoglobin": [51222, 50811, 51640],
    "Creatinine": [50912],
    "T_Bilirubin": [50885],
    "Albumin": [50862],
    "CRP": [50889],
}
ALL_LAB_ITEMIDS = sorted({i for ids in LAB_ITEMIDS.values() for i in ids})
ITEMID_TO_FIELD = {i: field for field, ids in LAB_ITEMIDS.items() for i in ids}
WINDOW_HOURS = 24


def _load_cohort_index():
    adm = pd.read_parquet(COHORT_DIR / "admissions.parquet")
    return adm[["subject_id", "hadm_id", "index_ts", "admittime", "dischtime"]].copy()


def stage_labs() -> None:
    t0 = time.time()
    idx = _load_cohort_index()
    hadm_set = set(idx["hadm_id"].tolist())
    usecols = ["subject_id", "hadm_id", "charttime", "itemid", "valuenum"]

    rows = []
    n_chunks = 0
    n_rows_scanned = 0
    for chunk in pd.read_csv(HOSP / "labevents.csv", usecols=usecols, chunksize=CHUNK, low_memory=False):
        n_chunks += 1
        n_rows_scanned += len(chunk)
        c = chunk[chunk["itemid"].isin(ALL_LAB_ITEMIDS) & chunk["hadm_id"].isin(hadm_set)]
        if len(c):
            rows.append(c)
        if n_chunks % 20 == 0:
            print(f"[27b/labs] scanned {n_rows_scanned:,} rows, kept {sum(len(r) for r in rows):,}", flush=True)

    lab = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=usecols)
    lab["charttime"] = pd.to_datetime(lab["charttime"], errors="coerce")
    lab["field"] = lab["itemid"].map(ITEMID_TO_FIELD)
    lab = lab.dropna(subset=["charttime", "valuenum"])
    print(f"[27b/labs] {len(lab):,} candidate lab rows after itemid+hadm_id+na filter, "
          f"{lab['hadm_id'].nunique():,} distinct admissions represented", flush=True)

    lab = lab.merge(idx, on=["subject_id", "hadm_id"], how="inner")
    lab["dt_hours"] = (lab["charttime"] - lab["index_ts"]).dt.total_seconds() / 3600.0
    lab["abs_dt"] = lab["dt_hours"].abs()
    in_window = lab["abs_dt"] <= WINDOW_HOURS
    in_admission = (lab["charttime"] >= lab["admittime"]) & (lab["charttime"] <= lab["dischtime"])

    def pick_nearest(group: pd.DataFrame) -> pd.Series:
        win = group[in_window.loc[group.index]]
        pool = win if len(win) else group[in_admission.loc[group.index]]
        if not len(pool):
            return pd.Series({"valuenum": np.nan, "used_fallback_window": False, "n_candidates": 0})
        best = pool.loc[pool["abs_dt"].idxmin()]
        return pd.Series({
            "valuenum": best["valuenum"],
            "used_fallback_window": bool(not len(win)),
            "n_candidates": len(pool),
        })

    out_rows = []
    for field in LAB_ITEMIDS:
        sub = lab[lab["field"] == field]
        if sub.empty:
            continue
        picked = sub.groupby(["subject_id", "hadm_id"]).apply(pick_nearest).reset_index()
        picked["field"] = field
        out_rows.append(picked)

    long_df = pd.concat(out_rows, ignore_index=True) if out_rows else pd.DataFrame(
        columns=["subject_id", "hadm_id", "valuenum", "used_fallback_window", "n_candidates", "field"]
    )
    wide = long_df.pivot_table(index=["subject_id", "hadm_id"], columns="field", values="valuenum", aggfunc="first")
    wide = wide.reindex(columns=list(LAB_ITEMIDS.keys())).reset_index()
    wide = idx[["subject_id", "hadm_id"]].merge(wide, on=["subject_id", "hadm_id"], how="left")

    coverage = {f: float(wide[f].notna().mean()) if f in wide.columns else 0.0 for f in LAB_ITEMIDS}
    print(f"[27b/labs] coverage (fraction of admissions with a value within window): {coverage}", flush=True)
    _atomic_write_parquet(wide, COHORT_DIR / "labs.parquet")
    _atomic_write_json(
        {"coverage": coverage, "window_hours": WINDOW_HOURS, "n_admissions": len(idx),
         "elapsed_sec": round(time.time() - t0, 1)},
        COHORT_DIR / "labs_coverage.json",
    )
    print(f"[27b/labs] wrote labs.parquet ({time.time()-t0:.1f}s)", flush=True)


# ---------------------------------------------------------------------------
# Comorbidities: Charlson (Quan 2005 ICD-10 / Deyo-Quan ICD-9-CM, coarse --
# score + broad category only, NOT the Korean label set) + an
# independently-built 8-level source-of-infection heuristic. Both are
# MIMIC-side-only derivations, built without looking at any Korean
# label/cluster, over the WHOLE ADMISSION's diagnoses_icd rows (discharge
# billing codes -- same "whole_admission" timing caveat Task 12 already
# raised for the Korean Charlson fields; symmetric between cohorts).
# ---------------------------------------------------------------------------
CHARLSON_ICD10 = {
    "mi": (1, [r"^I21", r"^I22", r"^I252"]),
    "chf": (1, [r"^I099", r"^I110", r"^I130", r"^I132", r"^I255", r"^I420", r"^I425", r"^I426", r"^I427", r"^I428", r"^I429", r"^I43", r"^I50", r"^P290"]),
    "pvd": (1, [r"^I70", r"^I71", r"^I731", r"^I738", r"^I739", r"^I771", r"^I790", r"^I792", r"^K551", r"^K558", r"^K559", r"^Z958", r"^Z959"]),
    "cvd": (1, [r"^G45", r"^G46", r"^H340", r"^I6"]),
    "dementia": (1, [r"^F00", r"^F01", r"^F02", r"^F03", r"^F051", r"^G30", r"^G311", r"^G312"]),
    "copd": (1, [r"^I278", r"^I279", r"^J4", r"^J6", r"^J684", r"^J701", r"^J703"]),
    "rheumatic": (1, [r"^M05", r"^M06", r"^M315", r"^M32", r"^M33", r"^M34", r"^M351", r"^M353", r"^M360"]),
    "pud": (1, [r"^K25", r"^K26", r"^K27", r"^K28"]),
    "mild_liver": (1, [r"^B18", r"^K700", r"^K701", r"^K702", r"^K703", r"^K709", r"^K713", r"^K714", r"^K715", r"^K717", r"^K73", r"^K74", r"^K760", r"^K762", r"^K763", r"^K764", r"^K768", r"^K769", r"^Z944"]),
    "diabetes_uncomp": (1, [r"^E100", r"^E101", r"^E106", r"^E108", r"^E109", r"^E110", r"^E111", r"^E116", r"^E118", r"^E119", r"^E120", r"^E121", r"^E126", r"^E128", r"^E129", r"^E130", r"^E131", r"^E136", r"^E138", r"^E139", r"^E140", r"^E141", r"^E146", r"^E148", r"^E149"]),
    "diabetes_comp": (2, [r"^E102", r"^E103", r"^E104", r"^E105", r"^E107", r"^E112", r"^E113", r"^E114", r"^E115", r"^E117", r"^E122", r"^E123", r"^E124", r"^E125", r"^E127", r"^E132", r"^E133", r"^E134", r"^E135", r"^E137", r"^E142", r"^E143", r"^E144", r"^E145", r"^E147"]),
    "hemiplegia": (2, [r"^G041", r"^G114", r"^G80", r"^G81", r"^G82", r"^G830", r"^G831", r"^G832", r"^G833", r"^G834", r"^G839"]),
    "renal": (2, [r"^I120", r"^I131", r"^N032", r"^N033", r"^N034", r"^N035", r"^N036", r"^N037", r"^N052", r"^N053", r"^N054", r"^N055", r"^N056", r"^N057", r"^N18", r"^N19", r"^N250", r"^Z490", r"^Z491", r"^Z492", r"^Z940", r"^Z992"]),
    "malignancy": (2, [r"^C[0-9]", r"^C[0-8][0-9]", r"^C9[0-7]"]),
    "mod_severe_liver": (3, [r"^I850", r"^I859", r"^I864", r"^I982", r"^K704", r"^K711", r"^K721", r"^K729", r"^K765", r"^K766", r"^K767"]),
    "metastatic": (6, [r"^C77", r"^C78", r"^C79", r"^C80"]),
    "aids": (6, [r"^B2[0-4]"]),
}
CHARLSON_ICD9 = {
    "mi": (1, [r"^410", r"^412"]),
    "chf": (1, [r"^39891", r"^40201", r"^40211", r"^40291", r"^40401", r"^40403", r"^40411", r"^40413", r"^40491", r"^40493", r"^4254", r"^4255", r"^4256", r"^4257", r"^4258", r"^4259", r"^428"]),
    "pvd": (1, [r"^0930", r"^4373", r"^440", r"^441", r"^4471", r"^5571", r"^5579", r"^V434"]),
    "cvd": (1, [r"^36234", r"^43[0-8]"]),
    "dementia": (1, [r"^290", r"^2941", r"^3312"]),
    "copd": (1, [r"^416[89]", r"^49", r"^50[0-5]", r"^5064", r"^5081", r"^5088"]),
    "rheumatic": (1, [r"^446", r"^7010", r"^71[0-4]", r"^7193", r"^725"]),
    "pud": (1, [r"^53[1-4]"]),
    "mild_liver": (1, [r"^070[23]", r"^070[45]", r"^0709", r"^570", r"^571", r"^5733", r"^5734", r"^5738", r"^5739", r"^V427"]),
    "diabetes_uncomp": (1, [r"^250[0-3]", r"^2508", r"^2509"]),
    "diabetes_comp": (2, [r"^250[4-7]"]),
    "hemiplegia": (2, [r"^334[1]", r"^342", r"^343", r"^344[0-6]", r"^3449"]),
    "renal": (2, [r"^40301", r"^40311", r"^40391", r"^40402", r"^40403", r"^40412", r"^40413", r"^40492", r"^40493", r"^582", r"^583", r"^585", r"^586", r"^5880", r"^V420", r"^V451", r"^V56"]),
    "malignancy": (2, [r"^1[4-9][0-9]", r"^20[0-8]"]),
    "mod_severe_liver": (3, [r"^4560", r"^4561", r"^4562", r"^5722", r"^5723", r"^5724", r"^5728"]),
    "metastatic": (6, [r"^19[6-9]"]),
    "aids": (6, [r"^042", r"^043", r"^044"]),
}


def _compile_map(m):
    return {k: (w, [re.compile(p) for p in pats]) for k, (w, pats) in m.items()}


CHARLSON_ICD10_C = _compile_map(CHARLSON_ICD10)
CHARLSON_ICD9_C = _compile_map(CHARLSON_ICD9)

SOURCE_RULES = [
    ("urinary", ["N390", "N10", "N136", "N30", "N151", "N159", "590", "599.0", "5990"]),
    ("hepatobiliary", ["K80", "K81", "K82", "K83", "5750", "5751", "576", "574"]),
    ("gastrointestinal", ["K35", "K37", "K56", "K57", "K65", "540", "541", "542", "5589", "560"]),
    ("respiratory", ["J13", "J14", "J15", "J18", "J690", "480", "481", "482", "483", "485", "486"]),
    ("skin_soft_tissue", ["L03", "L089", "L97", "681", "682", "6826"]),
    ("catheter_device_related", ["T827", "T836", "T857", "9966", "9993"]),
    ("endovascular", ["I33", "I38", "I39", "421", "424"]),
]


def _icd_hits(codes: list, compiled_map: dict) -> set:
    hits = set()
    for cond, (w, pats) in compiled_map.items():
        for c in codes:
            if any(p.match(c) for p in pats):
                hits.add(cond)
                break
    return hits


def _charlson_score(hits: set, compiled_map: dict) -> int:
    weights = {"mi": 1, "chf": 1, "pvd": 1, "cvd": 1, "dementia": 1, "copd": 1, "rheumatic": 1,
               "pud": 1, "mild_liver": 1, "diabetes_uncomp": 1, "diabetes_comp": 2, "hemiplegia": 2,
               "renal": 2, "malignancy": 2, "mod_severe_liver": 3, "metastatic": 6, "aids": 6}
    score = sum(weights[h] for h in hits)
    if "mod_severe_liver" in hits:
        score -= weights["mild_liver"] if "mild_liver" in hits else 0
    if "metastatic" in hits:
        score -= weights["malignancy"] if "malignancy" in hits else 0
    if "diabetes_comp" in hits:
        score -= weights["diabetes_uncomp"] if "diabetes_uncomp" in hits else 0
    return max(score, 0)


def _charlson_category(score: int) -> str:
    if score == 0:
        return "none"
    if score <= 2:
        return "mild"
    if score <= 4:
        return "moderate"
    return "severe"


def _source_category(codes: list) -> str:
    for label, prefixes in SOURCE_RULES:
        for c in codes:
            if any(c.startswith(p) for p in prefixes):
                return label
    return "unknown_other"


def stage_comorbid() -> None:
    t0 = time.time()
    idx = _load_cohort_index()[["subject_id", "hadm_id"]]
    hadm_set = set(idx["hadm_id"].tolist())

    dx = pd.read_csv(HOSP / "diagnoses_icd.csv", usecols=["subject_id", "hadm_id", "icd_code", "icd_version"], low_memory=False)
    dx = dx[dx["hadm_id"].isin(hadm_set)].copy()
    dx["icd_code"] = dx["icd_code"].astype(str).str.replace(".", "", regex=False).str.upper()
    print(f"[27b/comorbid] {len(dx):,} diagnosis rows for {dx['hadm_id'].nunique():,} cohort admissions", flush=True)

    rows = []
    for (sid, hid), g in dx.groupby(["subject_id", "hadm_id"]):
        codes10 = g.loc[g["icd_version"] == 10, "icd_code"].tolist()
        codes9 = g.loc[g["icd_version"] == 9, "icd_code"].tolist()
        hits = _icd_hits(codes10, CHARLSON_ICD10_C) | _icd_hits(codes9, CHARLSON_ICD9_C)
        score = _charlson_score(hits, CHARLSON_ICD10_C)
        rows.append({
            "subject_id": sid, "hadm_id": hid,
            "charlson_score": score, "charlson_category": _charlson_category(score),
            "source_category": _source_category(codes10 + codes9),
            "n_diagnosis_codes": len(g),
        })
    out = pd.DataFrame(rows)
    out = idx.merge(out, on=["subject_id", "hadm_id"], how="left")
    out["charlson_score"] = out["charlson_score"].fillna(0).astype(int)
    out["charlson_category"] = out["charlson_category"].fillna("none")
    out["source_category"] = out["source_category"].fillna("unknown_other")

    print("[27b/comorbid] charlson_category:", out["charlson_category"].value_counts().to_dict(), flush=True)
    print("[27b/comorbid] source_category:", out["source_category"].value_counts().to_dict(), flush=True)
    _atomic_write_parquet(out, COHORT_DIR / "comorbidities.parquet")
    print(f"[27b/comorbid] wrote comorbidities.parquet ({time.time()-t0:.1f}s)", flush=True)


# ---------------------------------------------------------------------------
# Empiric antibiotics (window: [index_ts, index_ts+72h]) + spectrum flags,
# from prescriptions.csv (free-text `drug` field -- matched by keyword
# substring, case-insensitive; MIMIC has no clean generic-name column at
# this granularity).
# ---------------------------------------------------------------------------
SPECTRUM_KEYWORDS = {
    "antiMRSAab_empiric": ["VANCOMYCIN", "LINEZOLID", "DAPTOMYCIN", "CEFTAROLINE"],
    "antiVREab_empiric": ["LINEZOLID", "DAPTOMYCIN"],
    "antipseudoab_empiric": ["PIPERACILLIN", "CEFEPIME", "CEFTAZIDIME", "MEROPENEM", "IMIPENEM",
                             "CIPROFLOXACIN", "LEVOFLOXACIN", "AZTREONAM", "GENTAMICIN", "TOBRAMYCIN",
                             "AMIKACIN", "COLISTIN", "POLYMYXIN"],
    "carbaab_empiric": ["MEROPENEM", "IMIPENEM", "ERTAPENEM", "DORIPENEM"],
}
EMPIRIC_WINDOW_HOURS = 72
# A conservative allow-list of systemic-antibiotic substrings, used only to
# decide which `drug` rows count as "an antibiotic" for the narrative's
# empiric-therapy list (excludes topical/ophthalmic formulations, which
# `prescriptions.drug` free text does not otherwise flag).
ANTIBIOTIC_KEYWORDS = [
    "VANCOMYCIN", "PIPERACILLIN", "TAZOBACTAM", "CEFEPIME", "CEFTRIAXONE", "CEFTAZIDIME",
    "CEFAZOLIN", "CEFUROXIME", "MEROPENEM", "IMIPENEM", "ERTAPENEM", "CIPROFLOXACIN",
    "LEVOFLOXACIN", "MOXIFLOXACIN", "GENTAMICIN", "TOBRAMYCIN", "AMIKACIN", "AMPICILLIN",
    "SULBACTAM", "AMOXICILLIN", "LINEZOLID", "DAPTOMYCIN", "METRONIDAZOLE", "AZITHROMYCIN",
    "DOXYCYCLINE", "TRIMETHOPRIM", "SULFAMETHOXAZOLE", "CLINDAMYCIN", "OXACILLIN", "NAFCILLIN",
    "PENICILLIN", "AZTREONAM", "COLISTIN", "POLYMYXIN", "RIFAMPIN", "FLUCONAZOLE", "MICAFUNGIN",
    "CASPOFUNGIN",
]


def stage_abx() -> None:
    t0 = time.time()
    idx = _load_cohort_index()
    hadm_set = set(idx["hadm_id"].tolist())
    usecols = ["subject_id", "hadm_id", "starttime", "drug"]

    rows = []
    n_chunks = 0
    for chunk in pd.read_csv(HOSP / "prescriptions.csv", usecols=usecols, chunksize=CHUNK, low_memory=False):
        n_chunks += 1
        c = chunk[chunk["hadm_id"].isin(hadm_set) & chunk["drug"].notna()]
        c = c[c["drug"].str.upper().str.contains("|".join(ANTIBIOTIC_KEYWORDS), regex=True, na=False)]
        if len(c):
            rows.append(c)
        if n_chunks % 20 == 0:
            print(f"[27b/abx] scanned {n_chunks * CHUNK:,} rows, kept {sum(len(r) for r in rows):,}", flush=True)

    abx = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=usecols)
    abx["starttime"] = pd.to_datetime(abx["starttime"], errors="coerce")
    abx = abx.dropna(subset=["starttime"])
    abx = abx.merge(idx[["subject_id", "hadm_id", "index_ts"]], on=["subject_id", "hadm_id"], how="inner")
    abx["dt_hours"] = (abx["starttime"] - abx["index_ts"]).dt.total_seconds() / 3600.0
    empiric = abx[(abx["dt_hours"] >= -EMPIRIC_WINDOW_HOURS) & (abx["dt_hours"] <= EMPIRIC_WINDOW_HOURS)].copy()
    print(f"[27b/abx] {len(empiric):,} empiric antibiotic-order rows in +-{EMPIRIC_WINDOW_HOURS}h window, "
          f"{empiric['hadm_id'].nunique():,} admissions", flush=True)

    empiric["drug_u"] = empiric["drug"].str.upper()

    def summarize(g: pd.DataFrame) -> pd.Series:
        drugs_sorted = g.sort_values("starttime")["drug_u"].tolist()
        seen, uniq = set(), []
        for d in drugs_sorted:
            if d not in seen:
                seen.add(d)
                uniq.append(d)
        flags = {k: "yes" if any(kw in du for du in seen for kw in kws) else "no"
                 for k, kws in SPECTRUM_KEYWORDS.items()}
        out = {"empiric_antibiotics": uniq[:5], "n_empiric_antibiotics": len(uniq)}
        out.update(flags)
        return pd.Series(out)

    summ = empiric.groupby(["subject_id", "hadm_id"]).apply(summarize).reset_index()
    out = idx[["subject_id", "hadm_id"]].merge(summ, on=["subject_id", "hadm_id"], how="left")
    out["n_empiric_antibiotics"] = out["n_empiric_antibiotics"].fillna(0).astype(int)
    out["empiric_antibiotics"] = out["empiric_antibiotics"].apply(lambda v: v if isinstance(v, list) else [])
    for k in SPECTRUM_KEYWORDS:
        out[k] = out[k].fillna("no")

    for k in SPECTRUM_KEYWORDS:
        print(f"[27b/abx] {k}: {out[k].value_counts().to_dict()}", flush=True)
    _atomic_write_parquet(out, COHORT_DIR / "empiric_antibiotics.parquet")
    print(f"[27b/abx] wrote empiric_antibiotics.parquet ({time.time()-t0:.1f}s)", flush=True)


# ---------------------------------------------------------------------------
# AST detail over the shared 11-drug panel (8 Gram-negative + 3
# Gram-positive), restricted to the specimens making up the INDEX episode,
# plus the derived resistance flags (3GCR/CeftriaxoneR/CarbaR/MRSA/VRE/CRAB).
# ---------------------------------------------------------------------------
AB_MAP = {
    "AMPICILLIN": "Ampicillin", "CEFTRIAXONE": "Ceftriaxone", "CEFTAZIDIME": "Ceftazidime",
    "CIPROFLOXACIN": "Ciprofloxacin", "PIPERACILLIN/TAZO": "Piperacillin-Tazobactam",
    "MEROPENEM": "Meropenem", "ERTAPENEM": "Ertapenem", "IMIPENEM": "Imipenem",
    "TRIMETHOPRIM/SULFA": "Sulfamethoxazole-trimethoprim",
    "VANCOMYCIN": "Vancomycin", "OXACILLIN": "Oxacillin", "PENICILLIN G": "Penicillin G",
}
SHARED_PANEL_DISPLAY = [
    "Ampicillin", "Ceftriaxone", "Ceftazidime", "Ciprofloxacin", "Piperacillin-Tazobactam",
    "Meropenem", "Ertapenem", "Sulfamethoxazole-trimethoprim", "Vancomycin", "Oxacillin", "Penicillin G",
]


def stage_ast() -> None:
    from importlib import import_module
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    cohort_mod = import_module("27a_mimic_cohort")

    t0 = time.time()
    idx = pd.read_parquet(COHORT_DIR / "admissions.parquet")[["subject_id", "hadm_id", "index_ts"]]
    subject_set = set(idx["subject_id"].tolist())

    usecols = ["subject_id", "hadm_id", "micro_specimen_id", "chartdate", "charttime",
               "spec_type_desc", "org_name", "ab_name", "interpretation"]
    df = pd.read_csv(HOSP / "microbiologyevents.csv", usecols=usecols, low_memory=False)
    # IMPORTANT: filter by subject_id here, NOT by the raw `hadm_id` column.
    # 52% of this cohort's linked specimens (75,814/145,869, per
    # cohort_flow.json) only acquired a hadm_id via 27a's ED-linkage
    # recovery -- their RAW microbiologyevents.hadm_id is null. Filtering on
    # the raw column here (an earlier version of this function did) silently
    # dropped every recovered index episode and undercounted AST coverage by
    # >60%. Recover hadm_id the identical way 27a does, then join on
    # (subject_id, resolved_hadm_id, ts == index_ts) instead.
    df = df[(df["spec_type_desc"] == "BLOOD CULTURE") & df["subject_id"].isin(subject_set)
            & df["org_name"].notna() & (df["org_name"] != "CANCELLED")].copy()
    df["ts"] = pd.to_datetime(df["charttime"], errors="coerce").fillna(pd.to_datetime(df["chartdate"], errors="coerce"))
    df = df.dropna(subset=["ts"])
    df["drug"] = df["ab_name"].map(AB_MAP)
    df["org_category"] = df["org_name"].map(cohort_mod.classify_organism)

    adm_full = cohort_mod.load_admissions()
    resolved, _ = cohort_mod.recover_hadm_id(df, adm_full, left_edge="edregtime")
    df["resolved_hadm_id"] = resolved
    df = df.drop(columns=["hadm_id"])  # avoid a duplicate-named column after the rename below
    df = df.dropna(subset=["resolved_hadm_id"])
    df["resolved_hadm_id"] = df["resolved_hadm_id"].astype("int64")

    df = df.merge(
        idx.rename(columns={"hadm_id": "resolved_hadm_id"}),
        on=["subject_id", "resolved_hadm_id"], how="inner",
    )
    index_rows = df[df["ts"] == df["index_ts"]].copy()
    index_rows = index_rows.rename(columns={"resolved_hadm_id": "hadm_id"})
    print(f"[27b/ast] {len(index_rows):,} microbiologyevents rows belong to an index episode "
          f"({index_rows['hadm_id'].nunique():,} / {len(idx):,} admissions)", flush=True)

    sev = {"R": 2, "I": 1, "S": 0}
    ast_rows = index_rows.dropna(subset=["drug"]).copy()
    ast_rows["sev"] = ast_rows["interpretation"].map(sev)
    ast_rows = ast_rows.dropna(subset=["sev"])
    ast_rows = ast_rows.sort_values("sev", ascending=False).drop_duplicates(
        subset=["subject_id", "hadm_id", "drug"], keep="first"
    )

    def summarize(g: pd.DataFrame) -> pd.Series:
        d = dict(zip(g["drug"], g["interpretation"]))
        orgs = set(g["org_category"])
        ceftri_r = d.get("Ceftriaxone") == "R"
        ceftaz_r = d.get("Ceftazidime") == "R"
        carba_r = d.get("Meropenem") == "R" or d.get("Imipenem") == "R" or d.get("Ertapenem") == "R"
        mrsa = (d.get("Oxacillin") == "R") and ("Staph_aureus" in orgs)
        vre = (d.get("Vancomycin") == "R") and bool(orgs & {"Efaecium", "Efaecalis", "Other_enterococci"})
        crab = carba_r and ("Acinetobacter" in orgs)
        detail = [f"{disp}: {d[disp]}" for disp in SHARED_PANEL_DISPLAY if disp in d]
        return pd.Series({
            "BSI_3gcr": "yes" if (ceftri_r or ceftaz_r) else "no",
            "BSI_ceftriaxoner": "yes" if ceftri_r else "no",
            "BSI_carbar": "yes" if carba_r else "no",
            "BSI_mrsa": "yes" if mrsa else "no",
            "BSI_vre": "yes" if vre else "no",
            "BSI_crab": "yes" if crab else "no",
            "ast_detail": detail,
            "n_panel_drugs_tested": len(detail),
        })

    summ = ast_rows.groupby(["subject_id", "hadm_id"]).apply(summarize).reset_index()
    out = idx[["subject_id", "hadm_id"]].merge(summ, on=["subject_id", "hadm_id"], how="left")
    for f in ["BSI_3gcr", "BSI_ceftriaxoner", "BSI_carbar", "BSI_mrsa", "BSI_vre", "BSI_crab"]:
        out[f] = out[f].fillna("no")
    out["ast_detail"] = out["ast_detail"].apply(lambda v: v if isinstance(v, list) else [])
    out["n_panel_drugs_tested"] = out["n_panel_drugs_tested"].fillna(0).astype(int)

    coverage = {disp: float((ast_rows["drug"] == disp).groupby(ast_rows["hadm_id"]).any().reindex(idx["hadm_id"], fill_value=False).mean())
                for disp in SHARED_PANEL_DISPLAY}
    print(f"[27b/ast] per-drug testing coverage (fraction of cohort admissions): {coverage}", flush=True)
    for f in ["BSI_3gcr", "BSI_carbar", "BSI_mrsa", "BSI_vre", "BSI_crab"]:
        print(f"[27b/ast] {f}: {out[f].value_counts().to_dict()}", flush=True)

    _atomic_write_parquet(out, COHORT_DIR / "ast_detail.parquet")
    _atomic_write_json({"panel_coverage": coverage, "elapsed_sec": round(time.time() - t0, 1)},
                        COHORT_DIR / "ast_panel_coverage.json")
    print(f"[27b/ast] wrote ast_detail.parquet ({time.time()-t0:.1f}s)", flush=True)


def stage_merge() -> None:
    adm = pd.read_parquet(COHORT_DIR / "admissions.parquet")
    for name in ["labs", "comorbidities", "empiric_antibiotics", "ast_detail"]:
        p = COHORT_DIR / f"{name}.parquet"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing -- run --stage {name.split('_')[0]} first")
        piece = pd.read_parquet(p)
        adm = adm.merge(piece, on=["subject_id", "hadm_id"], how="left", validate="one_to_one")
    _atomic_write_parquet(adm, COHORT_DIR / "admissions_enriched.parquet")
    print(f"[27b/merge] wrote admissions_enriched.parquet, shape={adm.shape}", flush=True)


def _atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)


def _atomic_write_json(obj, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, dest)


STAGES = {
    "labs": stage_labs, "comorbid": stage_comorbid, "abx": stage_abx,
    "ast": stage_ast, "merge": stage_merge,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES.keys()), required=True)
    args = ap.parse_args()
    STAGES[args.stage]()


if __name__ == "__main__":
    main()
