#!/usr/bin/env python3
"""Task M, Step 2 -- render one narrative per MIMIC admission, structurally
analogous to the Korean published template (`reports_pilot_pre/*.report.md`,
the corpus that actually produced the published 16 clusters -- see
`.superpowers/sdd/progress.md`'s corpus-provenance resolution).

Field-for-field equivalence is impossible (design doc section B); this
renders the same SECTION TYPES where MIMIC supports them (administrative,
comorbidities, organ support, labs, spectrum flags, bacterial culture/BSI
episodes, infection treatment) and OMITS every field this project could not
harmonize -- see `FIELDS_NOT_FILLED.md` (written by this script) for the
complete list with reasons.

Hard rules (task brief, binding):
  * No absolute calendar dates anywhere (MIMIC's are shifted anyway; the
    Korean sample literally prints "Admission Date : 2011-02-25", which
    this template deliberately does NOT reproduce). Only hospital-day-
    relative timing is rendered ("Timing: hospital day N" for episodes).
  * No outcome tokens (mortality/discharge/death/LOS) anywhere -- none are
    even loaded into `admissions_enriched.parquet` upstream.
  * No raw patient/admission ID inside the TEXT (per the Korean-template
    audit finding that a raw ID is an untested high-cardinality nuisance
    token) -- the file NAME carries `mimic_<subject_id>_<hadm_id>` for
    joining back to structured data, never a line inside the document.

Run (SBATCH; a few thousand small file writes, i/o bound but simple):
    /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/27c_mimic_narratives.py
"""
import json
import os
from pathlib import Path

import pandas as pd

COHORT_DIR = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision/out/27_mimic/cohort")
NARR_DIR = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision/out/27_mimic/narratives")

SPECTRUM_FIELDS = ["antiMRSAab_empiric", "antiVREab_empiric", "antipseudoab_empiric", "carbaab_empiric"]
# CRP measured (`labs_coverage.json`) at 21.0% of MIMIC admissions vs ~96.1%
# in the Korean corpus (100% - 3.9% NA, per the corpus-wide binary-field
# scan) -- a 75pp gap, far past the pre-specified 20pp KEEP/DROP rule
# (DESIGN doc B.1's CRP row). US practice orders CRP far less often than
# Korean practice for this indication. Dropped from the rendered TEXT (a
# field that reads "NA" in ~79% of MIMIC narratives and ~4% of Korean ones
# is close to a cohort marker under the project's own B.0 principle);
# measured value retained nowhere else either -- reported here, not
# fabricated as a KEEP.
LAB_FIELDS = ["WBC_Count", "T_Bilirubin", "Creatinine", "Hemoglobin", "Albumin", "PLT_Count"]
RESISTANCE_FLAGS = [
    ("3GCR", "BSI_3gcr"), ("CarbaR", "BSI_carbar"), ("MRSA", "BSI_mrsa"),
    ("VRE", "BSI_vre"), ("CeftriaxoneR", "BSI_ceftriaxoner"), ("CRAB", "BSI_crab"),
]


def fmt(v, ndigits=1):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "NA"
    if isinstance(v, float):
        return f"{v:.{ndigits}f}"
    return str(v)


def render_one(row: pd.Series) -> str:
    lines = []

    lines.append("[Administrative]")
    lines.append(f"Age : {fmt(row['age_at_admission'], 0)}")
    lines.append(f"Sex : {row['sex']}")
    lines.append(f"Entry mode : {row['entry_mode']}")
    lines.append("")

    lines.append("[Fungemia]")
    lines.append(f"Fungemia : {row['fungemia']}")
    lines.append("")

    lines.append("[Comorbidities]")
    lines.append(f"Source of infection category : {row['source_category']}")
    lines.append(f"Charlson category : {row['charlson_category']}")
    lines.append(f"Charlson score : {fmt(row['charlson_score'], 0)}")
    lines.append("")

    lines.append("[Organ support]")
    lines.append(f"icu : {row['icu']}")
    lines.append("")

    lines.append("[Physiological and biological parameters]")
    for f in LAB_FIELDS:
        lines.append(f"{f} : {fmt(row.get(f))}")
    lines.append("")

    lines.append("[Spectrum flags]")
    for f in SPECTRUM_FIELDS:
        lines.append(f"{f} : {row[f]}")
    lines.append("")

    lines.append("[Bacterial culture]")
    hosp_day = 0
    try:
        hosp_day = max(0, int((row["index_ts"] - row["admittime"]).total_seconds() // 86400))
    except Exception:
        pass
    lines.append(f"Episode : 1")
    lines.append(f"Timing : hospital day {hosp_day}")
    lines.append(f"Organism summary : {row['organism_category']}")
    lines.append(f"Polymicrobial : {row['polymicrobial']}")
    for disp, col in RESISTANCE_FLAGS:
        lines.append(f"{disp} : {row[col]}")
    detail = row.get("ast_detail")
    detail = detail if isinstance(detail, (list, tuple)) and len(detail) else ["NA"]
    lines.append("AST_detailed :")
    for d in detail:
        lines.append(f"  o {d}")
    lines.append("")

    lines.append("[Infection treatment]")
    lines.append(f"Number of BSI episodes this admission : {fmt(row['n_bsi_episodes'], 0)}")
    emp = row.get("empiric_antibiotics")
    emp = emp if isinstance(emp, (list, tuple)) and len(emp) else []
    if emp:
        for i, drug in enumerate(emp, 1):
            lines.append(f"Antibiotic_{i}_name : {drug.title()}")
    else:
        lines.append("Antibiotic_1_name : NA")

    return "\n".join(lines) + "\n"


FIELDS_NOT_FILLED_MD = """# MIMIC narrative -- fields that could not be harmonized (Task M, Step 2)

Structurally analogous to the Korean published template
(`reports_pilot_pre/*.report.md`), same section TYPES where MIMIC supports
them. Field-for-field equivalence was not attempted (design doc section B).
Every Korean field/section below was considered and could NOT be filled, or
was filled only after a documented change of meaning; reason given for each.

## Dropped entirely (no line rendered, any cohort)

| Korean field/section | Why dropped |
|---|---|
| Patient pseudo-ID | High-cardinality nuisance token, never validated in the published embedding either; correct choice for both cohorts is to omit it from the TEXT (kept only as the file name for joining). |
| Admission Date | Task brief instruction: no absolute calendar dates. MIMIC's dates are randomly shifted per-patient (de-identification), so a literal date would be meaningless AND (per DESIGN doc B.0) a systematic distributional mismatch against Korea's real 2009-2024 dates -- a cohort marker. `admit_period` (3-yr MIMIC anchor-group band) is retained as a STRUCTURED covariate for stratified analysis (Step 4 token recoverability), never rendered in text. |
| Treatment department | No shared taxonomy exists: Korean uses ~45 named clinical departments; MIMIC's nearest analogs (`services.curr_service`, ~20 billing service codes, or `transfers.careunit`) have no validated crosswalk. Inventing one would be an unvalidated mapping doing real analytic work (DESIGN doc B.1). |
| All ICD10 codes during admission (raw code list) | MIMIC mixes ICD-9 and ICD-10 (`icd_version`); embedding literal codes from two different vocabularies is a guaranteed cohort marker. Replaced by a coarse, independently-built 8-level source-of-infection category (see below), which is NOT validated against the Korean `icd_categorized_onlyinfections` grouper -- built for narrative structure only. |
| Vasopressin_any | `inputevents` is MIMIC's ICU module ONLY; a non-ICU patient would read "no" by ascertainment, not physiology (54%+ of this cohort is non-ICU). Systematic bias, not random missingness -- DROP-BOTH per DESIGN doc B.1. |
| ventilator, central/arterial line, urinary/renal catheter | Same ICU-only ascertainment problem as Vasopressin_any (`procedureevents`/`chartevents`). |
| sofa_score_2, sepsis_2, quick_sofa_score_2, sepsis_quicksofa_2 | SOFA/qSOFA require vitals (chartevents, ICU-only) and this MIMIC-IV installation has no `mimic-iv-ed` module for non-ICU vitals -- computable for ~45% of the cohort (the ICU stratum) and 0% for the rest. Differential, severity-correlated missingness -- worse than dropping. |
| SBP, RR, PR, Body_temperature, GCS_2, DBP, MBP | Same ICU-only vitals problem (chartevents-only, no ED module). |
| VRE_colonization, CRE_colonization | Presence-only registers in the Korean corpus itself (near-zero recorded negatives at every pipeline level, per the author's full-corpus scan) -- not usable as a binary anywhere, and MIMIC surveillance-culture screening policy would additionally encode institutional protocol rather than colonization biology. |
| is_empiric_tt_excessivelylarge, is_admission_tt_excessivelylarge | Susceptibility-derived judgment flags specific to an under-documented Korean derivation rule on a different antibiotic vocabulary; not reconstructible. |
| ab_admission_category/name_clean | Whole-admission antibiotic summary fields (Task 12: `whole_admission` timing, already a defect in the Korean index-window template); most exposed to US/Korean formulary-naming mismatch. |
| Fungemia organism / timing detail | Rare (printed only when Fungemia=yes in Korea); the 17-value Korean fungal vocabulary has no MIMIC crosswalk. The binary Fungemia flag is kept; the detail is not. |
| CRP | Measured (not assumed): MIMIC availability within +-24h of index = 21.0% of admissions (`labs_coverage.json`) vs ~96.1% in the Korean corpus (100% - the corpus-wide scan's 3.9% NA rate) -- a 75pp gap, far past the pre-specified 20pp KEEP/DROP rule (DESIGN doc B.1). Dropped from the rendered text; a field reading "NA" in ~79% of MIMIC narratives vs ~4% of Korean ones is close to a cohort marker under this project's own B.0 principle. |

## Filled, with a documented change of meaning ("DERIVE")

| Field | MIMIC derivation | Caveat |
|---|---|---|
| Age | `patients.anchor_age + (admission year - anchor_year)` | MIMIC caps ages >89 at 91 (its own de-identification rule); Korean ages are exact. |
| icu | Any `icustays` row overlapping the admission | Exhaustive in MIMIC (icu module); in the PUBLISHED Korean narratives this field is `yes`/`NA` with zero explicit `no` (author-confirmed events-extract, `NA`->`no` verified) -- both now genuinely comparable. |
| Charlson score / category | Quan (2005) ICD-10 + Deyo-Quan ICD-9-CM comorbidity index over `diagnoses_icd` (whole-admission discharge codes, both ICD versions present in MIMIC) | Coarse (score + 4-level category: none/mild/moderate/severe), NOT matched to the Korean `charlson_category_withcancermerged` label set (per DESIGN doc B.1's explicit guidance not to attempt exact reproduction). Whole-admission timing on both sides (symmetric mismatch vs the Korean index-window ideal). |
| Source of infection category | An 8-level keyword/ICD-prefix heuristic built independently over `diagnoses_icd`, NOT validated against or matched to the Korean `icd_categorized_onlyinfections` grouper, and not clinician-reviewed. Structural placeholder for the "comorbidities" section type, not a validated harmonized variable. |
| Organism summary / Polymicrobial / 3GCR / CeftriaxoneR / CarbaR / MRSA / VRE / CRAB | `microbiologyevents` org_name/ab_name/interpretation, mapped onto the Korean 19-category organism taxonomy and the shared 11-drug AST panel (Ampicillin, Ceftriaxone, Ceftazidime, Ciprofloxacin, Piperacillin-Tazobactam, Meropenem, Ertapenem, Sulfamethoxazole-trimethoprim, Vancomycin, Oxacillin, Penicillin G) | Ertapenem tested in only ~0.1% of MIMIC blood isolates (40/33,746 raw AST rows) -- CarbaR relies mainly on Meropenem/Imipenem. Per-drug testing coverage is reported in `ast_panel_coverage.json`, not silently assumed adequate. |
| Empiric antibiotics | `prescriptions.drug` (free text) matched by a systemic-antibiotic keyword allow-list, window [index-72h, index+72h] | No clean generic-name column at this granularity in MIMIC; keyword matching is a coarser instrument than the Korean structured drug list. No per-drug "Efficacy" line is rendered (the Korean field requires matching the empiric drug to the AST panel result, which MIMIC's free-text drug strings do not support reliably enough to report as a real field). |

## Kept, direct or near-direct

Sex, WBC_Count/Creatinine/T_Bilirubin/PLT_Count/Albumin/Hemoglobin (labevents,
nearest value within +-24h of index culture, else nearest in the admission,
else NA -- see `labs_coverage.json` for per-field coverage), Fungemia (binary,
derived from a co-occurring fungal-positive blood culture at or before index).

## Never included, by construction (Global Constraints)

Any mortality/discharge/death/LOS field. None are read anywhere upstream of
this script (`27a_mimic_cohort.py`, `27b_mimic_enrich.py` never load
`hospital_expire_flag`, `deathtime`, `dischtime` as a modeled/rendered value
-- `dischtime`/`admittime` are used only for interval arithmetic).
"""


def main() -> None:
    NARR_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(COHORT_DIR / "admissions_enriched.parquet")
    df["admittime"] = pd.to_datetime(df["admittime"])
    df["index_ts"] = pd.to_datetime(df["index_ts"])

    manifest_rows = []
    n_written = 0
    for _, row in df.iterrows():
        fname = f"mimic_{row['subject_id']}_{row['hadm_id']}.report.md"
        text = render_one(row)
        (NARR_DIR / fname).write_text(text)
        manifest_rows.append({
            "admission_ID": f"mimic_{row['subject_id']}_{row['hadm_id']}",
            "subject_id": int(row["subject_id"]), "hadm_id": int(row["hadm_id"]),
            "report_file": str(NARR_DIR / fname),
            "organism_category": row["organism_category"], "icu": row["icu"],
            "care_unit": row["care_unit"], "admit_period": row["admit_period"],
            "n_bsi_episodes": int(row["n_bsi_episodes"]),
        })
        n_written += 1

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_parquet(COHORT_DIR / "narrative_manifest.parquet", index=False)
    (NARR_DIR.parent / "cohort" / "FIELDS_NOT_FILLED.md").write_text(FIELDS_NOT_FILLED_MD)

    lengths = manifest_df["report_file"].apply(lambda p: len(Path(p).read_text()))
    print(json.dumps({
        "n_written": n_written,
        "char_len_min": int(lengths.min()), "char_len_median": float(lengths.median()),
        "char_len_max": int(lengths.max()),
    }, indent=2), flush=True)
    print(f"[27c] wrote {n_written} narratives to {NARR_DIR}", flush=True)
    print(f"[27c] wrote {COHORT_DIR / 'FIELDS_NOT_FILLED.md'}", flush=True)


if __name__ == "__main__":
    main()
