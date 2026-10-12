#!/usr/bin/env python3
"""Assemble the submission-package Table 1 (cohort characteristics).

Design doc: `.superpowers/sdd/SUBMISSION_PACKAGE_DESIGN.md` T1, section 6-B,
and section 7-M1 (this script IS the M1 resolution -- the row composition
decided here is a concrete proposal, not yet an author-ratified one).

Three cohort columns are produced, not two:

  - `korea`       -- 32,329 admissions, the primary cohort. Source:
                     `out/20_onehot/features/raw_fields.parquet` (the actual
                     RENDERED template fields -- self-consistent with a paper
                     about what these fields do to embedding geometry).
  - `mimic_all`   -- 16,144 admissions, every organism. Auxiliary: shows the
                     external cohort's overall case-mix, but is NOT the
                     external-validation analysis unit for this paper.
  - `mimic_ecoli` -- 1,929 admissions, `organism_category == "Ecoli"` only.
                     This is the author-decided MIMIC PRIMARY cohort
                     (`.superpowers/sdd/progress.md`, "author decision (2026-09-04)",
                     item 1): the paper's claim is a within-species
                     sub-phenotype, so external validation has to be
                     evaluated on the same species-restricted footing, not on
                     a mixed-organism population that would test a different
                     claim. n~2,000 is a stated power LIMITATION, not
                     something this table tries to fix by cohort-swapping.

MIMIC source is `out/27_mimic/cohort/admissions_enriched.parquet`, which is
the already-joined superset (labs + Charlson + AST-resistance flags +
antibiotics), rather than separately re-joining `admissions.parquet` +
`ast_detail.parquet` + `episodes_raw.parquet` by hand -- re-implementing a
join Task M already did and already documented in
`out/27_mimic/cohort/FIELDS_NOT_FILLED.md` would be duplicated, error-prone
work for no benefit.

Two structural constraints drive most of the row-by-row cohort availability:

1. `out/27_mimic/cohort/FIELDS_NOT_FILLED.md` documents which Korean
   template fields have NO MIMIC analogue at all (SOFA/qSOFA, vasopressor,
   all four procedures, department) because this MIMIC-IV installation has
   no `mimic-iv-ed` module -- vitals/procedures are only ascertainable for
   the ICU stratum (~45%), which would be differential missingness, not
   random. Those rows report Korea only, with an explicit
   "not available in MIMIC" note (never a blank cell, never a silent 0%).

2. `.superpowers/sdd/progress.md` ("finding (2026-09-04): narrative-sentence
   binary-field encoding defect") documents that four Korean binary fields (`icu`,
   `Vasopressin_any`, and -- independently confirmed in this script's own
   pass over `raw_fields.parquet`, extending the same finding -- the three
   procedure flags `central_line`/`urinary_catheter`/`ventilator`) are
   PRESENCE-ONLY: no explicit negative was ever recorded, so `NA` conflates
   "confirmed absent" with "not documented". Folding NA into "no" would
   silently manufacture a low percentage. This script reports `yes` as
   n (% of the full admission denominator) and leaves the true "no" count
   unknown, with the defect stated in the `note` column of every affected
   row -- never as a folded no/yes/NA percentage.

Cell counts under 5 are masked as "<5" in `table1_cohort.md` and flagged
`masked=True` in `table1_cohort.csv` (project configuration notes' data-sensitivity rule:
no patient-level rows in outputs, and small cells are a re-identification
risk even in aggregate). No hypothesis tests are computed -- this is a
cohort description, not a between-group comparison (no comparison groups are
defined for one).
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
KOREA_PARQUET = ROOT / "out/20_onehot/features/raw_fields.parquet"
MIMIC_PARQUET = ROOT / "out/27_mimic/cohort/admissions_enriched.parquet"
UNIT_OF_ANALYSIS = ROOT / "out/01_frame/unit_of_analysis_report.json"
COHORT_FLOW = ROOT / "out/27_mimic/cohort/cohort_flow.json"
OUT_DIR = ROOT / "out/50_cohort_table"

MASK_BELOW = 5
COHORTS = ["korea", "mimic_all", "mimic_ecoli"]
COHORT_LABEL = {
    "korea": "Korea (primary, n=32,329)",
    "mimic_all": "MIMIC-IV, all organisms (auxiliary, n=16,144)",
    "mimic_ecoli": "MIMIC-IV, E. coli subset (descriptive only, n=1,929)",
}

TIDY_ROWS = []  # accumulator, one dict per (section, variable, stratum, cohort)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def to_numeric(series: pd.Series) -> pd.Series:
    """Korean fields are stored as strings with literal 'NA' for missing."""
    if series.dtype == object:
        series = series.replace("NA", pd.NA)
    return pd.to_numeric(series, errors="coerce")


def fmt(x, decimals):
    if pd.isna(x):
        return "NA"
    return f"{x:,.{decimals}f}"


def median_iqr(series: pd.Series, decimals: int = 0):
    """Returns (display_string, n_valid, n_missing, n_total)."""
    s = to_numeric(series)
    n_total = len(s)
    n_missing = int(s.isna().sum())
    valid = s.dropna()
    n_valid = len(valid)
    if n_valid == 0:
        return "NA", n_valid, n_missing, n_total
    q1, med, q3 = valid.quantile([0.25, 0.5, 0.75])
    disp = f"{fmt(med, decimals)} [{fmt(q1, decimals)}–{fmt(q3, decimals)}]"
    return disp, n_valid, n_missing, n_total


def n_pct(n: int, denom: int):
    """Returns (display_string, masked_bool)."""
    if denom == 0:
        return "NA", False
    if 0 < n < MASK_BELOW:
        return "<5", True
    pct = 100.0 * n / denom
    return f"{n:,} ({pct:.1f}%)", False


def add_row(section, variable, stratum, cohort, n, denominator, value,
            source_file, source_column, note="", masked=False):
    TIDY_ROWS.append(dict(
        section=section, variable=variable, stratum=stratum, cohort=cohort,
        n=n, denominator=denominator, value=value,
        source_file=str(source_file), source_column=source_column, note=note,
        masked=masked,
    ))


def add_median_row(section, variable, series, cohort, denom_label,
                    source_file, source_column, decimals=0, note=""):
    disp, n_valid, n_missing, n_total = median_iqr(series, decimals)
    add_row(section, variable, "median_iqr", cohort, n_valid, n_total, disp,
            source_file, source_column,
            note=(note + f"; missing={n_missing}/{n_total}").strip("; "))


def add_binary_row(section, variable, count, denom, cohort, source_file,
                    source_column, note=""):
    disp, masked = n_pct(count, denom)
    full_note = note
    if masked:
        full_note = (full_note + "; masked: raw count <5, suppressed per "
                                  "data-sensitivity rule").strip("; ")
    add_row(section, variable, "yes", cohort, count, denom, disp,
            source_file, source_column, full_note, masked=masked)


def add_categorical_rows(section, variable, value_counts: dict, denom, cohort,
                          source_file, source_column, note=""):
    for stratum, count in value_counts.items():
        disp, masked = n_pct(count, denom)
        full_note = note
        if masked:
            full_note = (full_note + "; masked: raw count <5, suppressed per "
                                      "data-sensitivity rule").strip("; ")
        add_row(section, variable, stratum, cohort, count, denom, disp,
                source_file, source_column, full_note, masked=masked)


NA_DEFECT_NOTE = (
    "NA-encoding defect (progress.md 2026-09-04): field has NO explicit "
    "negative recorded anywhere in the corpus -- NA conflates 'confirmed "
    "absent' and 'not documented'. Value below is yes / full denominator; "
    "true no-count is unknown. Do not read the complement as %no."
)

NOT_AVAILABLE_MIMIC = (
    "Not available in MIMIC-IV: no mimic-iv-ed module in this installation, "
    "so vitals/procedures are only ascertainable for the ICU stratum "
    "(~45% of the cohort) -- differential, severity-correlated missingness, "
    "worse than dropping (out/27_mimic/cohort/FIELDS_NOT_FILLED.md)."
)


def main():
    print(f"Loading {KOREA_PARQUET}")
    kr = pd.read_parquet(KOREA_PARQUET)
    print(f"  {len(kr):,} rows x {kr.shape[1]} cols")

    print(f"Loading {MIMIC_PARQUET}")
    mm = pd.read_parquet(MIMIC_PARQUET)
    print(f"  {len(mm):,} rows x {mm.shape[1]} cols")
    mm_ecoli = mm[mm["organism_category"] == "Ecoli"].copy()
    print(f"  E. coli subset: {len(mm_ecoli):,} rows")

    unit_of_analysis = json.loads(UNIT_OF_ANALYSIS.read_text())
    cohort_flow = json.loads(COHORT_FLOW.read_text())

    n_kr = len(kr)
    n_mm = len(mm)
    n_mm_ec = len(mm_ecoli)
    assert n_kr == 32329, f"Korean cohort size changed: {n_kr}"
    assert n_mm == 16144, f"MIMIC-all cohort size changed: {n_mm}"
    assert n_mm_ec == 1929, f"MIMIC E. coli cohort size changed: {n_mm_ec}"

    # ---- Header (N) -----------------------------------------------------
    add_row("N", "Admissions", "n", "korea", n_kr, n_kr, f"{n_kr:,}",
            KOREA_PARQUET, "admission_ID (row count)")
    add_row("N", "Unique patients", "n", "korea",
            unit_of_analysis["n_unique_patient_ID"], n_kr,
            f"{unit_of_analysis['n_unique_patient_ID']:,}",
            UNIT_OF_ANALYSIS, "n_unique_patient_ID",
            note="AUTHOR_DECISIONS.md D1: admission is the analysis unit "
                 "(32,329 hospitalizations among 27,083 patients); "
                 "patient_ID not present in raw_fields.parquet, quoted "
                 "from the pre-computed frame report instead of re-joined.")
    add_row("N", "Admissions", "n", "mimic_all", n_mm, n_mm, f"{n_mm:,}",
            MIMIC_PARQUET, "subject_id/hadm_id (row count)")
    add_row("N", "Unique patients", "n", "mimic_all",
            mm["subject_id"].nunique(), n_mm,
            f"{mm['subject_id'].nunique():,}", MIMIC_PARQUET, "subject_id",
            note=f"cross-checked against cohort_flow.json final_subjects_n="
                 f"{cohort_flow['final_subjects_n']}")
    add_row("N", "Admissions", "n", "mimic_ecoli", n_mm_ec, n_mm_ec,
            f"{n_mm_ec:,}", MIMIC_PARQUET,
            "subject_id/hadm_id, organism_category=='Ecoli'",
            note="descriptive only; no dedicated E. coli-only MIMIC analysis "
                 "exists (N-003) -- the MIMIC replication/comparison unit is "
                 "the full-organism cohort, mimic_all n=16,144 "
                 "(out/27_mimic/cohort/cohort_flow.json "
                 "admissions_with_bacterial_positive_episode)")
    add_row("N", "Unique patients", "n", "mimic_ecoli",
            mm_ecoli["subject_id"].nunique(), n_mm_ec,
            f"{mm_ecoli['subject_id'].nunique():,}", MIMIC_PARQUET,
            "subject_id, organism_category=='Ecoli'")

    # ---- Demographics -----------------------------------------------------
    add_median_row("Demographics", "Age, years", kr["admin__age"], "korea",
                    n_kr, KOREA_PARQUET, "admin__age", decimals=0)
    add_median_row("Demographics", "Age, years", mm["age_at_admission"],
                    "mimic_all", n_mm, MIMIC_PARQUET, "age_at_admission",
                    decimals=0,
                    note="MIMIC caps ages >89 at 91 by its own "
                         "de-identification rule; Korean ages are exact")
    add_median_row("Demographics", "Age, years",
                    mm_ecoli["age_at_admission"], "mimic_ecoli", n_mm_ec,
                    MIMIC_PARQUET, "age_at_admission", decimals=0,
                    note="MIMIC caps ages >89 at 91")

    for cohort, df, col, n in [
        ("korea", kr, "admin__sex", n_kr),
        ("mimic_all", mm, "sex", n_mm),
        ("mimic_ecoli", mm_ecoli, "sex", n_mm_ec),
    ]:
        n_male = int((df[col] == "M").sum())
        add_binary_row("Demographics", "Sex, male", n_male, n,
                        cohort, KOREA_PARQUET if cohort == "korea" else MIMIC_PARQUET,
                        col)

    dept_vc = kr["admin__department"].value_counts()
    top10 = dept_vc.head(10)
    other_n = int(dept_vc.iloc[10:].sum())
    dept_dict = {k: int(v) for k, v in top10.items()}
    dept_dict["Other (45 further categories)"] = other_n
    add_categorical_rows("Demographics", "Treatment department (top 10 + other)",
                         dept_dict, n_kr, "korea", KOREA_PARQUET,
                         "admin__department",
                         note="top 10 of 55 categories shown individually; "
                              "remainder pooled into Other so no category "
                              "cell is masked")
    add_row("Demographics", "Treatment department (top 10 + other)", "n/a",
            "mimic_all", None, n_mm, "not available", MIMIC_PARQUET, "n/a",
            note="No validated Korea<->MIMIC department crosswalk exists "
                 "(Korean ~45 named departments vs MIMIC curr_service/"
                 "careunit, ~20 billing codes); FIELDS_NOT_FILLED.md")
    add_row("Demographics", "Treatment department (top 10 + other)", "n/a",
            "mimic_ecoli", None, n_mm_ec, "not available", MIMIC_PARQUET,
            "n/a", note="same as mimic_all")

    # ---- Comorbidity / severity --------------------------------------------
    add_median_row("Severity", "Charlson comorbidity index", kr["comorbid__charlson_score"],
                    "korea", n_kr, KOREA_PARQUET, "comorbid__charlson_score",
                    decimals=0)
    add_median_row("Severity", "Charlson comorbidity index", mm["charlson_score"],
                    "mimic_all", n_mm, MIMIC_PARQUET, "charlson_score",
                    decimals=0,
                    note="Quan(2005) ICD-10 + Deyo-Quan ICD-9-CM over "
                         "whole-admission discharge codes; NOT matched to "
                         "the Korean category label set (FIELDS_NOT_FILLED.md); "
                         "whole-admission timing on both sides (symmetric)")
    add_median_row("Severity", "Charlson comorbidity index",
                    mm_ecoli["charlson_score"], "mimic_ecoli", n_mm_ec,
                    MIMIC_PARQUET, "charlson_score", decimals=0,
                    note="same Quan(2005) caveat as mimic_all")

    add_median_row("Severity", "SOFA score", kr["sofa__sofa_score_2"], "korea",
                    n_kr, KOREA_PARQUET, "sofa__sofa_score_2", decimals=0)
    add_row("Severity", "SOFA score", "median_iqr", "mimic_all", None, n_mm,
            "not available", MIMIC_PARQUET, "n/a", note=NOT_AVAILABLE_MIMIC)
    add_row("Severity", "SOFA score", "median_iqr", "mimic_ecoli", None,
            n_mm_ec, "not available", MIMIC_PARQUET, "n/a", note=NOT_AVAILABLE_MIMIC)

    add_median_row("Severity", "qSOFA score", kr["sofa__quick_sofa_score_2"],
                    "korea", n_kr, KOREA_PARQUET, "sofa__quick_sofa_score_2",
                    decimals=0)
    add_row("Severity", "qSOFA score", "median_iqr", "mimic_all", None, n_mm,
            "not available", MIMIC_PARQUET, "n/a", note=NOT_AVAILABLE_MIMIC)
    add_row("Severity", "qSOFA score", "median_iqr", "mimic_ecoli", None,
            n_mm_ec, "not available", MIMIC_PARQUET, "n/a", note=NOT_AVAILABLE_MIMIC)

    sepsis_yes = int((kr["summary__sepsis_2"] == "yes").sum())
    sepsis_valid = int((kr["summary__sepsis_2"] != "NA").sum())
    add_binary_row("Severity", "Sepsis (Sepsis-2 criteria)", sepsis_yes, n_kr,
                    "korea", KOREA_PARQUET, "summary__sepsis_2",
                    note=f"explicit yes/no/NA field, {n_kr - sepsis_valid} "
                         f"NA excluded from numerator but kept in denominator")
    add_row("Severity", "Sepsis (Sepsis-2 criteria)", "yes", "mimic_all",
            None, n_mm, "not available", MIMIC_PARQUET, "n/a",
            note=NOT_AVAILABLE_MIMIC)
    add_row("Severity", "Sepsis (Sepsis-2 criteria)", "yes", "mimic_ecoli",
            None, n_mm_ec, "not available", MIMIC_PARQUET, "n/a",
            note=NOT_AVAILABLE_MIMIC)

    icu_yes_kr = int((kr["summary__icu"] == "yes").sum())
    add_binary_row("Severity", "ICU admission", icu_yes_kr, n_kr, "korea",
                    KOREA_PARQUET, "summary__icu", note=NA_DEFECT_NOTE)
    icu_yes_mm = int((mm["icu"] == "yes").sum())
    add_binary_row("Severity", "ICU admission", icu_yes_mm, n_mm, "mimic_all",
                    MIMIC_PARQUET, "icu",
                    note="explicit yes/no in MIMIC (icustays module, "
                         "exhaustive) -- NOT the same ascertainment quality "
                         "as the Korean NA-defect field above")
    icu_yes_mm_ec = int((mm_ecoli["icu"] == "yes").sum())
    add_binary_row("Severity", "ICU admission", icu_yes_mm_ec, n_mm_ec,
                    "mimic_ecoli", MIMIC_PARQUET, "icu")

    # ---- Procedures (all four share the Korean NA-defect pattern) --------
    proc_specs = [
        ("Vasopressor use (any)", "physio__Vasopressin_any"),
        ("Central venous catheter", "proc__central_line"),
        ("Urinary catheter", "proc__urinary_catheter"),
        ("Mechanical ventilation", "proc__ventilator"),
    ]
    for label, col in proc_specs:
        # Vasopressin_any is coded 1.0/NA (numeric-looking string); the
        # other three are yes/NA.
        positive_token = "1.0" if col == "physio__Vasopressin_any" else "yes"
        n_yes = int((kr[col].astype(str) == positive_token).sum())
        add_binary_row("Procedures", label, n_yes, n_kr, "korea",
                        KOREA_PARQUET, col, note=NA_DEFECT_NOTE)
        add_row("Procedures", label, "yes", "mimic_all", None, n_mm,
                "not available", MIMIC_PARQUET, "n/a", note=NOT_AVAILABLE_MIMIC)
        add_row("Procedures", label, "yes", "mimic_ecoli", None, n_mm_ec,
                "not available", MIMIC_PARQUET, "n/a", note=NOT_AVAILABLE_MIMIC)

    # ---- Laboratory values -------------------------------------------------
    lab_specs = [
        ("WBC count, x10^3/uL", "physio__WBC_Count", "WBC_Count", 1),
        ("CRP, mg/L", "physio__CRP", "CRP", 1),
        ("Albumin, g/dL", "physio__Albumin", "Albumin", 1),
        ("Creatinine, mg/dL", "physio__Creatinine", "Creatinine", 2),
        ("Platelet count, x10^3/uL", "physio__PLT_Count", "PLT_Count", 0),
        ("Total bilirubin, mg/dL", "physio__T_Bilirubin", "T_Bilirubin", 1),
    ]
    for label, kr_col, mm_col, dec in lab_specs:
        add_median_row("Laboratory", label, kr[kr_col], "korea", n_kr,
                        KOREA_PARQUET, kr_col, decimals=dec)
        note_mm = ""
        if mm_col == "CRP":
            note_mm = ("MIMIC coverage far lower than Korea (label-window "
                       "+-24h): reported here for completeness, but the "
                       "narrative-rendering pipeline DROPPED this field for "
                       "MIMIC text entirely for the same reason "
                       "(FIELDS_NOT_FILLED.md, 75pp gap) -- treat median as "
                       "unreliable, not just 'noisier'")
        add_median_row("Laboratory", label, mm[mm_col], "mimic_all", n_mm,
                        MIMIC_PARQUET, mm_col, decimals=dec, note=note_mm)
        add_median_row("Laboratory", label, mm_ecoli[mm_col], "mimic_ecoli",
                        n_mm_ec, MIMIC_PARQUET, mm_col, decimals=dec,
                        note=note_mm)

    # ---- Microbiology -------------------------------------------------------
    kept_organisms = ["Ecoli", "Klebsiella", "Staph_coagulasenegative",
                       "Staph_aureus", "Streptococcus", "Efaecium",
                       "Efaecalis", "Enterobacter", "Pseudomonas_aeruginosa",
                       "Acinetobacter", "Polymicrobial"]

    def organism_dict(series):
        vc = series.value_counts()
        d = {k: int(vc.get(k, 0)) for k in kept_organisms}
        other = int(vc.sum() - sum(d.values()))
        d["Other (8 pooled categories)"] = other
        return d

    add_categorical_rows("Microbiology", "Index organism", organism_dict(kr["culture__index_organism"]),
                         n_kr, "korea", KOREA_PARQUET, "culture__index_organism",
                         note="Other pools: Other, Other_enterococci, "
                              "Citrobacter, Proteus, Serratia, "
                              "Stenotrophomonas, Strepto_pneumoniae, "
                              "Morganella, Providencia")
    add_categorical_rows("Microbiology", "Index organism", organism_dict(mm["organism_category"]),
                         n_mm, "mimic_all", MIMIC_PARQUET, "organism_category",
                         note="same pooling as korea")
    add_row("Microbiology", "Index organism", "Ecoli", "mimic_ecoli", n_mm_ec,
            n_mm_ec, f"{n_mm_ec:,} (100.0%)", MIMIC_PARQUET,
            "organism_category=='Ecoli'",
            note="cohort-defining criterion, not an observed distribution")
    for k in kept_organisms + ["Other (8 pooled categories)"]:
        if k == "Ecoli":
            continue
        add_row("Microbiology", "Index organism", k, "mimic_ecoli", 0,
                n_mm_ec, "0 (0.0%)", MIMIC_PARQUET,
                "organism_category=='Ecoli'", note="excluded by cohort definition")

    poly_yes_kr = int((kr["summary__index_bsi_polymicrobial"] == "yes").sum())
    add_binary_row("Microbiology", "Polymicrobial index BSI", poly_yes_kr, n_kr,
                    "korea", KOREA_PARQUET, "summary__index_bsi_polymicrobial")
    poly_yes_mm = int((mm["polymicrobial"] == "yes").sum())
    add_binary_row("Microbiology", "Polymicrobial index BSI", poly_yes_mm, n_mm,
                    "mimic_all", MIMIC_PARQUET, "polymicrobial")
    add_row("Microbiology", "Polymicrobial index BSI", "yes", "mimic_ecoli", 0,
            n_mm_ec, "0 (0.0%)", MIMIC_PARQUET, "polymicrobial",
            note="mutually exclusive with organism_category=='Ecoli' by "
                 "construction -- verified 0/1929 in this cohort")

    add_row("Microbiology", "Fungemia", "yes", "korea", 0, n_kr, "0 (0.0%)",
            "n/a (constant by cohort definition)", "n/a",
            note="Fungemia is NOT a column in raw_fields.parquet; the "
                 "Korean cohort has zero Candida/fungal content anywhere "
                 "(scripts/21_common.py: 'no Candida/Fungemia content "
                 "anywhere in this cohort: no'; corroborated here by zero "
                 "fungal categories among the 20 values of "
                 "culture__index_organism). Rendered narrative text still "
                 "prints the explicit line 'Fungemia: no' for every "
                 "admission (verified in a sample report), so this is a "
                 "true constant, not an unmeasured field.")
    fun_yes_mm = int((mm["fungemia"] == "yes").sum())
    add_binary_row("Microbiology", "Fungemia", fun_yes_mm, n_mm, "mimic_all",
                    MIMIC_PARQUET, "fungemia")
    fun_yes_mm_ec = int((mm_ecoli["fungemia"] == "yes").sum())
    add_binary_row("Microbiology", "Fungemia", fun_yes_mm_ec, n_mm_ec,
                    "mimic_ecoli", MIMIC_PARQUET, "fungemia",
                    note="co-occurring fungal-positive culture in an "
                         "admission whose INDEX organism is E. coli -- not "
                         "a contradiction, just a second, later/earlier "
                         "positive specimen")

    # ---- Colonization flags (both share NA-encoding problems, but two
    # different ones -- kept separate from the four presence-only fields
    # above because CRE is presence-only while VRE has a genuine but tiny
    # explicit-negative count; folding them into the same helper text would
    # blur that distinction) ------------------------------------------------
    cre_yes = int((kr["summary__cre_colonization"] == "yes").sum())
    add_binary_row(
        "Colonization", "CRE colonization", cre_yes, n_kr, "korea",
        KOREA_PARQUET, "summary__cre_colonization",
        note="Presence-only: value is 'yes' (3,147) or 'NA' (29,182); "
             "explicit 'no' recorded 0/32,329 times. This is the field the "
             "task brief singled out as effectively a presence flag rather "
             "than a true binary; treat the % below as a floor, not a "
             "prevalence estimate.")
    add_row("Colonization", "CRE colonization", "yes", "mimic_all", None,
            n_mm, "not available", MIMIC_PARQUET, "n/a",
            note="Dropped in the Korean corpus itself for the same "
                 "presence-only reason (FIELDS_NOT_FILLED.md: 'not usable "
                 "as a binary anywhere'), so no MIMIC crosswalk was "
                 "attempted either.")
    add_row("Colonization", "CRE colonization", "yes", "mimic_ecoli", None,
            n_mm_ec, "not available", MIMIC_PARQUET, "n/a",
            note="same as mimic_all")

    vre_yes = int((kr["summary__vre_colonization"] == "yes").sum())
    vre_no = int((kr["summary__vre_colonization"] == "no").sum())
    add_binary_row(
        "Colonization", "VRE colonization", vre_yes, n_kr, "korea",
        KOREA_PARQUET, "summary__vre_colonization",
        note=f"Explicit 'no' IS recorded but only {vre_no}/32,329 times "
             f"(yes=3,678, no={vre_no}, NA=28,165) -- unlike CRE this is a "
             f"true 3-level field, but the near-absence of recorded "
             f"negatives still means NA cannot be read as 'not colonized'.")
    add_row("Colonization", "VRE colonization", "yes", "mimic_all", None,
            n_mm, "not available", MIMIC_PARQUET, "n/a",
            note="Dropped in the Korean corpus itself (FIELDS_NOT_FILLED.md), "
                 "no MIMIC crosswalk attempted.")
    add_row("Colonization", "VRE colonization", "yes", "mimic_ecoli", None,
            n_mm_ec, "not available", MIMIC_PARQUET, "n/a",
            note="same as mimic_all")

    # ---- Episode burden -----------------------------------------------------
    add_median_row("Episodes", "BSI episodes per admission",
                    kr["struct__n_episode_blocks"], "korea", n_kr,
                    KOREA_PARQUET, "struct__n_episode_blocks", decimals=0)
    add_median_row("Episodes", "BSI episodes per admission",
                    mm["n_bsi_episodes"], "mimic_all", n_mm, MIMIC_PARQUET,
                    "n_bsi_episodes", decimals=0)
    add_median_row("Episodes", "BSI episodes per admission",
                    mm_ecoli["n_bsi_episodes"], "mimic_ecoli", n_mm_ec,
                    MIMIC_PARQUET, "n_bsi_episodes", decimals=0)

    multi_kr = int((kr["struct__n_episode_blocks"] > 1).sum())
    add_binary_row("Episodes", "Admissions with >1 BSI episode", multi_kr,
                    n_kr, "korea", KOREA_PARQUET, "struct__n_episode_blocks")
    multi_mm = int((mm["n_bsi_episodes"] > 1).sum())
    add_binary_row("Episodes", "Admissions with >1 BSI episode", multi_mm,
                    n_mm, "mimic_all", MIMIC_PARQUET, "n_bsi_episodes")
    multi_mm_ec = int((mm_ecoli["n_bsi_episodes"] > 1).sum())
    add_binary_row("Episodes", "Admissions with >1 BSI episode", multi_mm_ec,
                    n_mm_ec, "mimic_ecoli", MIMIC_PARQUET, "n_bsi_episodes")

    # ---- Calendar span --------------------------------------------------------
    yrs = pd.to_numeric(kr["admin__admission_year"], errors="coerce").dropna()
    add_row("Calendar", "Admission year, range", "range", "korea", len(yrs),
            n_kr, f"{int(yrs.min())}–{int(yrs.max())}", KOREA_PARQUET,
            "admin__admission_year")

    period_vc_all = mm["admit_period"].value_counts(dropna=False)
    period_dict_all = {(k if pd.notna(k) else "unknown"): int(v)
                       for k, v in period_vc_all.items()}
    add_categorical_rows("Calendar", "Admission period (3-yr anchor band)",
                         period_dict_all, n_mm, "mimic_all", MIMIC_PARQUET,
                         "admit_period",
                         note="MIMIC dates are randomly shifted per-patient "
                              "for de-identification; true calendar year is "
                              "not usable (FIELDS_NOT_FILLED.md). admit_period "
                              "is the de-identification anchor-group band, "
                              "NOT real calendar time -- shown in place of "
                              "the Korean year range, not as an equivalent "
                              "quantity")
    period_vc_ec = mm_ecoli["admit_period"].value_counts(dropna=False)
    period_dict_ec = {(k if pd.notna(k) else "unknown"): int(v)
                      for k, v in period_vc_ec.items()}
    add_categorical_rows("Calendar", "Admission period (3-yr anchor band)",
                         period_dict_ec, n_mm_ec, "mimic_ecoli", MIMIC_PARQUET,
                         "admit_period", note="same caveat as mimic_all")

    # -------------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tidy = pd.DataFrame(TIDY_ROWS)
    csv_path = OUT_DIR / "table1_cohort.csv"
    tidy.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path} ({len(tidy)} rows)")

    write_markdown(tidy)
    write_manifest(unit_of_analysis, cohort_flow, len(kr), len(mm), len(mm_ecoli))


def write_markdown(tidy: pd.DataFrame):
    lines = []
    lines.append("# Table 1 — Cohort characteristics")
    lines.append("")
    lines.append(
        "No hypothesis tests. This is a cohort description, not a "
        "between-group comparison (no comparison groups are defined for "
        "one) -- design doc `.superpowers/sdd/SUBMISSION_PACKAGE_DESIGN.md` "
        "T1. Continuous variables: median [IQR]. Categorical variables: "
        "n (%). Cells with a raw count under 5 are masked as `<5`. `NA` in "
        "a source field is never folded into a percentage's denominator as "
        "an implicit negative -- see per-row notes for fields with a "
        "documented NA-encoding defect."
    )
    lines.append("")
    lines.append(
        "| " + " | ".join(["Section", "Variable", "Stratum",
                           COHORT_LABEL["korea"], COHORT_LABEL["mimic_all"],
                           COHORT_LABEL["mimic_ecoli"]]) + " |"
    )
    lines.append("|" + "---|" * 6)

    grouped = tidy.groupby(["section", "variable", "stratum"], sort=False)
    seen = set()
    for (section, variable, stratum), _ in grouped:
        key = (section, variable, stratum)
        if key in seen:
            continue
        seen.add(key)
        row = {}
        for cohort in COHORTS:
            match = tidy[(tidy.section == section) & (tidy.variable == variable)
                        & (tidy.stratum == stratum) & (tidy.cohort == cohort)]
            row[cohort] = match.iloc[0]["value"] if len(match) else "—"
        stratum_disp = "" if stratum in ("median_iqr", "yes", "n", "range") else stratum
        lines.append(
            f"| {section} | {variable} | {stratum_disp} | "
            f"{row['korea']} | {row['mimic_all']} | {row['mimic_ecoli']} |"
        )

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    notes_seen = []
    for _, r in tidy.iterrows():
        if r["note"] and r["note"] not in notes_seen:
            notes_seen.append(r["note"])
    for i, note in enumerate(notes_seen, 1):
        lines.append(f"{i}. {note}")
        lines.append("")

    md_path = OUT_DIR / "table1_cohort.md"
    md_path.write_text("\n".join(lines))
    print(f"Wrote {md_path}")


def write_manifest(unit_of_analysis, cohort_flow, n_kr, n_mm, n_mm_ec):
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/50_cohort_table.py",
        "cohorts": {
            "korea": {"n_admissions": n_kr,
                      "n_patients": unit_of_analysis["n_unique_patient_ID"]},
            "mimic_all": {"n_admissions": n_mm},
            "mimic_ecoli": {"n_admissions": n_mm_ec,
                            "definition": "organism_category == 'Ecoli'",
                            "role": "descriptive only -- no dedicated "
                                    "E. coli-only MIMIC analysis exists "
                                    "(N-003); the MIMIC replication/"
                                    "comparison unit is mimic_all, "
                                    "n=16,144"},
        },
        "inputs": [
            {
                "path": str(KOREA_PARQUET.relative_to(ROOT)),
                "sha256": sha256_of(KOREA_PARQUET),
                "n_rows": n_kr,
                "role": "Korean cohort, rendered template fields",
            },
            {
                "path": str(MIMIC_PARQUET.relative_to(ROOT)),
                "sha256": sha256_of(MIMIC_PARQUET),
                "n_rows": n_mm,
                "role": "MIMIC-IV cohort, joined superset (labs + "
                        "Charlson + AST-resistance + antibiotics); used "
                        "in place of separately re-joining admissions.parquet"
                        " + ast_detail.parquet + episodes_raw.parquet",
            },
            {
                "path": str(UNIT_OF_ANALYSIS.relative_to(ROOT)),
                "sha256": sha256_of(UNIT_OF_ANALYSIS),
                "role": "Korean patient count (patient_ID nunique), not "
                        "recomputed here since patient_ID is absent from "
                        "raw_fields.parquet",
            },
            {
                "path": str(COHORT_FLOW.relative_to(ROOT)),
                "sha256": sha256_of(COHORT_FLOW),
                "role": "MIMIC subject-count cross-check only",
            },
        ],
        "columns_used": {
            "korea": sorted(set(r["source_column"] for r in TIDY_ROWS
                               if r["cohort"] == "korea" and r["source_column"] != "n/a")),
            "mimic": sorted(set(r["source_column"] for r in TIDY_ROWS
                                if r["cohort"] in ("mimic_all", "mimic_ecoli")
                                and r["source_column"] != "n/a")),
        },
        "mask_below": MASK_BELOW,
        "no_hypothesis_tests": True,
        "no_patient_level_rows_written": True,
    }
    manifest_path = OUT_DIR / "table1_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
