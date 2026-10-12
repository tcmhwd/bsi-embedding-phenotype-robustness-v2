#!/usr/bin/env python3
"""Analysis A (post hoc follow-up), PHASE 1 + PHASE 2 (Addendum 1) -- harmonized field manifest.

PHASE 2 (PREREGISTRATION_A_ADDENDUM_1.md, FROZEN, sha256 in qc/PREREG_HASHES_A_ADDENDUM1.txt) changes three
decisions of the frozen text and this script implements them:
  1. `AST_detailed` (header line + all per-drug lines + the `o NA` placeholder) is REMOVED from every arm;
  2. the duplicate line `Microbes isolated in this episode` is REMOVED from every arm;
  3. `harmonized_keep_therapy` differs from the primary arm ONLY by the whole-admission antibiotics block:
     the per-episode efficacy lines are dropped in BOTH arms.
The Phase-1 outputs were moved (not deleted) to A_harmonized_template/superseded_phase1/.

Writes, under out/followup_analyses/A_harmonized_template/:
  harmonized_field_manifest.csv   one row per Korean rendered line-key class
                                  (plus section headers, anchors, and the
                                  MIMIC-only lines with no Korean counterpart)
  harmonized_field_manifest.json  provenance (git HEAD, input hashes, counts)

PREREGISTRATION_A.md (FROZEN, sha256 in qc/PREREG_HASHES_AC.txt) fixes the
field decisions this manifest records; the manifest is a TRANSCRIPTION of the
repository's own harmonization record (out/27_mimic/cohort/FIELDS_NOT_FILLED.md,
generated from scripts/27c_mimic_narratives.py:128-178) reconciled against
(i) the MIMIC renderer's actual lines (27c_mimic_narratives.py:63-125),
(ii) the machine-readable Korean field inventory
(out/20_onehot/features/raw_fields.parquet) and (iii) the rendered Korean
corpus of record itself (reports_pilot_pre, scanned line by line). It is NOT
derived from manuscript prose, and no field is mapped speculatively.

Integrity checks that abort the run on failure:
  * PREREGISTRATION_A.md sha256 equals the frozen hash.
  * Every cited file:line evidence entry contains its declared snippet.
  * Every line-key class observed in the corpus is covered by exactly one
    manifest row; every raw_fields.parquet column is assigned to a row.
  * The five retained summary-token categories equal the set MIMIC renders
    (27c_mimic_narratives.py:83,92-93); 5 + 14 = 19 categories.

The render stage (61_harmonized_render.py) derives every drop operation from
the `mechanism` / `retained_*` columns of THIS table, so the file the author
signs off is the file that drives the rendering.

Stages:   --stage manifest      (only stage; makes no API call, no GPU use,
                                 no embedding, no clustering)
Env:      /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime as dt
import hashlib
import importlib
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c21 = importlib.import_module("21_common")
c18 = c21.c18

REV = c21.REV
SF = REV / "out" / "followup_analyses"
A_DIR = SF / "A_harmonized_template"
PREREG = A_DIR / "PREREGISTRATION_A.md"
PREREG_HASHES = SF / "qc" / "PREREG_HASHES_AC.txt"
ADDENDUM = A_DIR / "PREREGISTRATION_A_ADDENDUM_1.md"
ADDENDUM_HASHES = SF / "qc" / "PREREG_HASHES_A_ADDENDUM1.txt"
FNF = REV / "out" / "27_mimic" / "cohort" / "FIELDS_NOT_FILLED.md"
RAW_FIELDS = REV / "out" / "20_onehot" / "features" / "raw_fields.parquet"
MANIFEST_CSV = A_DIR / "harmonized_field_manifest.csv"
MANIFEST_JSON = A_DIR / "harmonized_field_manifest.json"

ARMS_DET = ["korean_mimic_compatible", "harmonized_keep_therapy"]  # deterministic, manifest-driven
ARM_RANDOM = "thickness_matched_random"

# --- constants of the harmonization decision (PREREGISTRATION_A.md D-A-alpha) ---
RETAINED_SUMMARY = ["icu", "antiMRSAab_empiric", "antiVREab_empiric",
                    "antipseudoab_empiric", "carbaab_empiric"]
MIMIC_SPECTRUM = ["antiMRSAab_empiric", "antiVREab_empiric", "antipseudoab_empiric", "carbaab_empiric"]

# Korean line-key -> section (used only by norm_key below)
ST = "Summary tokens"
AD = "Administrative informations"
CM = "Underlying comorbidities"
TR = "Treatments during admission"
PR = "Procedures during the admission"
SO = "Sofa/qsofa"
PH = "Physiological and biological parameters"
BC = "Bacterial culture"
IT = "Infection treatment"

ANTIBIOTICS_SUBHEADER = "Antibiotics"
# The five Korean AST_detailed drugs that are NOT in MIMIC's shared 11-drug panel
# (27b_mimic_enrich.py:383-386); Korean display names as they occur in the corpus.
OFF_PANEL_AST = ["Cefotaxime", "Levofloxacin", "Tetracycline", "Ampicillin Sulbactam", "Amoxicillin clavulanate"]
PANEL_AST = ["Ampicillin", "Ceftriaxone", "Ceftazidime", "Ciprofloxacin", "Piperacillin Tazobactam",
             "Meropenem", "Ertapenem", "Sulfamethoxazole trimethoprim", "Vancomycin", "Oxacillin", "Penicillin G"]


# ----------------------------------------------------------------------------
# Line classification -- one canonical key per rendered line (used by BOTH the
# manifest scan and the render-stage QA so counts are comparable).
# ----------------------------------------------------------------------------
def norm_key(section: str | None, line: str) -> str:
    s = line.strip()
    ind = c21._indent_len(line)
    if c21._is_header(line):
        return f"HEADER::{s.strip('[]')}"
    if section == ST:
        return f"{ST}::{c21.summary_token_category(s)}"
    if c21._is_episode_anchor(line):
        return f"{section}::BSI_episode_num"
    if s == c21.INFECTION_TREATMENT_PREAMBLE:
        return f"{IT}::<preamble sentence>"
    if section == TR and s == ANTIBIOTICS_SUBHEADER:
        return f"{TR}::<Antibiotics subheader>"
    key = c21._line_key(s)
    if section == BC and ind == 2:
        return f"{BC}::AST_detailed/<NA placeholder>" if key is None else f"{BC}::AST_detailed/{key}"
    if section == IT and ind == 2:
        if key is None:
            return f"{IT}::Empiric/<NA placeholder>"
        if re.match(r"^Antibiotic_\d+_name$", key):
            return f"{IT}::Empiric/Antibiotic_N_name"
    if section == IT and ind == 4 and key and re.match(r"^Antibiotic_\d+_efficacy$", key):
        return f"{IT}::Empiric/Antibiotic_N_efficacy"
    if section == TR:
        if key is None and s.startswith("- No recorded antibiotics"):
            return f"{TR}::WholeAdm/<no-antibiotics placeholder>"
        if key and re.match(r"^Antibiotic_\d+_name$", key):
            return f"{TR}::WholeAdm/Antibiotic_N_name"
        if key and re.match(r"^Antibiotic_\d+_duration_days$", key):
            return f"{TR}::WholeAdm/Antibiotic_N_duration_days"
    if key is None:
        return f"{section}::<nokey:{s[:40]}>"
    return f"{section}::{key}"


def scan_text(text: str):
    """Yield (norm_key) for every non-blank line of a document."""
    sec = None
    for ln in text.split("\n"):
        if ln.strip() == "":
            continue
        if c21._is_header(ln):
            sec = ln.strip().strip("[]")
        yield norm_key(sec, ln)


# ----------------------------------------------------------------------------
# Corpus reading (I/O-latency-bound: 32,329 small files -> thread pool)
# ----------------------------------------------------------------------------
def n_workers() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))


def read_corpus(limit: int | None = None, with_hash: bool = False):
    """Corpus of record, in out_pre metadata row order (the order used by every
    embedding of this corpus: 18_common.load_metadata, embedding_index == row)."""
    meta = c18.load_metadata()
    if limit:
        meta = meta.iloc[:limit]
    aids = [int(a) for a in meta["admission_ID"]]
    nids = list(meta["newpatient_ID"])
    paths = [c18.report_path_for(n, a) for n, a in zip(nids, aids)]

    def _rd(p):
        b = p.read_bytes()
        return b.decode("utf-8", errors="replace"), (hashlib.sha256(b).hexdigest() if with_hash else None)

    with cf.ThreadPoolExecutor(max_workers=n_workers() * 2) as ex:
        res = list(ex.map(_rd, paths))
    texts = [r[0] for r in res]
    hashes = [r[1] for r in res]
    return aids, nids, texts, hashes


# ----------------------------------------------------------------------------
# Evidence: (relative path, line number, snippet that must occur on that line)
# ----------------------------------------------------------------------------
FNF_P = "out/27_mimic/cohort/FIELDS_NOT_FILLED.md"
C27 = "scripts/27c_mimic_narratives.py"
B27 = "scripts/27b_mimic_enrich.py"
A27 = "scripts/27a_mimic_cohort.py"
C21 = "scripts/21_common.py"
P20 = "scripts/20a_parse_template_fields.py"
PRE = "out/followup_analyses/A_harmonized_template/PREREGISTRATION_A.md"
ADD = "out/followup_analyses/A_harmonized_template/PREREGISTRATION_A_ADDENDUM_1.md"


def E(path, line, snip):
    return (path, line, snip)


EV_DA_ALPHA = E(PRE, 102, "retain the `[Summary tokens]` header")
EV_DA_BETA = E(PRE, 117, "drop the Korean date lines")
EV_DA_GAMMA = E(PRE, 125, "antibiotics items")
EV_DA_GAMMA_EFF = E(PRE, 127, "efficacy")
EV_DA_GLUCO = E(PRE, 131, "Glucocorticoid lines")
EV_DA_DELTA = E(PRE, 138, "remaining drops")
EV_DA_DELTA_RET = E(PRE, 145, "Retained, per")
EV_C27_ADMIN = E(C27, 66, "[Administrative]")
EV_ADD1 = E(ADD, 10, "AST_detailed")
EV_ADD2 = E(ADD, 16, "Microbes isolated in this episode")
EV_ADD3 = E(ADD, 20, "ONLY by the whole-admission antibiotics")
EV_C27_LABS = E(C27, 48, "LAB_FIELDS")
EV_C27_SPEC = E(C27, 38, "SPECTRUM_FIELDS")


def R(id_, kind, field, section, keys, mimic, in_m, sem, status, kmc, kkt, mech, src,
      reason="", confirm=False, confirm_reason="", cols=(), ev=(), notes="", in_k=True, all_arms=False):
    return dict(field_id=id_, row_kind=kind, korean_field=field, korean_section=section,
                korean_line_keys=list(keys), mimic_equivalent=mimic, rendered_in_korea=in_k,
                rendered_in_mimic=in_m, semantic_equivalence=sem, harmonization_status=status,
                retained_korean_mimic_compatible=kmc, retained_harmonized_keep_therapy=kkt,
                mechanism=mech, decision_source=src, reason_for_exclusion=reason,
                author_confirm=confirm, author_confirm_reason=confirm_reason,
                raw_fields_columns=list(cols), evidence=list(ev), notes=notes,
                all_arms_removal=all_arms)


X = "exact/near-direct"
M = "mapped/coarsened"
U = "unavailable"
C = "meaning_changed"
Z = "unsafe_to_harmonize"

ROWS: list[dict] = []
ROWS += [
    # ============================ [Summary tokens] =============================
    R("S00", "section_header", "[Summary tokens]", ST, [f"HEADER::{ST}"],
      "none (MIMIC renders no summary block; its icu / spectrum-flag lines sit under [Organ support] / [Spectrum flags])",
      False, "Header kept only as the container of the five retained lines (D-A-alpha).", U, True, True,
      "section_header", "prereg D-A-alpha", ev=[EV_DA_ALPHA, E(C27, 82, "[Organ support]"), E(C27, 91, "[Spectrum flags]")]),
    R("S01", "field", "icu (summary token)", ST, [f"{ST}::icu"], "icu : (27c:83, [Organ support])", True,
      "Korean summary_token 'icu_yes/NA' (yes/NA, zero explicit 'no') vs MIMIC any-icustays 'yes'/'no'.", C, True, True,
      "retained", "prereg D-A-alpha; author-approved (AUTHOR_DECISIONS_2026-09-21)",
      cols=["summary__icu"],
      ev=[EV_DA_ALPHA, E(C27, 83, "icu :"), E(FNF_P, 32, "icu")],
      notes="Documented change of meaning (FNF:32). In the Korean template this field is rendered ONLY inside [Summary tokens]."),
]
for cat in MIMIC_SPECTRUM:
    ROWS.append(R(
        f"S02_{cat}", "field", f"{cat} (summary token)", ST, [f"{ST}::{cat}"],
        f"{cat} : (27c:92-93, [Spectrum flags])", True,
        "Same concept (empiric regimen covers spectrum X). Korean value from the data provider's flag; MIMIC value by "
        "keyword match over prescriptions.drug within +-72 h of the index culture (27b:293-300). Not validated against each other.",
        C, True, True, "retained", "prereg D-A-alpha; author-approved (AUTHOR_DECISIONS_2026-09-21)",
        cols=[f"summary__{cat}"],
        ev=[EV_DA_ALPHA, EV_C27_SPEC, E(C27, 92, "SPECTRUM_FIELDS"), E(B27, 293, "SPECTRUM_KEYWORDS"),
            E(C21, 367, "AGGREGATE OVER ALL")],
        notes="Korean definition per AUTHOR_DECISIONS.md D4 (antiXab_empiric from the provider data dictionary); "
              "21_common.py:367-380 records these tokens as aggregated over all episodes of the admission. "
              "FNF does not list these four; they are retained because D-A-alpha names them and MIMIC renders them."))
_SUM_DROP = [
    ("admission_cluster", U, "Pre-computed cluster identifier from a prior clustering of the Korean corpus; no MIMIC counterpart. "
     "It is BGE-M3's modal top field (PREREGISTRATION_A D-A-alpha, disclosed consequence): dropping it makes the harmonized arm "
     "a circularity control for BGE-M3.", ["summary__admission_cluster"]),
    ("empiric_cluster", U, "Pre-computed cluster identifier (empiric-therapy clustering); no MIMIC counterpart.", ["summary__empiric_cluster"]),
    ("cre_colonization", U, "FNF:21 (VRE/CRE colonization dropped: presence-only registers).", ["summary__cre_colonization"]),
    ("vre_colonization", U, "FNF:21 (VRE/CRE colonization dropped: presence-only registers).", ["summary__vre_colonization"]),
    ("empiric_tt_adapted", U, "No MIMIC counterpart (27c renders no such line). Not named in FNF; dropped per prereg D-A-alpha/D-A-delta.", ["summary__empiric_tt_adapted"]),
    ("sepsis_2", U, "FNF:19 (sepsis_2 requires ICU-only vitals in MIMIC).", ["summary__sepsis_2"]),
    ("sepsis_quicksofa_2", U, "FNF:19 (sepsis_quicksofa_2 requires ICU-only vitals in MIMIC).", ["summary__sepsis_quicksofa_2"]),
]
for cat, st, why, cols in _SUM_DROP:
    ev = [EV_DA_ALPHA, EV_DA_DELTA]
    if cat in ("cre_colonization", "vre_colonization"):
        ev.append(E(FNF_P, 21, "colonization"))
    if cat.startswith("sepsis"):
        ev.append(E(FNF_P, 19, cat))
    ROWS.append(R(f"S03_{cat}", "field", f"{cat} (summary token)", ST, [f"{ST}::{cat}"],
                  "none", False, "No MIMIC line for this concept.", st, False, False, "summary_category",
                  "prereg D-A-alpha (14 dropped)", reason=why, cols=cols, ev=ev))
for cat, name in [("index_bsi_3gcr", "3GCR"), ("index_bsi_carbar", "CarbaR"), ("index_bsi_ceftriaxoner", "CeftriaxoneR"),
                  ("index_bsi_crab", "CRAB"), ("index_bsi_mrsa", "MRSA"), ("index_bsi_vre", "VRE"),
                  ("index_bsi_polymicrobial", "BSI_polymicrobial / Polymicrobial")]:
    ROWS.append(R(
        f"S04_{cat}", "field", f"{cat} (summary token)", ST, [f"{ST}::{cat}"],
        "none as a summary line (MIMIC renders the same concept as a line in [Bacterial culture], 27c:105-107)", False,
        f"Summary-token duplicate of the index-episode {name} line; the content survives in the retained Korean line.", U,
        False, False, "summary_category", "prereg D-A-alpha (14 dropped)",
        reason=f"D-A-alpha keeps exactly five summary lines; this token is a duplicate of the index-episode {name} line "
               "(20a docstring: index-episode flags equal the summary tokens) which is retained in the body.",
        confirm=(cat == "index_bsi_polymicrobial"),
        confirm_reason=("PREREGISTRATION_A is internally inconsistent: D-A-alpha (line 102) drops all but five summary lines, "
                        "but D-A-delta (line 148) lists `summary__index_bsi_polymicrobial` under 'Retained'. Implemented D-A-alpha "
                        "(explicit, numbered, enumerates the 14 dropped). Polymicrobial content is still present via the retained "
                        "BSI_polymicrobial line in [Infection treatment].") if cat == "index_bsi_polymicrobial" else "",
        cols=[f"summary__{cat}"],
        ev=[EV_DA_ALPHA, E(C27, 105, "Polymicrobial"), E(P20, 257, "parse_bacterial_culture_index_episode")]
           + ([E(PRE, 148, "index_bsi_polymicrobial")] if cat == "index_bsi_polymicrobial" else [])))

ROWS += [
    # ===================== [Administrative informations] ======================
    R("A00", "section_header", "[Administrative informations]", AD, [f"HEADER::{AD}"], "[Administrative] (27c:66)", True,
      "Same section type; header text differs ('Administrative informations' vs 'Administrative').", M, True, True,
      "section_header", "prereg D-A-delta (Korean syntax/section order preserved)", ev=[EV_C27_ADMIN]),
    R("A01", "field", "Patient ID", AD, [f"{AD}::Patient ID"], "none (omitted from text; file name only)", False,
      "Identifier; not rendered in MIMIC text (27c docstring).", U, False, False, "remove_items", "FNF:13",
      reason="FNF:13 - 'correct choice for both cohorts is to omit it from the TEXT'. High-cardinality nuisance token.",
      confirm=True,
      confirm_reason="Not enumerated in PREREGISTRATION_A D-A-delta (which says its drops are 'transcribed from FIELDS_NOT_FILLED.md:11-25' "
                     "but omits FNF:13). Dropped here because the manifest rule is 'MIMIC-renderable lines only' and FNF:13 states it is omitted "
                     "from the text in both cohorts. Retaining it would leave a unique per-patient token in the Korean text only.",
      ev=[E(FNF_P, 13, "Patient pseudo-ID"), E(C21, 176, "PATIENT_ID_KEY")]),
    R("A02", "field", "Age", AD, [f"{AD}::Age"], "Age : (27c:67)", True,
      "Korean exact age; MIMIC anchor_age + year offset, capped at 91 for >89 (de-identification).", C, True, True,
      "retained", "prereg D-A-delta; FNF:31", cols=["admin__age"], ev=[E(C27, 67, "Age :"), E(FNF_P, 31, "Age"), EV_DA_DELTA_RET]),
    R("A03", "field", "Sex", AD, [f"{AD}::Sex"], "Sex : (27c:68)", True, "Same concept and coding (M/F).", X, True, True,
      "retained", "prereg D-A-delta; FNF:40", cols=["admin__sex"], ev=[E(C27, 68, "Sex :"), E(FNF_P, 40, "Sex")]),
    R("A04", "field", "Admission Date", AD, [f"{AD}::Admission Date"], "none (MIMIC dates are patient-shifted; never rendered)", False,
      "Absolute calendar date vs no date.", U, False, False, "remove_items", "prereg D-A-beta; FNF:14",
      reason="FNF:14 no absolute calendar dates in MIMIC; D-A-beta drops the Korean date lines, no timing substitute.",
      cols=["admin__admission_date_raw", "admin__admission_year", "admin__admission_month"],
      ev=[EV_DA_BETA, E(FNF_P, 14, "Admission Date"), E(C21, 178, "ADMISSION_DATE_KEY")]),
    R("A05", "field", "Entry mode", AD, [f"{AD}::Entry mode"], "Entry mode : (27c:69)", True,
      "Korean codes I/E/O (29,266 / 2,809 / 254 in raw_fields) vs MIMIC 'ED' if edregtime present else 'Direct' (27a:397). "
      "Different vocabularies; no crosswalk documented anywhere; not mentioned in FNF.", Z, True, True,
      "retained", "prereg D-A-delta (retained)",
      confirm=True,
      confirm_reason="Equivalence is undocumented (FNF silent; value vocabularies differ). Retained only because PREREGISTRATION_A D-A-delta "
                     "(line 145) names admin__entry_mode as retained; under the 'uncertain -> unsafe_to_harmonize -> exclude' rule it would be dropped.",
      cols=["admin__entry_mode"], ev=[E(C27, 69, "Entry mode :"), E(A27, 397, "entry_mode"), EV_DA_DELTA_RET]),
    R("A06", "field", "Treatment departement", AD, [f"{AD}::{c21.DEPARTMENT_KEY}"], "none", False,
      "Korean ~45 named departments; no shared taxonomy.", U, False, False, "remove_items", "FNF:15; prereg D-A-delta",
      reason="FNF:15 no shared taxonomy; inventing a crosswalk would be an unvalidated mapping. This is 3-small's dominant field (prereg D-A-delta).",
      cols=["admin__department"], ev=[E(FNF_P, 15, "Treatment department"), E(C21, 177, "DEPARTMENT_KEY"), EV_DA_DELTA]),
    # ======================= [Underlying comorbidities] ========================
    R("C00", "section_header", "[Underlying comorbidities]", CM, [f"HEADER::{CM}"], "[Comorbidities] (27c:76)", True,
      "Same section type; header text differs.", M, True, True, "section_header", "prereg D-A-delta",
      ev=[E(C27, 76, "[Comorbidities]")]),
    R("C01", "field", "Infectious comorbidities as defined through ICD-10 coding during the admission", CM,
      [f"{CM}::Infectious comorbidities as defined through ICD-10 coding during the admission"],
      "Source of infection category : (27c:77)", True,
      "Korean: multi-label list from the icd_categorized_onlyinfections grouper. MIMIC: single label from an independently built "
      "8-level keyword/ICD-prefix heuristic, 'NOT validated against or matched to the Korean grouper ... not a validated harmonized variable'.",
      C, True, True, "retained", "prereg D-A-delta; FNF:34",
      confirm=True,
      confirm_reason="FNF:34 itself states the MIMIC counterpart is an unvalidated structural placeholder. Retained because D-A-delta names "
                     "comorbid__infectious as retained; the pairing is by section slot, not by validated equivalence.",
      cols=["comorbid__infectious"], ev=[E(C27, 77, "Source of infection category"), E(FNF_P, 34, "Source of infection category"),
                                         E(B27, 245, "_source_category"), EV_DA_DELTA_RET],
      notes="Korean values are unchanged in the harmonized arm; only presence is decided."),
    R("C02", "field", "All ICD10 codes during the admission stay whether infectious or not", CM,
      [f"{CM}::All ICD10 codes during the admission stay whether infectious or not"], "none (replaced by the 8-level source category)", False,
      "Raw ICD code list; MIMIC mixes ICD-9/ICD-10.", U, False, False, "remove_items", "FNF:16; prereg D-A-delta",
      reason="FNF:16 mixed ICD-9/10 vocabularies would be a cohort marker.", cols=["comorbid__icd10"],
      ev=[E(FNF_P, 16, "ICD10"), EV_DA_DELTA]),
    R("C03", "field", "All comorbidities as classified for the Charlson Comorbidities Index calculation", CM,
      [f"{CM}::All comorbidities as classified for the Charlson Comorbidities Index calculation"], "Charlson category : (27c:78)", True,
      "Korean: multi-label list of Charlson categories (e.g. ['other','cancer_lymphoma_leukemia']). MIMIC: single 4-level label "
      "(none/mild/moderate/severe), 'NOT matched to the Korean charlson_category_withcancermerged label set'.",
      C, True, True, "retained", "prereg D-A-delta; FNF:33",
      confirm=True,
      confirm_reason="Label vocabularies differ (multi-label categories vs 4-level severity); FNF:33 documents the change of meaning. Retained per D-A-delta.",
      cols=["comorbid__charlson_class"], ev=[E(C27, 78, "Charlson category"), E(FNF_P, 33, "Charlson"), EV_DA_DELTA_RET]),
    R("C04", "field", "Charlson score", CM, [f"{CM}::Charlson score"], "Charlson score : (27c:79)", True,
      "Numeric score, computed differently (Korean index-window vs MIMIC whole-admission discharge codes, Quan/Deyo-Quan).",
      C, True, True, "retained", "prereg D-A-delta; FNF:33", cols=["comorbid__charlson_score"],
      ev=[E(C27, 79, "Charlson score"), E(FNF_P, 33, "Charlson"), EV_DA_DELTA_RET]),
    # ===================== [Treatments during admission] ======================
    R("T00", "section_header", "[Treatments during admission]", TR, [f"HEADER::{TR}"], "none (MIMIC has no whole-admission treatment section)", False,
      "Section holds the whole-admission antibiotics block (dropped in the primary arm) and the glucocorticoid lines (dropped in both arms).",
      U, False, True, "section_header", "prereg D-A-gamma (Rule R: header removed only when the section is empty)",
      reason="Primary arm: section is empty after the whole-admission antibiotics block and the glucocorticoid lines are removed.",
      confirm=True,
      confirm_reason="The prereg does not say what happens to the section header / bare 'Antibiotics' subheader once the block is gone. "
                     "Rule R (implemented): a section header is removed iff the section is left with no content line; prereg D-A-delta already "
                     "removes the whole [Sofa/qsofa] section. Alternative (keep an empty header + 'Antibiotics' subheader, as the existing "
                     "`no_therapy` variant does) differs by 2 lines/document in the primary arm.",
      ev=[EV_DA_GAMMA, E(C21, 436, "Antibiotics")]),
    R("T01", "anchor", "Antibiotics (bare subheader)", TR, [f"{TR}::<Antibiotics subheader>"], "none", False,
      "Label with no value; only meaningful with the whole-admission block beneath it.", U, False, True, "admission_antibiotic_walk",
      "prereg D-A-gamma (Rule R)", reason="Orphan subheader once its block is removed.", ev=[E(C21, 436, "Antibiotics")]),
    R("T02", "field", "Antibiotic_N_name (whole-admission antibiotics list)", TR, [f"{TR}::WholeAdm/Antibiotic_N_name"], "none", False,
      "Whole-admission antibiotic identity (FNF:23: 'most exposed to US/Korean formulary-naming mismatch').", U, False, True,
      "admission_antibiotic_walk", "prereg D-A-gamma (first clause); FNF:23",
      reason="FNF:23 drops the whole-admission antibiotic summary fields. THIS IS THE BLOCK THAT CREATES H-A3: first_antibiotic (Qwen3's dominant field) "
             "is parsed from exactly these lines (40_field_dominance.py:185-196). Retained in the companion arm harmonized_keep_therapy.",
      cols=["tx__admission_antibiotics"], ev=[EV_DA_GAMMA, E(FNF_P, 23, "ab_admission"), E(P20, 188, "parse_treatments"),
                                              E("scripts/40_field_dominance.py", 196, "first_antibiotic")]),
    R("T03", "field", "Antibiotic_N_duration_days (whole-admission)", TR, [f"{TR}::WholeAdm/Antibiotic_N_duration_days"], "none", False,
      "Continuation line of the whole-admission antibiotic bullet.", U, False, True, "admission_antibiotic_walk",
      "prereg D-A-gamma (first clause); FNF:23", reason="Part of the whole-admission antibiotics block (FNF:23).",
      cols=["tx__total_antibiotic_duration_days"], ev=[EV_DA_GAMMA, E(FNF_P, 23, "ab_admission")]),
    R("T04", "field", "- No recorded antibiotics during admission (NA)", TR, [f"{TR}::WholeAdm/<no-antibiotics placeholder>"], "none", False,
      "Placeholder of the whole-admission block.", U, False, True, "admission_antibiotic_walk",
      "prereg D-A-gamma (first clause)", reason="Placeholder of the whole-admission antibiotics block.", ev=[EV_DA_GAMMA]),
]
for nm, col in [("Short_acting_glucocorticoids", "tx__gluco_short"), ("Intermediate_acting_glucocorticoids", "tx__gluco_intermediate"),
                ("Long_acting_glucocorticoids", "tx__gluco_long")]:
    ROWS.append(R(f"T05_{nm}", "field", nm, TR, [f"{TR}::{nm}"], "none", False, "MIMIC renders no glucocorticoid line (27c:63-125).", U,
                  False, False, "remove_items", "prereg D-A-gamma (glucocorticoid sentence)",
                  reason="MIMIC renders none (prereg D-A-gamma: 'so tx__gluco_{short,intermediate,long} are dropped separately'). Not listed in FNF.",
                  cols=[col], ev=[EV_DA_GLUCO]))
ROWS += [
    # ===================== [Procedures during the admission] ==================
    R("P00", "section_header", "[Procedures during the admission]", PR, [f"HEADER::{PR}"], "none", False,
      "Section is empty once its three ICU-only procedure lines are dropped.", U, False, False, "section_header",
      "FNF:18 + Rule R", reason="All three lines dropped (FNF:18); empty section header removed (Rule R).",
      confirm=True, confirm_reason="See T00: header removal on an emptied section is an implementation rule not stated in the prereg.",
      ev=[E(FNF_P, 18, "ventilator")]),
]
for nm, col in [("central_orarterial_line", "proc__central_line"), ("urinary_orrenal_catheter", "proc__urinary_catheter"),
                ("ventilator", "proc__ventilator")]:
    ROWS.append(R(f"P01_{nm}", "field", nm, PR, [f"{PR}::{nm}"], "none", False, "ICU-only ascertainment in MIMIC (FNF:18).", U,
                  False, False, "remove_items", "FNF:18; prereg D-A-delta", reason="FNF:18 ICU-only ascertainment (procedureevents/chartevents).",
                  cols=[col], ev=[E(FNF_P, 18, "ventilator"), EV_DA_DELTA]))
ROWS.append(R("O00", "section_header", "[Sofa/qsofa]", SO, [f"HEADER::{SO}"], "none", False, "Whole section dropped.", U, False, False,
              "section_header", "prereg D-A-delta ('the whole [Sofa/qsofa] section'); FNF:19",
              reason="All four lines dropped (FNF:19: SOFA/qSOFA need ICU-only vitals).", ev=[E(PRE, 142, "Sofa/qsofa"), E(FNF_P, 19, "sofa_score_2")]))
for nm, col in [("sofa_score_2", "sofa__sofa_score_2"), ("sepsis_2", None), ("quick_sofa_score_2", "sofa__quick_sofa_score_2"),
                ("sepsis_quicksofa_2", None)]:
    ROWS.append(R(f"O01_{nm}", "field", nm, SO, [f"{SO}::{nm}"], "none", False, "Requires ICU-only vitals in MIMIC (FNF:19).", U, False, False,
                  "remove_items", "FNF:19; prereg D-A-delta", reason="FNF:19 SOFA/qSOFA computable for ~45% of MIMIC (ICU stratum) only.",
                  cols=[col] if col else [], ev=[E(FNF_P, 19, nm if nm != "sepsis_2" else "sepsis_2"), EV_DA_DELTA],
                  notes=("Duplicates the summary token of the same name (20a docstring); the raw_fields column is summary__" + nm) if col is None else ""))
ROWS.append(R("H00", "section_header", "[Physiological and biological parameters]", PH, [f"HEADER::{PH}"],
              "[Physiological and biological parameters] (27c:86)", True, "Identical header text.", X, True, True, "section_header",
              "prereg D-A-delta", ev=[E(C27, 86, "[Physiological and biological parameters]")]))
for nm in ["WBC_Count", "T_Bilirubin", "Creatinine", "Hemoglobin", "Albumin", "PLT_Count"]:
    ROWS.append(R(f"H01_{nm}", "field", nm, PH, [f"{PH}::{nm}"], f"{nm} : (27c:87-88, LAB_FIELDS 27c:48)", True,
                  "Same laboratory concept; MIMIC value = nearest lab within +-24 h of the index culture (labevents); Korean index-window value.",
                  X, True, True, "retained", "prereg D-A-delta; FNF:40", cols=[f"physio__{nm}"],
                  ev=[EV_C27_LABS, E(FNF_P, 40, "Sex, WBC_Count")]))
for nm, why, ref in [("SBP", "FNF:20 ICU-only vitals", 20), ("DBP", "FNF:20 ICU-only vitals", 20), ("MBP", "FNF:20 ICU-only vitals", 20),
                     ("RR", "FNF:20 ICU-only vitals", 20), ("Body_temperature", "FNF:20 ICU-only vitals", 20),
                     ("PR", "FNF:20 ICU-only vitals", 20), ("GCS_2", "FNF:20 ICU-only vitals", 20),
                     ("Vasopressin_any", "FNF:17 ICU-module-only ascertainment", 17), ("CRP", "FNF:25 75 pp availability gap", 25)]:
    ROWS.append(R(f"H02_{nm}", "field", nm, PH, [f"{PH}::{nm}"], "none", False, "Not rendered in MIMIC.", U, False, False, "remove_items",
                  f"FNF:{ref}; prereg D-A-delta", reason=why, cols=[f"physio__{nm}"], ev=[E(FNF_P, ref, nm.split('_')[0] if nm != "Body_temperature" else "Body_temperature"), EV_DA_DELTA]))
for nm in ["AST", "ALT", "pO2_Arterial"]:
    ROWS.append(R(f"H03_{nm}", "field", nm, PH, [f"{PH}::{nm}"], "none (not in MIMIC LAB_FIELDS, 27c:48)", False, "Not rendered in MIMIC.", U,
                  False, False, "remove_items", "prereg D-A-delta; absent from 27c LAB_FIELDS",
                  reason="Not in MIMIC's rendered lab set (27c:48). Not enumerated in FNF; named as dropped in prereg D-A-delta.",
                  cols=[f"physio__{nm}"], ev=[EV_C27_LABS, EV_DA_DELTA]))
ROWS += [
    # ============================ [Bacterial culture] =========================
    R("B00", "section_header", "[Bacterial culture]", BC, [f"HEADER::{BC}"], "[Bacterial culture] (27c:96)", True, "Identical header text.", X,
      True, True, "section_header", "prereg D-A-delta", ev=[E(C27, 96, "[Bacterial culture]")]),
    R("B01", "anchor", "BSI_episode_num (episode anchor, [Bacterial culture])", BC, [f"{BC}::BSI_episode_num"], "Episode : 1 (27c:102, literal)", True,
      "Korean value is a database episode identifier (e.g. 32893); MIMIC prints the literal 1.", C, True, True, "retained",
      "prereg D-A-delta ('BSI_episode_num / episode count'); anchor kept by every existing variant (21_common.py:181)",
      confirm=True, confirm_reason="Low priority: anchor value is a global episode ID in Korea vs the literal '1' in MIMIC. Kept as a structural anchor, as in all 20 existing variants.",
      ev=[E(C27, 102, "Episode : 1"), E(C21, 181, "EPISODE_ANCHOR_PREFIX"), EV_DA_DELTA_RET]),
    R("B02", "field", "Test_date", BC, [f"{BC}::Test_date"], "Timing : hospital day N (27c:103)", True,
      "Absolute calendar date (Korea) vs relative hospital day (MIMIC). Any alignment needs a NEW derived Korean field.", Z, False, False,
      "remove_items", "prereg D-A-beta", reason="D-A-beta: drop the Korean date lines, render NO timing substitute (deriving a Korean hospital-day would be a "
                                                "new computed field = the speculative mapping the spec forbids).",
      cols=[], ev=[EV_DA_BETA, E(C27, 103, "Timing"), E(FNF_P, 14, "Admission Date"), E(C21, 179, "TEST_DATE_KEY")],
      notes="MIMIC's `Timing` line is therefore rendered in MIMIC but in NEITHER Korean arm (see M02)."),
    R("B03", "field", "BSI_episode_short_clean", BC, [f"{BC}::BSI_episode_short_clean"], "Organism summary : (27c:104)", True,
      "Same concept; MIMIC organism_category is mapped ONTO the Korean 19-category taxonomy (27a:119-162).", M, True, True, "retained",
      "prereg D-A-delta; FNF:35", cols=["culture__index_organism"], ev=[E(C27, 104, "Organism summary"), E(FNF_P, 35, "Organism summary"), EV_DA_DELTA_RET]),
]
for nm in ["3GCR", "CarbaR", "MRSA", "VRE", "CeftriaxoneR", "CRAB"]:
    ROWS.append(R(f"B04_{nm}", "field", nm, BC, [f"{BC}::{nm}"], f"{nm} : (27c:106-107)", True,
                  "Same flag; MIMIC derived from microbiologyevents AST over the shared drug panel (Ertapenem tested in ~0.1% -> CarbaR mostly via Meropenem/Imipenem).",
                  M, True, True, "retained", "prereg D-A-delta; FNF:35", cols=[],
                  ev=[E(C27, 49, "RESISTANCE_FLAGS"), E(FNF_P, 35, "3GCR"), EV_DA_DELTA_RET],
                  notes="raw_fields has these only as the summary__index_bsi_* duplicates."))
ROWS += [
    R("B05", "field", "AST_summary_tokens", BC, [f"{BC}::AST_summary_tokens"], "none as a line (the six flags are rendered individually, 27c:106-107)", False,
      "Concatenated re-encoding of the six flag lines ('3GCR_no ; CarbaR_no ; ...').", U, True, True, "retained",
      "prereg D-A-delta ('AST_summary_tokens rendered in [Bacterial culture]')",
      confirm=True, confirm_reason="MIMIC renders no AST_summary_tokens line; retained because D-A-delta names it. It is a deterministic re-encoding of the six retained flags "
                                   "(equality is verified over the whole corpus by 61 qa: see render_qa.json 'derivation_checks').",
      ev=[E(PRE, 147, "AST_summary_tokens"), E(C27, 106, "RESISTANCE_FLAGS")]),
    R("B06", "field", "AST_detailed (header line; the item removal takes its whole indented block)", BC, [f"{BC}::AST_detailed"],
      "AST_detailed : + a single bullet `o NA` (27c:108-110)", True,
      "Key exists in both, but MIMIC renders the single bullet `o NA` in 16,144 of 16,144 documents (Addendum 1 decision 1), i.e. the content is not genuinely available; "
      "the Korean block carries real per-drug results.", U, False, False, "remove_items",
      "Addendum 1 decision 1 (changes frozen D-A-delta 'retain whole')",
      reason="Addendum 1 decision 1: retaining it would leave the 'MIMIC-fillable' Korean template richer in organism information than MIMIC itself.",
      ev=[EV_ADD1, E(C27, 110, "AST_detailed"), E(PRE, 147, "AST_detailed")], all_arms=True,
      notes="Superseded Phase-1 status: exact/near-direct, retained. Removed from the primary, keep_therapy AND random arms."),
    R("B07", "field", "AST_detailed drug lines: " + ", ".join(PANEL_AST) + " (MIMIC shared panel)", BC,
      [f"{BC}::AST_detailed/{d}" for d in PANEL_AST], "  o <drug : result> over SHARED_PANEL_DISPLAY (27c:108-112; 27b:383-386) -- rendered as `o NA` only", True,
      "Same drugs are in MIMIC's panel by design, but the MIMIC block is `o NA` in 16,144/16,144 documents (Addendum 1 decision 1).", U, False, False,
      "block_continuation", "Addendum 1 decision 1", reason="Continuation lines of the removed AST_detailed item (B06).",
      ev=[EV_ADD1, E(B27, 383, "SHARED_PANEL_DISPLAY")], all_arms=True, notes="Superseded Phase-1 status: mapped/coarsened, retained."),
    R("B08", "field", "AST_detailed drug lines OUTSIDE MIMIC's panel: " + ", ".join(OFF_PANEL_AST), BC,
      [f"{BC}::AST_detailed/{d}" for d in OFF_PANEL_AST], "none (MIMIC renders only the 11 shared-panel drugs, 27b:383-386)", False,
      "Drug-level lines that MIMIC cannot render.", U, False, False, "block_continuation", "Addendum 1 decision 1",
      reason="Continuation lines of the removed AST_detailed item (B06); also not in MIMIC's 11-drug panel.",
      ev=[EV_ADD1, E(B27, 383, "SHARED_PANEL_DISPLAY")], all_arms=True, notes="Superseded Phase-1 flag: TOP PRIORITY confirm; settled by Addendum 1 decision 1."),
    R("B09", "field", "AST_detailed: 'o NA (no AST for selected antibiotics)'", BC, [f"{BC}::AST_detailed/<NA placeholder>"], "  o NA (27c:108-109)", True,
      "Same meaning (no AST); Korean text carries a parenthetical. Removed with the block (Addendum 1 decision 1).", U, False, False, "block_continuation", "Addendum 1 decision 1",
      reason="Continuation line of the removed AST_detailed item (B06).", ev=[EV_ADD1, E(C27, 108, "detail")], all_arms=True),
    # =========================== [Infection treatment] ========================
    R("I00", "section_header", "[Infection treatment]", IT, [f"HEADER::{IT}"], "[Infection treatment] (27c:115)", True, "Identical header text.", X, True, True,
      "section_header", "prereg D-A-gamma (empiric names retained)", ev=[E(C27, 115, "[Infection treatment]")]),
    R("I01", "anchor", "For BSI we organise the data in episodes: (preamble sentence)", IT, [f"{IT}::<preamble sentence>"], "none", False,
      "Fixed template sentence.", U, True, True, "retained", "structural anchor, kept by every existing variant (21_common.py:182)",
      ev=[E(C21, 182, "INFECTION_TREATMENT_PREAMBLE")]),
    R("I02", "anchor", "BSI_episode_num (episode anchor, [Infection treatment])", IT, [f"{IT}::BSI_episode_num"], "none (MIMIC has one implicit episode)", False,
      "See B01.", C, True, True, "retained", "prereg D-A-delta; structural anchor", ev=[E(C21, 181, "EPISODE_ANCHOR_PREFIX")]),
    R("I03", "field", "BSI_polymicrobial", IT, [f"{IT}::BSI_polymicrobial"], "Polymicrobial : (27c:105, in [Bacterial culture])", True,
      "Same concept; section placement differs (Korean [Infection treatment] vs MIMIC [Bacterial culture]).", M, True, True, "retained",
      "prereg D-A-delta (polymicrobial retained); FNF:35", ev=[E(C27, 105, "Polymicrobial"), E(FNF_P, 35, "Polymicrobial")]),
    R("I04", "field", "Microbes isolated in this episode", IT, [f"{IT}::Microbes isolated in this episode"], "none in [Infection treatment] (organism printed once, 27c:104)", False,
      "Second organism line per episode (ORGANISM_KEYS = both lines, 21_common.py:170); equal to BSI_episode_short_clean in 36,610 of 36,610 episodes (Addendum 1); MIMIC prints the organism once.",
      U, False, False, "remove_items", "Addendum 1 decision 2 (not mentioned in the frozen text)",
      reason="Addendum 1 decision 2: exact duplicate of BSI_episode_short_clean, so Korean text names the organism twice per episode against once in MIMIC.",
      ev=[EV_ADD2, E(C21, 170, "ORGANISM_KEYS"), E(C27, 104, "Organism summary")], all_arms=True,
      notes="Superseded Phase-1 status: retained by default. BSI_episode_short_clean (the field used in attribution) is kept."),
    R("I05", "field", "Empiric treatment (block header line)", IT, [f"{IT}::Empiric treatment"], "none (MIMIC lists empiric drugs directly under the section, 27c:117-123)", False,
      "Container line for the per-episode empiric drug list.", U, True, True, "retained", "prereg D-A-gamma ('retain the [Infection treatment] per-episode empiric names')",
      ev=[E(PRE, 128, "`[Infection treatment]` per-episode"), E(C27, 117, "emp")]),
    R("I06", "field", "Antibiotic_N_name (per-episode empiric)", IT, [f"{IT}::Empiric/Antibiotic_N_name"], "Antibiotic_{i}_name : (27c:121)", True,
      "Same concept (empiric regimen names); MIMIC = free-text keyword-matched prescriptions in [index-72h, index+72h] (<=5), Korean = structured per-episode list.",
      C, True, True, "retained", "prereg D-A-gamma; FNF:36", ev=[E(C27, 121, "Antibiotic_{i}_name"), E(FNF_P, 36, "Empiric antibiotics"), E(PRE, 128, "`[Infection treatment]` per-episode")]),
    R("I07", "field", "Antibiotic_N_efficacy (per-episode)", IT, [f"{IT}::Empiric/Antibiotic_N_efficacy"], "none ('No per-drug Efficacy line is rendered', FNF:36)", False,
      "Per-drug efficacy (values yes/no/NA in Korean).", U, False, False, "efficacy_subline",
      "prereg D-A-gamma (efficacy sub-lines dropped); Addendum 1 decision 3 (dropped in BOTH arms)",
      reason="FNF:36: MIMIC free-text drug strings cannot be matched to AST reliably enough to render an efficacy line. Addendum 1 decision 3: dropped in the primary AND the keep_therapy arm, "
             "so the two arms differ only by the whole-admission antibiotics block.",
      ev=[EV_ADD3, E(PRE, 127, "efficacy"), E(FNF_P, 36, "Efficacy")],
      notes="Superseded Phase-1 implementation: retained in keep_therapy. The random arm is field-agnostic for this line (it may remove or keep it at random)."),
    R("I08", "field", "Empiric treatment: 'o NA' placeholder", IT, [f"{IT}::Empiric/<NA placeholder>"], "Antibiotic_1_name : NA (27c:123)", True,
      "No empiric antibiotic recorded.", X, True, True, "retained", "prereg D-A-gamma", ev=[E(C27, 123, "Antibiotic_1_name : NA")]),
]

# Rows for content that exists ONLY on the other side (no Korean line to retain/drop).
ROWS += [
    R("M01", "mimic_only", "(no Korean line) Fungemia", "(none)", [], "[Fungemia] / Fungemia : (27c:72-73)", True,
      "Korean corpus of record renders NO Fungemia line (21_common.py:78: 'no Candida/Fungemia content anywhere in this cohort (checked)'; confirmed by the full-corpus scan of this stage).",
      U, None, None, "not_rendered", "FNF:41-43 says 'kept' for MIMIC; prereg D-A-delta lists 'the Fungemia binary' as retained",
      reason="Nothing to retain: the field does not exist in the Korean rendered text. Not added (would be a new field).",
      confirm=True, confirm_reason="PREREGISTRATION_A D-A-delta (line 150) lists 'the Fungemia binary' as retained, but the Korean narratives contain no such line. No action possible; recorded so the sign-off is informed.",
      in_k=False, ev=[E(C27, 72, "[Fungemia]"), E(C27, 73, "Fungemia :"), E(C21, 78, "no Candida/Fungemia"), E(PRE, 150, "Fungemia")]),
    R("M02", "mimic_only", "(no Korean line) Timing : hospital day N", "(none)", [], "Timing : hospital day {N} (27c:103)", True,
      "Relative day; Korea renders only absolute Test_date (B02).", Z, None, None, "not_rendered", "prereg D-A-beta",
      reason="D-A-beta: no timing substitute.", in_k=False, ev=[EV_DA_BETA, E(C27, 103, "Timing")]),
    R("M03", "mimic_only", "(no Korean line) Number of BSI episodes this admission", "(none)", [], "Number of BSI episodes this admission : (27c:116)", True,
      "Korean text has no rendered episode-count line (n_episode_blocks is a structural feature computed by the parser, 20a:274).", Z, None, None, "not_rendered",
      "prereg D-A-delta ('episode count'); no Korean line exists", reason="Adding a count line would create a new computed field.", in_k=False,
      ev=[E(C27, 116, "Number of BSI episodes"), E(P20, 274, "structural_features")]),
    R("M04", "mimic_only", "(no Korean line) [Organ support] / [Spectrum flags] section headers", "(none)", [], "[Organ support] (27c:82), [Spectrum flags] (27c:91)", True,
      "In Korea the same lines live inside [Summary tokens] (S01, S02_*).", M, None, None, "not_rendered", "prereg D-A-alpha",
      reason="Korean keeps its own section structure (spec A1: preserve Korean syntax/order).", in_k=False, ev=[E(C27, 82, "[Organ support]"), E(C27, 91, "[Spectrum flags]")]),
    R("M05", "mimic_only", "(no Korean line) is_empiric_tt_excessivelylarge / is_admission_tt_excessivelylarge", "(none)", [], "none", False,
      "Absent from the Korean rendered corpus (21_common.py:159-160); dropped in FNF:22.", U, None, None, "not_rendered", "FNF:22",
      reason="Not present in the corpus of record.", in_k=False, ev=[E(FNF_P, 22, "excessivelylarge"), E(C21, 159, "excessivelylarge")]),
    R("M06", "mimic_only", "(no Korean line) Fungemia organism / timing detail", "(none)", [], "none", False,
      "Absent from the Korean rendered corpus (no fungal content, 21_common.py:78); FNF:24.", U, None, None, "not_rendered", "FNF:24",
      reason="Not present in the corpus of record.", in_k=False, ev=[E(FNF_P, 24, "Fungemia organism"), E(C21, 78, "no Candida/Fungemia")]),
    R("M07", "mimic_only", "raw_fields struct__* (n_episode_blocks, byte_length, line_count, ast_line_count)", "(parser features)", [],
      "n/a", False, "Structural features computed by the parser from the rendered text (20a:274); not rendered lines.", U, None, None, "not_rendered",
      "n/a", reason="Derived features, not text.", in_k=False,
      cols=["struct__n_episode_blocks", "struct__byte_length", "struct__line_count", "struct__ast_line_count"], ev=[E(P20, 274, "structural_features")]),
]

# Disposition of every Phase-1 "author must confirm" flag by PREREGISTRATION_A_ADDENDUM_1.md.
DISPOSITION = {
    "B06": ("SETTLED", "Addendum 1 decision 1: AST_detailed removed from every arm"),
    "B07": ("SETTLED", "Addendum 1 decision 1"), "B08": ("SETTLED", "Addendum 1 decision 1"), "B09": ("SETTLED", "Addendum 1 decision 1"),
    "I04": ("SETTLED", "Addendum 1 decision 2: duplicate organism line removed from every arm"),
    "I07": ("SETTLED", "Addendum 1 decision 3: efficacy lines dropped in both arms"),
    "T00": ("SETTLED", "Addendum 1: section-header rule adopted (removed only when the section is empty)"),
    "P00": ("SETTLED", "Addendum 1: section-header rule adopted"),
    "A01": ("SETTLED", "Addendum 1: Patient ID dropped (FIELDS_NOT_FILLED line 13)"),
    "S04_index_bsi_polymicrobial": ("SETTLED", "Addendum 1: frozen D-A-alpha governs (five summary lines kept)"),
    "M01": ("SETTLED", "Addendum 1: no Korean document contains the line; nothing retained or added"),
    "A05": ("LIMITATION", "Addendum 1: retained as in the frozen text, reported as a limitation of Analysis A"),
    "C01": ("LIMITATION", "Addendum 1: retained as in the frozen text, reported as a limitation of Analysis A"),
    "C03": ("LIMITATION", "Addendum 1: retained as in the frozen text, reported as a limitation of Analysis A"),
    "B05": ("LIMITATION", "Addendum 1: retained as in the frozen text, reported as a limitation of Analysis A"),
    "B01": ("LIMITATION", "Addendum 1: retained as in the frozen text, reported as a limitation of Analysis A"),
}
_unflagged = [r["field_id"] for r in ROWS if r["author_confirm"] and r["field_id"] not in DISPOSITION]
assert not _unflagged, f"Phase-1 flags without an Addendum-1 disposition: {_unflagged}"

FIELD_IDS = [r["field_id"] for r in ROWS]
assert len(FIELD_IDS) == len(set(FIELD_IDS)), "duplicate field_id in manifest table"


# ----------------------------------------------------------------------------
# helpers shared with 61
# ----------------------------------------------------------------------------
def arm_retains(row: dict, arm: str):
    if arm == "korean_mimic_compatible":
        return row["retained_korean_mimic_compatible"]
    if arm == "harmonized_keep_therapy":
        return row["retained_harmonized_keep_therapy"]
    raise KeyError(arm)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "-C", str(REV), "rev-parse", "HEAD"], text=True).strip()


def verify_evidence() -> list[dict]:
    """Every cited file:line must contain its snippet; returns the verified list."""
    cache: dict[str, list[str]] = {}
    seen = {}
    for r in ROWS:
        for (path, line, snip) in r["evidence"]:
            if path not in cache:
                cache[path] = (REV / path).read_text(encoding="utf-8").split("\n")
            lines = cache[path]
            if not (1 <= line <= len(lines)) or snip not in lines[line - 1]:
                raise SystemExit(f"[60] EVIDENCE FAILURE {r['field_id']}: {path}:{line} does not contain {snip!r}: "
                                 f"{lines[line - 1][:120] if 1 <= line <= len(lines) else '<out of range>'!r}")
            seen[(path, line)] = snip
    return [{"path": p, "line": l, "snippet": s} for (p, l), s in sorted(seen.items())]


def _frozen_hash(hash_file: Path, suffix: str):
    for ln in hash_file.read_text().splitlines():
        if ln.strip().endswith(suffix):
            return ln.split()[0]
    return None


def check_prereg_frozen() -> str:
    h = sha256_file(PREREG)
    frozen = _frozen_hash(PREREG_HASHES, "A_harmonized_template/PREREGISTRATION_A.md")
    if frozen != h:
        raise SystemExit(f"[60] PREREGISTRATION_A.md hash {h} != frozen {frozen}: REFUSING (prereg must not change)")
    ha = sha256_file(ADDENDUM)
    frozen_a = _frozen_hash(ADDENDUM_HASHES, "A_harmonized_template/PREREGISTRATION_A_ADDENDUM_1.md")
    if frozen_a != ha:
        raise SystemExit(f"[60] PREREGISTRATION_A_ADDENDUM_1.md hash {ha} != frozen {frozen_a}: REFUSING")
    return h


def git_state(files=None) -> dict:
    """HEAD + a dirty flag (uncommitted changes in the given pathspec, or anywhere in tracked files)."""
    args = ["git", "-C", str(REV), "status", "--porcelain", "--untracked-files=no"]
    if files:
        args += ["--", *files]
    out = subprocess.check_output(args, text=True).strip()
    return {"git_head": git_head(), "git_dirty": bool(out), "git_dirty_detail": out.splitlines()[:20]}


def stage_manifest(limit: int | None, out_dir: Path) -> None:
    import pandas as pd

    prereg_hash = check_prereg_frozen()
    ev_verified = verify_evidence()

    # summary-token retention set == what MIMIC renders (D-A-alpha)
    summ_rows = [r for r in ROWS if r["korean_section"] == ST and r["row_kind"] == "field"]
    cats = [r["korean_line_keys"][0].split("::")[1] for r in summ_rows]
    assert sorted(cats) == sorted(c21.SUMMARY_TOKEN_CATEGORIES), "manifest summary categories != 21_common categories"
    retained = sorted(r["korean_line_keys"][0].split("::")[1] for r in summ_rows if r["retained_korean_mimic_compatible"])
    assert retained == sorted(RETAINED_SUMMARY), retained
    assert len(cats) == 19 and len(cats) - len(retained) == 14

    # raw_fields.parquet reconciliation: every column assigned to exactly one row
    raw_cols = [c for c in pd.read_parquet(RAW_FIELDS).columns if c != "admission_ID"]
    assigned = collections.Counter(c for r in ROWS for c in r["raw_fields_columns"])
    missing = [c for c in raw_cols if assigned[c] == 0]
    extra = [c for c in assigned if c not in raw_cols]
    multi = [c for c, n in assigned.items() if n > 1]
    # 6 resistance-flag rows are documented to have no own column; sepsis rows likewise.
    if missing or extra or multi:
        raise SystemExit(f"[60] raw_fields reconciliation FAILED missing={missing} extra={extra} multi={multi}")

    # Corpus scan (line-key classes)
    aids, nids, texts, _ = read_corpus(limit)
    n_docs = len(texts)
    lines_ct: collections.Counter = collections.Counter()
    docs_ct: collections.Counter = collections.Counter()
    per_doc_lines = []
    for t in texts:
        seen = set()
        n = 0
        for k in scan_text(t):
            lines_ct[k] += 1
            n += 1
            if k not in seen:
                seen.add(k)
                docs_ct[k] += 1
        per_doc_lines.append(n)
    covered = collections.Counter(k for r in ROWS for k in r["korean_line_keys"])
    dup = [k for k, n in covered.items() if n > 1]
    unc = sorted(k for k in lines_ct if covered[k] == 0)
    if dup or unc:
        raise SystemExit(f"[60] corpus coverage FAILED: keys in >1 row={dup}; observed but unmapped={unc}")
    declared_absent = [k for r in ROWS if r["rendered_in_korea"] for k in r["korean_line_keys"] if lines_ct[k] == 0]
    declared_present_but_absent = [k for k in declared_absent]  # may be legitimately 0 in a --limit sample
    fungemia_hits = sum(1 for t in texts if "ungemia" in t or "Candida" in t)

    # per-row counts and predicted removals
    rows_out = []
    pred_removed = {a: 0 for a in ARMS_DET}
    for r in ROWS:
        nl = sum(lines_ct[k] for k in r["korean_line_keys"])
        nd = max([docs_ct[k] for k in r["korean_line_keys"]] or [0])
        rec = {
            "field_id": r["field_id"], "row_kind": r["row_kind"], "korean_field": r["korean_field"],
            "korean_section": r["korean_section"], "mimic_equivalent": r["mimic_equivalent"],
            "rendered_in_korea": r["rendered_in_korea"], "rendered_in_mimic": r["rendered_in_mimic"],
            "semantic_equivalence": r["semantic_equivalence"], "harmonization_status": r["harmonization_status"],
        }
        for arm, key in [("korean_mimic_compatible", "retained_korean_mimic_compatible"),
                         ("harmonized_keep_therapy", "retained_harmonized_keep_therapy")]:
            v = r[key]
            rec[f"retained_{arm}"] = "n/a" if v is None else ("yes" if v else "no")
            if v is False:
                pred_removed[arm] += nl
        rec["all_arms_removal_addendum1"] = "yes" if r["all_arms_removal"] else ""
        rec["retained_thickness_matched_random"] = (
            "no (Addendum 1 decisions 1-2 apply to every arm)" if r["all_arms_removal"] else
            ("stochastic (top-level items removed at random; no field-level decision)" if r["rendered_in_korea"] else "n/a"))
        rec["arms_retaining"] = ";".join(a for a in ARMS_DET if arm_retains(r, a)) if r["retained_korean_mimic_compatible"] is not None else ""
        rec["reason_for_exclusion"] = r["reason_for_exclusion"]
        rec["mechanism"] = r["mechanism"]
        rec["decision_source"] = r["decision_source"]
        disp = DISPOSITION.get(r["field_id"])
        rec["author_confirm"] = (disp[0] if disp else "")            # SETTLED | LIMITATION | "" (nothing was flagged)
        rec["addendum1_disposition"] = disp[1] if disp else ""
        rec["open_limitation"] = "YES" if (disp and disp[0] == "LIMITATION") else ""
        rec["phase1_flag_reason"] = r["author_confirm_reason"]
        rec["korean_line_keys"] = " ;; ".join(r["korean_line_keys"])
        rec["raw_fields_columns"] = ";".join(r["raw_fields_columns"])
        rec["n_docs_with_field"] = nd
        rec["n_lines_corpus"] = nl
        rec["evidence"] = " | ".join(f"{p}:{l}" for p, l, _ in r["evidence"])
        rec["notes"] = r["notes"]
        rows_out.append(rec)
    df = pd.DataFrame(rows_out)

    # ---- summary counts ------------------------------------------------------
    rk = df["row_kind"].value_counts().to_dict()
    in_k = df[df["rendered_in_korea"]]
    obs_keys = len(lines_ct)
    total_lines = int(sum(lines_ct.values()))

    def by_arm(arm):
        col = f"retained_{arm}"
        removed_rows = in_k[in_k[col] == "no"]
        kept_rows = in_k[in_k[col] == "yes"]
        return {
            "rows_retained": int(len(kept_rows)), "rows_removed": int(len(removed_rows)),
            "line_classes_retained": int(sum(len(r["korean_line_keys"]) for r in ROWS if r["rendered_in_korea"] and arm_retains(r, arm))),
            "line_classes_removed": int(sum(len(r["korean_line_keys"]) for r in ROWS if r["rendered_in_korea"] and arm_retains(r, arm) is False)),
            "lines_removed_corpus_predicted": int(pred_removed[arm]),
            "lines_retained_corpus_predicted": int(total_lines - pred_removed[arm]),
            "median_lines_per_doc_before": float(statistics.median(per_doc_lines)),
            "sections_with_any_removal": sorted({r["korean_section"] for r in ROWS
                                                 if r["rendered_in_korea"] and r["row_kind"] in ("field", "anchor") and arm_retains(r, arm) is False}),
            "section_headers_removed": sorted(r["korean_section"] for r in ROWS if r["row_kind"] == "section_header" and arm_retains(r, arm) is False),
            "status_counts_retained_rows": in_k[in_k[col] == "yes"]["harmonization_status"].value_counts().to_dict(),
            "status_counts_removed_rows": in_k[in_k[col] == "no"]["harmonization_status"].value_counts().to_dict(),
        }

    summary = {
        "n_docs_scanned": n_docs,
        "n_distinct_line_key_classes_observed": obs_keys,
        "n_manifest_rows": int(len(df)), "row_kind_counts": rk,
        "n_lines_corpus_total": total_lines,
        "lines_per_doc_median_nonblank": float(statistics.median(per_doc_lines)),
        "harmonization_status_counts_all_rows": df["harmonization_status"].value_counts().to_dict(),
        "harmonization_status_counts_korean_rendered_rows": in_k["harmonization_status"].value_counts().to_dict(),
        "harmonization_status_counts_field_rows_only": in_k[in_k["row_kind"] == "field"]["harmonization_status"].value_counts().to_dict(),
        "n_field_rows_korean_rendered": int((in_k["row_kind"] == "field").sum()),
        "field_rows_by_arm_and_status": {
            a: {"retained": in_k[(in_k["row_kind"] == "field") & (in_k[f"retained_{a}"] == "yes")]["harmonization_status"].value_counts().to_dict(),
                "removed": in_k[(in_k["row_kind"] == "field") & (in_k[f"retained_{a}"] == "no")]["harmonization_status"].value_counts().to_dict()}
            for a in ARMS_DET},
        "rows_flagged_in_phase1_now_settled": df[df["author_confirm"] == "SETTLED"]["field_id"].tolist(),
        "rows_retained_as_documented_limitations": df[df["author_confirm"] == "LIMITATION"]["field_id"].tolist(),
        "rows_still_needing_author_confirmation": [],
        "arms": {a: by_arm(a) for a in ARMS_DET},
        "fungemia_or_candida_token_docs_in_corpus": fungemia_hits,
        "declared_present_in_korea_but_zero_lines_in_scan": declared_present_but_absent,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_p = out_dir / MANIFEST_CSV.name
    json_p = out_dir / MANIFEST_JSON.name
    tmp = csv_p.with_name(f".tmp_{os.getpid()}_{csv_p.name}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv_p)

    prov = {
        "stage": "manifest", "script": "scripts/60_harmonized_manifest.py",
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        **git_state(["scripts/60_harmonized_manifest.py", "scripts/61_harmonized_render.py"]), "sys_executable": sys.executable,
        "pandas": pd.__version__, "numpy": __import__("numpy").__version__,
        "addendum_sha256": sha256_file(ADDENDUM), "addendum_frozen_hash_verified": True,
        "limit": limit,
        "prereg_sha256": prereg_hash, "prereg_frozen_hash_verified": True,
        "inputs_sha256": {
            "FIELDS_NOT_FILLED.md": sha256_file(FNF), "raw_fields.parquet": sha256_file(RAW_FIELDS),
            "27c_mimic_narratives.py": sha256_file(REV / C27), "21_common.py": sha256_file(REV / C21),
            "27b_mimic_enrich.py": sha256_file(REV / B27), "27a_mimic_cohort.py": sha256_file(REV / A27),
            "20a_parse_template_fields.py": sha256_file(REV / P20),
        },
        "manifest_csv_sha256": sha256_file(csv_p),
        "script_sha256": {"60_harmonized_manifest.py": sha256_file(Path(__file__).resolve())},
        "corpus_of_record": str(c18.REPORTS_DIR), "n_corpus_files_expected": 32329,
        "evidence_entries_verified": len(ev_verified),
        "summary": summary,
        "phase": "2 (Addendum 1)",
        "decisions_implemented": {
            "D-A-alpha": "retain [Summary tokens] header + exactly 5 lines " + ",".join(RETAINED_SUMMARY) + "; drop the other 14",
            "D-A-beta": "drop Admission Date and every Test_date line; no timing substitute",
            "D-A-gamma": "primary arm: drop whole-admission antibiotics block (Antibiotic_N_name / duration_days / placeholder), keep empiric names; "
                         "per-episode Antibiotic_N_efficacy lines dropped in BOTH deterministic arms (Addendum 1 decision 3); glucocorticoid lines "
                         "dropped in both arms; keep_therapy retains the whole-admission block ONLY",
            "Addendum-1 decision 1": "AST_detailed header + all per-drug lines + `o NA` placeholder removed from every arm (incl. random)",
            "Addendum-1 decision 2": "`Microbes isolated in this episode` removed from every arm (incl. random)",
            "D-A-delta": "remaining drops per FNF / prereg (see manifest rows with mechanism remove_items)",
            "Rule R": "a section header is removed iff the section is empty after surgery (Sofa, Procedures always; Treatments in the primary arm)",
            "D-A-epsilon": "three arms; the random arm is built by 61 render, not by this manifest",
        },
    }
    tmp = json_p.with_name(f".tmp_{os.getpid()}_{json_p.name}")
    tmp.write_text(json.dumps(prov, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, json_p)
    print(json.dumps({"manifest_csv": str(csv_p), "n_rows": len(df), "status_counts_korean_rendered_rows":
                      summary["harmonization_status_counts_korean_rendered_rows"],
                      "pred_removed": pred_removed, "n_docs": n_docs}, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["manifest"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="(testing) scan only the first N documents")
    ap.add_argument("--out-dir", type=Path, default=A_DIR, help="(testing) override output directory")
    args = ap.parse_args()
    if args.limit and args.out_dir == A_DIR:
        raise SystemExit("--limit refuses to write into the real output directory; pass --out-dir")
    stage_manifest(args.limit, args.out_dir)


if __name__ == "__main__":
    main()
