#!/usr/bin/env python3
"""Task 1 (one-hot comparator) -- Step 1: parse the published template's
fields directly out of `reports_pilot_pre/*.report.md` (the same
`field: value` lines the embedding saw -- never the source CSVs, per the
task brief). Corpus loader is `18_common.load_texts()` (already-verified
`reports_pilot_pre` corpus, admission_ID/newpatient_ID aligned via
`out_pre` metadata).

Two stages, resumable, atomic writes:
  --stage parse      : per-document field extraction -> raw_fields.parquet
                        (one row per admission_ID, JSON-able values, no
                        one-hot yet). Cheap (pure text/regex), safe to run
                        on the login node.
  --stage vectorize   : raw_fields.parquet -> the two feature matrices
                        (variant a: template fields only; variant b: a +
                        rendered structural features), written as
                        out/20_onehot/features/onehot_<variant>.npy +
                        ids_<variant>.json + colnames_<variant>.json.

Template structure (empirically verified, not assumed -- see
`_verify_summary_token_grammar` and the docstring notes below each parser):

- `[Summary tokens]`: one flattened `<prefix>_<level>` token per line, no
  colon. Full-corpus-generic grammar (regex `^(.*)_(yes|no|NA)$` else
  `^(.*)_(\\d+)$`), verified 0 unmatched tokens across a 3,000-doc sample.
  19 fields: 17 categorical yes/no/NA(-subset) flags + 2 numeric-ID
  cluster labels (`admission_cluster`, `empiric_cluster`) -- all treated as
  categorical here (they are opaque IDs, not scores).
- `[Administrative informations]`: Age (numeric), Sex / Entry mode /
  Treatment departement (categorical), Admission Date (split into
  categorical year + month -- exact day-level dates are effectively
  unique identifiers, not a clinical "field level").
- `[Underlying comorbidities]`: three comma-separated multi-label fields
  (Infectious comorbidities, All ICD10 codes, Charlson classification) +
  Charlson score (numeric).
- `[Treatments during admission]`: admission-level `Antibiotic_N_name`
  lines (N unbounded, up to 20 in this corpus) -> multi-hot drug-presence
  (positional "which slot" is not clinically meaningful, so slots are
  pooled into a presence set, not encoded as separate fields); total
  admission antibiotic duration summed (numeric); three glucocorticoid
  fields (categorical yes/NA).
- `[Procedures during the admission]`: 3 categorical yes/NA fields.
- `[Sofa/qsofa]`: `sofa_score_2` / `quick_sofa_score_2` numeric (the two
  sepsis flags in this same block duplicate `[Summary tokens]`'s
  `sepsis_2`/`sepsis_quicksofa_2` and are NOT re-parsed here, to avoid
  encoding the same fact twice under two names).
- `[Physiological and biological parameters]`: all numeric.
- `[Bacterial culture]`: INDEX episode only (empirically verified: the
  first `BSI_episode_num` block's resistance flags equal `[Summary
  tokens]`'s `index_bsi_*` flags exactly) -- only the one field not
  already covered by Summary tokens is extracted here: the organism
  (`BSI_episode_short_clean`). All later episode blocks in this section,
  and all of `[Infection treatment]`, are per-episode content and are
  deliberately NOT flattened into variant (a) -- a document with 14
  episodes would otherwise inject 14x as many organism/AST fields as a
  document with 1, which is exactly the episode-count confound variant
  (a) exists to exclude structurally. Variant (b) instead adds the
  document-level COUNT/SIZE of that per-episode content explicitly (see
  Step 2, `20b_...` companion / `_structural_features` below).

High-cardinality multi-label fields (`All ICD10 codes`, admission
antibiotic names) are frequency-thresholded at MIN_LEVEL_COUNT=20
corpus-wide occurrences; rarer levels collapse into one `<field>__OTHER`
indicator column rather than being silently dropped (still contributes a
binary signal: "has a rare code/drug").

Outcome-blind (Global Constraints): no mortality/discharge/death column is
read anywhere in this module -- the source is narrative text, and no
outcome field is ever part of the rendered template body used here.
"""
import argparse
import importlib
import json
import os
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c18 = importlib.import_module("18_common")

OUT = c18.REV / "out" / "20_onehot"
FEAT_DIR = OUT / "features"

MIN_LEVEL_COUNT = 20

SUMMARY_TOKEN_RE_YN = re.compile(r"^(.*)_(yes|no|NA)$")
SUMMARY_TOKEN_RE_NUM = re.compile(r"^(.*)_(\d+)$")

NUMERIC_PHYS_FIELDS = [
    "SBP", "DBP", "MBP", "RR", "Body_temperature", "PR", "GCS_2",
    "Vasopressin_any", "WBC_Count", "CRP", "AST", "ALT", "T_Bilirubin",
    "Creatinine", "pO2_Arterial", "Hemoglobin", "Albumin", "PLT_Count",
]

MISSING_LINE = "__LINE_ABSENT__"


def _section_spans(text: str) -> dict:
    """Return {header_name: (start_line_idx, end_line_idx_exclusive)} for
    every `[Header]` block, using line indices into text.split('\\n')."""
    lines = text.split("\n")
    spans = {}
    cur_name, cur_start = None, None
    for i, line in enumerate(lines):
        if line.startswith("["):
            if cur_name is not None:
                spans[cur_name] = (cur_start, i)
            cur_name = line.strip().strip("[]")
            cur_start = i + 1
    if cur_name is not None:
        spans[cur_name] = (cur_start, len(lines))
    return spans


def _kv(line: str) -> tuple:
    """`- key : value` / `• key : value` / `key : value` -> (key, value),
    stripping leading bullet markers. Returns (None, None) if no colon."""
    s = line.strip()
    if s.startswith("- "):
        s = s[2:]
    elif s.startswith("• "):
        s = s[2:]
    if ":" not in s:
        return None, None
    k, _, v = s.partition(":")
    return k.strip(), v.strip()


def parse_summary_tokens(lines: list) -> dict:
    out = {}
    for line in lines:
        tok = line.strip()
        if not tok:
            continue
        m = SUMMARY_TOKEN_RE_YN.match(tok)
        if m:
            out[f"summary__{m.group(1)}"] = m.group(2)
            continue
        m = SUMMARY_TOKEN_RE_NUM.match(tok)
        if m:
            out[f"summary__{m.group(1)}"] = m.group(2)
            continue
        out.setdefault("summary__UNPARSED", []).append(tok)
    return out


def parse_administrative(lines: list) -> dict:
    out = {}
    for line in lines:
        k, v = _kv(line)
        if k is None:
            continue
        if k.startswith("Age"):
            out["admin__age"] = v
        elif k.startswith("Sex"):
            out["admin__sex"] = v
        elif k.startswith("Entry mode"):
            out["admin__entry_mode"] = v
        elif k.startswith("Treatment departement"):
            out["admin__department"] = v
        elif k.startswith("Admission Date"):
            out["admin__admission_date_raw"] = v
            if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
                out["admin__admission_year"] = v[:4]
                out["admin__admission_month"] = v[5:7]
    return out


def parse_comorbidities(lines: list) -> dict:
    out = {}
    for line in lines:
        k, v = _kv(line)
        if k is None:
            continue
        if k.startswith("Infectious comorbidities"):
            out["comorbid__infectious"] = [t.strip() for t in v.split(",") if t.strip()]
        elif k.startswith("All ICD10 codes"):
            out["comorbid__icd10"] = [t.strip() for t in v.split(",") if t.strip()]
        elif k.startswith("All comorbidities as classified"):
            out["comorbid__charlson_class"] = [t.strip() for t in v.split(",") if t.strip()]
        elif k.startswith("Charlson score"):
            out["comorbid__charlson_score"] = v
    return out


def parse_treatments(lines: list) -> dict:
    drugs = []
    total_duration = 0.0
    gluco = {}
    for line in lines:
        k, v = _kv(line)
        if k is None:
            continue
        m = re.match(r"^Antibiotic_(\d+)_name$", k)
        if m:
            drugs.append(v)
            continue
        m = re.match(r"^Antibiotic_(\d+)_duration_days$", k)
        if m:
            try:
                total_duration += float(v)
            except ValueError:
                pass
            continue
        if k == "Short_acting_glucocorticoids":
            gluco["tx__gluco_short"] = v
        elif k == "Intermediate_acting_glucocorticoids":
            gluco["tx__gluco_intermediate"] = v
        elif k == "Long_acting_glucocorticoids":
            gluco["tx__gluco_long"] = v
    out = {"tx__admission_antibiotics": drugs, "tx__total_antibiotic_duration_days": total_duration}
    out.update(gluco)
    return out


def parse_procedures(lines: list) -> dict:
    out = {}
    for line in lines:
        k, v = _kv(line)
        if k is None:
            continue
        if k == "central_orarterial_line":
            out["proc__central_line"] = v
        elif k == "urinary_orrenal_catheter":
            out["proc__urinary_catheter"] = v
        elif k == "ventilator":
            out["proc__ventilator"] = v
    return out


def parse_sofa(lines: list) -> dict:
    out = {}
    for line in lines:
        k, v = _kv(line)
        if k is None:
            continue
        if k == "sofa_score_2":
            out["sofa__sofa_score_2"] = v
        elif k == "quick_sofa_score_2":
            out["sofa__quick_sofa_score_2"] = v
    return out


def parse_physio(lines: list) -> dict:
    out = {}
    for line in lines:
        k, v = _kv(line)
        if k is None:
            continue
        if k in NUMERIC_PHYS_FIELDS:
            out[f"physio__{k}"] = v
    return out


def parse_bacterial_culture_index_episode(lines: list) -> dict:
    """Only the FIRST `BSI_episode_num` block (verified == the index
    episode whose resistance flags equal Summary tokens' index_bsi_*, see
    module docstring). Extracts only the organism -- the resistance flags
    are already covered by Summary tokens."""
    for i, line in enumerate(lines):
        if line.strip().startswith("BSI_episode_num"):
            for j in range(i + 1, min(i + 15, len(lines))):
                if lines[j].strip().startswith("BSI_episode_num"):
                    break
                k, v = _kv(lines[j])
                if k == "BSI_episode_short_clean":
                    return {"culture__index_organism": v}
            return {}
    return {}


def structural_features(text: str, bact_lines: list) -> dict:
    n_episode_blocks = sum(1 for ln in bact_lines if ln.strip().startswith("BSI_episode_num"))
    ast_line_count = sum(1 for ln in bact_lines if ln.strip().startswith("o "))
    return {
        "struct__n_episode_blocks": n_episode_blocks,
        "struct__byte_length": len(text.encode("utf-8")),
        "struct__line_count": text.count("\n") + 1,
        "struct__ast_line_count": ast_line_count,
    }


def parse_one(admission_id: int, text: str) -> dict:
    lines = text.split("\n")
    spans = _section_spans(text)

    def sec(name):
        if name not in spans:
            return []
        a, b = spans[name]
        return lines[a:b]

    rec = {"admission_ID": int(admission_id)}
    rec.update(parse_summary_tokens(sec("Summary tokens")))
    rec.update(parse_administrative(sec("Administrative informations")))
    rec.update(parse_comorbidities(sec("Underlying comorbidities")))
    rec.update(parse_treatments(sec("Treatments during admission")))
    rec.update(parse_procedures(sec("Procedures during the admission")))
    rec.update(parse_sofa(sec("Sofa/qsofa")))
    rec.update(parse_physio(sec("Physiological and biological parameters")))
    bact_lines = sec("Bacterial culture")
    rec.update(parse_bacterial_culture_index_episode(bact_lines))
    rec.update(structural_features(text, bact_lines))
    return rec


def stage_parse() -> dict:
    import pandas as pd

    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    dest = FEAT_DIR / "raw_fields.parquet"

    meta = c18.load_texts()  # admission_ID, newpatient_ID, embedding_index, text -- full 32,329-doc corpus
    print(f"[20a] parsing {len(meta)} narratives from reports_pilot_pre/ ...", flush=True)
    records = []
    n_unparsed_summary = 0
    for i, (aid, text) in enumerate(zip(meta["admission_ID"], meta["text"])):
        rec = parse_one(aid, text)
        if "summary__UNPARSED" in rec:
            n_unparsed_summary += 1
        records.append(rec)
        if (i + 1) % 5000 == 0:
            print(f"[20a] parsed {i + 1}/{len(meta)}", flush=True)

    df = pd.DataFrame.from_records(records)
    # list-valued / dict-of-list columns must be JSON-encoded for parquet
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, list)).any():
            df[col] = df[col].map(lambda v: json.dumps(v) if isinstance(v, list) else v)

    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)

    manifest = {
        "n_rows": len(df),
        "n_fields": len([c for c in df.columns if c != "admission_ID"]),
        "n_docs_with_unparsed_summary_token": int(n_unparsed_summary),
        "columns": sorted(df.columns.tolist()),
        "dest": str(dest),
    }
    c18.atomic_write_json(manifest, FEAT_DIR / ".parse_manifest.json")
    print(json.dumps({k: v for k, v in manifest.items() if k != "columns"}, indent=2), flush=True)
    if n_unparsed_summary:
        print(f"WARNING: {n_unparsed_summary} docs had an unparsed summary token "
              f"(grammar assumption may not be fully exhaustive) -- see raw_fields.parquet "
              f"summary__UNPARSED column", flush=True)
    return manifest


CAT_SINGLE_FIELDS = [
    "summary__admission_cluster", "summary__antiMRSAab_empiric", "summary__antiVREab_empiric",
    "summary__antipseudoab_empiric", "summary__carbaab_empiric", "summary__cre_colonization",
    "summary__empiric_cluster", "summary__empiric_tt_adapted", "summary__icu",
    "summary__index_bsi_3gcr", "summary__index_bsi_carbar", "summary__index_bsi_ceftriaxoner",
    "summary__index_bsi_crab", "summary__index_bsi_mrsa", "summary__index_bsi_polymicrobial",
    "summary__index_bsi_vre", "summary__sepsis_2", "summary__sepsis_quicksofa_2",
    "summary__vre_colonization",
    "admin__sex", "admin__entry_mode", "admin__department", "admin__admission_year",
    "admin__admission_month",
    "tx__gluco_short", "tx__gluco_intermediate", "tx__gluco_long",
    "proc__central_line", "proc__urinary_catheter", "proc__ventilator",
    "culture__index_organism",
]

MULTI_HOT_FIELDS = ["comorbid__infectious", "comorbid__icd10", "comorbid__charlson_class",
                     "tx__admission_antibiotics"]

NUMERIC_FIELDS = (
    ["admin__age", "comorbid__charlson_score", "tx__total_antibiotic_duration_days",
     "sofa__sofa_score_2", "sofa__quick_sofa_score_2"]
    + [f"physio__{f}" for f in NUMERIC_PHYS_FIELDS]
)

STRUCTURAL_FIELDS = ["struct__n_episode_blocks", "struct__byte_length", "struct__line_count",
                      "struct__ast_line_count"]


def _standardize(values, add_indicator: bool = True):
    """Returns (z, colnames, isna_indicator_or_None). NaN -> 0 after
    standardization (mean-imputation-equivalent); missingness is kept as an
    explicit extra indicator column rather than silently imputed away,
    mirroring the "NA is its own level" rule used for categoricals."""
    import numpy as np

    values = values.astype(float)
    isna = np.isnan(values)
    mean = np.nanmean(values) if not isna.all() else 0.0
    std = np.nanstd(values) if not isna.all() else 1.0
    if not np.isfinite(std) or std == 0:
        std = 1.0
    z = (values - mean) / std
    z[isna] = 0.0
    if add_indicator and isna.any():
        return z, isna.astype(float)
    return z, None


def _multi_hot(series, min_count: int) -> "pd.DataFrame":
    import pandas as pd
    from collections import Counter

    parsed = series.map(lambda s: json.loads(s) if isinstance(s, str) else [])
    counts = Counter(t for lst in parsed for t in lst)
    keep = {t for t, n in counts.items() if n >= min_count}
    dropped = {t for t, n in counts.items() if n < min_count}
    cols = {}
    for t in sorted(keep):
        cols[t] = parsed.map(lambda lst, t=t: 1.0 if t in lst else 0.0)
    if dropped:
        cols["__OTHER__"] = parsed.map(lambda lst: 1.0 if any(t in dropped for t in lst) else 0.0)
    return pd.DataFrame(cols), sorted(keep), sorted(dropped)


def build_feature_matrix(df, fields_numeric, fields_cat, fields_multi, min_level_count=MIN_LEVEL_COUNT):
    import numpy as np
    import pandas as pd

    blocks = []
    colnames = []
    field_report = {}

    for f in fields_numeric:
        raw = pd.to_numeric(df[f], errors="coerce").to_numpy()
        z, isna_ind = _standardize(raw)
        blocks.append(z.reshape(-1, 1))
        colnames.append(f"{f}__z")
        if isna_ind is not None:
            blocks.append(isna_ind.reshape(-1, 1))
            colnames.append(f"{f}__isNA")
        field_report[f] = {"kind": "numeric", "n_missing": int(isna_ind.sum()) if isna_ind is not None else 0}

    for f in fields_cat:
        dummies = pd.get_dummies(df[f].astype(str), prefix=f)
        blocks.append(dummies.to_numpy(dtype=float))
        colnames.extend(dummies.columns.tolist())
        field_report[f] = {"kind": "categorical", "n_levels": dummies.shape[1]}

    for f in fields_multi:
        dummies, kept, dropped = _multi_hot(df[f], min_level_count)
        blocks.append(dummies.to_numpy(dtype=float))
        colnames.extend([f"{f}__{c}" for c in dummies.columns])
        field_report[f] = {"kind": "multi_hot", "n_levels_kept": len(kept),
                            "n_levels_dropped_to_OTHER": len(dropped), "min_level_count": min_level_count}

    X = np.concatenate(blocks, axis=1).astype(np.float32)
    return X, colnames, field_report


def stage_vectorize() -> dict:
    import numpy as np
    import pandas as pd

    src = FEAT_DIR / "raw_fields.parquet"
    if not src.exists():
        raise FileNotFoundError(f"{src} missing -- run --stage parse first")
    df = pd.read_parquet(src)

    manifest = {}
    for variant, extra_numeric in [("a", []), ("b", STRUCTURAL_FIELDS)]:
        X, colnames, field_report = build_feature_matrix(
            df, NUMERIC_FIELDS + extra_numeric, CAT_SINGLE_FIELDS, MULTI_HOT_FIELDS,
        )
        ids = df["admission_ID"].to_numpy()

        vec_path = FEAT_DIR / f"onehot_{variant}.npy"
        id_path = FEAT_DIR / f"onehot_{variant}_ids.json"
        col_path = FEAT_DIR / f"onehot_{variant}_colnames.json"
        tmp_vec = vec_path.with_name(f".tmp_{os.getpid()}_{vec_path.name}")
        np.save(tmp_vec, X)
        os.replace(tmp_vec, vec_path)
        tmp_id = id_path.with_name(f".tmp_{os.getpid()}_{id_path.name}")
        tmp_id.write_text(json.dumps(ids.tolist()))
        os.replace(tmp_id, id_path)
        tmp_col = col_path.with_name(f".tmp_{os.getpid()}_{col_path.name}")
        tmp_col.write_text(json.dumps(colnames))
        os.replace(tmp_col, col_path)

        manifest[variant] = {
            "n_rows": int(X.shape[0]), "n_cols": int(X.shape[1]),
            "n_numeric_fields": len(NUMERIC_FIELDS + extra_numeric),
            "n_categorical_fields": len(CAT_SINGLE_FIELDS),
            "n_multi_hot_fields": len(MULTI_HOT_FIELDS),
            "field_report": field_report,
        }
        print(f"[20a/vectorize] variant {variant}: X shape={X.shape}", flush=True)

    c18.atomic_write_json(manifest, FEAT_DIR / ".vectorize_manifest.json")
    print(json.dumps({v: {k: vv for k, vv in m.items() if k != "field_report"} for v, m in manifest.items()},
                      indent=2), flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["parse", "vectorize"])
    args = ap.parse_args()
    if args.stage == "parse":
        stage_parse()
    elif args.stage == "vectorize":
        stage_vectorize()


if __name__ == "__main__":
    main()
