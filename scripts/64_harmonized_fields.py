#!/usr/bin/env python3
"""Analysis A, PHASE 2 -- field attribution, conditional agreement, degeneracy, cross-encoder concordance,
the preregistered decision-rule outputs, and comparison_table.csv.

Governing documents (FROZEN): PREREGISTRATION_A.md sections 3.2 (pipeline), 4.0-4.5 (decision rules, reporting
units) and PREREGISTRATION_A_ADDENDUM_1.md.  This script COMPUTES the quantities and applies the pre-written
mechanical rules (zone / threshold assignments); it writes NO interpretation.

Inputs (all read-only):
  * this analysis: matched/<arm>/<encoder>/{d??_s?.npy, whole_s42.npy} (63), manifest CSV (60)
  * stored Korean full-template matched fits: out/39_audit6/matched/draw{01..10}.npz (labels 'bge-m3|s',
    'qwen3-0.6b|s' and the subsets); stored summaries out/39_audit6/matched_encoders.json,
    out/40_field_dominance/field_dominance.json, out/41_audit7/dominance_checks.json, out/33_figures/*.csv
  * stored MIMIC-IV fits: out/27_mimic/replicate/labels/<encoder>_seed{1..10}.parquet, cohort/narrative_manifest.parquet
  * stored Qwen3 `no_therapy` whole-corpus labels: out/47_qwen3_seeds/labels/no_therapy__seed{42,1..10}.npy

Reused (imported, not re-implemented): 40_field_dominance.{load_korea_fields, build_fields, field_ami,
summarise_dominance, KOREA_CATEGORICAL, KOREA_CONTINUOUS}; audit7_dominance_checks.{conditional_agreement}
(NOT its hard-coded dominant-field dictionary at :144-145 -- the dominant field is always computed);
audit6_matched_encoders.{agree, summarize}.

Conventions (PREREGISTRATION_A section 3.2): agreement = both-non-noise ARI with AMI alongside; degeneracy = k >= 5 in
BOTH partitions of a pair; encoder-level degeneracy = fraction of fits with k <= 4; conditional agreement =
size-weighted mean of per-stratum AMI/ARI (min_n = 50) within index-organism strata, median across fits.

FIELD ATTRIBUTION is computed over the union of (i) the 40_field_dominance field list and (ii) the fields that script
silently dropped because of column-name mismatches (entry_mode, index_bsi_*), the latter taken from
raw_fields.parquet (parsed from the rendered text itself).  For every arm the "rendered" subset is derived from the
reviewed manifest (field -> manifest row -> retained_<arm>); dominance is reported over the RENDERED fields (primary)
and over all fields (diagnostic, comparable with the stored Korean values).  For the random arm every field that
exists in the full template counts as rendered (removal is per-document random).

Stages:  --stage attribution | aggregate | all      Env: /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
"""
import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import collections
import importlib
import importlib.metadata
import itertools
import json
import multiprocessing as mp
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
f40 = importlib.import_module("40_field_dominance")
a7 = importlib.import_module("audit7_dominance_checks")
a6 = importlib.import_module("audit6_matched_encoders")

REV = SCRIPT_DIR.parent
A_DIR = REV / "out/followup_analyses/A_harmonized_template"
MATCHED = A_DIR / "matched"
MAN_CSV = A_DIR / "harmonized_field_manifest.csv"
DRAWS = REV / "out/39_audit6/matched"
STORED_ENC = REV / "out/39_audit6/matched_encoders.json"
STORED_DOM = REV / "out/40_field_dominance/field_dominance.json"
STORED_COND = REV / "out/41_audit7/dominance_checks.json"
MIMIC_LABELS = REV / "out/27_mimic/replicate/labels"
MIMIC_COHORT = REV / "out/27_mimic/cohort"
NO_THERAPY = REV / "out/47_qwen3_seeds/labels"
IDS_JSON = REV / "out/18_encoder_dec/embeddings/bge-m3/merged_ids.json"
RAW_FIELDS = f40.RAW_FIELDS
OUT_DIR = A_DIR                                  # outputs; overridden by --out-dir for smoke tests
ARMS = ("korean_mimic_compatible", "harmonized_keep_therapy", "thickness_matched_random")
MODELS = ("bge-m3", "qwen3-0.6b")
CONDITIONS = ("korea_full_template",) + ARMS + ("mimic_iv",)
DRAW_IDS = list(range(1, 11))
SEEDS = [1, 2, 3, 4, 5]
N_FULL = 32_329
THIS_FILES = ["scripts/62_harmonized_embed.py", "scripts/63_harmonized_matched.py", "scripts/64_harmonized_fields.py",
              "slurm/62_harmonized_A_embed.sh", "slurm/63_harmonized_A_matched.sh", "slurm/64_harmonized_A_aggregate.sh"]

# --- constants FROZEN in PREREGISTRATION_A.md (section 4); recorded, and cross-checked against the stored data ------------
PREREG = {
    "korea_full_bge_vs_qwen_ari_range": [-0.00281, 0.15910], "korea_full_ari_median": 0.00139, "korea_full_ami_median": 0.00499,
    "mimic_ari_range": [0.44684, 0.51630], "mimic_ari_median": 0.48965, "mimic_ami_median": 0.80332,
    "korea_full_bge_k_median": 3.0, "korea_full_bge_frac_k_le_4": 0.64, "korea_full_bge_largest_share": 0.9743,
    "korea_full_qwen_k_median": 24.0, "korea_full_qwen_largest_share": 0.1943,
    "g0_frac_k_le_4_max": 0.5,
    "h_a2_thresholds": {"resolved": 0.10, "unresolved": 0.50},          # [AUTHOR-CONFIRMABLE] 0.10 adopted (AUTHOR_DECISIONS)
    "h_a3_ari_threshold": 0.50,                                          # [AUTHOR-CONFIRMABLE] adopted
    "reference_values_for_h_a3": {"qwen3_first_antibiotic_ami_full_template": 0.947, "qwen3_first_antibiotic_ami_no_therapy": 0.063,
                                  "qwen3_organism_ami_no_therapy": 0.619},
}

# field (as named in 40_field_dominance / raw_fields) -> manifest row(s) whose retention decides "rendered"
FIELD_TO_MANIFEST = {
    "first_antibiotic": ["T02"], "antibiotic_set": ["T02"], "n_antibiotic_lines": ["T02"],
    "admission_cluster": ["S03_admission_cluster"], "empiric_cluster": ["S03_empiric_cluster"],
    "BSI_episode_short_clean": ["B03"], "treatment_department_eng": ["A06"], "admit_year": ["A04"],
    "n_bsi_episodes": ["B01"], "sex": ["A03"], "entry_mode": ["A05"], "hospital_code": None,
    "icu": ["S01"], "charlson_category_withcancermerged": ["C03"], "charlson_class": ["C03"], "comorbid_infectious": ["C01"],
    "index_bsi_3gcr": ["B04_3GCR"], "index_bsi_mrsa": ["B04_MRSA"], "index_bsi_vre": ["B04_VRE"], "index_bsi_carbar": ["B04_CarbaR"],
    "index_bsi_ceftriaxoner": ["B04_CeftriaxoneR"], "index_bsi_crab": ["B04_CRAB"], "index_bsi_polymicrobial": ["I03"],
    "antiMRSAab_empiric": ["S02_antiMRSAab_empiric"], "antiVREab_empiric": ["S02_antiVREab_empiric"],
    "antipseudoab_empiric": ["S02_antipseudoab_empiric"], "carbaab_empiric": ["S02_carbaab_empiric"],
    "ventilator": ["P01_ventilator"], "central_orarterial_line": ["P01_central_orarterial_line"],
    "urinary_orrenal_catheter": ["P01_urinary_orrenal_catheter"],
    "sofa_score_2 (decile)": ["O01_sofa_score_2"], "quick_sofa_score_2 (decile)": ["O01_quick_sofa_score_2"],
    "age_at_test (decile)": ["A02"], "WBC_Count (decile)": ["H01_WBC_Count"], "CRP (decile)": ["H02_CRP"],
    "T_Bilirubin (decile)": ["H01_T_Bilirubin"], "Creatinine (decile)": ["H01_Creatinine"],
    "Hemoglobin (decile)": ["H01_Hemoglobin"], "Albumin (decile)": ["H01_Albumin"],
}
EXTRA_CATEGORICAL = ["index_bsi_ceftriaxoner", "index_bsi_crab", "comorbid_infectious", "charlson_class"]
SPECIFIC_FIELDS = ["first_antibiotic", "BSI_episode_short_clean", "admission_cluster", "treatment_department_eng",
                   "n_bsi_episodes", "antibiotic_set"]


# =============================================================================================================
# generic helpers
# =============================================================================================================
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def git_info() -> dict:
    def run(*args) -> str:
        return subprocess.run(["git", "-C", str(REV), *args], capture_output=True, text=True, check=True).stdout.strip()

    st = run("status", "--porcelain", "--", *THIS_FILES)
    return {"git_commit": run("rev-parse", "HEAD"), "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(st), "git_dirty_detail": st.splitlines()[:20]}


def versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scikit-learn", "scipy"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def atomic_write_json(obj, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(json.dumps(obj, indent=2, cls=_Enc))
    os.replace(tmp, dest)


def med(v):
    return float(np.median(v)) if len(v) else None


def rng3(v):
    return {"n": int(len(v)), "median": med(v), "min": float(np.min(v)) if len(v) else None, "max": float(np.max(v)) if len(v) else None}


def fit_stats(lab: np.ndarray) -> dict:
    nn = lab[lab != -1]
    k = int(len(set(nn.tolist())))
    largest = float(np.bincount(nn).max() / len(lab)) if len(nn) else 1.0     # audit6 definition: share of ALL rows
    return {"k": k, "noise": float((lab == -1).mean()), "largest_share": largest}


# =============================================================================================================
# label loading
# =============================================================================================================
def load_ids() -> list:
    return [int(i) for i in json.load(open(IDS_JSON))]


def load_condition_labels(cond: str, model: str) -> dict:
    """{(draw, seed): (labels, subset)} for the 50 matched fits; whole-corpus {'whole': labels} when available."""
    out = {}
    for d in DRAW_IDS:
        z = np.load(DRAWS / f"draw{d:02d}.npz")
        for s in SEEDS:
            if cond == "korea_full_template":
                lab = z[f"{model}|{s}"]
            else:
                p = MATCHED / cond / model / f"d{d:02d}_s{s}.npy"
                if not p.exists():
                    return {}
                lab = np.load(p)
            out[(d, s)] = (lab, z["subset"])
    return out


def load_whole(cond: str, model: str, seed: int = 42):
    if cond == "korea_full_template":
        z = np.load(REV / "out/18_encoder_dec/recluster" / model / "labels.npz")
        return z["labels"].astype(np.int32)
    p = MATCHED / cond / model / f"whole_s{seed}.npy"
    return np.load(p) if p.exists() else None


# =============================================================================================================
# fields
# =============================================================================================================
FIELDS = None           # dict name -> pd.Series (32,329 rows, embedding order); set in build_fields_global()


def build_fields_global():
    global FIELDS
    ids = load_ids()
    frame = f40.load_korea_fields()
    frame = frame.set_index("admission_ID").loc[ids].reset_index()
    raw = pd.read_parquet(RAW_FIELDS).set_index("admission_ID").loc[ids].reset_index()
    assert list(frame["admission_ID"]) == list(raw["admission_ID"]) == ids
    extra = {"entry_mode": "admin__entry_mode", "index_bsi_3gcr": "summary__index_bsi_3gcr", "index_bsi_mrsa": "summary__index_bsi_mrsa",
             "index_bsi_vre": "summary__index_bsi_vre", "index_bsi_carbar": "summary__index_bsi_carbar",
             "index_bsi_polymicrobial": "summary__index_bsi_polymicrobial", "index_bsi_ceftriaxoner": "summary__index_bsi_ceftriaxoner",
             "index_bsi_crab": "summary__index_bsi_crab", "comorbid_infectious": "comorbid__infectious", "charlson_class": "comorbid__charlson_class"}
    for name, col in extra.items():
        if name not in frame.columns:                    # only fields 40_field_dominance could not see (column-name mismatch)
            frame[name] = raw[col].astype(str).to_numpy()
    cats = list(f40.KOREA_CATEGORICAL) + [c for c in EXTRA_CATEGORICAL if c not in f40.KOREA_CATEGORICAL]
    FIELDS = f40.build_fields(frame, cats, f40.KOREA_CONTINUOUS)
    missing = [n for n in FIELDS if n not in FIELD_TO_MANIFEST]
    if missing:
        raise SystemExit(f"[64] fields without a manifest mapping: {missing}")
    return FIELDS


def rendered_fields(cond: str, man: pd.DataFrame) -> list:
    """Field names rendered in the condition's text (from the reviewed manifest)."""
    if cond in ("korea_full_template", "thickness_matched_random"):
        return [n for n in FIELDS if FIELD_TO_MANIFEST[n] is not None]
    ret = man.set_index("field_id")[f"retained_{cond}"]
    out = []
    for n in FIELDS:
        rows = FIELD_TO_MANIFEST[n]
        if rows is not None and all(ret[r] == "yes" for r in rows):
            out.append(n)
    return out


# =============================================================================================================
# stage 1: field attribution (parallel over fits)
# =============================================================================================================
def _attr_job(job):
    cond, model, key = job
    if key == "whole":
        lab = load_whole(cond, model)
        sub = FIELDS
        subset = None
    else:
        d, s = key
        z = np.load(DRAWS / f"draw{d:02d}.npz")
        subset = z["subset"]
        lab = z[f"{model}|{s}"] if cond == "korea_full_template" else np.load(MATCHED / cond / model / f"d{d:02d}_s{s}.npy")
    fields = FIELDS if subset is None else {n: v.iloc[subset].reset_index(drop=True) for n, v in FIELDS.items()}
    return (cond, model, key, f40.field_ami(lab, fields))


def summarise_group(per_fit: list, names: list) -> dict:
    if not per_fit or not names:
        return None
    sub = [{n: s[n] for n in names} for s in per_fit]
    blk = f40.summarise_dominance(sub)
    fm = blk["per_field_ami_median"]
    order = list(fm.items())
    runner = collections.Counter()
    for s in sub:
        o = sorted(s.items(), key=lambda kv: -kv[1])
        if len(o) > 1:
            runner[o[1][0]] += 1
    blk["runner_up_by_median_ami"] = order[1][0] if len(order) > 1 else None
    blk["runner_up_median_ami"] = order[1][1] if len(order) > 1 else None
    blk["runner_up_modal_across_fits"] = runner.most_common(1)[0][0] if runner else None
    blk["top_field_by_median_ami"] = order[0][0]
    return blk


def stage_attribution(workers: int, allow_missing: bool) -> dict:
    man = pd.read_csv(MAN_CSV, keep_default_na=False)
    _bad = sorted({r for rows in FIELD_TO_MANIFEST.values() if rows for r in rows} - set(man["field_id"]))
    if _bad:
        raise SystemExit(f"[64] FIELD_TO_MANIFEST references unknown manifest ids: {_bad}")
    build_fields_global()
    print(f"[64/attr] {len(FIELDS)} fields; rendered per condition: "
          + ", ".join(f"{c}={len(rendered_fields(c, man))}" for c in CONDITIONS if c != 'mimic_iv'), flush=True)
    jobs = []
    for cond in ("korea_full_template",) + ARMS:
        for model in MODELS:
            if cond != "korea_full_template" and not (MATCHED / cond / model / "d01_s1.npy").exists():
                if allow_missing:
                    continue
                raise SystemExit(f"[64] labels for {cond}/{model} missing")
            for d in DRAW_IDS:
                for s in SEEDS:
                    jobs.append((cond, model, (d, s)))
            if load_whole(cond, model) is not None:
                jobs.append((cond, model, "whole"))
    print(f"[64/attr] {len(jobs)} partitions x {len(FIELDS)} fields; {workers} workers", flush=True)
    t0 = time.time()
    results = collections.defaultdict(dict)
    ctx = mp.get_context("fork")                       # workers inherit FIELDS; no numba/UMAP here, so fork is safe
    with ctx.Pool(processes=workers) as pool:
        for i, (cond, model, key, ami) in enumerate(pool.imap_unordered(_attr_job, jobs, chunksize=1)):
            results[(cond, model)][key] = ami
            if (i + 1) % 25 == 0:
                print(f"[64/attr] {i + 1}/{len(jobs)} done ({time.time() - t0:.0f}s)", flush=True)
    out = {"n_fields": len(FIELDS), "fields": list(FIELDS), "field_to_manifest_rows": FIELD_TO_MANIFEST,
           "conditions": {}, "seconds": round(time.time() - t0, 1)}
    for (cond, model), byk in sorted(results.items()):
        matched_keys = sorted(k for k in byk if k != "whole")
        per_fit = [byk[k] for k in matched_keys]
        rend = rendered_fields(cond, man)
        blk = {"n_matched_fits": len(per_fit), "rendered_fields": rend,
               "dominance_rendered_fields": summarise_group(per_fit, rend),
               "dominance_all_fields": summarise_group(per_fit, list(FIELDS)),
               "specific_fields_ami": {f: rng3([s[f] for s in per_fit]) for f in SPECIFIC_FIELDS if f in FIELDS},
               "whole_corpus_seed42_ami": byk.get("whole"),
               "per_fit_ami": {f"d{k[0]:02d}_s{k[1]}": v for k, v in zip(matched_keys, per_fit)}}
        out["conditions"].setdefault(cond, {})[model] = blk
    atomic_write_json({**out, "written": now_iso(), "sys_executable": sys.executable, "versions": versions(), **git_info()},
                      OUT_DIR / "field_attribution.json")
    print(f"[64/attr] wrote {OUT_DIR / 'field_attribution.json'}", flush=True)
    return out


# =============================================================================================================
# stage 2: aggregation
# =============================================================================================================
def pair_metrics(cond: str, labs: dict) -> dict:
    """labs[model] = {(d,s): (labels, subset)}; returns per-fit lists for the seed-matched pair."""
    rows = []
    for key in sorted(labs["bge-m3"]):
        la, sub = labs["bge-m3"][key]
        lb, _ = labs["qwen3-0.6b"][key]
        ari, ami = a6.agree(la, lb)
        from sklearn.metrics import adjusted_rand_score
        ka, kb = fit_stats(la)["k"], fit_stats(lb)["k"]
        rows.append({"draw": key[0], "seed": key[1], "ari": ari, "ami": ami, "ari_all_rows": float(adjusted_rand_score(la, lb)),
                     "k_bge": ka, "k_qwen": kb, "nondegenerate": bool(ka >= 5 and kb >= 5)})
    return rows


def condition_stats(cond: str, organism_all: np.ndarray, allow_missing: bool):
    labs = {m: load_condition_labels(cond, m) for m in MODELS}
    if not all(labs[m] for m in MODELS):
        if allow_missing:
            return None
        raise SystemExit(f"[64] labels for {cond} missing")
    rows = pair_metrics(cond, labs)
    per_enc = {}
    for m in MODELS:
        st = [fit_stats(labs[m][k][0]) for k in sorted(labs[m])]
        ks = [s["k"] for s in st]
        per_enc[m] = {"n_fits": len(st), "k_median": med(ks), "k_range": [min(ks), max(ks)], "frac_k_le_4": float(np.mean(np.array(ks) <= 4)),
                      "largest_share_median": med([s["largest_share"] for s in st]), "largest_share_range": [min(s["largest_share"] for s in st), max(s["largest_share"] for s in st)],
                      "noise_median": med([s["noise"] for s in st]), "k_values": ks}
    nd = [r for r in rows if r["nondegenerate"]]
    # conditional agreement within index-organism strata (n-weighted mean of per-stratum AMI/ARI, min_n = 50)
    cond_rows = []
    for key in sorted(labs["bge-m3"]):
        la, sub = labs["bge-m3"][key]
        lb, _ = labs["qwen3-0.6b"][key]
        cond_rows.append(a7.conditional_agreement(la, lb, organism_all[sub], min_n=50))
    g0_ok = all(per_enc[m]["frac_k_le_4"] <= PREREG["g0_frac_k_le_4_max"] for m in MODELS)
    return {
        "n_pairs": len(rows),
        "cross_encoder_unrestricted": {"ari": rng3([r["ari"] for r in rows]), "ami": rng3([r["ami"] for r in rows]),
                                       "ari_all_rows_median": med([r["ari_all_rows"] for r in rows])},
        "cross_encoder_nondegenerate_k_ge_5_both": {"n": len(nd), "ari": rng3([r["ari"] for r in nd]), "ami": rng3([r["ami"] for r in nd])},
        "per_encoder": per_enc,
        "g0_interpretable": bool(g0_ok), "g0_verdict": "interpretable" if g0_ok else "degeneracy-limited",
        "conditional_agreement_within_organism": {
            "n_fits": len(cond_rows), "ami_within_median": med([c["ami_weighted"] for c in cond_rows]),
            "ami_within_range": [min(c["ami_weighted"] for c in cond_rows), max(c["ami_weighted"] for c in cond_rows)],
            "ari_within_median": med([c["ari_weighted"] for c in cond_rows]),
            "ami_unconditional_median": med([c["ami_unconditional"] for c in cond_rows]),
            "n_strata_median": med([c["n_strata"] for c in cond_rows]), "n_rows_median": med([c["n_rows"] for c in cond_rows]),
            "min_n": 50},
        "pairs": rows,
    }


def mimic_stats() -> dict:
    """MIMIC-IV values recomputed from the stored 10-seed fits (both encoders, same seeds)."""
    from sklearn.metrics import adjusted_rand_score
    nm = pd.read_parquet(MIMIC_COHORT / "narrative_manifest.parquet")
    per_enc = {m: [] for m in MODELS}
    rows, cond_rows, stats_by_enc = [], [], {m: [] for m in MODELS}
    for s in range(1, 11):
        ta = pd.read_parquet(MIMIC_LABELS / f"bge-m3_seed{s}.parquet").rename(columns={"cluster": "a"})
        tb = pd.read_parquet(MIMIC_LABELS / f"qwen3-0.6b_seed{s}.parquet").rename(columns={"cluster": "b"})
        m = nm.merge(ta, on="admission_ID", validate="one_to_one").merge(tb, on="admission_ID", validate="one_to_one")
        la, lb = m["a"].to_numpy(), m["b"].to_numpy()
        ari, ami = a6.agree(la, lb)
        rows.append({"seed": s, "ari": ari, "ami": ami, "ari_all_rows": float(adjusted_rand_score(la, lb)),
                     "k_bge": fit_stats(la)["k"], "k_qwen": fit_stats(lb)["k"], "nondegenerate": fit_stats(la)["k"] >= 5 and fit_stats(lb)["k"] >= 5})
        stats_by_enc["bge-m3"].append(fit_stats(la))
        stats_by_enc["qwen3-0.6b"].append(fit_stats(lb))
        cond_rows.append(a7.conditional_agreement(la, lb, m["organism_category"].astype(str).to_numpy(), min_n=50))
    for mdl in MODELS:
        st = stats_by_enc[mdl]
        ks = [x["k"] for x in st]
        per_enc[mdl] = {"n_fits": len(st), "k_median": med(ks), "k_range": [min(ks), max(ks)], "frac_k_le_4": float(np.mean(np.array(ks) <= 4)),
                        "largest_share_median": med([x["largest_share"] for x in st]),
                        "largest_share_range": [min(x["largest_share"] for x in st), max(x["largest_share"] for x in st)],
                        "noise_median": med([x["noise"] for x in st]), "k_values": ks}
    g0_ok = all(per_enc[mm]["frac_k_le_4"] <= PREREG["g0_frac_k_le_4_max"] for mm in MODELS)
    nd = [r for r in rows if r["nondegenerate"]]
    return {"n_pairs": len(rows), "replicate_structure": "10 seeds on the full MIMIC cohort (16,144 admissions), no draws",
            "cross_encoder_unrestricted": {"ari": rng3([r["ari"] for r in rows]), "ami": rng3([r["ami"] for r in rows]),
                                           "ari_all_rows_median": med([r["ari_all_rows"] for r in rows])},
            "cross_encoder_nondegenerate_k_ge_5_both": {"n": len(nd), "ari": rng3([r["ari"] for r in nd]), "ami": rng3([r["ami"] for r in nd])},
            "per_encoder": per_enc, "g0_interpretable": bool(g0_ok), "g0_verdict": "interpretable" if g0_ok else "degeneracy-limited",
            "conditional_agreement_within_organism": {
                "n_fits": len(cond_rows), "ami_within_median": med([c["ami_weighted"] for c in cond_rows]),
                "ami_within_range": [min(c["ami_weighted"] for c in cond_rows), max(c["ami_weighted"] for c in cond_rows)],
                "ari_within_median": med([c["ari_weighted"] for c in cond_rows]),
                "ami_unconditional_median": med([c["ami_unconditional"] for c in cond_rows]),
                "n_strata_median": med([c["n_strata"] for c in cond_rows]), "n_rows_median": med([c["n_rows"] for c in cond_rows]), "min_n": 50},
            "pairs": rows}


def zone(r, korea_max, mimic_min):
    if r is None:
        return None
    return 1 if r <= korea_max else (2 if r < mimic_min else 3)


def score_vs_no_therapy(cond: str) -> dict:
    """PREREG 4.3(1): both-non-noise ARI between the Qwen3 labels of `cond` and the stored Qwen3 no_therapy labels."""
    out = {"matched_fits": None, "whole_corpus": None}
    vals = []
    for d in DRAW_IDS:
        z = np.load(DRAWS / f"draw{d:02d}.npz")
        sub = z["subset"]
        for s in SEEDS:
            nt = np.load(NO_THERAPY / f"no_therapy__seed{s}.npy")
            lab = z[f"qwen3-0.6b|{s}"] if cond == "korea_full_template" else np.load(MATCHED / cond / "qwen3-0.6b" / f"d{d:02d}_s{s}.npy")
            ari, ami = a6.agree(lab, nt[sub])
            vals.append((ari, ami))
    out["matched_fits"] = {"n": len(vals), "ari": rng3([v[0] for v in vals]), "ami": rng3([v[1] for v in vals]),
                           "note": "matched fit (draw d, seed s) on the draw subset vs the stored whole-corpus no_therapy labels of the same seed restricted to that subset"}
    w = load_whole(cond, "qwen3-0.6b")
    if w is not None:
        nt = np.load(NO_THERAPY / "no_therapy__seed42.npy")
        ari, ami = a6.agree(w, nt)
        out["whole_corpus"] = {"seed": 42, "n": int(len(w)), "ari": ari, "ami": ami}
    return out


def stage_aggregate(allow_missing: bool) -> None:
    man = pd.read_csv(MAN_CSV, keep_default_na=False)
    attr_p = OUT_DIR / "field_attribution.json"
    attr = json.loads(attr_p.read_text()) if attr_p.exists() else None
    build_fields_global()
    organism_all = FIELDS["BSI_episode_short_clean"].astype(str).to_numpy()
    stored_enc = json.loads(STORED_ENC.read_text())
    stored_dom = json.loads(STORED_DOM.read_text())
    stored_cond = json.loads(STORED_COND.read_text())

    stats = {}
    for cond in CONDITIONS:
        stats[cond] = mimic_stats() if cond == "mimic_iv" else condition_stats(cond, organism_all, allow_missing)
        if stats[cond] is None:
            print(f"[64/agg] {cond}: labels missing, skipped (allow-missing)", flush=True)
    stats = {k: v for k, v in stats.items() if v is not None}

    # --- cross-checks of the recomputed Korean-full / MIMIC anchors against the stored files and the frozen prereg numbers
    kf = stats["korea_full_template"]
    stored_pair = stored_enc["cross_encoder_seed_matched"]["bge-m3 vs qwen3-0.6b"]
    mim = stats["mimic_iv"]
    checks = {
        "korea_full_ari_median_recomputed_vs_stored_matched_encoders_json": [kf["cross_encoder_unrestricted"]["ari"]["median"], stored_pair["ari"]["median"]],
        "korea_full_ari_max_recomputed_vs_stored": [kf["cross_encoder_unrestricted"]["ari"]["max"], stored_pair["ari"]["max"]],
        "korea_full_ami_median_recomputed_vs_stored": [kf["cross_encoder_unrestricted"]["ami"]["median"], stored_pair["ami"]["median"]],
        "korea_full_bge_k_median_vs_stored": [kf["per_encoder"]["bge-m3"]["k_median"], stored_enc["k_per_fit"]["bge-m3"]["median"]],
        "korea_full_bge_frac_k_le_4_vs_stored": [kf["per_encoder"]["bge-m3"]["frac_k_le_4"], stored_enc["k_per_fit"]["bge-m3"]["frac_k_le_4"]],
        "korea_full_bge_largest_share_vs_stored": [kf["per_encoder"]["bge-m3"]["largest_share_median"], stored_enc["largest_cluster_share"]["bge-m3"]["median"]],
        "korea_full_within_organism_ami_vs_stored_dominance_checks": [kf["conditional_agreement_within_organism"]["ami_within_median"],
                                                                      stored_cond["korea_conditional_agreement"]["bge-m3 vs qwen3-0.6b"]["organism"]["ami_within_median"]],
        "mimic_ari_median_recomputed_vs_prereg_0.48965": [mim["cross_encoder_unrestricted"]["ari"]["median"], PREREG["mimic_ari_median"]],
        "mimic_ari_min_recomputed_vs_prereg_0.44684": [mim["cross_encoder_unrestricted"]["ari"]["min"], PREREG["mimic_ari_range"][0]],
        "mimic_ami_median_recomputed_vs_prereg_0.80332": [mim["cross_encoder_unrestricted"]["ami"]["median"], PREREG["mimic_ami_median"]],
        "mimic_within_organism_ami_vs_stored_dominance_checks": [mim["conditional_agreement_within_organism"]["ami_within_median"],
                                                                 stored_cond["mimic_conditional_agreement"]["bge-m3 vs qwen3-0.6b"]["organism"]["ami_within_median"]],
    }
    tol = {"mimic_ari_median_recomputed_vs_prereg_0.48965": 5e-5, "mimic_ari_min_recomputed_vs_prereg_0.44684": 5e-5, "mimic_ami_median_recomputed_vs_prereg_0.80332": 5e-5}
    checks_out = {k: {"recomputed": v[0], "reference": v[1], "abs_diff": abs(v[0] - v[1]), "ok": bool(abs(v[0] - v[1]) <= tol.get(k, 1e-9))} for k, v in checks.items()}
    korea_max = kf["cross_encoder_unrestricted"]["ari"]["max"]           # Korea's own observed maximum across its 50 fits
    mimic_min = mim["cross_encoder_unrestricted"]["ari"]["min"]           # MIMIC's observed minimum
    korea_max_ami = kf["cross_encoder_unrestricted"]["ami"]["max"]
    mimic_min_ami = mim["cross_encoder_unrestricted"]["ami"]["min"]

    # --- decision-rule outputs (mechanical; no prose interpretation)
    rules = {"anchors": {"korea_full_ari_max": korea_max, "mimic_ari_min": mimic_min, "korea_full_ami_max_extremes_reading": korea_max_ami,
                         "mimic_ami_min_extremes_reading": mimic_min_ami, "prereg_ami_median_anchors": [PREREG["korea_full_ami_median"], PREREG["mimic_ami_median"]],
                         "ami_zone_note": "PREREG 4.2 says AMI uses 'the identical three-zone rule against 0.00499 and 0.80332' (medians). The identical rule is applied here to the "
                                          "extremes (Korea max, MIMIC min); the position relative to the two medians is recorded too. Both are reported."},
             "by_arm": {}}
    for arm in ARMS:
        if arm not in stats:
            continue
        s = stats[arm]
        unr = s["cross_encoder_unrestricted"]
        ndg = s["cross_encoder_nondegenerate_k_ge_5_both"]
        interpretable = s["g0_interpretable"]
        prim_ari = unr["ari"]["median"] if interpretable else (ndg["ari"]["median"] if ndg["n"] else None)
        prim_ami = unr["ami"]["median"] if interpretable else (ndg["ami"]["median"] if ndg["n"] else None)
        f_bge = s["per_encoder"]["bge-m3"]["frac_k_le_4"]
        r = {
            "g0_verdict": s["g0_verdict"], "primary_statistic_basis": "unrestricted 50 pairs" if interpretable else "pair-level non-degenerate subset (k>=5 in both)",
            "primary_statistic_n_pairs": s["n_pairs"] if interpretable else ndg["n"],
            "R_unrestricted_median_ari": unr["ari"]["median"], "R_nondegenerate_median_ari": ndg["ari"]["median"], "n_nondegenerate": ndg["n"],
            "R_primary_median_ari": prim_ari, "A_primary_median_ami": prim_ami,
            "zone_ari_primary": zone(prim_ari, korea_max, mimic_min), "zone_ari_unrestricted": zone(unr["ari"]["median"], korea_max, mimic_min),
            "zone_ari_nondegenerate": zone(ndg["ari"]["median"], korea_max, mimic_min) if ndg["n"] else None,
            "zone_ami_extremes_reading_primary": zone(prim_ami, korea_max_ami, mimic_min_ami),
            "ami_relative_to_prereg_median_anchors": (None if prim_ami is None else
                                                      ("<=korea_median" if prim_ami <= PREREG["korea_full_ami_median"] else ("<mimic_median" if prim_ami < PREREG["mimic_ami_median"] else ">=mimic_median"))),
            "h_a2_bge_frac_k_le_4": f_bge, "h_a2_band": ("resolved(<=0.10)" if f_bge <= PREREG["h_a2_thresholds"]["resolved"] else
                                                          ("attenuates_not_resolved(0.10<f<0.50)" if f_bge < PREREG["h_a2_thresholds"]["unresolved"] else "remains_degenerate(>=0.50)")),
            "h_a2_bge_k_median": s["per_encoder"]["bge-m3"]["k_median"], "h_a2_bge_largest_share_median": s["per_encoder"]["bge-m3"]["largest_share_median"],
        }
        rules["by_arm"][arm] = r
    prim = rules["by_arm"].get("korean_mimic_compatible")
    keep = rules["by_arm"].get("harmonized_keep_therapy")
    rnd = rules["by_arm"].get("thickness_matched_random")
    if prim:
        zh = prim["zone_ari_primary"]
        rules["rule_4_3_applicable_zone_2_or_3_reached"] = bool(zh in (2, 3))
        nt = {c: score_vs_no_therapy(c) for c in ("korea_full_template",) + ARMS if (c == "korea_full_template" or c in stats)}
        rules["rule_4_3_1_vs_no_therapy"] = {c: v for c, v in nt.items()}
        rules["rule_4_3_1_primary_arm"] = {
            "median_ari_matched_fits": nt["korean_mimic_compatible"]["matched_fits"]["ari"]["median"],
            "threshold": PREREG["h_a3_ari_threshold"],
            "branch": ("median_ari_ge_threshold" if nt["korean_mimic_compatible"]["matched_fits"]["ari"]["median"] >= PREREG["h_a3_ari_threshold"] else "median_ari_lt_threshold"),
            "whole_corpus_seed42_ari": (nt["korean_mimic_compatible"]["whole_corpus"] or {}).get("ari")}
        if keep:
            zk = keep["zone_ari_primary"]
            rk, rh = keep["R_primary_median_ari"], prim["R_primary_median_ari"]
            if rk is not None and rh is not None and rk > rh:
                br = "R_keep_above_R_harm"
            elif zk == 1 and zh in (2, 3):
                br = "R_keep_zone1_while_R_harm_zone2_or_3"
            elif zk == zh:
                br = "R_keep_same_zone_as_R_harm"
            else:
                br = "other(R_keep_not_above_R_harm; zones differ but not the zone1-vs-2/3 pattern)"
            rules["rule_4_3_2_vs_keep_therapy"] = {"R_keep_median_ari": rk, "R_harm_median_ari": rh, "zone_keep": zk, "zone_harm": zh, "branch": br,
                                                   "h_a4_R_keep_lower_than_R_harm": (None if rk is None or rh is None else bool(rk < rh))}
        if rnd:
            zr = rnd["zone_ari_primary"]
            rules["rule_4_4_thickness"] = {"zone_random": zr, "zone_harm": zh, "R_random_median_ari": rnd["R_primary_median_ari"], "R_harm_median_ari": prim["R_primary_median_ari"],
                                           "branch": ("random_same_zone_as_harmonized" if zr == zh else ("random_zone1_harmonized_higher" if zr == 1 and zh in (2, 3) else "other"))}
    # --- comparison table (PREREG 4.5) ---------------------------------------------------------------------------------
    rows = []
    for cond in CONDITIONS:
        if cond not in stats:
            continue
        s = stats[cond]
        unr, ndg, ca = s["cross_encoder_unrestricted"], s["cross_encoder_nondegenerate_k_ge_5_both"], s["conditional_agreement_within_organism"]
        row = {"condition": cond, "n_cross_encoder_pairs": s["n_pairs"], "replicate_structure": ("10 seeds, full cohort, no draws" if cond == "mimic_iv" else "10 draws x 5 seeds (n=16,144)"),
               "cross_ari_median_unrestricted": unr["ari"]["median"], "cross_ari_min_unrestricted": unr["ari"]["min"], "cross_ari_max_unrestricted": unr["ari"]["max"],
               "cross_ami_median_unrestricted": unr["ami"]["median"], "cross_ami_min_unrestricted": unr["ami"]["min"], "cross_ami_max_unrestricted": unr["ami"]["max"],
               "n_pairs_nondegenerate_k_ge_5_both": ndg["n"], "cross_ari_median_nondegenerate": ndg["ari"]["median"], "cross_ari_min_nondegenerate": ndg["ari"]["min"],
               "cross_ari_max_nondegenerate": ndg["ari"]["max"], "cross_ami_median_nondegenerate": ndg["ami"]["median"],
               "cross_ami_min_nondegenerate": ndg["ami"]["min"], "cross_ami_max_nondegenerate": ndg["ami"]["max"],
               "within_organism_ami_median": ca["ami_within_median"], "within_organism_ari_median": ca["ari_within_median"],
               "within_organism_ami_unconditional_median": ca["ami_unconditional_median"], "within_organism_n_strata_median": ca["n_strata_median"],
               "within_organism_n_rows_median": ca["n_rows_median"], "degeneracy_verdict_G0": s["g0_verdict"]}
        for m, short in (("bge-m3", "bge"), ("qwen3-0.6b", "qwen")):
            pe = s["per_encoder"][m]
            row.update({f"{short}_k_median": pe["k_median"], f"{short}_frac_k_le_4": pe["frac_k_le_4"], f"{short}_largest_share_median": pe["largest_share_median"],
                        f"{short}_noise_median": pe["noise_median"]})
            if cond == "mimic_iv":
                d = stored_dom["cohorts"]["mimic"][m]
                fm = list(d["per_field_ami_median"].items())
                row.update({f"{short}_dominant_field": d["modal_dominant_field"], f"{short}_dominant_ami_median": d["top_ami_median"],
                            f"{short}_runner_up_field": fm[1][0], f"{short}_runner_up_ami_median": fm[1][1],
                            f"{short}_dominance_margin_median": d["margin_over_runner_up_median"], f"{short}_top_field_consistency": d["modal_share"],
                            f"{short}_dominance_source": "stored out/40_field_dominance (MIMIC field set)"})
            elif attr and cond in attr["conditions"] and m in attr["conditions"][cond]:
                blk = attr["conditions"][cond][m]
                d = blk["dominance_rendered_fields"]
                row.update({f"{short}_dominant_field": d["modal_dominant_field"], f"{short}_dominant_ami_median": d["top_ami_median"],
                            f"{short}_runner_up_field": d["runner_up_by_median_ami"], f"{short}_runner_up_ami_median": d["runner_up_median_ami"],
                            f"{short}_dominance_margin_median": d["margin_over_runner_up_median"], f"{short}_top_field_consistency": d["modal_share"],
                            f"{short}_dominance_source": "recomputed here over RENDERED fields",
                            f"{short}_dominant_all_fields": blk["dominance_all_fields"]["modal_dominant_field"],
                            f"{short}_dominant_all_fields_ami_median": blk["dominance_all_fields"]["top_ami_median"],
                            f"{short}_first_antibiotic_ami_median": blk["specific_fields_ami"]["first_antibiotic"]["median"],
                            f"{short}_organism_ami_median": blk["specific_fields_ami"]["BSI_episode_short_clean"]["median"]})
            if cond == "korea_full_template":
                d0 = stored_dom["cohorts"]["korea_matched_refits"][m]
                row.update({f"{short}_stored_dominant_field_all_stored_fields": d0["modal_dominant_field"], f"{short}_stored_top_ami_median": d0["top_ami_median"],
                            f"{short}_stored_top_field_consistency": d0["modal_share"], f"{short}_stored_margin_median": d0["margin_over_runner_up_median"]})
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_DIR / "comparison_table.csv", index=False)

    meta = {"written": now_iso(), "sys_executable": sys.executable, "versions": versions(), **git_info(),
            "seeds": SEEDS, "draws": DRAW_IDS, "n_matched": a6.MIMIC_N, "min_cluster_size_matched": a6.min_cluster_size(a6.MIMIC_N),
            "umap_n_neighbors": a6.UMAP_NEIGHBORS, "noise_convention": "both-non-noise ARI/AMI (audit6_matched_encoders.agree)", "degeneracy_rule": "k >= 5 in both partitions of a pair; encoder-level frac_k_le_4",
            "prereg_constants": PREREG, "conditions_computed": list(stats), "cross_checks_recomputed_vs_stored": checks_out,
            "allow_missing": allow_missing, "attribution_json_present": attr is not None,
            "stored_sources": {"matched_encoders": str(STORED_ENC), "field_dominance": str(STORED_DOM), "dominance_checks": str(STORED_COND)}}
    atomic_write_json({**meta, "stats": {c: {k: v for k, v in s.items() if k != "pairs"} for c, s in stats.items()},
                       "pairs": {c: s["pairs"] for c, s in stats.items()}}, OUT_DIR / "matched_encoders.json")
    atomic_write_json({**meta, "conditional_agreement": {c: s["conditional_agreement_within_organism"] for c, s in stats.items()}}, OUT_DIR / "conditional_agreement.json")
    atomic_write_json({**meta, "rules": rules}, OUT_DIR / "hypothesis_rules.json")
    bad = [k for k, v in checks_out.items() if not v["ok"]]
    print(json.dumps({"conditions": list(stats), "cross_check_failures": bad, "comparison_table": str(OUT_DIR / "comparison_table.csv")}, indent=2), flush=True)
    if bad:
        print(f"[64/agg] WARNING: {len(bad)} cross-check(s) differ from stored/prereg values: {bad}", flush=True)


def main() -> None:
    global OUT_DIR, MATCHED
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["attribution", "aggregate", "all"], required=True)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--allow-missing", action="store_true", help="(testing) skip conditions whose labels do not exist")
    ap.add_argument("--out-dir", type=Path, default=A_DIR, help="(testing) override the output directory")
    ap.add_argument("--matched-root", type=Path, default=MATCHED, help="(testing) override the root of the harmonized-arm partitions")
    args = ap.parse_args()
    OUT_DIR, MATCHED = args.out_dir, args.matched_root
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    if args.stage in ("attribution", "all"):
        stage_attribution(workers, args.allow_missing)
    if args.stage in ("aggregate", "all"):
        stage_aggregate(args.allow_missing)


if __name__ == "__main__":
    main()
