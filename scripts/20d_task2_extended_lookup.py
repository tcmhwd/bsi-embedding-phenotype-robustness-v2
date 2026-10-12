#!/usr/bin/env python3
"""Task 2 -- extended token-lookup baseline + corrected S6 verdict.

`17d_token_lookup.py` (read and reused: `lookup_eval`/`score` copied
verbatim) rebuilt with the variables the evidence and the manuscript
itself name -- episode count, index->discharge day band, ICU,
ventilation, department, organism, hospital, year, plus narrative length
band -- instead of the original battery's organism/+hospital/+year/
+department only.

Same protocol as `out/12_audit/second_audit/g4_perclass.py` (read and
reused): StratifiedGroupKFold(5, groups=patient_ID), noise excluded,
per-published-cluster macro-F1 from a per-fold majority-vote crosstab
lookup (no embedding).

Cumulative combos, in the order the brief lists the variables (episode
count first, since that is the mechanism cluster 0's own audit trail
points to): each combo adds one more key to the join-key string. The
per-cluster "best_lookup_f1" is the max F1 across all 9 cumulative combos
-- directly comparable to (and a replacement for) `17d_token_lookup.py`'s
/ `survival_matrix.csv`'s `best_lookup_f1` column, which used only 3 combos
and none of the 6 new variables.

Inputs (`17_common`/`18_common`, read-only):
  - `pipeline_new/out_pre/phenotype_clusters_labels.csv` (published labels)
  - `manuscript/phenotype_ml_dataset.csv` (patient_ID, hospital_code,
    treatment_department_eng, BSI_episode_short_clean, admit_year, icu,
    ventilator, n_bsi_episodes, length_of_stay,
    days_from_admission_to_index_bsi -- no outcome column)
  - `reports_pilot_pre/*.report.md` (narrative byte length only, for the
    length band -- via `18_common.load_texts`)

Then recomputes the S6 verdict (PRESPEC rule verbatim, from `18f_findings.py`:
PASS iff in >=2 of 3 encoders, F1_enc(c) >= 0.60 AND F1_enc(c) >=
F1_lookup(c) + 0.10) for all 16 clusters using this CORRECTED lookup
in place of `out/18_encoder_dec/s6_matrix.csv`'s original `best_lookup_f1`
column. Encoder F1 values themselves are reused verbatim from that file,
never recomputed.

Outcome-blind (Global Constraints): no mortality/discharge/death column is
read anywhere in this module.

No SLURM needed -- runs directly in the bsi_pheno interpreter (a few
seconds; no embeddings are loaded).
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c17 = importlib.import_module("17_common")
c18 = importlib.import_module("18_common")

OUT_DIR = c18.OUT.parent / "20_onehot"
DATASET = c17.DATASET

# S6 thresholds read from `prespec_thresholds.json` (a single JSON block
# shared with `22a_ecoli_common.py`'s S1/S2/S4/S5, so a reviewer can see
# that one file governs every gate -- AUDIT_3 N9 / AUDIT_4 M6).
# Not hand-typed here any more; the values themselves are unchanged (0.60 /
# 0.10 / 2), copied verbatim from PRESPEC section 3 same as before.
with open(SCRIPT_DIR / "prespec_thresholds.json") as _f:
    _ALL_THRESHOLDS = json.load(_f)
F1_PASS_MIN = _ALL_THRESHOLDS["S6"]["f1_min"]
F1_MARGIN_OVER_LOOKUP = _ALL_THRESHOLDS["S6"]["f1_margin_over_lookup"]
S6_MIN_ENCODERS_PASSING = _ALL_THRESHOLDS["S6"]["min_encoders_passing"]
print(f"[20d] S6 thresholds loaded from prespec_thresholds.json: "
      f"f1_min={F1_PASS_MIN}, f1_margin_over_lookup={F1_MARGIN_OVER_LOOKUP}, "
      f"min_encoders_passing={S6_MIN_ENCODERS_PASSING}", flush=True)
ENCODERS = ["bge-m3", "qwen3-0.6b", "openai-3small"]


def lookup_eval(key, y, folds):
    yt, yp = [], []
    for tr_, te_ in folds:
        tab = pd.crosstab(key[tr_], y[tr_])
        maj = tab.idxmax(axis=1)
        glob = pd.Series(y[tr_]).value_counts().idxmax()
        yp.append(pd.Series(key[te_]).map(maj).fillna(glob).astype(int).to_numpy())
        yt.append(y[te_])
    return np.concatenate(yt), np.concatenate(yp)


def score(yt, yp, labels_sorted):
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labels_sorted, zero_division=0)
    return dict(
        macro_f1=float(f1_score(yt, yp, average="macro")),
        per_cluster_f1={int(c): float(ff) for c, ff in zip(labels_sorted, f)},
    )


def qbin(x: pd.Series, q: int = 4, label_prefix: str = "") -> pd.Series:
    """Quantile-bin a numeric series into `q` bands, NaN as its own band
    (matches project convention: missingness is its own level, not
    imputed away)."""
    finite = x.dropna()
    try:
        cats = pd.qcut(finite, q=q, duplicates="drop")
    except ValueError:
        cats = pd.qcut(finite, q=min(q, finite.nunique()), duplicates="drop")
    out = pd.Series(index=x.index, dtype=object)
    out.loc[finite.index] = [f"{label_prefix}{iv}" for iv in cats.astype(str)]
    out.loc[x.isna()] = f"{label_prefix}NA"
    return out


def build_extended_frame() -> pd.DataFrame:
    frame = c17.build_full_frame()  # admission_ID, newpatient_ID, phenotype_cluster, patient_ID, hospital_code,
    # treatment_department_eng, admit_year, BSI_episode_short_clean -- row order = out_pre order
    ds = pd.read_csv(
        DATASET,
        usecols=["admission_ID", "icu", "ventilator", "n_bsi_episodes",
                 "length_of_stay", "days_from_admission_to_index_bsi"],
        low_memory=False,
    ).drop_duplicates(subset="admission_ID")
    frame = frame.merge(ds, on="admission_ID", how="left")
    if frame["icu"].isna().any() or frame["ventilator"].isna().any():
        raise RuntimeError("icu/ventilator merge produced NaN -- admission_ID join incomplete")

    # narrative byte length -> band (document-level, from reports_pilot_pre, aligned by admission_ID)
    texts = c18.load_texts(admission_ids=frame["admission_ID"].to_numpy())
    texts["byte_length"] = texts["text"].map(lambda t: len(t.encode("utf-8")))
    frame = frame.merge(texts[["admission_ID", "byte_length"]], on="admission_ID", how="left")
    if frame["byte_length"].isna().any():
        raise RuntimeError("byte_length merge produced NaN")

    frame["day_band_index_to_discharge"] = qbin(
        frame["length_of_stay"] - frame["days_from_admission_to_index_bsi"], q=4, label_prefix="day:"
    )
    frame["narrative_length_band"] = qbin(frame["byte_length"], q=4, label_prefix="len:")
    frame["episode_count"] = frame["n_bsi_episodes"].astype(str)
    frame["icu_str"] = frame["icu"].astype(str)
    frame["ventilator_str"] = frame["ventilator"].astype(str)
    return frame


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = build_extended_frame()
    pub = frame[c17.PUBLISHED_CLUSTER_COL].to_numpy()
    keep = pub != -1
    y = pub[keep]
    groups = frame["patient_ID"].to_numpy()[keep]

    ep = frame["episode_count"].to_numpy()[keep]
    day = frame["day_band_index_to_discharge"].astype(str).to_numpy()[keep]
    icu = frame["icu_str"].to_numpy()[keep]
    vent = frame["ventilator_str"].to_numpy()[keep]
    dept = frame["treatment_department_eng"].astype(str).to_numpy()[keep]
    org = frame["BSI_episode_short_clean"].astype(str).to_numpy()[keep]
    hosp = frame["hospital_code"].astype(str).to_numpy()[keep]
    year = frame["admit_year"].astype(str).to_numpy()[keep]
    length_band = frame["narrative_length_band"].astype(str).to_numpy()[keep]

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(len(y)), y, groups))
    labels_sorted = sorted(set(y.tolist()))

    # Cumulative combos, in the brief's listed variable order.
    ordered_vars = [
        ("episode_count", ep), ("day_band", day), ("icu", icu), ("ventilation", vent),
        ("department", dept), ("organism", org), ("hospital", hosp), ("year", year),
        ("narrative_length_band", length_band),
    ]
    combos = {}
    acc = None
    name_parts = []
    for name, arr in ordered_vars:
        name_parts.append(name)
        acc = arr.astype(str) if acc is None else np.char.add(np.char.add(acc, "|"), arr.astype(str))
        combos["+".join(name_parts)] = acc.copy()

    res = {}
    for name, key in combos.items():
        res[name] = score(*lookup_eval(key, y, folds), labels_sorted)
        print(name, {k: v for k, v in res[name].items() if k != "per_cluster_f1"}, flush=True)

    per_cluster = pd.DataFrame({name: r["per_cluster_f1"] for name, r in res.items()})
    per_cluster["best_lookup_f1_corrected"] = per_cluster.max(axis=1)
    per_cluster["best_lookup_combo_corrected"] = per_cluster[list(combos.keys())].idxmax(axis=1)
    per_cluster.index.name = "published_cluster"
    per_cluster = per_cluster.reset_index()

    c17.atomic_write_json(res, OUT_DIR / "task2_extended_lookup.json")
    c17.atomic_write_csv(per_cluster, OUT_DIR / "task2_extended_lookup.csv")
    print(per_cluster[["published_cluster", "best_lookup_f1_corrected", "best_lookup_combo_corrected"]]
          .round(4).to_string(index=False))
    print(f"[20d] wrote {OUT_DIR / 'task2_extended_lookup.csv'}")

    # -------------------------------------------------------------
    # Corrected S6 verdict: reuse encoder F1s verbatim from s6_matrix.csv,
    # replace only best_lookup_f1 with the corrected value above.
    # -------------------------------------------------------------
    s6 = pd.read_csv(c18.OUT / "s6_matrix.csv")
    corrected_lookup = dict(zip(per_cluster["published_cluster"], per_cluster["best_lookup_f1_corrected"]))
    corrected_combo = dict(zip(per_cluster["published_cluster"], per_cluster["best_lookup_combo_corrected"]))

    rows = []
    for _, r in s6.iterrows():
        c = int(r["cluster"])
        row = {"cluster": c, "size": int(r["size"]),
               "best_lookup_f1_original": float(r["best_lookup_f1"]),
               "best_lookup_f1_corrected": float(corrected_lookup[c]),
               "best_lookup_combo_corrected": corrected_combo[c]}
        pass_count = 0
        n_measured = 0
        for enc in ENCODERS:
            f1 = r.get(f"F1_{enc}")
            row[f"F1_{enc}"] = f1
            if pd.notna(f1):
                n_measured += 1
                enc_pass = (f1 >= F1_PASS_MIN) and (f1 >= corrected_lookup[c] + F1_MARGIN_OVER_LOOKUP)
                row[f"S6_pass_{enc}_corrected"] = bool(enc_pass)
                pass_count += int(enc_pass)
            else:
                row[f"S6_pass_{enc}_corrected"] = None
        row["S6_pass_original"] = bool(r["S6_pass"]) if pd.notna(r["S6_pass"]) else None
        row["S6_pass_corrected"] = (pass_count >= S6_MIN_ENCODERS_PASSING) if n_measured == len(ENCODERS) else None
        rows.append(row)

    s6_corrected = pd.DataFrame(rows)
    c17.atomic_write_csv(s6_corrected, OUT_DIR / "task2_s6_corrected.csv")

    n_pass_orig = int(s6_corrected["S6_pass_original"].fillna(False).sum())
    n_pass_corr = int(s6_corrected["S6_pass_corrected"].fillna(False).sum())
    cluster0 = s6_corrected[s6_corrected["cluster"] == 0].iloc[0]

    print()
    print(s6_corrected[["cluster", "size", "best_lookup_f1_original", "best_lookup_f1_corrected",
                         "S6_pass_original", "S6_pass_corrected"]].round(4).to_string(index=False))
    print()
    print(f"S6 pass count -- ORIGINAL lookup: {n_pass_orig}/16; CORRECTED lookup: {n_pass_corr}/16")
    print(f"Cluster 0: best_lookup_f1 original={cluster0['best_lookup_f1_original']:.4f} -> "
          f"corrected={cluster0['best_lookup_f1_corrected']:.4f} "
          f"(combo={cluster0['best_lookup_combo_corrected']}); "
          f"S6_pass original={cluster0['S6_pass_original']} -> corrected={cluster0['S6_pass_corrected']}")

    summary = {
        "n_s6_pass_original": n_pass_orig, "n_s6_pass_corrected": n_pass_corr,
        "cluster0_best_lookup_f1_original": float(cluster0["best_lookup_f1_original"]),
        "cluster0_best_lookup_f1_corrected": float(cluster0["best_lookup_f1_corrected"]),
        "cluster0_best_lookup_combo_corrected": cluster0["best_lookup_combo_corrected"],
        "cluster0_S6_pass_original": cluster0["S6_pass_original"],
        "cluster0_S6_pass_corrected": cluster0["S6_pass_corrected"],
    }
    c17.atomic_write_json(summary, OUT_DIR / "task2_summary.json")
    print(f"[20d] wrote {OUT_DIR / 'task2_s6_corrected.csv'} and task2_summary.json")


if __name__ == "__main__":
    main()
