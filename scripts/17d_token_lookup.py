#!/usr/bin/env python3
"""Task 17, token-lookup baseline (the "robust <=> token" dissociation
addendum) -- exactly `out/12_audit/second_audit/g4_perclass.py`'s method
(StratifiedGroupKFold(5, groups=patient_ID), per-fold majority-vote
crosstab lookup, no embedding at all), but scored against the ACTUAL
published 16-cluster partition (`out_pre/phenotype_clusters_labels.csv`)
instead of that audit's reference.

Progressive key combinations (same three as g4_perclass.py, read and
reused): organism only; organism+hospital+year; organism+hospital+year+
department. Reports per-published-cluster F1 for each, plus the BEST
(max) across the three per PRESPEC's "F1_lookup(c)" usage.

No embeddings are loaded -- this needs only
`out_pre/phenotype_clusters_labels.csv` + `manuscript/phenotype_ml_dataset.csv`
(via `17_common`) -- so it runs directly in the bsi_pheno interpreter, no
SLURM submission needed (a few seconds).

Outcome-blind (Global Constraints): no mortality/discharge/death column is
read.
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

OUT_DIR = c17.OUT / "token_lookup"


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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = c17.build_full_frame()
    pub = frame[c17.PUBLISHED_CLUSTER_COL].to_numpy()
    keep = pub != -1
    y = pub[keep]
    groups = frame["patient_ID"].to_numpy()[keep]
    org = frame["BSI_episode_short_clean"].astype(str).to_numpy()[keep]
    hosp = frame["hospital_code"].astype(str).to_numpy()[keep]
    year = frame["admit_year"].astype(str).to_numpy()[keep]
    dept = frame["treatment_department_eng"].astype(str).to_numpy()[keep]

    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(len(y)), y, groups))
    labels_sorted = sorted(set(y.tolist()))

    combos = {
        "lookup_organism": org,
        "lookup_organism_hospital_year": np.array([f"{a}|{b}|{c}" for a, b, c in zip(org, hosp, year)]),
        "lookup_organism_hospital_year_dept": np.array(
            [f"{a}|{b}|{c}|{d}" for a, b, c, d in zip(org, hosp, year, dept)]
        ),
    }

    res = {}
    for name, key in combos.items():
        res[name] = score(*lookup_eval(key, y, folds), labels_sorted)
        print(name, {k: v for k, v in res[name].items() if k != "per_cluster_f1"}, flush=True)

    per_cluster = pd.DataFrame({name: r["per_cluster_f1"] for name, r in res.items()})
    per_cluster["best_lookup_f1"] = per_cluster.max(axis=1)
    per_cluster["best_lookup_combo"] = per_cluster[list(combos.keys())].idxmax(axis=1)
    per_cluster.index.name = "published_cluster"
    per_cluster = per_cluster.reset_index()

    (OUT_DIR / "g4_perclass_published.json").write_text(json.dumps(res, indent=2))
    per_cluster.to_csv(OUT_DIR / "g4_perclass_published.csv", index=False)
    print(per_cluster.round(3).to_string())
    print(f"[token_lookup] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
