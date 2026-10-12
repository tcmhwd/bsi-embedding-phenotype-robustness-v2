#!/usr/bin/env python3
"""Token recoverability of the clusters, both cohorts, done correctly.

Supersedes the extended-lookup arm of `31_mimic_extended_lookup.py`, whose
conclusion ("MIMIC clusters are not token-recoverable") was an artefact of
the estimator rather than a property of the data.

WHAT WENT WRONG, so it is not repeated.  A majority-class-per-cell lookup
predicts a test row by the majority label of its exact token combination in
the training folds.  That works while cells are populous.  Once the number
of distinct cells approaches the number of rows, most test rows land in a
combination the training folds never contained, and fall back to the global
majority.  In the MIMIC extended lookup, 11,090 cells over 14,957 rows put
**66-70% of test rows in unseen cells**, so macro-F1 was capped near 0.3-0.4
by construction.  The earlier reading -- "the cell count is saturated,
therefore the low F1 is not a capacity artefact" -- inverts the logic:
saturation is precisely what makes the lookup unable to generalise.

Accuracy on the rows whose cell WAS seen is 0.86 (BGE-M3) and 0.98 (Qwen3):
the tokens do determine the cluster; the lookup just cannot carry that
information across unseen combinations.

So this script reports three things per cohort and token set:
  - the lookup (kept, because the pre-specified S6 criterion uses it),
  - the fraction of test rows in unseen cells (the diagnostic that says
    whether the lookup is trustworthy at that cell count),
  - a gradient-boosted classifier over the SAME token fields, which
    generalises across cells and therefore answers the actual question,
    "is cluster membership a function of the rendered tokens?"

Both cohorts get the same treatment so the comparison is like-for-like --
the earlier double-dissociation table compared Korean per-cluster F1 for
seven organism-defined clusters against MIMIC macro-F1 over 38-62 clusters,
which are not the same quantity.

Also adds the four rendered empiric-spectrum flags that `31`'s EXTENDED
omitted (`27c_mimic_narratives.py` SPECTRUM_FIELDS).
"""
import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder

ROOT = Path("/data/tcmhwd/AI/AST_ai")
REV = ROOT / "phenotype_revision"
MIMIC_COHORT = REV / "out/27_mimic/cohort"
MIMIC_LABELS = REV / "out/27_mimic/replicate/labels"
PUBLISHED = ROOT / "pipeline_new/out_pre/phenotype_clusters_labels.csv"
DATASET = ROOT / "manuscript/phenotype_ml_dataset.csv"
OUT = REV / "out/38_token_recoverability"

N_SPLITS = 5
FOLD_SEED = 0
ANCHOR_SEED = 42

MIMIC_PRESPEC = ["organism_category", "care_unit", "admit_period", "n_bsi_episodes"]
MIMIC_EXTENDED = [
    "charlson_category", "source_category",
    "BSI_3gcr", "BSI_mrsa", "BSI_vre", "BSI_carbar",
    "polymicrobial", "fungemia", "sex", "age_band", "icu",
    # Rendered by 27c_mimic_narratives.py (SPECTRUM_FIELDS) but missing from
    # the earlier extended set.
    "antiMRSAab_empiric", "antiVREab_empiric", "antipseudoab_empiric", "carbaab_empiric",
]
KOREA_PRESPEC = ["BSI_episode_short_clean", "hospital_code", "admit_year",
                 "treatment_department_eng"]
# charlson_category_withcancermerged is a comma-joined multi-label string with
# 3,675 distinct values -- a quasi-unique identifier, not a category, and the
# same trap the age banding avoids on the MIMIC side. The banded Charlson
# score is used instead, which also matches MIMIC's 4-level charlson_category.
KOREA_EXTENDED = ["n_bsi_episodes", "icu", "charlson_band", "sex"]


def unseen_cell_fraction(key: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    """How much of the test set the lookup cannot even address.

    Reported alongside every lookup, because a lookup whose test rows mostly
    fall outside the training cells is measuring sparsity, not structure.
    """
    unseen = seen_correct = seen_total = 0
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED)
    for train_idx, test_idx in splitter.split(key.reshape(-1, 1), y, groups):
        known = pd.DataFrame({"k": key[train_idx], "y": y[train_idx]})
        majority = known.groupby("k")["y"].agg(lambda s: s.value_counts().idxmax())
        for i in test_idx:
            if key[i] in majority.index:
                seen_total += 1
                seen_correct += int(majority[key[i]] == y[i])
            else:
                unseen += 1
    return {
        "frac_test_rows_in_unseen_cell": float(unseen / len(y)),
        "accuracy_on_seen_cells": float(seen_correct / seen_total) if seen_total else None,
        "n_seen": int(seen_total),
    }


def lookup_macro_f1(key: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    predictions = np.full(len(y), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED)
    for train_idx, test_idx in splitter.split(key.reshape(-1, 1), y, groups):
        majority = (
            pd.DataFrame({"k": key[train_idx], "y": y[train_idx]})
            .groupby("k")["y"].agg(lambda s: s.value_counts().idxmax())
        )
        global_majority = pd.Series(y[train_idx]).value_counts().idxmax()
        predictions[test_idx] = (
            pd.Series(key[test_idx]).map(majority).fillna(global_majority).to_numpy()
        )
    return float(f1_score(y, predictions, average="macro"))


def classifier_scores(frame: pd.DataFrame, columns: list, y: np.ndarray,
                      groups: np.ndarray) -> dict:
    """Gradient-boosted trees over the same token fields.

    Unlike the lookup this generalises across token combinations, so it
    answers "is membership a function of these tokens?" rather than "did
    this exact combination appear in training?".
    """
    data = frame[columns].astype(str)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    predictions = np.full(len(y), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED)
    for train_idx, test_idx in splitter.split(data, y, groups):
        x_train = pd.DataFrame(encoder.fit_transform(data.iloc[train_idx])).astype("category")
        x_test = pd.DataFrame(encoder.transform(data.iloc[test_idx]))
        for column in x_train.columns:
            x_test[column] = pd.Categorical(x_test[column],
                                            categories=x_train[column].cat.categories)
        model = HistGradientBoostingClassifier(
            categorical_features="from_dtype", max_iter=200, learning_rate=0.1,
            max_leaf_nodes=31, min_samples_leaf=20, random_state=0,
        )
        model.fit(x_train, y[train_idx])
        predictions[test_idx] = model.predict(x_test)
    per_cluster = f1_score(y, predictions, average=None, labels=np.unique(y))
    return {
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "accuracy": float((predictions == y).mean()),
        "per_cluster_f1": {int(c): float(v) for c, v in zip(np.unique(y), per_cluster)},
    }


def evaluate(frame: pd.DataFrame, columns: list, y: np.ndarray,
             groups: np.ndarray) -> dict:
    key = frame[columns].astype(str).agg("\x1f".join, axis=1).to_numpy()
    return {
        "n_cells": int(pd.Series(key).nunique()),
        "rows_per_cell": float(len(y) / pd.Series(key).nunique()),
        "lookup_macro_f1": lookup_macro_f1(key, y, groups),
        "lookup_diagnostic": unseen_cell_fraction(key, y, groups),
        "classifier": classifier_scores(frame, columns, y, groups),
    }


def load_mimic() -> pd.DataFrame:
    keys = ["subject_id", "hadm_id"]
    frame = pd.read_parquet(MIMIC_COHORT / "narrative_manifest.parquet")
    enriched = pd.read_parquet(MIMIC_COHORT / "admissions_enriched.parquet")
    frame = frame.merge(
        enriched[keys + ["age_at_admission", "sex", "polymicrobial", "fungemia"]],
        on=keys, how="left", validate="one_to_one")
    frame = frame.merge(
        pd.read_parquet(MIMIC_COHORT / "comorbidities.parquet")[
            keys + ["charlson_category", "source_category"]],
        on=keys, how="left", validate="one_to_one")
    frame = frame.merge(
        pd.read_parquet(MIMIC_COHORT / "ast_detail.parquet")[
            keys + ["BSI_3gcr", "BSI_mrsa", "BSI_vre", "BSI_carbar"]],
        on=keys, how="left", validate="one_to_one")
    frame = frame.merge(
        pd.read_parquet(MIMIC_COHORT / "empiric_antibiotics.parquet")[
            keys + ["antiMRSAab_empiric", "antiVREab_empiric",
                    "antipseudoab_empiric", "carbaab_empiric"]],
        on=keys, how="left", validate="one_to_one")
    frame["age_band"] = pd.cut(
        frame["age_at_admission"], bins=[0, 40, 50, 60, 70, 80, 200],
        labels=["<40", "40s", "50s", "60s", "70s", "80+"], right=False).astype(str)
    return frame


def load_korea() -> pd.DataFrame:
    published = pd.read_csv(PUBLISHED, usecols=["admission_ID", "phenotype_cluster"])
    columns = ["admission_ID", "patient_ID", "BSI_episode_short_clean", "hospital_code",
               "test_date", "treatment_department_eng", "n_bsi_episodes", "icu",
               "charlson_score", "sex"]
    dataset = pd.read_csv(DATASET, low_memory=False, usecols=columns).drop_duplicates("admission_ID")
    frame = published.merge(dataset, on="admission_ID", validate="one_to_one")
    frame["admit_year"] = pd.to_datetime(frame["test_date"], errors="coerce").dt.year.astype(str)
    frame["charlson_band"] = pd.cut(
        frame["charlson_score"], bins=[-0.5, 0.5, 2.5, 4.5, 100],
        labels=["0", "1-2", "3-4", "5+"]).astype(str)
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    result = {
        "design": (
            "Token recoverability with both a majority-per-cell lookup (the "
            "pre-specified S6 instrument) and a gradient-boosted classifier over "
            "the same fields (which generalises across cells). The unseen-cell "
            "fraction says when the lookup is trustworthy."
        ),
        "fold_seed": FOLD_SEED,
        "anchor_seed": ANCHOR_SEED,
        "cohorts": {},
    }

    mimic = load_mimic()
    result["cohorts"]["mimic"] = {}
    for encoder in ("bge-m3", "qwen3-0.6b"):
        labels = pd.read_parquet(MIMIC_LABELS / f"{encoder}_seed{ANCHOR_SEED}.parquet")
        merged = mimic.merge(labels, on="admission_ID", how="inner", validate="one_to_one")
        merged = merged[merged["cluster"] != -1].reset_index(drop=True)
        y = merged["cluster"].to_numpy()
        groups = merged["subject_id"].to_numpy()
        block = {"n": int(len(y)), "k": int(len(set(y.tolist())))}
        for name, columns in {
            "prespec": MIMIC_PRESPEC,
            "prespec_plus_extended": MIMIC_PRESPEC + MIMIC_EXTENDED,
        }.items():
            block[name] = evaluate(merged, columns, y, groups)
            print(f"[mimic/{encoder}/{name}] lookup "
                  f"{block[name]['lookup_macro_f1']:.3f} | classifier "
                  f"{block[name]['classifier']['macro_f1']:.3f} | unseen "
                  f"{block[name]['lookup_diagnostic']['frac_test_rows_in_unseen_cell']:.3f}",
                  flush=True)
        result["cohorts"]["mimic"][encoder] = block

    korea = load_korea()
    korea = korea[korea["phenotype_cluster"] != -1].reset_index(drop=True)
    y = korea["phenotype_cluster"].to_numpy()
    groups = korea["patient_ID"].to_numpy()
    block = {"n": int(len(y)), "k": int(len(set(y.tolist())))}
    for name, columns in {
        "prespec": KOREA_PRESPEC,
        "prespec_plus_extended": KOREA_PRESPEC + KOREA_EXTENDED,
    }.items():
        block[name] = evaluate(korea, columns, y, groups)
        print(f"[korea/{name}] lookup {block[name]['lookup_macro_f1']:.3f} | classifier "
              f"{block[name]['classifier']['macro_f1']:.3f} | unseen "
              f"{block[name]['lookup_diagnostic']['frac_test_rows_in_unseen_cell']:.3f}",
              flush=True)
    result["cohorts"]["korea"] = {"published-3-large": block}

    (OUT / "token_recoverability.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\n[token] wrote {OUT}/token_recoverability.json")


if __name__ == "__main__":
    main()
