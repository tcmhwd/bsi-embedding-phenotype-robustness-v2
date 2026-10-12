#!/usr/bin/env python3
"""Audit 6 -- independent checks on scripts/38_token_recoverability.py (R1).

Writes out/39_audit6/token_checks.json.  Does not modify 38's outputs.

Checks
  A. Mechanics of classifier_scores(): fold-level assertions that no patient
     is in both train and test, that no admission is duplicated, and that the
     number of NaN cells in the realigned test matrix equals the number of
     OrdinalEncoder unknown (-1) codes -- i.e. unseen levels become missing,
     never a valid train code.
  B. Re-derivation of 38's six headline rows with chance-corrected metrics
     that are comparable across k (Cohen's kappa, AMI(pred, true), balanced
     accuracy) beside macro-F1, plus a patient-block permutation null for
     macro-F1 so each number sits next to its own chance level.
  C. Korea ablations: is the Charlson band doing the work?  prespec + each
     extended field alone; extended minus the band; the raw charlson_score
     (18 levels) in place of the band; the original 3,674-level string via
     the lookup only (HGB cannot take it) and via a top-254 + 'other' recode.
  D. Korea with a MIMIC-equivalent token set (age band, resistance flags,
     polymicrobial, empiric-spectrum flags, entry mode, ventilator, Charlson
     band, sex, ICU, episode count) -- all rendered in the Korean narrative --
     so the two cohorts are scored on comparable vocabularies.
"""
import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "4")

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (adjusted_mutual_info_score, balanced_accuracy_score,
                             cohen_kappa_score, f1_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OrdinalEncoder

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
OUT = REV / "out/39_audit6"
OUT.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("t38", REV / "scripts/38_token_recoverability.py")
t38 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t38)

N_SPLITS, FOLD_SEED = t38.N_SPLITS, t38.FOLD_SEED


def hgb():
    return HistGradientBoostingClassifier(
        categorical_features="from_dtype", max_iter=200, learning_rate=0.1,
        max_leaf_nodes=31, min_samples_leaf=20, random_state=0)


def classifier_full(frame, columns, y, groups, label):
    """38's classifier, re-implemented line-for-line, plus the mechanics assertions."""
    data = frame[columns].astype(str)
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    predictions = np.full(len(y), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=FOLD_SEED)
    mech = {"group_overlap": 0, "unknown_codes": 0, "nan_after_realign": 0, "max_cardinality": 0}
    for train_idx, test_idx in splitter.split(data, y, groups):
        mech["group_overlap"] += len(set(groups[train_idx]) & set(groups[test_idx]))
        x_train = pd.DataFrame(encoder.fit_transform(data.iloc[train_idx])).astype("category")
        raw_test = encoder.transform(data.iloc[test_idx])
        mech["unknown_codes"] += int((raw_test == -1).sum())
        x_test = pd.DataFrame(raw_test)
        for column in x_train.columns:
            x_test[column] = pd.Categorical(x_test[column], categories=x_train[column].cat.categories)
            mech["max_cardinality"] = max(mech["max_cardinality"], len(x_train[column].cat.categories))
        mech["nan_after_realign"] += int(x_test.isna().sum().sum())
        model = hgb()
        model.fit(x_train, y[train_idx])
        predictions[test_idx] = model.predict(x_test)
    assert (predictions >= 0).all()
    out = {
        "label": label, "n": int(len(y)), "k": int(len(np.unique(y))), "columns": columns,
        "macro_f1": float(f1_score(y, predictions, average="macro")),
        "weighted_f1": float(f1_score(y, predictions, average="weighted")),
        "accuracy": float((predictions == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "kappa": float(cohen_kappa_score(y, predictions)),
        "ami_pred_vs_true": float(adjusted_mutual_info_score(y, predictions)),
        "majority_class_share": float(pd.Series(y).value_counts(normalize=True).iloc[0]),
        "mechanics": mech,
    }
    print(f"[clf] {label}: macroF1 {out['macro_f1']:.3f} kappa {out['kappa']:.3f} "
          f"AMI {out['ami_pred_vs_true']:.3f} balacc {out['balanced_accuracy']:.3f} "
          f"mech {mech}", flush=True)
    return out


def permutation_null(frame, columns, y, groups, n_perm=3, seed=0):
    """Macro-F1 when cluster labels are permuted across patients (labels kept
    constant within a patient so the grouping structure is preserved)."""
    rng = np.random.default_rng(seed)
    vals = []
    patients = pd.Series(groups)
    first = patients.drop_duplicates().index.to_numpy()
    for _ in range(n_perm):
        # permute the per-patient first label, then broadcast to admissions
        perm = rng.permutation(y[first])
        mapping = dict(zip(patients.iloc[first], perm))
        y_perm = patients.map(mapping).to_numpy()
        vals.append(classifier_full(frame, columns, y_perm, groups, "perm")["macro_f1"])
    return {"n_perm": n_perm, "macro_f1_values": vals, "macro_f1_mean": float(np.mean(vals))}


def main():
    result = {"A_B_headline": {}, "C_korea_ablations": {}, "D_korea_mimic_equivalent": {}}

    # ---------------- MIMIC
    mimic = t38.load_mimic()
    assert not mimic["admission_ID"].duplicated().any()
    for enc in ("bge-m3", "qwen3-0.6b"):
        labels = pd.read_parquet(t38.MIMIC_LABELS / f"{enc}_seed42.parquet")
        merged = mimic.merge(labels, on="admission_ID", how="inner", validate="one_to_one")
        merged = merged[merged["cluster"] != -1].reset_index(drop=True)
        y = merged["cluster"].to_numpy(); g = merged["subject_id"].to_numpy()
        block = {}
        for name, cols in {"prespec": t38.MIMIC_PRESPEC,
                           "prespec_plus_extended": t38.MIMIC_PRESPEC + t38.MIMIC_EXTENDED}.items():
            block[name] = classifier_full(merged, cols, y, g, f"mimic/{enc}/{name}")
        block["prespec_plus_extended"]["permutation_null"] = permutation_null(
            merged, t38.MIMIC_PRESPEC + t38.MIMIC_EXTENDED, y, g)
        # organism-only, to see how much of the token function is the organism line
        block["organism_only"] = classifier_full(merged, ["organism_category"], y, g, f"mimic/{enc}/organism_only")
        block["extended_minus_organism"] = classifier_full(
            merged, [c for c in t38.MIMIC_PRESPEC + t38.MIMIC_EXTENDED if c != "organism_category"],
            y, g, f"mimic/{enc}/extended_minus_organism")
        result["A_B_headline"][f"mimic/{enc}"] = block

    # ---------------- Korea
    korea = t38.load_korea()
    korea = korea[korea["phenotype_cluster"] != -1].reset_index(drop=True)
    extra_cols = ["admission_ID", "age_at_test", "BSI_episode_3GCR", "BSI_episode_CarbaR",
                  "BSI_episode_VRE", "BSI_episode_MRSA", "BSI_episode_CRAB",
                  "BSI_episode_CeftriaxoneR", "BSI_episode_polymicrobial", "antiMRSAab_empiric",
                  "antiVREab_empiric", "antipseudoab_empiric", "carbaab_empiric",
                  "outpatient_inpatient_ER", "ventilator", "charlson_category_withcancermerged",
                  "empiric_cluster", "admission_cluster", "sepsis_2", "VRE_colonization",
                  "CRE_colonization"]
    extra = pd.read_csv(t38.DATASET, low_memory=False, usecols=extra_cols).drop_duplicates("admission_ID")
    korea = korea.merge(extra, on="admission_ID", validate="one_to_one")
    korea["age_band"] = pd.cut(korea["age_at_test"], bins=[0, 40, 50, 60, 70, 80, 200],
                               labels=["<40", "40s", "50s", "60s", "70s", "80+"], right=False).astype(str)
    top = korea["charlson_category_withcancermerged"].value_counts().index[:254]
    korea["charlson_str_top254"] = np.where(
        korea["charlson_category_withcancermerged"].isin(top),
        korea["charlson_category_withcancermerged"], "other_rare")
    y = korea["phenotype_cluster"].to_numpy(); g = korea["patient_ID"].to_numpy()
    assert not korea["admission_ID"].duplicated().any()
    P, E = t38.KOREA_PRESPEC, t38.KOREA_EXTENDED

    hb = {}
    hb["prespec"] = classifier_full(korea, P, y, g, "korea/prespec")
    hb["prespec_plus_extended"] = classifier_full(korea, P + E, y, g, "korea/prespec_plus_extended")
    hb["prespec_plus_extended"]["permutation_null"] = permutation_null(korea, P + E, y, g)
    hb["organism_only"] = classifier_full(korea, ["BSI_episode_short_clean"], y, g, "korea/organism_only")
    hb["extended_minus_organism"] = classifier_full(
        korea, [c for c in P + E if c != "BSI_episode_short_clean"], y, g, "korea/extended_minus_organism")
    result["A_B_headline"]["korea/published-3-large"] = hb

    ab = {}
    for field in E:
        ab[f"prespec+{field}"] = classifier_full(korea, P + [field], y, g, f"korea/prespec+{field}")
    ab["extended_minus_charlson_band"] = classifier_full(
        korea, P + [c for c in E if c != "charlson_band"], y, g, "korea/extended_minus_charlson_band")
    ab["extended_with_raw_charlson_score"] = classifier_full(
        korea, P + [c if c != "charlson_band" else "charlson_score" for c in E], y, g,
        "korea/extended_with_raw_charlson_score")
    ab["extended_with_charlson_string_top254"] = classifier_full(
        korea, P + [c if c != "charlson_band" else "charlson_str_top254" for c in E], y, g,
        "korea/extended_with_charlson_string_top254")
    cols_str = P + [c if c != "charlson_band" else "charlson_category_withcancermerged" for c in E]
    key = korea[cols_str].astype(str).agg("\x1f".join, axis=1).to_numpy()
    ab["extended_with_charlson_string_LOOKUP_only"] = {
        "lookup_macro_f1": t38.lookup_macro_f1(key, y, g),
        "lookup_diagnostic": t38.unseen_cell_fraction(key, y, g),
        "n_cells": int(pd.Series(key).nunique()),
    }
    result["C_korea_ablations"] = ab

    MIMIC_EQUIV = P + ["n_bsi_episodes", "icu", "charlson_band", "sex", "age_band",
                       "BSI_episode_3GCR", "BSI_episode_CarbaR", "BSI_episode_VRE", "BSI_episode_MRSA",
                       "BSI_episode_CRAB", "BSI_episode_CeftriaxoneR", "BSI_episode_polymicrobial",
                       "antiMRSAab_empiric", "antiVREab_empiric", "antipseudoab_empiric",
                       "carbaab_empiric", "outpatient_inpatient_ER", "ventilator"]
    FULL_RENDERED = MIMIC_EQUIV + ["empiric_cluster", "admission_cluster", "sepsis_2",
                                   "VRE_colonization", "CRE_colonization"]
    d = {}
    d["mimic_equivalent"] = classifier_full(korea, MIMIC_EQUIV, y, g, "korea/mimic_equivalent")
    d["mimic_equivalent"]["lookup"] = {
        "lookup_macro_f1": t38.lookup_macro_f1(
            korea[MIMIC_EQUIV].astype(str).agg("\x1f".join, axis=1).to_numpy(), y, g),
        "lookup_diagnostic": t38.unseen_cell_fraction(
            korea[MIMIC_EQUIV].astype(str).agg("\x1f".join, axis=1).to_numpy(), y, g)}
    d["full_rendered_summary_tokens"] = classifier_full(korea, FULL_RENDERED, y, g, "korea/full_rendered")
    result["D_korea_mimic_equivalent"] = d

    (OUT / "token_checks.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"[audit6] wrote {OUT}/token_checks.json")


if __name__ == "__main__":
    main()
