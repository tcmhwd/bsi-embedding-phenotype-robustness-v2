#!/usr/bin/env python3
"""Task 22 Task C addendum -- add the missing comparison baseline: the
PUBLISHED embedding-derived phenotypes themselves, scored under the EXACT
SAME protocol as the five structured comparators, so the comparator table's
AUROC column has something to be compared against.

Two rows, both reusing existing, unmodified functions -- reused, not
reimplemented (reported explicitly below):
  - `published_16`: `phenotype_cluster` column, joined via `admission_ID`
    onto the SAME frame (`01_build_analysis_frame.load_frame`, N=32,329)
    that `22g_comparator_common.load_common()` already uses for the other
    ten rows -- same population, same admission_ID/patient_ID arrays.
  - `published_ecoli_10`: `ecoli_cluster` column from
    `pipeline_new/out_pre/ecoli_umap_hdbscan_labels.csv`
    (`22a_ecoli_common.load_reference_frame()`, N=9,595 -- a DIFFERENT,
    smaller population than the other nine rows; flagged explicitly in the
    output, not silently mixed into the same comparison).

AUROC: `22g_comparator_common.evaluate_prognostic_separation_grouped`
(StratifiedGroupKFold(5, groups=patient_ID), cluster-label one-hot as the
sole feature, logistic regression, 90d_mortality outcome) -- the identical
function object already used for all ten comparator rows, called here with
no changes, just different (labels, admission_ids, patient_ids) inputs.

ARI: `22g_comparator_common.ari_against_reference` -- also reused unchanged.
For `published_16` vs the published labels themselves this is a pipeline
sanity check (must equal 1.0 exactly). For `published_ecoli_10`, the
published-16/embedding reference frames are first restricted to the
intersecting admission_IDs (the E. coli subset) before computing ARI, since
the two label vectors live on different row sets -- this is the ONLY
population-specific accommodation made anywhere in this script; the AUROC
call itself needs no such accommodation (it joins by admission_ID and reads
whatever population is handed to it).

Runs on the login node (label vectors + existing functions only, no new
fitting) -- a few seconds.

Writes only under
`phenotype_revision/out/22_published_objects/task_c_comparators/`:
  - `point_published_16.json`, `point_published_ecoli_10.json` (same shape
    as the other `point_*.json` files, `22g3_assemble_table.py` picks them
    up automatically once added to its METHODS list)
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c22g = importlib.import_module("22g_comparator_common")
c22a = importlib.import_module("22a_ecoli_common")

OUT = c22g.OUT
PUBLISHED_LABELS_PATH = Path("/data/tcmhwd/AI/AST_ai/pipeline_new/out_pre/phenotype_clusters_labels.csv")


def main() -> None:
    embedding_labels_df = pd.read_parquet(c22g.EMBEDDING_LABELS_PATH)

    # ---------------------------------------------------------------
    # Row 1: published_16, on the SAME N=32,329 population/frame as the
    # other ten comparator rows.
    # ---------------------------------------------------------------
    frame, matrix, categorical_cols, imputed, admission_ids, patient_ids = c22g.load_common()
    labels_16 = frame["phenotype_cluster"].to_numpy()
    published_labels_df = frame[["admission_ID", "phenotype_cluster"]]

    n_clusters_16_total = int(len(set(labels_16.tolist())))
    n_clusters_16_nonoise = int(len(set(labels_16.tolist()) - {-1}))

    ari_vs_embedding_16 = c22g.ari_against_reference(admission_ids, labels_16, embedding_labels_df, "label")
    ari_vs_published_16 = c22g.ari_against_reference(admission_ids, labels_16, published_labels_df, "phenotype_cluster")
    assert abs(ari_vs_published_16 - 1.0) < 1e-9, (
        f"published_16 vs itself should be ARI=1.0 exactly, got {ari_vs_published_16} -- pipeline join bug"
    )

    auroc_mean_16, auroc_std_16 = c22g.evaluate_prognostic_separation_grouped(
        labels_16, admission_ids, patient_ids, dataset_path=c22g.DATASET, seed=c22g.SEED,
    )

    row_16 = {
        "method": "published_16",
        "k": n_clusters_16_nonoise,
        "n_clusters_actual": n_clusters_16_total,
        "n_clusters_actual_noise_excluded": n_clusters_16_nonoise,
        "noise_label_treatment": "-1 (noise) kept as its own predictor category in the one-hot "
                                  "encoding, exactly like every other method's labels are encoded "
                                  "(pd.get_dummies of the raw label column, no special-casing) -- "
                                  "same encoding as the five structured comparators.",
        "ari_vs_embedding": ari_vs_embedding_16,
        "ari_vs_published_paper": ari_vs_published_16,
        "ari_vs_published_paper_note": "self-comparison sanity check, asserted == 1.0",
        "auroc_90d_mortality_mean": auroc_mean_16,
        "auroc_90d_mortality_std": auroc_std_16,
        "auroc_cv_scheme": "StratifiedGroupKFold(5, groups=patient_ID), seed=" + str(c22g.SEED)
                            + " -- IDENTICAL function call (evaluate_prognostic_separation_grouped) "
                              "used for all ten structured-comparator rows",
        "n_rows": int(len(frame)),
        "population_note": "N=32,329, SAME frame/admission_ID/patient_ID population as all five "
                            "structured comparators (kprototypes/famd_kmeans/gmm_structured/gower_pam/lca) "
                            "-- directly comparable AUROC.",
        "reused_functions": [
            "22g_comparator_common.load_common (-> 01_build_analysis_frame.load_frame)",
            "22g_comparator_common.ari_against_reference (-> 05c_comparison_table.ari_against_reference)",
            "22g_comparator_common.evaluate_prognostic_separation_grouped",
        ],
    }
    (OUT / "point_published_16.json").write_text(json.dumps(row_16, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(row_16, indent=2, ensure_ascii=False, default=str), flush=True)

    # ---------------------------------------------------------------
    # Row 2: published_ecoli_10, on the E. coli subset's OWN population
    # (N=9,595) -- explicitly NOT the same population as the other rows.
    # ---------------------------------------------------------------
    eco_frame = c22a.load_reference_frame()
    labels_eco = eco_frame[c22a.REFERENCE_CLUSTER_COL].to_numpy()
    eco_admission_ids = eco_frame["admission_ID"].to_numpy()
    eco_patient_ids = eco_frame["patient_ID"].to_numpy()

    n_clusters_eco_total = int(len(set(labels_eco.tolist())))
    n_clusters_eco_nonoise = int(len(set(labels_eco.tolist()) - {-1}))

    # ARI columns need the reference frames restricted to the E. coli
    # admission_ID subset first -- the labels live on different row sets,
    # this is the one population-specific accommodation in this script.
    eco_id_set = set(eco_admission_ids.tolist())
    published_labels_eco_subset = published_labels_df[published_labels_df["admission_ID"].isin(eco_id_set)]
    embedding_labels_eco_subset = embedding_labels_df[embedding_labels_df["admission_ID"].isin(eco_id_set)]
    ari_vs_published_eco = c22g.ari_against_reference(
        eco_admission_ids, labels_eco, published_labels_eco_subset, "phenotype_cluster",
    )
    ari_vs_embedding_eco = c22g.ari_against_reference(
        eco_admission_ids, labels_eco, embedding_labels_eco_subset, "label",
    )

    auroc_mean_eco, auroc_std_eco = c22g.evaluate_prognostic_separation_grouped(
        labels_eco, eco_admission_ids, eco_patient_ids, dataset_path=c22g.DATASET, seed=c22g.SEED,
    )

    row_eco = {
        "method": "published_ecoli_10",
        "k": n_clusters_eco_nonoise,
        "n_clusters_actual": n_clusters_eco_total,
        "n_clusters_actual_noise_excluded": n_clusters_eco_nonoise,
        "noise_label_treatment": "-1 (noise) kept as its own predictor category, same convention as published_16.",
        "ari_vs_embedding": ari_vs_embedding_eco,
        "ari_vs_embedding_note": "computed WITHIN the n=9,595 E. coli subset only (embedding reference "
                                  "restricted to the intersecting admission_IDs first) -- not comparable "
                                  "in magnitude to the other rows' ari_vs_embedding, which is computed on "
                                  "the full N=32,329 cohort.",
        "ari_vs_published_paper": ari_vs_published_eco,
        "ari_vs_published_paper_note": "published_16 restricted to the E. coli admission_ID subset first, "
                                        "then compared to ecoli_cluster -- answers 'how much does the "
                                        "E. coli subphenotyping resemble the parent published clusters "
                                        "restricted to E. coli patients', NOT a self-comparison.",
        "auroc_90d_mortality_mean": auroc_mean_eco,
        "auroc_90d_mortality_std": auroc_std_eco,
        "auroc_cv_scheme": "StratifiedGroupKFold(5, groups=patient_ID), seed=" + str(c22g.SEED)
                            + " -- IDENTICAL function call (evaluate_prognostic_separation_grouped), "
                              "same encoding, applied to the E. coli subset's own labels/population",
        "n_rows": int(len(eco_frame)),
        "population_note": "N=9,595 (E. coli subphenotype object, Task A) -- a DIFFERENT, SMALLER "
                            "population than the other nine comparator rows (all N=32,329). This row's "
                            "AUROC answers a different question (within E. coli patients only, do the "
                            "10 E. coli subphenotypes predict 90-day mortality) and is NOT directly "
                            "comparable in magnitude to the full-cohort AUROC numbers in the other rows "
                            "-- reported for completeness, not as an apples-to-apples baseline for "
                            "kprototypes/famd_kmeans/gmm_structured/gower_pam/lca/published_16.",
        "reused_functions": [
            "22a_ecoli_common.load_reference_frame",
            "22g_comparator_common.ari_against_reference (-> 05c_comparison_table.ari_against_reference)",
            "22g_comparator_common.evaluate_prognostic_separation_grouped",
        ],
    }
    (OUT / "point_published_ecoli_10.json").write_text(json.dumps(row_eco, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(row_eco, indent=2, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
