#!/usr/bin/env python3
"""Task 22 Task C, step 1 -- point-estimate fit for ONE (method, k) combo:
n_clusters_actual, ari_vs_embedding, ari_vs_published_paper,
auroc_90d_mortality (StratifiedGroupKFold(5, patient_ID) -- see
`22g_comparator_common.py` "Fix 1"), top-3 characteristic features, labels
saved for the bootstrap step to reuse as `base_labels`.

MUST be invoked with:
    <bsi_pheno python3> 22g1_comparator_point.py --method NAME --k K

Writes only under
`phenotype_revision/out/22_published_objects/task_c_comparators/`:
  - `point_{method}_k{k}.json` -- the comparison-table row for this combo
  - `labels_{method}_k{k}.npz` -- fitted labels (bootstrap step's base_labels)
  - `top_features_{method}_k{k}.csv`
One JSON file per combo (not a shared CSV) so concurrent SLURM tasks for
different combos never race on the same file. `22g3_assemble_table.py`
combines them afterward. Resumable: skips a combo whose JSON already exists.
"""
import os

os.environ["OMP_NUM_THREADS"] = os.environ["NUMBA_NUM_THREADS"] = \
    os.environ["MKL_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c22g = importlib.import_module("22g_comparator_common")
recl = importlib.import_module("02_recluster")  # _thread_config, _library_versions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(c22g.METHOD_FIT_FUNCS))
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seed", type=int, default=c22g.SEED)
    args = ap.parse_args()

    c22g.OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{args.method}_k{args.k}"
    result_path = c22g.OUT / f"point_{tag}.json"
    if result_path.exists():
        print(f"[{tag}] already done, skipping (resume)", flush=True)
        return

    print(f"[{tag}] python executable: {sys.executable}", flush=True)
    print(f"[{tag}] loading frame + structured matrix", flush=True)
    frame, matrix, categorical_cols, imputed, admission_ids, patient_ids = c22g.load_common()

    embedding_labels_df = pd.read_parquet(c22g.EMBEDDING_LABELS_PATH)
    published_labels_df = frame[["admission_ID", "phenotype_cluster"]]

    fit_func = c22g.METHOD_FIT_FUNCS[args.method]
    fit_kwargs = {"seed": args.seed}
    if args.method == "gower_pam":
        c22g.assert_gower_row_order_matches(admission_ids)
        full_gower = c22g.load_gower_matrix()
        fit_kwargs["precomputed_distance"] = np.asarray(full_gower)  # materialize once for the point fit

    numeric_cols_used = [c for c in matrix.columns if c not in categorical_cols]
    print(f"[{tag}] config actually used: fit_func={fit_func.__name__} k={args.k} seed={args.seed} "
          f"n_rows={len(matrix)} precomputed_distance={'yes' if args.method == 'gower_pam' else 'no'}",
          flush=True)
    print(f"[{tag}] numeric_cols actually used ({len(numeric_cols_used)}): {numeric_cols_used}", flush=True)
    print(f"[{tag}] categorical_cols actually used ({len(categorical_cols)}): {categorical_cols}", flush=True)

    t0 = time.time()
    labels = np.asarray(fit_func(imputed, categorical_cols, args.k, **fit_kwargs))
    fit_sec = time.time() - t0
    n_clusters_actual = int(len(set(labels.tolist())))
    print(f"[{tag}] fit done in {fit_sec:.1f}s, n_clusters_actual={n_clusters_actual}", flush=True)

    ari_vs_embedding = c22g.ari_against_reference(admission_ids, labels, embedding_labels_df, "label")
    ari_vs_published = c22g.ari_against_reference(admission_ids, labels, published_labels_df, "phenotype_cluster")

    t1 = time.time()
    auroc_mean, auroc_std = c22g.evaluate_prognostic_separation_grouped(
        labels, admission_ids, patient_ids, dataset_path=c22g.DATASET, seed=args.seed,
    )
    auroc_sec = time.time() - t1
    print(f"[{tag}] AUROC(grouped)={auroc_mean:.4f}+-{auroc_std:.4f} ({auroc_sec:.1f}s)", flush=True)

    top3 = c22g.characteristic_features_per_cluster(imputed, categorical_cols, labels)
    top3.to_csv(c22g.OUT / f"top_features_{tag}.csv", index=False)

    np.savez_compressed(c22g.OUT / f"labels_{tag}.npz", labels=labels.astype(np.int32), admission_ids=admission_ids)

    row = {
        "method": args.method,
        "k": args.k,
        "n_clusters_actual": n_clusters_actual,
        "ari_vs_embedding": ari_vs_embedding,
        "ari_vs_published_paper": ari_vs_published,
        "auroc_90d_mortality_mean": auroc_mean,
        "auroc_90d_mortality_std": auroc_std,
        "auroc_cv_scheme": "StratifiedGroupKFold(5, groups=patient_ID), seed=" + str(args.seed),
        "top_features_path": f"top_features_{tag}.csv",
        "labels_path": f"labels_{tag}.npz",
        "fit_wall_time_sec": round(fit_sec, 3),
        "seed": args.seed,
        "n_rows": int(len(matrix)),
        "numeric_cols": numeric_cols_used,
        "categorical_cols": categorical_cols,
        "dense_variant_fix": {
            "applied": True,
            "sparse_to_dense_map": c22g.comparators.DENSE_VARIANT_FIX_MAP,
        },
        "python_executable": sys.executable,
        "python_prefix": sys.prefix,
    }
    tmp = result_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(row, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, result_path)
    print(json.dumps(row, indent=2, ensure_ascii=False, default=str), flush=True)
    print(f"[{tag}] wrote {result_path}", flush=True)


if __name__ == "__main__":
    main()
