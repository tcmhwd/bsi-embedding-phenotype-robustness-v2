#!/usr/bin/env python3
"""Task 22 Task A, S4 (institution transport) / S5 (era transport) applied
to the published E. coli subphenotype object -- fit-on-source +
`umap.UMAP.transform` + `hdbscan.approximate_predict` (never refit the
target), at Table A.2's E. coli configuration, on
`ecoli_case_report_embeddings.npy`, in the frozen original execution
environment. Plus a target-side independent reference refit (for
ARI(transported, refit)), and a 25-random-patient-split null per direction.

MUST be invoked with:
    /home/programs/anaconda3/envs/gpt/bin/python3 22d_ecoli_s4s5_transport.py --run-index I

--run-index in [0, 103]: 0-3 are the 4 observed directions
(hospital_10_to_20, hospital_20_to_10, period_early_to_late,
period_late_to_early); 4-103 are 25 nulls per direction (patient-level
random split matching the real source size).

Reference partition for the per-cluster T metric:
`ecoli_umap_hdbscan_labels.csv[ecoli_cluster]` (not a refit).

Reuses `02_recluster.run_config` and `08_transport.transport_predict`/
`unassigned_fraction`/`best_whole_cohort_cluster` DIRECTLY via importlib
(both are corpus-agnostic pure numpy/pandas/umap/hdbscan code) -- not
reimplemented, per this repo's "reuse, don't reimplement" convention.

Writes only under
`phenotype_revision/out/22_published_objects/task_a_ecoli/s4s5_transport/`:
  - `{tag}.json` -- direction/null summary
  - `{tag}_percluster.csv` -- one row per reference cluster
Resumable: skips a run whose summary JSON already exists.
"""
import os
import sys

for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import importlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c22 = importlib.import_module("22a_ecoli_common")
recl = importlib.import_module("02_recluster")  # run_config(return_models=True)
tr = importlib.import_module("08_transport")    # transport_predict, unassigned_fraction,
                                                  # best_whole_cohort_cluster

OUT_DIR = c22.OUT / "s4s5_transport"
SEED = c22.CLUSTER_SEED


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-index", type=int, required=True)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run = c22.resolve_transport_run(args.run_index)
    tag = run["tag"]
    summary_path = OUT_DIR / f"{tag}.json"
    if summary_path.exists():
        print(f"[transport {tag}] already done, skipping (resume)", flush=True)
        return

    frame = c22.load_reference_frame()
    emb = c22.load_embeddings()
    reference = frame[c22.REFERENCE_CLUSTER_COL].to_numpy(dtype=np.int64)
    patient_ids = frame["patient_ID"].to_numpy()

    direction = run["direction"]
    if not run["is_null"]:
        src_idx = np.where(c22.stratum_mask(frame, direction["axis"], direction["source"]))[0]
        tgt_idx = np.where(c22.stratum_mask(frame, direction["axis"], direction["target"]))[0]
    else:
        n_source_real = int(c22.stratum_mask(frame, direction["axis"], direction["source"]).sum())
        split_seed = run["direction_idx"] * 1000 + run["null_rep"]
        src_idx, tgt_idx = c22.patient_split(patient_ids, n_source_real, split_seed)
        assert len(set(patient_ids[src_idx].tolist()) & set(patient_ids[tgt_idx].tolist())) == 0, (
            "null split leaked a patient across source/target"
        )

    mcs = c22.mcs_for_stratum(len(src_idx))
    # min_samples scaled by the same n_src/n_full ratio as min_cluster_size,
    # with the same floor rule (mirrors `mcs_for_stratum`'s own floor=50).
    ms = max(50, round(c22.MIN_SAMPLES * len(src_idx) / c22.N_ECOLI))
    cfg = dict(c22.UMAP_KW)
    cfg.update(min_cluster_size=mcs, min_samples=ms, cluster_selection_method=c22.HDBSCAN_KW["cluster_selection_method"])
    print(f"[transport {tag}] config actually used: {cfg} (n_source={len(src_idx)}, "
          f"n_target={len(tgt_idx)})", flush=True)

    t0 = time.time()
    fit = recl.run_config(emb[src_idx], seed=SEED, return_models=True, **cfg)
    fit_wall = time.time() - t0
    source_labels = np.asarray(fit["labels"])

    t1 = time.time()
    target_labels, strengths, _ = tr.transport_predict(fit, emb[tgt_idx])
    transport_wall = time.time() - t1
    target_labels = np.asarray(target_labels)

    t2 = time.time()
    ref = recl.run_config(emb[tgt_idx], seed=SEED, **cfg)
    ref_wall = time.time() - t2

    target_unassigned = tr.unassigned_fraction(target_labels)
    both = (target_labels != -1) & (np.asarray(ref["labels"]) != -1)
    ari_all = float(adjusted_rand_score(target_labels, ref["labels"]))
    ari_assigned_both = (
        float(adjusted_rand_score(target_labels[both], np.asarray(ref["labels"])[both]))
        if both.any() else float("nan")
    )

    ref_src = reference[src_idx]
    ref_tgt = reference[tgt_idx]
    rows = []
    for c in sorted(set(reference.tolist()) - {-1}):
        n_src_c = int((ref_src == c).sum())
        n_tgt_c = int((ref_tgt == c).sum())
        matched_k, src_jac = tr.best_whole_cohort_cluster(ref_src, source_labels, c)
        if matched_k is None or (isinstance(src_jac, float) and (np.isnan(src_jac) or src_jac < 0.50)):
            matched_k_report = None
            T = float("nan") if n_src_c == 0 else 0.0
        else:
            matched_k_report = int(matched_k)
            T = float("nan") if n_tgt_c == 0 else float((target_labels[ref_tgt == c] == matched_k).mean())
        rows.append(dict(
            reference_cluster=int(c), n_src_c=n_src_c, n_tgt_c=n_tgt_c,
            matched_source_cluster=matched_k_report,
            source_side_jaccard=float(src_jac) if src_jac is not None else float("nan"),
            T=T,
        ))
    percluster_df = pd.DataFrame(rows)

    summary = dict(
        tag=tag, direction=direction["name"], axis=direction["axis"],
        source_stratum=direction["source"], target_stratum=direction["target"],
        is_null=run["is_null"], null_rep=run["null_rep"], direction_idx=run["direction_idx"],
        seed=SEED, n_source=int(len(src_idx)), n_target=int(len(tgt_idx)),
        min_cluster_size=mcs, min_samples=ms,
        source_n_clusters=int(fit["n_clusters"]), source_noise_frac=float(fit["noise_frac"]),
        target_unassigned_frac=float(target_unassigned),
        reference_target_n_clusters=int(ref["n_clusters"]),
        reference_target_noise_frac=float(ref["noise_frac"]),
        ari_transported_vs_refit_all=ari_all,
        ari_transported_vs_refit_assigned_both=ari_assigned_both,
        frac_assigned_both=float(both.mean()),
        fit_wall_time_sec=round(fit_wall, 2), transport_wall_time_sec=round(transport_wall, 2),
        reference_fit_wall_time_sec=round(ref_wall, 2),
        config_used=cfg,
    )
    c22.atomic_write_json(summary, summary_path)
    c22.atomic_write_csv(percluster_df, OUT_DIR / f"{tag}_percluster.csv")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str), flush=True)
    print(f"[transport {tag}] done -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
