#!/usr/bin/env python3
"""Task 17, S4 (institution transport) / S5 (era transport) -- fit-on-source
+ `umap.UMAP.transform` + `hdbscan.approximate_predict` (never refit the
target -- refitting would give a different UMAP space, making cluster-ID
comparison meaningless), at the published configuration, on the
contemporaneous `out_pre` embeddings, in the frozen original execution
environment. Plus a target-side INDEPENDENT reference refit (for
ARI(transported, refit)), and a 25-random-patient-split null per direction.

MUST be invoked with:
    /home/programs/anaconda3/envs/gpt/bin/python3 17c_s4s5_transport.py --run-index I [--out-dir DIR]

--run-index in [0, 103]: 0-3 are the 4 observed directions
(hospital_10_to_20, hospital_20_to_10, period_early_to_late,
period_late_to_early); 4-103 are 25 nulls per direction (patient-level
random split matching the real source size, `17_common.patient_split`,
identical method to `14_a0_remeasure.patient_split`).

--out-dir optionally overrides the output directory (default: the original
`out/17_correct_env/s4s5_transport/`). Used for the AUDIT_3 N7 rerun so
the pre-fix results stay on disk for a before/after comparison.

Reference partition for the per-cluster T metric: the PUBLISHED labels
(`out_pre/phenotype_clusters_labels.csv`), not a refit.

N7 fix (2026-09-06): the independent target-side reference refit (used only
for `ari_transported_vs_refit_*`, never for the per-cluster T metric) is now
scaled by `a0_min_cluster_size(len(target))` -- the TARGET stratum's own
size -- instead of reusing the source-derived `min_cluster_size`. Previously
both the source fit and the target reference refit used the mcs derived from
the source stratum, so for asymmetric strata (e.g. hospital 10 vs 20) the two
directions' reference partitions were built at very different granularities
(k=4 vs k=35) and `ari_transported_vs_refit_*` was not comparable across
directions. `min_cluster_size`/`config_used` in the summary JSON now refer to
the source fit only; `reference_min_cluster_size`/`reference_config_used` are
the (new) target-scaled values used for the reference refit.

Reuses `02_recluster.run_config` (UMAP+HDBSCAN fit, `return_models=True`
for the source so its trained reducer/clusterer can transform/predict the
target) and `08_transport._stratum_mask`/`transport_predict`/
`unassigned_fraction`/`best_whole_cohort_cluster` DIRECTLY via importlib
(both modules are corpus-agnostic pure numpy/pandas/umap/hdbscan code, so
they are safe to import under the frozen interpreter; verified importable
in this env before use) -- not reimplemented, per this repo's own
"reuse, don't reimplement" convention. `frame_mod.load_frame`/
`load_embeddings` are NOT used (wrong corpus) -- embeddings and the
hospital/era/patient metadata frame come from `17_common` instead.

Writes only under `phenotype_revision/out/17_correct_env/s4s5_transport/`:
  - `{tag}.json` -- direction/null summary (sizes, mcs used, ARI vs refit,
    unassigned fraction, wall times, config actually used)
  - `{tag}_percluster.csv` -- one row per published cluster: matched source
    cluster, source-side Jaccard, T (transport recovery fraction)
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
c17 = importlib.import_module("17_common")
recl = importlib.import_module("02_recluster")  # run_config(return_models=True)
tr = importlib.import_module("08_transport")    # _stratum_mask, transport_predict,
                                                  # unassigned_fraction, best_whole_cohort_cluster

OUT_DIR = c17.OUT / "s4s5_transport"
SEED = c17.CLUSTER_SEED


def build_frame_for_transport() -> pd.DataFrame:
    """`08_transport._stratum_mask` reads `frame["hospital_code"]` and
    `frame["admit_year"]` -- build a frame with exactly those columns, in
    `out_pre` row order, plus the published label and patient_ID."""
    frame = c17.build_full_frame()
    assert frame["hospital_code"].isin([10, 20]).all()
    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-index", type=int, required=True)
    ap.add_argument(
        "--out-dir", type=str, default=None,
        help="Override the output directory (default: the original "
             "out/17_correct_env/s4s5_transport/). Used for the N7 rerun so the "
             "original (pre-fix) results are preserved for the before/after report.",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    run = c17.resolve_transport_run(args.run_index)
    tag = run["tag"]
    summary_path = out_dir / f"{tag}.json"
    if summary_path.exists():
        print(f"[transport {tag}] already done, skipping (resume)", flush=True)
        return

    frame = build_frame_for_transport()
    emb = c17.load_embeddings()
    published = frame[c17.PUBLISHED_CLUSTER_COL].to_numpy(dtype=np.int64)
    patient_ids = frame["patient_ID"].to_numpy()

    direction = run["direction"]
    if not run["is_null"]:
        src_idx = np.where(tr._stratum_mask(frame, direction["axis"], direction["source"]).to_numpy())[0]
        tgt_idx = np.where(tr._stratum_mask(frame, direction["axis"], direction["target"]).to_numpy())[0]
    else:
        n_source_real = int(tr._stratum_mask(frame, direction["axis"], direction["source"]).sum())
        split_seed = run["direction_idx"] * 1000 + run["null_rep"]
        src_idx, tgt_idx = c17.patient_split(patient_ids, n_source_real, split_seed)
        assert len(set(patient_ids[src_idx].tolist()) & set(patient_ids[tgt_idx].tolist())) == 0, (
            "null split leaked a patient across source/target"
        )

    mcs = c17.a0_min_cluster_size(len(src_idx))
    cfg = dict(c17.UMAP_KW)
    cfg.update(min_cluster_size=mcs, **c17.HDBSCAN_KW)
    # N7 fix: the independent target-side reference refit must be scaled by the
    # TARGET stratum's own size, not the source's -- refitting the target with
    # a source-derived mcs makes the reference partition (and therefore
    # ari_transported_vs_refit_*) incomparable across directions of differing
    # stratum size (e.g. hospital_10_to_20 mcs=123 on a 7,669-row target vs
    # hospital_20_to_10 mcs=50 on a 24,660-row target). See AUDIT_3 N7.
    mcs_ref = c17.a0_min_cluster_size(len(tgt_idx))
    ref_cfg = dict(c17.UMAP_KW)
    ref_cfg.update(min_cluster_size=mcs_ref, **c17.HDBSCAN_KW)
    print(f"[transport {tag}] config actually used: source={cfg} (n_source={len(src_idx)}); "
          f"target reference={ref_cfg} (n_target={len(tgt_idx)})", flush=True)

    t0 = time.time()
    fit = recl.run_config(emb[src_idx], seed=SEED, return_models=True, **cfg)
    fit_wall = time.time() - t0
    source_labels = np.asarray(fit["labels"])

    t1 = time.time()
    target_labels, strengths, _ = tr.transport_predict(fit, emb[tgt_idx])
    transport_wall = time.time() - t1
    target_labels = np.asarray(target_labels)

    t2 = time.time()
    ref = recl.run_config(emb[tgt_idx], seed=SEED, **ref_cfg)
    ref_wall = time.time() - t2

    target_unassigned = tr.unassigned_fraction(target_labels)
    both = (target_labels != -1) & (np.asarray(ref["labels"]) != -1)
    ari_all = float(adjusted_rand_score(target_labels, ref["labels"]))
    ari_assigned_both = (
        float(adjusted_rand_score(target_labels[both], np.asarray(ref["labels"])[both]))
        if both.any() else float("nan")
    )

    pub_src = published[src_idx]
    pub_tgt = published[tgt_idx]
    rows = []
    for c in sorted(set(published.tolist()) - {-1}):
        n_src_c = int((pub_src == c).sum())
        n_tgt_c = int((pub_tgt == c).sum())
        matched_k, src_jac = tr.best_whole_cohort_cluster(pub_src, source_labels, c)
        if matched_k is None or (isinstance(src_jac, float) and (np.isnan(src_jac) or src_jac < 0.50)):
            matched_k_report = None
            T = float("nan") if n_src_c == 0 else 0.0
        else:
            matched_k_report = int(matched_k)
            T = float("nan") if n_tgt_c == 0 else float((target_labels[pub_tgt == c] == matched_k).mean())
        rows.append(dict(
            published_cluster=int(c), n_src_c=n_src_c, n_tgt_c=n_tgt_c,
            matched_source_cluster=matched_k_report,
            source_side_jaccard=float(src_jac) if src_jac is not None else float("nan"),
            T=T,
        ))
    percluster_df = pd.DataFrame(rows)

    summary = dict(
        tag=tag, direction=direction["name"], axis=direction["axis"],
        source_stratum=direction["source"], target_stratum=direction["target"],
        is_null=run["is_null"], null_rep=run["null_rep"], direction_idx=run["direction_idx"],
        seed=SEED, n_source=int(len(src_idx)), n_target=int(len(tgt_idx)), min_cluster_size=mcs,
        reference_min_cluster_size=mcs_ref,
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
        reference_config_used=ref_cfg,
    )
    c17.atomic_write_json(summary, summary_path)
    c17.atomic_write_csv(percluster_df, out_dir / f"{tag}_percluster.csv")
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str), flush=True)
    print(f"[transport {tag}] done -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
