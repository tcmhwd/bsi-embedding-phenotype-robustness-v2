#!/usr/bin/env python3
"""Task 68 -- Analysis C (FOLLOWUP-2026-09-21-C): cluster the five MIMIC arms and score them.

Stages
  cluster    --arm A --encoder E   11 UMAP+HDBSCAN fits (seeds 42, 1..10) of one arm x encoder.
                                    Labels -> labels/<arm>/<enc>_seed<s>.parquet; per-fit run JSON
                                    -> runs/<arm>/<enc>_seed<s>.json.  Resumable per fit.
  aggregate                         reads every fit + the stored MIMIC partitions and writes
                                    mimic_ablation.csv (+ cross-encoder / conditional / summary /
                                    cross-seed CSVs), reproduction_check.json, summary_raw.md.

Clustering is `02_recluster.run_config` called EXACTLY as 27e_mimic_replicate.py:102-105 calls it
(UMAP n_components=2, n_neighbors=15, min_dist=0.1, metric=euclidean, random_state=seed, n_jobs=1;
HDBSCAN min_cluster_size=max(10, n//200)=80, min_samples=None, cluster_selection_method=eom,
core_dist_n_jobs=1).  Thread pins (OMP/NUMBA/MKL/OPENBLAS = 1) are set BEFORE any numeric import
(27e:329-332; 02_recluster.py:49-52).  Env: /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
(umap-learn 0.5.7 / numpy 1.26.4, the env that produced the stored MIMIC arm; NOT the frozen env).

Scoring (PREREGISTRATION_C.md 3.3, 3.5, 4.0-4.3)
  * ARI convention: `ari_noise_excluded` = both-non-noise rows (the prereg's primary convention),
    with `ari_all_rows` (noise = one label) beside it, exactly the 27e:140-142 naming; AMI alongside.
  * within encoder: each arm's seed-s fit vs (a) the SAME encoder's stored seed-s MIMIC fit
    (out/27_mimic/replicate/labels) and (b) the new `unmodified` arm's seed-s fit.
  * k, noise fraction, largest-cluster share, degeneracy (k <= 4 flag; pair rule k >= 5).
  * field attribution: 40_field_dominance.field_ami over the 19 MIMIC fields; top field, margin.
  * across encoders per arm and seed: BGE-M3 vs Qwen3 ARI/AMI; conditional agreement within
    organism_category and organism_category x source_category strata via
    audit7_dominance_checks.conditional_agreement(min_n=50) (its hard-coded Korean dict is NOT used).
  * Addendum 1 (PREREGISTRATION_C_ADDENDUM_1.md): the seed-42 partition-identity criterion of prereg G0.1 is
    REPLACED.  Nothing here halts on partition identity.  The gross-error detector (min cosine >= 0.99 of the
    re-embedded unmodified corpus vs the stored vectors) is computed in 67b_C_embed.py and REQUIRED here
    (cluster: own encoder must have passed, the other must not have failed; aggregate: both).  The `unmodified`
    fits' k, noise, largest-cluster share and ARI/AMI vs the stored partitions (seed 42 and seeds 1-10) are
    written descriptively as the numerical noise floor (reproduction_check.json, summary_raw.md); every arm is
    scored both vs the stored seed-matched fit and vs the new unmodified arm.
  * `stored_reference` rows (the stored partitions themselves) are computed by the same code so the
    anchors in the preregistration (0.5714 conditional AMI, field AMIs) can be checked against it.

NO interpretation is produced here: summary_raw.md contains numbers only.
"""
from __future__ import annotations

import os

for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import importlib
import itertools
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
v67 = importlib.import_module("67_C_mimic_variants")

ENCODERS = v67.ENCODERS
SEEDS = v67.SEEDS
ARM_ORDER = v67.ARM_ORDER
REF = "stored_reference"
MIN_CLUSTER_SIZE_RULE = "max(10, n // 200)"

UMAP_KWARGS = {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1, "metric": "euclidean",
               "random_state": "<seed>", "n_jobs": 1}
HDBSCAN_KWARGS = {"min_cluster_size": "<mcs>", "min_samples": None, "metric": "euclidean",
                  "gen_min_span_tree": True, "cluster_selection_method": "eom", "core_dist_n_jobs": 1,
                  "prediction_data": False,
                  "cluster_selection_epsilon": "library default (0.0); never set anywhere in this repo"}


def min_cluster_size_for(n: int) -> int:            # 27e_mimic_replicate.py:87-88
    return max(10, n // 200)


def label_path(root: Path, arm: str, enc: str, seed: int) -> Path:
    return root / "labels" / arm / f"{enc}_seed{seed}.parquet"


def run_path(root: Path, arm: str, enc: str, seed: int) -> Path:
    return root / "runs" / arm / f"{enc}_seed{seed}.json"


# ============================================================================ cluster stage
def stage_cluster(root: Path, embed_root: Path, arm: str, enc: str, seeds: list) -> None:
    other = [e for e in ENCODERS if e != enc]
    v67.require_gate(embed_root, [enc], other)                # Addendum 1 item 2a
    recl = importlib.import_module("02_recluster")
    edir = embed_root / "embed" / arm / enc
    manifest = json.loads((edir / "embed_manifest.json").read_text())
    vecs = np.load(edir / "merged.npy")
    ids = json.loads((edir / "merged_ids.json").read_text())
    man = v67.load_manifest()
    assert ids == man["admission_ID"].tolist(), "embedding row order != manifest sorted by admission_ID"
    n = len(vecs)
    assert n == v67.N_EXPECTED and vecs.shape[1] == manifest["dim"]
    n_subj = int(man["subject_id"].nunique())
    assert n_subj == v67.N_SUBJECTS_EXPECTED
    vec_sha = v67.sha256_file(edir / "merged.npy")
    assert vec_sha == manifest["vectors_sha256"], "merged.npy differs from its embed manifest"
    mcs = min_cluster_size_for(n)
    print(f"[68/cluster] {arm}/{enc} n={n} dim={vecs.shape[1]} mcs={mcs} seeds={seeds}", flush=True)

    for seed in seeds:
        rp, lp = run_path(root, arm, enc, seed), label_path(root, arm, enc, seed)
        if rp.exists() and lp.exists():
            prev = json.loads(rp.read_text())
            if prev.get("embedding_vectors_sha256") == vec_sha:
                print(f"[68/cluster] {arm}/{enc}/seed{seed}: exists (same embedding sha), skipped", flush=True)
                continue
        t0 = time.time()
        r = recl.run_config(
            vecs, n_components=2, n_neighbors=15, min_cluster_size=mcs, seed=seed,
            min_dist=0.1, metric="euclidean", min_samples=None, cluster_selection_method="eom",
        )                                                                  # == 27e_mimic_replicate.py:102-105
        fit_sec = time.time() - t0
        labels = np.asarray(r["labels"]).astype(int)
        sizes = Counter(labels[labels != -1].tolist())
        largest = max(sizes.values()) if sizes else 0
        k = int(len(sizes))
        assert k == r["n_clusters"]
        df = pd.DataFrame({"admission_ID": ids, "cluster": labels})
        v67.atomic_write_parquet(df, lp)
        rec = {
            "analysis": "C_mimic_organism_ablation", "arm": arm, "encoder": enc, "seed": seed,
            "seed_in_1_10": seed != 42, "seeds_planned": SEEDS,
            "cohort_n": n, "unique_subject_n": n_subj,
            "model": manifest["model_source_repo"], "model_revision": manifest["model_revision"],
            "embedding_device": manifest["device"], "embedding_model_param_dtypes": manifest["model_param_dtypes"],
            "embedding_dim": int(vecs.shape[1]), "embedding_vectors_sha256": vec_sha,
            "embedding_corpus_sha256": manifest["corpus_sha256"],
            "umap_kwargs": {**UMAP_KWARGS, "random_state": seed},
            "hdbscan_kwargs": {**HDBSCAN_KWARGS, "min_cluster_size": mcs},
            "min_cluster_size_rule": MIN_CLUSTER_SIZE_RULE, "realised_min_cluster_size": mcs,
            "call": "02_recluster.run_config exactly as 27e_mimic_replicate.py:102-105",
            "n_clusters": k, "noise_frac": float((labels == -1).mean()), "n_noise": int((labels == -1).sum()),
            "largest_cluster_n": int(largest),
            "largest_cluster_frac_of_all_rows": float(largest / n),
            "largest_cluster_frac_of_clustered_rows": float(largest / max(1, int((labels != -1).sum()))),
            "k_le_4": bool(k <= 4), "dbcv": r["dbcv"], "fit_sec": round(fit_sec, 2),
            "labels_file": str(lp), "labels_sha256": v67.sha256_file(lp),
            "thread_config": recl._thread_config(), "library_versions": recl._library_versions(),
            **v67.env_record(),
        }
        v67.atomic_write_json(rec, rp)
        print(f"[68/cluster] {arm}/{enc}/seed{seed}: k={k} noise={rec['noise_frac']:.4%} "
              f"largest={largest} ({fit_sec:.0f}s)", flush=True)


# ============================================================================ aggregate stage
def _stats(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if len(x) == 0:
        return {"median": np.nan, "min": np.nan, "max": np.nan, "n": 0}
    return {"median": float(x.median()), "min": float(x.min()), "max": float(x.max()), "n": int(len(x))}


def agreement(a: np.ndarray, b: np.ndarray, prefix: str = "") -> dict:
    """ARI/AMI, both-non-noise (primary) and all-rows -- naming as 27e_mimic_replicate.py:140-151."""
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    keep = (a != -1) & (b != -1)
    return {
        f"ari_noise_excluded{prefix}": float(adjusted_rand_score(a[keep], b[keep])),
        f"ami_noise_excluded{prefix}": float(adjusted_mutual_info_score(a[keep], b[keep])),
        f"ari_all_rows{prefix}": float(adjusted_rand_score(a, b)),
        f"ami_all_rows{prefix}": float(adjusted_mutual_info_score(a, b)),
        f"n_both_non_noise{prefix}": int(keep.sum()),
    }


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append("NA" if np.isnan(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def stage_aggregate(root: Path, embed_root: Path, allow_partial: bool, arms: list, seeds: list) -> None:
    from sklearn.metrics import adjusted_rand_score
    v67.require_gate(embed_root, list(ENCODERS))              # Addendum 1 item 2a
    s40 = importlib.import_module("40_field_dominance")
    a7 = importlib.import_module("audit7_dominance_checks")     # conditional_agreement, combine ONLY
    man = v67.load_manifest()
    ids = man["admission_ID"].tolist()
    frame = v67.load_mimic_fields()
    assert frame["admission_ID"].tolist() == ids
    fields = s40.build_fields(frame, s40.MIMIC_CATEGORICAL, s40.MIMIC_CONTINUOUS)
    assert len(fields) == 19, len(fields)
    org = frame["organism_category"].astype(str).to_numpy()
    src = frame["source_category"].astype(str).to_numpy()
    strata_defs = {"organism_category": org, "organism_category x source_category": a7.combine(org, src)}
    strata_rendered = {  # is the stratifying field still rendered in the arm's text?
        "unmodified": {"organism_category": True, "organism_category x source_category": True},
        "no_organism": {"organism_category": False, "organism_category x source_category": False},
        "no_organism_no_resistance": {"organism_category": False, "organism_category x source_category": False},
        "no_charlson": {"organism_category": True, "organism_category x source_category": True},
        "no_source": {"organism_category": True, "organism_category x source_category": False},
        REF: {"organism_category": True, "organism_category x source_category": True},
    }

    all_arms = list(arms) + [REF]
    cache: dict = {}

    def labels(arm, enc, seed):
        key = (arm, enc, seed)
        if key not in cache:
            p = (v67.STORED_LABELS / f"{enc}_seed{seed}.parquet") if arm == REF else label_path(root, arm, enc, seed)
            if not p.exists():
                cache[key] = None
            else:
                df = pd.read_parquet(p).set_index("admission_ID")
                assert set(df.index) == set(ids) and len(df) == len(ids)
                cache[key] = df.loc[ids, "cluster"].to_numpy().astype(int)
        return cache[key]

    missing = [(a, e, s) for a in all_arms for e in ENCODERS for s in seeds if labels(a, e, s) is None]
    if missing and not allow_partial:
        raise SystemExit(f"{len(missing)} fits missing, e.g. {missing[:5]}; aggregate aborted (use --allow-partial to override)")
    if missing:
        print(f"[68/aggregate] WARNING partial: {len(missing)} fits missing", flush=True)

    # ------------------------------------------------ within-encoder rows
    rows = []
    for arm in all_arms:
        for enc in ENCODERS:
            for seed in seeds:
                y = labels(arm, enc, seed)
                if y is None:
                    continue
                n = len(y)
                sizes = Counter(y[y != -1].tolist())
                k = len(sizes)
                largest = max(sizes.values()) if sizes else 0
                ref = labels(REF, enc, seed)
                c0 = labels("unmodified", enc, seed)
                row = {
                    "arm": arm, "encoder": enc, "seed": seed, "seed_in_1_10": seed != 42,
                    "n": n, "n_unique_subjects": v67.N_SUBJECTS_EXPECTED,
                    "k": k, "noise_frac": float((y == -1).mean()),
                    "largest_cluster_n": int(largest),
                    "largest_cluster_frac_of_all_rows": largest / n,
                    "largest_cluster_frac_of_clustered_rows": largest / max(1, int((y != -1).sum())),
                    "k_le_4": bool(k <= 4), "degenerate_k_lt_5": bool(k < 5),
                }
                if ref is not None:
                    row.update(agreement(y, ref, "_vs_stored"))
                if c0 is not None:
                    row.update(agreement(y, c0, "_vs_c0"))
                # per-partition field attribution (noise dropped), 40_field_dominance.field_ami
                ami = s40.field_ami(y, fields)
                ordered = sorted(ami.items(), key=lambda kv: -kv[1])
                row["top_field"] = ordered[0][0]
                row["top_field_ami"] = ordered[0][1]
                row["dominance_margin"] = ordered[0][1] - ordered[1][1]
                row["runner_up_field"] = ordered[1][0]
                row["organism_category_ami"] = ami["organism_category"]
                for f, v in ami.items():
                    row[f"field_ami__{f}"] = v
                rows.append(row)
                print(f"[68/aggregate] within {arm}/{enc}/seed{seed}: k={k} top={row['top_field']} "
                      f"orgAMI={row['organism_category_ami']:.3f}", flush=True)
    within = pd.DataFrame(rows)

    # ------------------------------------------------ cross-encoder + conditional rows
    xrows, crows = [], []
    for arm in all_arms:
        for seed in seeds:
            a, b = labels(arm, ENCODERS[0], seed), labels(arm, ENCODERS[1], seed)
            if a is None or b is None:
                continue
            ka, kb = len(set(a[a != -1].tolist())), len(set(b[b != -1].tolist()))
            xr = {"arm": arm, "seed": seed, "seed_in_1_10": seed != 42, "k_bge_m3": ka, "k_qwen3_0_6b": kb,
                  "noise_frac_bge_m3": float((a == -1).mean()), "noise_frac_qwen3_0_6b": float((b == -1).mean()),
                  "pair_nondegenerate_k_ge_5_both": bool(ka >= 5 and kb >= 5), **agreement(a, b)}
            xrows.append(xr)
            for sname, strata in strata_defs.items():
                c = a7.conditional_agreement(a, b, strata, min_n=50)
                crows.append({"arm": arm, "seed": seed, "seed_in_1_10": seed != 42, "strata": sname,
                              "strata_field_rendered_in_arm": strata_rendered[arm][sname],
                              "n_strata": c["n_strata"], "n_rows": c["n_rows"],
                              "ami_within_strata_weighted": c["ami_weighted"],
                              "ari_within_strata_weighted": c["ari_weighted"],
                              "ami_unconditional_both_non_noise": c["ami_unconditional"],
                              "ari_unconditional_both_non_noise": c["ari_unconditional"],
                              "pair_nondegenerate_k_ge_5_both": xr["pair_nondegenerate_k_ge_5_both"]})
    cross = pd.DataFrame(xrows)
    cond = pd.DataFrame(crows)

    # ------------------------------------------------ within-encoder cross-seed stability (seeds 1..10)
    srows = []
    for arm in all_arms:
        for enc in ENCODERS:
            ys = {s: labels(arm, enc, s) for s in seeds if s != 42}
            if any(v is None for v in ys.values()) or len(ys) < 2:
                continue
            ari_x, ari_a = [], []
            for s1, s2 in itertools.combinations(sorted(ys), 2):
                ya, yb = ys[s1], ys[s2]
                keep = (ya != -1) & (yb != -1)
                ari_x.append(adjusted_rand_score(ya[keep], yb[keep]))
                ari_a.append(adjusted_rand_score(ya, yb))
            srows.append({"arm": arm, "encoder": enc, "n_pairs": len(ari_x),
                          "cross_seed_ari_noise_excluded_median": float(np.median(ari_x)),
                          "cross_seed_ari_noise_excluded_min": float(np.min(ari_x)),
                          "cross_seed_ari_noise_excluded_max": float(np.max(ari_x)),
                          "cross_seed_ari_all_rows_median": float(np.median(ari_a)),
                          "cross_seed_ari_all_rows_min": float(np.min(ari_a)),
                          "cross_seed_ari_all_rows_max": float(np.max(ari_a))})
    xseed = pd.DataFrame(srows)

    # ------------------------------------------------ reproduction check (G0.1 numbers)
    repro = {"note": ("NUMERICAL NOISE FLOOR (Addendum 1 item 2b): new `unmodified` fits vs the stored MIMIC fits (same "
                      "encoder, same seed); descriptive, no pass/fail criterion and nothing halts on it. The gross-error "
                      "detector (item 2a) is in gross_error_check_<encoder>.json."),
             "gross_error_check": {e: json.loads(v67.gate_path(embed_root, e).read_text()) for e in ENCODERS},
             "prereg_expected_seed42": {"bge-m3": {"k": 38, "noise_frac": 0.123575},
                                        "qwen3-0.6b": {"k": 62, "noise_frac": 0.073526}},
             "per_encoder": {}}
    for enc in ENCODERS:
        per_seed = {}
        for seed in seeds:
            y, r = labels("unmodified", enc, seed), labels(REF, enc, seed)
            if y is None or r is None:
                continue
            sy, sr = Counter(y[y != -1].tolist()), Counter(r[r != -1].tolist())
            per_seed[str(seed)] = {"k_new": len(sy), "k_stored": len(sr),
                                   "noise_new": float((y == -1).mean()), "noise_stored": float((r == -1).mean()),
                                   "largest_share_of_all_new": max(sy.values()) / len(y),
                                   "largest_share_of_all_stored": max(sr.values()) / len(r),
                                   "labels_array_identical": bool(np.array_equal(y, r)),
                                   **agreement(y, r)}
        mf = embed_root / "embed" / "unmodified" / enc / "embed_manifest.json"
        repro["per_encoder"][enc] = {
            "seed_42": per_seed.get("42"), "all_seeds": per_seed,
            "embedding_numeric_diagnostic_vs_stored": (json.loads(mf.read_text()).get("numeric_diagnostic_vs_stored_cpu_vectors")
                                                       if mf.exists() else None),
            "embedding_device": json.loads(mf.read_text()).get("device") if mf.exists() else None,
            "embedding_model_param_dtypes": json.loads(mf.read_text()).get("model_param_dtypes") if mf.exists() else None,
        }

    # ------------------------------------------------ write tables
    within_out = within.copy()
    v67.atomic_write_csv(within_out, root / "mimic_ablation.csv")
    v67.atomic_write_csv(cross, root / "mimic_ablation_cross_encoder.csv")
    v67.atomic_write_csv(cond, root / "mimic_ablation_conditional.csv")
    v67.atomic_write_csv(xseed, root / "mimic_ablation_cross_seed.csv")
    v67.atomic_write_json(repro, root / "reproduction_check.json")

    # ------------------------------------------------ summaries (numbers only)
    seedsets = {"seeds_42_and_1_10 (n=11)": None, "seeds_1_10 (n=10)": True}
    srows_ = []
    for arm in all_arms:
        for enc in ENCODERS:
            for sname, flag in seedsets.items():
                d = within[(within["arm"] == arm) & (within["encoder"] == enc)]
                if flag:
                    d = d[d["seed_in_1_10"]]
                if d.empty:
                    continue
                r = {"arm": arm, "encoder": enc, "seed_set": sname, "n_fits": len(d),
                     "k_median": d["k"].median(), "k_min": d["k"].min(), "k_max": d["k"].max(),
                     "frac_k_le_4": d["k_le_4"].mean(),
                     "noise_median": d["noise_frac"].median(), "noise_min": d["noise_frac"].min(),
                     "noise_max": d["noise_frac"].max(),
                     "largest_share_of_all_median": d["largest_cluster_frac_of_all_rows"].median(),
                     "largest_share_of_all_max": d["largest_cluster_frac_of_all_rows"].max(),
                     "organism_ami_median": d["organism_category_ami"].median(),
                     "organism_ami_min": d["organism_category_ami"].min(),
                     "organism_ami_max": d["organism_category_ami"].max(),
                     "top_field_margin_median": d["dominance_margin"].median()}
                mc = Counter(d["top_field"])
                top, cnt = mc.most_common(1)[0]
                r["modal_top_field"], r["modal_top_field_fits"] = top, f"{cnt}/{len(d)}"
                for ref in ("c0", "stored"):
                    for m in ("ari_noise_excluded", "ami_noise_excluded", "ari_all_rows", "ami_all_rows"):
                        col = f"{m}_vs_{ref}"
                        if col in d:
                            r[f"{col}_median"] = d[col].median()
                            r[f"{col}_min"] = d[col].min()
                            r[f"{col}_max"] = d[col].max()
                srows_.append(r)
    summary_within = pd.DataFrame(srows_)

    xs = []
    for arm in all_arms:
        for sname, flag in seedsets.items():
            d = cross[cross["arm"] == arm]
            if flag:
                d = d[d["seed_in_1_10"]]
            if d.empty:
                continue
            r = {"arm": arm, "seed_set": sname, "n_pairs": len(d)}
            for m in ("ari_noise_excluded", "ami_noise_excluded", "ari_all_rows", "ami_all_rows"):
                r[f"{m}_median"], r[f"{m}_min"], r[f"{m}_max"] = d[m].median(), d[m].min(), d[m].max()
            nd = d[d["pair_nondegenerate_k_ge_5_both"]]
            r["n_nondegenerate_pairs"] = len(nd)
            for m in ("ari_noise_excluded", "ami_noise_excluded"):
                r[f"{m}_median_nondegenerate_pairs"] = nd[m].median() if len(nd) else np.nan
            for sname2 in strata_defs:
                dc = cond[(cond["arm"] == arm) & (cond["strata"] == sname2)]
                if flag:
                    dc = dc[dc["seed_in_1_10"]]
                tag = "org" if sname2 == "organism_category" else "orgxsrc"
                r[f"cond_ami_{tag}_median"] = dc["ami_within_strata_weighted"].median()
                r[f"cond_ami_{tag}_min"] = dc["ami_within_strata_weighted"].min()
                r[f"cond_ami_{tag}_max"] = dc["ami_within_strata_weighted"].max()
                r[f"cond_ari_{tag}_median"] = dc["ari_within_strata_weighted"].median()
                r[f"cond_{tag}_n_strata_median"] = dc["n_strata"].median()
                r[f"cond_{tag}_n_rows_median"] = dc["n_rows"].median()
                r[f"cond_{tag}_strata_field_rendered"] = bool(dc["strata_field_rendered_in_arm"].iloc[0])
            xs.append(r)
    summary_cross = pd.DataFrame(xs)
    v67.atomic_write_csv(summary_within, root / "mimic_ablation_summary_within_encoder.csv")
    v67.atomic_write_csv(summary_cross, root / "mimic_ablation_summary_cross_encoder.csv")

    # ------------------------------------------------ edit sizes (from render QA + embed manifests)
    qa = json.loads((root / "render_qa.json").read_text())
    edit_rows = []
    for arm in arms:
        q = qa["arms"][arm]
        r = {"arm": arm, "lines_removed_per_doc_min": q["lines_removed_per_doc"]["min"],
             "lines_removed_per_doc_median": q["lines_removed_per_doc"]["median"],
             "lines_removed_per_doc_max": q["lines_removed_per_doc"]["max"],
             "lines_before_median": q["lines_per_doc_before"]["median"],
             "lines_after_median": q["lines_per_doc_after"]["median"],
             "chars_removed_median": q["chars_removed_per_doc"]["median"]}
        for enc in ENCODERS:
            mf = embed_root / "embed" / arm / enc / "embed_manifest.json"
            if mf.exists():
                m = json.loads(mf.read_text())
                r[f"{enc}_tokens_median"] = m["n_tokens_median"]
                r[f"{enc}_tokens_max"] = m["n_tokens_max"]
                r[f"{enc}_n_truncated"] = m["n_truncated"]
                tr = m.get("tokens_removed_vs_unmodified")
                r[f"{enc}_tokens_removed_median"] = tr["tokens_removed_per_doc_median"] if tr else 0.0
        edit_rows.append(r)
    edit = pd.DataFrame(edit_rows)

    # ------------------------------------------------ summary_raw.md  (NUMBERS ONLY)
    md = ["# Analysis C -- raw numbers (no interpretation)", "",
          f"Generated by `scripts/68_C_cluster.py --stage aggregate`. Seeds {seeds}. "
          f"n={v67.N_EXPECTED} admissions, {v67.N_SUBJECTS_EXPECTED} subjects. "
          "ARI/AMI columns named `*_noise_excluded` (both-non-noise rows) and `*_all_rows` (noise as one label). "
          "`stored_reference` = the stored out/27_mimic/replicate partitions (same code path).", "",
          "## Numerical noise floor (Addendum 1 2b): new unmodified fits vs stored fits, descriptive", ""]
    rp_rows = []
    for enc in ENCODERS:
        for seed, v in repro["per_encoder"][enc]["all_seeds"].items():
            rp_rows.append({"encoder": enc, "seed": seed, "k_new": v["k_new"], "k_stored": v["k_stored"],
                            "noise_new": v["noise_new"], "noise_stored": v["noise_stored"],
                            "largest_share_new": v["largest_share_of_all_new"],
                            "largest_share_stored": v["largest_share_of_all_stored"],
                            "labels_identical": v["labels_array_identical"],
                            "ari_noise_excluded": v["ari_noise_excluded"], "ari_all_rows": v["ari_all_rows"],
                            "ami_noise_excluded": v["ami_noise_excluded"]})
    md += [md_table(pd.DataFrame(rp_rows)), ""]
    for enc in ENCODERS:
        g = repro["gross_error_check"][enc]
        md.append(f"- gross-error detector {enc}: cosine min {g['cosine_min']:.8f} (threshold {g['threshold_cosine_min']}), "
                  f"median {g['cosine_median']:.8f}, n={g['n_docs_compared']}, pass={g['pass']}")
    md.append("")
    for enc in ENCODERS:
        d = repro["per_encoder"][enc].get("embedding_numeric_diagnostic_vs_stored")
        if d:
            md.append(f"- embedding vs stored vectors, {enc} (device={repro['per_encoder'][enc]['embedding_device']}, "
                      f"dtype={repro['per_encoder'][enc]['embedding_model_param_dtypes']}): cosine min {d['cosine_min']:.8f}, "
                      f"median {d['cosine_median']:.8f}, max|diff| {d['max_abs_elementwise_diff']:.3e}, "
                      f"bitwise-identical rows {d['bitwise_identical_rows']}/{d['n']}")
    md += ["", "## Edit sizes (arms are not size-matched)", "", md_table(edit, "{:.1f}"), "",
           "## Within-encoder: k, noise, largest cluster, distance from C0 and from stored, organism AMI", ""]
    keep_cols = ["arm", "encoder", "seed_set", "n_fits", "k_median", "k_min", "k_max", "frac_k_le_4",
                 "noise_median", "largest_share_of_all_median", "organism_ami_median", "organism_ami_min",
                 "organism_ami_max", "modal_top_field", "modal_top_field_fits", "top_field_margin_median"]
    md += [md_table(summary_within[[c for c in keep_cols if c in summary_within]]), "",
           "### vs the new unmodified arm (same encoder, same seed)", ""]
    c0_cols = ["arm", "encoder", "seed_set"] + [c for c in summary_within if "_vs_c0_" in c]
    md += [md_table(summary_within[c0_cols]), "", "### vs the stored seed-matched MIMIC fit", ""]
    st_cols = ["arm", "encoder", "seed_set"] + [c for c in summary_within if "_vs_stored_" in c]
    md += [md_table(summary_within[st_cols]), "", "### Within-encoder cross-seed ARI (seeds 1..10, 45 pairs)", "",
           md_table(xseed), "", "## Cross-encoder (BGE-M3 vs Qwen3-0.6B, same seed) per arm", ""]
    md += [md_table(summary_cross), "", "## Per-seed cross-encoder values", "",
           md_table(cross[["arm", "seed", "k_bge_m3", "k_qwen3_0_6b", "ari_noise_excluded", "ami_noise_excluded",
                           "ari_all_rows", "ami_all_rows", "pair_nondegenerate_k_ge_5_both"]]), "",
           "## Field attribution: median AMI over the 19 MIMIC fields (seeds 42 + 1..10)", ""]
    fcols = [c for c in within.columns if c.startswith("field_ami__")]
    med = within.groupby(["arm", "encoder"])[fcols].median().reset_index()
    med.columns = ["arm", "encoder"] + [c.replace("field_ami__", "") for c in fcols]
    md += [md_table(med, "{:.3f}"), "", "## Provenance", "",
           f"- git commit {v67.git_info()['git_commit']}; sys.executable {sys.executable}",
           f"- versions {json.dumps(v67.package_versions())}", ""]
    v67.atomic_write_text("\n".join(md) + "\n", root / "summary_raw.md")

    meta = {"analysis": "C_mimic_organism_ablation aggregate", "arms": all_arms, "encoders": ENCODERS, "seeds": seeds,
            "n_within_rows": len(within), "n_cross_rows": len(cross), "n_cond_rows": len(cond),
            "missing_fits": [list(m) for m in missing],
            "inputs": {f"{a}/{e}": json.loads((embed_root / "embed" / a / e / "embed_manifest.json").read_text())["vectors_sha256"]
                       for a in arms for e in ENCODERS if (embed_root / "embed" / a / e / "embed_manifest.json").exists()},
            **v67.env_record()}
    v67.atomic_write_json(meta, root / "aggregate_meta.json")
    print(f"[68/aggregate] wrote tables + summary_raw.md under {root}", flush=True)


# ============================================================================ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["cluster", "aggregate"], required=True)
    ap.add_argument("--arm", choices=ARM_ORDER)
    ap.add_argument("--encoder", choices=ENCODERS)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--out-root", default=str(v67.C_ROOT))
    ap.add_argument("--embed-root", default=None, help="default: same as --out-root (tests only)")
    ap.add_argument("--arms", nargs="+", default=ARM_ORDER, choices=ARM_ORDER, help="aggregate: arms to include")
    ap.add_argument("--allow-partial", action="store_true")
    args = ap.parse_args()
    root = Path(args.out_root)
    eroot = Path(args.embed_root) if args.embed_root else root
    if args.stage == "cluster":
        if not (args.arm and args.encoder):
            ap.error("--stage cluster needs --arm and --encoder")
        stage_cluster(root, eroot, args.arm, args.encoder, args.seeds)
    else:
        stage_aggregate(root, eroot, args.allow_partial, args.arms, args.seeds)


if __name__ == "__main__":
    main()
