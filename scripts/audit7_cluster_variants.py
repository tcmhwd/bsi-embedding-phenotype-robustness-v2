#!/usr/bin/env python3
"""Audit 7 (question B) -- does the Korean Qwen3 partition survive removal of
the rendered cluster-ID tokens, and does its seed reproducibility survive?

MUST run in the frozen environment (/home/programs/anaconda3/envs/gpt), the
same one that produced out/18_encoder_dec/recluster/qwen3-0.6b (seed 42,
k=24).  Nothing in that environment is modified; it is only executed.

For each embedding matrix
    original          out/18_encoder_dec/embeddings/qwen3-0.6b/merged.npy
    no_summary_tokens out/41_audit7/embeddings/no_summary_tokens/merged.npy
    no_cluster_ids    out/41_audit7/embeddings/no_cluster_ids/merged.npy
and each seed in {42, 1, 2, 3, 4}: UMAP(2, nn15, md0.1, euclidean, seed) ->
HDBSCAN(mcs=max(10, n//200)=161, eom).  Thread count pinned to 1.

Reports: k / noise per fit; cross-seed ARI+AMI within each variant; ARI+AMI
of every fit against the STORED original seed-42 partition (the 'original'
seed-42 refit must reproduce it at ARI ~1 -- that is the environment check);
AMI of every fit with admission_cluster / empiric_cluster / organism /
department; and a 100-draw permutation null for AMI(fit, admission_cluster).
"""
import os

for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
ROOT = REV.parent
OUT = REV / "out/41_audit7/cluster_variants"
SEEDS = [42, 1, 2, 3, 4]
VARIANTS = {
    "original": REV / "out/18_encoder_dec/embeddings/qwen3-0.6b/merged.npy",
    "no_summary_tokens": REV / "out/41_audit7/embeddings/no_summary_tokens/merged.npy",
    "no_cluster_ids": REV / "out/41_audit7/embeddings/no_cluster_ids/merged.npy",
}
STORED = REV / "out/18_encoder_dec/recluster/qwen3-0.6b/labels.npz"
IDS = REV / "out/18_encoder_dec/embeddings/qwen3-0.6b/merged_ids.json"
DATASET = ROOT / "manuscript/phenotype_ml_dataset.csv"


def fit(x: np.ndarray, seed: int) -> np.ndarray:
    import hdbscan
    import umap
    coords = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean",
                       random_state=seed).fit_transform(x)
    return hdbscan.HDBSCAN(min_cluster_size=max(10, len(x) // 200), min_samples=None,
                           metric="euclidean", cluster_selection_method="eom").fit_predict(coords).astype(np.int32)


def agree(a, b):
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    both = (a != -1) & (b != -1)
    return float(adjusted_rand_score(a[both], b[both])), float(adjusted_mutual_info_score(a[both], b[both]))


def field_ami(lab, values):
    from sklearn.metrics import adjusted_mutual_info_score
    keep = lab != -1
    return float(adjusted_mutual_info_score(lab[keep], values[keep]))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[audit7-cluster] python: {sys.executable}", flush=True)
    import umap, hdbscan, numpy
    print(f"[audit7-cluster] umap {umap.__version__} numpy {numpy.__version__}", flush=True)

    ids = json.load(open(IDS))
    stored = np.load(STORED)
    assert [int(i) for i in stored["admission_ID"]] == [int(i) for i in ids]
    stored_lab = stored["labels"].astype(np.int32)

    ds = pd.read_csv(DATASET, low_memory=False, usecols=[
        "admission_ID", "admission_cluster", "empiric_cluster", "BSI_episode_short_clean",
        "treatment_department_eng", "ab_admission_category_clean", "n_bsi_episodes"]).drop_duplicates("admission_ID")
    ds = ds.set_index("admission_ID").loc[ids].reset_index()
    fields = {c: ds[c].astype(str).to_numpy() for c in
              ["admission_cluster", "empiric_cluster", "BSI_episode_short_clean",
               "treatment_department_eng", "ab_admission_category_clean", "n_bsi_episodes"]}

    labels = {}
    for variant, path in VARIANTS.items():
        if not path.exists():
            print(f"[audit7-cluster] {variant}: {path} missing, skipped", flush=True)
            continue
        x = np.load(path)
        assert len(x) == len(ids)
        for seed in SEEDS:
            key = f"{variant}|{seed}"
            cache = OUT / f"labels_{variant}_seed{seed}.npy"
            if cache.exists():
                labels[key] = np.load(cache)
            else:
                t0 = time.time()
                labels[key] = fit(x, seed)
                np.save(cache, labels[key])
                print(f"[audit7-cluster] {key}: k={len(set(labels[key].tolist()) - {-1})} "
                      f"noise={(labels[key] == -1).mean():.4f} ({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.default_rng(7)
    result = {"seeds": SEEDS, "per_fit": {}, "cross_seed": {}, "vs_stored_original_seed42": {},
              "cross_variant_seed_matched": {}}
    for key, lab in labels.items():
        ami_fields = {f: field_ami(lab, v) for f, v in fields.items()}
        null = [field_ami(lab, rng.permutation(fields["admission_cluster"])) for _ in range(100)]
        r, m = agree(lab, stored_lab)
        result["per_fit"][key] = {
            "k": int(len(set(lab.tolist()) - {-1})), "noise": float((lab == -1).mean()),
            "ami_by_field": ami_fields,
            "ami_admission_cluster_perm_null": {"mean": float(np.mean(null)), "max": float(np.max(null))},
            "ari_vs_stored": r, "ami_vs_stored": m,
        }
    for variant in VARIANTS:
        keys = [k for k in labels if k.startswith(variant + "|")]
        if len(keys) < 2:
            continue
        aris, amis = [], []
        for a, b in itertools.combinations(keys, 2):
            r, m = agree(labels[a], labels[b]); aris.append(r); amis.append(m)
        result["cross_seed"][variant] = {
            "n_pairs": len(aris), "ari_median": float(np.median(aris)), "ari_min": float(np.min(aris)),
            "ari_max": float(np.max(aris)), "ami_median": float(np.median(amis)), "ami_min": float(np.min(amis))}
    for va, vb in itertools.combinations(VARIANTS, 2):
        vals = []
        for seed in SEEDS:
            ka, kb = f"{va}|{seed}", f"{vb}|{seed}"
            if ka in labels and kb in labels:
                r, m = agree(labels[ka], labels[kb]); vals.append({"seed": seed, "ari": r, "ami": m})
        if vals:
            result["cross_variant_seed_matched"][f"{va} vs {vb}"] = {
                "per_seed": vals, "ari_median": float(np.median([v["ari"] for v in vals])),
                "ami_median": float(np.median([v["ami"] for v in vals]))}
    (OUT / "cluster_variants.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_fit"}, indent=1))
    for key, blk in result["per_fit"].items():
        print(f"{key}: k={blk['k']} noise={blk['noise']:.3f} ARIvsStored={blk['ari_vs_stored']:.3f} "
              f"AMI adm={blk['ami_by_field']['admission_cluster']:.3f} emp={blk['ami_by_field']['empiric_cluster']:.3f} "
              f"abcat={blk['ami_by_field']['ab_admission_category_clean']:.3f} null_max={blk['ami_admission_cluster_perm_null']['max']:.4f}")


if __name__ == "__main__":
    main()
