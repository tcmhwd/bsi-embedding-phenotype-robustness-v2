#!/usr/bin/env python3
"""Audit 9 -- cluster the antibiotic-name-masked Qwen3 embeddings in the FROZEN
environment with the code-of-record configuration (copied verbatim from
scripts/44_ablation_cluster.py: UMAP n_components=2, n_neighbors=15,
min_dist=0.1, euclidean, random_state=seed; HDBSCAN min_cluster_size=
max(10, n//200), min_samples=None, eom; threads pinned to 1).

Variants: mask_ab_names_all, mask_ab1_name_only (out/45_audit9/embeddings).
Seeds: 42, 1, 2, 3 -- the same set as out/44_ablation so the cross-seed and
vs-stored numbers are directly comparable.
Writes out/45_audit9/mask_ablation.json and labels_<variant>.npz.
"""
import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import itertools
import json
from pathlib import Path

import numpy as np

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
ROOT = REV.parent
OUT = REV / "out/45_audit9"
EMB = OUT / "embeddings"
SEEDS = [42, 1, 2, 3]
VARIANTS = ["mask_ab_names_all", "mask_ab1_name_only"]
STORED = REV / "out/18_encoder_dec/recluster/qwen3-0.6b/labels.npz"
IDS = REV / "out/18_encoder_dec/embeddings/qwen3-0.6b/merged_ids.json"
RAW_FIELDS = REV / "out/20_onehot/features/raw_fields.parquet"


def fit(x, seed):
    import hdbscan
    import umap
    coords = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                       metric="euclidean", random_state=seed).fit_transform(x)
    return hdbscan.HDBSCAN(min_cluster_size=max(10, len(x) // 200), min_samples=None,
                           metric="euclidean", cluster_selection_method="eom",
                           ).fit_predict(coords).astype(np.int32)


def agree(a, b):
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    both = (a != -1) & (b != -1)
    return (float(adjusted_rand_score(a[both], b[both])),
            float(adjusted_mutual_info_score(a[both], b[both])))


def field_ami(labels, values):
    from sklearn.metrics import adjusted_mutual_info_score
    keep = labels != -1
    return float(adjusted_mutual_info_score(labels[keep], values[keep]))


def main():
    import pandas as pd
    ids = json.loads(IDS.read_text())
    stored = np.load(STORED, allow_pickle=True)["labels"]
    for v in VARIANTS:
        if json.loads((EMB / v / "merged_ids.json").read_text()) != ids:
            raise RuntimeError(f"{v}: id order differs")
    raw = pd.read_parquet(RAW_FIELDS)[["admission_ID", "tx__admission_antibiotics"]]

    def first_ab(value):
        if isinstance(value, str) and value.startswith("["):
            p = json.loads(value)
            return str(p[0]) if p else "NA"
        return "NA"

    def second_ab(value):
        if isinstance(value, str) and value.startswith("["):
            p = json.loads(value)
            return str(p[1]) if len(p) > 1 else "NA"
        return "NA"

    raw["first_antibiotic"] = raw["tx__admission_antibiotics"].apply(first_ab)
    raw["second_antibiotic"] = raw["tx__admission_antibiotics"].apply(second_ab)
    raw["n_antibiotic_lines"] = raw["tx__admission_antibiotics"].apply(
        lambda v: len(json.loads(v)) if isinstance(v, str) and v.startswith("[") else 0)
    ds = pd.read_csv(ROOT / "manuscript/phenotype_ml_dataset.csv", low_memory=False,
                     usecols=["admission_ID", "BSI_episode_short_clean"]).drop_duplicates("admission_ID")
    f = raw.merge(ds, on="admission_ID", validate="one_to_one").set_index("admission_ID").loc[ids]
    ab1 = f["first_antibiotic"].astype(str).to_numpy()
    ab2 = f["second_antibiotic"].astype(str).to_numpy()
    nab = f["n_antibiotic_lines"].astype(str).to_numpy()
    org = f["BSI_episode_short_clean"].astype(str).to_numpy()

    labels, res = {}, {"seeds": SEEDS, "per_fit": {}, "cross_seed": {}}
    for v in VARIANTS:
        x = np.load(EMB / v / "merged.npy")
        for s in SEEDS:
            lab = fit(x, s)
            labels[f"{v}|{s}"] = lab
            ari, ami = agree(lab, stored)
            res["per_fit"][f"{v}|{s}"] = dict(
                k=int(len(set(lab.tolist()) - {-1})), noise=float((lab == -1).mean()),
                ari_vs_stored=ari, ami_vs_stored=ami,
                ami_first_antibiotic=field_ami(lab, ab1), ami_second_antibiotic=field_ami(lab, ab2),
                ami_n_antibiotic_lines=field_ami(lab, nab), ami_organism=field_ami(lab, org))
            print(v, s, res["per_fit"][f"{v}|{s}"], flush=True)
        np.savez_compressed(OUT / f"labels_{v}.npz", **{str(s): labels[f"{v}|{s}"] for s in SEEDS})
        pairs = [agree(labels[f"{v}|{a}"], labels[f"{v}|{b}"])[0] for a, b in itertools.combinations(SEEDS, 2)]
        res["cross_seed"][v] = {"ari_median": float(np.median(pairs)), "ari_min": float(np.min(pairs)),
                                "pairs": [float(p) for p in pairs]}
    (OUT / "mask_ablation.json").write_text(json.dumps(res, indent=2, sort_keys=True))
    print("[audit9] wrote", OUT / "mask_ablation.json")


if __name__ == "__main__":
    main()
