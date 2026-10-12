#!/usr/bin/env python3
"""Field-dominance ablation for Qwen3: does removing the dominant field
destroy the partition, while removing a non-dominant one does not?

Everything so far says Qwen3's Korean partition is ALIGNED with the first
antibiotic name (AMI 0.947, 50/50 fits).  Alignment is not causation: the
partition could be organised by something the antibiotic merely correlates
with.  The test that separates those is removal.

The precedent is already on disk for a different encoder.  For OpenAI
3-large, whose dominant field is the organism line, the published-template
variants give:

    no_organism    k = 2,  ARI 0.007 vs the published partition   (destroyed)
    no_department  k = 18, ARI 0.881                              (intact)

That is an ablation, not a correlation: removing the dominant field
collapses the partition and removing a non-dominant field does not.

This script builds the same contrast for Qwen3, whose dominant field is the
therapy block rather than the organism line, so the two encoders' predicted
outcomes are swapped:

    no_therapy    -- removes Qwen3's dominant field   -> expect collapse
    no_organism   -- removes a NON-dominant field for -> expect intact
                     Qwen3 (its organism AMI is 0.039)
    full          -- unmodified template               -> must reproduce the
                     stored Qwen3 partition, which is what makes the other
                     two interpretable at all

The `full` arm is not a formality.  These narratives were re-embedded on a
different date with the same script; if `full` did not reproduce the stored
partition, a collapse in `no_therapy` could be an embedding-run artefact
rather than the ablation.

Clustering runs in the FROZEN environment, which is why this is a separate
script from the embedding: /home/programs/anaconda3/envs/gpt is the only
environment that reproduces this project's published partitions, and it is
read-only.
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
OUT = REV / "out/44_ablation"
EMB = REV / "out/41_audit7/embeddings"

SEEDS = [42, 1, 2, 3]
VARIANTS = {
    "full": EMB / "ablation_full/merged.npy",
    "no_therapy": EMB / "ablation_no_therapy/merged.npy",
    "no_organism": EMB / "ablation_no_organism/merged.npy",
}
STORED = REV / "out/18_encoder_dec/recluster/qwen3-0.6b/labels.npz"
IDS = REV / "out/18_encoder_dec/embeddings/qwen3-0.6b/merged_ids.json"
RAW_FIELDS = REV / "out/20_onehot/features/raw_fields.parquet"


def fit(x: np.ndarray, seed: int) -> np.ndarray:
    import hdbscan
    import umap
    coords = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                       metric="euclidean", random_state=seed).fit_transform(x)
    return hdbscan.HDBSCAN(
        min_cluster_size=max(10, len(x) // 200), min_samples=None,
        metric="euclidean", cluster_selection_method="eom",
    ).fit_predict(coords).astype(np.int32)


def agree(a: np.ndarray, b: np.ndarray) -> tuple:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    both = (a != -1) & (b != -1)
    return (float(adjusted_rand_score(a[both], b[both])),
            float(adjusted_mutual_info_score(a[both], b[both])))


def field_ami(labels: np.ndarray, values: np.ndarray) -> float:
    from sklearn.metrics import adjusted_mutual_info_score
    keep = labels != -1
    return float(adjusted_mutual_info_score(labels[keep], values[keep]))


def main() -> None:
    import pandas as pd

    OUT.mkdir(parents=True, exist_ok=True)

    ids = json.loads(IDS.read_text())
    stored_payload = np.load(STORED, allow_pickle=True)
    stored = stored_payload[[k for k in stored_payload.files if "label" in k.lower()][0]]

    # Every variant must carry the same admissions in the same order as the
    # stored partition, or the comparisons below are meaningless.
    for name, path in VARIANTS.items():
        variant_ids = json.loads((path.parent / "merged_ids.json").read_text())
        if variant_ids != ids:
            raise RuntimeError(f"{name}: id order differs from the stored partition")

    raw = pd.read_parquet(RAW_FIELDS)[["admission_ID", "tx__admission_antibiotics"]]

    def first_antibiotic(value):
        if isinstance(value, str) and value.startswith("["):
            parsed = json.loads(value)
            return str(parsed[0]) if parsed else "NA"
        return "NA"

    raw["first_antibiotic"] = raw["tx__admission_antibiotics"].apply(first_antibiotic)
    dataset = pd.read_csv(
        ROOT / "manuscript/phenotype_ml_dataset.csv", low_memory=False,
        usecols=["admission_ID", "BSI_episode_short_clean"],
    ).drop_duplicates("admission_ID")
    fields = raw.merge(dataset, on="admission_ID", validate="one_to_one")
    fields = fields.set_index("admission_ID").loc[ids]
    antibiotic = fields["first_antibiotic"].astype(str).to_numpy()
    organism = fields["BSI_episode_short_clean"].astype(str).to_numpy()

    labels: dict = {}
    result = {"seeds": SEEDS, "per_fit": {}, "vs_stored": {},
              "cross_seed": {}, "cross_variant_seed_matched": {}}

    for name, path in VARIANTS.items():
        x = np.load(path)
        for seed in SEEDS:
            key = f"{name}|{seed}"
            fitted = fit(x, seed)
            labels[key] = fitted
            ari, ami = agree(fitted, stored)
            result["per_fit"][key] = {
                "k": int(len(set(fitted.tolist()) - {-1})),
                "noise": float((fitted == -1).mean()),
                "ari_vs_stored": ari,
                "ami_vs_stored": ami,
                "ami_first_antibiotic": field_ami(fitted, antibiotic),
                "ami_organism": field_ami(fitted, organism),
            }
            block = result["per_fit"][key]
            print(f"[{key}] k={block['k']:3d} noise={block['noise']:.3f} "
                  f"ARIvsStored={ari:.4f} AMI_ab={block['ami_first_antibiotic']:.4f} "
                  f"AMI_org={block['ami_organism']:.4f}", flush=True)
        np.savez_compressed(OUT / f"labels_{name}.npz",
                            **{str(s): labels[f"{name}|{s}"] for s in SEEDS})

    for name in VARIANTS:
        pairs = [agree(labels[f"{name}|{a}"], labels[f"{name}|{b}"])
                 for a, b in itertools.combinations(SEEDS, 2)]
        result["cross_seed"][name] = {
            "ari_median": float(np.median([p[0] for p in pairs])),
            "ari_min": float(np.min([p[0] for p in pairs])),
            "ami_median": float(np.median([p[1] for p in pairs])),
            "n_pairs": len(pairs),
        }

    for a, b in itertools.combinations(VARIANTS, 2):
        per_seed = [dict(seed=s, ari=agree(labels[f"{a}|{s}"], labels[f"{b}|{s}"])[0],
                         ami=agree(labels[f"{a}|{s}"], labels[f"{b}|{s}"])[1])
                    for s in SEEDS]
        result["cross_variant_seed_matched"][f"{a} vs {b}"] = {
            "per_seed": per_seed,
            "ari_median": float(np.median([p["ari"] for p in per_seed])),
            "ami_median": float(np.median([p["ami"] for p in per_seed])),
        }

    (OUT / "ablation.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\n[ablation] wrote {OUT}/ablation.json")


if __name__ == "__main__":
    main()
