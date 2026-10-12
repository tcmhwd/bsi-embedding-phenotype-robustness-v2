#!/usr/bin/env python3
"""Seed replication for the 3-large ablation arms.

The ablation figure rests on two encoders, but the two arms are not equally
supported: the Qwen3 arms carry four clustering seeds each, while the
3-large arms are a single fit per variant from the published-template pass.
A reviewer will ask whether ARI 0.008 for `no_organism` is a seed accident,
and the honest answer today is that nobody has looked.

The conclusion is very unlikely to move -- 0.008 against 0.954 for the
control is not a gap seeds close -- but "unlikely to move" is not the same
as "checked", and this project has been wrong before about exactly that
kind of assumption.  The embeddings are already on disk, so the check costs
CPU time and no API calls.

Runs in the FROZEN environment, the only stack that reproduces this
project's published partitions, and uses the published configuration
verbatim.
"""
import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data/tcmhwd/AI/AST_ai")
REV = ROOT / "phenotype_revision"
EMB = REV / "out/21_published_variants/embeddings"
PUBLISHED = ROOT / "pipeline_new/out_pre/phenotype_clusters_labels.csv"
OUT = REV / "out/45_3large_seeds"

# Seed 42 is the published configuration's seed and is included so the new
# fits can be checked against the values already reported.
SEEDS = [42, 1, 2, 3]
VARIANTS = ["full", "no_department", "no_therapy", "no_organism"]


def fit(x: np.ndarray, seed: int) -> np.ndarray:
    import hdbscan
    import umap
    coords = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                       metric="euclidean", random_state=seed).fit_transform(x)
    return hdbscan.HDBSCAN(
        min_cluster_size=max(10, len(x) // 200), min_samples=None,
        metric="euclidean", cluster_selection_method="eom",
    ).fit_predict(coords).astype(np.int32)


def agree(a: np.ndarray, b: np.ndarray) -> dict:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    both = (a != -1) & (b != -1)
    return {
        "ari_noise_excl": float(adjusted_rand_score(a[both], b[both])),
        "ami_noise_excl": float(adjusted_mutual_info_score(a[both], b[both])),
        "ari_all_row": float(adjusted_rand_score(a, b)),
    }


def main() -> None:
    import pandas as pd
    OUT.mkdir(parents=True, exist_ok=True)

    published = pd.read_csv(PUBLISHED, usecols=["admission_ID", "phenotype_cluster"])
    reference = published["phenotype_cluster"].to_numpy()
    published_ids = published["admission_ID"].tolist()

    # The variant embeddings are NOT stored in the published label order --
    # same 32,329 admissions, different sequence (the first version of this
    # script compared labels[i] against reference[i] and produced ARI ~0 for
    # every arm including the unmodified one, which is how the bug surfaced).
    # Every variant is therefore reindexed into the published order before
    # any comparison, and the reindex is asserted rather than assumed.
    def load_aligned(path: Path) -> np.ndarray:
        ids = json.loads((path.parent / "merged_ids.json").read_text())
        if set(map(str, ids)) != set(map(str, published_ids)):
            raise RuntimeError(f"{path.parent.name}: admission id sets differ")
        position = pd.Series(range(len(ids)), index=[str(i) for i in ids])
        order = position.loc[[str(i) for i in published_ids]].to_numpy()
        return np.load(path)[order]

    dataset = pd.read_csv(
        ROOT / "manuscript/phenotype_ml_dataset.csv", low_memory=False,
        usecols=["admission_ID", "BSI_episode_short_clean"],
    ).drop_duplicates("admission_ID")
    organism = (published.merge(dataset, on="admission_ID", validate="one_to_one")
                ["BSI_episode_short_clean"].astype(str).to_numpy())

    from sklearn.metrics import adjusted_mutual_info_score

    labels: dict = {}
    result = {"seeds": SEEDS, "per_fit": {}, "cross_seed": {}, "summary": {}}

    for variant in VARIANTS:
        path = EMB / variant / "merged.npy"
        if not path.exists():
            print(f"[skip] {variant}: no embedding", flush=True)
            continue
        x = load_aligned(path)
        if len(x) != len(reference):
            raise RuntimeError(f"{variant}: {len(x)} rows vs {len(reference)} labels")
        for seed in SEEDS:
            fitted = fit(x, seed)
            labels[f"{variant}|{seed}"] = fitted
            keep = fitted != -1
            block = {
                "k": int(len(set(fitted.tolist()) - {-1})),
                "noise": float((fitted == -1).mean()),
                **agree(fitted, reference),
                "ami_organism": float(
                    adjusted_mutual_info_score(fitted[keep], organism[keep])),
            }
            result["per_fit"][f"{variant}|{seed}"] = block
            print(f"[{variant}|{seed}] k={block['k']:3d} noise={block['noise']:.3f} "
                  f"ARI_ne={block['ari_noise_excl']:.4f} "
                  f"AMI_org={block['ami_organism']:.4f}", flush=True)
        np.savez_compressed(OUT / f"labels_{variant}.npz",
                            **{str(s): labels[f"{variant}|{s}"] for s in SEEDS})

        values = [result["per_fit"][f"{variant}|{s}"]["ari_noise_excl"] for s in SEEDS]
        pairs = [agree(labels[f"{variant}|{a}"], labels[f"{variant}|{b}"])["ari_noise_excl"]
                 for a, b in itertools.combinations(SEEDS, 2)]
        result["cross_seed"][variant] = {
            "ari_median": float(np.median(pairs)),
            "ari_min": float(np.min(pairs)),
            "n_pairs": len(pairs),
        }
        result["summary"][variant] = {
            "ari_vs_published_median": float(np.median(values)),
            "ari_vs_published_min": float(np.min(values)),
            "ari_vs_published_max": float(np.max(values)),
            "k_range": [int(min(result["per_fit"][f"{variant}|{s}"]["k"] for s in SEEDS)),
                        int(max(result["per_fit"][f"{variant}|{s}"]["k"] for s in SEEDS))],
        }

    (OUT / "three_large_seeds.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\n[3large-seeds] wrote {OUT}/three_large_seeds.json")


if __name__ == "__main__":
    main()
