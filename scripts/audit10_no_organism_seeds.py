#!/usr/bin/env python3
"""Audit 10: does the 3-large `no_organism` arm ever escape the k=2 solution?

out/25_audit4/seeds/ shows seeds 1-5,7-10 give k=2 / ARI 0.0081 but seed 6
gives k=29, 27% noise, ARI 0.095.  Refit seed 6 (to characterise it: organism
AMI, published-C0 recovery) and ten fresh seeds 11-20 (to estimate how often
the escape happens).  Frozen env, published config verbatim (same as
scripts/45_3large_ablation_seeds.py), reindexed to published order.
"""
import os
for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/data/tcmhwd/AI/AST_ai"); REV = ROOT / "phenotype_revision"
EMB = REV / "out/21_published_variants/embeddings/no_organism/merged.npy"
PUBLISHED = ROOT / "pipeline_new/out_pre/phenotype_clusters_labels.csv"
OUT = REV / "out/46_audit10"; OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [6] + list(range(11, 21))

def fit(x, seed):
    import hdbscan, umap
    coords = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean",
                       random_state=seed).fit_transform(x)
    return hdbscan.HDBSCAN(min_cluster_size=max(10, len(x) // 200), min_samples=None,
                           metric="euclidean", cluster_selection_method="eom").fit_predict(coords).astype(np.int32)

def one(seed):
    import pandas as pd
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    pub = pd.read_csv(PUBLISHED, usecols=["admission_ID", "phenotype_cluster"])
    ref = pub.phenotype_cluster.to_numpy(); pid = [str(i) for i in pub.admission_ID]
    ids = json.loads((EMB.parent / "merged_ids.json").read_text())
    pos = pd.Series(range(len(ids)), index=[str(i) for i in ids])
    x = np.load(EMB)[pos.loc[pid].to_numpy()]
    ds = pd.read_csv(ROOT / "manuscript/phenotype_ml_dataset.csv", low_memory=False,
                     usecols=["admission_ID", "BSI_episode_short_clean"]).drop_duplicates("admission_ID")
    org = pub.merge(ds, on="admission_ID", validate="one_to_one")["BSI_episode_short_clean"].astype(str).to_numpy()
    lab = fit(x, seed)
    keep = lab != -1; both = keep & (ref != -1)
    sizes = np.bincount(lab[keep])
    c0 = ref == 0
    # best-Jaccard recovery of published C0
    bestj = max(((lab == c) & c0).sum() / ((lab == c) | c0).sum() for c in range(sizes.size)) if sizes.size else 0.0
    res = dict(seed=seed, k=int(sizes.size), noise=float((~keep).mean()),
               largest_share=float(sizes.max() / keep.sum()) if sizes.size else None,
               ari_noise_excl=float(adjusted_rand_score(lab[both], ref[both])),
               ami_noise_excl=float(adjusted_mutual_info_score(lab[both], ref[both])),
               ari_all_row=float(adjusted_rand_score(lab, ref)),
               ami_organism=float(adjusted_mutual_info_score(lab[keep], org[keep])),
               c0_best_jaccard=float(bestj))
    np.save(OUT / f"no_organism_labels_seed{seed}.npy", lab)
    print(json.dumps(res), flush=True)
    return res

if __name__ == "__main__":
    from multiprocessing import Pool
    with Pool(len(SEEDS)) as p:
        out = p.map(one, SEEDS)
    (OUT / "no_organism_extra_seeds.json").write_text(json.dumps(out, indent=1))
    print("[audit10] wrote", OUT / "no_organism_extra_seeds.json")
