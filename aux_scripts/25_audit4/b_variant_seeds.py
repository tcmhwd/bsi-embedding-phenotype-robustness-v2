#!/usr/bin/env python3
"""Audit 4, check A -- seed sensitivity of the Task-21 variant ARIs.
Same code-of-record config as scripts/21c_cluster_frozen.py (frozen env,
threads pinned to 1), but random_state = --seed instead of 42. One
(variant, seed) per invocation; writes out/25_audit4/seeds/<variant>_seed<s>.json.
MUST run with /home/programs/anaconda3/envs/gpt/bin/python3.
"""
import os
for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"
import argparse, importlib, json, sys, time
from pathlib import Path
import numpy as np
import hdbscan, umap

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
sys.path.insert(0, str(REV / "scripts"))
c21 = importlib.import_module("21_common")
EMB = REV / "out/21_published_variants/embeddings"
OUT = REV / "out/25_audit4/seeds"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{a.variant}_seed{a.seed}.json"
    if dest.exists():
        print("exists, skip", dest); return
    pub = c21.load_published_frame()
    ids = pub["admission_ID"].to_numpy()
    ref = pub[c21.PUBLISHED_CLUSTER_COL].to_numpy(dtype=np.int64)
    vecs = np.load(EMB / a.variant / "merged.npy")
    vid = json.loads((EMB / a.variant / "merged_ids.json").read_text())
    X = c21.align_embeddings_to_admission_ids(vecs, vid, ids)
    mcs = c21.min_cluster_size_for(X.shape[0])
    t0 = time.time()
    space = umap.UMAP(random_state=a.seed, **c21.UMAP_KW).fit_transform(X)
    lab = hdbscan.HDBSCAN(min_cluster_size=mcs, core_dist_n_jobs=1, **c21.HDBSCAN_KW).fit_predict(
        np.ascontiguousarray(space, dtype=np.float64))
    res = dict(variant=a.variant, seed=a.seed, k=int(len(set(lab.tolist()) - {-1})),
               noise_frac=float((lab == -1).mean()),
               agreement_vs_published=c21.agreement(ref, lab),
               jaccard_vs_published={str(c): c21.jaccard_for_cluster(ref, lab, c) for c in sorted(set(ref.tolist()) - {-1})},
               wall_sec=time.time() - t0, python=sys.executable, umap=umap.__version__)
    c21.atomic_write_json(res, dest)
    print(json.dumps({k: v for k, v in res.items() if k != "jaccard_vs_published"}), flush=True)


if __name__ == "__main__":
    main()
