#!/usr/bin/env python3
"""Audit 4, check A -- does the *embedding* lose the information under
shuffle_order / json_format, or only the UMAP+HDBSCAN partition?

For each Task-21 variant embedding (3-large, out/21_published_variants/embeddings):
  (i)   kNN label transfer (k=25, cosine, StratifiedGroupKFold(5, patient_ID),
        seed 42, noise excluded) predicting the PUBLISHED 16 labels -- identical
        protocol to scripts/18d_s6_knn.py -- per-cluster F1 + macro-F1.
  (ii)  15-NN overlap with the `full` re-embedding's 15-NN graph (fraction of
        each row's 15 nearest neighbours shared), the graph UMAP(n_neighbors=15)
        consumes.
  (iii) 15-NN purity of published cluster 0 in the variant embedding.
Writes only under out/25_audit4/. bsi_pheno interpreter.
"""
import json, sys, os, time
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, precision_recall_fscore_support

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
sys.path.insert(0, str(REV / "scripts"))
import importlib
c18 = importlib.import_module("18_common")
EMB = REV / "out/21_published_variants/embeddings"
OUT = REV / "out/25_audit4"
VARIANTS = sys.argv[1:] or ["full", "shuffle_order", "json_format", "no_organism", "no_definitions",
                             "null_dropped", "no_headers", "index_window", "no_dates", "no_department",
                             "no_summary_tokens", "host_source_severity_only"]
K_GRAPH = 15
K_KNN = 25
BLOCK = 2000


def load_variant(v, target_ids):
    vecs = np.load(EMB / v / "merged.npy")
    ids = json.loads((EMB / v / "merged_ids.json").read_text())
    X = c18.align_embeddings_to_admission_ids(vecs, ids, target_ids).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def knn_graph(X, k):
    n = X.shape[0]
    out = np.empty((n, k), dtype=np.int32)
    for s in range(0, n, BLOCK):
        e = min(n, s + BLOCK)
        S = X[s:e] @ X.T
        S[np.arange(e - s), np.arange(s, e)] = -np.inf
        part = np.argpartition(-S, k, axis=1)[:, :k]
        out[s:e] = part
    return out


def knn_transfer(X, y, groups):
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    yt, yp = [], []
    for tr, te in skf.split(np.zeros(len(y)), y, groups):
        Xtr, Xte = X[tr], X[te]
        preds = np.empty(len(te), dtype=y.dtype)
        for s in range(0, len(te), BLOCK):
            e = min(len(te), s + BLOCK)
            S = Xte[s:e] @ Xtr.T
            nn = np.argpartition(-S, K_KNN, axis=1)[:, :K_KNN]
            lab = y[tr][nn]
            for i in range(e - s):
                vals, cnt = np.unique(lab[i], return_counts=True)
                preds[s + i] = vals[np.argmax(cnt)]
        yt.append(y[te]); yp.append(preds)
    yt = np.concatenate(yt); yp = np.concatenate(yp)
    labels = sorted(set(y.tolist()))
    p, r, f, s_ = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    return float(f1_score(yt, yp, average="macro")), {int(c): float(ff) for c, ff in zip(labels, f)}


def main():
    frame = c18.load_published_frame()
    ids = frame["admission_ID"].to_numpy()
    pub = frame[c18.PUBLISHED_CLUSTER_COL].to_numpy()
    groups = frame["patient_ID"].to_numpy()
    keep = pub != -1
    Xfull = load_variant("full", ids)
    Gfull = knn_graph(Xfull, K_GRAPH)
    Gfull_sets = [set(r.tolist()) for r in Gfull]
    results = {}
    for v in VARIANTS:
        t0 = time.time()
        X = Xfull if v == "full" else load_variant(v, ids)
        G = Gfull if v == "full" else knn_graph(X, K_GRAPH)
        overlap = np.array([len(Gfull_sets[i].intersection(G[i].tolist())) / K_GRAPH for i in range(len(ids))])
        # cluster-0 15-NN purity
        m0 = pub == 0
        purity0 = float(np.mean(m0[G[m0]].mean(axis=1)))
        # per published cluster mean overlap
        per_cl_overlap = {int(c): float(overlap[pub == c].mean()) for c in sorted(set(pub.tolist()))}
        macro, per = knn_transfer(X[keep], pub[keep], groups[keep])
        results[v] = dict(knn15_overlap_with_full_mean=float(overlap.mean()),
                          knn15_overlap_with_full_median=float(np.median(overlap)),
                          knn15_overlap_by_published_cluster=per_cl_overlap,
                          cluster0_knn15_purity=purity0,
                          label_transfer_macro_f1=macro, label_transfer_per_cluster_f1=per,
                          wall_sec=time.time() - t0)
        print(v, json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in results[v].items()
                             if not isinstance(x, dict)}), "cl0 F1", round(per.get(0, float('nan')), 3),
              "cl3 F1", round(per.get(3, float('nan')), 3), flush=True)
        (OUT / "a_variant_geometry.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
