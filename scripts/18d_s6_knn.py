#!/usr/bin/env python3
"""Task 18, Step 2 -- S6 recoverability metric: kNN (k=25, cosine,
StratifiedGroupKFold(5, groups=patient_ID), seed 42) predicting the
published 16 labels from each encoder's embedding, noise excluded. Method
is identical to `out/12_audit/second_audit/g4_perclass.py` (read and
mirrored, not imported -- that script is wired to the wrong corpus for
this task) and to `17d_token_lookup.py`'s fold construction, so numbers are
directly comparable to `out/17_correct_env/survival_matrix.csv`'s
`best_lookup_f1` column.

`--encoder` in {bge-m3, qwen3-0.6b, openai-3small, openai-3large}. The
first three read `out/18_encoder_dec/embeddings/<slug>/merged.npy` (Step 1
output of this task); `openai-3large` reads the STORED `out_pre/
case_report_embeddings.npy` directly (the ceiling -- same model that
produced the published clusters, no new embedding needed).

All four runs use the IDENTICAL StratifiedGroupKFold split (same y/groups
construction order, same seed) so results are directly comparable fold-
for-fold.

Outcome-blind (Global Constraints): only phenotype_cluster + patient_ID are
read from the reference files; no mortality/discharge/death column.
"""
import importlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c18 = importlib.import_module("18_common")

OUT_DIR = c18.OUT / "s6_knn"

ENCODER_EMBED_PATHS = {
    "bge-m3": c18.OUT / "embeddings" / "bge-m3" / "merged.npy",
    "qwen3-0.6b": c18.OUT / "embeddings" / "qwen3-0.6b" / "merged.npy",
    "openai-3small": c18.OUT / "embeddings" / "openai-3small" / "merged.npy",
}
ENCODER_IDS_PATHS = {
    "bge-m3": c18.OUT / "embeddings" / "bge-m3" / "merged_ids.json",
    "qwen3-0.6b": c18.OUT / "embeddings" / "qwen3-0.6b" / "merged_ids.json",
    "openai-3small": c18.OUT / "embeddings" / "openai-3small" / "merged_ids.json",
}


def _load_encoder_vecs(encoder: str, target_admission_ids):
    import numpy as np

    if encoder == "openai-3large":
        meta = c18.load_metadata()
        stored = np.load(c18.OUT_PRE_EMB)
        return c18.align_embeddings_to_admission_ids(
            stored, meta["admission_ID"].to_numpy(), target_admission_ids,
        )
    vec_path = ENCODER_EMBED_PATHS[encoder]
    id_path = ENCODER_IDS_PATHS[encoder]
    if not vec_path.exists():
        raise FileNotFoundError(f"{vec_path} missing -- run Step 1 embedding for {encoder!r} first")
    vecs = np.load(vec_path)
    ids = json.loads(id_path.read_text())
    return c18.align_embeddings_to_admission_ids(vecs, ids, target_admission_ids)


def main() -> None:
    import argparse
    import numpy as np
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.neighbors import KNeighborsClassifier

    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True,
                     choices=["bge-m3", "qwen3-0.6b", "openai-3small", "openai-3large"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{args.encoder}.json"

    frame = c18.load_published_frame()
    pub = frame[c18.PUBLISHED_CLUSTER_COL].to_numpy()
    keep = pub != -1
    y = pub[keep]
    groups = frame["patient_ID"].to_numpy()[keep]
    admission_ids_kept = frame["admission_ID"].to_numpy()[keep]
    print(f"[18d/{args.encoder}] n_rows_total={len(frame)} n_kept(noise-excl)={keep.sum()} "
          f"n_clusters={len(set(y.tolist()))}", flush=True)

    X = _load_encoder_vecs(args.encoder, admission_ids_kept)
    print(f"[18d/{args.encoder}] embedding aligned: shape={X.shape}", flush=True)

    skf = StratifiedGroupKFold(n_splits=c18.CV_SPLITS, shuffle=True, random_state=c18.CV_SEED)
    folds = list(skf.split(np.zeros(len(y)), y, groups))
    labels_sorted = sorted(set(y.tolist()))

    yt, yp = [], []
    for i, (tr, te) in enumerate(folds):
        # n_jobs=1 deliberately (matches g4_perclass.py's precedent exactly):
        # BLAS-level parallelism already comes from OMP/MKL/OPENBLAS_NUM_THREADS
        # (set by the SLURM wrapper from SLURM_CPUS_PER_TASK); n_jobs>1 here
        # would additionally fork joblib workers on top of that -- on a
        # shared node joblib's default worker count is the NODE's full core
        # count, not the cgroup allocation, so n_jobs=-1 risks severe
        # oversubscription (this repo's documented detectCores()-style gotcha).
        clf = KNeighborsClassifier(n_neighbors=c18.KNN_K, metric="cosine", n_jobs=1)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        yt.append(y[te])
        yp.append(pred)
        print(f"[18d/{args.encoder}] fold {i + 1}/{c18.CV_SPLITS} done "
              f"(train={len(tr)}, test={len(te)})", flush=True)
    yt = np.concatenate(yt)
    yp = np.concatenate(yp)

    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labels_sorted, zero_division=0)
    result = {
        "encoder": args.encoder,
        "n_rows": int(len(y)),
        "n_classes": int(len(labels_sorted)),
        "k": c18.KNN_K,
        "n_splits": c18.CV_SPLITS,
        "seed": c18.CV_SEED,
        "macro_f1": float(f1_score(yt, yp, average="macro")),
        "per_cluster_f1": {int(c): float(ff) for c, ff in zip(labels_sorted, f)},
        "per_cluster_precision": {int(c): float(pp) for c, pp in zip(labels_sorted, p)},
        "per_cluster_recall": {int(c): float(rr) for c, rr in zip(labels_sorted, r)},
        "per_cluster_support": {int(c): int(ss) for c, ss in zip(labels_sorted, s)},
    }
    c18.atomic_write_json(result, out_path)
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("per_cluster")}, indent=2), flush=True)
    print(json.dumps(result["per_cluster_f1"], indent=2), flush=True)
    print(f"[18d/{args.encoder}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
