#!/usr/bin/env python3
"""Task M, Step 4 -- the three replications on MIMIC-IV embeddings.

Thread pinning MUST happen before numpy/sklearn/umap/hdbscan/numba are
imported (this repo's own established gotcha: clustering is non-
deterministic across thread counts even with a fixed seed -- 8 vs 24
threads gave different cluster counts for the identical config/seed in the
Korean cohort's Task 2). Every stage in this module that clusters therefore
pins threads to 1 at the top of `main()`, before any of those libraries are
imported.

Clustering config ("code-of-record", identical to `17_common.py`'s anchor
config used throughout this project's Korean-cohort work, and to the task
brief's own specification): UMAP(n_components=2, n_neighbors=15,
min_dist=0.1, metric='euclidean', random_state=seed) + HDBSCAN(
min_cluster_size=max(10, n // 200), min_samples=None,
cluster_selection_method='eom'). Reuses `02_recluster.run_config` via
importlib (mirrors this project's own convention for reusing that function
across corpora, e.g. `04_open_embedding.py`'s stage_floors_seed).

Stages:
  --stage cluster --encoder {bge-m3,qwen3-0.6b} --seed N
      Fit once, write labels_<encoder>_seed<N>.parquet.
  --stage cross_encoder
      ARI/AMI between bge-m3 and qwen3-0.6b's seed-42 labels, both all-row
      and noise-excluded.
  --stage cross_seed --encoder {bge-m3,qwen3-0.6b}
      Requires seeds 1-10 already clustered (this stage does not cluster;
      run --stage cluster in a loop or array first). Reports k per seed and
      the pairwise ARI distribution (median [min-max]), all-row and
      noise-excluded.
  --stage token_recoverability
      Requires both encoders clustered at seed 42. For EACH encoder's
      cluster labels (as the target), per cluster: macro-F1 of a
      categorical lookup (organism, care_unit, admit_period, n_bsi_episodes,
      progressive combinations) vs. a kNN(k=25, cosine) label-transfer
      classifier trained on the OTHER encoder's embeddings -- identical
      method to `17d_token_lookup.py` / `18d_s6_knn.py` in the Korean arm,
      under StratifiedGroupKFold(5, groups=subject_id).

Outcome-blind (Global Constraints): no mortality/discharge/death/LOS column
is read anywhere in this module -- only organism/care_unit/admit_period/
n_bsi_episodes/subject_id/embeddings are used.
"""
import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REV = SCRIPT_DIR.parent
COHORT_DIR = REV / "out/27_mimic/cohort"
EMBED_DIR = REV / "out/27_mimic/embed"
OUT = REV / "out/27_mimic/replicate"

ENCODERS = ["bge-m3", "qwen3-0.6b"]
KNN_K = 25
CV_SPLITS = 5
CV_SEED = 42


def _atomic_write_json(obj, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, dest)


def _atomic_write_parquet(df, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)


def _load_embeddings(encoder: str):
    import json as _json

    import numpy as np

    vecs = np.load(EMBED_DIR / encoder / "merged.npy")
    ids = _json.loads((EMBED_DIR / encoder / "merged_ids.json").read_text())
    return vecs, ids


def _min_cluster_size_for(n: int) -> int:
    return max(10, n // 200)


def stage_cluster(encoder: str, seed: int) -> None:
    import importlib

    import numpy as np
    import pandas as pd

    sys.path.insert(0, str(SCRIPT_DIR))
    recl = importlib.import_module("02_recluster")

    vecs, ids = _load_embeddings(encoder)
    mcs = _min_cluster_size_for(len(vecs))
    r = recl.run_config(
        vecs, n_components=2, n_neighbors=15, min_cluster_size=mcs, seed=seed,
        min_dist=0.1, metric="euclidean", min_samples=None, cluster_selection_method="eom",
    )
    df = pd.DataFrame({"admission_ID": ids, "cluster": r["labels"]})
    out_dir = OUT / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_parquet(df, out_dir / f"{encoder}_seed{seed}.parquet")
    result = {
        "encoder": encoder, "seed": seed, "n": len(vecs), "min_cluster_size": mcs,
        "n_clusters": r["n_clusters"], "noise_frac": r["noise_frac"], "dbcv": r["dbcv"],
    }
    _atomic_write_json(result, out_dir / f".summary_{encoder}_seed{seed}.json")
    print(json.dumps(result, indent=2), flush=True)


def _load_labels(encoder: str, seed: int):
    import pandas as pd
    return pd.read_parquet(OUT / "labels" / f"{encoder}_seed{seed}.parquet")


def _ari_ami(y1, y2):
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    return float(adjusted_rand_score(y1, y2)), float(adjusted_mutual_info_score(y1, y2))


def stage_cross_encoder() -> None:
    import numpy as np

    a = _load_labels("bge-m3", 42).set_index("admission_ID")
    b = _load_labels("qwen3-0.6b", 42).set_index("admission_ID")
    common = a.index.intersection(b.index)
    assert len(common) == len(a) == len(b), (
        f"admission_ID sets differ between encoders: bge-m3={len(a)}, qwen3={len(b)}, common={len(common)}"
    )
    ya = a.loc[common, "cluster"].to_numpy()
    yb = b.loc[common, "cluster"].to_numpy()

    ari_all, ami_all = _ari_ami(ya, yb)
    keep = (ya != -1) & (yb != -1)
    ari_excl, ami_excl = _ari_ami(ya[keep], yb[keep])

    result = {
        "n_rows": int(len(common)),
        "n_clusters_bge_m3": int(len(set(ya.tolist()) - {-1})),
        "n_clusters_qwen3": int(len(set(yb.tolist()) - {-1})),
        "noise_frac_bge_m3": float((ya == -1).mean()),
        "noise_frac_qwen3": float((yb == -1).mean()),
        "ari_all_rows": ari_all, "ami_all_rows": ami_all,
        "ari_noise_excluded": ari_excl, "ami_noise_excluded": ami_excl,
        "n_noise_excluded_rows": int(keep.sum()),
    }
    _atomic_write_json(result, OUT / "cross_encoder.json")
    print(json.dumps(result, indent=2), flush=True)


def stage_cross_seed(encoder: str, seeds=range(1, 11)) -> None:
    import itertools

    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    seeds = list(seeds)
    all_labels, ks, noise_fracs = {}, {}, {}
    for s in seeds:
        df = _load_labels(encoder, s)
        y = df.sort_values("admission_ID")["cluster"].to_numpy()
        all_labels[s] = y
        ks[s] = int(len(set(y.tolist()) - {-1}))
        noise_fracs[s] = float((y == -1).mean())

    pairs = list(itertools.combinations(seeds, 2))
    ari_all = np.array([adjusted_rand_score(all_labels[a], all_labels[b]) for a, b in pairs])

    ari_excl = []
    for a, b in pairs:
        ya, yb = all_labels[a], all_labels[b]
        keep = (ya != -1) & (yb != -1)
        ari_excl.append(adjusted_rand_score(ya[keep], yb[keep]))
    ari_excl = np.array(ari_excl)

    result = {
        "encoder": encoder, "seeds": seeds, "k_per_seed": ks, "noise_frac_per_seed": noise_fracs,
        "ari_all_rows": {
            "median": float(np.median(ari_all)), "min": float(ari_all.min()), "max": float(ari_all.max()),
        },
        "ari_noise_excluded": {
            "median": float(np.median(ari_excl)), "min": float(ari_excl.min()), "max": float(ari_excl.max()),
        },
        "n_pairs": len(pairs),
    }
    _atomic_write_json(result, OUT / f"cross_seed_{encoder}.json")
    print(json.dumps(result, indent=2), flush=True)


def _load_metadata_for_recoverability():
    import pandas as pd

    nm = pd.read_parquet(COHORT_DIR / "narrative_manifest.parquet")
    return nm[["admission_ID", "subject_id", "organism_category", "care_unit", "admit_period", "n_bsi_episodes"]]


def _lookup_eval(key, y, folds):
    import numpy as np
    import pandas as pd

    yt, yp = [], []
    for tr, te in folds:
        tab = pd.crosstab(pd.Series(key[tr]), pd.Series(y[tr]))
        maj = tab.idxmax(axis=1)
        glob = pd.Series(y[tr]).value_counts().idxmax()
        pred = pd.Series(key[te]).map(maj).fillna(glob)
        yp.append(pred.to_numpy())
        yt.append(y[te])
    return np.concatenate(yt), np.concatenate(yp)


def _macro_f1_and_per_cluster(yt, yp, labels_sorted):
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labels_sorted, zero_division=0)
    return {
        "macro_f1": float(f1_score(yt, yp, average="macro")),
        "per_cluster_f1": {int(c): float(ff) for c, ff in zip(labels_sorted, f)},
    }


def _knn_eval(X, y, folds):
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier

    yt, yp = [], []
    for tr, te in folds:
        clf = KNeighborsClassifier(n_neighbors=KNN_K, metric="cosine", n_jobs=1)
        clf.fit(X[tr], y[tr])
        yp.append(clf.predict(X[te]))
        yt.append(y[te])
    return np.concatenate(yt), np.concatenate(yp)


def _run_recoverability_for_target(target_encoder: str, source_encoder: str, meta) -> dict:
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    labels = _load_labels(target_encoder, 42).merge(meta, on="admission_ID", how="inner")
    labels = labels[labels["cluster"] != -1].reset_index(drop=True)

    y = labels["cluster"].to_numpy()
    groups = labels["subject_id"].to_numpy()
    org = labels["organism_category"].astype(str).to_numpy()
    unit = labels["care_unit"].astype(str).to_numpy()
    period = labels["admit_period"].astype(str).to_numpy()
    nep = labels["n_bsi_episodes"].astype(str).to_numpy()

    skf = StratifiedGroupKFold(n_splits=CV_SPLITS, shuffle=True, random_state=CV_SEED)
    folds = list(skf.split(np.zeros(len(y)), y, groups))
    labels_sorted = sorted(set(y.tolist()))

    combos = {
        "organism": org,
        "organism_unit_period": np.array([f"{a}|{b}|{c}" for a, b, c in zip(org, unit, period)]),
        "organism_unit_period_episodes": np.array(
            [f"{a}|{b}|{c}|{d}" for a, b, c, d in zip(org, unit, period, nep)]
        ),
    }
    lookup_res = {name: _macro_f1_and_per_cluster(*_lookup_eval(key, y, folds), labels_sorted)
                  for name, key in combos.items()}

    src_vecs, src_ids = _load_embeddings(source_encoder)
    src_df = {aid: i for i, aid in enumerate(src_ids)}
    idx_in_src = labels["admission_ID"].map(src_df)
    assert idx_in_src.notna().all(), "some admission_IDs missing from source encoder embeddings"
    X = src_vecs[idx_in_src.to_numpy().astype(int)]

    knn_res = _macro_f1_and_per_cluster(*_knn_eval(X, y, folds), labels_sorted)

    best_lookup = {c: max(lookup_res[name]["per_cluster_f1"][c] for name in combos) for c in labels_sorted}
    best_lookup_combo = {
        c: max(combos.keys(), key=lambda name: lookup_res[name]["per_cluster_f1"][c])
        for c in labels_sorted
    }
    margin = {c: knn_res["per_cluster_f1"][c] - best_lookup[c] for c in labels_sorted}
    cluster_size = {int(c): int((y == c).sum()) for c in labels_sorted}

    return {
        "target_encoder": target_encoder, "source_encoder_for_knn": source_encoder,
        "n_rows_noise_excluded": int(len(y)), "n_clusters": len(labels_sorted),
        "lookup_macro_f1": {name: r["macro_f1"] for name, r in lookup_res.items()},
        "knn_macro_f1": knn_res["macro_f1"],
        "per_cluster": {
            int(c): {
                "size": cluster_size[c], "best_lookup_f1": best_lookup[c],
                "best_lookup_combo": best_lookup_combo[c], "knn_f1": knn_res["per_cluster_f1"][c],
                "margin_knn_minus_lookup": margin[c],
            }
            for c in labels_sorted
        },
    }


def stage_token_recoverability() -> None:
    meta = _load_metadata_for_recoverability()
    results = {}
    for target, source in [("bge-m3", "qwen3-0.6b"), ("qwen3-0.6b", "bge-m3")]:
        res = _run_recoverability_for_target(target, source, meta)
        results[target] = res
        n_robust = sum(1 for v in res["per_cluster"].values() if v["margin_knn_minus_lookup"] <= 0.10)
        n_embed_only = sum(1 for v in res["per_cluster"].values() if v["margin_knn_minus_lookup"] > 0.10)
        print(f"[27e/token_recoverability] target={target}: n_clusters={res['n_clusters']} "
              f"lookup>=knn-0.10 (token-explained)={n_robust} embedding-only(margin>0.10)={n_embed_only}", flush=True)
    _atomic_write_json(results, OUT / "token_recoverability.json")
    print(f"[27e/token_recoverability] wrote {OUT / 'token_recoverability.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                     choices=["cluster", "cross_encoder", "cross_seed", "token_recoverability"])
    ap.add_argument("--encoder", choices=ENCODERS)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Thread pinning BEFORE any numpy/sklearn/umap/hdbscan/numba import --
    # applies to every stage in this module that clusters (cluster,
    # cross_encoder/cross_seed/token_recoverability only import sklearn
    # metrics, which are not thread-count-sensitive, but pinning
    # unconditionally here is simpler and strictly safe).
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    OUT.mkdir(parents=True, exist_ok=True)

    if args.stage == "cluster":
        assert args.encoder, "--encoder required for --stage cluster"
        stage_cluster(args.encoder, args.seed)
    elif args.stage == "cross_encoder":
        stage_cross_encoder()
    elif args.stage == "cross_seed":
        assert args.encoder, "--encoder required for --stage cross_seed"
        stage_cross_seed(args.encoder)
    elif args.stage == "token_recoverability":
        stage_token_recoverability()


if __name__ == "__main__":
    main()
