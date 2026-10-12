#!/usr/bin/env python3
"""Audit 6 (R2, R6) -- n-matched Korean draws for ALL FOUR Korean encoders,
with labels saved, so that (a) the fig1 Korean cross-seed point can be
recomputed as a pooled median over raw pairs instead of a median of draw
medians, and (b) the cross-encoder concordance axis can be evaluated on
non-degenerate encoder pairs (Qwen3 vs 3-small, Qwen3 vs 3-large, 3-small vs
3-large) at the same n and min_cluster_size as MIMIC.

Mirrors scripts/30_structure_matched.py exactly (same rng per draw, same
subset, same UMAP/HDBSCAN configuration, same thread pinning, same env);
only the encoder list and the label persistence are new.

  --stage matched --draw D --seeds 1 2 3 4 5   -> out/39_audit6/matched/draw{D:02d}.npz
  --stage aggregate                             -> out/39_audit6/matched_encoders.json
"""
import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data/tcmhwd/AI/AST_ai")
REV = ROOT / "phenotype_revision"
KOREA = REV / "out/18_encoder_dec/embeddings"
OUT = REV / "out/39_audit6"
MIMIC_N = 16144
UMAP_NEIGHBORS = 15
ENCODERS = ("bge-m3", "qwen3-0.6b", "openai-3small", "openai-3large")


def load(encoder: str) -> np.ndarray:
    if encoder == "openai-3large":
        # row order verified identical to the 18_encoder_dec merged_ids order
        return np.load(ROOT / "pipeline_new/out_pre/case_report_embeddings.npy")
    return np.load(KOREA / encoder / "merged.npy")


def min_cluster_size(n: int) -> int:
    return max(10, n // 200)


def fit_partition(x: np.ndarray, seed: int) -> np.ndarray:
    import hdbscan
    import umap
    coords = umap.UMAP(n_components=2, n_neighbors=UMAP_NEIGHBORS, min_dist=0.1,
                       metric="euclidean", random_state=seed).fit_transform(x)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size(len(x)), min_samples=None,
                                metric="euclidean", cluster_selection_method="eom")
    return clusterer.fit_predict(coords).astype(np.int32)


def stage_matched(args):
    rng = np.random.default_rng(10_000 + args.draw)
    n_korea = len(np.load(KOREA / "bge-m3" / "merged.npy", mmap_mode="r"))
    subset = rng.choice(n_korea, size=MIMIC_N, replace=False)
    store = {"subset": subset}
    for enc in ENCODERS:
        x = load(enc)[subset]
        for seed in args.seeds:
            labels = fit_partition(x, seed)
            store[f"{enc}|{seed}"] = labels
            print(f"[audit6-matched] draw={args.draw} {enc} seed={seed} "
                  f"k={len(set(labels.tolist()) - {-1})} noise={(labels == -1).mean():.4f}", flush=True)
    (OUT / "matched").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "matched" / f"draw{args.draw:02d}.npz", **store)


def agree(a, b):
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    both = (a != -1) & (b != -1)
    return (float(adjusted_rand_score(a[both], b[both])),
            float(adjusted_mutual_info_score(a[both], b[both])))


def summarize(v):
    v = np.asarray(v, float)
    return {"n": int(len(v)), "median": float(np.median(v)), "min": float(v.min()),
            "max": float(v.max()), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75))}


def stage_aggregate(args):
    import pandas as pd
    files = sorted((OUT / "matched").glob("draw*.npz"))
    seeds = None
    cross_seed = {e: {"ari": [], "ami": []} for e in ENCODERS}
    cross_seed_draw_medians = {e: [] for e in ENCODERS}
    ks = {e: [] for e in ENCODERS}
    largest = {e: [] for e in ENCODERS}
    pairs = {f"{a} vs {b}": {"ari": [], "ami": [], "ari_nondegenerate": [], "ami_nondegenerate": []}
             for a, b in itertools.combinations(ENCODERS, 2)}
    # token alignment on the subsample
    ds = pd.read_csv(ROOT / "manuscript/phenotype_ml_dataset.csv", low_memory=False,
                     usecols=["admission_ID", "BSI_episode_short_clean", "hospital_code", "test_date",
                              "treatment_department_eng", "n_bsi_episodes"]).drop_duplicates("admission_ID")
    ids = json.load(open(KOREA / "bge-m3" / "merged_ids.json"))
    ds = ds.set_index("admission_ID").loc[ids].reset_index()
    ds["admit_year"] = pd.to_datetime(ds["test_date"], errors="coerce").dt.year.astype(str)
    tokens = ["BSI_episode_short_clean", "hospital_code", "admit_year", "treatment_department_eng", "n_bsi_episodes"]
    token_ami = {e: {t: [] for t in tokens} for e in ENCODERS}
    from sklearn.metrics import adjusted_mutual_info_score
    for f in files:
        z = np.load(f)
        subset = z["subset"]
        keys = [k for k in z.files if "|" in k]
        seeds = sorted({int(k.split("|")[1]) for k in keys})
        for e in ENCODERS:
            vals_ari, vals_ami = [], []
            for a, b in itertools.combinations(seeds, 2):
                r, m = agree(z[f"{e}|{a}"], z[f"{e}|{b}"])
                vals_ari.append(r); vals_ami.append(m)
            cross_seed[e]["ari"] += vals_ari; cross_seed[e]["ami"] += vals_ami
            cross_seed_draw_medians[e].append(float(np.median(vals_ari)))
            for s in seeds:
                lab = z[f"{e}|{s}"]
                k = len(set(lab.tolist()) - {-1}); ks[e].append(k)
                nn = lab[lab != -1]
                largest[e].append(float(pd.Series(nn).value_counts().iloc[0] / len(lab)) if len(nn) else 1.0)
                sub = ds.iloc[subset]
                keep = lab != -1
                for t in tokens:
                    token_ami[e][t].append(float(adjusted_mutual_info_score(
                        sub[t].astype(str).to_numpy()[keep], lab[keep])))
        for a, b in itertools.combinations(ENCODERS, 2):
            for s in seeds:
                la, lb = z[f"{a}|{s}"], z[f"{b}|{s}"]
                r, m = agree(la, lb)
                pairs[f"{a} vs {b}"]["ari"].append(r); pairs[f"{a} vs {b}"]["ami"].append(m)
                ka = len(set(la.tolist()) - {-1}); kb = len(set(lb.tolist()) - {-1})
                if ka >= 5 and kb >= 5:
                    pairs[f"{a} vs {b}"]["ari_nondegenerate"].append(r)
                    pairs[f"{a} vs {b}"]["ami_nondegenerate"].append(m)
    out = {
        "n_draws": len(files), "seeds": seeds, "n": MIMIC_N, "min_cluster_size": min_cluster_size(MIMIC_N),
        "cross_seed_raw_pairs": {e: {"ari": summarize(v["ari"]), "ami": summarize(v["ami"]),
                                     "median_of_draw_medians_ari": float(np.median(cross_seed_draw_medians[e])),
                                     "draw_medians_ari": cross_seed_draw_medians[e]}
                                 for e, v in cross_seed.items()},
        "k_per_fit": {e: {"values": v, "median": float(np.median(v)), "frac_k_le_4": float(np.mean(np.array(v) <= 4))}
                      for e, v in ks.items()},
        "largest_cluster_share": {e: summarize(v) for e, v in largest.items()},
        "cross_encoder_seed_matched": {
            p: {"ari": summarize(v["ari"]), "ami": summarize(v["ami"]),
                "nondegenerate_both_k_ge_5": {
                    "n": len(v["ari_nondegenerate"]),
                    "ari": summarize(v["ari_nondegenerate"]) if v["ari_nondegenerate"] else None,
                    "ami": summarize(v["ami_nondegenerate"]) if v["ami_nondegenerate"] else None}}
            for p, v in pairs.items()},
        "token_alignment_ami": {e: {t: summarize(v) for t, v in d.items()} for e, d in token_ami.items()},
    }
    (OUT / "matched_encoders.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("cross_seed_raw_pairs", "cross_encoder_seed_matched")}, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["matched", "aggregate"], required=True)
    ap.add_argument("--draw", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stage_matched(args) if args.stage == "matched" else stage_aggregate(args)


if __name__ == "__main__":
    main()
