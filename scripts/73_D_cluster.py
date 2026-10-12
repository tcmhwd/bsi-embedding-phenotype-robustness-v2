#!/usr/bin/env python3
"""Analysis D (optional, exploratory): the Figure-1 matched cross-encoder design with
ONE factor changed -- UMAP n_components 2 -> 20. Everything else is audit6's.

Frozen design: out/followup_analyses/D_alt_pipeline_optional/PREDECLARED_PIPELINE.md
(sha256 in out/followup_analyses/qc/PREREG_HASHES_D.txt). Do not change any constant below
without a numbered addendum to that document.

  --stage anchor_stored                       -> anchor/d_anchor_stored.csv        (G0.a, no fitting)
  --stage anchor_refit --draw 1 --seeds 1..5  -> anchor/d_anchor_refit.csv (G0.b) + anchor/d_anchor_result.json
                                                 + 20 config=orig_2d rows in d_alt_fits.csv
  --stage alt --draw D --seeds 1 2 3 4 5      -> fits/alt_draw{D:02d}.npz (+ fit-meta sidecar json)
  --stage aggregate                           -> d_alt_*.csv, d_alt_summary.json

Exploratory, single alternative configuration, not confirmatory. No CI / SE / p-value is
computed anywhere; dispersion is always an observed range (min/max) or IQR over fits.
"""
import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"  # BEFORE any numeric import -- determinism, not speed

import argparse
import csv
import datetime
import fcntl
import hashlib
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data/tcmhwd/AI/AST_ai")
REV = ROOT / "phenotype_revision"
KOREA = REV / "out/18_encoder_dec/embeddings"
SRC = REV / "out/39_audit6"  # stored draws + labels
OUT = REV / "out/followup_analyses/D_alt_pipeline_optional"
FIG = REV / "out/33_figures"
DATASET = ROOT / "manuscript/phenotype_ml_dataset.csv"

MIMIC_N = 16144
ENCODERS = ("bge-m3", "qwen3-0.6b", "openai-3small", "openai-3large")
UMAP_NEIGHBORS = 15
MIN_DIST = 0.1
UMAP_METRIC = "euclidean"
ALT_NCOMP = 20  # <-- THE ONLY CHANGED FACTOR (PREDECLARED_PIPELINE.md 2.1)
ORIG_NCOMP = 2
EXPECTED_MCS = 80
ANCHOR_DRAW = 1
DRAWS = tuple(range(1, 11))
SEEDS = (1, 2, 3, 4, 5)
TOKENS = ["BSI_episode_short_clean", "hospital_code", "admit_year",
          "treatment_department_eng", "n_bsi_episodes"]

# Section 3.0 tolerances (fixed in the frozen document)
TOL_PASS = 1e-9
TOL_NOTE = 1e-6
# Section 3.1 tolerances
EXACT_ARI_TOL = 1e-12
DRIFT_ARI_MIN = 0.99
DRIFT_K = 1
DRIFT_NOISE_FRAC = 0.005
# Section 5 anchors (observed extremes of data already on disk)
MAX_2D_ARI = 0.07318
MAX_2D_AMI = 0.12380
MIMIC_MIN_ARI = 0.44684
MIMIC_AMI = 0.80332
MIN_2D_WITHIN_ARI = 0.46746

NONDEGENERATE_TRIO = ("qwen3-0.6b", "openai-3small", "openai-3large")
TRIO_PAIRS = (("qwen3-0.6b", "openai-3small"), ("qwen3-0.6b", "openai-3large"),
              ("openai-3small", "openai-3large"))

FITS_COLS = ["config", "encoder", "draw", "seed", "n", "n_components", "n_neighbors", "min_dist",
             "umap_metric", "random_state", "min_cluster_size", "min_samples", "hdbscan_metric",
             "cluster_selection_method", "k", "n_noise", "noise_frac", "largest_cluster_share",
             "k_le_4", "fit_seconds"]
PAIRS_COLS = ["config", "draw", "seed", "encoder_a", "encoder_b", "k_a", "k_b", "n_both_nonnoise",
              "ari_both_nonnoise", "ami_both_nonnoise", "nondegenerate_k_ge_5"]
CROSSSEED_COLS = ["config", "encoder", "draw", "seed_a", "seed_b", "k_a", "k_b", "n_both_nonnoise",
                  "ari_both_nonnoise", "ami_both_nonnoise"]
TOKEN_COLS = ["config", "encoder", "draw", "seed", "token", "ami", "n_nonnoise"]
STORED_COLS = ["quantity", "published", "recomputed", "abs_diff", "tolerance", "verdict"]
REFIT_COLS = ["encoder", "draw", "seed", "k_new", "k_stored", "n_noise_new", "n_noise_stored",
              "noise_frac_new", "noise_frac_stored", "n_both_nonnoise", "ari_vs_stored",
              "ami_vs_stored", "exact", "within_drift_band", "verdict"]

# Pre-written readings, verbatim from PREDECLARED_PIPELINE.md 5.1 / 5.3
ZONE_READING = {
    "zone1_configuration_specific": (
        "Under a 20-component UMAP, these two encoders agree at a level comparable to the MIMIC-IV "
        "cross-encoder comparison. The near-zero agreement reported in Figure 1a is, for this pair, "
        "specific to the 2-component UMAP/HDBSCAN configuration and is not a property of the "
        "representations alone."),
    "zone3_robust": (
        "Agreement for this pair remains within the range already observed under the 2-component "
        "configuration. Changing the UMAP dimension from 2 to 20 does not recover cross-encoder "
        "agreement; the disagreement is not specific to the 2-component reduction."),
    "zone2_indeterminate": (
        "Agreement moved but did not reach the level the manuscript calls substantial. A single "
        "alternative configuration cannot distinguish partial recovery from a dimension-dependent "
        "metric artefact. Reported as indeterminate and used in neither direction."),
}
BGE_READING = {
    "attenuated": (
        "BGE-M3's degeneracy at the matched n is dependent on the 2-component reduction and is not "
        "solely a property of the BGE-M3 representation."),
    "persists": "BGE-M3 remains degenerate at the matched n under both configurations tested.",
}
LABEL = "exploratory, single alternative configuration, not confirmatory"


# ---------------------------------------------------------------- generic helpers
def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_versions():
    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for name in ("numpy", "umap-learn", "hdbscan", "scikit-learn", "scipy", "numba", "pandas"):
        try:
            out[name] = metadata.version(name)
        except Exception as exc:  # noqa: BLE001
            out[name] = f"unavailable ({type(exc).__name__})"
    return out


def git_head():
    try:
        return subprocess.run(["git", "-C", str(REV), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=30, check=True).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable ({type(exc).__name__})"


def locked_json_update(path, mutate):
    """Read-modify-write a shared JSON under an flock (concurrent stages/tasks write the same file)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = OUT / ".d_state.lock"
    with open(lock_path, "w") as lf:
        try:
            fcntl.flock(lf, fcntl.LOCK_EX)
        except OSError as exc:
            print(f"[D] WARNING: flock unavailable ({exc}); proceeding without lock", flush=True)
        data = json.loads(path.read_text()) if path.exists() else {}
        mutate(data)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        os.replace(tmp, path)


def write_stage_metadata(stage_key, t0, inputs, extra=None):
    meta = {
        "label": LABEL,
        "sys.executable": sys.executable,
        "versions": env_versions(),
        "git_head": git_head(),
        "script_sha256": sha256_file(Path(__file__)),
        "thread_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "n": MIMIC_N, "draws": list(DRAWS), "seeds": list(SEEDS),
        "settings_declared": {
            "n_components_alt": ALT_NCOMP, "n_components_orig": ORIG_NCOMP,
            "umap_n_neighbors": UMAP_NEIGHBORS, "umap_min_dist": MIN_DIST, "umap_metric": UMAP_METRIC,
            "hdbscan_min_cluster_size": min_cluster_size(MIMIC_N), "hdbscan_min_samples": None,
            "hdbscan_metric": "euclidean", "hdbscan_cluster_selection_method": "eom"},
        "input_sha256": {str(p): sha256_file(p) for p in inputs},
        "finished": now_iso(),
        "wall_seconds": round(time.perf_counter() - t0, 2),
    }
    if extra:
        meta.update(extra)

    def mutate(d):
        d.setdefault("run_metadata", {})[stage_key] = meta
    locked_json_update(OUT / "d_alt_summary.json", mutate)
    return meta


def write_csv(path, cols, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})


# ---------------------------------------------------------------- pipeline pieces
def load(encoder):  # verbatim from audit6_matched_encoders.py:37-41
    if encoder == "openai-3large":
        return np.load(ROOT / "pipeline_new/out_pre/case_report_embeddings.npy")
    return np.load(KOREA / encoder / "merged.npy")


def input_paths_for(encoders):
    return [ROOT / "pipeline_new/out_pre/case_report_embeddings.npy" if e == "openai-3large"
            else KOREA / e / "merged.npy" for e in encoders]


def min_cluster_size(n):  # verbatim from audit6:44-45 -> 80 at n=16,144
    return max(10, n // 200)


def fit_partition(x, seed, n_components):
    """audit6 fit_partition with n_components a parameter. Returns (labels, realised settings)."""
    import hdbscan
    import umap
    reducer = umap.UMAP(n_components=n_components, n_neighbors=UMAP_NEIGHBORS,
                        min_dist=MIN_DIST, metric=UMAP_METRIC, random_state=seed)
    coords = reducer.fit_transform(x)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size(len(x)), min_samples=None,
                                metric="euclidean", cluster_selection_method="eom")
    labels = clusterer.fit_predict(coords).astype(np.int32)
    realised = {
        "n": int(len(x)), "n_components": int(reducer.n_components), "n_neighbors": int(reducer.n_neighbors),
        "min_dist": float(reducer.min_dist), "umap_metric": str(reducer.metric),
        "random_state": reducer.random_state, "min_cluster_size": int(clusterer.min_cluster_size),
        "min_samples": clusterer.min_samples, "hdbscan_metric": str(clusterer.metric),
        "cluster_selection_method": str(clusterer.cluster_selection_method),
    }
    return labels, realised


def agree(a, b):  # verbatim from audit6:74-78 -- both-non-noise ONLY
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    both = (a != -1) & (b != -1)
    if both.sum() <= 1:
        return float("nan"), float("nan"), int(both.sum())
    return (float(adjusted_rand_score(a[both], b[both])),
            float(adjusted_mutual_info_score(a[both], b[both])), int(both.sum()))


def stored_draw(draw):  # NEVER regenerate the subset -- read it
    return np.load(SRC / "matched" / f"draw{draw:02d}.npz")


def summarize(v):  # audit6:81-83
    v = np.asarray(v, float)
    return {"n": int(len(v)), "median": float(np.median(v)), "min": float(v.min()),
            "max": float(v.max()), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75))}


def n_clusters(lab):
    return int(len(set(lab.tolist()) - {-1}))


def largest_share(lab):  # audit6:123 -- largest non-noise cluster / n; 1.0 if everything is noise
    nn = lab[lab != -1]
    if len(nn) == 0:
        return 1.0
    return float(np.unique(nn, return_counts=True)[1].max() / len(lab))


def fit_stats(lab):
    n_noise = int((lab == -1).sum())
    k = n_clusters(lab)
    return {"k": k, "n_noise": n_noise, "noise_frac": n_noise / len(lab),
            "largest_cluster_share": largest_share(lab), "k_le_4": bool(k <= 4)}


def read_labels(z):
    """z: npz. Returns subset and {(enc, seed): labels}; asserts the 20-key layout."""
    keys = [k for k in z.files if "|" in k]
    seeds = sorted({int(k.split("|")[1]) for k in keys})
    assert seeds == list(SEEDS), f"unexpected seeds {seeds}"
    labs = {(e, s): z[f"{e}|{s}"] for e in ENCODERS for s in seeds}
    assert len(keys) == len(labs) == len(ENCODERS) * len(SEEDS), f"unexpected keys {sorted(keys)}"
    subset = z["subset"]
    assert len(subset) == MIMIC_N
    for lab in labs.values():
        assert len(lab) == MIMIC_N
    return subset, labs


def compute_tables(draw_labels, tokens_df=None):
    """draw_labels: {draw: (subset, {(enc, seed): labels})}. Mirrors the audit6 aggregate loops."""
    fits, cross_seed, pairs, token_rows = [], [], [], []
    for draw in sorted(draw_labels):
        subset, labs = draw_labels[draw]
        for e in ENCODERS:
            for a, b in itertools.combinations(SEEDS, 2):
                r, m, nb = agree(labs[(e, a)], labs[(e, b)])
                cross_seed.append({"encoder": e, "draw": draw, "seed_a": a, "seed_b": b,
                                   "k_a": n_clusters(labs[(e, a)]), "k_b": n_clusters(labs[(e, b)]),
                                   "n_both_nonnoise": nb, "ari_both_nonnoise": r, "ami_both_nonnoise": m})
            for s in SEEDS:
                lab = labs[(e, s)]
                fits.append({"encoder": e, "draw": draw, "seed": s, **fit_stats(lab)})
                if tokens_df is not None:
                    sub = tokens_df.iloc[subset]
                    keep = lab != -1
                    from sklearn.metrics import adjusted_mutual_info_score
                    for t in TOKENS:
                        token_rows.append({"encoder": e, "draw": draw, "seed": s, "token": t,
                                           "ami": float(adjusted_mutual_info_score(
                                               sub[t].astype(str).to_numpy()[keep], lab[keep])),
                                           "n_nonnoise": int(keep.sum())})
        for a, b in itertools.combinations(ENCODERS, 2):
            for s in SEEDS:
                la, lb = labs[(a, s)], labs[(b, s)]
                r, m, nb = agree(la, lb)
                ka, kb = n_clusters(la), n_clusters(lb)
                pairs.append({"draw": draw, "seed": s, "encoder_a": a, "encoder_b": b, "k_a": ka, "k_b": kb,
                              "n_both_nonnoise": nb, "ari_both_nonnoise": r, "ami_both_nonnoise": m,
                              "nondegenerate_k_ge_5": bool(ka >= 5 and kb >= 5)})
    return fits, cross_seed, pairs, token_rows


def aggregate_tables(fits, cross_seed, pairs):
    """audit6 matched_encoders.json layout (the parts G0.a and the D aggregate need)."""
    out = {"cross_seed_raw_pairs": {}, "k_per_fit": {}, "largest_cluster_share": {},
           "cross_encoder_seed_matched": {}}
    for e in ENCODERS:
        rows = [r for r in cross_seed if r["encoder"] == e]
        ari = [r["ari_both_nonnoise"] for r in rows]
        ami = [r["ami_both_nonnoise"] for r in rows]
        draw_med = [float(np.median([r["ari_both_nonnoise"] for r in rows if r["draw"] == d]))
                    for d in sorted({r["draw"] for r in rows})]
        out["cross_seed_raw_pairs"][e] = {"ari": summarize(ari), "ami": summarize(ami),
                                          "median_of_draw_medians_ari": float(np.median(draw_med)),
                                          "draw_medians_ari": draw_med}
        fe = [r for r in fits if r["encoder"] == e]
        ks = [r["k"] for r in fe]
        out["k_per_fit"][e] = {"values": ks, "median": float(np.median(ks)),
                               "frac_k_le_4": float(np.mean(np.array(ks) <= 4))}
        out["largest_cluster_share"][e] = summarize([r["largest_cluster_share"] for r in fe])
    for a, b in itertools.combinations(ENCODERS, 2):
        rows = [r for r in pairs if r["encoder_a"] == a and r["encoder_b"] == b]
        nd = [r for r in rows if r["nondegenerate_k_ge_5"]]
        out["cross_encoder_seed_matched"][f"{a} vs {b}"] = {
            "ari": summarize([r["ari_both_nonnoise"] for r in rows]),
            "ami": summarize([r["ami_both_nonnoise"] for r in rows]),
            "nondegenerate_both_k_ge_5": {
                "n": len(nd),
                "ari": summarize([r["ari_both_nonnoise"] for r in nd]) if nd else None,
                "ami": summarize([r["ami_both_nonnoise"] for r in nd]) if nd else None}}
    return out


def load_stored_draws():
    files = [SRC / "matched" / f"draw{d:02d}.npz" for d in DRAWS]
    missing = [str(f) for f in files if not f.exists()]
    assert not missing, f"missing stored draws: {missing}"
    return {d: read_labels(stored_draw(d)) for d in DRAWS}, files


# ---------------------------------------------------------------- anchor result (both anchors share one file)
def combine_anchor(d):
    g0a, g0b = d.get("G0a"), d.get("G0b")
    d["label"] = LABEL
    if g0a and g0a["verdict"] == "fail":
        d["verdict"] = "fail"
        d["reason"] = "G0.a metric-code anchor failed (a deviation > 1e-6): D halts (PREDECLARED_PIPELINE.md 3.0)"
    elif g0b and g0b["verdict"] == "fail":
        d["verdict"] = "fail"
        d["reason"] = "G0.b environment anchor failed: D halts (PREDECLARED_PIPELINE.md 3.1)"
    elif g0a and g0b:
        d["verdict"] = g0b["verdict"]  # exact | drift
        d["reason"] = ("G0.a " + g0a["verdict"] + "; G0.b " + g0b["verdict"] +
                       ("; every D conclusion must carry the label 'reproduction with numerical drift'"
                        if g0b["verdict"] == "drift" else ""))
        d["g0a_numerical_note"] = bool(g0a["verdict"] == "pass_numerical_note")
    else:
        d["verdict"] = None
        d["reason"] = "pending: " + ", ".join(k for k, v in (("G0a", g0a), ("G0b", g0b)) if not v)
    d["written"] = now_iso()


# ---------------------------------------------------------------- stage: anchor_stored (G0.a)
def tol_verdict(pub, rec):
    try:
        diff = abs(float(rec) - float(pub))
    except (TypeError, ValueError):
        return float("nan"), "fail"
    if not np.isfinite(diff):
        return diff, "fail"
    if diff <= TOL_PASS:
        return diff, "pass"
    if diff <= TOL_NOTE:
        return diff, "pass_numerical_note"
    return diff, "fail"


def stage_anchor_stored(args):
    import pandas as pd
    t0 = time.perf_counter()
    draw_labels, files = load_stored_draws()
    fits, cross_seed, pairs, _ = compute_tables(draw_labels)
    rec = aggregate_tables(fits, cross_seed, pairs)
    pub = json.load(open(SRC / "matched_encoders.json"))
    assert pub["seeds"] == list(SEEDS)
    agreement = pd.read_csv(FIG / "agreement.csv")
    degen = pd.read_csv(FIG / "encoder_degeneracy.csv")

    rows = []

    def add(q, p, r):
        d, v = tol_verdict(p, r)
        rows.append({"quantity": q, "published": repr(float(p)) if p is not None else "",
                     "recomputed": repr(float(r)) if r is not None else "",
                     "abs_diff": repr(float(d)), "tolerance": TOL_PASS, "verdict": v})

    def add_summary(prefix, p, r):
        for key in ("n", "median", "min", "max", "q25", "q75"):
            add(f"{prefix}/{key}", p[key], r[key])

    # -- configuration constants recorded in matched_encoders.json
    add("json:n", pub["n"], MIMIC_N)
    add("json:min_cluster_size", pub["min_cluster_size"], min_cluster_size(MIMIC_N))
    add("json:n_draws", pub["n_draws"], len(files))
    # -- matched_encoders.json: cross-seed, k, largest share, cross-encoder
    for e in ENCODERS:
        pj, rj = pub["cross_seed_raw_pairs"][e], rec["cross_seed_raw_pairs"][e]
        add_summary(f"json:cross_seed_raw_pairs/{e}/ari", pj["ari"], rj["ari"])
        add_summary(f"json:cross_seed_raw_pairs/{e}/ami", pj["ami"], rj["ami"])
        add(f"json:cross_seed_raw_pairs/{e}/median_of_draw_medians_ari",
            pj["median_of_draw_medians_ari"], rj["median_of_draw_medians_ari"])
        for i, (p, r) in enumerate(zip(pj["draw_medians_ari"], rj["draw_medians_ari"]), start=1):
            add(f"json:cross_seed_raw_pairs/{e}/draw_medians_ari[draw{i:02d}]", p, r)
        assert len(pj["draw_medians_ari"]) == len(rj["draw_medians_ari"])
        pk, rk = pub["k_per_fit"][e], rec["k_per_fit"][e]
        add(f"json:k_per_fit/{e}/median", pk["median"], rk["median"])
        add(f"json:k_per_fit/{e}/frac_k_le_4", pk["frac_k_le_4"], rk["frac_k_le_4"])
        assert len(pk["values"]) == len(rk["values"])
        for i, (p, r) in enumerate(zip(pk["values"], rk["values"])):
            add(f"json:k_per_fit/{e}/values[{i}]", p, r)
        add_summary(f"json:largest_cluster_share/{e}", pub["largest_cluster_share"][e],
                    rec["largest_cluster_share"][e])
    for pair, pj in pub["cross_encoder_seed_matched"].items():
        rj = rec["cross_encoder_seed_matched"][pair]
        add_summary(f"json:cross_encoder_seed_matched/{pair}/ari", pj["ari"], rj["ari"])
        add_summary(f"json:cross_encoder_seed_matched/{pair}/ami", pj["ami"], rj["ami"])
        add(f"json:cross_encoder_seed_matched/{pair}/nondegenerate_both_k_ge_5/n",
            pj["nondegenerate_both_k_ge_5"]["n"], rj["nondegenerate_both_k_ge_5"]["n"])
        for m in ("ari", "ami"):
            pn, rn = pj["nondegenerate_both_k_ge_5"][m], rj["nondegenerate_both_k_ge_5"][m]
            if pn is None or rn is None:
                assert pn is None and rn is None, f"nondegenerate subset presence differs for {pair}/{m}"
                continue
            add_summary(f"json:cross_encoder_seed_matched/{pair}/nondegenerate_both_k_ge_5/{m}", pn, rn)

    # -- agreement.csv, Korea (n-matched) panel only (the MIMIC rows are not recomputable here)
    korea = agreement[agreement["cohort"] == "Korea (n-matched)"]
    label_to_pair = {"Qwen3 vs 3-small": "qwen3-0.6b vs openai-3small",
                     "Qwen3 vs 3-large": "qwen3-0.6b vs openai-3large",
                     "3-small vs 3-large": "openai-3small vs openai-3large"}
    for _, row in korea.iterrows():
        if row["comparison"] == "Cross-seed":
            r = rec["cross_seed_raw_pairs"][row["encoder"]]
            tag = f"agreement.csv:Cross-seed/{row['encoder']}"
            add(f"{tag}/estimate(median_of_draw_medians)", row["estimate"], r["median_of_draw_medians_ari"])
            add(f"{tag}/lo(min)", row["lo"], r["ari"]["min"])
            add(f"{tag}/hi(max)", row["hi"], r["ari"]["max"])
            add(f"{tag}/ami(pooled median)", row["ami"], r["ami"]["median"])
            add(f"{tag}/n_values", row["n_values"], r["ari"]["n"])
        elif row["comparison"] == "Cross-encoder":
            r = rec["cross_encoder_seed_matched"][label_to_pair[row["label"]]]
            tag = f"agreement.csv:Cross-encoder/{row['label']}"
            add(f"{tag}/estimate(median)", row["estimate"], r["ari"]["median"])
            add(f"{tag}/lo(min)", row["lo"], r["ari"]["min"])
            add(f"{tag}/hi(max)", row["hi"], r["ari"]["max"])
            add(f"{tag}/ami(median)", row["ami"], r["ami"]["median"])
            add(f"{tag}/n_values", row["n_values"], r["ari"]["n"])
        else:
            raise SystemExit(f"unexpected comparison in agreement.csv: {row['comparison']}")
    # -- encoder_degeneracy.csv (only the columns recomputable from the stored draws)
    for _, row in degen.iterrows():
        e = row["encoder"]
        tag = f"encoder_degeneracy.csv:{e}"
        ks = np.array(rec["k_per_fit"][e]["values"])
        add(f"{tag}/k_median_refits", row["k_median_refits"], rec["k_per_fit"][e]["median"])
        add(f"{tag}/n_fits", row["n_fits"], len(ks))
        add(f"{tag}/frac_k_le_4", row["frac_k_le_4"], rec["k_per_fit"][e]["frac_k_le_4"])
        add(f"{tag}/n_k_le_4", row["n_k_le_4"], int((ks <= 4).sum()))
        add(f"{tag}/largest_share_median_refits", row["largest_share_median_refits"],
            rec["largest_cluster_share"][e]["median"])

    write_csv(OUT / "anchor" / "d_anchor_stored.csv", STORED_COLS, rows)
    verdicts = [r["verdict"] for r in rows]
    overall = "fail" if "fail" in verdicts else ("pass_numerical_note" if "pass_numerical_note" in verdicts else "pass")
    diffs = np.array([float(r["abs_diff"]) for r in rows])
    g0a = {"verdict": overall, "n_quantities": len(rows), "n_pass": verdicts.count("pass"),
           "n_pass_numerical_note": verdicts.count("pass_numerical_note"), "n_fail": verdicts.count("fail"),
           "max_abs_diff": float(np.nanmax(diffs)),
           "failed_quantities": [r["quantity"] for r in rows if r["verdict"] == "fail"][:50],
           "note_quantities": [r["quantity"] for r in rows if r["verdict"] == "pass_numerical_note"][:50],
           "csv": str(OUT / "anchor" / "d_anchor_stored.csv"), "finished": now_iso(),
           "not_recomputed": ["encoder_degeneracy.csv:k_wholecorpus / largest_share_wholecorpus / degenerate "
                              "(whole-corpus fits and a hard-coded literal, not derivable from the stored draws)",
                              "MIMIC rows of agreement.csv (MIMIC labels are not part of the stored Korean draws)",
                              "token_alignment_ami (not part of the section 3.0 list)"]}

    def mutate(d):
        d["G0a"] = g0a
        combine_anchor(d)
    locked_json_update(OUT / "anchor" / "d_anchor_result.json", mutate)
    write_stage_metadata("anchor_stored", t0, files + [SRC / "matched_encoders.json",
                                                       FIG / "agreement.csv", FIG / "encoder_degeneracy.csv"])
    print(f"[D anchor_stored] rows={len(rows)} verdict={overall} max_abs_diff={g0a['max_abs_diff']:.3e} "
          f"pass={g0a['n_pass']} note={g0a['n_pass_numerical_note']} fail={g0a['n_fail']}", flush=True)
    if overall == "fail":
        print("[D anchor_stored] FAILED quantities (first 50): " + "; ".join(g0a["failed_quantities"]), flush=True)
        sys.exit(2)


# ---------------------------------------------------------------- stage: anchor_refit (G0.b)
def stage_anchor_refit(args):
    t0 = time.perf_counter()
    if args.draw != ANCHOR_DRAW:
        raise SystemExit(f"anchor_refit is draw {ANCHOR_DRAW:02d} only (PREDECLARED_PIPELINE.md 3.1); got --draw {args.draw}")
    if list(args.seeds) != list(SEEDS):
        raise SystemExit(f"anchor_refit uses seeds {list(SEEDS)}; got {args.seeds}")
    z = stored_draw(ANCHOR_DRAW)
    subset, stored = read_labels(z)
    refit_rows, fit_rows = [], []
    for enc in ENCODERS:
        x = load(enc)[subset]
        assert x.shape[0] == MIMIC_N
        for seed in SEEDS:
            t1 = time.perf_counter()
            labels, real = fit_partition(x, seed, ORIG_NCOMP)
            sec = time.perf_counter() - t1
            assert real["min_cluster_size"] == EXPECTED_MCS, f"min_cluster_size {real['min_cluster_size']} != {EXPECTED_MCS}"
            assert real["n_components"] == ORIG_NCOMP and real["n"] == MIMIC_N
            old = stored[(enc, seed)]
            new_s, old_s = fit_stats(labels), fit_stats(old)
            ari, ami, nb = agree(labels, old)
            exact = bool(np.isfinite(ari) and abs(ari - 1.0) <= EXACT_ARI_TOL and new_s["k"] == old_s["k"]
                         and new_s["n_noise"] == old_s["n_noise"])
            within = bool(np.isfinite(ari) and ari >= DRIFT_ARI_MIN and abs(new_s["k"] - old_s["k"]) <= DRIFT_K
                          and abs(new_s["noise_frac"] - old_s["noise_frac"]) <= DRIFT_NOISE_FRAC)
            verdict = "exact" if exact else ("drift" if within else "fail")
            refit_rows.append({"encoder": enc, "draw": ANCHOR_DRAW, "seed": seed, "k_new": new_s["k"],
                               "k_stored": old_s["k"], "n_noise_new": new_s["n_noise"],
                               "n_noise_stored": old_s["n_noise"], "noise_frac_new": new_s["noise_frac"],
                               "noise_frac_stored": old_s["noise_frac"], "n_both_nonnoise": nb,
                               "ari_vs_stored": ari, "ami_vs_stored": ami, "exact": exact,
                               "within_drift_band": within, "verdict": verdict,
                               "_bit_identical": bool(np.array_equal(labels, old))})
            fit_rows.append({"config": "orig_2d", "encoder": enc, "draw": ANCHOR_DRAW, "seed": seed,
                             "n": real["n"], "n_components": real["n_components"],
                             "n_neighbors": real["n_neighbors"], "min_dist": real["min_dist"],
                             "umap_metric": real["umap_metric"], "random_state": real["random_state"],
                             "min_cluster_size": real["min_cluster_size"],
                             "min_samples": "" if real["min_samples"] is None else real["min_samples"],
                             "hdbscan_metric": real["hdbscan_metric"],
                             "cluster_selection_method": real["cluster_selection_method"],
                             "k": new_s["k"], "n_noise": new_s["n_noise"], "noise_frac": new_s["noise_frac"],
                             "largest_cluster_share": new_s["largest_cluster_share"], "k_le_4": new_s["k_le_4"],
                             "fit_seconds": round(sec, 3)})
            print(f"[D anchor_refit] {enc} seed={seed} k_new={new_s['k']} k_stored={old_s['k']} "
                  f"noise_new={new_s['n_noise']} noise_stored={old_s['n_noise']} ari={ari:.12f} "
                  f"bit_identical={refit_rows[-1]['_bit_identical']} verdict={verdict} sec={sec:.1f}", flush=True)

    write_csv(OUT / "anchor" / "d_anchor_refit.csv", REFIT_COLS, refit_rows)
    # d_alt_fits.csv: append the 20 orig_2d rows (idempotent: replace any earlier orig_2d rows, keep the rest)
    fits_path = OUT / "d_alt_fits.csv"
    keep = []
    if fits_path.exists():
        with open(fits_path, newline="") as f:
            keep = [r for r in csv.DictReader(f) if r["config"] != "orig_2d"]
    write_csv(fits_path, FITS_COLS, keep + fit_rows)

    verdicts = [r["verdict"] for r in refit_rows]
    overall = "fail" if "fail" in verdicts else ("drift" if "drift" in verdicts else "exact")
    aris = np.array([r["ari_vs_stored"] for r in refit_rows], float)
    g0b = {"verdict": overall, "n_fits": len(refit_rows), "n_exact": verdicts.count("exact"),
           "n_drift": verdicts.count("drift"), "n_fail": verdicts.count("fail"),
           "n_bit_identical_labels": int(sum(r["_bit_identical"] for r in refit_rows)),
           "min_ari_vs_stored": float(np.nanmin(aris)),
           "max_abs_dk": int(max(abs(r["k_new"] - r["k_stored"]) for r in refit_rows)),
           "max_abs_d_noise_frac": float(max(abs(r["noise_frac_new"] - r["noise_frac_stored"]) for r in refit_rows)),
           "fit_seconds": {"median": float(np.median([r["fit_seconds"] for r in fit_rows])),
                           "min": float(min(r["fit_seconds"] for r in fit_rows)),
                           "max": float(max(r["fit_seconds"] for r in fit_rows))},
           "per_fit_verdicts": {f"{r['encoder']}|{r['seed']}": r["verdict"] for r in refit_rows},
           "csv": str(OUT / "anchor" / "d_anchor_refit.csv"), "finished": now_iso()}

    def mutate(d):
        d["G0b"] = g0b
        combine_anchor(d)
    locked_json_update(OUT / "anchor" / "d_anchor_result.json", mutate)
    write_stage_metadata("anchor_refit", t0, input_paths_for(ENCODERS) + [SRC / "matched" / f"draw{ANCHOR_DRAW:02d}.npz"],
                         extra={"realised_settings_example": {k: (v if not isinstance(v, np.generic) else v.item())
                                                              for k, v in real.items()}})
    print(f"[D anchor_refit] verdict={overall} exact={g0b['n_exact']} drift={g0b['n_drift']} fail={g0b['n_fail']} "
          f"min_ari={g0b['min_ari_vs_stored']:.12f}", flush=True)
    if overall == "fail":
        sys.exit(2)


# ---------------------------------------------------------------- stage: alt (NOT run in the anchor phase)
def stage_alt(args):
    t0 = time.perf_counter()
    z = stored_draw(args.draw)
    subset = z["subset"]
    assert len(subset) == MIMIC_N
    store = {"subset": subset}
    fitmeta = {}
    for enc in ENCODERS:
        x = load(enc)[subset]
        assert x.shape[0] == MIMIC_N
        for seed in args.seeds:
            t1 = time.perf_counter()
            labels, real = fit_partition(x, seed, ALT_NCOMP)
            sec = time.perf_counter() - t1
            assert real["min_cluster_size"] == EXPECTED_MCS, f"min_cluster_size {real['min_cluster_size']} != {EXPECTED_MCS}"
            assert real["n_components"] == ALT_NCOMP and real["n"] == MIMIC_N
            store[f"{enc}|{seed}"] = labels
            fitmeta[f"{enc}|{seed}"] = {**{k: (v.item() if isinstance(v, np.generic) else v) for k, v in real.items()},
                                        "fit_seconds": round(sec, 3)}
            print(f"[D alt] draw={args.draw} {enc} seed={seed} k={n_clusters(labels)} "
                  f"noise={(labels == -1).mean():.4f} sec={sec:.1f}", flush=True)
    (OUT / "fits").mkdir(parents=True, exist_ok=True)
    final = OUT / "fits" / f"alt_draw{args.draw:02d}.npz"
    tmp = final.with_name(final.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **store)
    os.replace(tmp, final)
    (OUT / "fits" / f"alt_draw{args.draw:02d}_fitmeta.json").write_text(json.dumps(fitmeta, indent=2))
    write_stage_metadata(f"alt_draw{args.draw:02d}", t0, input_paths_for(ENCODERS) + [SRC / "matched" / f"draw{args.draw:02d}.npz"])


# ---------------------------------------------------------------- stage: aggregate
def zone_of(ari, ami):
    if ari >= MIMIC_MIN_ARI and ami >= MIMIC_AMI:
        return "zone1_configuration_specific"
    if ari <= MAX_2D_ARI and ami <= MAX_2D_AMI:
        return "zone3_robust"
    return "zone2_indeterminate"


def stage_aggregate(args):
    import pandas as pd
    t0 = time.perf_counter()
    result_path = OUT / "anchor" / "d_anchor_result.json"
    if not result_path.exists() or json.loads(result_path.read_text()).get("verdict") not in ("exact", "drift"):
        raise SystemExit("anchor verdict is not exact|drift: aggregate refuses to run (PREDECLARED_PIPELINE.md A.6)")
    anchor = json.loads(result_path.read_text())

    stored, stored_files = load_stored_draws()
    alt, alt_files, fitmeta = {}, [], {}
    for d in DRAWS:
        f = OUT / "fits" / f"alt_draw{d:02d}.npz"
        assert f.exists(), f"missing {f}"
        alt_files.append(f)
        z = np.load(f)
        subset, labs = read_labels(z)
        assert np.array_equal(subset, stored[d][0]) and subset.dtype == stored[d][0].dtype, \
            f"draw {d}: subset differs from the stored draw"  # A.7 check 1
        alt[d] = (subset, labs)
        fitmeta[d] = json.loads((OUT / "fits" / f"alt_draw{d:02d}_fitmeta.json").read_text())

    ids = json.load(open(KOREA / "bge-m3" / "merged_ids.json"))
    ds = pd.read_csv(DATASET, low_memory=False,
                     usecols=["admission_ID", "BSI_episode_short_clean", "hospital_code", "test_date",
                              "treatment_department_eng", "n_bsi_episodes"]).drop_duplicates("admission_ID")
    ds = ds.set_index("admission_ID").loc[ids].reset_index()
    assert len(ds) == len(ids) and ds["admission_ID"].notna().all(), "covariate reindex produced NaN ids"  # A.7 check 6
    ds["admit_year"] = pd.to_datetime(ds["test_date"], errors="coerce").dt.year.astype(str)

    tables = {}
    for config, dl in (("orig_2d", stored), ("alt_20d", alt)):
        tables[config] = compute_tables(dl, tokens_df=ds)

    # d_alt_fits.csv: keep the orig_2d rows written by anchor_refit, (re)write the alt_20d rows
    fits_path = OUT / "d_alt_fits.csv"
    with open(fits_path, newline="") as f:
        orig_rows = [r for r in csv.DictReader(f) if r["config"] == "orig_2d"]
    assert len(orig_rows) == 20, f"expected 20 orig_2d rows from anchor_refit, found {len(orig_rows)}"
    alt_rows = []
    for r in tables["alt_20d"][0]:
        m = fitmeta[r["draw"]][f"{r['encoder']}|{r['seed']}"]
        alt_rows.append({"config": "alt_20d", "encoder": r["encoder"], "draw": r["draw"], "seed": r["seed"],
                         "n": m["n"], "n_components": m["n_components"], "n_neighbors": m["n_neighbors"],
                         "min_dist": m["min_dist"], "umap_metric": m["umap_metric"],
                         "random_state": m["random_state"], "min_cluster_size": m["min_cluster_size"],
                         "min_samples": "" if m["min_samples"] is None else m["min_samples"],
                         "hdbscan_metric": m["hdbscan_metric"],
                         "cluster_selection_method": m["cluster_selection_method"], "k": r["k"],
                         "n_noise": r["n_noise"], "noise_frac": r["noise_frac"],
                         "largest_cluster_share": r["largest_cluster_share"], "k_le_4": r["k_le_4"],
                         "fit_seconds": m["fit_seconds"]})
    write_csv(fits_path, FITS_COLS, alt_rows + orig_rows)

    def tag(config, rows):
        return [{"config": config, **r} for r in rows]
    write_csv(OUT / "d_alt_pairs.csv", PAIRS_COLS,
              tag("alt_20d", tables["alt_20d"][2]) + tag("orig_2d", tables["orig_2d"][2]))
    write_csv(OUT / "d_alt_crossseed.csv", CROSSSEED_COLS,
              tag("alt_20d", tables["alt_20d"][1]) + tag("orig_2d", tables["orig_2d"][1]))
    write_csv(OUT / "d_alt_token_ami.csv", TOKEN_COLS,
              tag("alt_20d", tables["alt_20d"][3]) + tag("orig_2d", tables["orig_2d"][3]))

    agg = {c: aggregate_tables(*tables[c][:3]) for c in tables}
    pub = json.load(open(SRC / "matched_encoders.json"))

    # -- section 5.0 gate G1 (per encoder) and degeneracy (A rule), alt arm
    alt_agg = agg["alt_20d"]
    encoders = {}
    for e in ENCODERS:
        cs = alt_agg["cross_seed_raw_pairs"][e]["ari"]["median"]
        frac = alt_agg["k_per_fit"][e]["frac_k_le_4"]
        encoders[e] = {"cross_seed_ari_median": cs,
                       "stable": bool(cs >= MIN_2D_WITHIN_ARI),
                       "k_median": alt_agg["k_per_fit"][e]["median"], "frac_k_le_4": frac,
                       "largest_share_median": alt_agg["largest_cluster_share"][e]["median"],
                       "degeneracy_limited": bool(frac > 0.5)}
    n_unstable = sum(not encoders[e]["stable"] for e in NONDEGENERATE_TRIO)
    # -- section 5.1 three-zone rule
    pair_zones = {}
    for a, b in TRIO_PAIRS:
        key = f"{a} vs {b}"
        p = alt_agg["cross_encoder_seed_matched"][key]
        ari, ami = p["ari"]["median"], p["ami"]["median"]
        limited = [e for e in (a, b) if not encoders[e]["stable"] or encoders[e]["degeneracy_limited"]]
        z = zone_of(ari, ami)
        p2 = pub["cross_encoder_seed_matched"][key]
        pair_zones[key] = {
            "ari_median_20d": ari, "ami_median_20d": ami, "ari_range_20d": [p["ari"]["min"], p["ari"]["max"]],
            "ami_range_20d": [p["ami"]["min"], p["ami"]["max"]],
            "ari_median_2d_published": p2["ari"]["median"], "ami_median_2d_published": p2["ami"]["median"],
            "ari_signed_diff_20d_minus_2d": ari - p2["ari"]["median"],
            "ami_signed_diff_20d_minus_2d": ami - p2["ami"]["median"],
            "zone": z, "reading": ZONE_READING[z], "limited_by": limited,
            "contributes_to_overall": not limited}
    if n_unstable >= 2:
        overall = "indeterminate"
    else:
        zs = [v["zone"] for v in pair_zones.values()]
        all_contrib = all(v["contributes_to_overall"] for v in pair_zones.values())
        if all_contrib and all(z == "zone1_configuration_specific" for z in zs):
            overall = "configuration-specific"
        elif all_contrib and all(z == "zone3_robust" for z in zs):
            overall = "robust"
        else:
            overall = "mixed/indeterminate"
    bge = encoders["bge-m3"]
    bge_reading = BGE_READING["attenuated"] if bge["frac_k_le_4"] <= 0.5 else BGE_READING["persists"]

    summary = {
        "label": LABEL, "anchor_verdict": anchor["verdict"], "anchor_g0a": anchor["G0a"]["verdict"],
        "anchor_g0b": anchor["G0b"]["verdict"],
        "boundaries": {"MAX_2D_ARI": MAX_2D_ARI, "MAX_2D_AMI": MAX_2D_AMI, "MIMIC_MIN_ARI": MIMIC_MIN_ARI,
                       "MIMIC_AMI": MIMIC_AMI, "MIN_2D_WITHIN_ARI": MIN_2D_WITHIN_ARI,
                       "note": "MIMIC values are fixed 2-D reference boundaries only; MIMIC was not re-run at 20-D"},
        "aggregates": agg, "encoders_gate_G1": encoders, "n_nondegenerate_encoders_instability_limited": n_unstable,
        "pair_zones": pair_zones, "overall_verdict": overall,
        "bge_m3_H_D4": {"frac_k_le_4_20d": bge["frac_k_le_4"], "k_median_20d": bge["k_median"],
                        "largest_share_median_20d": bge["largest_share_median"],
                        "published_2d": {"frac_k_le_4": 0.64, "k_median": 3.0, "largest_share_median": 0.9743},
                        "reading": bge_reading},
        "fit_seconds_alt": {"median": float(np.median([r["fit_seconds"] for r in alt_rows])),
                            "min": float(min(r["fit_seconds"] for r in alt_rows)),
                            "max": float(max(r["fit_seconds"] for r in alt_rows))},
        "token_ami_note": "five coarse covariates only; NOT the manuscript field-attribution set; "
                          "no dominant field is named (PREDECLARED_PIPELINE.md 5.4)",
    }

    def mutate(d):
        d.update(summary)
    locked_json_update(OUT / "d_alt_summary.json", mutate)
    write_stage_metadata("aggregate", t0, stored_files + alt_files + [DATASET, KOREA / "bge-m3" / "merged_ids.json"])
    print(f"[D aggregate] overall={overall} rows: fits={len(alt_rows) + len(orig_rows)} "
          f"pairs={len(tables['alt_20d'][2]) + len(tables['orig_2d'][2])} "
          f"crossseed={len(tables['alt_20d'][1]) + len(tables['orig_2d'][1])} "
          f"token={len(tables['alt_20d'][3]) + len(tables['orig_2d'][3])}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["anchor_stored", "anchor_refit", "alt", "aggregate"], required=True)
    ap.add_argument("--draw", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "anchor").mkdir(parents=True, exist_ok=True)
    print(f"[D] stage={args.stage} draw={args.draw} seeds={args.seeds} exe={sys.executable} "
          f"threads={[os.environ[k] for k in ('OMP_NUM_THREADS', 'NUMBA_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')]}",
          flush=True)
    {"anchor_stored": stage_anchor_stored, "anchor_refit": stage_anchor_refit,
     "alt": stage_alt, "aggregate": stage_aggregate}[args.stage](args)


if __name__ == "__main__":
    main()
