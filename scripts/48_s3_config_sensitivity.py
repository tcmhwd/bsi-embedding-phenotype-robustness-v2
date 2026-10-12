#!/usr/bin/env python3
"""Task 48 -- the S3 axis (clustering-configuration sensitivity), closing
pre-registration deviation D2.

MUST be invoked with the FROZEN interpreter:
    /home/programs/anaconda3/envs/gpt/bin/python3 48_s3_config_sensitivity.py --stage ...

WHY THIS EXISTS
---------------
`PRESPEC_stress_criteria.md` section 2 specifies six stress axes and section 5
makes Core = S1 AND S2 AND S3. S3 was never evaluated for the originally
derived partition (DEVIATION NOTE 2026-09-06, D2), because the only
configuration sweeps on disk (`out/02_recluster/*mcs200*`) were produced under
the umap-learn 0.5.7 stack, which does not reproduce that partition at all and
therefore cannot serve as S3 for this object. Every survival count in the
manuscript has consequently been reported over S1/S2/S4/S5 (+S6), never over
"the pre-registered battery".

This script runs S3 as written, in the reproducing environment, so the count
can be reported over the axis set the pre-registration actually specifies.

WHAT THE FROZEN DOCUMENT SPECIFIES (transcribed, not paraphrased)
-----------------------------------------------------------------
section 2, S3 row:
    perturbation : "A1 (`nc20_nn30_mcs100`, seed 42) and its 20 seeds; plus
                    mcs in {100, 200} at the published 2-D setting"
    replicates   : 21 + 2
    metric       : "R_cfg(c) = coverage-purity retention against A1 seed 42;
                    J at mcs 100/200"
section 3, S3 row:
    PASS if      : "R_cfg(c) >= 0.75 AND J(c, A0@mcs200) >= 0.50"
section 2, definitions:
    J(c, P)      : "max over clusters k != -1 of partition P of
                    |c & k| / |c | k|, with noise rows counted as not
                    co-assigned"
    R(c, P)      : "fraction of c's members that lie in clusters k of P whose
                    members are >= 80% inside c"
section 1.3, A1 :  "nc20_nn30_mcs100 (min_dist 0, cosine, min_samples 10),
                    seed 42, on the stored vectors"
section 1.5     :  "Noise (label -1) is never a phenotype and is never scored."

IMPLEMENTATION DECISIONS (stated because the frozen text does not fix them)
---------------------------------------------------------------------------
D-a. In R(c, P), a candidate cluster k of P qualifies when
     |k & c| / |k| >= 0.80 -- purity of k with respect to c, k's own members as
     the denominator. This is the reading that makes R credit a SPLIT of c into
     pure sub-clusters, which section 2 gives as its whole motivation ("this
     credits splits of c into pure sub-clusters, which Jaccard does not").
     Noise (k = -1) never qualifies, per section 1.5.
D-b. The GATED R_cfg is the one against A1 seed 42, exactly as section 2 says.
     The 20 further A1 seeds are replicates: we report R_cfg against each of
     them as a distribution (median, min, max, 5th percentile) beside the gated
     value, and gate on nothing but seed 42. Reporting them costs nothing and
     lets a reader see whether the gated value sits at the edge of its own
     seed distribution.
D-c. "A0@mcs200" is the configuration of record (UMAP nc2/nn15/md0.1/euclidean/
     random_state=42) re-clustered with HDBSCAN min_cluster_size=200 on the
     SAME UMAP space; likewise mcs=100. Only min_cluster_size changes, since
     that is what the frozen text varies. The UMAP space is fitted once at
     seed 42 and reused for both, which is also what "at the published 2-D
     setting" means.
D-d. The reference partition is the stored originally derived labels (the
     object the manuscript reports), consistent with deviation D1.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It does not touch any existing survival matrix. It writes an S3 column set to
`out/48_s3/` only; merging S3 into the reported counts is a separate, explicit
step so that the pre-S3 and post-S3 readings can both be shown.

STAGES
------
  --stage a1     --seed N   one A1 fit (21 of these: seed 42 and 1-20)
  --stage mcs              the two published-2-D fits (mcs 100 and 200)
  --stage aggregate        R_cfg, J, PASS columns -> out/48_s3/s3_matrix.csv

Resumable: every stage skips work whose output already exists.
"""
import os
import sys

for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"

import argparse
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import time
from pathlib import Path

import hdbscan
import numba
import numpy as np
import pandas as pd
import sklearn
import umap

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c17 = importlib.import_module("17_common")

OUT_DIR = c17.REV / "out/48_s3"
A1_DIR = OUT_DIR / "a1"
MCS_DIR = OUT_DIR / "mcs"

# PRESPEC section 1.3: A1 = nc20_nn30_mcs100, min_dist 0, cosine, min_samples 10.
A1_UMAP_KW = dict(n_components=20, n_neighbors=30, min_dist=0.0, metric="cosine")
A1_HDBSCAN_KW = dict(min_cluster_size=100, min_samples=10, metric="euclidean",
                     cluster_selection_method="eom")
A1_SEEDS = [42] + list(range(1, 21))          # 21 replicates
MCS_VALUES = [100, 200]                        # 2 replicates, published 2-D setting

R_CFG_PURITY = 0.80    # PRESPEC section 2: "members are >= 80% inside c"
R_CFG_PASS = 0.75      # PRESPEC section 3
J_MCS200_PASS = 0.50   # PRESPEC section 3


def versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__,
            "umap-learn": umap.__version__,
            "hdbscan": importlib.metadata.version("hdbscan"),
            "numba": numba.__version__}


def coverage_purity_retention(reference: np.ndarray, derived: np.ndarray,
                              ref_cluster: int, purity: float = R_CFG_PURITY) -> float:
    """R(c, P): fraction of cluster c's members that fall in clusters of P
    whose own membership is at least `purity` inside c. Noise never qualifies
    (PRESPEC section 1.5). See decision D-a in the module docstring."""
    in_c = reference == ref_cluster
    n_c = int(in_c.sum())
    if n_c == 0:
        return float("nan")
    retained = 0
    for k in set(derived.tolist()) - {-1}:
        in_k = derived == k
        n_k = int(in_k.sum())
        if n_k == 0:
            continue
        overlap = int((in_k & in_c).sum())
        if overlap / n_k >= purity:
            retained += overlap
    return retained / n_c


def stage_a1(seed: int) -> None:
    A1_DIR.mkdir(parents=True, exist_ok=True)
    dest = A1_DIR / f"result_a1_seed{seed}.json"
    if dest.exists():
        print(f"[a1|{seed}] already done, skipping (resume)", flush=True)
        return

    ref = c17.load_reference_frame()
    reference = ref[c17.PUBLISHED_CLUSTER_COL].to_numpy(dtype=np.int64)
    emb = c17.load_embeddings()
    if emb.shape[0] != len(reference):
        raise ValueError(f"row mismatch: labels={len(reference)} emb={emb.shape[0]}")

    print(f"[a1|{seed}] UMAP={dict(**A1_UMAP_KW, random_state=seed)} "
          f"HDBSCAN={A1_HDBSCAN_KW}", flush=True)
    t0 = time.time()
    space = umap.UMAP(random_state=seed, **A1_UMAP_KW).fit_transform(emb)
    t_umap = time.time() - t0
    t1 = time.time()
    derived = hdbscan.HDBSCAN(core_dist_n_jobs=1, **A1_HDBSCAN_KW).fit_predict(
        np.ascontiguousarray(space, dtype=np.float64))
    t_hdb = time.time() - t1

    k = int(len(set(derived.tolist()) - {-1}))
    np.savez_compressed(A1_DIR / f"labels_a1_seed{seed}.npz", labels=derived.astype(np.int32))
    c17.atomic_write_json({
        "arm": "A1", "seed": seed,
        "umap": dict(**A1_UMAP_KW, random_state=seed), "hdbscan": A1_HDBSCAN_KW,
        "n_clusters": k, "n_noise": int((derived == -1).sum()),
        "noise_frac": float((derived == -1).mean()),
        "agreement_vs_published": c17.agreement(reference, derived),
        "python_executable": sys.executable, "versions": versions(),
        "wall_time_umap_seconds": t_umap, "wall_time_hdbscan_seconds": t_hdb,
    }, dest)
    print(f"[a1|{seed}] k={k} noise={(derived == -1).mean():.4f} -> {dest}", flush=True)


def stage_mcs() -> None:
    MCS_DIR.mkdir(parents=True, exist_ok=True)
    todo = [m for m in MCS_VALUES if not (MCS_DIR / f"result_mcs{m}.json").exists()]
    if not todo:
        print("[mcs] both already done, skipping (resume)", flush=True)
        return

    ref = c17.load_reference_frame()
    reference = ref[c17.PUBLISHED_CLUSTER_COL].to_numpy(dtype=np.int64)
    emb = c17.load_embeddings()

    # One UMAP space at the published 2-D setting, seed 42, reused for both
    # min_cluster_size values (decision D-c).
    print(f"[mcs] UMAP={dict(**c17.UMAP_KW, random_state=42)}", flush=True)
    t0 = time.time()
    space = np.ascontiguousarray(
        umap.UMAP(random_state=42, **c17.UMAP_KW).fit_transform(emb), dtype=np.float64)
    t_umap = time.time() - t0

    for m in todo:
        t1 = time.time()
        derived = hdbscan.HDBSCAN(min_cluster_size=m, core_dist_n_jobs=1,
                                  **c17.HDBSCAN_KW).fit_predict(space)
        t_hdb = time.time() - t1
        k = int(len(set(derived.tolist()) - {-1}))
        np.savez_compressed(MCS_DIR / f"labels_mcs{m}.npz", labels=derived.astype(np.int32))
        c17.atomic_write_json({
            "arm": "published_2d", "min_cluster_size": m,
            "umap": dict(**c17.UMAP_KW, random_state=42),
            "hdbscan": dict(min_cluster_size=m, **c17.HDBSCAN_KW),
            "n_clusters": k, "n_noise": int((derived == -1).sum()),
            "noise_frac": float((derived == -1).mean()),
            "agreement_vs_published": c17.agreement(reference, derived),
            "python_executable": sys.executable, "versions": versions(),
            "wall_time_umap_seconds": t_umap, "wall_time_hdbscan_seconds": t_hdb,
        }, MCS_DIR / f"result_mcs{m}.json")
        print(f"[mcs|{m}] k={k} noise={(derived == -1).mean():.4f}", flush=True)


def stage_aggregate() -> None:
    ref = c17.load_reference_frame()
    reference = ref[c17.PUBLISHED_CLUSTER_COL].to_numpy(dtype=np.int64)
    clusters = sorted(set(reference.tolist()) - {-1})

    missing = [s for s in A1_SEEDS if not (A1_DIR / f"labels_a1_seed{s}.npz").exists()]
    missing += [f"mcs{m}" for m in MCS_VALUES if not (MCS_DIR / f"labels_mcs{m}.npz").exists()]
    if missing:
        raise SystemExit(f"missing fits: {missing}")

    a1 = {s: np.load(A1_DIR / f"labels_a1_seed{s}.npz")["labels"] for s in A1_SEEDS}
    mcs = {m: np.load(MCS_DIR / f"labels_mcs{m}.npz")["labels"] for m in MCS_VALUES}

    rows = []
    for c in clusters:
        r_gated = coverage_purity_retention(reference, a1[42], c)
        r_seeds = [coverage_purity_retention(reference, a1[s], c) for s in A1_SEEDS if s != 42]
        j100 = c17.jaccard_for_cluster(reference, mcs[100], c)
        j200 = c17.jaccard_for_cluster(reference, mcs[200], c)
        rows.append({
            "cluster": int(c),
            "size": int((reference == c).sum()),
            "R_cfg_a1_seed42": r_gated,
            "R_cfg_a1_other_seeds_median": float(np.median(r_seeds)),
            "R_cfg_a1_other_seeds_min": float(np.min(r_seeds)),
            "R_cfg_a1_other_seeds_p05": float(np.percentile(r_seeds, 5)),
            "R_cfg_a1_other_seeds_max": float(np.max(r_seeds)),
            "J_mcs100": j100,
            "J_mcs200": j200,
            "R_cfg_pass": bool(r_gated >= R_CFG_PASS),
            "J_mcs200_pass": bool(j200 >= J_MCS200_PASS),
            "S3_pass": bool(r_gated >= R_CFG_PASS and j200 >= J_MCS200_PASS),
        })

    df = pd.DataFrame(rows).sort_values("cluster")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    c17.atomic_write_csv(df, OUT_DIR / "s3_matrix.csv")

    summary = {
        "axis": "S3 (clustering-configuration sensitivity)",
        "closes_deviation": "D2",
        "thresholds": {"R_cfg": R_CFG_PASS, "J_mcs200": J_MCS200_PASS,
                       "R_cfg_purity": R_CFG_PURITY},
        "replicates": {"A1_seeds": len(A1_SEEDS), "mcs_values": MCS_VALUES},
        "a1_k_per_seed": {str(s): int(len(set(a1[s].tolist()) - {-1})) for s in A1_SEEDS},
        "mcs_k": {str(m): int(len(set(mcs[m].tolist()) - {-1})) for m in MCS_VALUES},
        "S3_pass_count": int(df["S3_pass"].sum()),
        "S3_pass_clusters": [int(x) for x in df.loc[df["S3_pass"], "cluster"]],
        "R_cfg_pass_count": int(df["R_cfg_pass"].sum()),
        "J_mcs200_pass_count": int(df["J_mcs200_pass"].sum()),
        "versions": versions(),
        "implementation_decisions": "see module docstring D-a..D-d",
    }
    c17.atomic_write_json(summary, OUT_DIR / "s3_summary.json")

    print(df.to_string(index=False), flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "versions"},
                     indent=1, ensure_ascii=False), flush=True)
    print(f"[48] wrote {OUT_DIR / 's3_matrix.csv'} and s3_summary.json", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["a1", "mcs", "aggregate"], required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "a1":
        if args.seed is None:
            raise SystemExit("--stage a1 requires --seed")
        stage_a1(args.seed)
    elif args.stage == "mcs":
        stage_mcs()
    else:
        stage_aggregate()


if __name__ == "__main__":
    main()
