#!/usr/bin/env python3
"""Analysis A, PHASE 2 -- the EXISTING n-matched clustering design applied to one harmonized arm x one encoder.

One SLURM array task = one (arm, encoder) pair (same task numbering as 62_harmonized_embed.py) and runs
  * 10 draws x 5 seeds = 50 n-matched fits (n = 16,144), the design of out/39_audit6 / Figure 1, and
  * ONE whole-corpus fit (n = 32,329, seed 42) -- needed only for PREREGISTRATION_A section 4.3(1), which
    scores the harmonized Qwen3 partition against the stored Qwen3 `no_therapy` labels "on the full 32,329
    rows for the whole-corpus fit".  It is additional to the 300 matched fits and is reported as such.

NOTHING about the design is re-drawn or re-chosen: the fit function, UMAP/HDBSCAN configuration, seeds and the
min_cluster_size rule are IMPORTED from scripts/audit6_matched_encoders.py (fit_partition, min_cluster_size,
UMAP_NEIGHBORS); the 10 draw subsets are LOADED from out/39_audit6/matched/draw{01..10}.npz["subset"] (and
asserted bit-identical to np.random.default_rng(10_000 + draw).choice(32329, 16144, replace=False)).
Noise convention (both-non-noise ARI/AMI) and the degeneracy rule (k >= 5) live in the aggregation stage (64).

Environment: /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 (NOT the frozen env).  OMP/NUMBA/MKL/OPENBLAS
threads are pinned to 1 BEFORE any numeric import (audit6_matched_encoders does it at import; done here too so the
pool workers inherit it).  Fits are independent, so a process pool spreads them over SLURM_CPUS_PER_TASK cores;
UMAP with a fixed random_state is single-threaded and deterministic, so parallelism across fits cannot change a label.

Writes  out/followup_analyses/A_harmonized_template/matched/<arm>/<encoder>/
    d{DD}_s{S}.npy, d{DD}_s{S}.json     one per matched fit (labels int32 of length 16,144; per-fit metrics)
    whole_s42.npy, whole_s42.json       the whole-corpus fit
    run.json                            run record (env, versions, git+dirty, seeds, n, kwargs, all per-fit metrics)
Resumable: a fit whose .npy and .json both exist is skipped.
"""
import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import multiprocessing as mp
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
a6 = importlib.import_module("audit6_matched_encoders")     # fit_partition, min_cluster_size, UMAP_NEIGHBORS, MIMIC_N, OUT

REV = SCRIPT_DIR.parent
A_DIR = REV / "out/followup_analyses/A_harmonized_template"
EMBED = A_DIR / "embed"
MATCHED = A_DIR / "matched"
DRAWS = a6.OUT / "matched"                                   # out/39_audit6/matched
ARMS = ("korean_mimic_compatible", "harmonized_keep_therapy", "thickness_matched_random")
MODELS = ("bge-m3", "qwen3-0.6b")
TASKS = [(a, m) for a in ARMS for m in MODELS]
N_FULL = 32_329
WHOLE_SEED = 42
THIS_FILES = ["scripts/62_harmonized_embed.py", "scripts/63_harmonized_matched.py", "scripts/64_harmonized_fields.py",
              "slurm/62_harmonized_A_embed.sh", "slurm/63_harmonized_A_matched.sh", "slurm/64_harmonized_A_aggregate.sh"]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def git_info() -> dict:
    def run(*args) -> str:
        return subprocess.run(["git", "-C", str(REV), *args], capture_output=True, text=True, check=True).stdout.strip()

    st = run("status", "--porcelain", "--", *THIS_FILES)
    return {"git_commit": run("rev-parse", "HEAD"), "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(st), "git_dirty_detail": st.splitlines()[:20]}


def versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "umap-learn", "hdbscan", "scikit-learn", "numba", "pynndescent", "scipy", "pandas"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def atomic_write_json(obj, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, dest)


def atomic_save_npy(arr, dest: Path) -> None:
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    os.replace(tmp, dest)


def load_subset(draw: int) -> np.ndarray:
    return np.load(DRAWS / f"draw{draw:02d}.npz")["subset"]


def fit_job(job):
    """job = (arm, model, kind, draw, seed) -> per-fit metrics dict (labels written to disk)."""
    arm, model, kind, draw, seed, embed_root, out_root = job
    d = Path(out_root) / arm / model
    name = f"d{draw:02d}_s{seed}" if kind == "matched" else f"whole_s{seed}"
    x = np.load(Path(embed_root) / arm / model / "merged.npy", mmap_mode="r")
    if kind == "matched":
        subset = load_subset(draw)
        xs = np.ascontiguousarray(x[subset], dtype=np.float32)
    else:
        subset = None
        xs = np.ascontiguousarray(x, dtype=np.float32)
    t0 = time.time()
    labels = a6.fit_partition(xs, seed).astype(np.int32)
    sec = time.time() - t0
    nn = labels[labels != -1]
    k = int(len(set(nn.tolist())))
    largest = float(np.bincount(nn).max() / len(labels)) if len(nn) else 1.0
    atomic_save_npy(labels, d / f"{name}.npy")
    meta = {"name": name, "kind": kind, "draw": draw if kind == "matched" else None, "seed": seed, "n": int(len(labels)),
            "dim": int(xs.shape[1]), "min_cluster_size": int(a6.min_cluster_size(len(labels))), "k": k,
            "noise_frac": float((labels == -1).mean()), "largest_cluster_share": largest, "fit_seconds": round(sec, 1),
            "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest(), "subset_sha256": (hashlib.sha256(subset.tobytes()).hexdigest() if subset is not None else None),
            "pid": os.getpid()}
    atomic_write_json(meta, d / f"{name}.json")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=None, help="1-based array task id (same numbering as 62)")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--draws", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--no-whole", action="store_true", help="skip the whole-corpus seed-42 fit")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--embed-root", type=Path, default=EMBED, help="(testing) override the embedding root")
    ap.add_argument("--out-root", type=Path, default=MATCHED, help="(testing) override the output root")
    ap.add_argument("--smoke", type=int, default=None, help="run only the first N jobs (testing; writes real files, resumable)")
    args = ap.parse_args()
    if args.task is not None:
        args.arm, args.model = TASKS[args.task - 1]
    if not (args.arm and args.model):
        raise SystemExit("give --task N or --arm and --model")
    arm, model = args.arm, args.model
    tag = f"[63/{arm}/{model}]"
    EMB, OUTR = args.embed_root, args.out_root

    # ---- precondition: the embedding exists, is untruncated and is the file its manifest describes
    man_p = EMB / arm / model / "embed_manifest.json"
    if not man_p.exists():
        raise SystemExit(f"{tag} {man_p} missing -- embed first")
    man = json.loads(man_p.read_text())
    vec_p = EMB / arm / model / "merged.npy"
    x = np.load(vec_p, mmap_mode="r")
    if x.shape != (N_FULL, 1024) or man.get("n_truncated") != 0 or man.get("vectors_sha256") != sha256_file(vec_p):
        raise SystemExit(f"{tag} embedding file / manifest mismatch (shape {x.shape}, n_truncated {man.get('n_truncated')})")
    if not np.isfinite(np.asarray(x[:: max(1, N_FULL // 512)])).all():
        raise SystemExit(f"{tag} non-finite values in the embedding")

    # ---- the draw subsets are the stored ones (asserted reproducible; never redrawn)
    subset_ok = {}
    for dr in args.draws:
        st = load_subset(dr)
        regen = np.random.default_rng(10_000 + dr).choice(N_FULL, size=a6.MIMIC_N, replace=False)
        subset_ok[dr] = bool(np.array_equal(st, regen)) and st.shape == (a6.MIMIC_N,)
    if not all(subset_ok.values()):
        raise SystemExit(f"{tag} stored draw subsets are not reproducible from default_rng(10_000+draw): {subset_ok}")

    jobs = []
    if not args.no_whole:
        jobs.append((arm, model, "whole", None, WHOLE_SEED, str(EMB), str(OUTR)))               # longest fit first
    for dr in args.draws:
        for sd in args.seeds:
            jobs.append((arm, model, "matched", dr, sd, str(EMB), str(OUTR)))
    (OUTR / arm / model).mkdir(parents=True, exist_ok=True)

    def done(job) -> bool:
        _, _, kind, dr, sd = job[:5]
        name = f"d{dr:02d}_s{sd}" if kind == "matched" else f"whole_s{sd}"
        d = OUTR / arm / model
        return (d / f"{name}.npy").exists() and (d / f"{name}.json").exists()

    todo = [j for j in jobs if not done(j)]
    if args.smoke:
        todo = todo[: args.smoke]
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    workers = max(1, min(workers, len(todo) or 1))
    print(f"{tag} {len(jobs)} fits in the design, {len(todo)} to run, {workers} worker processes (1 thread each); "
          f"UMAP n_neighbors={a6.UMAP_NEIGHBORS}, HDBSCAN min_cluster_size(16144)={a6.min_cluster_size(a6.MIMIC_N)} "
          f"min_cluster_size(32329)={a6.min_cluster_size(N_FULL)}", flush=True)
    t0 = time.time()
    if todo:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for meta in pool.imap_unordered(fit_job, todo, chunksize=1):
                print(f"{tag} {meta['name']}: n={meta['n']} k={meta['k']} noise={meta['noise_frac']:.4f} "
                      f"largest={meta['largest_cluster_share']:.4f} ({meta['fit_seconds']}s)", flush=True)
    wall = time.time() - t0

    # ---- run record over ALL fits on disk
    metas = []
    for j in jobs:
        _, _, kind, dr, sd = j[:5]
        name = f"d{dr:02d}_s{sd}" if kind == "matched" else f"whole_s{sd}"
        p = OUTR / arm / model / f"{name}.json"
        if p.exists():
            metas.append(json.loads(p.read_text()))
    complete = len(metas) == len(jobs)
    run = {
        "analysis": "A_harmonized_template", "phase": "2 (Addendum 1)", "arm": arm, "encoder": model,
        "n_matched": a6.MIMIC_N, "n_whole": N_FULL, "draws": args.draws, "seeds": args.seeds, "whole_seed": None if args.no_whole else WHOLE_SEED,
        "n_fits_expected": len(jobs), "n_fits_on_disk": len(metas), "complete": complete,
        "draw_subsets_are_the_stored_ones_and_reproducible": subset_ok,
        "draw_source": str(DRAWS / "draw{01..10}.npz['subset']"),
        "embedding": str(vec_p), "embedding_dim": int(x.shape[1]), "embedding_vectors_sha256": man["vectors_sha256"],
        "embedding_n_truncated": man["n_truncated"],
        "umap_kwargs": {"n_components": 2, "n_neighbors": a6.UMAP_NEIGHBORS, "min_dist": 0.1, "metric": "euclidean", "random_state": "seed"},
        "hdbscan_kwargs": {"min_cluster_size": {"16144": a6.min_cluster_size(a6.MIMIC_N), "32329": a6.min_cluster_size(N_FULL)},
                           "min_samples": None, "metric": "euclidean", "cluster_selection_method": "eom"},
        "design_source": "scripts/audit6_matched_encoders.py: fit_partition / min_cluster_size (imported, not re-implemented)",
        "noise_convention_for_agreement": "both-non-noise (audit6_matched_encoders.agree), applied in 64",
        "degeneracy_rule": "k >= 5 in both partitions (applied in 64)",
        "workers": workers, "wall_seconds_this_invocation": round(wall, 1),
        "sys_executable": sys.executable, "versions": versions(), "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "written": now_iso(), **git_info(), "fits": metas,
    }
    atomic_write_json(run, OUTR / arm / model / "run.json")
    print(f"{tag} run.json written: {len(metas)}/{len(jobs)} fits on disk, complete={complete}", flush=True)
    if not complete and not args.smoke:
        raise SystemExit(f"{tag} INCOMPLETE: {len(metas)}/{len(jobs)} fits")


if __name__ == "__main__":
    main()
