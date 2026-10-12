#!/usr/bin/env python3
"""Task 66 -- Analysis B (FOLLOWUP-2026-09-21-B): cluster the three 3-small arms at
the preregistered 11 seeds and score them against the encoder-specific reference.

    B1a  unmodified_seed_only    the EXISTING stored embedding
                                 out/18_encoder_dec/embeddings/openai-3small/merged.npy
                                 re-clustered (zero API cost)
    B1b  unmodified_reembedded   65_B_variants.py embedding of narratives/full/
    B2   no_therapy              65_B_variants.py embedding of narratives/no_therapy/

Governing document (FROZEN): out/followup_analyses/B_symmetric_perturbation/PREREGISTRATION_B.md
  * seeds  [42, 1..10]  (s49.SEEDS, imported)
  * UMAP / HDBSCAN configuration: `49_3small_no_department.fit` (imported, not
    copied) -- n_components=2, n_neighbors=15, min_dist=0.1, euclidean,
    random_state=seed; HDBSCAN min_cluster_size=max(10, n//200)=161,
    min_samples=None, euclidean, eom
  * reference: out/18_encoder_dec/recluster/openai-3small/labels.npz (k=22, 277
    noise) for EVERY arm; nothing is re-normalised to a common reference
  * noise convention: BOTH-NON-NOISE. Computed with 17_common.agreement (its
    `ari_noise_excl` / `ami_noise_excl`) and cross-checked to 1e-9 against the
    arm's own `49_3small_no_department.agree`
  * field attribution: 40_field_dominance.field_ami / summarise_dominance over
    the full KOREA_CATEGORICAL + KOREA_CONTINUOUS field set. Dominant field for
    3-small = treatment_department_eng; context field = BSI_episode_short_clean.
    (audit7_dominance_checks.py's hard-coded dictionary is NOT used.)
  * environment: the FROZEN interpreter /home/programs/anaconda3/envs/gpt (the
    one that produced every Figure-3 arm). It is read-only and cannot embed.

STAGES
------
cluster --arm {B1a,B1b,B2}   one UMAP+HDBSCAN fit per seed; each label vector is
                             written atomically before the next fit (resumable);
                             a per-run JSON is written under runs/.
aggregate                    requires every arm's 11 label files. Writes
                             perturbation_matrix.csv (rows for these three arms
                             replace any earlier rows for the same arm_id; rows of
                             other arms are preserved), cross_seed_ari.csv,
                             field_ami_per_fit.csv and runs/aggregate_*.json.
                             No interpretation and no pass/fail verdict is
                             computed here; the preregistered decision rules are
                             applied by the reader.

Cross-seed ARI is written to its own file and is never mixed into the
`ari_vs_reference` column.

Writes ONLY under out/followup_analyses/B_symmetric_perturbation/.
"""
import os

# Pinned BEFORE any numeric import (thread count changes HDBSCAN's k at fixed seed;
# see 18e_recluster_frozen.py:14-16).
for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
import importlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
v65 = importlib.import_module("65_B_variants")     # paths, git/env records, sha256 helpers
s49 = v65.s49                                        # fit(), agree(), SEEDS, REFERENCE, constants
c17 = importlib.import_module("17_common")           # agreement, atomic_write_json/csv

REV = v65.REV
OUT_B = v65.OUT_B
LABELS = OUT_B / "labels"
RUNS = OUT_B / "runs"
MATRIX = OUT_B / "perturbation_matrix.csv"

SEEDS = list(s49.SEEDS)                       # [42, 1..10]
REFERENCE = s49.REFERENCE
STORED = v65.STORED_3SMALL
DOMINANT_FIELD = "treatment_department_eng"
CONTEXT_FIELD = "BSI_episode_short_clean"
CONVENTION = "both_nonnoise"
CONVENTION_SOURCE = ("17_common.agreement[ari_noise_excl, ami_noise_excl]; identical to "
                     "49_3small_no_department.agree (cross-checked)")
DEGENERATE_K_BELOW = 5                        # audit6_matched_encoders.py:134-137 (k >= 5 is non-degenerate)

ENCODER_LABEL = "OpenAI 3-small"              # naming used by out/33_figures/ablation.csv
DOMINANT_FIELD_LABEL = "treatment department"

ARMS = {
    "B1a": {"arm_id": "B1a_unmodified_seed_only", "variant": "full", "arm_class": "unmodified",
            "role": "reference band, seed noise only (stored embedding re-clustered)",
            "emb": STORED / "merged.npy", "ids": STORED / "merged_ids.json"},
    "B1b": {"arm_id": "B1b_unmodified_reembedded", "variant": "full", "arm_class": "unmodified",
            "role": "reference band, API re-embedding + seed noise"},
    "B2": {"arm_id": "B2_no_therapy", "variant": "no_therapy", "arm_class": "other field removed",
           "role": "size-graded non-dominant control"},
}
for _k in ("B1b", "B2"):
    _d = v65.arm_dir(_k)
    ARMS[_k]["emb"] = _d / "merged.npy"
    ARMS[_k]["ids"] = _d / "merged_ids.json"


def label_path(arm: str, seed: int) -> Path:
    return LABELS / f"{ARMS[arm]['arm_id']}__seed{seed}.npy"


def load_reference():
    payload = np.load(REFERENCE, allow_pickle=True)
    return payload["labels"].astype(np.int64), [int(x) for x in payload["admission_ID"]]


def n_clusters(lab: np.ndarray) -> int:
    return int(len(set(lab.tolist()) - {-1}))


def largest_share(lab: np.ndarray) -> float:
    """Largest non-noise cluster / all rows (audit6_matched_encoders.py:124-125)."""
    nn = lab[lab != -1]
    return float(np.bincount(nn).max() / len(lab)) if len(nn) else 1.0


def stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


# --------------------------------------------------------------------------- cluster
def stage_cluster(arm: str) -> None:
    cfg = ARMS[arm]
    reference, ids_ref = load_reference()
    n = len(ids_ref)
    for p in (cfg["emb"], cfg["ids"]):
        if not p.exists():
            raise SystemExit(f"{p} missing")
    ids = [int(x) for x in json.loads(cfg["ids"].read_text())]
    if ids != ids_ref:
        raise SystemExit(f"{arm}: embedding row order differs from the reference partition")
    x = np.load(cfg["emb"])
    if x.shape != (n, s49.EMBED_DIM) or not np.isfinite(x).all():
        raise SystemExit(f"{arm}: unexpected embedding {x.shape}")

    LABELS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    started = v65.now_iso()
    run = {
        "stage": "cluster", "arm": arm, "arm_id": cfg["arm_id"], "role": cfg["role"],
        "variant": cfg["variant"], "encoder": s49.MODEL_NAME, "n": n, "seeds": SEEDS,
        "umap_kwargs": dict(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean",
                            random_state="<seed>"),
        "hdbscan_kwargs": dict(min_cluster_size=max(10, n // 200), min_samples=None,
                               metric="euclidean", cluster_selection_method="eom"),
        "fit_function": "49_3small_no_department.fit (imported)",
        "embedding_path": str(cfg["emb"]), "embedding_shape": list(x.shape),
        "embedding_dtype": str(x.dtype), "embedding_sha256": v65.sha256_file(cfg["emb"]),
        "reference_path": str(REFERENCE), "reference_sha256": v65.sha256_file(REFERENCE),
        "reference_k": n_clusters(reference), "reference_noise": int((reference == -1).sum()),
        "started": started, "fits": {}, **v65.env_record(),
    }

    labels = {}
    for seed in SEEDS:
        dest = label_path(arm, seed)
        if dest.exists():
            lab = np.load(dest)
            if lab.shape != (n,):
                raise SystemExit(f"{dest}: cached labels have shape {lab.shape}")
            labels[seed] = lab
            run["fits"][str(seed)] = {"cached": True, "k": n_clusters(lab),
                                      "noise_fraction": float((lab == -1).mean())}
            print(f"[66/{arm}|{seed}] cached k={n_clusters(lab)}", flush=True)
            continue
        t0 = time.time()
        lab = s49.fit(x, seed)
        tmp = dest.with_name(dest.name + ".tmp.npy")   # np.save appends .npy otherwise
        np.save(tmp, lab)
        tmp.replace(dest)
        labels[seed] = lab
        secs = round(time.time() - t0, 1)
        run["fits"][str(seed)] = {"cached": False, "seconds": secs, "k": n_clusters(lab),
                                  "noise_fraction": float((lab == -1).mean())}
        print(f"[66/{arm}|{seed}] k={n_clusters(lab)} noise={(lab == -1).mean():.3f} ({secs}s)", flush=True)

    # numbers only, for the reproduction record (B1a seed 42 reproduces the stored fit of record)
    ag = c17.agreement(reference, labels[42])
    run["seed42_vs_reference"] = {"ari_both_nonnoise": ag["ari_noise_excl"],
                                  "ami_both_nonnoise": ag["ami_noise_excl"]}
    run["finished"] = v65.now_iso()
    dest = RUNS / f"{cfg['arm_id']}__cluster__{stamp()}_{run['slurm_job_id'] or 'nojob'}.json"
    c17.atomic_write_json(run, dest)
    print(f"[66/{arm}] wrote {dest}", flush=True)


# --------------------------------------------------------------------------- aggregate
def edit_sizes() -> dict:
    out = {}
    with open(REV / "out/21_published_variants/generation_summary.csv") as fh:
        for row in csv.DictReader(fh):
            out[row["variant"]] = (int(row["corpus_wide_lines_removed"]), int(row["n_docs"]))
    return out


def summarise(vals) -> dict:
    v = np.asarray(vals, dtype=float)
    return {"n": int(len(v)), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def stage_aggregate(arm_keys) -> None:
    import pandas as pd

    v40 = importlib.import_module("40_field_dominance")
    reference, ids_ref = load_reference()
    n = len(ids_ref)

    missing = [f"{a}|{s}" for a in arm_keys for s in SEEDS if not label_path(a, s).exists()]
    if missing:
        raise SystemExit(f"missing {len(missing)} fits, e.g. {missing[:5]}")

    # fields on the reference row order (by admission_ID, never by position)
    frame = v40.load_korea_fields()
    frame = frame.set_index("admission_ID").loc[ids_ref].reset_index()
    fields = v40.build_fields(frame, v40.KOREA_CATEGORICAL, v40.KOREA_CONTINUOUS)
    for f in (DOMINANT_FIELD, CONTEXT_FIELD):
        if f not in fields:
            raise SystemExit(f"field {f} not available")
    sizes = edit_sizes()

    git = v65.git_info()
    rows, field_rows, cross_rows = [], [], []
    per_arm = {}
    max_diff = 0.0
    for arm in arm_keys:
        cfg = ARMS[arm]
        removed, n_docs = sizes[cfg["variant"]]
        labs = {s: np.load(label_path(arm, s)) for s in SEEDS}
        per_fit_scores = []
        for seed in SEEDS:
            lab = labs[seed]
            if lab.shape != (n,):
                raise SystemExit(f"{arm}|{seed}: labels shape {lab.shape}")
            ag = c17.agreement(reference, lab)
            ari, ami = ag["ari_noise_excl"], ag["ami_noise_excl"]
            a49 = s49.agree(lab, reference)
            diff = max(abs(a49[0] - ari), abs(a49[1] - ami))
            max_diff = max(max_diff, diff)
            if diff > 1e-9:
                raise SystemExit(f"{arm}|{seed}: 17_common.agreement and 49.agree disagree by {diff}")
            scores = v40.field_ami(lab, fields)
            per_fit_scores.append(scores)
            ordered = sorted(scores.items(), key=lambda kv: -kv[1])
            (top, top_ami), (runner, run_ami) = ordered[0], ordered[1]
            k = n_clusters(lab)
            rows.append({
                "analysis": "B", "encoder": ENCODER_LABEL, "dominant_field": DOMINANT_FIELD_LABEL,
                "arm_id": cfg["arm_id"], "variant": cfg["variant"], "arm": cfg["arm_class"],
                "role": cfg["role"], "seed": seed, "n": n, "k": k,
                "noise_fraction": float((lab == -1).mean()), "largest_cluster_share": largest_share(lab),
                "degenerate_k_lt_5": bool(k < DEGENERATE_K_BELOW),
                "agreement_convention": CONVENTION,
                "ari_vs_reference": ari, "ami_vs_reference": ami,
                "n_both_nonnoise": ag["n_both_nonnoise"],
                "reference": "out/18_encoder_dec/recluster/openai-3small/labels.npz",
                "reference_k": n_clusters(reference), "reference_noise": int((reference == -1).sum()),
                "dominant_field_name": DOMINANT_FIELD, "dominant_field_ami": scores[DOMINANT_FIELD],
                "organism_field_name": CONTEXT_FIELD, "organism_ami": scores[CONTEXT_FIELD],
                "top_field": top, "top_field_ami": top_ami, "runner_up_field": runner,
                "dominance_margin": top_ami - run_ami,
                "edit_lines_removed_total": removed, "edit_lines_removed_per_doc": removed / n_docs,
                "embedding_source": str(cfg["emb"].relative_to(REV)),
                "git_commit": git["git_commit"],
            })
            for fname, val in scores.items():
                field_rows.append({"arm_id": cfg["arm_id"], "seed": seed, "field": fname, "ami": val})
        # cross-seed ARI within the arm: separate file, never mixed into ari_vs_reference
        cs = []
        for a, b in itertools.combinations(SEEDS, 2):
            v = c17.agreement(labs[a], labs[b])["ari_noise_excl"]
            cs.append(v)
            cross_rows.append({"arm_id": cfg["arm_id"], "seed_a": a, "seed_b": b,
                               "agreement_convention": CONVENTION, "ari_cross_seed": v})
        arm_rows = [r for r in rows if r["arm_id"] == cfg["arm_id"]]
        per_arm[cfg["arm_id"]] = {
            "arm": arm, "n_seeds": len(SEEDS),
            "ari_vs_reference": summarise([r["ari_vs_reference"] for r in arm_rows]),
            "ami_vs_reference": summarise([r["ami_vs_reference"] for r in arm_rows]),
            "dominant_field_ami": summarise([r["dominant_field_ami"] for r in arm_rows]),
            "organism_ami": summarise([r["organism_ami"] for r in arm_rows]),
            "k": summarise([r["k"] for r in arm_rows]),
            "noise_fraction": summarise([r["noise_fraction"] for r in arm_rows]),
            "largest_cluster_share": summarise([r["largest_cluster_share"] for r in arm_rows]),
            "n_degenerate_fits": int(sum(r["degenerate_k_lt_5"] for r in arm_rows)),
            "cross_seed_ari": {**summarise(cs), "n_pairs": len(cs)},
            "dominance_summary_40": v40.summarise_dominance(per_fit_scores),
            "edit_lines_removed_per_doc": removed / n_docs,
            "embedding_sha256": v65.sha256_file(cfg["emb"]),
        }
        print(f"[66/aggregate/{arm}] ARI median {per_arm[cfg['arm_id']]['ari_vs_reference']['median']:+.4f} "
              f"[{per_arm[cfg['arm_id']]['ari_vs_reference']['min']:+.4f}, "
              f"{per_arm[cfg['arm_id']]['ari_vs_reference']['max']:+.4f}]", flush=True)

    OUT_B.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    mine = set(new["arm_id"])
    if MATRIX.exists():
        old = pd.read_csv(MATRIX)
        new = pd.concat([old[~old["arm_id"].isin(mine)], new], ignore_index=True, sort=False)
    c17.atomic_write_csv(new, MATRIX)
    c17.atomic_write_csv(pd.DataFrame(field_rows), OUT_B / "field_ami_per_fit.csv")
    c17.atomic_write_csv(pd.DataFrame(cross_rows), OUT_B / "cross_seed_ari.csv")

    RUNS.mkdir(parents=True, exist_ok=True)
    run = {
        "stage": "aggregate", "arms": arm_keys, "seeds": SEEDS, "n": n,
        "agreement_convention": CONVENTION, "agreement_source": CONVENTION_SOURCE,
        "max_abs_diff_17_common_vs_49_agree": max_diff,
        "dominant_field": DOMINANT_FIELD, "context_field": CONTEXT_FIELD,
        "degenerate_rule": f"k < {DEGENERATE_K_BELOW}",
        "n_fields_scored": len(fields), "fields_scored": list(fields),
        "reference_path": str(REFERENCE), "reference_sha256": v65.sha256_file(REFERENCE),
        "reference_k": n_clusters(reference), "reference_noise": int((reference == -1).sum()),
        "per_arm": per_arm, "matrix_csv": str(MATRIX), "written": v65.now_iso(),
        **v65.env_record(),
    }
    dest = RUNS / f"aggregate__{stamp()}_{run['slurm_job_id'] or 'nojob'}.json"
    c17.atomic_write_json(run, dest)
    print(f"[66/aggregate] wrote {MATRIX} ({len(new)} rows total) and {dest}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cluster", "aggregate"])
    ap.add_argument("--arm", default=None, choices=list(ARMS))
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="aggregate only: comma-separated arm keys (default all three)")
    args = ap.parse_args()
    if args.stage == "cluster":
        if not args.arm:
            raise SystemExit("--arm is required for --stage cluster")
        stage_cluster(args.arm)
    else:
        stage_aggregate([a for a in args.arms.split(",") if a])


if __name__ == "__main__":
    main()
