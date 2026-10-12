#!/usr/bin/env python3
"""Task 66b -- Analysis B, arm B3 (FOLLOWUP-2026-09-21-B): cluster the Qwen3-Embedding-0.6B
`no_department` embedding at the preregistered 11 seeds and score it against the Qwen3
encoder-specific reference.

    B3  qwen3_no_department   65b_B3_embed_qwen3.py embedding of narratives/no_department/

Governing documents (FROZEN): out/followup_analyses/B_symmetric_perturbation/PREREGISTRATION_B.md
(H-B3, sections 3.1-3.5, 4.4) and out/followup_analyses/qc/AUTHOR_DECISIONS_2026-09-21.md
(section 7.1: B3 is run, labelled presentational / symmetry-completing; reported whatever it shows).

Everything that defines the measurement is IMPORTED from the arms it extends; nothing is redefined:
  * seeds          [42, 1..10]                    47_qwen3_extra_seeds.SEEDS
  * fit            UMAP(n_components=2, n_neighbors=15, min_dist=0.1, euclidean, random_state=seed)
                   -> HDBSCAN(min_cluster_size=max(10, n//200)=161, min_samples=None, euclidean, eom)
                                                47_qwen3_extra_seeds.fit  (the function that fitted
                                                every existing Qwen3 Figure-3 arm; no float64 cast,
                                                unlike 49's fit -- see EXISTING_PIPELINE_MAP.md 3.5)
  * reference      out/18_encoder_dec/recluster/qwen3-0.6b/labels.npz  (k=24, 851 noise), the same
                   object for every Qwen3 arm; row order ASSERTED, not reindexed
                                                47_qwen3_extra_seeds.STORED / IDS
  * noise/ARI/AMI  BOTH-NON-NOISE convention. Computed with 17_common.agreement (`ari_noise_excl`,
                   `ami_noise_excl`) and cross-checked to 1e-9 against 47_qwen3_extra_seeds.agree
                   (the Qwen3 arms' own function). Recorded in every output row as
                   `agreement_convention = both_nonnoise`.
  * field AMI      40_field_dominance.field_ami / summarise_dominance over the full
                   KOREA_CATEGORICAL + KOREA_CONTINUOUS set. Qwen3's dominant field is
                   `first_antibiotic`; context field `BSI_episode_short_clean` (organism). Cross-checked
                   to 1e-9 against 47_qwen3_extra_seeds.field_ami. audit7_dominance_checks.py's
                   hard-coded dictionary is NOT used.
  * degeneracy     k < 5 flagged (66_B_cluster.DEGENERATE_K_BELOW; audit6_matched_encoders.py:134-137)
  * helpers        66_B_cluster.{n_clusters, largest_share, summarise, edit_sizes, stamp};
                   65_B_variants.{sha256_file, now_iso, package_versions}
  * environment    the FROZEN interpreter /home/programs/anaconda3/envs/gpt (the one that produced every
                   Figure-3 arm). Read-only; it cannot embed. Threads pinned to 1 BEFORE numeric imports.

STAGES
------
cluster --seed-index i | --seed s
                 ONE UMAP+HDBSCAN fit; labels written atomically; per-run JSON under runs/.
                 (SLURM array 0-10, one task per seed; seeds are independent, and at a fixed
                 random_state the frozen stack is thread-independent -- 47_qwen3_extra_seeds.py.)
aggregate        requires all 11 label files. Writes, SEPARATE from the B1/B2 files:
                   perturbation_matrix_B3.csv   (same columns as perturbation_matrix.csv)
                   cross_seed_ari_B3.csv        (cross-seed ARI, never mixed into ari_vs_reference)
                   field_ami_per_fit_B3.csv
                   knn_readout_B3.csv           (15-NN cosine purity, first antibiotic + organism)
                   runs/aggregate_B3__*.json
                 No interpretation and no pass/fail verdict is computed here; the preregistered decision
                 rule (median ARI >= 0.90) is applied by the reader.

15-NN readout: PREREGISTRATION_B.md section 3.5 (last paragraph) extends it to this arm at k = 15 over all
32,329 admissions (chance 0.093 first antibiotic / 0.128 organism). It is a RE-RUN of the existing
definition in `audit9_knn_purity.py` (raw cosine graph, self excluded, no UMAP/HDBSCAN, purity = mean over
rows of the fraction of the 15 neighbours sharing the row's value). That script is a top-level program that
writes into the pre-existing out/45_audit9, so its lines are restated here, and the restatement is
VERIFIED by re-running it on the stored full Qwen3 embedding and comparing with the value already in
out/45_audit9/knn_purity.json (and the chance rates likewise). No new endpoint is introduced; the
existing definition's third field (department) is not in the preregistration and is not computed.

Does NOT modify 65_B_variants.py / 66_B_cluster.py / any committed script, and does not write
perturbation_matrix.csv. Writes ONLY under out/followup_analyses/B_symmetric_perturbation/.
"""
import os

# Pinned BEFORE any numeric import (thread count changes HDBSCAN's k at fixed seed;
# see 18e_recluster_frozen.py:14-16).
for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import importlib
import itertools
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
v65 = importlib.import_module("65_B_variants")        # read-only import: sha256_file, now_iso, package_versions
v66 = importlib.import_module("66_B_cluster")         # read-only import: n_clusters, largest_share, summarise, ...
q47 = importlib.import_module("47_qwen3_extra_seeds")  # fit, agree, field_ami, load_fields, SEEDS, STORED, IDS
c17 = importlib.import_module("17_common")            # agreement, atomic_write_json/csv

REV = v65.REV
OUT_B = v65.OUT_B
LABELS = OUT_B / "labels"
RUNS = OUT_B / "runs"
MATRIX_B3 = OUT_B / "perturbation_matrix_B3.csv"
CROSS_B3 = OUT_B / "cross_seed_ari_B3.csv"
FIELD_B3 = OUT_B / "field_ami_per_fit_B3.csv"
KNN_B3 = OUT_B / "knn_readout_B3.csv"

ARM_ID = "B3_qwen3_no_department"
VARIANT = "no_department"
ARM_CLASS = "other field removed"
ROLE = ("additional non-dominant control; presentational / symmetry-completing "
        "(AUTHOR_DECISIONS_2026-09-21 B 7.1)")
EMB_DIR = OUT_B / "embeddings" / ARM_ID
EMB = EMB_DIR / "merged.npy"
EMB_IDS = EMB_DIR / "merged_ids.json"
EMB_MANIFEST = EMB_DIR / "manifest.json"

SEEDS = list(q47.SEEDS)                               # [42, 1..10]
REFERENCE = q47.STORED
DOMINANT_FIELD = "first_antibiotic"
CONTEXT_FIELD = "BSI_episode_short_clean"
ENCODER_LABEL = "Qwen3-0.6B"                          # naming used by out/33_figures/ablation.csv
DOMINANT_FIELD_LABEL = "first antibiotic"
CONVENTION = "both_nonnoise"
CONVENTION_SOURCE = ("17_common.agreement[ari_noise_excl, ami_noise_excl]; identical to "
                     "47_qwen3_extra_seeds.agree (cross-checked to 1e-9)")
DEGENERATE_K_BELOW = v66.DEGENERATE_K_BELOW           # 5
EXPECTED_DIM = 1024

K_KNN = 15
KNN_EXISTING = REV / "out/45_audit9/knn_purity.json"
KNN_STORED_FULL = REV / "out/18_encoder_dec/embeddings/qwen3-0.6b/merged.npy"
KNN_REPRO_TOL = 5e-4
THIS_FILES = ["scripts/65b_B3_embed_qwen3.py", "slurm/65b_B3_embed.sh",
              "scripts/66b_B3_cluster.py", "slurm/66b_B3_cluster.sh"]


def git_info() -> dict:
    def run(*args) -> str:
        return subprocess.run(["git", "-C", str(REV), *args], capture_output=True,
                              text=True, check=True).stdout.strip()

    dirty = run("status", "--porcelain", "--", *THIS_FILES)
    return {"git_commit": run("rev-parse", "HEAD"), "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "b3_scripts_dirty": bool(dirty), "b3_scripts_dirty_detail": dirty}


def env_record() -> dict:
    return {
        "sys_executable": sys.executable,
        "versions": v65.package_versions(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "thread_pins": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        **git_info(),
    }


def label_path(seed: int) -> Path:
    return LABELS / f"{ARM_ID}__seed{seed}.npy"


def load_reference():
    payload = np.load(REFERENCE, allow_pickle=True)
    ref_ids = [int(x) for x in payload["admission_ID"]]
    ids = [int(x) for x in q47.load_ids()]
    if ref_ids != ids:                                # asserted, not reindexed (as 47:157-159)
        raise RuntimeError("id order differs between the stored Qwen3 partition and the stored Qwen3 ids")
    return payload["labels"].astype(np.int64), ids


def load_embedding(ids):
    for p in (EMB, EMB_IDS, EMB_MANIFEST):
        if not p.exists():
            raise SystemExit(f"{p} missing -- run slurm/65b_B3_embed.sh first")
    emb_ids = [int(x) for x in json.loads(EMB_IDS.read_text())]
    if emb_ids != ids:
        raise RuntimeError("id order differs between the B3 embedding and the stored Qwen3 partition")
    man = json.loads(EMB_MANIFEST.read_text())
    if man.get("n_truncated") != 0 or man.get("frac_truncated") != 0.0:
        raise SystemExit(f"B3 embedding manifest reports truncation: {man.get('n_truncated')}")
    x = np.load(EMB)
    if x.shape != (len(ids), EXPECTED_DIM) or not np.isfinite(x).all():
        raise SystemExit(f"unexpected B3 embedding {x.shape}")
    return x, man


# --------------------------------------------------------------------------- cluster
def stage_cluster(seed: int) -> None:
    reference, ids = load_reference()
    n = len(ids)
    x, man = load_embedding(ids)
    LABELS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    started = v65.now_iso()
    dest = label_path(seed)
    run = {
        "stage": "cluster", "arm_id": ARM_ID, "variant": VARIANT, "role": ROLE,
        "encoder": "Qwen/Qwen3-Embedding-0.6B", "n": n, "seed": seed, "seeds_full_set": SEEDS,
        "umap_kwargs": dict(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean",
                            random_state=seed),
        "hdbscan_kwargs": dict(min_cluster_size=max(10, n // 200), min_samples=None,
                               metric="euclidean", cluster_selection_method="eom"),
        "fit_function": "47_qwen3_extra_seeds.fit (imported)",
        "embedding_path": str(EMB), "embedding_shape": list(x.shape), "embedding_dtype": str(x.dtype),
        "embedding_sha256": v65.sha256_file(EMB), "embedding_manifest": str(EMB_MANIFEST),
        "embedding_n_tokens_median": man.get("n_tokens_median"), "embedding_n_tokens_max": man.get("n_tokens_max"),
        "embedding_n_truncated": man.get("n_truncated"),
        "reference_path": str(REFERENCE), "reference_sha256": v65.sha256_file(REFERENCE),
        "reference_k": v66.n_clusters(reference), "reference_noise": int((reference == -1).sum()),
        "started": started, **env_record(),
    }
    if dest.exists():
        lab = np.load(dest)
        if lab.shape != (n,):
            raise SystemExit(f"{dest}: cached labels have shape {lab.shape}")
        run["fit"] = {"cached": True}
        print(f"[66b|{seed}] cached k={v66.n_clusters(lab)}", flush=True)
    else:
        t0 = time.time()
        lab = q47.fit(x, seed)
        tmp = dest.with_name(dest.name + ".tmp.npy")   # np.save appends .npy otherwise
        np.save(tmp, lab)
        tmp.replace(dest)
        run["fit"] = {"cached": False, "seconds": round(time.time() - t0, 1)}
        print(f"[66b|{seed}] k={v66.n_clusters(lab)} noise={(lab == -1).mean():.3f} "
              f"({run['fit']['seconds']}s)", flush=True)
    run["fit"].update({"k": v66.n_clusters(lab), "noise_fraction": float((lab == -1).mean())})
    ag = c17.agreement(reference, lab)     # numbers only, for the log; scoring happens in `aggregate`
    run["vs_reference"] = {"ari_both_nonnoise": ag["ari_noise_excl"], "ami_both_nonnoise": ag["ami_noise_excl"],
                           "agreement_convention": CONVENTION}
    run["finished"] = v65.now_iso()
    out = RUNS / f"{ARM_ID}__cluster_seed{seed}__{v66.stamp()}_{run['slurm_job_id'] or 'nojob'}.json"
    c17.atomic_write_json(run, out)
    print(f"[66b|{seed}] wrote {out}", flush=True)


# --------------------------------------------------------------------------- kNN readout
def knn_purity(x: np.ndarray, fields: dict) -> dict:
    """audit9_knn_purity.py, restated line for line: rows L2-normalised (float32), brute-force cosine
    similarity in 2048-row chunks, self excluded with -inf, top-K by argpartition, purity = mean over
    rows of the share of the K neighbours whose field value equals the row's."""
    x = x.astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    n = len(x)
    nn = np.empty((n, K_KNN), dtype=np.int64)
    for s in range(0, n, 2048):
        sims = x[s:s + 2048] @ x.T
        sims[np.arange(sims.shape[0]), np.arange(s, s + sims.shape[0])] = -np.inf
        part = np.argpartition(-sims, K_KNN, axis=1)[:, :K_KNN]
        nn[s:s + 2048] = part
    return {f: float((v[nn] == v[:, None]).mean(axis=1).mean()) for f, v in fields.items()}


def stage_knn(ids) -> dict:
    ab1, org = q47.load_fields(ids)                   # first-antibiotic value / index organism, stored row order
    fields = {"first_antibiotic": ab1, "organism": org}
    # chance: audit9 draws its permutations from one default_rng(0) in the order
    # first_antibiotic, organism, department -- the first two are reproduced exactly.
    rng = np.random.default_rng(0)
    chance = {f: float(np.mean(v[rng.permutation(len(v))] == v)) for f, v in fields.items()}
    existing = json.loads(KNN_EXISTING.read_text())
    chance_diff = {f: abs(chance[f] - existing["chance_match_rate"][f]) for f in fields}

    x_full = np.load(KNN_STORED_FULL)
    repro = knn_purity(x_full, fields)
    repro_diff = {f: abs(repro[f] - existing["arms"]["stored_full"][f"purity_{f}"]) for f in fields}
    repro_ok = (existing["k"] == K_KNN and max(chance_diff.values()) < 1e-12
                and max(repro_diff.values()) < KNN_REPRO_TOL)
    del x_full
    x, _ = load_embedding(ids)
    b3 = knn_purity(x, fields)
    return {"k": K_KNN, "metric": "cosine (L2-normalised dot product), self excluded, brute force",
            "definition_source": "scripts/audit9_knn_purity.py (restated; verified below)",
            "chance_match_rate": chance, "chance_abs_diff_vs_existing": chance_diff,
            "existing_json": str(KNN_EXISTING),
            "reproduction_of_existing_stored_full": {"recomputed": repro, "abs_diff_vs_existing": repro_diff,
                                                     "tolerance": KNN_REPRO_TOL, "reproduces": repro_ok},
            "b3": b3}


# --------------------------------------------------------------------------- aggregate
def stage_aggregate() -> None:
    import pandas as pd

    v40 = importlib.import_module("40_field_dominance")
    reference, ids = load_reference()
    n = len(ids)
    x_probe, man = load_embedding(ids)                # existence / order / truncation / shape checks
    del x_probe

    missing = [s for s in SEEDS if not label_path(s).exists()]
    if missing:
        raise SystemExit(f"missing {len(missing)} fits, seeds {missing}")

    frame = v40.load_korea_fields()
    frame = frame.set_index("admission_ID").loc[ids].reset_index()
    fields = v40.build_fields(frame, v40.KOREA_CATEGORICAL, v40.KOREA_CONTINUOUS)
    for f in (DOMINANT_FIELD, CONTEXT_FIELD):
        if f not in fields:
            raise SystemExit(f"field {f} not available")
    ab1, org = q47.load_fields(ids)
    removed, n_docs = v66.edit_sizes()[VARIANT]

    git = git_info()
    rows, field_rows, cross_rows, per_fit_scores = [], [], [], []
    labs = {s: np.load(label_path(s)) for s in SEEDS}
    max_diff = 0.0
    for seed in SEEDS:
        lab = labs[seed]
        if lab.shape != (n,):
            raise SystemExit(f"seed {seed}: labels shape {lab.shape}")
        ag = c17.agreement(reference, lab)
        ari, ami = ag["ari_noise_excl"], ag["ami_noise_excl"]
        a47 = q47.agree(lab, reference)
        diff = max(abs(a47[0] - ari), abs(a47[1] - ami))
        max_diff = max(max_diff, diff)
        if diff > 1e-9:
            raise SystemExit(f"seed {seed}: 17_common.agreement and 47.agree disagree by {diff}")
        scores = v40.field_ami(lab, fields)
        d_fa = abs(scores[DOMINANT_FIELD] - q47.field_ami(lab, ab1))
        d_org = abs(scores[CONTEXT_FIELD] - q47.field_ami(lab, org))
        max_diff = max(max_diff, d_fa, d_org)
        if max(d_fa, d_org) > 1e-9:
            raise SystemExit(f"seed {seed}: 40.field_ami and 47.field_ami disagree by {max(d_fa, d_org)}")
        per_fit_scores.append(scores)
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        (top, top_ami), (runner, run_ami) = ordered[0], ordered[1]
        k = v66.n_clusters(lab)
        rows.append({
            "analysis": "B", "encoder": ENCODER_LABEL, "dominant_field": DOMINANT_FIELD_LABEL,
            "arm_id": ARM_ID, "variant": VARIANT, "arm": ARM_CLASS, "role": ROLE, "seed": seed, "n": n,
            "k": k, "noise_fraction": float((lab == -1).mean()),
            "largest_cluster_share": v66.largest_share(lab),
            "degenerate_k_lt_5": bool(k < DEGENERATE_K_BELOW),
            "agreement_convention": CONVENTION,
            "ari_vs_reference": ari, "ami_vs_reference": ami, "n_both_nonnoise": ag["n_both_nonnoise"],
            "reference": "out/18_encoder_dec/recluster/qwen3-0.6b/labels.npz",
            "reference_k": v66.n_clusters(reference), "reference_noise": int((reference == -1).sum()),
            "dominant_field_name": DOMINANT_FIELD, "dominant_field_ami": scores[DOMINANT_FIELD],
            "organism_field_name": CONTEXT_FIELD, "organism_ami": scores[CONTEXT_FIELD],
            "top_field": top, "top_field_ami": top_ami, "runner_up_field": runner,
            "dominance_margin": top_ami - run_ami,
            "edit_lines_removed_total": removed, "edit_lines_removed_per_doc": removed / n_docs,
            "embedding_source": str(EMB.relative_to(REV)), "git_commit": git["git_commit"],
        })
        for fname, val in scores.items():
            field_rows.append({"arm_id": ARM_ID, "seed": seed, "field": fname, "ami": val})
    cs = []
    for a, b in itertools.combinations(SEEDS, 2):
        v = c17.agreement(labs[a], labs[b])["ari_noise_excl"]
        cs.append(v)
        cross_rows.append({"arm_id": ARM_ID, "seed_a": a, "seed_b": b,
                           "agreement_convention": CONVENTION, "ari_cross_seed": v})

    OUT_B.mkdir(parents=True, exist_ok=True)
    c17.atomic_write_csv(pd.DataFrame(rows), MATRIX_B3)
    c17.atomic_write_csv(pd.DataFrame(field_rows), FIELD_B3)
    c17.atomic_write_csv(pd.DataFrame(cross_rows), CROSS_B3)

    per_arm = {
        "n_seeds": len(SEEDS),
        "ari_vs_reference": v66.summarise([r["ari_vs_reference"] for r in rows]),
        "ami_vs_reference": v66.summarise([r["ami_vs_reference"] for r in rows]),
        "dominant_field_ami": v66.summarise([r["dominant_field_ami"] for r in rows]),
        "organism_ami": v66.summarise([r["organism_ami"] for r in rows]),
        "k": v66.summarise([r["k"] for r in rows]),
        "noise_fraction": v66.summarise([r["noise_fraction"] for r in rows]),
        "largest_cluster_share": v66.summarise([r["largest_cluster_share"] for r in rows]),
        "n_degenerate_fits": int(sum(r["degenerate_k_lt_5"] for r in rows)),
        "cross_seed_ari": {**v66.summarise(cs), "n_pairs": len(cs)},
        "dominance_summary_40": v40.summarise_dominance(per_fit_scores),
        "edit_lines_removed_per_doc": removed / n_docs,
    }
    print(f"[66b/aggregate] ARI median {per_arm['ari_vs_reference']['median']:+.4f} "
          f"[{per_arm['ari_vs_reference']['min']:+.4f}, {per_arm['ari_vs_reference']['max']:+.4f}] "
          f"(n={len(SEEDS)}, {CONVENTION})", flush=True)

    knn = stage_knn(ids)
    knn_rows = [{"arm_id": ARM_ID, "encoder": ENCODER_LABEL, "k": K_KNN, "field": f,
                 "purity": knn["b3"][f], "chance_match_rate": knn["chance_match_rate"][f],
                 "definition_source": "audit9_knn_purity.py (restated; reproduces existing stored_full: "
                                      f"{knn['reproduction_of_existing_stored_full']['reproduces']})"}
                for f in ("first_antibiotic", "organism")]
    c17.atomic_write_csv(pd.DataFrame(knn_rows), KNN_B3)
    print(f"[66b/aggregate] 15-NN purity B3: {knn['b3']}; chance {knn['chance_match_rate']}; "
          f"existing definition reproduced: {knn['reproduction_of_existing_stored_full']['reproduces']}", flush=True)

    RUNS.mkdir(parents=True, exist_ok=True)
    run = {
        "stage": "aggregate", "arm_id": ARM_ID, "seeds": SEEDS, "n": n,
        "agreement_convention": CONVENTION, "agreement_source": CONVENTION_SOURCE,
        "max_abs_diff_cross_checks": max_diff,
        "dominant_field": DOMINANT_FIELD, "context_field": CONTEXT_FIELD,
        "degenerate_rule": f"k < {DEGENERATE_K_BELOW}",
        "n_fields_scored": len(fields), "fields_scored": list(fields),
        "reference_path": str(REFERENCE), "reference_sha256": v65.sha256_file(REFERENCE),
        "reference_k": v66.n_clusters(reference), "reference_noise": int((reference == -1).sum()),
        "embedding_path": str(EMB), "embedding_sha256": v65.sha256_file(EMB),
        "embedding_manifest": {k: man.get(k) for k in (
            "model_load_path", "model_provenance", "pooling", "normalize", "device", "device_name",
            "dtype_out", "batch_size", "n_docs", "dim", "docs_per_sec", "enc_sec", "n_tokens_median",
            "n_tokens_max", "n_truncated", "frac_truncated", "max_seq_length", "repro_check_vs_existing_full_arm",
            "narrative_hash_manifest_sha256", "git_commit", "slurm_job_id")},
        "per_arm": {ARM_ID: per_arm}, "knn_readout": knn,
        "outputs": {"perturbation_matrix_B3": str(MATRIX_B3), "cross_seed_ari_B3": str(CROSS_B3),
                    "field_ami_per_fit_B3": str(FIELD_B3), "knn_readout_B3": str(KNN_B3)},
        "note": "perturbation_matrix.csv, cross_seed_ari.csv and field_ami_per_fit.csv (B1/B2) are not touched.",
        "written": v65.now_iso(), **env_record(),
    }
    dest = RUNS / f"aggregate_B3__{v66.stamp()}_{run['slurm_job_id'] or 'nojob'}.json"
    c17.atomic_write_json(run, dest)
    print(f"[66b/aggregate] wrote {MATRIX_B3} ({len(rows)} rows) and {dest}", flush=True)
    if not knn["reproduction_of_existing_stored_full"]["reproduces"]:
        raise SystemExit("15-NN restatement did NOT reproduce the existing stored_full value / chance rates "
                         "-- knn_readout_B3.csv is flagged accordingly; do not use it without investigation")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["cluster", "aggregate"])
    ap.add_argument("--seed-index", type=int, default=None, help="index into [42, 1..10] (SLURM array 0-10)")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    if args.stage == "cluster":
        if (args.seed_index is None) == (args.seed is None):
            raise SystemExit("give exactly one of --seed-index / --seed")
        seed = SEEDS[args.seed_index] if args.seed_index is not None else args.seed
        if seed not in SEEDS:
            raise SystemExit(f"seed {seed} not in the preregistered set {SEEDS}")
        stage_cluster(seed)
    else:
        stage_aggregate()


if __name__ == "__main__":
    main()
