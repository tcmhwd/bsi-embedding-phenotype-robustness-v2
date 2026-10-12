#!/usr/bin/env python3
"""Task 47 -- extend the Qwen3 ablation/masking arms from 4 seeds to 11.

WHY. `out/44_ablation/ablation.json` and `out/45_audit9/mask_ablation.json`
ran seeds [42, 1, 2, 3] only, while the 3-large arms carry 11 seeds
(21 for `no_organism`). Two things changed the cost of that asymmetry:

  1. The Phase 2A audit (MJ-2) promoted the Qwen3 `no_therapy` arm from a
     "collapse" to a FIELD TAKEOVER -- in 3 of 4 seeds the pipeline converges
     on a new, mutually consistent partition aligned with the index organism
     (organism AMI 0.039 -> 0.63; cross-seed ARI 0.976-0.989), while seed 3
     gives a k=9 solution. That is now load-bearing mechanistic evidence,
     and it rests on n=4.

     (RESULT, added after the run: at 11 seeds the takeover holds in ALL
     eleven. Nine seeds give k=17-19 with organism AMI 0.591-0.637; the two
     minority seeds -- 3 and 8, k=9-10 -- are a DIFFERENT BASIN rather than a
     degenerate or merely coarser one, and they show the takeover too, at
     organism AMI 0.420 and 0.521 against 0.040 unmodified. The word
     "near-degenerate" above was the four-seed reading and does not survive.)
  2. A reviewer's most obvious question about the ablation figure is why one
     encoder has 11 (21) seeds and the other has 4.

WHAT THIS DOES. Re-fits every Qwen3 arm at seeds 42 and 1-10 (11 seeds), i.e.
the ORIGINAL four seeds plus seven new ones, reusing the stored variant
embeddings. Nothing is re-embedded. Re-fitting the original four is
deliberate: it makes the 11-seed matrices self-contained and doubles as a
reproduction check on `44_ablation`/`audit9_cluster_mask` (the frozen stack is
thread-independent at a fixed random_state, R-009, so the original seeds must
come back identical -- the aggregate stage asserts this).

DESIGN. SLURM array over the five arms (one task per arm, 11 fits each,
comparable to the 12-fit job 588163 that finished inside 3 h). Each fit's
labels are written to disk before the next starts, so a timeout loses at most
one fit and a rerun resumes. A separate `--stage aggregate` pass then computes
every cross-seed and cross-variant quantity from the saved labels.

METHOD (identical to `44_ablation_cluster.py` / `audit9_cluster_mask.py`, whose
functions this mirrors rather than imports, because those modules run their
own `main()` on import of their constants):
  UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean",
       random_state=seed)
  HDBSCAN(min_cluster_size=max(10, n//200) = 161, min_samples=None,
          metric="euclidean", cluster_selection_method="eom")
  agreement on rows non-noise in BOTH partitions (`both_nonnoise`).

Runs in the FROZEN environment (/home/programs/anaconda3/envs/gpt), the only
stack that reproduces this project's partitions. Threads pinned to 1 before any
numeric import.

Writes only under out/47_qwen3_seeds/.
"""
import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
ROOT = REV.parent
OUT = REV / "out/47_qwen3_seeds"
LABELS = OUT / "labels"

# seeds 42 + 1..10; the first four are the original run (44/45), refit here so
# the 11-seed matrices are self-contained and the originals are re-verified.
SEEDS = [42] + list(range(1, 11))
ORIGINAL_SEEDS = [42, 1, 2, 3]

VARIANTS = {
    "full":               REV / "out/41_audit7/embeddings/ablation_full/merged.npy",
    "no_therapy":         REV / "out/41_audit7/embeddings/ablation_no_therapy/merged.npy",
    "no_organism":        REV / "out/41_audit7/embeddings/ablation_no_organism/merged.npy",
    "mask_ab1_name_only": REV / "out/45_audit9/embeddings/mask_ab1_name_only/merged.npy",
    "mask_ab_names_all":  REV / "out/45_audit9/embeddings/mask_ab_names_all/merged.npy",
}
ARM_ORDER = list(VARIANTS)

STORED = REV / "out/18_encoder_dec/recluster/qwen3-0.6b/labels.npz"
IDS = REV / "out/18_encoder_dec/embeddings/qwen3-0.6b/merged_ids.json"
RAW_FIELDS = REV / "out/20_onehot/features/raw_fields.parquet"

# The values the original 4-seed runs recorded, asserted in the aggregate
# stage so a silent environment drift cannot pass unnoticed.
ORIGINAL_JSONS = {
    "44": REV / "out/44_ablation/ablation.json",
    "45": REV / "out/45_audit9/mask_ablation.json",
}


def fit(x: np.ndarray, seed: int) -> np.ndarray:
    import hdbscan
    import umap
    coords = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                       metric="euclidean", random_state=seed).fit_transform(x)
    return hdbscan.HDBSCAN(
        min_cluster_size=max(10, len(x) // 200), min_samples=None,
        metric="euclidean", cluster_selection_method="eom",
    ).fit_predict(coords).astype(np.int32)


def agree(a: np.ndarray, b: np.ndarray) -> tuple:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    both = (a != -1) & (b != -1)
    return (float(adjusted_rand_score(a[both], b[both])),
            float(adjusted_mutual_info_score(a[both], b[both])))


def field_ami(labels: np.ndarray, values: np.ndarray) -> float:
    from sklearn.metrics import adjusted_mutual_info_score
    keep = labels != -1
    return float(adjusted_mutual_info_score(labels[keep], values[keep]))


def load_ids():
    return json.loads(IDS.read_text())


def load_fields(ids):
    """first-antibiotic and index-organism values on the stored row order."""
    import pandas as pd
    raw = pd.read_parquet(RAW_FIELDS)[["admission_ID", "tx__admission_antibiotics"]]

    def first_antibiotic(value):
        if isinstance(value, str) and value.startswith("["):
            parsed = json.loads(value)
            return str(parsed[0]) if parsed else "NA"
        return "NA"

    raw["first_antibiotic"] = raw["tx__admission_antibiotics"].apply(first_antibiotic)
    dataset = pd.read_csv(
        ROOT / "manuscript/phenotype_ml_dataset.csv", low_memory=False,
        usecols=["admission_ID", "BSI_episode_short_clean"],
    ).drop_duplicates("admission_ID")
    fields = raw.merge(dataset, on="admission_ID", validate="one_to_one")
    fields = fields.set_index("admission_ID").loc[ids]
    return (fields["first_antibiotic"].astype(str).to_numpy(),
            fields["BSI_episode_short_clean"].astype(str).to_numpy())


def label_path(arm: str, seed: int) -> Path:
    return LABELS / f"{arm}__seed{seed}.npy"


def stage_fit(arm: str) -> None:
    if arm not in VARIANTS:
        raise SystemExit(f"unknown arm {arm!r}; expected one of {ARM_ORDER}")
    LABELS.mkdir(parents=True, exist_ok=True)

    ids = load_ids()
    path = VARIANTS[arm]
    variant_ids = json.loads((path.parent / "merged_ids.json").read_text())
    if variant_ids != ids:
        raise RuntimeError(f"{arm}: id order differs from the stored partition")

    x = np.load(path)
    for seed in SEEDS:
        dest = label_path(arm, seed)
        if dest.exists():
            print(f"[{arm}|{seed}] cached, skipping", flush=True)
            continue
        labels = fit(x, seed)
        # NOTE: the temp name must itself end in ".npy" -- np.save silently
        # APPENDS ".npy" to any path that does not, which would leave the
        # rename target non-existent (observed: job 590044 died on its first
        # fit for exactly this reason).
        tmp = dest.with_name(dest.name + ".tmp.npy")
        np.save(tmp, labels)
        tmp.replace(dest)
        k = int(len(set(labels.tolist()) - {-1}))
        print(f"[{arm}|{seed}] k={k:3d} noise={(labels == -1).mean():.3f}", flush=True)


def stage_aggregate() -> None:
    ids = load_ids()
    antibiotic, organism = load_fields(ids)

    stored_payload = np.load(STORED, allow_pickle=True)
    stored = stored_payload[[k for k in stored_payload.files if "label" in k.lower()][0]]

    missing = [f"{a}|{s}" for a in ARM_ORDER for s in SEEDS if not label_path(a, s).exists()]
    if missing:
        raise SystemExit(f"missing {len(missing)} fits, e.g. {missing[:5]}")

    labels = {(a, s): np.load(label_path(a, s)) for a in ARM_ORDER for s in SEEDS}

    result = {
        "seeds": SEEDS,
        "original_seeds": ORIGINAL_SEEDS,
        "arms": ARM_ORDER,
        "method": ("UMAP nc2/nn15/md0.1/euclidean/random_state=seed -> HDBSCAN "
                   "mcs=max(10,n//200)=161, ms=None, eom; frozen env; threads=1; "
                   "agreement on both_nonnoise rows"),
        "per_fit": {},
        "vs_stored": {},
        "cross_seed": {},
        "reproduction_check": {},
    }

    for arm in ARM_ORDER:
        for seed in SEEDS:
            lab = labels[(arm, seed)]
            ari, ami = agree(lab, stored)
            result["per_fit"][f"{arm}|{seed}"] = {
                "k": int(len(set(lab.tolist()) - {-1})),
                "noise": float((lab == -1).mean()),
                "ari_vs_stored": ari,
                "ami_vs_stored": ami,
                "ami_first_antibiotic": field_ami(lab, antibiotic),
                "ami_organism": field_ami(lab, organism),
            }

        vals = [result["per_fit"][f"{arm}|{s}"]["ari_vs_stored"] for s in SEEDS]
        org = [result["per_fit"][f"{arm}|{s}"]["ami_organism"] for s in SEEDS]
        ab = [result["per_fit"][f"{arm}|{s}"]["ami_first_antibiotic"] for s in SEEDS]
        result["vs_stored"][arm] = {
            "n_seeds": len(SEEDS),
            "ari_median": float(np.median(vals)),
            "ari_min": float(np.min(vals)),
            "ari_max": float(np.max(vals)),
            "ami_organism_median": float(np.median(org)),
            "ami_organism_min": float(np.min(org)),
            "ami_organism_max": float(np.max(org)),
            "ami_first_antibiotic_median": float(np.median(ab)),
            "ami_first_antibiotic_min": float(np.min(ab)),
            "ami_first_antibiotic_max": float(np.max(ab)),
        }

        pairs = {}
        for s1, s2 in itertools.combinations(SEEDS, 2):
            pairs[f"{s1}-{s2}"] = agree(labels[(arm, s1)], labels[(arm, s2)])[0]
        pv = list(pairs.values())
        result["cross_seed"][arm] = {
            "n_pairs": len(pv),
            "ari_median": float(np.median(pv)),
            "ari_min": float(np.min(pv)),
            "ari_max": float(np.max(pv)),
            "pairs": pairs,
        }

    # Reproduction check against the original 4-seed runs. The frozen stack is
    # thread-independent at a fixed random_state (R-009), so these must match.
    for tag, path in ORIGINAL_JSONS.items():
        if not path.exists():
            result["reproduction_check"][tag] = {"status": "source json absent"}
            continue
        prev = json.loads(path.read_text()).get("per_fit", {})
        checks = {}
        for key, old in prev.items():
            if key not in result["per_fit"]:
                continue
            new = result["per_fit"][key]
            checks[key] = {
                "k_old": old.get("k"), "k_new": new["k"],
                "ari_old": old.get("ari_vs_stored"), "ari_new": new["ari_vs_stored"],
                "k_match": old.get("k") == new["k"],
                "ari_close": (old.get("ari_vs_stored") is not None
                              and abs(old["ari_vs_stored"] - new["ari_vs_stored"]) < 1e-9),
            }
        result["reproduction_check"][tag] = {
            "n_compared": len(checks),
            "n_k_mismatch": sum(1 for v in checks.values() if not v["k_match"]),
            "n_ari_mismatch": sum(1 for v in checks.values() if not v["ari_close"]),
            "detail": checks,
        }

    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "qwen3_extra_seeds.json"
    dest.write_text(json.dumps(result, indent=1, sort_keys=True))
    print(f"[47] wrote {dest}", flush=True)

    for arm in ARM_ORDER:
        v = result["vs_stored"][arm]
        c = result["cross_seed"][arm]
        print(f"  {arm:19s} ARIvsStored median {v['ari_median']:+.4f} "
              f"[{v['ari_min']:+.4f}, {v['ari_max']:+.4f}]  "
              f"cross-seed median {c['ari_median']:.4f} [{c['ari_min']:.4f}, {c['ari_max']:.4f}]  "
              f"AMI_org median {v['ami_organism_median']:.4f}", flush=True)
    for tag, chk in result["reproduction_check"].items():
        if "n_compared" in chk:
            print(f"  reproduction vs {tag}: {chk['n_compared']} fits compared, "
                  f"{chk['n_k_mismatch']} k mismatch, {chk['n_ari_mismatch']} ARI mismatch", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["fit", "aggregate"], required=True)
    ap.add_argument("--arm", default=None, help="arm name, or an integer array index 0-4")
    args = ap.parse_args()

    if args.stage == "aggregate":
        stage_aggregate()
        return

    arm = args.arm
    if arm is None:
        raise SystemExit("--stage fit requires --arm")
    if arm.isdigit():
        arm = ARM_ORDER[int(arm)]
    stage_fit(arm)


if __name__ == "__main__":
    main()
