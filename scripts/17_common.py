#!/usr/bin/env python3
"""Task 17: shared utilities for the stress-test battery run against the
ACTUAL published phenotypes (16 clusters + noise), on the contemporaneous
`out_pre` embeddings, in the corrected (frozen, `/home/programs/anaconda3/envs/gpt`)
UMAP/HDBSCAN stack that reproduces them bit-for-bit
(`out/16_c1_sweep/result_original_env_seed42.json`: ARI=1.0, k=16, n_noise=1792
both all-row and noise-excluded).

Deliberately depends on nothing beyond numpy/pandas/sklearn.metrics -- no
umap/hdbscan import here -- so this module is safe to `importlib` from BOTH
the frozen clustering interpreter and the bsi_pheno metrics interpreter.

Does NOT reuse `01_build_analysis_frame.load_embeddings` (that loads the
WRONG corpus, `pipeline_new/out/embeddings_shards`) or the stored
`phenotype_cluster` column via that module's frame (which is built from the
same wrong-corpus embedding_index, not from `out_pre`). Instead this module
loads `out_pre/phenotype_clusters_labels.csv` and
`out_pre/case_report_embeddings.npy` directly -- their row order is
POSITIONALLY identical (verified: the frozen-stack refit at seed 42
reproduces the `phenotype_cluster` column bit-for-bit with no join needed).
Patient/hospital/department/year/organism metadata is separately joined
onto that fixed row order by `admission_ID` (never by row position) from
`manuscript/phenotype_ml_dataset.csv`.

Algorithm reuse (mirrored, not imported, since the originals are wired to
the wrong-corpus frame): `patient_subsample_indices`/`scale_threshold`/
`classify_outcome`/`_pairwise_coassignment` from `03_stability.py`;
`patient_split`/`a0_min_cluster_size` from `14_a0_remeasure.py`. Where the
original IS corpus-agnostic and safe to import directly (`02_recluster.run_config`,
`08_transport._stratum_mask`/`transport_predict`/`unassigned_fraction`/
`best_whole_cohort_cluster`), the frozen-env scripts import it directly via
importlib -- see `17c_s4s5_transport.py`.

Outcome-blind (Global Constraints): no mortality/discharge/death column is
read anywhere in this module.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/tcmhwd/AI/AST_ai")
REV = ROOT / "phenotype_revision"
OUT_PRE_LABELS = ROOT / "pipeline_new/out_pre/phenotype_clusters_labels.csv"
OUT_PRE_EMB = ROOT / "pipeline_new/out_pre/case_report_embeddings.npy"
DATASET = ROOT / "manuscript/phenotype_ml_dataset.csv"
OUT = REV / "out/17_correct_env"

PUBLISHED_CLUSTER_COL = "phenotype_cluster"

# Code-of-record configuration (`pipeline_new/phenotype_umap_hdbscan.py`,
# also the authors' earlier supplementary material (Table S2)).
UMAP_KW = dict(n_components=2, n_neighbors=15, min_dist=0.1, metric="euclidean")
HDBSCAN_KW = dict(min_samples=None, metric="euclidean", cluster_selection_method="eom")
N_FULL = 32_329


def min_cluster_size_for(n: int) -> int:
    return max(10, n // 200)


# NOTE: thread pinning (OMP/NUMBA/MKL/OPENBLAS_NUM_THREADS) is deliberately
# NOT a function in this module -- it must execute before numpy/pandas are
# imported (this module imports both at module scope), so every entry
# script pins threads itself, in its own first lines, before `import
# importlib.util` even loads this file. See any `17[abc]_*.py` header.


def load_reference_frame() -> pd.DataFrame:
    """`out_pre` labels CSV in its own row order == `OUT_PRE_EMB`'s row
    order (positionally verified, not merged)."""
    df = pd.read_csv(OUT_PRE_LABELS, usecols=["admission_ID", "newpatient_ID", PUBLISHED_CLUSTER_COL])
    return df.reset_index(drop=True)


def load_embeddings() -> np.ndarray:
    return np.load(OUT_PRE_EMB)


def load_metadata_aligned(admission_ids) -> pd.DataFrame:
    """patient_ID / hospital_code / treatment_department_eng / admit_year /
    BSI_episode_short_clean (organism), joined on `admission_ID` from
    `phenotype_ml_dataset.csv`, then reindexed to exactly match
    `admission_ids`'s order (the `out_pre` row order). Raises if any
    admission_ID is missing (full 32,329/32,329 coverage was verified
    before this module was written)."""
    ds = pd.read_csv(
        DATASET,
        usecols=["admission_ID", "patient_ID", "hospital_code", "treatment_department_eng",
                 "test_date", "BSI_episode_short_clean"],
        low_memory=False,
    )
    ds["admit_year"] = pd.to_datetime(ds["test_date"], errors="coerce").dt.year
    ds = ds.drop(columns=["test_date"]).drop_duplicates(subset="admission_ID")
    admission_ids = np.asarray(admission_ids)
    have = set(ds["admission_ID"].tolist())
    missing = [a for a in admission_ids.tolist() if a not in have]
    if missing:
        raise RuntimeError(f"{len(missing)} admission_ID(s) missing from {DATASET}, e.g. {missing[:5]}")
    aligned = ds.set_index("admission_ID").loc[admission_ids].reset_index()
    if len(aligned) != len(admission_ids):
        raise RuntimeError("metadata alignment length mismatch")
    return aligned


def build_full_frame() -> pd.DataFrame:
    """Reference frame + aligned metadata, one row per `out_pre` row, in
    `out_pre` order. Convenience wrapper used by every stage."""
    ref = load_reference_frame()
    meta = load_metadata_aligned(ref["admission_ID"].to_numpy())
    assert (meta["admission_ID"].to_numpy() == ref["admission_ID"].to_numpy()).all()
    frame = ref.merge(meta.drop(columns=["admission_ID"]), left_index=True, right_index=True)
    return frame


# ---------------------------------------------------------------------------
# S4/S5 transport constants (matches `08_transport.DIRECTIONS`/`EARLY_YEAR_CUTOFF`
# exactly -- hospital_code 10/20, admit_year<=2017 "early" vs >2017 "late").
# ---------------------------------------------------------------------------
EARLY_YEAR_CUTOFF = 2017
DIRECTIONS = [
    {"name": "hospital_10_to_20", "axis": "hospital", "source": 10, "target": 20},
    {"name": "hospital_20_to_10", "axis": "hospital", "source": 20, "target": 10},
    {"name": "period_early_to_late", "axis": "period", "source": "early", "target": "late"},
    {"name": "period_late_to_early", "axis": "period", "source": "late", "target": "early"},
]
N_NULL_PER_DIRECTION = 25
N_DIRECTIONS = len(DIRECTIONS)
N_TRANSPORT_RUNS = N_DIRECTIONS + N_DIRECTIONS * N_NULL_PER_DIRECTION  # 4 + 100 = 104


def resolve_transport_run(run_index: int) -> dict:
    if not (0 <= run_index < N_TRANSPORT_RUNS):
        raise SystemExit(f"--run-index {run_index} out of range [0, {N_TRANSPORT_RUNS - 1}]")
    if run_index < N_DIRECTIONS:
        d = DIRECTIONS[run_index]
        return dict(is_null=False, direction=d, direction_idx=run_index, null_rep=None,
                    tag=f"observed_{d['name']}")
    rem = run_index - N_DIRECTIONS
    direction_idx, null_rep = divmod(rem, N_NULL_PER_DIRECTION)
    d = DIRECTIONS[direction_idx]
    return dict(is_null=True, direction=d, direction_idx=direction_idx, null_rep=null_rep,
                tag=f"null_{d['name']}_seed{null_rep}")


def patient_split(patient_ids, n_source: int, seed: int):
    """Random patient-level split into (source, rest); matches
    `14_a0_remeasure.patient_split` exactly (greedy seeded draw,
    skip-if-would-overshoot, no patient ever split across source/target)."""
    pids = np.asarray(patient_ids)
    uniq, inv, counts = np.unique(pids, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    sel = np.zeros(len(uniq), dtype=bool)
    total = 0
    for pos in order:
        if total + counts[pos] > n_source:
            continue
        sel[pos] = True
        total += counts[pos]
        if total == n_source:
            break
    src = np.where(sel[inv])[0]
    rest = np.where(~sel[inv])[0]
    return src, rest


def a0_min_cluster_size(n_source: int, n_full: int = N_FULL, floor: int = 50) -> int:
    """PRESPEC S4: "mcs = 161*n_src/32,329, floor 50"."""
    return max(floor, round(min_cluster_size_for(n_full) * n_source / n_full))


# ---------------------------------------------------------------------------
# S2 bootstrap (patient-level subsampling, f=0.632) -- mirrors
# `03_stability.patient_subsample_indices`/`scale_threshold`/
# `classify_outcome`/`_pairwise_coassignment` exactly.
# ---------------------------------------------------------------------------
F_SUBSAMPLE = 0.632
N_BOOT = 200
CLUSTER_SEED = 42


def patient_subsample_indices(patient_ids, f: float, seed: int) -> np.ndarray:
    patient_ids = np.asarray(patient_ids)
    n = len(patient_ids)
    target = f * n
    rng = np.random.default_rng(seed)
    unique_patients, inverse, counts = np.unique(patient_ids, return_inverse=True, return_counts=True)
    order = rng.permutation(len(unique_patients))
    selected_mask = np.zeros(len(unique_patients), dtype=bool)
    total = 0
    for pos in order:
        selected_mask[pos] = True
        total += counts[pos]
        if total >= target:
            break
    row_mask = selected_mask[inverse]
    return np.where(row_mask)[0]


def scale_threshold(value_full: int, f: float) -> int:
    return max(2, round(f * value_full))


def jaccard_for_cluster(reference, derived, ref_cluster) -> float:
    a = np.asarray(reference) == ref_cluster
    best = 0.0
    for cand in sorted(set(np.asarray(derived).tolist()) - {-1}):
        b = np.asarray(derived) == cand
        union = np.count_nonzero(a | b)
        if union == 0:
            continue
        j = np.count_nonzero(a & b) / union
        if j > best:
            best = j
    return float(best)


def _pairwise_coassignment(replicate_labels_of_retained_members) -> float:
    labels = np.asarray(replicate_labels_of_retained_members)
    n = len(labels)
    if n < 2:
        return float("nan")
    total_pairs = n * (n - 1) // 2
    non_noise = labels[labels != -1]
    if len(non_noise) < 2:
        return 0.0
    _, counts = np.unique(non_noise, return_counts=True)
    coassigned_pairs = int(np.sum(counts * (counts - 1) // 2))
    return coassigned_pairs / total_pairs


def classify_outcome(base_labels_sub, new_labels, k):
    """DESIGN §2.5 five-way classification, matches `03_stability.classify_outcome`."""
    base_labels_sub = np.asarray(base_labels_sub)
    new_labels = np.asarray(new_labels)
    mask = base_labels_sub == k
    retained_size = int(mask.sum())
    if retained_size == 0:
        return None
    sub_labels = new_labels[mask]
    noise_frac = float(np.mean(sub_labels == -1))
    non_noise = sub_labels[sub_labels != -1]
    if len(non_noise) > 0:
        _, counts = np.unique(non_noise, return_counts=True)
        fracs = counts / retained_size
        best_recall = float(fracs.max())
        n_major = int((fracs >= 0.25).sum())
    else:
        best_recall = 0.0
        n_major = 0
    split = n_major >= 2
    jaccard = jaccard_for_cluster(base_labels_sub, new_labels, k)
    if split:
        outcome = "split"
    elif jaccard >= 0.75:
        outcome = "recovered"
    elif jaccard >= 0.50:
        outcome = "perturbed"
    elif best_recall >= 0.80:
        outcome = "merged"
    else:
        outcome = "dissolved"
    return dict(cluster=int(k), retained_size=retained_size, jaccard=jaccard,
                coassign=_pairwise_coassignment(sub_labels), noise_frac=noise_frac,
                best_recall=best_recall, outcome=outcome)


def agreement(reference, derived) -> dict:
    """ARI/AMI, both all-row (noise as its own label) and noise-excluded,
    each explicitly labelled by convention (project gotcha)."""
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    reference = np.asarray(reference)
    derived = np.asarray(derived)
    both = (reference != -1) & (derived != -1)
    out = dict(
        ari_all_row=float(adjusted_rand_score(reference, derived)),
        ami_all_row=float(adjusted_mutual_info_score(reference, derived)),
        n_both_nonnoise=int(both.sum()),
    )
    if both.sum() > 1:
        out["ari_noise_excl"] = float(adjusted_rand_score(reference[both], derived[both]))
        out["ami_noise_excl"] = float(adjusted_mutual_info_score(reference[both], derived[both]))
    else:
        out["ari_noise_excl"] = float("nan")
        out["ami_noise_excl"] = float("nan")
    return out


def atomic_write_json(obj: dict, dest: Path) -> None:
    import json
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, dest)


def atomic_write_csv(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, dest)
