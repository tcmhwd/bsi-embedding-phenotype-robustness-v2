#!/usr/bin/env python3
"""Task 22, Task A -- shared utilities for the stress-test battery applied to
the PUBLISHED E. coli subphenotype object (manuscript Table A.2 "E. coli
subset" row; Table A.11 / Fig A.4 / Fig A.5), never tested against the
PRESPEC battery before this task.

## Parameters as found in the manuscript (verbatim from Table A.2)

    UMAP:    n_neighbors=30, min_dist=0.05, n_components=2, metric='cosine',
             random_state=42
    HDBSCAN: min_cluster_size=300, min_samples=50,
             cluster_selection_method='eom', metric='euclidean'

(Source: the authors' earlier supplementary material (Table A.2),
extracted verbatim 2026-09-05 -- see task report for the extraction method.
HDBSCAN's own `metric='euclidean'` is applied on the 2-D UMAP output space,
identical convention to the whole-cohort object in `17_common.py`.)

## n and cluster-count discrepancy -- resolved, reported here

The task brief quoted n=9,595 / 10 clusters for this object. Table A.2 itself
gives parameters only, no n. Table A.11 (the manuscript's own per-cluster
characteristics table for these subphenotypes) sums to n=8,268 across 10
non-noise clusters + noise -- which is exactly the cohort's monomicrobial
"Escherichia coli" count (`BSI_episode_short_clean == "Ecoli"`, Table A.3,
also this repo's own established E. coli stratum in `08_transport.py`
--organism-filter, `10_within_pathogen.py`, `11_rarefaction.py`).

However, `pipeline_new/out_pre/ecoli_case_report_embeddings.npy` (9,595 x
3,072) + `ecoli_umap_hdbscan_labels.csv` (column `ecoli_cluster`) is an
on-disk ORIGINAL PIPELINE ARTIFACT -- not something built for this task --
whose cluster-size distribution (1593,1580,1113,960,865,645,620,500,411,397
non-noise + 911 noise = 9,595 exactly) is the one that matches the task
brief's n=9,595/10-clusters figure exactly. Its `ecoli_hdbscan_summary.csv`
companion file is the per-cluster descriptive summary in the same style as
`phenotype_clusters_summary.csv` for the whole cohort. This is therefore
almost certainly the actual UMAP+HDBSCAN fit that produced Fig A.4/Fig A.5,
run on a broader E. coli selection than the further-restricted n=8,268 used
for Table A.11's characteristics table (the exact upstream selection rule
for the 9,595 is not documented in the manuscript text; row-level provenance
past this artifact was not re-derived, since re-deriving it is not necessary
to run the stress battery on the object that actually produced the figures).

Conclusion adopted here (report, not silently substitute): use n=9,595 /
10 clusters (`ecoli_case_report_embeddings.npy` + `ecoli_cluster`) as THE
anchor object under test, because it is the actual on-disk fit artifact
underlying the published E. coli figures, and it is the one that matches
the task brief's stated n/k. The n=8,268 Table-A.11 alternative is reported
alongside as a fact, not adopted as the object.

Outcome-blind (Global Constraints): no mortality/discharge/death column is
read anywhere in this module (30d/90d/in_hospital_mortality, death_date are
present in the source CSV but never selected into any frame built here).
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/tcmhwd/AI/AST_ai")
REV = ROOT / "phenotype_revision"
ECOLI_EMB = ROOT / "pipeline_new/out_pre/ecoli_case_report_embeddings.npy"
ECOLI_LABELS = ROOT / "pipeline_new/out_pre/ecoli_umap_hdbscan_labels.csv"
OUT = REV / "out/22_published_objects/task_a_ecoli"

REFERENCE_CLUSTER_COL = "ecoli_cluster"
N_ECOLI = 9_595

# Table A.2, "E. coli subset" row, verbatim. min_samples is NOT included in
# HDBSCAN_KW (kept as the separate MIN_CLUSTER_SIZE/MIN_SAMPLES constants
# instead) because every call site needs to pass a *scaled* min_cluster_size/
# min_samples for S2/S4/S5 -- folding min_samples into HDBSCAN_KW would force
# every caller to pop it back out first to override it.
UMAP_KW = dict(n_components=2, n_neighbors=30, min_dist=0.05, metric="cosine")
HDBSCAN_KW = dict(metric="euclidean", cluster_selection_method="eom")
MIN_CLUSTER_SIZE = 300
MIN_SAMPLES = 50

# ---------------------------------------------------------------------------
# S2 bootstrap constants (PRESPEC S2: f=0.632, n_boot=200, clustering seed
# fixed at 42; mcs/min_samples scaled by f -- same rule as 17_common.py /
# 03_stability.scale_threshold, applied here to the E. coli config's own
# mcs=300/min_samples=50 rather than the whole-cohort 161/161).
# ---------------------------------------------------------------------------
F_SUBSAMPLE = 0.632
N_BOOT = 200
CLUSTER_SEED = 42


def scale_threshold(value_full: int, f: float) -> int:
    return max(2, round(f * value_full))


NEEDED_COLS = [
    "admission_ID", "patient_ID", "hospital_code", "test_date",
    "treatment_department_eng", "BSI_episode_short_clean", REFERENCE_CLUSTER_COL,
]


def load_reference_frame() -> pd.DataFrame:
    """`ecoli_umap_hdbscan_labels.csv`, row order == `ECOLI_EMB`'s row order
    (positionally verified 2026-09-05: `ecoli_case_report_embeddings.npy[i]`
    is bit-identical to `case_report_embeddings.npy[embedding_index[i]]` for
    every spot-checked row)."""
    df = pd.read_csv(ECOLI_LABELS, usecols=NEEDED_COLS, low_memory=False)
    df["admit_year"] = pd.to_datetime(df["test_date"], errors="coerce").dt.year
    df = df.drop(columns=["test_date"]).reset_index(drop=True)
    return df


def load_embeddings() -> np.ndarray:
    return np.load(ECOLI_EMB)


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


def stratum_mask(frame: pd.DataFrame, axis: str, value) -> np.ndarray:
    if axis == "hospital":
        return (frame["hospital_code"] == value).to_numpy()
    if axis == "period":
        if value == "early":
            return (frame["admit_year"] <= EARLY_YEAR_CUTOFF).to_numpy()
        if value == "late":
            return (frame["admit_year"] > EARLY_YEAR_CUTOFF).to_numpy()
    raise ValueError(f"unknown axis/value: {axis}/{value}")


def patient_split(patient_ids, n_source: int, seed: int):
    """Random patient-level split into (source, rest); identical algorithm to
    `17_common.patient_split` / `14_a0_remeasure.patient_split` (greedy
    seeded draw, skip-if-would-overshoot, no patient split across sides)."""
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


def mcs_for_stratum(n_source: int, n_full: int = N_ECOLI, base_mcs: int = MIN_CLUSTER_SIZE,
                     floor: int = 50) -> int:
    """S4/S5 mcs scaling for a hospital/era sub-stratum of the E. coli object,
    generalising PRESPEC's own A0 rule ("mcs = 161*n_src/32,329, floor 50")
    to this object's base mcs=300 / n_full=9,595. Not a new methodological
    choice -- same proportional-scaling-with-floor rule PRESPEC already
    specifies for S4 on A0, applied to this object's own base mcs/n_full."""
    return max(floor, round(base_mcs * n_source / n_full))


def patient_subsample_indices(patient_ids, f: float, seed: int) -> np.ndarray:
    """Identical algorithm to `17_common.patient_subsample_indices` /
    `03_stability.patient_subsample_indices`."""
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
    """PRESPEC/DESIGN S2.5 five-way classification, identical rule to
    `17_common.classify_outcome` / `03_stability.classify_outcome`."""
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
    each explicitly labelled by convention (project gotcha: this project has
    mixed the two conventions before and reversed an ordering as a result)."""
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


# ---------------------------------------------------------------------------
# PRESPEC thresholds (S1/S2/S4/S5 only -- S3/S6 are out of scope for Task A
# per the task brief). Read from `prespec_thresholds.json`, a single JSON
# block shared with `20d_task2_extended_lookup.py`'s S6 verdict, so a
# reviewer can see that one file governs every gate (AUDIT_3 N9 /
# AUDIT_4 M6) -- NOT hand-copied into this module any more. The
# file's own values are still the ones copied verbatim from
# `.superpowers/sdd/PRESPEC_stress_criteria.md` section 3, 2026-09-05; this
# extraction changed no threshold value (verified: 22e_ecoli_aggregate.py's
# survival_matrix_s1s2s4s5.csv / SUMMARY.json are byte-identical before and
# after, aside from key order).
# ---------------------------------------------------------------------------
_THRESHOLDS_PATH = Path(__file__).resolve().parent / "prespec_thresholds.json"
with open(_THRESHOLDS_PATH) as _f:
    _ALL_THRESHOLDS = json.load(_f)
THRESHOLDS = {k: _ALL_THRESHOLDS[k] for k in ("S1", "S2", "S4", "S5")}
print(f"[22a_ecoli_common] thresholds loaded from {_THRESHOLDS_PATH}: {json.dumps(THRESHOLDS)}", flush=True)
