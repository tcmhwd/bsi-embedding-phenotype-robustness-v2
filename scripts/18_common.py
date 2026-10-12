#!/usr/bin/env python3
"""Task 18 (encoder axis / S6): shared utilities.

Corpus for this task is `reports_pilot_pre/` (32,329 `.report.md`,
2025-12-02 snapshot) -- the preserved copy of the narratives that produced
the published `phenotype_cluster` labels -- NOT `reports_all` (a later,
differently-templated regeneration) and NOT `01_build_analysis_frame`'s
frame (which is wired to `pipeline_new/out/embeddings_shards`, a different
corpus again; see that module's docstring history in Task 17).

Row-order convention (verified in Task 17, `17_common.py`, reused here):
`out_pre/phenotype_clusters_labels.csv` and `out_pre/case_report_embeddings.npy`
are POSITIONALLY identical, and `out_pre/case_report_embeddings_metadata.csv`'s
`embedding_index` column equals its own row index (verified below in
`load_metadata`). This module aligns everything it loads by `admission_ID`
explicitly (via merge), never by bare row position, even though the
underlying files happen to already share one order -- the project's own
stated gotcha ("a patch in this repo once silently ran with a different
config without erroring") is exactly the kind of silent misalignment this
guards against.

Deliberately imports `17_common` (Task 17's module, read-only, not edited
here) for the pieces that are corpus-agnostic: `PUBLISHED_CLUSTER_COL`,
`UMAP_KW`/`HDBSCAN_KW`/`min_cluster_size_for` (code-of-record clustering
config), `jaccard_for_cluster`, `agreement`, `atomic_write_json`/
`atomic_write_csv`. Does NOT import `17_common.OUT` for anything (that
points at `out/17_correct_env/`, read-only for this task) -- this module
defines its own `OUT` under `out/18_encoder_dec/`.

Outcome-blind (Global Constraints): no mortality/discharge/death column is
read anywhere in this module.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/tcmhwd/AI/AST_ai")
REV = ROOT / "phenotype_revision"
OUT = REV / "out" / "18_encoder_dec"

REPORTS_DIR = ROOT / "reports_pilot_pre"
OUT_PRE_LABELS = ROOT / "pipeline_new/out_pre/phenotype_clusters_labels.csv"
OUT_PRE_EMB = ROOT / "pipeline_new/out_pre/case_report_embeddings.npy"
OUT_PRE_META = ROOT / "pipeline_new/out_pre/case_report_embeddings_metadata.csv"
DATASET = ROOT / "manuscript/phenotype_ml_dataset.csv"
SURVIVAL_MATRIX = REV / "out/17_correct_env/survival_matrix.csv"  # read-only, reused verbatim

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c17 = importlib.import_module("17_common")

PUBLISHED_CLUSTER_COL = c17.PUBLISHED_CLUSTER_COL
UMAP_KW = c17.UMAP_KW
HDBSCAN_KW = c17.HDBSCAN_KW
min_cluster_size_for = c17.min_cluster_size_for
jaccard_for_cluster = c17.jaccard_for_cluster
agreement = c17.agreement
atomic_write_json = c17.atomic_write_json
atomic_write_csv = c17.atomic_write_csv

KNN_K = 25
CV_SPLITS = 5
CV_SEED = 42


def load_metadata() -> pd.DataFrame:
    """`admission_ID`, `newpatient_ID`, `embedding_index` -- one row per
    `out_pre` embedding row, `embedding_index` == row index (asserted)."""
    df = pd.read_csv(OUT_PRE_META, usecols=["newpatient_ID", "admission_ID", "embedding_index"])
    if not (df["embedding_index"].to_numpy() == np.arange(len(df))).all():
        raise RuntimeError(f"{OUT_PRE_META}: embedding_index is not the row index -- alignment assumption broken")
    if df["admission_ID"].duplicated().any():
        raise RuntimeError(f"{OUT_PRE_META}: duplicate admission_ID -- cannot align uniquely")
    return df


def report_path_for(newpatient_id: str, admission_id: int) -> Path:
    return REPORTS_DIR / f"{newpatient_id}_{admission_id}.report.md"


def load_texts(admission_ids=None) -> pd.DataFrame:
    """Returns a frame with columns admission_ID, newpatient_ID,
    embedding_index, text -- one row per requested admission_ID (default:
    every row in `out_pre`'s metadata, i.e. the full 32,329-doc corpus),
    in metadata row order. Raises if any report file is missing."""
    meta = load_metadata()
    if admission_ids is not None:
        keep = set(int(a) for a in admission_ids)
        meta = meta[meta["admission_ID"].isin(keep)].reset_index(drop=True)
    texts = []
    missing = []
    for nid, aid in zip(meta["newpatient_ID"], meta["admission_ID"]):
        p = report_path_for(nid, int(aid))
        if not p.exists():
            missing.append(str(p))
            texts.append(None)
            continue
        texts.append(p.read_text(encoding="utf-8", errors="replace"))
    if missing:
        raise FileNotFoundError(f"{len(missing)} report file(s) missing under {REPORTS_DIR}, e.g. {missing[:5]}")
    meta = meta.copy()
    meta["text"] = texts
    return meta


def load_published_frame() -> pd.DataFrame:
    """admission_ID, newpatient_ID, phenotype_cluster, patient_ID (joined
    from `phenotype_ml_dataset.csv` by admission_ID, outcome-blind: only
    patient_ID is read from that file here)."""
    ref = pd.read_csv(OUT_PRE_LABELS, usecols=["admission_ID", "newpatient_ID", PUBLISHED_CLUSTER_COL])
    ds = pd.read_csv(DATASET, usecols=["admission_ID", "patient_ID"], low_memory=False).drop_duplicates(
        subset="admission_ID"
    )
    have = set(ds["admission_ID"].tolist())
    missing = [a for a in ref["admission_ID"].tolist() if a not in have]
    if missing:
        raise RuntimeError(f"{len(missing)} admission_ID(s) in {OUT_PRE_LABELS} missing patient_ID in {DATASET}")
    out = ref.merge(ds, on="admission_ID", how="left")
    if len(out) != len(ref):
        raise RuntimeError("load_published_frame: merge changed row count")
    return out


def align_embeddings_to_admission_ids(vecs: np.ndarray, ids, target_admission_ids) -> np.ndarray:
    """Reindex `vecs` (rows keyed by `ids`, in whatever order they were
    saved) to exactly match `target_admission_ids`'s order, by admission_ID
    -- never by row position. Raises if any target id is missing."""
    ids = np.asarray(ids)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    target = np.asarray(target_admission_ids)
    missing = [int(a) for a in target if a not in pos.index]
    if missing:
        raise RuntimeError(f"{len(missing)} admission_ID(s) missing from embedding ids, e.g. {missing[:5]}")
    if pd.Index(ids).duplicated().any():
        raise RuntimeError("duplicate admission_ID in embedding ids -- alignment would be ambiguous")
    idx = pos.loc[target].to_numpy()
    return vecs[idx]
