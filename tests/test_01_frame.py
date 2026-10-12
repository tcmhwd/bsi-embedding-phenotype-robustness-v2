import os

import numpy as np
import sys
from pathlib import Path
import pytest
SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from importlib import import_module
mod = import_module("01_build_analysis_frame")

SHARD_DIR = "/data/tcmhwd/AI/AST_ai/pipeline_new/out/embeddings_shards"

# This module is an integration test against restricted, patient-level
# clinical data (the full analysis frame and embedding shards), which is
# not distributed with this repository. Skip gracefully when that data is
# not present on the machine running the tests.
pytestmark = pytest.mark.skipif(
    not (Path(mod.DATASET).is_file() and Path(SHARD_DIR).is_dir()),
    reason="requires restricted patient-level data",
)


def test_frame_rows_and_unique_admissions():
    f = mod.load_frame()
    assert len(f) == 32329
    assert f["admission_ID"].is_unique


def test_every_report_file_exists():
    f = mod.load_frame()
    paths = f["report_file"].tolist()
    assert len(paths) == 32329
    missing = [p for p in paths if not os.path.exists(p)]
    assert missing == [], (
        f"{len(missing)} of {len(paths)} report files missing; "
        f"first few: {missing[:5]}"
    )


def test_embeddings_align_with_frame():
    f = mod.load_frame()
    e = mod.load_embeddings(f)
    assert e.shape == (32329, 3072)
    assert e.dtype == np.float32

    # Independent ground truth: read each shard's actual row count from disk
    # (mmap, never mutated) rather than re-deriving it from mod.SHARD_ROWS —
    # otherwise a wrong SHARD_ROWS would shift both the test's expectation
    # and load_embeddings()'s internal computation identically, and this
    # test would never catch it.
    shard_sizes = []
    for shard_id in range(16):
        arr = np.load(f"{SHARD_DIR}/shard_{shard_id:05d}.npy", mmap_mode="r")
        shard_sizes.append(arr.shape[0])

    # Pin the observed on-disk layout as literals: this is what the code
    # (SHARD_ROWS = 2048) assumes, and the test must fail loudly if the
    # shard files ever disagree with that assumption.
    assert shard_sizes == [2048] * 15 + [1609]
    assert sum(shard_sizes) == 32329

    # Build the true index -> (shard, row) mapping from those observed
    # sizes via a cumulative offset walk, independent of mod.SHARD_ROWS.
    index_to_shard_row = {}
    emb_idx = 0
    for shard_id, size in enumerate(shard_sizes):
        for shard_row in range(size):
            index_to_shard_row[emb_idx] = (shard_id, shard_row)
            emb_idx += 1

    # Indices spanning shard 0 (row 0 and a non-zero row), shard 1
    # (boundary row and a non-zero row), shard 15 (start row and a
    # non-zero row), plus the very first and very last rows overall.
    candidate_indices = {0, 1024, 2048, 2053, 30720, 30820, 32328}
    present = set(f["embedding_index"].to_numpy().tolist())
    check_indices = sorted(candidate_indices & present)
    assert len(check_indices) >= 5, (
        "expected embedding_index values spanning shards 0, 1, and 15 to be "
        f"present in the frame; got {check_indices}"
    )
    # Confirm coverage across the required shards.
    checked_shards = {index_to_shard_row[i][0] for i in check_indices}
    assert {0, 1, 15}.issubset(checked_shards)

    shard_cache = {}
    for emb_idx in check_indices:
        shard_id, shard_row = index_to_shard_row[emb_idx]
        if shard_id not in shard_cache:
            shard_cache[shard_id] = np.load(
                f"{SHARD_DIR}/shard_{shard_id:05d}.npy", mmap_mode="r"
            )
        shard = shard_cache[shard_id]
        row = f.index[f["embedding_index"] == emb_idx][0]
        assert np.array_equal(
            e[f.index.get_loc(row)], np.asarray(shard[shard_row])
        ), f"mismatch at embedding_index={emb_idx} (shard {shard_id}, row {shard_row})"


def test_no_outcome_columns_leak_into_frame():
    f = mod.load_frame()
    banned = {"30d_mortality", "90d_mortality", "in_hospital_mortality", "death_date", "discharge_date"}
    assert banned.isdisjoint(set(f.columns))
