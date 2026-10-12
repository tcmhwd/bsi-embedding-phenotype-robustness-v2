#!/usr/bin/env python3
"""Rebuilds the analysis frame and clarifies the unit of analysis (patient vs.
admission).

The manuscript states "32,329 adult patients ... each patient contributed
one index hospitalization", but the actual data's unique patient_ID count
is 27,083. This script quantifies that discrepancy and records it in
unit_of_analysis_report.json. The outcome (mortality) variable is
deliberately not loaded (Global Constraints).
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/tcmhwd/AI/AST_ai")
DATASET = ROOT / "manuscript/phenotype_ml_dataset.csv"
SHARD_DIR = ROOT / "pipeline_new/out/embeddings_shards"
REPORT_DIR = ROOT / "reports_all"
OUT = ROOT / "phenotype_revision/out/01_frame"
SHARD_ROWS = 2048

FRAME_COLS = [
    "admission_ID", "patient_ID", "newpatient_ID", "embedding_index",
    "hospital_code", "test_date", "phenotype_cluster", "report_path",
    "treatment_department_eng", "n_bsi_episodes",
]


def load_frame(verify_reports: bool = True) -> pd.DataFrame:
    """Loads the analysis frame.

    When verify_reports=True (the default), verifies that every path in the
    returned `report_file` column actually exists (an existence-verified
    absolute-path contract). If even one is missing, raises
    FileNotFoundError, so that downstream tasks (2-12) never inherit an
    unverified path. Callers for whom speed matters can skip verification
    with verify_reports=False.
    """
    df = pd.read_csv(DATASET, usecols=lambda c: c in FRAME_COLS, low_memory=False)
    df["admit_year"] = pd.to_datetime(df["test_date"], errors="coerce").dt.year
    # report_path points at the old path (reports_pilot/), so it is reassembled by basename
    df["report_file"] = df["report_path"].map(
        lambda p: str(REPORT_DIR / os.path.basename(str(p)))
    )
    df = df.drop(columns=["report_path", "test_date"])
    df = df.reset_index(drop=True)
    if verify_reports:
        missing = [p for p in df["report_file"] if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} report_file path(s) do not exist; "
                f"first few: {missing[:5]}"
            )
    return df


def load_embeddings(frame: pd.DataFrame) -> np.ndarray:
    out = np.empty((len(frame), 3072), dtype=np.float32)
    idx = frame["embedding_index"].to_numpy()
    for shard_id in sorted(set(idx // SHARD_ROWS)):
        arr = np.load(SHARD_DIR / f"shard_{shard_id:05d}.npy", mmap_mode="r")
        sel = np.where(idx // SHARD_ROWS == shard_id)[0]
        out[sel] = np.asarray(arr[idx[sel] % SHARD_ROWS], dtype=np.float32)
    return out


def unit_of_analysis_report(frame: pd.DataFrame) -> dict:
    per_patient = frame.groupby("patient_ID").size()
    return {
        "n_rows": int(len(frame)),
        "n_unique_admission_ID": int(frame["admission_ID"].nunique()),
        "n_unique_patient_ID": int(frame["patient_ID"].nunique()),
        "n_unique_newpatient_ID": int(frame["newpatient_ID"].nunique()),
        "n_patients_with_multiple_rows": int((per_patient > 1).sum()),
        "n_extra_rows_from_repeat_patients": int((per_patient - 1).clip(lower=0).sum()),
        "max_rows_per_patient": int(per_patient.max()),
        "newpatient_to_patient_1to1": bool(
            frame.groupby("newpatient_ID")["patient_ID"].nunique().max() == 1
        ),
        "patient_to_newpatient_1to1": bool(
            frame.groupby("patient_ID")["newpatient_ID"].nunique().max() == 1
        ),
        "manuscript_claim": "32,329 adult patients; each patient contributed one index hospitalization",
        "verdict": "MISMATCH -- either revise the manuscript wording to an hospitalization basis, or re-apply the one-patient-one-admission rule",
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed", type=int, default=42,
        help="seed recorded in the reproducibility manifest (this script is deterministic and uses no RNG)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # Regardless of the verify_reports argument value passed to load_frame(),
    # missing_report_files is derived from an actual filesystem check. This
    # way, even if main() is later changed to call
    # load_frame(verify_reports=False), the manifest can never claim 0
    # missing files without verification.
    frame = load_frame()
    missing_report_files = [
        p for p in frame["report_file"] if not os.path.exists(p)
    ]
    missing = len(missing_report_files)
    frame.to_parquet(OUT / "analysis_frame.parquet", index=False)
    rep = unit_of_analysis_report(frame)
    (OUT / "unit_of_analysis_report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    (OUT / "01_manifest.json").write_text(json.dumps({
        "rows": len(frame),
        "missing_report_files": missing,
        "hospital_counts": frame["hospital_code"].value_counts().to_dict(),
        "year_range": [int(frame["admit_year"].min()), int(frame["admit_year"].max())],
        "seed": args.seed,
        "pandas": pd.__version__, "numpy": np.__version__,
    }, indent=2))
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"missing report files: {missing}")


if __name__ == "__main__":
    main()
