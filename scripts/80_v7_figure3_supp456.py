#!/usr/bin/env python3
"""Manuscript package v7 figures -- data-prep step (Figure 3, Supplementary Figures 4-6).

PURE RESTYLE round: no analysis is re-run, no number changes. This script copies the finished, already-verified
tables_v6 CSVs that scripts/80_v7_figure3_supp456.R needs into a new v7 tables directory, re-verifying byte-for-byte
identity against out/followup_analyses/tables_v6/*.csv on the way (so a v7-scoped, independently-provenanced copy exists
without ever touching scripts/74_v6_figure_data.py or its output directory). Nothing is recomputed; nothing is
retyped or reshaped beyond a straight CSV copy.

Source of truth for values: out/followup_analyses/tables_v6/*.csv, written by scripts/74_v6_figure_data.py (not edited by
this script or by 80_v7_figure3_supp456.R). That directory's figA_dominance_summary.csv already reflects the
runner-up-field correction applied post hoc by scripts/75_fix_runnerup.py (2026-09-21T16:19); this script copies
whatever is on disk there unchanged, so 75's correction carries through automatically. The one figure-visible
correction made in this v7 round (Figure 3 / Supplementary Fig. 5, item D of the task) is a text-label fix only
(no CSV value changes) and is verified directly against figA_dominance_summary.csv and out/33_figures/field_dominance.csv
inside 80_v7_figure3_supp456.R's comments / this script's provenance JSON -- see CHANGELOG_FIGB.md.

Env: /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
"""
import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parent.parent
SF = REV / "out/followup_analyses"
SRC = SF / "tables_v6"
OUT = SF / "tables_v7"
THIS_FILES = ["scripts/80_v7_figure3_supp456.py", "scripts/80_v7_figure3_supp456.R", "slurm/80_v7_figure3_supp456.sh"]

# Every CSV scripts/80_v7_figure3_supp456.R reads, unchanged from tables_v6 (Figure 3, Supplementary Figs. 4-6 only;
# Figures 1/2/4 and their tables are out of scope for this script and are not touched).
NEEDED = [
    "figA_concordance.csv", "figA_degeneracy.csv",                       # Supplementary Fig. 4
    "figA_alignment_cells.csv", "figA_dominance_summary.csv",            # Supplementary Fig. 5
    "figB_points.csv", "figB_arm_grid.csv", "figB_knn.csv",              # Figure 3
    "figC_within_points.csv", "figC_cross_points.csv",                   # Supplementary Fig. 6
    "figC_noise_floor.csv", "figC_edit_sizes.csv",
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def git_info() -> dict:
    import subprocess

    def run(*args) -> str:
        return subprocess.run(["git", "-C", str(REV), *args], capture_output=True, text=True, check=True).stdout.strip()

    st = run("status", "--porcelain", "--", *THIS_FILES)
    return {"git_commit": run("rev-parse", "HEAD"), "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(st), "git_dirty_detail": st.splitlines()[:20], "git_dirty_scope": THIS_FILES}


def main() -> None:
    import shutil

    OUT.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in NEEDED:
        src = SRC / name
        if not src.exists():
            raise FileNotFoundError(f"expected input missing: {src}")
        dst = OUT / name
        shutil.copy2(src, dst)   # byte-for-byte copy -- no pandas round trip, so no float-formatting drift is possible
        assert sha256(src) == sha256(dst), f"copy of {name} is not byte-identical to tables_v6 source"
        # still parse it once, to confirm the copy is readable and well-formed (not just bit-identical)
        df = pd.read_csv(dst)
        copied[name] = {"sha256": sha256(dst), "rows": len(df), "source": str(src.relative_to(REV))}

    # Point-verification for item D (Figure 3 / Supplementary Fig. 5 "therapy block" -> dominant-field label check).
    # Nothing is changed here; this only records the evidence the R script's in-panel labels were checked against.
    dom = pd.read_csv(OUT / "figA_dominance_summary.csv")
    qrow = dom[(dom.condition == "korea_full_template") & (dom.encoder == "qwen3-0.6b")].iloc[0]
    fd = pd.read_csv(REV / "out/33_figures/field_dominance.csv")
    fdrow = fd[(fd.cohort == "Korea") & (fd.encoder == "qwen3-0.6b") & (fd.is_dominant == True)].iloc[0]
    item_d_check = {
        "figA_dominance_summary.csv (korea_full_template, qwen3-0.6b)": {
            "dominant_field": qrow["dominant_field"], "dominant_field_display_key": qrow["dominant_field_display_key"],
            "top_ami_median": float(qrow["top_ami_median"])},
        "out/33_figures/field_dominance.csv (Korea, qwen3-0.6b, is_dominant=True)": {
            "field": fdrow["field"], "ami": float(fdrow["ami"])},
        "expected_per_task": {"dominant_field_display_key": "first_antibiotic", "ami_approx": 0.947},
        "verdict": "CONFIRMED: dominant field is first_antibiotic (\"first antibiotic named\"), NOT \"therapy block\". "
                   "Figure 3 / Supplementary Fig. 5 canvases already say \"first antibiotic named\" everywhere "
                   "(scripts/74_v6_figures.R DOM_FIELD, FIELD_LABELS); no canvas text needed correcting. "
                   "The mislabel this item warned about ('Qwen3-Embedding-0.6B's dominant field is the therapy "
                   "block') is present instead in the Fig. 3 LEGEND TEXT (.md, out of scope for this script/task) "
                   "-- see CHANGELOG_FIGB.md."}
    assert qrow["dominant_field_display_key"] == "first_antibiotic"
    assert abs(qrow["top_ami_median"] - 0.947226101384192) < 1e-9
    assert fdrow["field"] == "first_antibiotic" and abs(fdrow["ami"] - 0.947226101384192) < 1e-6

    prov = {"written": datetime.now().astimezone().isoformat(timespec="seconds"), "script": "scripts/80_v7_figure3_supp456.py",
            "sys_executable": sys.executable, "python": platform.python_version(), "pandas": pd.__version__,
            **git_info(), "copied_from": "out/followup_analyses/tables_v6/", "copied": copied, "item_d_verification": item_d_check}
    (OUT / "figure_data_provenance_v7.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps({"copied": sorted(copied), "item_d_verdict": item_d_check["verdict"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
