#!/usr/bin/env python3
"""Manuscript package v6 figures -- tidy source tables (Figure 3, Supplementary Figures 4-6).

Reads ONLY finished, on-disk outputs and reshapes them; nothing is recomputed.  Writes to out/followup_analyses/tables_v6/ (a NEW directory;
out/followup_analyses/tables/ -- the tables behind the earlier candidate figures -- is not touched):

  * every table scripts/71_followup_figure_data.py writes (figA_*, figB_points/arm_grid, figC_*) -- produced by IMPORTING the 71 functions unchanged,
    then asserted cell-for-cell identical to the existing out/followup_analyses/tables/*.csv (so the v6 figures and the candidates share one source);
  * figB_knn.csv -- Figure 3 panel b, the two clustering-free readouts: the stored readouts of the published Figure 3
    (out/33_figures/knn_readouts.csv) plus the Qwen3-0.6B department-removal arm added post hoc
    (B_symmetric_perturbation/knn_readout_B3.csv, field = first_antibiotic), with the chance level cross-checked between the two files.

Env: /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
"""
import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import importlib.util
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parent.parent
SF = REV / "out/followup_analyses"
OUT = SF / "tables_v6"
THIS_FILES = ["scripts/74_v6_figure_data.py", "scripts/74_v6_figures.R", "slurm/74_v6_figure_data.sh", "slurm/74_v6_figures.sh"]

spec = importlib.util.spec_from_file_location("followup71", REV / "scripts/71_followup_figure_data.py")
m71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m71)


def git_info() -> dict:
    import subprocess

    def run(*args) -> str:
        return subprocess.run(["git", "-C", str(REV), *args], capture_output=True, text=True, check=True).stdout.strip()

    st = run("status", "--porcelain", "--", *THIS_FILES)
    return {"git_commit": run("rev-parse", "HEAD"), "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(st), "git_dirty_detail": st.splitlines()[:20], "git_dirty_scope": THIS_FILES}


def fig3_knn(out: Path, src: list) -> dict:
    stored = pd.read_csv(REV / "out/33_figures/knn_readouts.csv")
    b3 = pd.read_csv(m71.B_DIR / "knn_readout_B3.csv")
    src += [REV / "out/33_figures/knn_readouts.csv", m71.B_DIR / "knn_readout_B3.csv"]
    rows = []
    left_key = "3-large label transfer (cluster labels)"
    right_key = "Qwen3 first-antibiotic neighbourhood purity (measured field)"
    for _, r in stored.iterrows():
        if r["subpanel"] == left_key:
            side, enc, k = "left", "text-3-large", 25
        elif r["subpanel"] == right_key:
            side, enc, k = "right", "Qwen3-0.6B", 15
        else:
            raise AssertionError(r["subpanel"])
        rows.append({"side": side, "encoder_compact": enc, "k": k, "variant": r["variant"], "value": r["value"], "metric": r["metric"],
                     "source": "stored (out/33_figures/knn_readouts.csv)"})
    # Qwen3 department removal (added post hoc): first-antibiotic purity, k = 15
    a = b3[(b3.arm_id == "B3_qwen3_no_department") & (b3.field == "first_antibiotic") & (b3.k == 15)]
    assert len(a) == 1
    a = a.iloc[0]
    chance_stored = float(stored.loc[(stored.subpanel == right_key) & (stored.variant == "chance"), "value"].iloc[0])
    assert abs(float(a["chance_match_rate"]) - chance_stored) < 1e-12, "chance level differs between the stored and the B3 readout"
    rows.append({"side": "right", "encoder_compact": "Qwen3-0.6B", "k": 15, "variant": "no_department", "value": float(a["purity"]),
                 "metric": "k=15 neighbourhood purity vs measured first-antibiotic value", "source": "added post hoc (B/knn_readout_B3.csv)"})
    d = pd.DataFrame(rows)
    d.to_csv(out / "figB_knn.csv", index=False)
    return {"figB_knn_rows": len(d)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    src: list = []
    counts: dict = {}
    counts.update(m71.fig_a_concordance(OUT, src))
    counts.update(m71.fig_a_alignment(OUT, src))
    counts.update(m71.fig_b(OUT, src))
    counts.update(m71.fig_c(OUT, src))
    # identity check against the candidate tables
    checked = {}
    for p in sorted(OUT.glob("fig[ABC]_*.csv")) + sorted(OUT.glob("figA_*.csv")) + sorted(OUT.glob("figB_points.csv")) + sorted(OUT.glob("figB_arm_grid.csv")):
        q = SF / "tables" / p.name
        if not q.exists() or p.name in checked:
            continue
        a, b = pd.read_csv(p), pd.read_csv(q)
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)
        checked[p.name] = "identical to out/followup_analyses/tables/" + p.name
    counts.update(fig3_knn(OUT, src))
    prov = {"written": datetime.now().astimezone().isoformat(timespec="seconds"), "script": "scripts/74_v6_figure_data.py",
            "sys_executable": sys.executable, "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
            **git_info(), "inputs": {str(p.relative_to(REV)): m71.sha256(p) for p in dict.fromkeys(src)},
            "outputs": {p.name: m71.sha256(p) for p in sorted(OUT.glob("fig*.csv"))}, "row_counts": counts,
            "identity_check_vs_candidate_tables": checked}
    (OUT / "figure_data_provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps({"row_counts": counts, "identity_checked": sorted(checked), "outputs": sorted(p.name for p in OUT.glob("fig*.csv"))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
