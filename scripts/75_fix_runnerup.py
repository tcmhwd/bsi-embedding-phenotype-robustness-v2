#!/usr/bin/env python3
"""Fix of the runner-up columns of figA_dominance_summary.csv (audit item 15, ROUND-2; nothing is re-fitted or re-scored).

The defect
----------
scripts/71_followup_figure_data.py:155-156 (inside `fig_a_alignment.add`) builds

    order = list(fm.items())                     # fm = dom["per_field_ami_median"]
    "runner_up_field": order[1][0], "runner_up_ami_median": order[1][1]

i.e. it takes the SECOND INSERTED item of the per-field median-AMI dictionary as "runner-up", not the field with the second-highest median AMI.
Consequences for the stored rows of tables/figA_dominance_summary.csv and tables_v6/figA_dominance_summary.csv: `runner_up_field` and
`runner_up_ami_median` are wrong (for example, Qwen3 on the corpus of record shows BSI_episode_short_clean 0.0405 where antibiotic_set 0.2937 is the runner-up).
The margin column (`dominance_margin_median`, a median across fits of the per-fit difference) is NOT affected, and the manuscript and the figures do not use
the runner-up columns.  Scripts 71 and 72 are not edited (frozen); this NEW script re-derives the two columns and writes the corrected table.

What this script does
---------------------
  * reads the same sources as script 71 (out/40_field_dominance/field_dominance.json for the stored 29-field / 19-field sets and
    out/followup_analyses/A_harmonized_template/field_attribution.json for the rendered fields of the harmonized arms);
  * for every (condition, encoder) row sorts the per-field MEDIAN AMI in descending order (ties by field name) and sets
    runner_up_field / runner_up_ami_median to the second entry; adds `top_field_by_median_ami` (first entry) so the meaning is explicit -- the runner-up is
    "second-highest per-field median AMI", the same convention as `runner_up_by_median_ami` in field_attribution.json and the comparison table;
  * keeps every other column byte-for-byte (asserted), in particular `dominance_margin_median`, `top_field_consistency`, `n_fields_scored`;
  * verifies against A_harmonized_template/comparison_table.csv (columns bge_/qwen_ runner_up_field / runner_up_ami_median) and against the
    `runner_up_by_median_ami` keys of field_attribution.json, and reports every row where comparison_table.csv itself disagrees;
  * sanity check: (top - runner-up) by per-field medians vs the margin column (a median of per-fit differences, so equality is not expected; |diff| < 0.05).

Outputs (out/followup_analyses/tables_v6/):
  figA_dominance_summary.csv                       corrected table (previous version kept as figA_dominance_summary_before_75.csv)
  figA_dominance_summary_runnerup_check.csv        row-by-row comparison old / new / comparison_table / source key
  figA_dominance_summary_75_provenance.json

Env: /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
"""
import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parent.parent
SF = REV / "out/followup_analyses"
TV6 = SF / "tables_v6"
THIS_FILES = ["scripts/75_fix_runnerup.py", "slurm/75_fix_runnerup.sh"]

spec = importlib.util.spec_from_file_location("followup71", REV / "scripts/71_followup_figure_data.py")
m71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m71)

MODELS = list(m71.MODELS)
HARMONIZED = ("korean_mimic_compatible", "harmonized_keep_therapy", "thickness_matched_random")
CT_PREFIX = {"bge-m3": "bge", "qwen3-0.6b": "qwen"}


def per_field_sources():
    """{(condition, encoder): (per_field_ami_median dict, runner_up_by_median_ami or None, runner_up_median_ami or None)} -- same sources as script 71."""
    attr = json.loads((m71.A_DIR / "field_attribution.json").read_text())
    stored = json.loads(m71.STORED_DOM.read_text())
    out = {}
    for m in MODELS:
        dom = stored["cohorts"]["korea_matched_refits"][m]
        out[("korea_full_template", m)] = (dom["per_field_ami_median"], dom.get("runner_up_by_median_ami"), dom.get("runner_up_median_ami"))
    for cond in HARMONIZED:
        for m in MODELS:
            dom = attr["conditions"][cond][m]["dominance_rendered_fields"]
            out[(cond, m)] = (dom["per_field_ami_median"], dom.get("runner_up_by_median_ami"), dom.get("runner_up_median_ami"))
    for m in MODELS:
        dom = stored["cohorts"]["mimic"][m]
        out[("mimic_iv", m)] = (dom["per_field_ami_median"], dom.get("runner_up_by_median_ami"), dom.get("runner_up_median_ami"))
    return out, [m71.A_DIR / "field_attribution.json", m71.STORED_DOM]


def git_info() -> dict:
    def run(*args) -> str:
        return subprocess.run(["git", "-C", str(REV), *args], capture_output=True, text=True, check=True).stdout.strip()
    st = run("status", "--porcelain", "--", *THIS_FILES)
    return {"git_commit": run("rev-parse", "HEAD"), "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(st), "git_dirty_detail": st.splitlines()[:20], "git_dirty_scope": THIS_FILES}


def main() -> int:
    path = TV6 / "figA_dominance_summary.csv"
    old = pd.read_csv(path)
    backup = TV6 / "figA_dominance_summary_before_75.csv"
    if not backup.exists():                      # keep the first (buggy) version; never overwrite the backup
        shutil.copyfile(path, backup)
    old0 = pd.read_csv(backup)
    assert old0.shape == old.shape and list(old0.columns) == list(old.columns)
    src, src_files = per_field_sources()
    comp = pd.read_csv(m71.A_DIR / "comparison_table.csv").set_index("condition")

    new = old0.copy()
    new["top_field_by_median_ami"] = None
    check = []
    problems = []
    for i, r in old0.iterrows():
        pf, key_field, key_val = src[(r["condition"], r["encoder"])]
        assert len(pf) == int(r["n_fields_scored"]), (r["condition"], r["encoder"], len(pf), r["n_fields_scored"])
        ranked = sorted(pf.items(), key=lambda kv: (-kv[1], kv[0]))
        (top_f, top_v), (ru_f, ru_v) = ranked[0], ranked[1]
        new.loc[i, "runner_up_field"] = ru_f
        new.loc[i, "runner_up_ami_median"] = ru_v
        new.loc[i, "top_field_by_median_ami"] = top_f
        row = {"condition": r["condition"], "encoder": r["encoder"], "n_fields_scored": len(pf),
               "old_runner_up_field": r["runner_up_field"], "old_runner_up_ami_median": r["runner_up_ami_median"],
               "new_runner_up_field": ru_f, "new_runner_up_ami_median": ru_v, "top_field_by_median_ami": top_f, "top_field_median_ami": top_v,
               "modal_dominant_field": r["dominant_field"], "margin_column": r["dominance_margin_median"],
               "top_minus_runner_up_by_medians": top_v - ru_v}
        # (1) internal consistency with the margin column
        row["sanity_margin_close"] = bool(abs((top_v - ru_v) - r["dominance_margin_median"]) < 0.05)
        # (2) source key runner_up_by_median_ami (present in field_attribution.json blocks)
        if key_field is not None:
            row["source_key_runner_up_field"], row["source_key_runner_up_ami"] = key_field, key_val
            row["matches_source_key"] = bool(key_field == ru_f and abs(key_val - ru_v) < 1e-9)
        else:
            row["source_key_runner_up_field"], row["source_key_runner_up_ami"], row["matches_source_key"] = None, np.nan, None
        # (3) comparison_table.csv
        if r["condition"] in comp.index:
            pfx = CT_PREFIX[r["encoder"]]
            ct_f, ct_v = comp.loc[r["condition"], f"{pfx}_runner_up_field"], comp.loc[r["condition"], f"{pfx}_runner_up_ami_median"]
            ct_margin = comp.loc[r["condition"], f"{pfx}_dominance_margin_median"]
            row["comparison_table_runner_up_field"], row["comparison_table_runner_up_ami"] = ct_f, ct_v
            row["comparison_table_margin"] = ct_margin
            row["same_field_set_as_comparison_table"] = bool(abs(ct_margin - r["dominance_margin_median"]) < 1e-6)
            row["matches_comparison_table"] = bool(ct_f == ru_f and abs(ct_v - ru_v) < 1e-6)
        check.append(row)
        if not row["sanity_margin_close"]:
            problems.append(f"margin sanity fails: {r['condition']} {r['encoder']} {row['top_minus_runner_up_by_medians']:.4f} vs {r['dominance_margin_median']:.4f}")
        if row["matches_source_key"] is False:
            problems.append(f"source key mismatch: {r['condition']} {r['encoder']}")
    chk = pd.DataFrame(check)

    # every column other than the two corrected ones (and the added one) is unchanged
    keep = [c for c in old0.columns if c not in ("runner_up_field", "runner_up_ami_median")]
    pd.testing.assert_frame_equal(old0[keep], new[keep], check_dtype=False)
    new.to_csv(path, index=False)
    chk.to_csv(TV6 / "figA_dominance_summary_runnerup_check.csv", index=False)

    n_changed = int(((old0["runner_up_field"] != new["runner_up_field"]) | (np.abs(old0["runner_up_ami_median"] - new["runner_up_ami_median"]) > 1e-12)).sum())
    print(f"[75] rows {len(new)}; rows whose runner-up changed: {n_changed}")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print(chk[["condition", "encoder", "old_runner_up_field", "old_runner_up_ami_median", "new_runner_up_field", "new_runner_up_ami_median",
               "top_field_by_median_ami", "modal_dominant_field", "top_minus_runner_up_by_medians", "margin_column", "sanity_margin_close",
               "matches_source_key"]].to_string())
    ct = chk[chk.get("matches_comparison_table").notna()] if "matches_comparison_table" in chk else chk.iloc[0:0]
    print("\n[75] comparison_table.csv verification (rows present there):")
    print(ct[["condition", "encoder", "same_field_set_as_comparison_table", "new_runner_up_field", "comparison_table_runner_up_field",
              "new_runner_up_ami_median", "comparison_table_runner_up_ami", "matches_comparison_table"]].to_string())
    comparable = ct[ct["same_field_set_as_comparison_table"]]
    disagree = comparable[~comparable["matches_comparison_table"]]
    print(f"[75] comparable rows (same field set / same margin): {len(comparable)}; agree with comparison_table.csv: {int(comparable['matches_comparison_table'].sum())}; disagree: {len(disagree)}")
    if len(disagree):
        print("[75] NOTE -- comparison_table.csv disagrees with the sorted runner-up on these rows (its own runner-up columns are then the inconsistent ones; "
              "see the margin sanity check):")
        print(disagree[["condition", "encoder", "new_runner_up_field", "comparison_table_runner_up_field", "top_minus_runner_up_by_medians", "margin_column"]].to_string())

    prov = {"written": datetime.now().astimezone().isoformat(timespec="seconds"), "script": "scripts/75_fix_runnerup.py",
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sources": [str(p) for p in src_files] +
            [str(m71.A_DIR / "comparison_table.csv"), str(backup)], "rows": int(len(new)), "rows_runner_up_changed": n_changed,
            "comparison_rows_comparable": int(len(comparable)), "comparison_rows_agree": int(comparable["matches_comparison_table"].sum()),
            "comparison_rows_disagree": [f"{a}|{b}" for a, b in zip(disagree["condition"], disagree["encoder"])], "problems": problems, **git_info()}
    (TV6 / "figA_dominance_summary_75_provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    if problems:
        print("[75] PROBLEMS:", *problems, sep="\n  ")
        return 1
    print("[75] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
