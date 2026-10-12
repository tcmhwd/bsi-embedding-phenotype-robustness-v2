#!/usr/bin/env python3
"""Task 2 (this revision round) -- the S6 three-reading table.

`PRESPEC_stress_criteria.md` Section 2, axis S6(c) defines the token
baseline as "the same F1 from an organism + hospital + year + department
lookup under identical folds" -- exactly what `17_correct_env/17d_token_lookup.py`
computed (max F1 over 3 cumulative combos: organism; organism+hospital+year;
organism+hospital+year+department) and what `out/18_encoder_dec/s6_matrix.csv`
stores as `best_lookup_f1`. `20d_task2_extended_lookup.py` extended this to
9 keys (episode_count, day_band, icu, ventilation, department, organism,
hospital, year, narrative_length_band) and took the max over 9 CUMULATIVE
combos in that order -- a change to the pre-specified lookup that, per the
frozen document's opening paragraph ("Any change after the whole-cohort
battery is run is a new, labelled exploratory analysis"), must be reported
as exploratory, not substituted silently for the pre-specified reading.
Critically, cluster 0's winning combo in that extension
(`episode_count+day_band+icu+ventilation`) includes `day_band` -- derived
from `length_of_stay`, NOT a field the published narrative actually
renders, and outcome-adjacent (a long index-to-discharge window is itself
prognostically informative).

This script produces THREE readings of the SAME lookup, side by side, for
all 16 published clusters, reusing the encoder F1 values already computed
in `s6_matrix.csv` verbatim (never recomputed -- those come from a kNN
classifier against the embedding, entirely independent of which lookup
reading is used):

  1. PRESPEC lookup -- organism / +hospital / +year / +department, exactly
     as Section 2 defines it. Reused verbatim from `s6_matrix.csv`'s
     `best_lookup_f1` (== `17d_token_lookup.py`'s `best_lookup_f1`, which
     computed exactly these 3 cumulative combos -- reproduced here, not
     just trusted, via `prespec_branch()` below, and asserted equal to the
     stored value within floating-point tolerance).
  2. Narrative-only extended lookup (EXPLORATORY, PRESPEC Section 0) -- the
     PRESPEC keys plus ONLY the variables actually rendered in the
     published narrative text: episode count, ICU, ventilation. Computed
     as the per-cluster MAX of two independent branches: (a) the PRESPEC
     branch itself (organism/+hospital+year/+department, reading 1 above)
     and (b) a fresh cumulative branch -- episode_count; +icu;
     +ventilation -- built with `20d`'s own `build_extended_frame`/
     `lookup_eval`/`score` functions, reused verbatim. For cluster 0 this
     branch alone gives 0.6561 / 0.6616 / 0.6695 at the three cumulative
     steps -- reproducing (not just quoting) the 0.656/0.662/0.669 figures
     from the fourth audit (`AUDIT_4.md`) -- confirming those numbers
     rather than trusting them.
  3. Full extended lookup (EXPLORATORY, PRESPEC Section 0) -- exactly
     `20d_task2_extended_lookup.py`'s existing computation (9 keys
     including `day_band`), reused verbatim from
     `out/20_onehot/task2_s6_corrected.csv`'s `best_lookup_f1_corrected` --
     NOT recomputed (task instruction).

For each reading: per-cluster lookup F1, S6 pass/fail under the PRESPEC
rule (>=2 of 3 encoders with F1_enc >= 0.60 AND F1_enc >= lookup + 0.10),
the S6 pass count (/16), and the count that additionally passes all FIVE
PRESPEC axes (S1 and S2 and S3(*) and S4 and S5 -- via `s6_matrix.csv`'s
`ALL_AXES_pass_S1S2S4S5` column, computed independently of S6 -- AND this
reading's S6 verdict). (*) `ALL_AXES_pass_S1S2S4S5`'s name lists S1/S2/S4/S5
only -- reused as-is from `s6_matrix.csv`, not altered here; see that
file's own generating script for what it actually gates on.

Outcome-blind (Global Constraints): no mortality/discharge/death column is
read anywhere in this module -- `day_band` itself is derived from
`length_of_stay` (a timing variable, not an outcome), consistent with
every other Task 17/18/20 script; it is flagged here as outcome-ADJACENT
(a long post-index stay is itself prognostic), not outcome-identical.

No SLURM needed -- pure lookup/aggregation on data already on disk, plus
one cheap StratifiedGroupKFold lookup pass (a few seconds, same cost class
as `20d`/`17d`). Writes under `out/28_openpool/` per this task's write
scope (not `out/20_onehot/`, where the underlying Task 2 artifacts already
live).
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c17 = importlib.import_module("17_common")
c20d = importlib.import_module("20d_task2_extended_lookup")

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
OUT_DIR = REV / "out/28_openpool"
S6_MATRIX_PATH = REV / "out/18_encoder_dec/s6_matrix.csv"
TASK2_S6_CORRECTED_PATH = REV / "out/20_onehot/task2_s6_corrected.csv"
G4_PUBLISHED_PATH = REV / "out/17_correct_env/token_lookup/g4_perclass_published.csv"

F1_PASS_MIN = 0.60
F1_MARGIN_OVER_LOOKUP = 0.10
S6_MIN_ENCODERS_PASSING = 2
ENCODERS = ["bge-m3", "qwen3-0.6b", "openai-3small"]


def cat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.char.add(np.char.add(np.asarray(a, dtype=str), "|"), np.asarray(b, dtype=str))


def prespec_branch(frame: pd.DataFrame, y: np.ndarray, folds, labels_sorted) -> dict:
    """Reproduces `17d_token_lookup.py` exactly: organism; +hospital+year;
    +department (3 cumulative combos), max F1 per cluster. Independent
    re-derivation, used only to ASSERT agreement with the stored
    `best_lookup_f1` (never overrides it)."""
    org = frame["BSI_episode_short_clean"].astype(str).to_numpy()
    hosp = frame["hospital_code"].astype(str).to_numpy()
    year = frame["admit_year"].astype(str).to_numpy()
    dept = frame["treatment_department_eng"].astype(str).to_numpy()

    acc = np.asarray(org, dtype=str)
    combos = {"organism": acc.copy()}
    acc = cat(cat(acc, hosp), year)
    combos["organism+hospital+year"] = acc.copy()
    acc = cat(acc, dept)
    combos["organism+hospital+year+department"] = acc.copy()

    per_combo = {}
    for name, key in combos.items():
        per_combo[name] = c20d.score(*c20d.lookup_eval(key, y, folds), labels_sorted)["per_cluster_f1"]
    df = pd.DataFrame(per_combo)
    return {"per_cluster_max": df.max(axis=1).to_dict(), "per_combo": per_combo}


def narrative_branch(frame: pd.DataFrame, y: np.ndarray, folds, labels_sorted) -> dict:
    """episode_count; +icu; +ventilation -- cumulative, standalone (NOT
    chained onto the PRESPEC keys). Reading 2's per-cluster value is the
    max of THIS branch and the PRESPEC branch (see module docstring)."""
    ep = frame["episode_count"].to_numpy()
    icu = frame["icu_str"].to_numpy()
    vent = frame["ventilator_str"].to_numpy()

    acc = np.asarray(ep, dtype=str)
    combos = {"episode_count": acc.copy()}
    acc = cat(acc, icu)
    combos["episode_count+icu"] = acc.copy()
    acc = cat(acc, vent)
    combos["episode_count+icu+ventilation"] = acc.copy()

    per_combo = {}
    for name, key in combos.items():
        per_combo[name] = c20d.score(*c20d.lookup_eval(key, y, folds), labels_sorted)["per_cluster_f1"]
    df = pd.DataFrame(per_combo)
    return {"per_cluster_max": df.max(axis=1).to_dict(), "per_combo": per_combo}


def s6_pass_row(lookup_f1: float, f1s: dict) -> tuple[dict, int]:
    per_encoder = {}
    n_pass = 0
    for enc in ENCODERS:
        f1 = f1s[enc]
        ok = bool((f1 >= F1_PASS_MIN) and (f1 >= lookup_f1 + F1_MARGIN_OVER_LOOKUP))
        per_encoder[enc] = ok
        n_pass += int(ok)
    return per_encoder, n_pass


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    s6 = pd.read_csv(S6_MATRIX_PATH).set_index("cluster")
    s6_corrected = pd.read_csv(TASK2_S6_CORRECTED_PATH).set_index("cluster")

    frame = c20d.build_extended_frame()
    pub = frame[c20d.c17.PUBLISHED_CLUSTER_COL].to_numpy()
    keep = pub != -1
    y = pub[keep]
    groups = frame["patient_ID"].to_numpy()[keep]
    frame_keep = frame.loc[keep].reset_index(drop=True)

    from sklearn.model_selection import StratifiedGroupKFold
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(skf.split(np.zeros(len(y)), y, groups))
    labels_sorted = sorted(set(y.tolist()))

    prespec = prespec_branch(frame_keep, y, folds, labels_sorted)
    narrative = narrative_branch(frame_keep, y, folds, labels_sorted)

    # Sanity: reproduced PRESPEC branch must match the stored best_lookup_f1
    # (17d's own output, carried into s6_matrix.csv) within float tolerance.
    mismatches = []
    for c in labels_sorted:
        stored = float(s6.loc[c, "best_lookup_f1"])
        reproduced = float(prespec["per_cluster_max"][c])
        if abs(stored - reproduced) > 1e-6:
            mismatches.append((c, stored, reproduced))
    if mismatches:
        raise RuntimeError(f"PRESPEC branch reproduction mismatch vs s6_matrix.csv: {mismatches}")
    print(f"[28b] PRESPEC branch reproduction verified against s6_matrix.csv for all "
          f"{len(labels_sorted)} clusters (max abs diff < 1e-6)", flush=True)

    # Cross-check the 0.656/0.662/0.669 figures cited for cluster 0 in AUDIT_4.md.
    c0_narrative_steps = narrative["per_combo"]
    c0_vals = [round(c0_narrative_steps[k][0], 4) for k in
               ["episode_count", "episode_count+icu", "episode_count+icu+ventilation"]]
    print(f"[28b] cluster 0 narrative-branch cumulative F1 (episode_count / +icu / +ventilation): "
          f"{c0_vals} -- audit-cited figures were 0.656 / 0.662 / 0.669", flush=True)
    expected = [0.656, 0.662, 0.669]
    if any(abs(a - b) > 0.001 for a, b in zip(c0_vals, expected)):
        print(f"[28b] WARNING: reproduced narrative-branch values differ from the audit-cited "
              f"figures by > 0.001: reproduced={c0_vals} vs cited={expected}", flush=True)

    rows = []
    for c in labels_sorted:
        size = int(s6.loc[c, "size"])
        f1s = {enc: float(s6.loc[c, f"F1_{enc}"]) for enc in ENCODERS}
        all4_axes = bool(s6.loc[c, "ALL_AXES_pass_S1S2S4S5"])

        lookup1 = float(prespec["per_cluster_max"][c])
        lookup2 = max(lookup1, float(narrative["per_cluster_max"][c]))
        lookup3 = float(s6_corrected.loc[c, "best_lookup_f1_corrected"])

        row = {"cluster": c, "size": size, **{f"F1_{enc}": f1s[enc] for enc in ENCODERS},
               "ALL_AXES_pass_S1S2S4S5": all4_axes}

        for reading_idx, (label, lookup) in enumerate(
            [("reading1_prespec", lookup1), ("reading2_narrative_exploratory", lookup2),
             ("reading3_full_exploratory", lookup3)], start=1
        ):
            per_enc, n_pass = s6_pass_row(lookup, f1s)
            s6_pass = n_pass >= S6_MIN_ENCODERS_PASSING
            row[f"lookup_f1_{label}"] = lookup
            for enc in ENCODERS:
                row[f"S6_pass_{enc}_{label}"] = per_enc[enc]
            row[f"n_encoders_passing_{label}"] = n_pass
            row[f"S6_pass_{label}"] = s6_pass
            row[f"all5_axes_pass_{label}"] = bool(all4_axes and s6_pass)

        # BGE-M3 margin explicitly, at each reading (F1_bge-m3 - lookup - 0.10).
        row["margin_bge-m3_reading1"] = f1s["bge-m3"] - lookup1 - F1_MARGIN_OVER_LOOKUP
        row["margin_bge-m3_reading2"] = f1s["bge-m3"] - lookup2 - F1_MARGIN_OVER_LOOKUP
        row["margin_bge-m3_reading3"] = f1s["bge-m3"] - lookup3 - F1_MARGIN_OVER_LOOKUP

        rows.append(row)

    table = pd.DataFrame(rows)
    c17.atomic_write_csv(table, OUT_DIR / "s6_three_readings.csv")

    summary = {}
    for label in ["reading1_prespec", "reading2_narrative_exploratory", "reading3_full_exploratory"]:
        summary[f"S6_pass_count_{label}"] = int(table[f"S6_pass_{label}"].sum())
        summary[f"all5_axes_pass_count_{label}"] = int(table[f"all5_axes_pass_{label}"].sum())
    cluster0 = table[table["cluster"] == 0].iloc[0]
    summary["cluster0_lookup_f1"] = {
        "reading1_prespec": cluster0["lookup_f1_reading1_prespec"],
        "reading2_narrative_exploratory": cluster0["lookup_f1_reading2_narrative_exploratory"],
        "reading3_full_exploratory": cluster0["lookup_f1_reading3_full_exploratory"],
    }
    summary["cluster0_margin_bge-m3"] = {
        "reading1_prespec": cluster0["margin_bge-m3_reading1"],
        "reading2_narrative_exploratory": cluster0["margin_bge-m3_reading2"],
        "reading3_full_exploratory": cluster0["margin_bge-m3_reading3"],
    }
    summary["cluster0_S6_pass"] = {
        "reading1_prespec": bool(cluster0["S6_pass_reading1_prespec"]),
        "reading2_narrative_exploratory": bool(cluster0["S6_pass_reading2_narrative_exploratory"]),
        "reading3_full_exploratory": bool(cluster0["S6_pass_reading3_full_exploratory"]),
    }
    summary["narrative_branch_cluster0_cumulative_steps"] = dict(zip(
        ["episode_count", "+icu", "+ventilation"], c0_vals))
    summary["note"] = ("Readings 2 and 3 are EXPLORATORY per PRESPEC_stress_criteria.md's opening "
                        "paragraph (any change to the S6(c) lookup after the whole-cohort battery "
                        "is run is a new, labelled exploratory analysis) -- only reading 1 is the "
                        "pre-specified S6(c) definition. Reading 3 includes day_band (derived from "
                        "length_of_stay, not rendered in the published narrative, outcome-adjacent); "
                        "reading 2 deliberately excludes it.")
    c17.atomic_write_json(summary, OUT_DIR / "s6_three_readings_summary.json")

    print()
    print(table[["cluster", "size", "lookup_f1_reading1_prespec", "lookup_f1_reading2_narrative_exploratory",
                 "lookup_f1_reading3_full_exploratory", "S6_pass_reading1_prespec",
                 "S6_pass_reading2_narrative_exploratory", "S6_pass_reading3_full_exploratory",
                 "all5_axes_pass_reading1_prespec", "all5_axes_pass_reading2_narrative_exploratory",
                 "all5_axes_pass_reading3_full_exploratory"]].round(4).to_string(index=False))
    print()
    print(json.dumps(summary, indent=2, default=str))
    print(f"[28b] wrote {OUT_DIR / 's6_three_readings.csv'} and s6_three_readings_summary.json")


if __name__ == "__main__":
    main()
