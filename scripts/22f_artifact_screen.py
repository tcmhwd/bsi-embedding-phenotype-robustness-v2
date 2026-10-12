#!/usr/bin/env python3
"""Task 22 Task B -- PRESPEC section 4 artifact screen, all six rules,
applied to all sixteen PUBLISHED clusters (`phenotype_clusters_labels.csv`
column `phenotype_cluster`, the actual stored 16+noise object -- NOT the A0
refit, per the task brief's "all sixteen published clusters").

Rules copied VERBATIM from `.superpowers/sdd/PRESPEC_stress_criteria.md`
section 4 (2026-09-05 read) into `RULES` below; thresholds are never typed
elsewhere in this script.

Derived variables (none of these exist as a column already; each is
documented at its point of use):
  - `day_band` = length_of_stay - days_from_admission_to_index_bsi
                 ("index -> discharge day band")
  - `hospital_onset` = days_from_admission_to_index_bsi >= 2
  - `short_stay` = day_band <= 2
  - `n_bsi_episodes` is NOT derived (already a column in
    `phenotype_ml_dataset.csv`); reported per-cluster as descriptive context
    only -- it does not gate any of the six section-4 rules, which read only
    organism/department/year/LOS/timing/empiric-field columns.

The "long stay / post-index" rule's third conjunct requires "the cluster's J
against the index-window narrative variant (section 6)".

  2026-09-05 (first run): that variant was among PRESPEC section 6's
  "required inputs not yet on disk" ("has not been run"), so this script
  evaluated only the two computable conjuncts and marked the overall label
  INDETERMINATE -- never PASS, never FAIL -- rather than assuming the
  missing conjunct satisfied.

  2026-09-06 (this run): the variant WAS subsequently generated, embedded
  and clustered in the published-template variant pass (Task 21), so the
  third conjunct is now computed from
  `out/21_published_variants/cluster/index_window/result.json`
  (`best_match_jaccard_vs_published`, the per-published-cluster best-match
  Jaccard against the index-window partition). INDETERMINATE remains the
  label only if that file is absent, so this script still degrades to the
  2026-09-05 behaviour rather than silently passing clusters.

Nothing else about the rule changed: the threshold `max_j_index_window`
0.50 was already present in `RULES` from the verbatim 2026-09-05 copy and
is not re-typed here.

F1_lookup(c) for the year/era rule's second disjunct reuses
`out/17_correct_env/token_lookup/g4_perclass_published.csv`
(`17d_token_lookup.py`, already computed against these same published
labels: StratifiedGroupKFold(5, patient_ID) majority-vote lookup) --
not recomputed here.

Outcome-blind (Global Constraints, PRESPEC section 9): `30d_mortality`,
`90d_mortality`, `in_hospital_mortality`, `death_date`, `discharge_date` are
never read. `length_of_stay`/`days_from_admission_to_index_bsi` are read
only for the section-4 timing screen, per PRESPEC section 9's explicit
carve-out.

Runs directly in `bsi_pheno` on the login node (mechanical pandas
aggregation over already-materialized files; a few seconds) -- no SLURM
submission needed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/data/tcmhwd/AI/AST_ai")
REV = ROOT / "phenotype_revision"
PUBLISHED_LABELS = ROOT / "pipeline_new/out_pre/phenotype_clusters_labels.csv"
DATASET = ROOT / "manuscript/phenotype_ml_dataset.csv"
LOOKUP_F1 = REV / "out/17_correct_env/token_lookup/g4_perclass_published.csv"
INDEX_WINDOW_RESULT = REV / "out/21_published_variants/cluster/index_window/result.json"
OUT_DIR = REV / "out/22_published_objects/task_b_artifact_screen"

# PRESPEC section 4 thresholds, copied verbatim, 2026-09-05.
RULES = {
    "organism_defined": {"min_share": 0.90, "field": "BSI_episode_short_clean"},
    "department_defined": {"min_share": 0.90, "field": "treatment_department_eng"},
    "year_era_defined": {"min_share_3yr_window": 0.90, "min_f1_lookup_delta": 0.30},
    "timing_short_stay": {"min_frac": 0.60, "max_day_band": 2},
    "timing_long_stay": {"min_frac_hospital_onset": 0.60, "min_median_day_band": 30,
                          "max_j_index_window": 0.50, "hospital_onset_min_day": 2},
    "sparse_template": {"min_frac": 0.80},
}

NEEDED_COLS = [
    "admission_ID", "BSI_episode_short_clean", "treatment_department_eng", "test_date",
    "length_of_stay", "days_from_admission_to_index_bsi", "is_empiric_tt_adapted",
    "ab_admission_name_clean", "n_bsi_episodes",
]


def load_index_window_jaccard() -> dict | None:
    """Per-published-cluster best-match Jaccard against the index-window
    partition, or None when that variant has not been produced.

    Returning None (rather than an empty dict or zeros) is deliberate: a
    missing file must reproduce the 2026-09-05 INDETERMINATE behaviour, and
    a zero would silently SATISFY the `< 0.50` conjunct and manufacture
    long-stay labels for every cluster.
    """
    if not INDEX_WINDOW_RESULT.exists():
        return None
    payload = json.loads(INDEX_WINDOW_RESULT.read_text())
    return {int(k): float(v) for k, v in payload["best_match_jaccard_vs_published"].items()}


def load_frame() -> pd.DataFrame:
    pub = pd.read_csv(PUBLISHED_LABELS, usecols=["admission_ID", "phenotype_cluster"])
    ds = pd.read_csv(DATASET, usecols=NEEDED_COLS, low_memory=False).drop_duplicates(subset="admission_ID")
    frame = pub.merge(ds, on="admission_ID", how="left", validate="one_to_one")
    missing = frame["BSI_episode_short_clean"].isna().sum()
    if missing:
        raise RuntimeError(f"{missing} admission_ID(s) missing from {DATASET} after join")
    frame["admit_year"] = pd.to_datetime(frame["test_date"], errors="coerce").dt.year
    frame["day_band"] = frame["length_of_stay"] - frame["days_from_admission_to_index_bsi"]
    frame["hospital_onset"] = frame["days_from_admission_to_index_bsi"] >= RULES["timing_long_stay"]["hospital_onset_min_day"]
    frame["short_stay"] = frame["day_band"] <= RULES["timing_short_stay"]["max_day_band"]
    return frame


def best_3yr_window_share(years: pd.Series) -> tuple[float, tuple[int, int] | None]:
    """Max fraction of `years` (non-null) falling in any 3-consecutive-year
    window, and that window. Denominator is the CLUSTER'S TOTAL member count
    (passed in by caller via `years` already being the full per-cluster
    Series including NaNs) -- a member with an unparseable admit_year counts
    against every window (conservative: missing does not manufacture a
    year-defined cluster)."""
    total = len(years)
    valid = years.dropna().astype(int)
    if valid.empty or total == 0:
        return 0.0, None
    counts = valid.value_counts()
    all_years = sorted(counts.index)
    best_frac, best_window = 0.0, None
    for y0 in all_years:
        n_in_window = int(counts[(counts.index >= y0) & (counts.index <= y0 + 2)].sum())
        frac = n_in_window / total
        if frac > best_frac:
            best_frac, best_window = frac, (y0, y0 + 2)
    return best_frac, best_window


def top_share(series: pd.Series) -> tuple[float, object]:
    total = len(series)
    if total == 0:
        return 0.0, None
    vc = series.value_counts(dropna=False)
    top_val = vc.index[0]
    return float(vc.iloc[0] / total), top_val


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = load_frame()
    lookup = pd.read_csv(LOOKUP_F1)
    index_window_j = load_index_window_jaccard()

    clusters = sorted(c for c in frame["phenotype_cluster"].unique() if c != -1)
    rows = []
    for c in clusters:
        sub = frame[frame["phenotype_cluster"] == c]
        n = len(sub)

        org_share, org_val = top_share(sub["BSI_episode_short_clean"])
        dept_share, dept_val = top_share(sub["treatment_department_eng"])
        yr_share, yr_window = best_3yr_window_share(sub["admit_year"])

        lk = lookup.loc[lookup["published_cluster"] == c]
        f1_org = float(lk["lookup_organism"].iloc[0]) if len(lk) else float("nan")
        f1_org_hosp_yr = float(lk["lookup_organism_hospital_year"].iloc[0]) if len(lk) else float("nan")
        f1_year_delta = f1_org_hosp_yr - f1_org if len(lk) else float("nan")

        short_stay_frac = float(sub["short_stay"].mean())
        hospital_onset_frac = float(sub["hospital_onset"].mean())
        day_band_median = float(sub["day_band"].median())

        empiric_missing_frac = float(sub["is_empiric_tt_adapted"].isna().mean())
        no_admission_ab = (
            sub["ab_admission_name_clean"].isna()
            | (sub["ab_admission_name_clean"].astype(str).str.strip() == "")
        )
        no_admission_ab_frac = float(no_admission_ab.mean())
        sparse_frac = float((sub["is_empiric_tt_adapted"].isna()
                              | sub["ab_admission_name_clean"].isna()
                              | (sub["ab_admission_name_clean"].astype(str).str.strip() == "")).mean())

        n_bsi_episodes_mean = float(sub["n_bsi_episodes"].mean())
        n_bsi_episodes_median = float(sub["n_bsi_episodes"].median())

        organism_defined = org_share >= RULES["organism_defined"]["min_share"]
        department_defined = dept_share >= RULES["department_defined"]["min_share"]
        year_era_defined = (
            yr_share >= RULES["year_era_defined"]["min_share_3yr_window"]
            or (not np.isnan(f1_year_delta) and f1_year_delta >= RULES["year_era_defined"]["min_f1_lookup_delta"])
        )
        timing_short_stay = short_stay_frac >= RULES["timing_short_stay"]["min_frac"]
        # Long-stay: three conjuncts. The first two are computed here; the
        # third is the cluster's best-match Jaccard against the index-window
        # partition, available since Task 21 produced that variant.
        long_stay_computable_conjuncts = (
            hospital_onset_frac >= RULES["timing_long_stay"]["min_frac_hospital_onset"]
            and day_band_median >= RULES["timing_long_stay"]["min_median_day_band"]
        )
        j_index_window = (
            float("nan") if index_window_j is None else index_window_j.get(int(c), float("nan"))
        )
        if index_window_j is None or np.isnan(j_index_window):
            # Reproduces the 2026-09-05 behaviour exactly when the variant is
            # absent: never PASS, never FAIL.
            timing_long_stay = False
            timing_long_stay_label = (
                "INDETERMINATE (index-window variant not on disk)"
                if long_stay_computable_conjuncts
                else "FALSE"
            )
        else:
            timing_long_stay = long_stay_computable_conjuncts and (
                j_index_window < RULES["timing_long_stay"]["max_j_index_window"]
            )
            timing_long_stay_label = str(timing_long_stay).upper()
        sparse_template = sparse_frac >= RULES["sparse_template"]["min_frac"]

        labels = []
        if organism_defined:
            labels.append("organism-defined")
        if department_defined:
            labels.append("department-defined")
        if year_era_defined:
            labels.append("year/era-defined")
        if timing_short_stay:
            labels.append("timing-defined (short stay)")
        if timing_long_stay:
            labels.append("timing-defined (long stay / post-index)")
        elif long_stay_computable_conjuncts and timing_long_stay_label.startswith("INDETERMINATE"):
            labels.append("timing-defined (long stay / post-index) [INDETERMINATE: 3rd conjunct unavailable]")
        if sparse_template:
            labels.append("sparse-template")
        template_timing_flag = any(l for l in labels if l != "organism-defined")

        rows.append(dict(
            cluster=int(c), n=n,
            organism_defined=organism_defined, organism_share=round(org_share, 4), organism_top_value=org_val,
            department_defined=department_defined, department_share=round(dept_share, 4), department_top_value=dept_val,
            year_era_defined=year_era_defined, year_3yr_window_share=round(yr_share, 4),
            year_3yr_window=str(yr_window), f1_lookup_organism=round(f1_org, 4) if not np.isnan(f1_org) else None,
            f1_lookup_organism_hosp_year=round(f1_org_hosp_yr, 4) if not np.isnan(f1_org_hosp_yr) else None,
            f1_lookup_year_delta=round(f1_year_delta, 4) if not np.isnan(f1_year_delta) else None,
            timing_short_stay=timing_short_stay, short_stay_frac=round(short_stay_frac, 4),
            timing_long_stay=timing_long_stay,
            timing_long_stay_conjuncts_1_2=long_stay_computable_conjuncts,
            timing_long_stay_j_index_window=(
                None if np.isnan(j_index_window) else round(j_index_window, 4)
            ),
            timing_long_stay_label=timing_long_stay_label,
            hospital_onset_frac=round(hospital_onset_frac, 4), day_band_median=day_band_median,
            sparse_template=sparse_template, sparse_frac=round(sparse_frac, 4),
            empiric_field_missing_frac=round(empiric_missing_frac, 4),
            no_admission_ab_frac=round(no_admission_ab_frac, 4),
            n_bsi_episodes_mean=round(n_bsi_episodes_mean, 3), n_bsi_episodes_median=n_bsi_episodes_median,
            labels="; ".join(labels) if labels else "(no label at all)",
            template_or_timing_flagged=bool([l for l in labels if l != "organism-defined"]),
        ))

    matrix = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    matrix.to_csv(OUT_DIR / "artifact_screen_matrix.csv", index=False)

    n_organism_only = int(((matrix["organism_defined"]) & (~matrix["template_or_timing_flagged"])).sum())
    n_any_template_or_timing = int(matrix["template_or_timing_flagged"].sum())
    n_no_label_at_all = int((~matrix["organism_defined"] & ~matrix["template_or_timing_flagged"]).sum())

    summary = dict(
        n_clusters=len(matrix),
        rule_counts={
            "organism_defined": int(matrix["organism_defined"].sum()),
            "department_defined": int(matrix["department_defined"].sum()),
            "year_era_defined": int(matrix["year_era_defined"].sum()),
            "timing_short_stay": int(matrix["timing_short_stay"].sum()),
            "timing_long_stay": int(matrix["timing_long_stay"].sum()),
            "timing_long_stay_conjuncts_1_2_only": int(matrix["timing_long_stay_conjuncts_1_2"].sum()),
            "sparse_template": int(matrix["sparse_template"].sum()),
        },
        n_organism_defined_only_no_other_label=n_organism_only,
        n_flagged_template_or_timing_other_than_organism=n_any_template_or_timing,
        n_carrying_no_label_at_all=n_no_label_at_all,
        long_stay_third_conjunct_source=(
            None if index_window_j is None else str(INDEX_WINDOW_RESULT)
        ),
        note_long_stay_rule=(
            "All three conjuncts evaluated. The 3rd (J vs the index-window narrative "
            "variant, threshold < 0.50) was INDETERMINATE on 2026-09-05 because that "
            "variant was not yet on disk; Task 21 subsequently generated, embedded and "
            "clustered it, so it is now read from "
            "out/21_published_variants/cluster/index_window/result.json. If that file is "
            "absent this script reverts to reporting INDETERMINATE rather than PASS."
            if index_window_j is not None else
            "The long-stay rule's 3rd conjunct (J vs index-window narrative variant) is "
            "not on disk; clusters satisfying the other two conjuncts are reported as "
            "INDETERMINATE, not PASS."
        ),
    )
    (OUT_DIR / "SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    print()
    print(matrix[["cluster", "n", "organism_defined", "department_defined", "year_era_defined",
                   "timing_short_stay", "timing_long_stay_label", "sparse_template", "labels"]].to_string(index=False))
    print(f"\n[artifact_screen] wrote {OUT_DIR}/artifact_screen_matrix.csv and SUMMARY.json")


if __name__ == "__main__":
    main()
