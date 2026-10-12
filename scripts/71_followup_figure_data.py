#!/usr/bin/env python3
"""Follow-up figures (candidates only) -- tidy source tables.

Reads ONLY finished, on-disk outputs and reshapes them; nothing is recomputed.  Writes tidy CSVs to out/followup_analyses/tables/
for scripts/72_followup_figures.R (which draws PNG + SVG into out/followup_analyses/figures/):

  Figure_A_harmonized_template_concordance
      figA_concordance.csv     cross-encoder (BGE-M3 vs Qwen3-0.6B) ARI and AMI, both-non-noise: all pairs and the pair-level
                               non-degenerate subset (k >= 5 in both), median and observed range, subset size
      figA_degeneracy.csv      per encoder: median k (+ range), share of fits with k <= 4, largest-cluster share (median + range)
      source: A_harmonized_template/comparison_table.csv, matched_encoders.json
  Figure_A_field_alignment
      figA_alignment_cells.csv     field x condition x encoder: median AMI across fits, status (scored | not_rendered | not_scored),
                                   is_dominant, dominance margin (median across fits), consistency (modal share)
      figA_dominance_summary.csv   per condition x encoder: dominant field, top-field AMI (median, observed range), margin, consistency
      source: A field_attribution.json (harmonized arms), out/40_field_dominance/field_dominance.json (Korea full, MIMIC-IV)
  Figure_B_completed_perturbation_matrix
      figB_points.csv, figB_arm_grid.csv   from B perturbation_matrix_completed.csv (script 69)
  Figure_C_mimic_organism_ablation
      figC_within_points.csv, figC_cross_points.csv, figC_noise_floor.csv, figC_edit_sizes.csv
      source: C mimic_ablation.csv, mimic_ablation_cross_encoder.csv, render_qa.json

Conventions: ARI / AMI = both-non-noise (the manuscript headline convention) unless a column says otherwise.  Ranges are OBSERVED ranges
(min, max) across fits/pairs/seeds -- not confidence intervals.  Display crosswalk: Korean `BSI_episode_short_clean` and MIMIC
`organism_category` share one display row ("index organism"); Korean `charlson_category_withcancermerged` and MIMIC `charlson_category` share
one ("Charlson category").  The source names are kept in `field_source_name`.

Env: /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
"""
import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parent.parent
SF = REV / "out/followup_analyses"
A_DIR, B_DIR, C_DIR = SF / "A_harmonized_template", SF / "B_symmetric_perturbation", SF / "C_mimic_organism_ablation"
STORED_DOM = REV / "out/40_field_dominance/field_dominance.json"
THIS_FILES = ["scripts/71_followup_figure_data.py", "scripts/72_followup_figures.R",
              "slurm/71_followup_figure_data.sh", "slurm/72_followup_figures.sh"]

CONDITIONS = ["korea_full_template", "korean_mimic_compatible", "harmonized_keep_therapy", "thickness_matched_random", "mimic_iv"]
COND_LABEL = {"korea_full_template": "Korea full template", "korean_mimic_compatible": "Korea MIMIC-compatible",
              "harmonized_keep_therapy": "Korea harmonized, therapy kept", "thickness_matched_random": "Korea thickness-matched random",
              "mimic_iv": "MIMIC-IV"}
MODELS = ["bge-m3", "qwen3-0.6b"]
ENC_LABEL = {"bge-m3": "BGE-M3", "qwen3-0.6b": "Qwen3-0.6B"}
SHORT = {"bge-m3": "bge", "qwen3-0.6b": "qwen"}
DISPLAY_ROW = {"BSI_episode_short_clean": "index_organism", "organism_category": "index_organism",
               "charlson_category_withcancermerged": "charlson_category", "charlson_category": "charlson_category"}
C_ARMS = ["unmodified", "no_organism", "no_organism_no_resistance", "no_charlson", "no_source"]
C_ENC = ["bge-m3", "qwen3-0.6b"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict:
    def run(*args) -> str:
        return subprocess.run(["git", "-C", str(REV), *args], capture_output=True, text=True, check=True).stdout.strip()

    st = run("status", "--porcelain", "--", *THIS_FILES)
    return {"git_commit": run("rev-parse", "HEAD"), "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(st), "git_dirty_detail": st.splitlines()[:20], "git_dirty_scope": THIS_FILES}


# =============================================================================================================
# Figure A -- concordance
# =============================================================================================================
def fig_a_concordance(out: Path, src: list) -> dict:
    ct = pd.read_csv(A_DIR / "comparison_table.csv")
    me = json.loads((A_DIR / "matched_encoders.json").read_text())
    src += [A_DIR / "comparison_table.csv", A_DIR / "matched_encoders.json"]
    rows = []
    for cond in CONDITIONS:
        r = ct[ct["condition"] == cond].iloc[0]
        n_all, n_nd = int(r["n_cross_encoder_pairs"]), int(r["n_pairs_nondegenerate_k_ge_5_both"])
        for metric in ("ari", "ami"):
            for pair_set, suf, n in (("all_pairs", "unrestricted", n_all), ("nondegenerate_pairs", "nondegenerate", n_nd)):
                med, lo, hi = (r[f"cross_{metric}_{k}_{suf}"] for k in ("median", "min", "max"))
                rows.append({"condition": cond, "condition_label": COND_LABEL[cond], "metric": metric.upper(), "pair_set": pair_set,
                             "n_pairs": n, "n_pairs_total": n_all, "median": med, "min": lo, "max": hi,
                             "replicate_structure": r["replicate_structure"],
                             "convention": "both-non-noise", "range_type": "observed range across pairs (not a CI)"})
    conc = pd.DataFrame(rows)
    # cross-check tidy values against matched_encoders.json (same numbers, second source)
    for cond in CONDITIONS:
        s = me["stats"][cond]
        for metric in ("ari", "ami"):
            got = conc[(conc.condition == cond) & (conc.metric == metric.upper()) & (conc.pair_set == "all_pairs")].iloc[0]
            assert abs(got["median"] - s["cross_encoder_unrestricted"][metric]["median"]) < 1e-12, (cond, metric)
            assert int(got["n_pairs"]) == s["n_pairs"]
            nd = conc[(conc.condition == cond) & (conc.metric == metric.upper()) & (conc.pair_set == "nondegenerate_pairs")].iloc[0]
            assert int(nd["n_pairs"]) == s["cross_encoder_nondegenerate_k_ge_5_both"]["n"], (cond, metric)
            assert abs(nd["median"] - s["cross_encoder_nondegenerate_k_ge_5_both"][metric]["median"]) < 1e-12, (cond, metric, "nd")
    conc.to_csv(out / "figA_concordance.csv", index=False)

    rows = []
    for cond in CONDITIONS:
        for m in MODELS:
            pe = me["stats"][cond]["per_encoder"][m]
            base = {"condition": cond, "condition_label": COND_LABEL[cond], "encoder": m, "encoder_label": ENC_LABEL[m], "n_fits": pe["n_fits"]}
            rows.append({**base, "indicator": "k", "value": pe["k_median"], "lo": pe["k_range"][0], "hi": pe["k_range"][1]})
            rows.append({**base, "indicator": "frac_k_le_4", "value": pe["frac_k_le_4"], "lo": np.nan, "hi": np.nan})
            rows.append({**base, "indicator": "largest_share", "value": pe["largest_share_median"], "lo": pe["largest_share_range"][0],
                         "hi": pe["largest_share_range"][1]})
    pd.DataFrame(rows).to_csv(out / "figA_degeneracy.csv", index=False)
    return {"figA_concordance_rows": len(conc), "figA_degeneracy_rows": len(rows)}


# =============================================================================================================
# Figure A -- field alignment
# =============================================================================================================
def fig_a_alignment(out: Path, src: list) -> dict:
    attr = json.loads((A_DIR / "field_attribution.json").read_text())
    stored = json.loads(STORED_DOM.read_text())
    src += [A_DIR / "field_attribution.json", STORED_DOM]
    all39 = attr["fields"]
    cells, summ = [], []

    def add(cond, m, per_field, scored_names, dom, source, all_fields_hint):
        fm = dom["per_field_ami_median"]
        for name in all_fields_hint:
            if name in per_field and name in scored_names:
                status, ami = "scored", per_field[name]
            elif name in scored_names:                               # cannot happen; kept as a guard
                status, ami = "not_scored", np.nan
            else:
                status, ami = source["missing_status"], np.nan
            cells.append({"condition": cond, "condition_label": COND_LABEL[cond], "encoder": m, "encoder_label": ENC_LABEL[m],
                          "field_source_name": name, "field_display_key": DISPLAY_ROW.get(name, name), "status": status, "ami_median": ami,
                          "is_dominant": bool(name == dom["modal_dominant_field"]) if status == "scored" else False,
                          "dominance_margin_median": dom["margin_over_runner_up_median"], "top_field_consistency": dom["modal_share"],
                          "n_fits": dom["n_fits"], "source": source["name"]})
        order = list(fm.items())
        summ.append({"condition": cond, "condition_label": COND_LABEL[cond], "encoder": m, "encoder_label": ENC_LABEL[m],
                     "dominant_field": dom["modal_dominant_field"], "dominant_field_display_key": DISPLAY_ROW.get(dom["modal_dominant_field"], dom["modal_dominant_field"]),
                     "top_ami_median": dom["top_ami_median"], "top_ami_min": dom["top_ami_range"][0], "top_ami_max": dom["top_ami_range"][1],
                     "runner_up_field": order[1][0] if len(order) > 1 else None, "runner_up_ami_median": order[1][1] if len(order) > 1 else np.nan,
                     "dominance_margin_median": dom["margin_over_runner_up_median"], "top_field_consistency": dom["modal_share"],
                     "n_fits": dom["n_fits"], "n_fields_scored": len(fm), "source": source["name"]})

    # Korea full template: the stored 29-field analysis (as manuscript Figure 2)
    for m in MODELS:
        dom = stored["cohorts"]["korea_matched_refits"][m]
        add("korea_full_template", m, dom["per_field_ami_median"], set(dom["per_field_ami_median"]), dom,
            {"name": "out/40_field_dominance/field_dominance.json (29 fields)", "missing_status": "not_scored"}, all39)
    # harmonized arms: rendered-field dominance of Analysis A (39-field set; fields removed from the arm = not rendered)
    for cond in ("korean_mimic_compatible", "harmonized_keep_therapy", "thickness_matched_random"):
        for m in MODELS:
            blk = attr["conditions"][cond][m]
            dom = blk["dominance_rendered_fields"]
            add(cond, m, dom["per_field_ami_median"], set(blk["rendered_fields"]), dom,
                {"name": "A field_attribution.json (rendered fields of the 39-field set)", "missing_status": "not_rendered"}, all39)
    # MIMIC-IV: stored 19-field analysis
    mimic_fields = list(stored["cohorts"]["mimic"]["bge-m3"]["per_field_ami_median"])
    for m in MODELS:
        dom = stored["cohorts"]["mimic"][m]
        add("mimic_iv", m, dom["per_field_ami_median"], set(dom["per_field_ami_median"]), dom,
            {"name": "out/40_field_dominance/field_dominance.json (19 MIMIC fields)", "missing_status": "not_rendered"}, mimic_fields)
    cells = pd.DataFrame(cells)

    # union of display rows; a Korean condition has no MIMIC-only row content and vice versa -> fill 'not_rendered' for absent (cond, row) combos
    rows = sorted(set(cells["field_display_key"]))
    full = []
    for cond in CONDITIONS:
        for m in MODELS:
            sub = cells[(cells.condition == cond) & (cells.encoder == m)]
            for key in rows:
                hit = sub[sub.field_display_key == key]
                if len(hit):
                    scored = hit[hit.status == "scored"]
                    full.append((scored if len(scored) else hit).iloc[0].to_dict())
                else:
                    full.append({"condition": cond, "condition_label": COND_LABEL[cond], "encoder": m, "encoder_label": ENC_LABEL[m],
                                 "field_source_name": key, "field_display_key": key, "status": "not_rendered", "ami_median": np.nan,
                                 "is_dominant": False, "dominance_margin_median": np.nan, "top_field_consistency": np.nan, "n_fits": np.nan,
                                 "source": "field not in this condition's field set"})
    full = pd.DataFrame(full)
    # attach the condition-level margin/consistency to every cell of that condition x encoder (used by the outline rule)
    dsum = pd.DataFrame(summ)
    full = full.drop(columns=["dominance_margin_median", "top_field_consistency", "n_fits"]).merge(
        dsum[["condition", "encoder", "dominance_margin_median", "top_field_consistency", "n_fits"]], on=["condition", "encoder"], how="left")
    full.to_csv(out / "figA_alignment_cells.csv", index=False)
    dsum.to_csv(out / "figA_dominance_summary.csv", index=False)
    return {"figA_alignment_rows": len(full), "figA_dominance_rows": len(dsum), "display_rows": len(rows)}


# =============================================================================================================
# Figure B
# =============================================================================================================
B_CATS = {  # (encoder) -> ordered arm ids with display label and the category they occupy in the Figure-3 layout
    "OpenAI 3-large": [("existing_full", "Unmodified"), ("existing_no_department", "Department\nremoved"),
                       ("existing_no_therapy", "Therapy block\nremoved"), ("existing_no_organism", "Organism\nremoved"),
                       (None, "Dominant-field\nvalue masked")],
    "OpenAI 3-small": [("B1a_unmodified_seed_only", "Unmodified:\nseed only"), ("B1b_unmodified_reembedded", "Unmodified:\nre-embedded"),
                       ("existing_no_organism", "Organism\nremoved"), ("B2_no_therapy", "Therapy block\nremoved"),
                       ("existing_no_department", "Department\nremoved"), (None, "Dominant-field\nvalue masked")],
    "Qwen3-0.6B": [("existing_full", "Unmodified"), ("existing_no_organism", "Organism\nremoved"),
                   ("B3_qwen3_no_department", "Department\nremoved"), ("existing_no_therapy", "Therapy block\nremoved"),
                   ("existing_mask_ab1_name_only", "First-antibiotic\nname masked")],
}
ENC_COMPACT = {"OpenAI 3-large": "text-3-large", "OpenAI 3-small": "text-3-small", "Qwen3-0.6B": "Qwen3-0.6B"}
DOM_LABEL = {"OpenAI 3-large": "index organism", "OpenAI 3-small": "treatment department", "Qwen3-0.6B": "first antibiotic named"}


def fig_b(out: Path, src: list) -> dict:
    p = B_DIR / "perturbation_matrix_completed.csv"
    src.append(p)
    d = pd.read_csv(p)
    keep = ["encoder", "dominant_field", "arm_id", "variant", "role", "source", "seed", "k", "ari_vs_reference", "ami_vs_reference", "noise_fraction"]
    pts = d[keep].copy()
    pts["encoder_compact"] = pts["encoder"].map(ENC_COMPACT)
    grid, order = [], []
    for enc, cats in B_CATS.items():
        for i, (arm, label) in enumerate(cats, start=1):
            g = d[(d.encoder == enc) & (d.arm_id == arm)] if arm else d.iloc[0:0]
            exists = len(g) > 0
            grid.append({"encoder": enc, "encoder_compact": ENC_COMPACT[enc], "dominant_field_label": DOM_LABEL[enc], "arm_id": arm if arm else "",
                         "x_label": label, "x_rank": i, "exists": exists, "n_seeds": int(len(g)),
                         "role": g["role"].iloc[0] if exists else "", "source": g["source"].iloc[0] if exists else "",
                         "ari_median": g["ari_vs_reference"].median() if exists else np.nan,
                         "ari_min": g["ari_vs_reference"].min() if exists else np.nan, "ari_max": g["ari_vs_reference"].max() if exists else np.nan})
            if exists:
                order.append((enc, arm, i, label))
    grid = pd.DataFrame(grid)
    # every merged row must land in exactly one grid cell
    lookup = {(e, a): (i, lab) for e, a, i, lab in order}
    pts["x_rank"] = [lookup[(e, a)][0] for e, a in zip(pts.encoder, pts.arm_id)]
    pts["x_label"] = [lookup[(e, a)][1] for e, a in zip(pts.encoder, pts.arm_id)]
    assert len(pts) == 164, len(pts)
    pts.to_csv(out / "figB_points.csv", index=False)
    grid.to_csv(out / "figB_arm_grid.csv", index=False)
    return {"figB_points": len(pts), "figB_grid_cells": len(grid), "figB_cells_not_existing": int((~grid["exists"]).sum())}


# =============================================================================================================
# Figure C
# =============================================================================================================
def fig_c(out: Path, src: list) -> dict:
    m = pd.read_csv(C_DIR / "mimic_ablation.csv")
    x = pd.read_csv(C_DIR / "mimic_ablation_cross_encoder.csv")
    qa = json.loads((C_DIR / "render_qa.json").read_text())
    src += [C_DIR / "mimic_ablation.csv", C_DIR / "mimic_ablation_cross_encoder.csv", C_DIR / "render_qa.json"]

    rows = []
    for _, r in m[m.arm.isin(C_ARMS)].iterrows():
        for ref, suf in (("unmodified re-embedded arm (same seed)", "c0"), ("stored reference partition (same seed)", "stored")):
            for metric in ("ari", "ami"):
                rows.append({"arm": r["arm"], "encoder": r["encoder"], "seed": int(r["seed"]), "seed_in_1_10": bool(r["seed_in_1_10"]),
                             "reference": ref, "reference_key": suf, "metric": metric.upper(),
                             "value": r[f"{metric}_noise_excluded_vs_{suf}"], "n_both_non_noise": r[f"n_both_non_noise_vs_{suf}"],
                             "k": r["k"], "noise_frac": r["noise_frac"], "convention": "both-non-noise"})
    within = pd.DataFrame(rows)
    within.to_csv(out / "figC_within_points.csv", index=False)

    xr = []
    for _, r in x[x.arm.isin(C_ARMS)].iterrows():
        for metric in ("ari", "ami"):
            xr.append({"arm": r["arm"], "seed": int(r["seed"]), "seed_in_1_10": bool(r["seed_in_1_10"]), "metric": metric.upper(),
                       "value": r[f"{metric}_noise_excluded"], "k_bge_m3": r["k_bge_m3"], "k_qwen3_0_6b": r["k_qwen3_0_6b"],
                       "pair_nondegenerate_k_ge_5_both": bool(r["pair_nondegenerate_k_ge_5_both"]), "convention": "both-non-noise"})
    cross = pd.DataFrame(xr)
    cross.to_csv(out / "figC_cross_points.csv", index=False)

    # numerical noise floor: the unmodified re-embedded arm scored against the stored reference partition (same encoder, same seed)
    nf = within[(within.arm == "unmodified") & (within.reference_key == "stored")]
    floor = (nf.groupby(["encoder", "metric"])["value"].agg(median="median", min="min", max="max", n_seeds="count").reset_index())
    floor["definition"] = "unmodified re-embedded arm vs stored reference partition, both-non-noise, seeds 42 and 1-10"
    floor.to_csv(out / "figC_noise_floor.csv", index=False)

    er = []
    for arm in C_ARMS:
        a = qa["arms"][arm]
        er.append({"arm": arm, "lines_removed_per_doc_median": a["lines_removed_per_doc"]["median"], "lines_per_doc_before": a["lines_per_doc_before"]["median"],
                   "lines_per_doc_after": a["lines_per_doc_after"]["median"], "chars_removed_per_doc_median": a["chars_removed_per_doc"]["median"],
                   "fraction_of_lines_removed_median": a["fraction_of_lines_removed_median"], "role": a["role"]})
    pd.DataFrame(er).to_csv(out / "figC_edit_sizes.csv", index=False)
    assert within["seed"].nunique() == 11 and cross["seed"].nunique() == 11
    return {"figC_within_rows": len(within), "figC_cross_rows": len(cross), "figC_floor_rows": len(floor), "figC_edit_rows": len(er)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=SF / "tables")
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    src: list = []
    counts = {}
    counts.update(fig_a_concordance(out, src))
    counts.update(fig_a_alignment(out, src))
    counts.update(fig_b(out, src))
    counts.update(fig_c(out, src))
    prov = {"written": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"), "script": "scripts/71_followup_figure_data.py",
            "sys_executable": sys.executable, "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, **git_info(),
            "inputs": {str(p.relative_to(REV)): sha256(p) for p in dict.fromkeys(src)},
            "outputs": {p.name: sha256(p) for p in sorted(out.glob("fig*.csv"))}, "row_counts": counts}
    (out / "figure_data_provenance.json").write_text(json.dumps(prov, indent=2))
    print(json.dumps({"row_counts": counts, "outputs": sorted(p.name for p in out.glob("fig*.csv"))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
