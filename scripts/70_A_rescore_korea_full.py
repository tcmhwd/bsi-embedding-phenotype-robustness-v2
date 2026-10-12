#!/usr/bin/env python3
"""Analysis A, supporting deliverable -- Korea-full partitions re-scored over the EXPANDED field set (post hoc).

WHY.  scripts/40_field_dominance.py scores every Korean partition against 29 rendered fields.  Six fields the narrative renders
are silently dropped there by column-name mismatches against the dataset CSV ([warn] path, 40_field_dominance.py:171-174):
    entry_mode, index_bsi_3gcr, index_bsi_mrsa, index_bsi_vre, index_bsi_carbar, index_bsi_polymicrobial.
scripts/64_harmonized_fields.py (Analysis A) attributes over an EXPANDED list that includes them (39 fields = the 29 + those six
+ four further raw fields: index_bsi_ceftriaxoner, index_bsi_crab, comorbid_infectious, charlson_class).  This script re-scores the
STORED Korea-full partitions over exactly that expanded set, like-for-like with Analysis A, so the Korea-full anchor of Figure 2
can be stated over the same field set as the harmonized arms.  Nothing is re-clustered or re-embedded.

FIELD SET AND FUNCTIONS are imported from 64_harmonized_fields (build_fields_global -> FIELDS; summarise_group) and from
40_field_dominance (field_ami, summarise_dominance) -- not re-derived.

PARTITIONS (all stored):
  matched_refits : out/39_audit6/matched/draw{01..10}.npz, keys '<encoder>|<seed>' (10 draws x 5 seeds = 50 fits per encoder), scored on the
                   draw subset exactly as 40_field_dominance does.  Requested: bge-m3, qwen3-0.6b.  Also scored (extension, flagged
                   requested=false): openai-3small, openai-3large -- the other two encoders of Figure 2 in the same npz files.
  whole_corpus   : out/18_encoder_dec/recluster/{bge-m3,qwen3-0.6b,openai-3small}/labels.npz  (32,329 rows; row order asserted equal to the
                   embedding id order) and, for 3-large, the originally derived partition pipeline_new/out_pre/phenotype_clusters_labels.csv
                   (aligned by admission_ID).  One fit each -> 'consistency across fits' is not defined and is written as null.

COMPARISON BLOCK (vs the 29-field stored values in out/40_field_dominance/field_dominance.json):
  * reproduction: the re-scored per-fit AMIs restricted to the 29 stored field names are re-summarised with 40_field_dominance.summarise_dominance;
    max |difference| of the 29 per-field medians must be ~1e-9 (check recorded; a larger value is reported, not hidden);
  * change: expanded-set values minus 29-field stored values (top field, top AMI, runner-up, margin, consistency).
  * cross-checks against Analysis A's own field_attribution.json (per-fit AMIs, matched and whole-corpus, bge-m3 / qwen3-0.6b).
NUMBERS ONLY -- no interpretation is written.

Env: /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3       Workers: SLURM_CPUS_PER_TASK (fork pool; the field table is copy-on-write).
"""
import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import collections
import importlib
import json
import multiprocessing as mp
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
f40 = importlib.import_module("40_field_dominance")
m64 = importlib.import_module("64_harmonized_fields")

REV = SCRIPT_DIR.parent
DRAWS = REV / "out/39_audit6/matched"
RECLUSTER = REV / "out/18_encoder_dec/recluster"
PUBLISHED = f40.PUBLISHED
STORED_DOM = REV / "out/40_field_dominance/field_dominance.json"
A_ATTR = REV / "out/followup_analyses/A_harmonized_template/field_attribution.json"
OUT_DIR = REV / "out/followup_analyses/A_harmonized_template"
THIS_FILES = ["scripts/70_A_rescore_korea_full.py", "slurm/70_A_rescore_korea_full.sh"]

SIX_DROPPED = ["entry_mode", "index_bsi_3gcr", "index_bsi_mrsa", "index_bsi_vre", "index_bsi_carbar", "index_bsi_polymicrobial"]
ENCODERS = ["bge-m3", "qwen3-0.6b", "openai-3small", "openai-3large"]
REQUESTED_MATCHED = {"bge-m3", "qwen3-0.6b"}
DISPLAY = {"bge-m3": "BGE-M3", "qwen3-0.6b": "Qwen3-0.6B", "openai-3small": "text-3-small", "openai-3large": "text-3-large"}
DRAW_IDS = list(range(1, 11))
SEEDS = [1, 2, 3, 4, 5]
TOL = 1e-9


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def git_info() -> dict:
    def run(*args) -> str:
        return subprocess.run(["git", "-C", str(REV), *args], capture_output=True, text=True, check=True).stdout.strip()

    st = run("status", "--porcelain", "--", *THIS_FILES)
    return {"git_commit": run("rev-parse", "HEAD"), "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(st), "git_dirty_detail": st.splitlines()[:20], "git_dirty_scope": THIS_FILES}


# ------------------------------------------------------------------------------------------------ jobs (fork workers)
WHOLE = {}     # encoder -> label vector in embedding order (filled before the pool forks)


def _job(job):
    kind, enc, key = job
    if kind == "whole":
        lab = WHOLE[enc]
        fields = m64.FIELDS
    else:
        d, s = key
        z = np.load(DRAWS / f"draw{d:02d}.npz")
        subset, lab = z["subset"], z[f"{enc}|{s}"]
        fields = {n: v.iloc[subset].reset_index(drop=True) for n, v in m64.FIELDS.items()}
    return (kind, enc, key, f40.field_ami(lab, fields))


def load_whole_partitions(ids: list) -> dict:
    out, src = {}, {}
    for enc in ("bge-m3", "qwen3-0.6b", "openai-3small"):
        p = RECLUSTER / enc / "labels.npz"
        z = np.load(p)
        adm = [int(i) for i in z["admission_ID"]]
        if adm != ids:
            raise SystemExit(f"[70] {p}: admission_ID order differs from the embedding id order")
        out[enc] = z["labels"].astype(np.int64)
        src[enc] = str(p.relative_to(REV))
    pub = pd.read_csv(PUBLISHED, usecols=["admission_ID", "phenotype_cluster"]).drop_duplicates("admission_ID").set_index("admission_ID")
    missing = [i for i in ids if i not in pub.index]
    if missing:
        raise SystemExit(f"[70] {len(missing)} embedding ids absent from the published label file")
    out["openai-3large"] = pub.loc[ids, "phenotype_cluster"].to_numpy().astype(np.int64)
    src["openai-3large"] = str(PUBLISHED) + " (originally derived partition, aligned by admission_ID)"
    return out, src


# ------------------------------------------------------------------------------------------------ summaries
def block(per_fit: list, names: list) -> dict:
    blk = m64.summarise_group(per_fit, names)
    order = list(blk["per_field_ami_median"].items())
    blk["top_field_by_median_ami"] = order[0][0]
    blk["top_field_median_ami_across_fits"] = order[0][1]
    return blk


def dropped_stats(per_fit: list) -> dict:
    out = {}
    for f in SIX_DROPPED:
        v = np.array([s[f] for s in per_fit], dtype=float)
        out[f] = {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}
    return out


def compare_to_stored(per_fit: list, stored_blk: dict, exp_blk: dict, names29: list) -> dict:
    """Reproduction (29-field restricted re-scoring vs stored) and change (expanded vs stored)."""
    restricted = f40.summarise_dominance([{n: s[n] for n in names29} for s in per_fit])
    st_fm = stored_blk["per_field_ami_median"]
    diffs = {n: abs(restricted["per_field_ami_median"][n] - st_fm[n]) for n in names29}
    worst = max(diffs, key=diffs.get)
    st_order = list(st_fm.items())
    st_runner = st_order[1] if len(st_order) > 1 else (None, None)
    ex_order = list(exp_blk["per_field_ami_median"].items())
    return {
        "reproduction_29_fields": {
            "n_shared_fields": len(names29), "max_abs_diff_per_field_ami_median": diffs[worst], "field_of_max_diff": worst,
            "abs_diff_top_ami_median": abs(restricted["top_ami_median"] - stored_blk["top_ami_median"]),
            "abs_diff_margin_median": abs(restricted["margin_over_runner_up_median"] - stored_blk["margin_over_runner_up_median"]),
            "abs_diff_modal_share": abs(restricted["modal_share"] - stored_blk["modal_share"]),
            "modal_field_equal": restricted["modal_dominant_field"] == stored_blk["modal_dominant_field"],
            "winner_counts_equal": dict(restricted["winner_counts"]) == dict(stored_blk["winner_counts"]),
            "within_1e-9": bool(diffs[worst] <= TOL),
            "restricted_top_field": restricted["modal_dominant_field"], "stored_top_field": stored_blk["modal_dominant_field"]},
        "stored_29_field_values": {
            "modal_dominant_field": stored_blk["modal_dominant_field"], "modal_share": stored_blk["modal_share"],
            "top_ami_median": stored_blk["top_ami_median"], "margin_over_runner_up_median": stored_blk["margin_over_runner_up_median"],
            "runner_up_by_median_ami": st_runner[0], "runner_up_median_ami": st_runner[1]},
        "expanded_minus_stored": {
            "top_field_expanded": exp_blk["modal_dominant_field"], "top_field_stored": stored_blk["modal_dominant_field"],
            "modal_top_field_same": exp_blk["modal_dominant_field"] == stored_blk["modal_dominant_field"],
            "top_ami_median_diff": exp_blk["top_ami_median"] - stored_blk["top_ami_median"],
            "margin_median_diff": exp_blk["margin_over_runner_up_median"] - stored_blk["margin_over_runner_up_median"],
            "modal_share_diff": exp_blk["modal_share"] - stored_blk["modal_share"],
            "runner_up_expanded": ex_order[1][0] if len(ex_order) > 1 else None,
            "runner_up_stored": st_runner[0],
            "runner_up_same": (ex_order[1][0] if len(ex_order) > 1 else None) == st_runner[0],
            "n_fields_expanded": len(exp_blk["per_field_ami_median"]), "n_fields_stored": len(st_fm)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR, help="(testing) override the output directory")
    args = ap.parse_args()
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    m64.build_fields_global()                     # 39-field expanded set, embedding row order
    fields = list(m64.FIELDS)
    stored = json.loads(STORED_DOM.read_text())
    st_matched = stored["cohorts"]["korea_matched_refits"]
    st_whole = stored["cohorts"]["korea_published_wholecorpus"]["ami_by_field"]
    names29 = [n for n in st_matched["bge-m3"]["per_field_ami_median"]]
    assert set(names29) == set(st_matched["qwen3-0.6b"]["per_field_ami_median"]) and len(names29) == 29
    missing29 = [n for n in names29 if n not in fields]
    assert not missing29, f"stored 29-field names absent from the expanded set: {missing29}"
    assert all(f in fields for f in SIX_DROPPED)
    extra_other = [f for f in fields if f not in names29 and f not in SIX_DROPPED]
    print(f"[70] expanded set {len(fields)} fields = {len(names29)} stored + {len(SIX_DROPPED)} previously dropped + {len(extra_other)} further "
          f"({extra_other}); workers={workers}", flush=True)

    ids = m64.load_ids()
    whole, whole_src = load_whole_partitions(ids)
    WHOLE.update(whole)

    jobs = [("matched", e, (d, s)) for e in ENCODERS for d in DRAW_IDS for s in SEEDS] + [("whole", e, "whole") for e in ENCODERS]
    res = collections.defaultdict(dict)
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=workers) as pool:
        for i, (kind, enc, key, ami) in enumerate(pool.imap_unordered(_job, jobs, chunksize=1)):
            res[(kind, enc)][key] = ami
            if (i + 1) % 25 == 0:
                print(f"[70] {i + 1}/{len(jobs)} partitions scored ({time.time() - t0:.0f}s)", flush=True)

    a_attr = json.loads(A_ATTR.read_text()) if A_ATTR.exists() else None
    partitions = {"matched_refits": {}, "whole_corpus": {}}
    rows = []
    checks = {"tolerance": TOL, "matched_29_field_max_abs_diff_per_encoder": {}, "whole_3large_vs_stored_ami_by_field_max_abs_diff": None,
              "vs_A_field_attribution_json": {}}

    for enc in ENCODERS:
        byk = res[("matched", enc)]
        keys = sorted(byk)
        assert len(keys) == 50, (enc, len(keys))
        per_fit = [byk[k] for k in keys]
        exp = block(per_fit, fields)
        cmp_ = compare_to_stored(per_fit, st_matched[enc], exp, names29)
        partitions["matched_refits"][enc] = {
            "display": DISPLAY[enc], "requested_in_task": enc in REQUESTED_MATCHED, "n_fits": len(per_fit),
            "n_draws": len(DRAW_IDS), "n_seeds": len(SEEDS), "source": f"out/39_audit6/matched/draw01..10.npz['{enc}|1..5']",
            "dominance_expanded_fields": exp, "ami_six_previously_dropped_fields": dropped_stats(per_fit),
            "comparison_vs_29_field_stored": cmp_,
            "per_fit_ami": {f"d{k[0]:02d}_s{k[1]}": v for k, v in zip(keys, per_fit)}}
        checks["matched_29_field_max_abs_diff_per_encoder"][enc] = cmp_["reproduction_29_fields"]["max_abs_diff_per_field_ami_median"]
        if a_attr and enc in a_attr["conditions"]["korea_full_template"]:
            blk = a_attr["conditions"]["korea_full_template"][enc]
            mx = 0.0
            for k, v in zip(keys, per_fit):
                ref = blk["per_fit_ami"][f"d{k[0]:02d}_s{k[1]}"]
                mx = max(mx, max(abs(v[f] - ref[f]) for f in fields))
            checks["vs_A_field_attribution_json"][f"matched|{enc}"] = {"max_abs_diff_over_50_fits_x_39_fields": mx}
        er = {"partition_set": "matched_refits", "encoder": enc, "display": DISPLAY[enc], "requested_in_task": enc in REQUESTED_MATCHED,
              "n_fits": len(per_fit)}
        er.update(row_from(exp, cmp_, partitions["matched_refits"][enc]["ami_six_previously_dropped_fields"], consistency_defined=True))
        rows.append(er)

    for enc in ENCODERS:
        ami = res[("whole", enc)]["whole"]
        exp = block([ami], fields)
        rec = {"display": DISPLAY[enc], "requested_in_task": True, "n_fits": 1, "source": whole_src[enc],
               "n_clusters_non_noise": int(len(set(whole[enc].tolist()) - {-1})), "noise_fraction": float((whole[enc] == -1).mean()),
               "dominance_expanded_fields": exp, "ami_six_previously_dropped_fields": dropped_stats([ami]), "ami_by_field": ami}
        if enc == "openai-3large":
            st_ord = sorted(st_whole.items(), key=lambda kv: -kv[1])
            diffs = {n: abs(ami[n] - st_whole[n]) for n in names29}
            worst = max(diffs, key=diffs.get)
            checks["whole_3large_vs_stored_ami_by_field_max_abs_diff"] = {"max_abs_diff": diffs[worst], "field": worst, "within_1e-9": bool(diffs[worst] <= TOL)}
            ex_order = sorted(ami.items(), key=lambda kv: -kv[1])
            rec["comparison_vs_29_field_stored"] = {
                "reproduction_29_fields": {"max_abs_diff_per_field_ami": diffs[worst], "field_of_max_diff": worst, "within_1e-9": bool(diffs[worst] <= TOL)},
                "stored_29_field_values": {"top_field": st_ord[0][0], "top_ami": st_ord[0][1], "runner_up": st_ord[1][0], "runner_up_ami": st_ord[1][1],
                                           "margin": st_ord[0][1] - st_ord[1][1]},
                "expanded_minus_stored": {"top_field_expanded": ex_order[0][0], "top_field_stored": st_ord[0][0], "top_field_same": ex_order[0][0] == st_ord[0][0],
                                          "top_ami_diff": ex_order[0][1] - st_ord[0][1], "margin_diff": (ex_order[0][1] - ex_order[1][1]) - (st_ord[0][1] - st_ord[1][1]),
                                          "runner_up_expanded": ex_order[1][0], "runner_up_stored": st_ord[1][0]}}
        elif a_attr and enc in a_attr["conditions"]["korea_full_template"]:
            ref = a_attr["conditions"]["korea_full_template"][enc]["whole_corpus_seed42_ami"]
            mx = max(abs(ami[f] - ref[f]) for f in fields)
            checks["vs_A_field_attribution_json"][f"whole|{enc}"] = {"max_abs_diff_over_39_fields": mx}
            rec["comparison_vs_29_field_stored"] = {"note": "no stored 29-field whole-corpus AMI exists for this encoder in field_dominance.json; "
                                                            "cross-check against Analysis A's field_attribution.json only (see checks)"}
        else:
            rec["comparison_vs_29_field_stored"] = {"note": "no stored whole-corpus field AMI for this encoder in field_dominance.json or field_attribution.json"}
        partitions["whole_corpus"][enc] = rec
        er = {"partition_set": "whole_corpus", "encoder": enc, "display": DISPLAY[enc], "requested_in_task": True, "n_fits": 1}
        er.update(row_from(exp, None, rec["ami_six_previously_dropped_fields"], consistency_defined=False))
        cmpw = rec["comparison_vs_29_field_stored"]
        if "expanded_minus_stored" in cmpw:
            er["stored_top_field"] = cmpw["stored_29_field_values"]["top_field"]
            er["stored_top_ami_median"] = cmpw["stored_29_field_values"]["top_ami"]
            er["stored_runner_up_field"] = cmpw["stored_29_field_values"]["runner_up"]
            er["stored_margin_median"] = cmpw["stored_29_field_values"]["margin"]
            er["top_field_same_as_stored"] = cmpw["expanded_minus_stored"]["top_field_same"]
            er["top_ami_diff_expanded_minus_stored"] = cmpw["expanded_minus_stored"]["top_ami_diff"]
            er["margin_diff_expanded_minus_stored"] = cmpw["expanded_minus_stored"]["margin_diff"]
            er["max_abs_diff_29_shared_fields"] = cmpw["reproduction_29_fields"]["max_abs_diff_per_field_ami"]
        rows.append(er)

    max29 = max(checks["matched_29_field_max_abs_diff_per_encoder"].values())
    checks["matched_29_field_max_abs_diff_all_encoders"] = max29
    checks["matched_29_field_within_1e-9"] = bool(max29 <= TOL)
    checks["all_reproduction_checks_within_1e-9"] = bool(
        checks["matched_29_field_within_1e-9"] and (checks["whole_3large_vs_stored_ami_by_field_max_abs_diff"] or {}).get("within_1e-9", False))

    result = {"written": now_iso(), "script": "scripts/70_A_rescore_korea_full.py", "sys_executable": sys.executable, "python": platform.python_version(),
              "numpy": np.__version__, "pandas": pd.__version__, **git_info(), "seconds": round(time.time() - t0, 1), "workers": workers,
              "post_hoc": True, "note": "Post hoc re-scoring of stored partitions over the expanded field set; nothing re-clustered. Numbers only.",
              "field_set": {"n_expanded": len(fields), "expanded": fields, "n_stored_29": len(names29), "stored_29": names29,
                            "six_previously_dropped": SIX_DROPPED, "further_expanded_fields": extra_other,
                            "source_function": "64_harmonized_fields.build_fields_global (imported)"},
              "stored_reference": str(STORED_DOM.relative_to(REV)), "partitions": partitions, "checks": checks}
    tmp = out_dir / f".tmp_{os.getpid()}_korea_full_rescored_expanded_fields.json"
    tmp.write_text(json.dumps(result, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, out_dir / "korea_full_rescored_expanded_fields.json")
    df = pd.DataFrame(rows)
    tmpc = out_dir / f".tmp_{os.getpid()}_korea_full_rescored_expanded_fields.csv"
    df.to_csv(tmpc, index=False)
    os.replace(tmpc, out_dir / "korea_full_rescored_expanded_fields.csv")
    print(json.dumps({"n_partitions_scored": len(jobs), "rows": len(df), "checks": {k: v for k, v in checks.items() if k != "vs_A_field_attribution_json"},
                      "vs_A_field_attribution_json": checks["vs_A_field_attribution_json"], "seconds": result["seconds"]}, indent=2), flush=True)
    if not checks["all_reproduction_checks_within_1e-9"]:
        print("[70] WARNING: a reproduction check exceeds 1e-9 (recorded in the json; values are NOT adjusted)", flush=True)


def row_from(exp: dict, cmp_, dropped: dict, consistency_defined: bool) -> dict:
    order = list(exp["per_field_ami_median"].items())
    r = {"top_field": exp["modal_dominant_field"], "top_field_by_median_ami": exp["top_field_by_median_ami"],
         "top_ami_median": exp["top_ami_median"], "top_ami_min": exp["top_ami_range"][0], "top_ami_max": exp["top_ami_range"][1],
         "runner_up_field_by_median_ami": exp["runner_up_by_median_ami"], "runner_up_ami_median": exp["runner_up_median_ami"],
         "runner_up_modal_across_fits": exp["runner_up_modal_across_fits"],
         "dominance_margin_median": exp["margin_over_runner_up_median"],
         "top_field_consistency": exp["modal_share"] if consistency_defined else None,
         "winner_counts": json.dumps(exp["winner_counts"])}
    for f in SIX_DROPPED:
        r[f"ami_{f}_median"] = dropped[f]["median"]
        r[f"ami_{f}_min"] = dropped[f]["min"]
        r[f"ami_{f}_max"] = dropped[f]["max"]
    r["n_fields_scored"] = len(order)
    if cmp_ is not None:
        s, d, p = cmp_["stored_29_field_values"], cmp_["expanded_minus_stored"], cmp_["reproduction_29_fields"]
        r.update({"stored_top_field": s["modal_dominant_field"], "stored_top_ami_median": s["top_ami_median"],
                  "stored_runner_up_field": s["runner_up_by_median_ami"], "stored_runner_up_ami_median": s["runner_up_median_ami"],
                  "stored_margin_median": s["margin_over_runner_up_median"], "stored_top_field_consistency": s["modal_share"],
                  "top_field_same_as_stored": d["modal_top_field_same"], "runner_up_same_as_stored": d["runner_up_same"],
                  "top_ami_diff_expanded_minus_stored": d["top_ami_median_diff"], "margin_diff_expanded_minus_stored": d["margin_median_diff"],
                  "consistency_diff_expanded_minus_stored": d["modal_share_diff"],
                  "max_abs_diff_29_shared_fields": p["max_abs_diff_per_field_ami_median"],
                  "reproduction_within_1e-9": p["within_1e-9"]})
    return r


if __name__ == "__main__":
    main()
