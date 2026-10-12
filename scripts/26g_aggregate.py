#!/usr/bin/env python3
"""Task 26 -- aggregate Part 1 (decisive control) and Part 2 (multi-seed
medians) into final CSV/JSON tables. Reads-only from:
  out/21_published_variants/cluster/<variant>/result.json   (seed-42 k/ARI/AMI/Jaccard)
  out/26_controls/cosine_to_full.json                        (mean cosine to `full`)
  out/26_controls/a_variant_geometry_all20.json               (15-NN overlap, kNN F1)
  out/26_controls/pairwise_ari.json                           (shuffle vs shuffle_seed2 vs fixed_permutation)
  out/25_audit4/seeds/<variant>_seed<1-10>.json                (multi-seed ARI/Jaccard)
Writes:
  out/26_controls/part1_decisive_control_table.csv/json
  out/26_controls/part2_multiseed_table.csv/json
Login-node safe (pure JSON/CSV aggregation, no scientific-stack imports needed).
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import pandas as pd

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
CLUSTER_DIR = REV / "out/21_published_variants/cluster"
CTRL_DIR = REV / "out/26_controls"
SEEDS_DIR = REV / "out/25_audit4/seeds"

ALL_20 = [
    "full", "index_window",
    "shuffle_order", "no_headers", "json_format", "null_dropped", "no_definitions",
    "fixed_permutation", "shuffle_order_seed2",
    "no_organism", "no_resistance", "no_therapy", "no_department", "no_year",
    "host_source_severity_only", "no_organism_no_resistance", "no_resistance_strict",
    "no_organism_no_resistance_strict", "no_dates", "no_summary_tokens",
]


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def part1_table():
    cosine = load_json(CTRL_DIR / "cosine_to_full.json") or {}
    geom = load_json(CTRL_DIR / "a_variant_geometry_all20.json") or {}
    pairwise = load_json(CTRL_DIR / "pairwise_ari.json") or {}

    rows = []
    for v in ["full", "shuffle_order", "shuffle_order_seed2", "fixed_permutation", "json_format"]:
        rp = CLUSTER_DIR / v / "result.json"
        r = load_json(rp)
        if r is None:
            rows.append({"variant": v, "status": "cluster result not yet available"})
            continue
        ap = r["agreement_vs_published"]
        g = geom.get(v, {})
        c = cosine.get(v, {})
        rows.append({
            "variant": v,
            "k": r["n_clusters"], "noise_frac": r["noise_frac"],
            "ari_all_row_vs_published": ap["ari_all_row"], "ari_noise_excl_vs_published": ap["ari_noise_excl"],
            "ami_all_row_vs_published": ap["ami_all_row"], "ami_noise_excl_vs_published": ap["ami_noise_excl"],
            "cluster0_jaccard": r["best_match_jaccard_vs_published"].get("0"),
            "mean_cosine_to_full": c.get("mean"),
            "knn15_overlap_with_full_mean": g.get("knn15_overlap_with_full_mean"),
            "label_transfer_macro_f1": g.get("label_transfer_macro_f1"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(CTRL_DIR / "part1_decisive_control_table.csv", index=False)

    summary = {"per_variant": rows, "pairwise_ari_between_shuffles_and_fixed_permutation": pairwise}
    (CTRL_DIR / "part1_decisive_control_table.json").write_text(json.dumps(summary, indent=2))
    print(df.to_string(index=False))
    print(json.dumps(pairwise, indent=2))
    return df, pairwise


def part2_table():
    rows = []
    per_cluster_by_variant = {}
    for v in ALL_20:
        # Exact match on `<variant>_seed<digits>.json` -- a plain glob
        # (`{v}_seed*.json`) wrongly matches a DIFFERENT variant's files
        # when one variant name is a prefix of another (e.g. "shuffle_order"
        # is a prefix of "shuffle_order_seed2": the glob "shuffle_order_seed*"
        # also matches "shuffle_order_seed2_seed1.json"). Anchored regex
        # avoids that grouping-key bug.
        pattern = re.compile(rf"^{re.escape(v)}_seed(\d+)\.json$")
        files = sorted(
            (p for p in SEEDS_DIR.glob("*.json") if pattern.match(p.name)),
            key=lambda p: int(pattern.match(p.name).group(1)),
        )
        if not files:
            rows.append({"variant": v, "n_seeds": 0, "note": "no seed sweep data"})
            continue
        recs = [load_json(f) for f in files]
        ks = [r["k"] for r in recs]
        ari_all = [r["agreement_vs_published"]["ari_all_row"] for r in recs]
        ari_ne = [r["agreement_vs_published"]["ari_noise_excl"] for r in recs]
        rows.append({
            "variant": v, "n_seeds": len(recs),
            "k_median": statistics.median(ks), "k_min": min(ks), "k_max": max(ks),
            "ari_all_row_median": statistics.median(ari_all),
            "ari_all_row_min": min(ari_all), "ari_all_row_max": max(ari_all),
            "ari_noise_excl_median": statistics.median(ari_ne),
            "ari_noise_excl_min": min(ari_ne), "ari_noise_excl_max": max(ari_ne),
        })
        # per-cluster Jaccard median/min/max across seeds
        clusters = sorted({c for r in recs for c in r["jaccard_vs_published"].keys()}, key=int)
        pc = {}
        for c in clusters:
            vals = [r["jaccard_vs_published"].get(c, 0.0) for r in recs]
            pc[c] = {"median": statistics.median(vals), "min": min(vals), "max": max(vals)}
        per_cluster_by_variant[v] = pc

    df = pd.DataFrame(rows)
    df.to_csv(CTRL_DIR / "part2_multiseed_table.csv", index=False)
    (CTRL_DIR / "part2_per_cluster_jaccard.json").write_text(json.dumps(per_cluster_by_variant, indent=2))
    print(df.to_string(index=False))
    return df, per_cluster_by_variant


if __name__ == "__main__":
    print("=== Part 1: decisive control ===")
    part1_table()
    print("\n=== Part 2: multi-seed medians ===")
    part2_table()
