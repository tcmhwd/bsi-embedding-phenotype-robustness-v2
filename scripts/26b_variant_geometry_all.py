#!/usr/bin/env python3
"""Task 26 Part 2 -- kNN label-transfer / 15-NN geometry for every one of
the 20 Task-21 variants (the 12 `a_variant_geometry.py` already covers,
the 6 AUDIT_4 routing flagged as missing, and the 2 new
AUDIT_4-C1 controls: fixed_permutation, shuffle_order_seed2).

Reuses `out/25_audit4/a_variant_geometry.py`'s functions VERBATIM (imported
by file path, not reimplemented) -- see that module's docstring for the
method. Writes to `out/26_controls/` only, so the original audit evidence
file `out/25_audit4/a_variant_geometry.json` (12 variants) is left
untouched; this file is the complete 20-variant superset.

Run: bsi_pheno interpreter, cpu_partition (see slurm/26b_variant_geometry_all.sh).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
sys.path.insert(0, str(REV / "scripts"))

spec = importlib.util.spec_from_file_location("a_variant_geometry", REV / "out/25_audit4/a_variant_geometry.py")
avg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(avg)

OUT_DIR = REV / "out" / "26_controls"
DEST = OUT_DIR / "a_variant_geometry_all20.json"

ALL_20 = [
    "full", "index_window",
    "shuffle_order", "no_headers", "json_format", "null_dropped", "no_definitions",
    "fixed_permutation", "shuffle_order_seed2",
    "no_organism", "no_resistance", "no_therapy", "no_department", "no_year",
    "host_source_severity_only", "no_organism_no_resistance", "no_resistance_strict",
    "no_organism_no_resistance_strict", "no_dates", "no_summary_tokens",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = avg.c18.load_published_frame()
    ids = frame["admission_ID"].to_numpy()
    pub = frame[avg.c18.PUBLISHED_CLUSTER_COL].to_numpy()
    groups = frame["patient_ID"].to_numpy()
    keep = pub != -1

    Xfull = avg.load_variant("full", ids)
    Gfull = avg.knn_graph(Xfull, avg.K_GRAPH)
    Gfull_sets = [set(r.tolist()) for r in Gfull]

    # Reuse an already-finished, previously-verified result for a variant
    # instead of recomputing (identical method, saves ~45s each) -- but
    # verify the source file actually covers it before trusting it.
    existing = {}
    prev_path = REV / "out/25_audit4/a_variant_geometry.json"
    if prev_path.exists():
        existing = json.loads(prev_path.read_text())

    results = {}
    for v in ALL_20:
        if v in existing:
            print(f"[26b] {v}: reusing out/25_audit4/a_variant_geometry.json (already computed)", flush=True)
            results[v] = existing[v]
            continue
        t0 = time.time()
        X = avg.load_variant(v, ids)
        G = avg.knn_graph(X, avg.K_GRAPH)
        overlap = [len(Gfull_sets[i].intersection(G[i].tolist())) / avg.K_GRAPH for i in range(len(ids))]
        import numpy as np
        overlap = np.array(overlap)
        m0 = pub == 0
        purity0 = float(np.mean(m0[G[m0]].mean(axis=1)))
        per_cl_overlap = {int(c): float(overlap[pub == c].mean()) for c in sorted(set(pub.tolist()))}
        macro, per = avg.knn_transfer(X[keep], pub[keep], groups[keep])
        results[v] = dict(
            knn15_overlap_with_full_mean=float(overlap.mean()),
            knn15_overlap_with_full_median=float(np.median(overlap)),
            knn15_overlap_by_published_cluster=per_cl_overlap,
            cluster0_knn15_purity=purity0,
            label_transfer_macro_f1=macro, label_transfer_per_cluster_f1=per,
            wall_sec=time.time() - t0,
        )
        print(v, json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in results[v].items()
                             if not isinstance(x, dict)}), "cl0 F1", round(per.get(0, float('nan')), 3),
              "cl3 F1", round(per.get(3, float('nan')), 3), flush=True)
        DEST.write_text(json.dumps(results, indent=2))

    DEST.write_text(json.dumps(results, indent=2))
    print(f"[26b] wrote {DEST} ({len(results)} variants)", flush=True)


if __name__ == "__main__":
    main()
