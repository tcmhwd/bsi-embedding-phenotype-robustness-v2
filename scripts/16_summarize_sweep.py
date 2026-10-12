#!/usr/bin/env python3
"""Summarize the HDBSCAN sweep in out/16_c1_sweep/sweep_results.jsonl.

Reports:
  - top combinations by ari_all_rows_noise_as_label
  - top combinations by ari_both_nonnoise
  - combinations closest to the published signature (k=16, noise_fraction~0.0554)
"""
import json
from pathlib import Path

OUT_DIR = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision/out/16_c1_sweep")
RESULTS_JSONL = OUT_DIR / "sweep_results.jsonl"

PUBLISHED_K = 16
PUBLISHED_NOISE_FRAC = 1792 / 32329


def load_rows():
    rows = []
    with RESULTS_JSONL.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt(row):
    return (
        f"mcs={row['min_cluster_size']:<4} ms={str(row['min_samples']):<5} "
        f"method={row['cluster_selection_method']:<4} k={row['k']:<3} "
        f"noise={row['noise_fraction']:.4f} "
        f"ari_all={row['agreement']['ari_all_rows_noise_as_label']:.4f} "
        f"ari_nonoise={row['agreement']['ari_both_nonnoise']:.4f} "
        f"ami_all={row['agreement']['ami_all_rows_noise_as_label']:.4f}"
    )


def main():
    rows = load_rows()
    print(f"[INFO] {len(rows)} combinations loaded")

    print("\n=== Top 15 by ari_all_rows_noise_as_label ===")
    for row in sorted(rows, key=lambda r: -r["agreement"]["ari_all_rows_noise_as_label"])[:15]:
        print(fmt(row))

    print("\n=== Top 15 by ari_both_nonnoise ===")
    for row in sorted(rows, key=lambda r: -r["agreement"]["ari_both_nonnoise"])[:15]:
        print(fmt(row))

    print(f"\n=== Top 15 closest to published signature (k={PUBLISHED_K}, noise={PUBLISHED_NOISE_FRAC:.4f}) ===")
    def dist(r):
        return abs(r["k"] - PUBLISHED_K) * 10 + abs(r["noise_fraction"] - PUBLISHED_NOISE_FRAC)
    for row in sorted(rows, key=dist)[:15]:
        print(fmt(row))

    print(f"\n=== All combinations with k == {PUBLISHED_K} exactly ===")
    exact_k = [r for r in rows if r["k"] == PUBLISHED_K]
    print(f"({len(exact_k)} found)")
    for row in sorted(exact_k, key=lambda r: -r["agreement"]["ari_all_rows_noise_as_label"]):
        print(fmt(row))


if __name__ == "__main__":
    main()
