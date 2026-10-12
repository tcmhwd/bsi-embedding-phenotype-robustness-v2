#!/usr/bin/env python3
"""Is cluster instability just "small clusters are unstable"?

The paper's central claim is that surviving clusters are token-defined.  The
obvious deflationary alternative is far more boring: HDBSCAN's small
clusters are noisy, the big ones are not, and every stress axis is really
measuring cluster size.  If that were true the survival pattern would carry
no information about tokens at all.

This tests it directly on the sixteen published clusters: Spearman rho
between cluster size and each pre-specified stability axis, with a
percentile bootstrap CI and an exact permutation p-value.

Two honesty constraints are built in rather than left to the write-up:

  n = 16.  Spearman rho on sixteen points has wide intervals, so a
  non-significant rho is NOT evidence of no relationship, and the CI is
  reported alongside every point estimate so the reader cannot read the
  point estimate alone.  The permutation null is exact (all 16! orderings
  are not enumerated, but the sampled permutation distribution is the exact
  null for a fixed pair of vectors under exchangeability).

  Size and survival are compared directly too -- rank-biserial correlation
  of size against the binary ALL_AXES_pass -- because that, not rho against
  a continuous axis, is the comparison the deflationary story actually
  makes.

Reads the N7/N8-corrected matrix (`out/29_fixes/17e_rerun_verify/`), not the
superseded `out/17_correct_env/` one.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
MATRIX = ROOT / "out/29_fixes/17e_rerun_verify/survival_matrix.csv"
OUT = ROOT / "out/32_size_stability"

# Higher = more stable for all of these except `frac_dissolved`, which is
# inverted at read time so that every axis points the same way.
STABILITY_AXES = {
    "J_seed_mean": "S1 seed Jaccard (mean)",
    "J_seed_p05": "S1 seed Jaccard (5th pct)",
    "jaccard_mean": "S2 patient bootstrap Jaccard (mean)",
    "coassign_mean": "S2 co-assignment (mean)",
    "frac_retained": "S2 fraction NOT dissolved",
    "T_obs_hospital_10_to_20": "S4 transport, hospital 10 to 20",
    "T_obs_hospital_20_to_10": "S4 transport, hospital 20 to 10",
    "T_obs_period_early_to_late": "S5 transport, early to late",
    "T_obs_period_late_to_early": "S5 transport, late to early",
}
N_BOOT = 10_000
N_PERM = 100_000
SEED = 2026


def spearman_with_uncertainty(x: np.ndarray, y: np.ndarray, rng) -> dict:
    keep = ~(np.isnan(x) | np.isnan(y))
    x, y = x[keep], y[keep]
    n = len(x)
    if n < 4:
        return {"n": int(n), "rho": None, "note": "too few non-missing pairs"}

    rho, p_asymptotic = stats.spearmanr(x, y)

    # Percentile bootstrap over cluster-level pairs.  With n=16 this is the
    # honest width of the estimate, and it is reported unconditionally.
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(x[idx])) < 2 or len(np.unique(y[idx])) < 2:
            boot[i] = np.nan
            continue
        boot[i] = stats.spearmanr(x[idx], y[idx]).statistic
    boot = boot[~np.isnan(boot)]

    # Permutation null: shuffle one vector, recompute.  Exact null for these
    # two fixed vectors under exchangeability, and free of the asymptotic
    # approximation that a t-based p-value makes at n=16.
    observed = abs(rho)
    perm = np.empty(N_PERM)
    y_work = y.copy()
    for i in range(N_PERM):
        rng.shuffle(y_work)
        perm[i] = stats.spearmanr(x, y_work).statistic
    p_perm = float((np.abs(perm) >= observed).mean())

    return {
        "n": int(n),
        "rho": float(rho),
        "ci95_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "p_permutation_two_sided": p_perm,
        "p_asymptotic": float(p_asymptotic),
        "n_bootstrap_usable": int(len(boot)),
    }


def rank_biserial(sizes: np.ndarray, passed: np.ndarray, rng) -> dict:
    """Do the surviving clusters differ in size from the rest?

    Rank-biserial via Mann-Whitney, with an exact permutation p-value.  This
    is the comparison the "it is just size" story actually asserts.
    """
    a, b = sizes[passed], sizes[~passed]
    if len(a) == 0 or len(b) == 0:
        return {"n_pass": int(len(a)), "n_fail": int(len(b)), "note": "one group empty"}
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    effect = 2.0 * u.statistic / (len(a) * len(b)) - 1.0

    labels = passed.copy()
    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        rng.shuffle(labels)
        aa, bb = sizes[labels], sizes[~labels]
        perm[i] = 2.0 * stats.mannwhitneyu(aa, bb, alternative="two-sided").statistic / (
            len(aa) * len(bb)
        ) - 1.0
    p_perm = float((np.abs(perm) >= abs(effect)).mean())

    return {
        "n_pass": int(len(a)),
        "n_fail": int(len(b)),
        "median_size_pass": float(np.median(a)),
        "median_size_fail": float(np.median(b)),
        "rank_biserial": float(effect),
        "p_permutation_two_sided": p_perm,
        "p_mannwhitney": float(u.pvalue),
    }


def main() -> None:
    matrix = pd.read_csv(MATRIX)
    matrix["frac_retained"] = 1.0 - matrix["frac_dissolved"]
    rng = np.random.default_rng(SEED)

    result = {
        "source_matrix": str(MATRIX),
        "n_clusters": int(len(matrix)),
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "n_permutation": N_PERM,
        "size_range": [int(matrix["size"].min()), int(matrix["size"].max())],
        "size_median": float(matrix["size"].median()),
        "spearman_size_vs_axis": {},
    }

    sizes = matrix["size"].to_numpy(dtype=float)
    for column, label in STABILITY_AXES.items():
        if column not in matrix.columns:
            result["spearman_size_vs_axis"][column] = {"note": f"column absent: {column}"}
            continue
        block = spearman_with_uncertainty(sizes, matrix[column].to_numpy(dtype=float), rng)
        block["axis_label"] = label
        result["spearman_size_vs_axis"][column] = block
        rho = block.get("rho")
        print(
            f"[rho] {label:42} rho={rho:+.3f}  "
            f"CI [{block['ci95_bootstrap'][0]:+.3f}, {block['ci95_bootstrap'][1]:+.3f}]  "
            f"p_perm={block['p_permutation_two_sided']:.4f}"
            if rho is not None else f"[rho] {label:42} {block.get('note')}",
            flush=True,
        )

    # Nine axes are tested at once, so nominal p-values are corrected before
    # anything is read off them. Without this, reporting "2 of 9 axes reach
    # p < 0.05" would be selective reporting of exactly the kind this
    # analysis exists to guard against.
    tested = [
        (column, block["p_permutation_two_sided"])
        for column, block in result["spearman_size_vs_axis"].items()
        if block.get("rho") is not None
    ]
    order = np.argsort([p for _, p in tested])
    m = len(tested)
    running = 0.0
    holm = {}
    for rank, position in enumerate(order):
        column, p = tested[position]
        running = max(running, min(1.0, (m - rank) * p))
        holm[column] = running
    for column, adjusted in holm.items():
        result["spearman_size_vs_axis"][column]["p_holm"] = float(adjusted)
        result["spearman_size_vs_axis"][column]["significant_after_holm"] = bool(adjusted < 0.05)
    result["multiplicity"] = {
        "method": "Holm-Bonferroni across the stability axes tested here",
        "n_tests": m,
        "n_nominally_significant": int(sum(p < 0.05 for _, p in tested)),
        "n_significant_after_holm": int(sum(v < 0.05 for v in holm.values())),
        "min_p_nominal": float(min(p for _, p in tested)),
        "min_p_holm": float(min(holm.values())),
    }
    print("\n[multiplicity]", json.dumps(result["multiplicity"], indent=2), flush=True)

    if "ALL_AXES_pass" in matrix.columns:
        passed = matrix["ALL_AXES_pass"].astype(bool).to_numpy()
        result["size_vs_survival"] = rank_biserial(sizes, passed, rng)
        result["surviving_clusters"] = matrix.loc[passed, "cluster"].tolist()
        result["surviving_sizes"] = matrix.loc[passed, "size"].tolist()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "size_stability.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    matrix[["cluster", "size"] + [c for c in STABILITY_AXES if c in matrix.columns]].to_csv(
        OUT / "size_stability_inputs.csv", index=False
    )
    print()
    print(json.dumps(result.get("size_vs_survival", {}), indent=2))
    print(f"\n[size_stability] wrote {OUT}")


if __name__ == "__main__":
    main()
