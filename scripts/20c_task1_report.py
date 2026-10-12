#!/usr/bin/env python3
"""Task 1 -- Step 3: assemble the one-hot comparator report from
`20b_recluster_onehot.py`'s `result.json` for variants a/b (already
contain k, noise_frac, ARI/AMI vs published both all-row and
noise-excluded, and per-published-cluster best-match Jaccard -- computed
in the frozen env via `17_common.agreement`/`jaccard_for_cluster`, reused
verbatim here, not recomputed).

Answers the decisive question: does a one-hot encoding of the rendered
template fields reproduce the published 16-cluster partition? And does
adding the rendered structural features (variant b) reproduce it
substantially better than content alone (variant a)? If so, that pins the
mechanism to document structure, not clinical content.

Login-node safe: reads only small JSON files already on disk.
"""
import json
from pathlib import Path

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
OUT = REV / "out" / "20_onehot"
RECLUSTER = OUT / "recluster"
FEAT_DIR = OUT / "features"

HIGHLIGHT_CLUSTERS = ["0", "1", "3", "6", "7"]


def load(variant: str) -> dict:
    p = RECLUSTER / variant / "result.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    import pandas as pd

    res = {v: load(v) for v in ["a", "b"]}
    missing = [v for v, r in res.items() if r is None]
    if missing:
        print(f"WARNING: variant(s) {missing} not yet clustered -- report incomplete")

    vec_manifest = json.loads((FEAT_DIR / ".vectorize_manifest.json").read_text())

    lines = ["# Task 1 -- the one-hot comparator (decisive test)", ""]
    lines.append(
        "Question: does a one-hot encoding of the rendered template fields reproduce the "
        "published 16-cluster partition? Variant (a) = template fields only (patient/clinical "
        "content). Variant (b) = (a) + rendered STRUCTURAL features (n_episode_blocks, "
        "byte_length, line_count, ast_line_count) -- properties of the document, not the patient."
    )
    lines.append("")
    lines.append(f"Feature dims: variant a = {vec_manifest['a']['n_cols']} cols "
                  f"({vec_manifest['a']['n_numeric_fields']} numeric incl. NA indicators, "
                  f"{vec_manifest['a']['n_categorical_fields']} categorical fields, "
                  f"{vec_manifest['a']['n_multi_hot_fields']} multi-hot fields); "
                  f"variant b = {vec_manifest['b']['n_cols']} cols "
                  f"(= a + {vec_manifest['b']['n_cols'] - vec_manifest['a']['n_cols']} structural).")
    lines.append("")

    rows = []
    for v in ["a", "b"]:
        r = res[v]
        if r is None:
            rows.append({"variant": v, "k": None, "noise_frac": None})
            continue
        ag = r["agreement_vs_published"]
        rows.append({
            "variant": v, "n_rows": r["n_rows"], "n_features": r["n_features"],
            "k": r["n_clusters"], "noise_frac": round(r["noise_frac"], 4),
            "ari_all_row": round(ag["ari_all_row"], 4), "ami_all_row": round(ag["ami_all_row"], 4),
            "ari_noise_excl": round(ag["ari_noise_excl"], 4), "ami_noise_excl": round(ag["ami_noise_excl"], 4),
            "n_both_nonnoise": ag["n_both_nonnoise"],
        })
    summary_df = pd.DataFrame(rows)
    lines.append("## Summary: k, noise, ARI/AMI vs published (both conventions labelled)")
    lines.append("")
    lines.append("```\n" + summary_df.to_string(index=False) + "\n```")
    lines.append("")

    lines.append("## Per-published-cluster best-match Jaccard, highlighted clusters (0,1,3,6,7)")
    lines.append("")
    jac_rows = []
    for v in ["a", "b"]:
        r = res[v]
        if r is None:
            continue
        jac = r["best_match_jaccard_vs_published"]
        jac_rows.append({"variant": v, **{f"cluster_{c}": round(jac.get(c, float("nan")), 4)
                                           for c in HIGHLIGHT_CLUSTERS}})
    if jac_rows:
        jac_df = pd.DataFrame(jac_rows)
        lines.append("```\n" + jac_df.to_string(index=False) + "\n```")
    lines.append("")

    lines.append("## Full per-published-cluster Jaccard (all 16 clusters)")
    lines.append("")
    full_rows = []
    for v in ["a", "b"]:
        r = res[v]
        if r is None:
            continue
        jac = r["best_match_jaccard_vs_published"]
        for c in sorted(jac.keys(), key=int):
            full_rows.append({"variant": v, "published_cluster": int(c), "jaccard": round(jac[c], 4)})
    if full_rows:
        full_df = pd.DataFrame(full_rows).pivot(index="published_cluster", columns="variant", values="jaccard")
        lines.append("```\n" + full_df.to_string() + "\n```")
        full_df.reset_index().to_csv(OUT / "task1_per_cluster_jaccard.csv", index=False)
    lines.append("")

    lines.append("## Verdict")
    lines.append("")
    if all(res[v] is not None for v in ["a", "b"]):
        ari_a = res["a"]["agreement_vs_published"]["ari_noise_excl"]
        ari_b = res["b"]["agreement_vs_published"]["ari_noise_excl"]
        k_a, k_b = res["a"]["n_clusters"], res["b"]["n_clusters"]
        THRESH = 0.50
        a_reproduces = ari_a >= THRESH
        b_reproduces = ari_b >= THRESH
        lines.append(f"Variant a (content only): ARI_noise_excl={ari_a:.4f}, k={k_a} "
                      f"({'REPRODUCES' if a_reproduces else 'does NOT reproduce'} the published partition "
                      f"at a {THRESH} ARI threshold).")
        lines.append(f"Variant b (content + structure): ARI_noise_excl={ari_b:.4f}, k={k_b} "
                      f"({'REPRODUCES' if b_reproduces else 'does NOT reproduce'} the published partition).")
        lines.append("")
        if b_reproduces and not a_reproduces:
            lines.append(
                "**Variant (b) reproduces the published partition substantially better than (a). "
                "This pins the mechanism to DOCUMENT STRUCTURE (episode-block count / narrative "
                "length / line count), not clinical content** -- consistent with the prior finding "
                "that cluster 0 is the >=3-episode group and that document length alone recovers it."
            )
        elif a_reproduces and b_reproduces:
            lines.append(
                "Both variants substantially reproduce the published partition -- clinical content "
                "alone (a one-hot of the same fields) is largely sufficient, and adding structural "
                "features does not materially change that. This would NOT support the "
                "document-structure mechanism claim on its own."
            )
        elif not a_reproduces and not b_reproduces:
            lines.append(
                "**Neither variant reproduces the published partition.** A one-hot encoding of the "
                "rendered fields -- content alone, or content plus explicit structural counts -- "
                "does not recover the embedding-derived clusters. This is a NEGATIVE result for the "
                "'lossy re-encoding of the table' hypothesis in its strong form: whatever the "
                "OpenAI embedding is keying on, it is not simply recoverable from a naive one-hot/"
                "standardized encoding of the same fields under this UMAP+HDBSCAN config -- report "
                "this plainly, it is not softened by the fact that (b) may still do relatively "
                "better than (a) on specific clusters (see per-cluster Jaccard above)."
            )
        else:
            lines.append(
                "Variant (a) reproduces the partition at this threshold but (b) does not -- an "
                "unexpected direction; inspect the per-cluster Jaccard table above before drawing "
                "conclusions."
            )
    else:
        lines.append("Incomplete -- one or both variants not yet clustered.")
    lines.append("")

    text = "\n".join(lines) + "\n"
    (OUT / "task1_FINDINGS.md").write_text(text)
    print(text)
    print(f"[20c] wrote {OUT / 'task1_FINDINGS.md'}")


if __name__ == "__main__":
    main()
