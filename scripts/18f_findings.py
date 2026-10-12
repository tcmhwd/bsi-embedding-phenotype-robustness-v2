#!/usr/bin/env python3
"""Task 18 -- assemble FINDINGS.md: the S6 (encoder) axis of the stress
battery against the ACTUAL published 16-cluster partition, applying the
PRESPEC S6 rule verbatim (`.superpowers/sdd/PRESPEC_stress_criteria.md`
SS2-3):

  PASS if in >=2 of 3 encoders, F1_enc(c) >= 0.60 AND
  F1_enc(c) >= F1_lookup(c) + 0.10

`F1_lookup(c)` is reused verbatim from `out/17_correct_env/survival_matrix.csv`
(`best_lookup_f1` column) -- not recomputed here.

Re-discoverability (Step 3, `18e_recluster_frozen.py`'s per-encoder
clustering) is reported next to the S6 table but never folds into the
PASS/FAIL verdict (PRESPEC: "not gated").

Login-node safe: reads only small JSON/CSV files already on disk.
"""
import importlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c18 = importlib.import_module("18_common")

ENCODERS = ["bge-m3", "qwen3-0.6b", "openai-3small"]
CEILING = "openai-3large"

F1_PASS_MIN = 0.60
F1_MARGIN_OVER_LOOKUP = 0.10
S6_MIN_ENCODERS_PASSING = 2


def _load_json(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main() -> None:
    import pandas as pd

    step0 = _load_json(c18.OUT / "step0_validate_corpus.json")
    knn = {enc: _load_json(c18.OUT / "s6_knn" / f"{enc}.json") for enc in ENCODERS + [CEILING]}
    recluster = {enc: _load_json(c18.OUT / "recluster" / enc / "result.json") for enc in ENCODERS}
    small_manifest = _load_json(c18.OUT / "embeddings" / "openai-3small" / ".openai_embed_manifest_3small.json")
    open_manifests = {
        enc: _load_json(c18.OUT / "embeddings" / enc / ".embed_manifest.json") for enc in ["bge-m3", "qwen3-0.6b"]
    }

    missing = [enc for enc in ENCODERS + [CEILING] if knn[enc] is None]
    if missing:
        print(f"WARNING: S6 kNN results missing for {missing} -- table below is INCOMPLETE", flush=True)

    survival = pd.read_csv(c18.SURVIVAL_MATRIX)
    lookup_f1 = dict(zip(survival["cluster"], survival["best_lookup_f1"]))
    all_axes_pass = dict(zip(survival["cluster"], survival["ALL_AXES_pass"]))
    cluster_size = dict(zip(survival["cluster"], survival["size"]))

    clusters = sorted(lookup_f1.keys())
    rows = []
    for c in clusters:
        row = {"cluster": c, "size": cluster_size[c], "ALL_AXES_pass_S1S2S4S5": bool(all_axes_pass[c]),
               "best_lookup_f1": lookup_f1[c]}
        pass_count = 0
        for enc in ENCODERS:
            f1 = None
            if knn[enc] is not None:
                f1 = knn[enc]["per_cluster_f1"].get(str(c))
            row[f"F1_{enc}"] = f1
            if f1 is not None:
                enc_pass = (f1 >= F1_PASS_MIN) and (f1 >= lookup_f1[c] + F1_MARGIN_OVER_LOOKUP)
                row[f"S6_pass_{enc}"] = enc_pass
                pass_count += int(enc_pass)
            else:
                row[f"S6_pass_{enc}"] = None
        row["F1_openai_3large_ceiling"] = knn[CEILING]["per_cluster_f1"].get(str(c)) if knn[CEILING] else None
        n_measured = sum(1 for enc in ENCODERS if row[f"F1_{enc}"] is not None)
        row["n_encoders_measured"] = n_measured
        row["n_encoders_passing"] = pass_count
        row["S6_pass"] = (pass_count >= S6_MIN_ENCODERS_PASSING) if n_measured == len(ENCODERS) else None
        for enc in ENCODERS:
            r = recluster.get(enc)
            if r is not None:
                row[f"reclust_J_{enc}"] = r["best_match_jaccard_vs_published"].get(str(c))
            else:
                row[f"reclust_J_{enc}"] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    c18.atomic_write_csv(df, c18.OUT / "s6_matrix.csv")

    n_s6_pass = int(df["S6_pass"].fillna(False).sum())
    n_measured_clusters = int(df["S6_pass"].notna().sum())

    cluster0 = df[df["cluster"] == 0].iloc[0].to_dict() if 0 in clusters else None

    lines = []
    lines.append("# Task 18 -- encoder axis (S6) against the actual published phenotypes")
    lines.append("")
    lines.append("Corpus: `reports_pilot_pre/` (32,329 `.report.md`, 2025-12-02 snapshot). "
                  "Reference labels: `pipeline_new/out_pre/phenotype_clusters_labels.csv` "
                  "(`phenotype_cluster`, 16 clusters + 1,792 noise).")
    lines.append("")
    lines.append("## Step 0 -- corpus validation")
    lines.append("")
    if step0:
        lines.append(f"n_sampled={step0['n_sampled']}, model={step0['model']}, "
                      f"cosine_mean={step0['cosine_mean']:.6f}, cosine_median={step0['cosine_median']:.6f}, "
                      f"cosine_min={step0['cosine_min']:.6f}, cosine_p05={step0['cosine_p05']:.6f}, "
                      f"cost=${step0['cost_usd']}, proceed={step0['proceed']}.")
        lines.append("")
        lines.append("**PASS** -- `reports_pilot_pre` is confirmed to be the corpus behind "
                      "`out_pre/case_report_embeddings.npy`; the rest of this task's assumption holds."
                      if step0["proceed"] else
                      "**FAIL** -- corpus assumption NOT validated, everything downstream is invalid.")
    else:
        lines.append("NOT RUN.")
    lines.append("")

    lines.append("## Step 1 -- embedding manifests")
    lines.append("")
    lines.append("| encoder | n_docs | dim | frac_truncated | docs_per_sec | device | cost_usd |")
    lines.append("|---|---|---|---|---|---|---|")
    for enc in ["bge-m3", "qwen3-0.6b"]:
        m = open_manifests[enc]
        if m:
            lines.append(f"| {enc} | {m['n_docs']} | {m['dim']} | {m['frac_truncated']} | "
                          f"{m.get('docs_per_sec')} | {m['device']} | 0 (local, no API) |")
        else:
            lines.append(f"| {enc} | NOT RUN | | | | | |")
    if small_manifest:
        ft = small_manifest.get("frac_truncated", "n/a")
        lines.append(f"| openai-3small | {small_manifest.get('n_docs_embedded')} | "
                      f"{small_manifest.get('dim')} | {ft} | n/a (API) | api | "
                      f"{small_manifest.get('cost_usd', 0):.4f} |")
    else:
        lines.append("| openai-3small | NOT RUN | | | | | |")
    lines.append("")
    total_api_cost = (small_manifest.get("cost_usd", 0) if small_manifest else 0)
    lines.append(f"Total API spend this task (embedding only, excludes Step 0's $"
                  f"{step0['cost_usd'] if step0 else 'n/a'}): ${total_api_cost:.4f} "
                  f"(hard cap was $2.00).")
    lines.append("")

    lines.append("## Step 2 -- S6 recoverability (kNN k=25 cosine, StratifiedGroupKFold(5, patient_ID), "
                  "noise excluded)")
    lines.append("")
    lines.append(f"PRESPEC S6 rule (verbatim): PASS if in >={S6_MIN_ENCODERS_PASSING} of "
                  f"{len(ENCODERS)} encoders, F1_enc(c) >= {F1_PASS_MIN} AND "
                  f"F1_enc(c) >= F1_lookup(c) + {F1_MARGIN_OVER_LOOKUP}.")
    lines.append("")
    show_cols = (["cluster", "size", "ALL_AXES_pass_S1S2S4S5"] +
                 [f"F1_{e}" for e in ENCODERS] + ["F1_openai_3large_ceiling", "best_lookup_f1", "S6_pass"])
    lines.append("```\n" + df[show_cols].round(4).to_string(index=False) + "\n```")
    lines.append("")
    lines.append(f"**{n_s6_pass}/{n_measured_clusters} measured published clusters PASS S6** "
                  f"(out of 16 total; {16 - n_measured_clusters} not measured because one or more "
                  "encoder results are missing).")
    lines.append("")

    lines.append("## Step 3 -- re-discoverability (reported, NOT gated per PRESPEC)")
    lines.append("")
    for enc in ENCODERS:
        r = recluster.get(enc)
        if r is None:
            lines.append(f"- {enc}: NOT RUN")
            continue
        ag = r["agreement_vs_published"]
        lines.append(f"- {enc}: k={r['n_clusters']}, noise_frac={r['noise_frac']:.4f} "
                      f"(n_noise={r['n_noise']}/{r['n_rows']}); "
                      f"ARI_all_row={ag['ari_all_row']:.4f}, ARI_noise_excl={ag['ari_noise_excl']:.4f}, "
                      f"AMI_all_row={ag['ami_all_row']:.4f}, AMI_noise_excl={ag['ami_noise_excl']:.4f}")
    lines.append("")
    reclust_cols = ["cluster", "size"] + [f"reclust_J_{e}" for e in ENCODERS]
    if all(recluster.get(e) is not None for e in ENCODERS):
        lines.append("```\n" + df[reclust_cols].round(4).to_string(index=False) + "\n```")
    else:
        lines.append("(incomplete -- one or more encoders' re-clustering not yet run)")
    lines.append("")

    lines.append("## Cluster 0 -- the finding this task exists to test")
    lines.append("")
    if cluster0:
        lines.append(f"n={int(cluster0['size'])}. Passed S1/S2/S4/S5 (seed, bootstrap, hospital "
                      f"transport, era transport): {cluster0['ALL_AXES_pass_S1S2S4S5']}. "
                      f"best_lookup_f1={cluster0['best_lookup_f1']:.4f} (token/organism lookup fails).")
        for enc in ENCODERS:
            f1 = cluster0[f"F1_{enc}"]
            f1s = f"{f1:.4f}" if f1 is not None else "NOT MEASURED"
            passv = cluster0[f"S6_pass_{enc}"]
            lines.append(f"- {enc}: F1={f1s}, individual-encoder pass "
                          f"(F1>={F1_PASS_MIN} AND F1>=lookup+{F1_MARGIN_OVER_LOOKUP})={passv}")
        ceil = cluster0["F1_openai_3large_ceiling"]
        lines.append(f"- OpenAI-3-large ceiling F1 = {ceil:.4f}" if ceil is not None else
                      "- OpenAI-3-large ceiling F1 = NOT MEASURED")
        lines.append("")
        s6v = cluster0["S6_pass"]
        if s6v is None:
            lines.append("**S6 verdict for cluster 0: NOT YET DETERMINED (results incomplete).**")
        elif s6v:
            lines.append("**S6 verdict for cluster 0: PASS -- cluster 0 survives the encoder axis. "
                          "It is recoverable from embeddings across >=2 independent encoders at a "
                          "margin over the token/organism lookup, while itself surviving seed, "
                          "bootstrap, hospital-transport and era-transport perturbation and being "
                          "invisible to categorical lookup. Under this pre-specified battery it "
                          "qualifies as a genuinely embedding-carried, non-token structure.**")
        else:
            lines.append("**S6 verdict for cluster 0: FAIL -- cluster 0 does NOT survive the encoder "
                          "axis. Its robustness to seed/bootstrap/transport perturbation and its "
                          "invisibility to token lookup were specific to the OpenAI "
                          "text-embedding-3-large representation, not a property that generalizes "
                          "across independent encoders.**")
        for enc in ENCODERS:
            r = recluster.get(enc)
            j = r["best_match_jaccard_vs_published"].get("0") if r else None
            js = f"{j:.4f}" if j is not None else "NOT MEASURED"
            lines.append(f"- {enc} own-clustering best-match Jaccard for cluster 0: {js}")
    else:
        lines.append("Cluster 0 not present in survival_matrix.csv -- unexpected, check inputs.")
    lines.append("")

    lines.append("## All 16 clusters -- summary")
    lines.append("")
    lines.append("```\n" + df[["cluster", "size", "S6_pass"]].to_string(index=False) + "\n```")
    lines.append("")
    lines.append(f"Count passing S6 (of 16): {n_s6_pass} (measured: {n_measured_clusters}/16).")

    text = "\n".join(lines) + "\n"
    (c18.OUT / "FINDINGS.md").write_text(text)
    print(text, flush=True)
    print(f"[18f] wrote {c18.OUT / 'FINDINGS.md'}", flush=True)


if __name__ == "__main__":
    main()
