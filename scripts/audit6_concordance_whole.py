#!/usr/bin/env python3
"""Audit 6 (R6) -- is the encoder-concordance axis sound?  Whole-corpus part.

Light (seconds).  Writes out/39_audit6/concordance_whole.json.

  1. Korea, full corpus (n=32,329, mcs=161): pairwise ARI/AMI among the four
     encoder partitions on disk (BGE-M3, Qwen3-0.6B, OpenAI 3-small, the
     published 3-large labels).  Non-degenerate pairs are what matter.
  2. MIMIC, seed 42 and seeds 1-10: BGE-M3 vs Qwen3 ARI/AMI (already in fig1)
     PLUS the same agreement restricted within each organism category, to
     ask whether the MIMIC concordance is agreement about structure or two
     encoders both sorting on the organism line.
  3. Token alignment: AMI(partition, token) for organism / hospital-or-unit /
     year-or-period / department / episode count, each encoder, both cohorts.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score as ami, adjusted_rand_score as ari

ROOT = Path("/data/tcmhwd/AI/AST_ai")
REV = ROOT / "phenotype_revision"
OUT = REV / "out/39_audit6"
OUT.mkdir(parents=True, exist_ok=True)


def agree(a, b):
    both = (a != -1) & (b != -1)
    return {"ari": float(ari(a[both], b[both])), "ami": float(ami(a[both], b[both])), "n_both": int(both.sum())}


def kinfo(lab):
    nn = lab[lab != -1]
    vc = pd.Series(nn).value_counts()
    return {"k": int(len(vc)), "noise": float((lab == -1).mean()), "largest_share": float(vc.iloc[0] / len(lab))}


def main():
    res = {}
    # ---- Korea whole corpus
    ids = json.load(open(REV / "out/18_encoder_dec/embeddings/bge-m3/merged_ids.json"))
    parts = {}
    for enc in ("bge-m3", "qwen3-0.6b", "openai-3small"):
        z = np.load(REV / f"out/18_encoder_dec/recluster/{enc}/labels.npz")
        assert z["admission_ID"].tolist() == ids
        parts[enc] = z["labels"].astype(int)
    pub = pd.read_csv(ROOT / "pipeline_new/out_pre/phenotype_clusters_labels.csv").set_index("admission_ID")
    parts["openai-3large-published"] = pub.loc[ids, "phenotype_cluster"].to_numpy().astype(int)
    ds = pd.read_csv(ROOT / "manuscript/phenotype_ml_dataset.csv", low_memory=False,
                     usecols=["admission_ID", "BSI_episode_short_clean", "hospital_code", "test_date",
                              "treatment_department_eng", "n_bsi_episodes", "icu", "sex",
                              "charlson_score"]).drop_duplicates("admission_ID").set_index("admission_ID").loc[ids]
    ds["admit_year"] = pd.to_datetime(ds["test_date"], errors="coerce").dt.year.astype(str)
    ktok = {"organism": "BSI_episode_short_clean", "hospital": "hospital_code", "year": "admit_year",
            "department": "treatment_department_eng", "n_episodes": "n_bsi_episodes", "icu": "icu",
            "sex": "sex", "charlson_score": "charlson_score"}
    korea = {"partitions": {e: kinfo(l) for e, l in parts.items()}, "pairwise": {}, "token_alignment_ami": {},
             "within_organism_cross_encoder": {}}
    names = list(parts)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            korea["pairwise"][f"{names[i]} vs {names[j]}"] = agree(parts[names[i]], parts[names[j]])
    for e, l in parts.items():
        keep = l != -1
        korea["token_alignment_ami"][e] = {
            t: float(ami(ds[c].astype(str).to_numpy()[keep], l[keep])) for t, c in ktok.items()}
    org = ds["BSI_episode_short_clean"].to_numpy()
    for pair in (("qwen3-0.6b", "openai-3small"), ("qwen3-0.6b", "openai-3large-published"),
                 ("openai-3small", "openai-3large-published")):
        d = {}
        for o in pd.Series(org).value_counts().index[:8]:
            m = org == o
            d[o] = {"n": int(m.sum()), **agree(parts[pair[0]][m], parts[pair[1]][m])}
        korea["within_organism_cross_encoder"][" vs ".join(pair)] = d
    res["korea_full_corpus"] = korea

    # ---- MIMIC
    man = pd.read_parquet(REV / "out/27_mimic/cohort/narrative_manifest.parquet").set_index("admission_ID")
    mtok = {"organism": "organism_category", "hospital": "care_unit", "year": "admit_period",
            "department": "care_unit", "n_episodes": "n_bsi_episodes", "icu": "icu"}
    mimic = {"seed42": {}, "seeds_1_10_within_organism": {}, "token_alignment_ami": {}}
    def ml(enc, seed):
        t = pd.read_parquet(REV / f"out/27_mimic/replicate/labels/{enc}_seed{seed}.parquet").set_index("admission_ID")
        return t.loc[man.index, "cluster"].to_numpy().astype(int)
    a, b = ml("bge-m3", 42), ml("qwen3-0.6b", 42)
    mimic["seed42"]["partitions"] = {"bge-m3": kinfo(a), "qwen3-0.6b": kinfo(b)}
    mimic["seed42"]["cross_encoder"] = agree(a, b)
    org = man["organism_category"].to_numpy()
    mimic["seed42"]["within_organism_cross_encoder"] = {
        o: {"n": int((org == o).sum()), **agree(a[org == o], b[org == o])}
        for o in pd.Series(org).value_counts().index[:8]}
    for enc, l in (("bge-m3", a), ("qwen3-0.6b", b)):
        keep = l != -1
        mimic["token_alignment_ami"][enc] = {
            t: float(ami(man[c].astype(str).to_numpy()[keep], l[keep])) for t, c in mtok.items()}
    # within-organism pooled across seeds 1-10 (median over seeds of per-organism AMI)
    per = {}
    for s in range(1, 11):
        a, b = ml("bge-m3", s), ml("qwen3-0.6b", s)
        for o in pd.Series(org).value_counts().index[:5]:
            per.setdefault(o, []).append(agree(a[org == o], b[org == o])["ami"])
    mimic["seeds_1_10_within_organism"] = {o: {"median_ami": float(np.median(v)), "min": float(min(v)),
                                               "max": float(max(v))} for o, v in per.items()}
    # cross-encoder with organism token removed: AMI conditional on organism (weighted mean of within-organism AMI)
    res["mimic"] = mimic
    (OUT / "concordance_whole.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
