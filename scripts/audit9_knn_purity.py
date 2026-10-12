#!/usr/bin/env python3
"""Audit 9 -- is field dominance a property of the 1024-d Qwen3 embedding or
only of the 2-D UMAP+HDBSCAN output?  k-NN (k=15, cosine) label purity of
first_antibiotic and organism in the RAW embedding for the stored corpus and
each ablation arm.  No UMAP, no HDBSCAN.  Writes out/45_audit9/knn_purity.json.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "8"))
import json
from pathlib import Path
import numpy as np
import pandas as pd

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
ROOT = REV.parent
K = 15
ARMS = {
    "stored_full": REV / "out/18_encoder_dec/embeddings/qwen3-0.6b/merged.npy",
    "no_organism": REV / "out/41_audit7/embeddings/ablation_no_organism/merged.npy",
    "no_therapy": REV / "out/41_audit7/embeddings/ablation_no_therapy/merged.npy",
    "mask_ab_names_all": REV / "out/45_audit9/embeddings/mask_ab_names_all/merged.npy",
    "mask_ab1_name_only": REV / "out/45_audit9/embeddings/mask_ab1_name_only/merged.npy",
}
ids = json.loads((REV / "out/18_encoder_dec/embeddings/qwen3-0.6b/merged_ids.json").read_text())
raw = pd.read_parquet(REV / "out/20_onehot/features/raw_fields.parquet").set_index("admission_ID").loc[ids]
ab1 = raw["tx__admission_antibiotics"].map(
    lambda v: str(json.loads(v)[0]) if isinstance(v, str) and v.startswith("[") and json.loads(v) else "NA").to_numpy()
ds = pd.read_csv(ROOT / "manuscript/phenotype_ml_dataset.csv", low_memory=False,
                 usecols=["admission_ID", "BSI_episode_short_clean"]).drop_duplicates("admission_ID").set_index("admission_ID")
org = ds.loc[ids, "BSI_episode_short_clean"].astype(str).to_numpy()
dept = raw["admin__department"].astype(str).to_numpy()
fields = {"first_antibiotic": ab1, "organism": org, "department": dept}
rng = np.random.default_rng(0)
chance = {f: float(np.mean(v[rng.permutation(len(v))] == v)) for f, v in fields.items()}

out = {"k": K, "chance_match_rate": chance, "arms": {}}
for name, path in ARMS.items():
    x = np.load(path).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    n = len(x)
    nn = np.empty((n, K), dtype=np.int64)
    for s in range(0, n, 2048):
        sims = x[s:s + 2048] @ x.T
        sims[np.arange(sims.shape[0]), np.arange(s, s + sims.shape[0])] = -np.inf
        part = np.argpartition(-sims, K, axis=1)[:, :K]
        nn[s:s + 2048] = part
    block = {}
    for f, v in fields.items():
        same = (v[nn] == v[:, None]).mean(axis=1)
        block[f"purity_{f}"] = float(same.mean())
    out["arms"][name] = block
    print(name, block, flush=True)
(REV / "out/45_audit9/knn_purity.json").write_text(json.dumps(out, indent=1))
print("chance", chance)
