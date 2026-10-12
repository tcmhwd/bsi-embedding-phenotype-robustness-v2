#!/usr/bin/env python3
"""Audit 4, check B -- Gower retrieval WITHOUT the four whole-admission
organ-support flags (icu, ventilator, central_orarterial_line,
urinary_orrenal_catheter) that scripts/05_baseline_comparators.py documents
as post-index exceptions. Everything else identical to Task 22/23
(build_structured_matrix -> compute_gower_matrix -> Task-23 retrieval,
k=25, same folds). Writes only under out/25_audit4/. bsi_pheno interpreter.
"""
import os
os.environ["OMP_NUM_THREADS"] = os.environ["NUMBA_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = "1"
import importlib, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
sys.path.insert(0, str(REV / "scripts"))
frame_mod = importlib.import_module("01_build_analysis_frame")
comparators = importlib.import_module("05_baseline_comparators")
c23 = importlib.import_module("23a_retrieval_common")
OUT = REV / "out/25_audit4"
DROP = ["icu", "ventilator", "central_orarterial_line", "urinary_orrenal_catheter"]
K = 25


def main():
    mpath = OUT / "gower_40_no_organ_support.npy"
    frame = frame_mod.load_frame(verify_reports=False)
    if not mpath.exists():
        matrix, cats = comparators.build_structured_matrix(frame)
        matrix = matrix.drop(columns=DROP); cats = [c for c in cats if c not in DROP]
        print("columns used:", list(matrix.columns), flush=True)
        t0 = time.time()
        D = comparators.compute_gower_matrix(matrix, cats)
        print("gower computed", D.shape, time.time() - t0, flush=True)
        tmp = mpath.with_name(mpath.stem + ".tmp.npy"); np.save(tmp, D); os.replace(tmp, mpath)
        del D
    D = np.load(mpath, mmap_mode="r")
    master = c23.build_master_frame()
    assert np.array_equal(master["admission_ID"].to_numpy(), frame["admission_ID"].to_numpy())
    sanity = c23.sanity_check_published_16_reproduces_0_6206(master)
    y = master[c23.MORTALITY_COL].to_numpy(); pid = master["patient_ID"].to_numpy()
    dates = master["test_date_ordinal"].to_numpy()
    folds = c23.build_folds(y, pid)
    res = {"sanity": sanity, "dropped_columns": DROP}
    for var in ["unrestricted", "past_only"]:
        aucs = []
        for tr, te in folds:
            valid = np.ones((len(te), len(tr)), dtype=bool) if var == "unrestricted" else (dates[tr][None, :] < dates[te][:, None])
            score = -np.asarray(D[np.ix_(te, tr)], dtype=np.float32)
            top, cnt = c23.topk_ordered(score, valid, K)
            risk, _ = c23.neighbour_risk_scores(top, cnt, y[tr], K)
            ok = ~np.isnan(risk)
            aucs.append(roc_auc_score(y[te][ok], risk[ok]))
        res[var] = dict(k=K, auroc_mean=float(np.mean(aucs)), auroc_std=float(np.std(aucs)))
        print(var, res[var], flush=True)
    (OUT / "d_gower_index_only.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
