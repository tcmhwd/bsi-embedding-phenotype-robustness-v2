#!/usr/bin/env python3
"""Recompute the MIMIC-IV row of Supplementary Table 15 (median lines and median tokens per document).

Why: PREREGISTRATION_C.md:273 states "MIMIC documents are 47 lines / 245-276 median tokens" and Supplementary Table 15 cites that figure, because no
document-statistics file was ever written for the MIMIC corpus.  This script produces that file (AGGREGATE ONLY -- no document id, no per-document value,
no text is written or printed; MIMIC-IV is PhysioNet-DUA data and every document is read only inside this job).

Corpora (both are read and compared):
  * C_unmodified  out/followup_analyses/C_mimic_organism_ablation/narratives/unmodified/<admission_ID>.report.md   (the Analysis C `unmodified` arm)
  * stored_27     out/27_mimic/narratives/<admission_ID>.report.md                                              (the corpus scripts/27c wrote and 27d embedded)

Statistic definitions are COPIED from the Analysis A `doc_stats.csv` writer (scripts/61_harmonized_render.py:783-805) so the MIMIC row is on the same footing as the Korean rows:
  * n_lines  = len(c21.content_lines(text)) = number of NON-BLANK lines, `[ln for ln in text.split("\\n") if ln.strip() != ""]`   (61:75-76, 795)
               Extra columns `*_physical` = len(text.splitlines()) = every physical line INCLUDING blank separator lines: this is the counter behind the "47 lines" of
               PREREGISTRATION_C.md:273 (67_C_mimic_variants.py:422 `n_lines_before = len(text.splitlines(keepends=True))`; render_qa.json lines_per_doc_before), NOT the A definition.
  * tokens   = len(tok(batch, add_special_tokens=True, truncation=False)["input_ids"]), batch 512   (61:483-488, 804; identical to embed_common.hf_token_counter)
               BGE-M3 tokenizer  = model_cache/bge-m3 (local),  Qwen3 tokenizer = the pinned local Qwen3-Embedding-0.6B snapshot (61:66-70)
               AutoTokenizer.from_pretrained(path, local_files_only=True), HF_HUB_OFFLINE=1, no special-token change.
  * Q1/Q3    = numpy.percentile linear interpolation; validated in-job by reproducing the corpus_of_record row of A_harmonized_template/doc_stats_arm_summary.csv
               from A_harmonized_template/doc_stats.csv with the same summary function (the run aborts if that does not match).

Guards (any failure aborts before the CSV is written): 16,144 documents in each corpus; identical file-name sets; C unmodified arm sha256/size == narratives/hashes.csv;
batched-vs-pipeline-counter token equality on a 40-document probe; A-summary reproduction as above.

Env: /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3   (login node: py_compile only; the run is sbatch, slurm/76_mimic_doc_stats.sh)
"""
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
_n = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))          # the tokenizer stage (Rust batch encode) is the only part that scales with cores
os.environ["RAYON_NUM_THREADS"] = str(_n)
os.environ["TOKENIZERS_PARALLELISM"] = "true"
for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"

import hashlib
import importlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REV / "scripts"))
SF = REV / "out/followup_analyses"
C_DIR = SF / "C_mimic_organism_ablation"
C_UNMOD = C_DIR / "narratives" / "unmodified"
C_HASHES = C_DIR / "narratives" / "hashes.csv"
S27_DIR = REV / "out/27_mimic/narratives"
A_DIR = SF / "A_harmonized_template"
OUT_CSV = C_DIR / "doc_stats_mimic_summary.csv"
OUT_JSON = C_DIR / "doc_stats_mimic_summary.json"
N_EXPECTED = 16144
CITED = {"median_lines": 47.0, "median_tokens_bge_m3": 245.0, "median_tokens_qwen3_0_6b": 276.0}   # PREREGISTRATION_C.md:273

TOKENIZERS = {   # copied verbatim from 61_harmonized_render.py:66-70
    "bge-m3": {"path": str(REV / "model_cache" / "bge-m3"), "max_seq_length": 8192},
    "qwen3-0.6b": {"path": "/home/tcmhwd/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/"
                           "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "max_seq_length": 32768},
}
COLS = {"bge-m3": "bge_m3", "qwen3-0.6b": "qwen3_0_6b"}


def content_lines(text):          # == 21_common.content_lines == 61_harmonized_render.nonblank
    return [ln for ln in text.split("\n") if ln.strip() != ""]


def token_lengths(texts, tok, batch=512):        # == 61_harmonized_render.token_lengths
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], add_special_tokens=True, truncation=False)["input_ids"]
        out.extend(len(e) for e in enc)
    return out


def summarize(name, n_lines, tok_by_name, n_lines_phys=None):
    """One summary row.  Inputs are per-document vectors; the row holds only aggregates."""
    q = lambda a, p: float(np.percentile(np.asarray(a, dtype=float), p))
    row = {"corpus": name, "n_documents": int(len(n_lines)),
           "median_lines": q(n_lines, 50), "lines_q1": q(n_lines, 25), "lines_q3": q(n_lines, 75)}
    for tname, v in tok_by_name.items():
        c = COLS[tname]
        row[f"median_tokens_{c}"] = q(v, 50)
        row[f"tokens_{c}_q1"] = q(v, 25)
        row[f"tokens_{c}_q3"] = q(v, 75)
    row["total_lines"] = int(np.sum(n_lines))
    if n_lines_phys is not None:
        row["median_lines_physical"] = q(n_lines_phys, 50)
        row["lines_physical_q1"] = q(n_lines_phys, 25)
        row["lines_physical_q3"] = q(n_lines_phys, 75)
    return row


def read_dir(d):
    """{admission_ID: bytes}, sorted by admission_ID.  Bytes are kept in memory only."""
    files = sorted(d.glob("*.report.md"))
    return {p.name[: -len(".report.md")]: p.read_bytes() for p in files}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    t0 = datetime.now()
    print(f"[76] host={platform.node()} job={os.environ.get('SLURM_JOB_ID')} cpus={os.environ.get('SLURM_CPUS_PER_TASK')} rayon={os.environ['RAYON_NUM_THREADS']}", flush=True)
    checks = {}

    # ---- read both corpora (in memory only) -----------------------------------------------------------------
    c_raw = read_dir(C_UNMOD)
    s_raw = read_dir(S27_DIR)
    print(f"[76] documents: C_unmodified={len(c_raw)} stored_27={len(s_raw)}", flush=True)
    assert len(c_raw) == N_EXPECTED and len(s_raw) == N_EXPECTED, "document count != 16,144"
    checks["n_docs_each_corpus_16144"] = True
    assert list(c_raw) == list(s_raw), "file-name sets differ between C unmodified arm and out/27_mimic/narratives"
    checks["identical_admission_id_sets_and_order"] = True

    # ---- integrity: C unmodified arm vs its own hashes.csv; C unmodified vs stored 27 (byte level) ----------
    hh = pd.read_csv(C_HASHES)
    hh = hh[hh["arm"] == "unmodified"].set_index("admission_ID")
    assert len(hh) == N_EXPECTED and set(hh.index) == set(c_raw)
    n_hash_ok = sum(1 for k, b in c_raw.items() if sha(b) == hh.at[k, "sha256"] and len(b) == int(hh.at[k, "n_bytes"]))
    n_byte_identical = sum(1 for k in c_raw if c_raw[k] == s_raw[k])
    print(f"[76] C_unmodified matches narratives/hashes.csv (sha256+bytes): {n_hash_ok}/{N_EXPECTED}", flush=True)
    print(f"[76] C_unmodified byte-identical to out/27_mimic/narratives:    {n_byte_identical}/{N_EXPECTED}", flush=True)
    assert n_hash_ok == N_EXPECTED, "C unmodified arm does not match its hashes.csv"
    checks["c_unmodified_matches_hashes_csv"] = int(n_hash_ok)
    checks["c_unmodified_byte_identical_to_stored_27"] = int(n_byte_identical)
    del hh

    corp = {"C_unmodified": [b.decode("utf-8") for b in c_raw.values()],
            "stored_27": [b.decode("utf-8") for b in s_raw.values()]}
    del c_raw, s_raw

    # ---- line counts ---------------------------------------------------------------------------------------
    lines = {n: [len(content_lines(t)) for t in ts] for n, ts in corp.items()}
    lines_phys = {n: [len(t.splitlines()) for t in ts] for n, ts in corp.items()}

    # ---- tokens (copied recipe) -----------------------------------------------------------------------------
    from transformers import AutoTokenizer
    ec = importlib.import_module("embed_common")
    toks = {n: {} for n in corp}
    tok_meta = {}
    for tname, spec in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True)
        cnt = ec.hf_token_counter(tok)
        probe = corp["C_unmodified"][:40]
        assert [cnt(t) for t in probe] == token_lengths(probe, tok), f"{tname}: batched != pipeline counter"
        checks[f"tokenizer_{tname}_batched_equals_pipeline_counter_probe40"] = True
        for n in corp:
            toks[n][tname] = token_lengths(corp[n], tok)
            print(f"[76] tokenized {n} with {tname}", flush=True)
        tok_meta[tname] = {"path": spec["path"], "max_seq_length": spec["max_seq_length"], "add_special_tokens": True, "truncation": False,
                           "n_over_max_seq_length_C_unmodified": int(sum(1 for v in toks["C_unmodified"][tname] if v > spec["max_seq_length"]))}
        del tok

    # ---- validate the summary function / percentile method against Analysis A's own summary ---------------------
    ds = pd.read_csv(A_DIR / "doc_stats.csv", usecols=["corpus_of_record__n_lines", "corpus_of_record__tokens_bge-m3", "corpus_of_record__tokens_qwen3-0.6b"])
    a_row = summarize("corpus_of_record", ds["corpus_of_record__n_lines"].tolist(),
                      {"bge-m3": ds["corpus_of_record__tokens_bge-m3"].tolist(), "qwen3-0.6b": ds["corpus_of_record__tokens_qwen3-0.6b"].tolist()})
    a_ref = pd.read_csv(A_DIR / "doc_stats_arm_summary.csv").set_index("arm").loc["corpus_of_record"]
    a_map = {"n_documents": "n_documents", "median_lines": "median_lines", "lines_q1": "lines_q1", "lines_q3": "lines_q3",
             "median_tokens_bge_m3": "median_tokens_bge_m3", "tokens_bge_m3_q1": "tokens_bge_m3_q1", "tokens_bge_m3_q3": "tokens_bge_m3_q3",
             "median_tokens_qwen3_0_6b": "median_tokens_qwen3_0_6b", "tokens_qwen3_0_6b_q1": "tokens_qwen3_0_6b_q1", "tokens_qwen3_0_6b_q3": "tokens_qwen3_0_6b_q3",
             "total_lines": "total_lines"}
    bad = {k: (a_row[k], float(a_ref[v])) for k, v in a_map.items() if abs(float(a_row[k]) - float(a_ref[v])) > 1e-9}
    assert not bad, f"summary function does not reproduce A_harmonized_template/doc_stats_arm_summary.csv corpus_of_record row: {bad}"
    checks["summary_function_reproduces_A_corpus_of_record_row"] = True
    print("[76] summary function reproduces the A corpus_of_record row of doc_stats_arm_summary.csv (all 11 columns)", flush=True)
    del ds

    # ---- summary --------------------------------------------------------------------------------------------
    rows = [summarize(n, lines[n], toks[n], lines_phys[n]) for n in corp]
    df = pd.DataFrame(rows)
    tmp = OUT_CSV.with_name(f".tmp_{os.getpid()}_{OUT_CSV.name}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, OUT_CSV)

    qa = json.loads((C_DIR / "render_qa.json").read_text())["arms"]["unmodified"]["lines_per_doc_before"]
    checks["render_qa_json_lines_per_doc_before_median_equals_physical_median"] = bool(qa["median"] == rows[0]["median_lines_physical"])
    checks["render_qa_json_lines_per_doc_before_min_max"] = [qa["min"], qa["max"]]
    cmp_ = {}
    for r in rows:
        cmp_[r["corpus"]] = {k: {"recomputed": r[k], "cited": v, "equal": bool(r[k] == v)} for k, v in CITED.items()}
        cmp_[r["corpus"]]["median_lines_physical_vs_cited_47"] = {"recomputed": r["median_lines_physical"], "cited": 47.0, "equal": bool(r["median_lines_physical"] == 47.0)}
    agree = all(abs(float(rows[0][k]) - float(rows[1][k])) < 1e-9 for k in rows[0] if k != "corpus")
    checks["C_unmodified_and_stored_27_summaries_identical"] = bool(agree)

    def _git(*a):
        try:
            return subprocess.run(["git", "-C", str(REV), *a], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            return None
    prov = {"task": "76 MIMIC-IV doc_stats (ST15 MIMIC row)", "aggregate_only": True, "n_expected": N_EXPECTED,
            "corpus_dirs": {"C_unmodified": str(C_UNMOD), "stored_27": str(S27_DIR)},
            "definitions": {"n_lines": "non-blank lines (21_common.content_lines; identical to A doc_stats n_lines)", "tokens": "add_special_tokens=True, truncation=False, batch 512",
                            "quantile": "numpy.percentile linear", "n_lines_physical": "len(text.splitlines()), blank lines included (C render_qa lines_per_doc_before)"},
            "tokenizers": tok_meta, "cited_values_PREREGISTRATION_C_md_273": CITED, "comparison_to_cited": cmp_, "checks": checks,
            "versions": {p: __import__(p).__version__ for p in ("numpy", "pandas", "transformers", "tokenizers")},
            "python": sys.version.split()[0], "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "git_commit": _git("rev-parse", "HEAD"), "timestamp": datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": round((datetime.now() - t0).total_seconds(), 1)}
    tmpj = OUT_JSON.with_name(f".tmp_{os.getpid()}_{OUT_JSON.name}")
    tmpj.write_text(json.dumps(prov, indent=2, default=str))
    os.replace(tmpj, OUT_JSON)

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("[76] SUMMARY (aggregate):\n" + df.T.to_string(), flush=True)
    for name, c in cmp_.items():
        print(f"[76] COMPARISON {name}: " + "; ".join(f"{k}: recomputed {v['recomputed']:g} vs cited {v['cited']:g} -> {'EQUAL' if v['equal'] else 'DIFFER'}" for k, v in c.items()), flush=True)
    print(f"[76] C_unmodified vs stored_27 summaries identical: {agree}", flush=True)
    print(f"[76] wrote {OUT_CSV} and {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
