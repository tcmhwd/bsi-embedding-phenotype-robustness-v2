#!/usr/bin/env python3
"""Task 18, Step 0 -- cheap validation that `reports_pilot_pre/` is in fact
the corpus behind `out_pre/case_report_embeddings.npy` before anything else
in this task is trusted.

Samples 500 admission_IDs (seed 42) from `out_pre`'s own metadata, reads
the CURRENT narrative text for each from `reports_pilot_pre/`, re-embeds
with OpenAI `text-embedding-3-large` (the same model that produced the
stored vectors), and compares row-wise cosine similarity against the
stored embedding for that exact admission_ID (matched via
`case_report_embeddings_metadata.csv`, never by row position).

Login-node safe: 500 docs, ~$0.05, a handful of API calls.

Expected: cosine ~0.999+. If materially lower, this prints a loud STOP and
writes proceed=false -- every downstream Task 18 script must refuse to run
if this file says proceed=false (checked explicitly, not just eyeballed).
"""
import importlib
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c18 = importlib.import_module("18_common")

N_SAMPLE = 500
SEED = 42
MODEL_NAME = "text-embedding-3-large"
PRICE_PER_MILLION_TOKENS_USD = 0.13
MAX_DOCS_PER_REQUEST = 100
COSINE_MIN_EXPECTED = 0.999


def _api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key
    key_path = Path.home() / ".secrets" / "openai_api_key"
    if not key_path.is_file():
        raise SystemExit(f"no OPENAI_API_KEY and {key_path} not found")
    return key_path.read_text().strip()


def main() -> None:
    import numpy as np
    import pandas as pd
    from openai import OpenAI

    c18.OUT.mkdir(parents=True, exist_ok=True)
    out_path = c18.OUT / "step0_validate_corpus.json"

    meta = c18.load_metadata()
    rng = np.random.default_rng(SEED)
    sample_idx = rng.choice(len(meta), size=N_SAMPLE, replace=False)
    sample_idx.sort()
    sample = meta.iloc[sample_idx].reset_index(drop=True)

    texts = []
    for nid, aid in zip(sample["newpatient_ID"], sample["admission_ID"]):
        p = c18.report_path_for(nid, int(aid))
        if not p.exists():
            raise FileNotFoundError(f"{p} missing")
        texts.append(p.read_text(encoding="utf-8", errors="replace"))

    client = OpenAI(api_key=_api_key())
    new_vecs = []
    total_tokens = 0
    t0 = time.time()
    for start in range(0, len(texts), MAX_DOCS_PER_REQUEST):
        batch = texts[start:start + MAX_DOCS_PER_REQUEST]
        resp = client.embeddings.create(model=MODEL_NAME, input=batch)
        new_vecs.extend(d.embedding for d in resp.data)
        total_tokens += resp.usage.total_tokens
        print(f"[step0] embedded {start + len(batch)}/{len(texts)} docs, "
              f"{total_tokens} tokens so far", flush=True)
    enc_sec = time.time() - t0
    new_vecs = np.asarray(new_vecs, dtype=np.float64)

    stored_full = np.load(c18.OUT_PRE_EMB, mmap_mode="r")
    stored_rows = stored_full[sample["embedding_index"].to_numpy()].astype(np.float64)

    def _rownorm(x):
        n = np.linalg.norm(x, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return x / n

    a = _rownorm(new_vecs)
    b = _rownorm(stored_rows)
    cosine = np.sum(a * b, axis=1)

    cost_usd = round(total_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD, 4)
    result = {
        "n_sampled": int(len(sample)),
        "seed": SEED,
        "model": MODEL_NAME,
        "total_tokens": int(total_tokens),
        "cost_usd": cost_usd,
        "enc_sec": round(enc_sec, 2),
        "cosine_mean": float(cosine.mean()),
        "cosine_median": float(np.median(cosine)),
        "cosine_min": float(cosine.min()),
        "cosine_p05": float(np.percentile(cosine, 5)),
        "cosine_below_expected_threshold": int((cosine < COSINE_MIN_EXPECTED).sum()),
        "expected_threshold": COSINE_MIN_EXPECTED,
        "corpus_dir": str(c18.REPORTS_DIR),
        "stored_embeddings_path": str(c18.OUT_PRE_EMB),
        "metadata_path": str(c18.OUT_PRE_META),
    }
    result["proceed"] = bool(result["cosine_mean"] >= COSINE_MIN_EXPECTED)
    c18.atomic_write_json(result, out_path)
    print(json.dumps(result, indent=2), flush=True)
    if not result["proceed"]:
        print(
            "STOP: mean cosine below the expected 0.999+ threshold -- the corpus "
            "assumption (reports_pilot_pre == the corpus behind out_pre embeddings) "
            "is NOT validated. Do not proceed to Step 1 until this is resolved.",
            flush=True,
        )
    else:
        print(f"[step0] PASS: mean cosine {result['cosine_mean']:.6f} >= {COSINE_MIN_EXPECTED} "
              f"-- corpus assumption validated, cost ${cost_usd}", flush=True)


if __name__ == "__main__":
    main()
