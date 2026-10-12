#!/usr/bin/env python3
"""Task 18, Step 1 -- embed the `reports_pilot_pre/` corpus (32,329 docs,
the corpus behind the published labels, Step 0-validated) with the two
open encoders (BGE-M3, Qwen3-Embedding-0.6B).

Model registry (`load_path`/`max_seq_length`/`dim`/`pooling`/`prefix`) is
REUSED verbatim from `04_open_embedding.MODEL_CONFIGS` (that module is
imported read-only, never edited) -- same models, same local BGE-M3
safetensors conversion, same "no query prefix for documents" convention.
Text loading is NEW here (`18_common.load_texts`), because
`04_open_embedding.py`'s own text loader is wired to the WRONG corpus
(`01_build_analysis_frame`, `reports_all` via `pipeline_new/out/
embeddings_shards`) for this task.

GPU load-order gotcha (measured repeatedly in this repo, jobs 586270/
586271/586273 vs 586275): `SentenceTransformer(model, device="cuda")`
directly segfaults with a zero-byte log. Load on cpu, THEN `.to("cuda")`.

Thread pinning happens before any scientific import (this is the embed
stage: thread-count-independent, so many threads are fine/desired on CPU;
irrelevant on GPU beyond the tokenizer's own CPU-side work).

Writes only under `out/18_encoder_dec/embeddings/<model_slug>/`:
  merged.npy, merged_ids.json (admission_ID, in the SAME row order as
  merged.npy), .embed_manifest.json (truncation-guard fields + timing +
  versions). Resumable: skips straight to a no-op if merged.npy already
  exists.
"""
import os
import sys

_THREADS = os.environ.get("STRESS_THREADS", "8")
for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = _THREADS

import argparse
import importlib
import json
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c18 = importlib.import_module("18_common")
oe = importlib.import_module("04_open_embedding")  # read-only import: MODEL_CONFIGS only
import embed_common

EMBED_OUT = c18.OUT / "embeddings"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(oe.MODEL_CONFIGS))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--batch-size", type=int, required=True)
    args = ap.parse_args()

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    step0 = c18.OUT / "step0_validate_corpus.json"
    if not step0.exists() or not json.loads(step0.read_text()).get("proceed", False):
        raise SystemExit(f"{step0} missing or proceed=false -- Step 0 corpus validation must pass first")

    out_dir = EMBED_OUT / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / "merged.npy"
    ids_path = out_dir / "merged_ids.json"
    manifest_path = out_dir / ".embed_manifest.json"
    if merged_path.exists() and ids_path.exists() and manifest_path.exists():
        print(f"[18b] {args.model}: merged.npy already present, skipping (resume)", flush=True)
        print(json.dumps(json.loads(manifest_path.read_text()), indent=2), flush=True)
        return

    cfg = oe.MODEL_CONFIGS[args.model]
    if args.model == "bge-m3":
        cfg["model_revision"] = oe._resolve_bge_m3_revision()

    frame = c18.load_texts()  # full corpus, metadata row order
    texts = frame["text"].tolist()
    admission_ids = frame["admission_ID"].tolist()
    print(f"[18b] {args.model}: {len(texts)} docs loaded from {c18.REPORTS_DIR}", flush=True)

    # --- G0: truncation guard, BEFORE encoding (hard precondition). ---
    tokenizer = AutoTokenizer.from_pretrained(cfg["load_path"])
    count_fn = embed_common.hf_token_counter(tokenizer)
    report = embed_common.token_report(texts, count_fn, cfg["max_seq_length"], cfg["tokenizer_name"])
    embed_common.assert_no_truncation(report)
    print(f"[18b] {args.model}: truncation guard OK, frac_truncated={report['frac_truncated']}", flush=True)

    model = SentenceTransformer(cfg["load_path"], device="cpu")
    model.max_seq_length = cfg["max_seq_length"]
    if args.device == "cuda":
        model = model.to("cuda")
    print(f"[18b] {args.model}: model loaded on device={args.device}, "
          f"cuda_available={torch.cuda.is_available()}", flush=True)

    t0 = time.time()
    vecs = model.encode(
        texts, batch_size=args.batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    enc_sec = time.time() - t0
    print(f"[18b] {args.model}: encoded {len(texts)} docs in {enc_sec:.1f}s "
          f"({len(texts) / enc_sec:.2f} docs/sec)", flush=True)

    tmp_vec = merged_path.with_name(f".tmp_{os.getpid()}_{merged_path.name}")
    np.save(tmp_vec, vecs.astype(np.float32))
    os.replace(tmp_vec, merged_path)
    tmp_id = ids_path.with_name(f".tmp_{os.getpid()}_{ids_path.name}")
    tmp_id.write_text(json.dumps(admission_ids))
    os.replace(tmp_id, ids_path)

    import sentence_transformers as _st
    import transformers as _tf
    manifest = {
        "model_slug": args.model, "model_load_path": cfg["load_path"],
        "model_revision": cfg.get("model_revision"), "pooling": cfg["pooling"],
        "prefix": cfg["prefix"], "normalize": True, "device": args.device,
        "batch_size": args.batch_size, "n_docs": len(texts), "dim": int(vecs.shape[1]),
        "enc_sec": round(enc_sec, 2),
        "docs_per_sec": round(len(texts) / enc_sec, 3) if enc_sec > 0 else None,
        "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
        "transformers_version": _tf.__version__, "sentence_transformers_version": _st.__version__,
        "corpus_dir": str(c18.REPORTS_DIR),
    }
    manifest = embed_common.merge_report_into_manifest(manifest, report)
    c18.atomic_write_json(manifest, manifest_path)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
