#!/usr/bin/env python3
"""Audit 7 (question B) -- embed a narrative variant with Qwen3-Embedding-0.6B.

Same model, pooling, normalisation, max_seq_length and load order as
18b_embed_open.py (MODEL_CONFIGS imported read-only from 04_open_embedding);
only the text source differs: a directory of <admission_ID>.txt files, read in
the out_pre metadata row order so the row order matches
out/18_encoder_dec/embeddings/qwen3-0.6b/merged.npy exactly.

    --variant no_summary_tokens --texts-dir out/21_published_variants/narratives/no_summary_tokens
    --variant no_cluster_ids    --texts-dir out/41_audit7/narratives/no_cluster_ids

Writes out/41_audit7/embeddings/<variant>/{merged.npy, merged_ids.json, .embed_manifest.json}.
"""
import os

_THREADS = os.environ.get("STRESS_THREADS", "8")
for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = _THREADS

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c18 = importlib.import_module("18_common")
oe = importlib.import_module("04_open_embedding")

OUT = SCRIPT_DIR.parent / "out/41_audit7/embeddings"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--texts-dir", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    out_dir = OUT / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = out_dir / "merged.npy"
    if merged_path.exists():
        print(f"[audit7-embed] {args.variant}: already present, skipping", flush=True)
        return

    cfg = oe.MODEL_CONFIGS["qwen3-0.6b"]
    meta = c18.load_metadata()
    texts_dir = Path(args.texts_dir)
    texts, ids = [], []
    for aid in meta["admission_ID"]:
        p = texts_dir / f"{int(aid)}.txt"
        texts.append(p.read_text(encoding="utf-8", errors="replace"))
        ids.append(int(aid))
    print(f"[audit7-embed] {args.variant}: {len(texts)} docs from {texts_dir}", flush=True)

    model = SentenceTransformer(cfg["load_path"], device="cpu")
    model.max_seq_length = cfg["max_seq_length"]
    if args.device == "cuda":
        model = model.to("cuda")
    print(f"[audit7-embed] model on {args.device}, cuda_available={torch.cuda.is_available()}", flush=True)

    t0 = time.time()
    vecs = model.encode(texts, batch_size=args.batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    enc_sec = time.time() - t0
    np.save(merged_path, vecs.astype(np.float32))
    (out_dir / "merged_ids.json").write_text(json.dumps(ids))
    manifest = {"variant": args.variant, "texts_dir": str(texts_dir), "model": cfg["load_path"],
                "pooling": cfg["pooling"], "max_seq_length": cfg["max_seq_length"],
                "normalize": True, "device": args.device, "batch_size": args.batch_size,
                "n_docs": len(texts), "dim": int(vecs.shape[1]), "enc_sec": round(enc_sec, 1),
                "torch_version": torch.__version__}
    (out_dir / ".embed_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
