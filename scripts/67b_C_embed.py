#!/usr/bin/env python3
"""Task 67b -- Analysis C (FOLLOWUP-2026-09-21-C): embed one MIMIC arm with one LOCAL encoder.

Everything about model loading, dtype, max_seq_length, pooling, normalisation and the
truncation guard is taken from scripts/27d_mimic_embed.py (MODEL_CONFIGS :38-49; load on CPU
then `.to("cuda")` :99-102; `normalize_embeddings=True`, no instruction prefix :105-108;
`assert_no_truncation` :91).  `MODEL_CONFIGS` is IMPORTED from 27d, not copied.  No text
leaves this machine: HF_HUB_OFFLINE=1 is forced, both encoders are loaded from local disk.

Every arm -- including `unmodified` -- is embedded FROM SCRATCH; the stored
out/27_mimic/embed vectors are never used as an embedding.  For the `unmodified` arm they are
read once, AFTER encoding, only to record a numeric diagnostic (cosine / max-abs difference
between this GPU run and the stored CPU run) in the embed manifest.

Batch size: 27d has no per-model constant on GPU (DEFAULT_BATCH_SIZE["cuda"] = 128, a CPU/GPU
generic default).  The batch sizes the stored MIMIC vectors were actually produced with are
recorded in out/27_mimic/embed/*/embed_manifest.json (bge-m3: 32, qwen3-0.6b: 8); those are
used here so that batch composition (sentence-transformers length-sorts and pads per batch)
matches the stored reference as closely as possible.  `--batch-size` overrides.

Output (under out/followup_analyses/C_mimic_organism_ablation/embed/<arm>/<encoder>/):
  merged.npy (16144 x 1024 float32), merged_ids.json, n_tokens.npy, embed_manifest.json

Addendum 1 (PREREGISTRATION_C_ADDENDUM_1.md): the model loading above is kept EXACTLY (Qwen3 loads as
bfloat16, BGE-M3 as float32), on GPU, with the stored run's batch sizes, for all five arms.  The
`unmodified` arm additionally runs the GROSS-ERROR DETECTOR: minimum cosine between the new and the
stored vectors over the full corpus must be >= 0.99 (v67.GROSS_ERROR_COS_MIN); the result is written to
<out-root>/gross_error_check_<encoder>.json and a failure exits non-zero.  Every non-`unmodified` embed
task first requires both encoders' checks to have passed, so a failure halts the rest of the chain.
There is NO partition-identity halt anywhere.

Env: /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3, one GPU on gpu_partition.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")             # no external service under any circumstance
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

v67 = importlib.import_module("67_C_mimic_variants")
s27d = importlib.import_module("27d_mimic_embed")
MODEL_CONFIGS = s27d.MODEL_CONFIGS                       # bge-m3 / qwen3-0.6b: load_path, max_seq_length
STORED_BATCH = {"bge-m3": 32, "qwen3-0.6b": 8}           # stored out/27_mimic/embed/*/embed_manifest.json
EXPECT_QWEN_SNAPSHOT = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"      # PREREGISTRATION_C.md 3.3
EXPECT_BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def model_provenance(model_slug: str) -> dict:
    cfg = MODEL_CONFIGS[model_slug]
    if model_slug == "bge-m3":
        prov = json.loads((Path(cfg["load_path"]) / "CONVERSION_PROVENANCE.json").read_text())
        st = Path(cfg["load_path"]) / "model.safetensors"
        sha = v67.sha256_file(st)
        assert prov["source_revision"] == EXPECT_BGE_REVISION, prov["source_revision"]
        assert sha == prov["sha256"]["model.safetensors"], "bge-m3 safetensors sha256 differs from provenance"
        return {"model_source_repo": prov["source_repo"], "model_revision": prov["source_revision"],
                "model_weights_file": str(st), "model_weights_sha256": sha}
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(cfg["load_path"], local_files_only=True))
    assert snap.name == EXPECT_QWEN_SNAPSHOT, snap.name
    return {"model_source_repo": cfg["load_path"], "model_revision": snap.name, "model_snapshot_dir": str(snap)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=v67.ARM_ORDER, required=True)
    ap.add_argument("--model", choices=list(MODEL_CONFIGS), required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--out-root", default=str(v67.C_ROOT))
    ap.add_argument("--limit", type=int, default=None, help="smoke-test only: first N documents")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.threads:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(args.threads)

    import numpy as np
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    import embed_common

    out_root = Path(args.out_root)
    out_dir = out_root / "embed" / args.arm / args.model
    if args.arm != "unmodified":
        v67.require_gate(out_root, list(v67.ENCODERS))          # Addendum 1 item 2a: halt the chain on failure
    if not args.force and (out_dir / "embed_manifest.json").exists() and (out_dir / "merged.npy").exists():
        print(f"[67b/{args.arm}/{args.model}] already embedded; --force to redo", flush=True)
        if args.arm == "unmodified":
            v67.require_gate(out_root, [args.model])
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")

    cfg = MODEL_CONFIGS[args.model]
    man = v67.load_manifest()
    if args.limit:
        man = man.iloc[: args.limit].copy()
    ids = man["admission_ID"].tolist()

    # ---- documents of this arm, verified against the render-stage sha256 manifest
    hashes = pd.read_csv(out_root / "narratives" / "hashes.csv")
    hmap = {r.admission_ID: r.sha256 for r in hashes[hashes["arm"] == args.arm].itertuples()}
    texts = []
    for aid in ids:
        b = v67.arm_doc_path(out_root, args.arm, aid).read_bytes()
        assert v67.sha256_bytes(b) == hmap[aid], f"{args.arm}/{aid}: file differs from hashes.csv"
        texts.append(b.decode("utf-8"))
    render_manifest = json.loads((out_root / "render_manifest.json").read_text())
    corpus_sha = render_manifest["arms"][args.arm]["corpus_sha256"]
    print(f"[67b/{args.arm}/{args.model}] n_docs={len(texts)} corpus_sha256={corpus_sha[:16]}", flush=True)

    # ---- G0: truncation guard, hard precondition (27d:88-91)
    tokenizer = AutoTokenizer.from_pretrained(cfg["load_path"])
    count_fn = embed_common.hf_token_counter(tokenizer)
    report = embed_common.token_report(texts, count_fn, cfg["max_seq_length"], cfg["tokenizer_name"])
    embed_common.assert_no_truncation(report)
    n_tokens = np.fromiter((count_fn(t) for t in texts), dtype=np.int64, count=len(texts))
    print(f"[67b/{args.arm}/{args.model}] tokens median={report['n_tokens_median']} max={report['n_tokens_max']} "
          f"n_truncated={report['n_truncated']}", flush=True)

    batch_size = args.batch_size or STORED_BATCH[args.model]
    prov = model_provenance(args.model)

    # ---- load exactly as 27d:99-102 (device="cuda" in the constructor segfaults silently)
    model = SentenceTransformer(cfg["load_path"], device="cpu")
    model.max_seq_length = cfg["max_seq_length"]
    if args.device == "cuda":
        model = model.to("cuda")
    # dtype is whatever 27d's `SentenceTransformer(path, device="cpu")` load yields in this stack -- it is
    # NOT overridden here (that would break comparability with the stored MIMIC vectors).  Recorded, and
    # any difference from PREREGISTRATION_C.md 3.3 ("float32 throughout") is flagged in the manifest.
    dtypes = {str(p.dtype) for p in model.parameters()}
    print(f"[67b/{args.arm}/{args.model}] model parameter dtypes after load: {sorted(dtypes)}", flush=True)

    t0 = time.time()
    vecs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    enc_sec = time.time() - t0
    vecs = vecs.astype(np.float32)
    assert vecs.shape == (len(texts), 1024) and np.isfinite(vecs).all()
    print(f"[67b/{args.arm}/{args.model}] encoded {vecs.shape} in {enc_sec:.1f}s ({len(texts) / enc_sec:.1f} docs/s)",
          flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    vec_path = out_dir / "merged.npy"
    tmp = vec_path.with_name(f".tmp_{os.getpid()}_{vec_path.name}")
    with open(tmp, "wb") as fh:
        np.save(fh, vecs)
    os.replace(tmp, vec_path)
    v67.atomic_write_json(ids, out_dir / "merged_ids.json")
    tmpn = out_dir / f".tmp_{os.getpid()}_n_tokens.npy"
    with open(tmpn, "wb") as fh:
        np.save(fh, n_tokens)
    os.replace(tmpn, out_dir / "n_tokens.npy")

    # ---- token edit size vs unmodified arm (same tokenizer)
    tok_removed = None
    if args.arm != "unmodified":
        u_hash = hashes[hashes["arm"] == "unmodified"].set_index("admission_ID")["sha256"]
        u_tokens = np.fromiter(
            (count_fn(v67.arm_doc_path(out_root, "unmodified", aid).read_text(encoding="utf-8")) for aid in ids),
            dtype=np.int64, count=len(ids))
        d = u_tokens - n_tokens
        tok_removed = {"unmodified_n_tokens_median": float(np.median(u_tokens)),
                       "arm_n_tokens_median": float(np.median(n_tokens)),
                       "tokens_removed_per_doc_min": int(d.min()), "tokens_removed_per_doc_median": float(np.median(d)),
                       "tokens_removed_per_doc_max": int(d.max())}

    # ---- numeric diagnostic vs the stored (CPU) vectors -- unmodified arm only, AFTER encoding
    diag = None
    if args.arm == "unmodified":
        sp = v67.STORED_EMBED / args.model
        stored = np.load(sp / "merged.npy")
        sids = json.loads((sp / "merged_ids.json").read_text())
        assert sids[: len(ids)] == ids
        stored = stored[: len(ids)]
        cos = (vecs * stored).sum(1)
        diag = {"n": int(len(ids)), "stored_vectors_path": str(sp / "merged.npy"),
                "cosine_min": float(cos.min()), "cosine_median": float(np.median(cos)),
                "cosine_mean": float(cos.mean()),
                "n_cosine_lt_0_999999": int((cos < 0.999999).sum()),
                "max_abs_elementwise_diff": float(np.abs(vecs - stored).max()),
                "bitwise_identical_rows": int((vecs == stored).all(1).sum()),
                "note": "diagnostic only; stored vectors were NOT used as this arm's embedding"}
        print(f"[67b/{args.arm}/{args.model}] vs stored: cos_min={diag['cosine_min']:.8f} "
              f"max_abs_diff={diag['max_abs_elementwise_diff']:.3e} identical_rows={diag['bitwise_identical_rows']}",
              flush=True)

    manifest = {
        "analysis": "C_mimic_organism_ablation", "arm": args.arm, "arm_role": v67.ARM_ROLE[args.arm],
        "model_slug": args.model, "model_load_path": cfg["load_path"], **prov,
        "device": args.device,
        "cuda_device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "cuda_runtime": torch.version.cuda, "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "model_param_dtypes": sorted(dtypes),
        "model_dtype_differs_from_prereg_text_float32": dtypes != {"torch.float32"},
        "pooling_and_normalisation": "as SentenceTransformer module config; normalize_embeddings=True; no prompt/prefix",
        "batch_size": batch_size, "batch_size_source": "stored out/27_mimic/embed/*/embed_manifest.json" if not args.batch_size else "--batch-size",
        "n_docs": len(texts), "n_unique_subjects": int(man["subject_id"].nunique()), "dim": int(vecs.shape[1]),
        "enc_sec": round(enc_sec, 2), "docs_per_sec": round(len(texts) / enc_sec, 3),
        "corpus_sha256": corpus_sha, "vectors_sha256": v67.sha256_file(vec_path),
        "narratives_dir": str(out_root / "narratives" / args.arm),
        "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
        "tokens_removed_vs_unmodified": tok_removed,
        "numeric_diagnostic_vs_stored_cpu_vectors": diag,
        **{k: report[k] for k in embed_common.MANIFEST_GUARD_KEYS},
        **v67.env_record(),
    }
    embed_common.assert_no_truncation(report)                       # again at manifest write (27d:91 / embed_common)
    assert manifest["n_truncated"] == 0
    manifest["addendum"] = "PREREGISTRATION_C_ADDENDUM_1.md (sha256 in qc/PREREG_HASHES_C_ADDENDUM1.txt)"
    v67.atomic_write_json(manifest, out_dir / "embed_manifest.json")
    print(f"[67b/{args.arm}/{args.model}] wrote {out_dir}", flush=True)

    if args.arm == "unmodified":
        gate = {
            "encoder": args.model, "check": "Addendum 1 item 2a gross-error detector",
            "n_docs_compared": diag["n"], "full_corpus": diag["n"] == v67.N_EXPECTED,
            "cosine_min": diag["cosine_min"], "cosine_median": diag["cosine_median"],
            "cosine_mean": diag["cosine_mean"], "threshold_cosine_min": v67.GROSS_ERROR_COS_MIN,
            "pass": bool(diag["cosine_min"] >= v67.GROSS_ERROR_COS_MIN),
            "max_abs_elementwise_diff": diag["max_abs_elementwise_diff"],
            "n_cosine_lt_0_999999": diag["n_cosine_lt_0_999999"],
            "bitwise_identical_rows": diag["bitwise_identical_rows"],
            "device": args.device, "model_param_dtypes": sorted(dtypes), "batch_size": batch_size,
            "vectors_sha256": manifest["vectors_sha256"], "corpus_sha256": corpus_sha,
            "stored_vectors_path": diag["stored_vectors_path"], **v67.env_record(),
        }
        v67.atomic_write_json(gate, v67.gate_path(out_root, args.model))
        print(f"[67b/{args.arm}/{args.model}] GROSS-ERROR CHECK cosine_min={gate['cosine_min']:.8f} "
              f"threshold={gate['threshold_cosine_min']} pass={gate['pass']}", flush=True)
        if not gate["pass"]:
            raise SystemExit(3)                                  # non-zero: `afterok`/`aftercorr` dependents never start


if __name__ == "__main__":
    main()
