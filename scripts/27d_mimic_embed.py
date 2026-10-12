#!/usr/bin/env python3
"""Task M, Step 3 -- embed the MIMIC narratives with BGE-M3 and
Qwen3-Embedding-0.6B (open, local models only -- no MIMIC text ever reaches
an external API, per the binding DUA constraint).

Mirrors `04_open_embedding.py`'s `stage_embed` exactly (model loading,
truncation guard, manifest fields, atomic writes) but does NOT import it,
because that module's `stage_embed` is wired to the Korean corpus loader
(`01_build_analysis_frame.load_frame`) and gates every embed behind the
Task-4 Korean concordance verdict (`embed_common.require_passing_gate`).
Neither applies here: the MIMIC arm does not attempt to transport the
Korean phenotypes (superseded design), so it needs no Korean-derived gate
-- it embeds MIMIC narratives directly and evaluates cross-encoder /
cross-seed / token-recoverability replication among THOSE embeddings.

Known gotcha (measured in this project, jobs 586270/586271/586273 vs
586275): `SentenceTransformer(model, device="cuda")` segfaults silently
with zero log output. Load on CPU, then `.to("cuda")` if needed.

Run (SBATCH; CPU is the default path -- GPU is contended by other jobs on
this cluster and this corpus is small, ~16k docs, so CPU is not a
bottleneck):
    /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/27d_mimic_embed.py --model bge-m3
    /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 scripts/27d_mimic_embed.py --model qwen3-0.6b
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REV = SCRIPT_DIR.parent
COHORT_DIR = REV / "out/27_mimic/cohort"
OUT = REV / "out/27_mimic/embed"

MODEL_CONFIGS = {
    "bge-m3": {
        "load_path": str(REV / "model_cache" / "bge-m3"),
        "max_seq_length": 8192,
        "tokenizer_name": "BAAI/bge-m3 (local safetensors conversion)",
    },
    "qwen3-0.6b": {
        "load_path": "Qwen/Qwen3-Embedding-0.6B",
        "max_seq_length": 32768,
        "tokenizer_name": "Qwen/Qwen3-Embedding-0.6B",
    },
}
DEFAULT_BATCH_SIZE = {"cpu": 16, "cuda": 128}


def _atomic_write_json(obj, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--threads", type=int, default=None, help="OMP/MKL thread count for encoding (thread-count-independent, so many threads is fine)")
    args = ap.parse_args()

    if args.threads:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[var] = str(args.threads)

    import numpy as np
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer

    sys.path.insert(0, str(SCRIPT_DIR))
    import embed_common

    cfg = MODEL_CONFIGS[args.model]
    manifest_df = pd.read_parquet(COHORT_DIR / "narrative_manifest.parquet").sort_values("admission_ID").reset_index(drop=True)
    texts = [Path(p).read_text(encoding="utf-8", errors="replace") for p in manifest_df["report_file"]]
    admission_ids = manifest_df["admission_ID"].tolist()
    print(f"[27d/{args.model}] n_docs={len(texts)}", flush=True)

    # --- G0: truncation guard, hard precondition before any encoding. ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["load_path"])
    count_fn = embed_common.hf_token_counter(tokenizer)
    report = embed_common.token_report(texts, count_fn, cfg["max_seq_length"], cfg["tokenizer_name"])
    embed_common.assert_no_truncation(report)
    print(f"[27d/{args.model}] token report: n_tokens_median={report['n_tokens_median']} "
          f"n_tokens_max={report['n_tokens_max']} frac_truncated={report['frac_truncated']}", flush=True)

    batch_size = args.batch_size or DEFAULT_BATCH_SIZE[args.device]

    # Measured constraint (this repo, jobs 586270/586271/586273 vs 586275):
    # direct device="cuda" segfaults silently. Load on CPU, then .to("cuda").
    model = SentenceTransformer(cfg["load_path"], device="cpu")
    model.max_seq_length = cfg["max_seq_length"]
    if args.device == "cuda":
        model = model.to("cuda")

    t0 = time.time()
    vecs = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    enc_sec = time.time() - t0

    out_dir = OUT / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_path = out_dir / "merged.npy"
    id_path = out_dir / "merged_ids.json"
    tmp_vec = vec_path.with_name(f".tmp_{os.getpid()}_{vec_path.name}")
    np.save(tmp_vec, vecs.astype(np.float32))
    os.replace(tmp_vec, vec_path)
    tmp_id = id_path.with_name(f".tmp_{os.getpid()}_{id_path.name}")
    tmp_id.write_text(json.dumps(admission_ids))
    os.replace(tmp_id, id_path)

    manifest = {
        "model_slug": args.model, "model_load_path": cfg["load_path"], "device": args.device,
        "batch_size": batch_size, "n_docs": len(texts), "dim": int(vecs.shape[1]),
        "enc_sec": round(enc_sec, 2), "docs_per_sec": round(len(texts) / enc_sec, 3) if enc_sec > 0 else None,
        "python": sys.executable, "sys_prefix": sys.prefix,
        "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
    }
    import sentence_transformers as _st
    import transformers as _tf
    manifest["transformers_version"] = _tf.__version__
    manifest["sentence_transformers_version"] = _st.__version__
    manifest.update({k: report[k] for k in embed_common.MANIFEST_GUARD_KEYS})
    _atomic_write_json(manifest, out_dir / "embed_manifest.json")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"[27d/{args.model}] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
