#!/usr/bin/env python3
"""Analysis A, PHASE 2 -- embed ONE harmonized Korean arm with ONE local open encoder (local GPU; no
external service, no API call).  6 tasks = 3 arms x {bge-m3, qwen3-0.6b}, 32,329 documents each.

Governing documents (FROZEN, not edited here): out/followup_analyses/A_harmonized_template/PREREGISTRATION_A.md
(section 3.2 "pipeline held exactly fixed") and PREREGISTRATION_A_ADDENDUM_1.md.

Everything that determines the vectors is taken from the script that produced the stored Korean full-corpus
embeddings, `18b_embed_open.py`, and from the stored manifests out/18_encoder_dec/embeddings/*/.embed_manifest.json
(read, not guessed):
  * model registry    04_open_embedding.MODEL_CONFIGS  (bge-m3: local safetensors conversion, max_seq_length 8192,
                      pooling cls; qwen3-0.6b: "Qwen/Qwen3-Embedding-0.6B", max_seq_length 32768, pooling last_token;
                      no prefix for documents) -- IMPORTED, not copied
  * load order        SentenceTransformer(load_path, device="cpu"), model.max_seq_length = cfg, THEN .to("cuda")
                      (18b_embed_open.py:96-100; a device="cuda" constructor segfaults silently in this repo)
  * dtype             whatever that constructor yields in this stack -- NOT overridden (recorded in the manifest)
  * batch size        bge-m3 32, qwen3-0.6b 16 (the values recorded in the stored .embed_manifest.json files)
  * normalisation     normalize_embeddings=True, float32 output
  * row order         narratives/row_order.csv == 18_common.load_metadata() == the stored merged_ids.json (asserted)
  * text decoding     UTF-8 file contents, exactly the bytes whose sha256 the render stage recorded
  * truncation guard  embed_common.token_report + assert_no_truncation with the model's own tokenizer, BEFORE encoding
  * environment       /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3 (the interpreter of the Figure-1 matched design)

Before any encoding this task writes  embed/<arm>/source_hashes.csv  (sha256 of every source narrative file)
and checks it against the render stage's hashes.csv; a mismatch stops the task.

INFORMATIONAL (non-gating) reproduction check: the first 64 documents of the FULL corpus of record
(reports_pilot_pre) are re-encoded with THIS environment/loader and compared with the stored vectors
out/18_encoder_dec/embeddings/<model>/merged.npy[:64]; the minimum cosine is written to the manifest.
Nothing halts on it (Addendum 1, "Numerical-path note").

Outputs under out/followup_analyses/A_harmonized_template/embed/<arm>/<model>/:
  merged.npy (32329 x 1024 float32), merged_ids.json, n_tokens.npy, embed_manifest.json
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")             # no external service under any circumstance
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
_THREADS = os.environ.get("SLURM_CPUS_PER_TASK") or "8"
for _k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = _THREADS

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c18 = importlib.import_module("18_common")
oe = importlib.import_module("04_open_embedding")          # MODEL_CONFIGS only (read-only)
import embed_common                                          # noqa: E402

REV = SCRIPT_DIR.parent
A_DIR = REV / "out/followup_analyses/A_harmonized_template"
NARR = A_DIR / "narratives"
EMBED = A_DIR / "embed"
STORED = REV / "out/18_encoder_dec/embeddings"
ARMS = ("korean_mimic_compatible", "harmonized_keep_therapy", "thickness_matched_random")
MODELS = ("bge-m3", "qwen3-0.6b")
# array task id -> (arm, model); arm-major so that arm 1's two embeddings finish first
TASKS = [(a, m) for a in ARMS for m in MODELS]
STORED_BATCH = {"bge-m3": 32, "qwen3-0.6b": 16}           # out/18_encoder_dec/embeddings/*/.embed_manifest.json
EXPECTED_N, EXPECTED_DIM = 32_329, 1024
EXPECT_QWEN_SNAPSHOT = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
EXPECT_BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
THIS_FILES = ["scripts/62_harmonized_embed.py", "scripts/63_harmonized_matched.py", "scripts/64_harmonized_fields.py",
              "slurm/62_harmonized_A_embed.sh", "slurm/63_harmonized_A_matched.sh", "slurm/64_harmonized_A_aggregate.sh"]


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_info() -> dict:
    def run(*args) -> str:
        return subprocess.run(["git", "-C", str(REV), *args], capture_output=True, text=True, check=True).stdout.strip()

    dirty = run("status", "--porcelain", "--untracked-files=no", "--", *THIS_FILES)
    untracked = run("status", "--porcelain", "--", *THIS_FILES)
    return {"git_commit": run("rev-parse", "HEAD"), "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(untracked), "git_dirty_detail": untracked.splitlines()[:20], "git_dirty_tracked_only": bool(dirty)}


def versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "torch", "transformers", "sentence-transformers", "tokenizers", "pandas"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def env_record() -> dict:
    return {"sys_executable": sys.executable, "versions": versions(), "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"), **git_info()}


def atomic_write_json(obj, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, dest)


def model_provenance(model_slug: str, cfg: dict) -> dict:
    if model_slug == "bge-m3":
        prov = json.loads((Path(cfg["load_path"]) / "CONVERSION_PROVENANCE.json").read_text())
        st = Path(cfg["load_path"]) / "model.safetensors"
        sha = sha256_file(st)
        assert prov["source_revision"] == EXPECT_BGE_REVISION, prov["source_revision"]
        assert sha == prov["sha256"]["model.safetensors"], "bge-m3 safetensors sha256 differs from CONVERSION_PROVENANCE.json"
        return {"model_source_repo": prov["source_repo"], "model_revision": prov["source_revision"],
                "model_weights_file": str(st), "model_weights_sha256": sha}
    from huggingface_hub import snapshot_download
    snap = Path(snapshot_download(cfg["load_path"], local_files_only=True))
    assert snap.name == EXPECT_QWEN_SNAPSHOT, snap.name
    st = snap / "model.safetensors"
    return {"model_source_repo": cfg["load_path"], "model_revision": snap.name, "model_snapshot_dir": str(snap),
            "model_weights_sha256": sha256_file(st) if st.exists() else None}


def write_source_manifest(arm: str, ids: list[int], embed_root: Path) -> dict:
    """sha256 of every source narrative file, written BEFORE any embedding, and reconciled with hashes.csv."""
    import pandas as pd
    hashes = pd.read_csv(NARR.parent / "hashes.csv")
    hmap = {int(r.admission_ID): r.sha256 for r in hashes[hashes["arm"] == arm].itertuples()}
    rows, bad = [], []
    for aid in ids:
        h = sha256_bytes((NARR / arm / f"{aid}.txt").read_bytes())
        rows.append((aid, h))
        if hmap.get(aid) != h:
            bad.append(aid)
    if bad:
        raise SystemExit(f"[62] {len(bad)} narrative file(s) of arm {arm} differ from render hashes.csv, e.g. {bad[:5]} -- STOP")
    dest = embed_root / arm / "source_hashes.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    with open(tmp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["admission_ID", "sha256_narrative_file"])
        w.writerows(rows)
    os.replace(tmp, dest)
    corpus_sha = sha256_bytes("\n".join(f"{a}:{h}" for a, h in rows).encode())
    return {"source_hashes_csv": str(dest), "source_hashes_csv_sha256": sha256_file(dest), "corpus_sha256": corpus_sha,
            "n_files_hashed": len(rows), "all_equal_to_render_hashes_csv": True}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=int, default=None, help="1-based array task id (arm-major x model)")
    ap.add_argument("--arm", choices=ARMS)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="(testing) first N documents only; requires --out-root")
    ap.add_argument("--out-root", type=Path, default=EMBED, help="(testing) override the embedding output root")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.task is not None:
        args.arm, args.model = TASKS[args.task - 1]
    if not (args.arm and args.model):
        raise SystemExit("give --task N or --arm and --model")

    import numpy as np
    import pandas as pd
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    if args.limit and args.out_root == EMBED:
        raise SystemExit("--limit refuses to write into the real embedding directory; pass --out-root")
    out_dir = args.out_root / args.arm / args.model
    tag = f"[62/{args.arm}/{args.model}]"
    if not args.force and args.limit is None and (out_dir / "embed_manifest.json").exists() and (out_dir / "merged.npy").exists():
        print(f"{tag} already embedded; --force to redo", flush=True)
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False -- refusing to fall back to CPU")

    cfg = dict(oe.MODEL_CONFIGS[args.model])
    if (cfg["max_seq_length"], cfg["dim"], cfg["pooling"], cfg["prefix"]) != \
            ({"bge-m3": 8192, "qwen3-0.6b": 32768}[args.model], EXPECTED_DIM, {"bge-m3": "cls", "qwen3-0.6b": "last_token"}[args.model], ""):
        raise SystemExit(f"MODEL_CONFIGS[{args.model!r}] drifted: {cfg}")
    prov = model_provenance(args.model, cfg)

    # ---- rows: same admission IDs and order as the corpus of record and the stored embeddings (asserted)
    ro = pd.read_csv(NARR / "row_order.csv")
    ids = [int(a) for a in ro["admission_ID"]]
    meta_ids = [int(a) for a in c18.load_metadata()["admission_ID"]]
    stored_ids = [int(a) for a in json.loads((STORED / args.model / "merged_ids.json").read_text())]
    if not (ids == meta_ids == stored_ids) or len(ids) != EXPECTED_N:
        raise SystemExit(f"{tag} row order differs from 18_common.load_metadata() / stored merged_ids.json -- STOP")
    if args.limit:
        ids = ids[: args.limit]

    # ---- sha256 manifest of the source narratives, BEFORE embedding
    srcman = write_source_manifest(args.arm, ids, args.out_root)
    texts = [(NARR / args.arm / f"{aid}.txt").read_bytes().decode("utf-8") for aid in ids]
    print(f"{tag} n_docs={len(texts)} corpus_sha256={srcman['corpus_sha256'][:16]} (source hashes written, equal to render hashes.csv)", flush=True)

    # ---- G0 truncation guard (hard precondition)
    tokenizer = AutoTokenizer.from_pretrained(cfg["load_path"])
    count_fn = embed_common.hf_token_counter(tokenizer)
    report = embed_common.token_report(texts, count_fn, cfg["max_seq_length"], cfg["tokenizer_name"])
    embed_common.assert_no_truncation(report)
    n_tokens = np.fromiter((count_fn(t) for t in texts), dtype=np.int64, count=len(texts))
    print(f"{tag} truncation guard OK: n_tokens_median={report['n_tokens_median']} n_tokens_max={report['n_tokens_max']} "
          f"n_truncated={report['n_truncated']}", flush=True)

    batch_size = args.batch_size or STORED_BATCH[args.model]
    model = SentenceTransformer(cfg["load_path"], device="cpu")    # load on CPU, THEN move (cuda-constructor segfault)
    model.max_seq_length = cfg["max_seq_length"]
    if args.device == "cuda":
        model = model.to("cuda")
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    dev_name = torch.cuda.get_device_name(0) if args.device == "cuda" else "cpu"
    print(f"{tag} model on {args.device} ({dev_name}); param dtypes {dtypes}; batch {batch_size}; snapshot {prov['model_revision']}", flush=True)

    t0 = time.time()
    vecs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    enc_sec = time.time() - t0
    vecs = vecs.astype(np.float32)
    print(f"{tag} encoded {vecs.shape} in {enc_sec:.1f}s ({len(texts) / enc_sec:.1f} docs/s)", flush=True)
    if vecs.shape != (len(ids), EXPECTED_DIM) or not np.isfinite(vecs).all():
        raise SystemExit(f"{tag} unexpected embedding {vecs.shape} finite={np.isfinite(vecs).all()}")
    norms = np.linalg.norm(vecs, axis=1)

    # ---- informational, NON-gating reproduction check on the FULL corpus of record (first 64 documents)
    repro = {"gating": False}
    try:
        n_rep = 8 if args.limit else 64
        meta = c18.load_metadata().iloc[:n_rep]
        sub = [c18.report_path_for(n, int(a)).read_text(encoding="utf-8", errors="replace")
               for n, a in zip(meta["newpatient_ID"], meta["admission_ID"])]
        v = model.encode(sub, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True,
                         show_progress_bar=False).astype(np.float32)
        r = np.asarray(np.load(STORED / args.model / "merged.npy", mmap_mode="r")[:n_rep], dtype=np.float32)
        cos = (v * r).sum(1)
        repro.update({"n_docs": n_rep, "reference": f"out/18_encoder_dec/embeddings/{args.model}/merged.npy[:{n_rep}]",
                      "corpus": "reports_pilot_pre (FULL template, corpus of record)",
                      "min_cosine": float(cos.min()), "median_cosine": float(np.median(cos)),
                      "max_abs_diff": float(np.abs(v - r).max()), "n_cosine_lt_0_9999": int((cos < 0.9999).sum())})
        print(f"{tag} repro (first 64 FULL docs vs stored): min cosine {repro['min_cosine']:.8f}", flush=True)
    except Exception as exc:  # noqa: BLE001 -- informational
        repro["error_class"] = type(exc).__name__
        repro["error"] = str(exc)[:200]

    out_dir.mkdir(parents=True, exist_ok=True)
    vec_path = out_dir / "merged.npy"
    tmp = vec_path.with_name(f".tmp_{os.getpid()}_merged.npy")
    with open(tmp, "wb") as fh:
        np.save(fh, vecs)
    os.replace(tmp, vec_path)
    atomic_write_json(ids, out_dir / "merged_ids.json")
    tmpn = out_dir / f".tmp_{os.getpid()}_n_tokens.npy"
    with open(tmpn, "wb") as fh:
        np.save(fh, n_tokens)
    os.replace(tmpn, out_dir / "n_tokens.npy")

    manifest = {
        "analysis": "A_harmonized_template", "phase": "2 (Addendum 1)", "arm": args.arm, "model_slug": args.model,
        "model_load_path": cfg["load_path"], **prov, "pooling": cfg["pooling"], "prefix": cfg["prefix"], "normalize": True,
        "device": args.device, "device_name": dev_name, "cuda_runtime": torch.version.cuda,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32), "model_param_dtypes": dtypes,
        "dtype_out": str(vecs.dtype), "batch_size": batch_size, "batch_size_source": "out/18_encoder_dec/embeddings/*/.embed_manifest.json",
        "n_docs": len(texts), "dim": int(vecs.shape[1]), "enc_sec": round(enc_sec, 2), "docs_per_sec": round(len(texts) / enc_sec, 3),
        "norm_min": float(norms.min()), "norm_max": float(norms.max()), "vectors_sha256": sha256_file(vec_path),
        "narratives_dir": str(NARR / args.arm), **srcman,
        "reference_stored_manifest": str(STORED / args.model / ".embed_manifest.json"),
        "repro_check_first64_full_corpus_vs_stored": repro, "torch_version": torch.__version__, "torch_threads": torch.get_num_threads(),
        "started_utc_iso": now_iso(), **env_record(),
    }
    manifest = embed_common.merge_report_into_manifest(manifest, report)          # re-asserts no truncation
    atomic_write_json(manifest, out_dir / "embed_manifest.json")
    print(f"{tag} wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
