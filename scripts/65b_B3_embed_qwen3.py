#!/usr/bin/env python3
"""Task 65b -- Analysis B, arm B3 (FOLLOWUP-2026-09-21-B): Qwen3-Embedding-0.6B embedding of
the Korean `no_department` narratives (local GPU; no external service, no API call).

    B3  qwen3_no_department   out/21_published_variants/narratives/no_department/
                              (32,329 docs, the SAME directory the 3-large and 3-small
                              no_department arms embedded)

Governing documents (FROZEN, not edited here):
  out/followup_analyses/B_symmetric_perturbation/PREREGISTRATION_B.md  (H-B3, sections 3.1, 3.4, 4.4)
  out/followup_analyses/qc/AUTHOR_DECISIONS_2026-09-21.md              (B section 7.1: B3 IS run;
                                                                     presentational / symmetry-completing)
B3 adds limited inference by design; it is reported whatever it shows.

Embedding recipe = `audit7_embed_qwen3.py` (the script that produced the existing Qwen3 ablation arms
out/41_audit7/embeddings/ablation_{full,no_therapy,no_organism}), re-stated here only because that
script's output directory is hard-wired to the pre-existing out/41_audit7 and it has no truncation
guard. Everything that determines the vectors is taken from the same sources:
  * model            04_open_embedding.MODEL_CONFIGS["qwen3-0.6b"]  (load_path
                     "Qwen/Qwen3-Embedding-0.6B", max_seq_length 32768, pooling last_token, no prefix)
  * load order       SentenceTransformer(load_path, device="cpu") THEN .to("cuda")
  * batch size       16 (the batch size in the original Qwen3 manifest, out/18_encoder_dec/
                     embeddings/qwen3-0.6b/.embed_manifest.json, and in slurm/audit7_embed_qwen3.sh)
  * normalisation    normalize_embeddings=True, float32 output
  * row order        18_common.load_metadata() admission_ID order == the stored Qwen3 merged_ids.json
  * text decoding    Path.read_text(encoding="utf-8", errors="replace")   (as audit7)
  * truncation guard embed_common.token_report + assert_no_truncation with the Qwen3 tokenizer, run
                     BEFORE encoding (as 18b_embed_open.py); audit7 lacked it.

STAGES
------
verify   CPU. Proves the narratives are what the preregistration says they are and writes the sha256
         manifest BEFORE any embedding:
           (1) 21_common.apply_variant(reports_pilot_pre text, "no_department") == narratives file,
               for every one of the 32,329 documents;
           (2) the directory is the one 21b_embed_openai (3-large, c21.NARRATIVES_DIR) and
               49_3small_no_department (s49.NARRATIVES) read;
           (3) cl100k token total == generation_summary.csv == the 3-large billed total == the 3-small
               billed total (identical text was embedded by both OpenAI arms);
           (4) row order == stored Qwen3 merged_ids.json == the Qwen3 reference labels' admission_ID.
         Writes source_hashes.csv and narrative_verification.json (`proceed`).
embed    GPU. Refuses to run unless `proceed` is true and every narrative file still hashes to the
         recorded value. Runs the truncation guard, encodes, validates the array and writes
         merged.npy / merged_ids.json / manifest.json.

Writes ONLY under out/followup_analyses/B_symmetric_perturbation/embeddings/B3_qwen3_no_department/.
"""
from __future__ import annotations

import os

_THREADS = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("STRESS_THREADS", "8")
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
c21 = importlib.import_module("21_common")
oe = importlib.import_module("04_open_embedding")          # MODEL_CONFIGS only (read-only)
import embed_common                                          # noqa: E402

REV = SCRIPT_DIR.parent
OUT_B = REV / "out/followup_analyses/B_symmetric_perturbation"
ARM_KEY = "B3"
ARM_ID = "B3_qwen3_no_department"
VARIANT = "no_department"
MODEL_SLUG = "qwen3-0.6b"
ARM_DIR = OUT_B / "embeddings" / ARM_ID
NARRATIVES = REV / "out/21_published_variants/narratives" / VARIANT
STORED_QWEN3 = REV / "out/18_encoder_dec/embeddings/qwen3-0.6b"
STORED_QWEN3_LABELS = REV / "out/18_encoder_dec/recluster/qwen3-0.6b/labels.npz"
BATCH_SIZE = 16
EXPECTED_DIM = 1024
EXPECTED_N = 32_329
THIS_FILES = ["scripts/65b_B3_embed_qwen3.py", "slurm/65b_B3_embed.sh",
              "scripts/66b_B3_cluster.py", "slurm/66b_B3_cluster.sh"]


# --------------------------------------------------------------------------- utilities
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
        return subprocess.run(["git", "-C", str(REV), *args], capture_output=True,
                              text=True, check=True).stdout.strip()

    dirty = run("status", "--porcelain", "--", *THIS_FILES)
    return {"git_commit": run("rev-parse", "HEAD"), "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "b3_scripts_dirty": bool(dirty), "b3_scripts_dirty_detail": dirty}


def versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "torch", "transformers", "sentence-transformers", "tiktoken", "tokenizers"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def env_record() -> dict:
    return {"sys_executable": sys.executable, "versions": versions(), "hostname": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "thread_env": {k: os.environ.get(k) for k in
                           ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"), **git_info()}


def atomic_write_json(obj: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, dest)


def ordered_ids() -> list:
    """Row order: 18_common.load_metadata() == the stored Qwen3 embedding's merged_ids.json."""
    meta = c18.load_metadata()
    return [int(a) for a in meta["admission_ID"]]


# --------------------------------------------------------------------------- verify
def stage_verify() -> dict:
    import numpy as np

    s49 = importlib.import_module("49_3small_no_department")
    ARM_DIR.mkdir(parents=True, exist_ok=True)
    meta = c18.load_metadata()
    ids = [int(a) for a in meta["admission_ID"]]
    if len(ids) != EXPECTED_N:
        raise SystemExit(f"expected {EXPECTED_N} admissions, found {len(ids)}")

    # (4) row order agreement with the stored Qwen3 objects
    stored_ids = [int(x) for x in json.loads((STORED_QWEN3 / "merged_ids.json").read_text())]
    ref_ids = [int(x) for x in np.load(STORED_QWEN3_LABELS, allow_pickle=True)["admission_ID"]]
    order_ok = (ids == stored_ids) and (ids == ref_ids)

    # (2) directory identity with the two OpenAI no_department arms
    dir_3large = c21.NARRATIVES_DIR / VARIANT
    dir_3small = s49.NARRATIVES / VARIANT
    same_dir = (NARRATIVES == dir_3large == dir_3small)

    # (1) predicate equality, document by document; sha256 manifest
    meta_i = meta.set_index("admission_ID")
    n_identical, mismatches, n_cr = 0, [], 0
    texts_for_tokens, rows = [], []
    mtimes = []
    for aid in ids:
        narr_path = NARRATIVES / f"{aid}.txt"
        if not narr_path.exists():
            raise SystemExit(f"no narrative for admission {aid} at {narr_path}")
        narr_raw = narr_path.read_bytes()
        mtimes.append(narr_path.stat().st_mtime)
        src_path = c18.report_path_for(meta_i.loc[aid, "newpatient_ID"], aid)
        src_raw = src_path.read_bytes()
        narr_text = narr_raw.decode("utf-8", errors="replace")
        src_text = src_raw.decode("utf-8", errors="replace")
        ok = (c21.apply_variant(src_text, VARIANT) == narr_text)
        n_identical += int(ok)
        if not ok and len(mismatches) < 5:
            mismatches.append(aid)
        n_cr += int(b"\r" in narr_raw)
        rows.append((aid, sha256_bytes(narr_raw), sha256_bytes(src_raw)))
        texts_for_tokens.append(narr_text)
    all_identical = (n_identical == len(ids))

    with open(ARM_DIR / "source_hashes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["admission_ID", "sha256_narrative_file", "sha256_source_report"])
        w.writerows(rows)
    manifest_sha = sha256_file(ARM_DIR / "source_hashes.csv")

    # (3) token-level identity with what the two OpenAI arms embedded (cl100k)
    count_fn = embed_common.openai_token_counter("cl100k_base")
    cl100k_total = int(sum(count_fn(t) for t in texts_for_tokens))
    with open(REV / "out/21_published_variants/generation_summary.csv") as fh:
        summ = {r["variant"]: r for r in csv.DictReader(fh)}[VARIANT]
    m3l = json.loads((REV / "out/21_published_variants/embeddings/no_department/"
                      ".openai_embed_manifest_no_department.json").read_text())
    m3s = json.loads((REV / "out/49_3small_ablation/embeddings/no_department/manifest.json").read_text())
    tokens = {"cl100k_total_recomputed": cl100k_total,
              "generation_summary_total_tokens_projected": int(summ["total_tokens_projected"]),
              "3large_total_tokens_billed": int(m3l["total_tokens_billed"]),
              "3small_total_tokens_billed": int(m3s["total_tokens_billed"])}
    tokens_ok = len(set(tokens.values())) == 1

    proceed = bool(order_ok and same_dir and all_identical and tokens_ok)
    result = {
        "arm": ARM_KEY, "arm_id": ARM_ID, "variant": VARIANT, "n_docs": len(ids),
        "narratives_dir": str(NARRATIVES),
        "check_1_apply_variant_equals_narrative_file": {
            "predicate": "21_common.apply_variant(reports_pilot_pre text, 'no_department')",
            "n_identical": n_identical, "n_docs": len(ids), "all_identical": all_identical,
            "first_mismatching_admission_IDs": mismatches},
        "check_2_same_directory_as_openai_no_department_arms": {
            "b3": str(NARRATIVES), "3large_21b_c21.NARRATIVES_DIR": str(dir_3large),
            "3small_49_s49.NARRATIVES": str(dir_3small), "same": same_dir},
        "check_3_token_identity_with_openai_arms": {**tokens, "all_equal": tokens_ok},
        "check_4_row_order_equals_stored_qwen3_ids_and_reference_labels": order_ok,
        "corpus_wide_lines_removed": int(summ["corpus_wide_lines_removed"]),
        "lines_removed_per_doc": int(summ["corpus_wide_lines_removed"]) / len(ids),
        "n_narrative_files_containing_CR": n_cr,
        "narrative_mtime_min": datetime.fromtimestamp(min(mtimes)).astimezone().isoformat(timespec="seconds"),
        "narrative_mtime_max": datetime.fromtimestamp(max(mtimes)).astimezone().isoformat(timespec="seconds"),
        "source_hashes_csv": str(ARM_DIR / "source_hashes.csv"),
        "source_hashes_csv_sha256": manifest_sha,
        "proceed": proceed, "written": now_iso(), "api_call_made": False, **env_record(),
    }
    atomic_write_json(result, ARM_DIR / "narrative_verification.json")
    print(json.dumps(result, indent=2), flush=True)
    if not proceed:
        raise SystemExit("STOP: narrative verification failed -- must not embed.")
    print(f"[65b/verify] OK: {n_identical}/{len(ids)} narratives == apply_variant(., 'no_department'); "
          f"tokens {cl100k_total} identical across generation_summary / 3-large / 3-small; "
          f"manifest sha256 {manifest_sha}", flush=True)
    return result


# --------------------------------------------------------------------------- embed
def _resolve_model_provenance(model, load_path: str) -> dict:
    prov = {"load_path": load_path, "commit_hash": None, "weights_file": None, "weights_sha256": None}
    try:
        prov["commit_hash"] = getattr(model[0].auto_model.config, "_commit_hash", None)
    except Exception:  # noqa: BLE001 -- provenance is best effort, never fatal
        pass
    try:
        from huggingface_hub import try_to_load_from_cache
        wf = try_to_load_from_cache(load_path, "model.safetensors")
        if isinstance(wf, str):
            prov["weights_file"] = wf
            prov["weights_sha256"] = sha256_file(Path(wf))
    except Exception:  # noqa: BLE001
        pass
    return prov


def stage_embed(device: str, batch_size: int, limit: int | None) -> None:
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    ver_path = ARM_DIR / "narrative_verification.json"
    if not ver_path.exists() or not json.loads(ver_path.read_text()).get("proceed", False):
        raise SystemExit(f"{ver_path} missing or proceed=false -- run --stage verify first")
    merged_path, ids_path, manifest_path = ARM_DIR / "merged.npy", ARM_DIR / "merged_ids.json", ARM_DIR / "manifest.json"
    if merged_path.exists() and ids_path.exists() and manifest_path.exists() and limit is None:
        print(f"[65b/embed] {ARM_ID}: already present, skipping", flush=True)
        return
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("device=cuda requested but torch.cuda.is_available() is False -- refusing to fall back to CPU")

    cfg = oe.MODEL_CONFIGS[MODEL_SLUG]
    if (cfg["max_seq_length"], cfg["dim"], cfg["pooling"], cfg["prefix"]) != (32768, EXPECTED_DIM, "last_token", ""):
        raise SystemExit(f"MODEL_CONFIGS['qwen3-0.6b'] drifted: {cfg}")
    ids = ordered_ids()
    if limit is not None:
        ids = ids[:limit]

    # narratives: hashes must still equal the verified manifest; text decoded as audit7 did
    recorded = {}
    with open(ARM_DIR / "source_hashes.csv") as fh:
        for row in csv.DictReader(fh):
            recorded[int(row["admission_ID"])] = row["sha256_narrative_file"]
    texts, changed = [], []
    for aid in ids:
        p = NARRATIVES / f"{aid}.txt"
        if sha256_bytes(p.read_bytes()) != recorded.get(aid):
            changed.append(aid)
        texts.append(p.read_text(encoding="utf-8", errors="replace"))
    if changed:
        raise SystemExit(f"{len(changed)} narratives changed since verify, e.g. {changed[:5]}")
    print(f"[65b/embed] {ARM_ID}: {len(texts)} docs from {NARRATIVES} (hashes match manifest)", flush=True)

    # G0 truncation guard BEFORE encoding (hard precondition)
    tokenizer = AutoTokenizer.from_pretrained(cfg["load_path"])
    report = embed_common.token_report(texts, embed_common.hf_token_counter(tokenizer),
                                       cfg["max_seq_length"], cfg["tokenizer_name"])
    embed_common.assert_no_truncation(report)
    print(f"[65b/embed] truncation guard OK: n_tokens_median={report['n_tokens_median']} "
          f"n_tokens_max={report['n_tokens_max']} n_truncated={report['n_truncated']}", flush=True)

    model = SentenceTransformer(cfg["load_path"], device="cpu")     # load on CPU, THEN move (cuda segfault gotcha)
    model.max_seq_length = cfg["max_seq_length"]                    # set explicitly, never rely on default
    if device == "cuda":
        model = model.to("cuda")
    prov = _resolve_model_provenance(model, cfg["load_path"])
    dev_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    print(f"[65b/embed] model on {device} ({dev_name}); snapshot {prov['commit_hash']}; batch {batch_size}", flush=True)

    t0 = time.time()
    vecs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    enc_sec = time.time() - t0
    vecs = vecs.astype(np.float32)
    if limit is not None:
        print(f"[65b/embed] --limit smoke test only ({len(ids)} docs, {enc_sec:.1f}s); nothing written", flush=True)
        return

    if vecs.shape != (EXPECTED_N, EXPECTED_DIM) or not np.isfinite(vecs).all():
        raise SystemExit(f"unexpected embedding {vecs.shape} finite={np.isfinite(vecs).all()}")
    norms = np.linalg.norm(vecs, axis=1)

    # informational, NON-gating: does this environment reproduce the stored vectors of the existing Qwen3
    # `full` arm? (first 64 docs of narratives/full re-encoded here vs out/41_audit7/.../ablation_full).
    repro = {"gating": False}
    try:
        full_dir = REV / "out/21_published_variants/narratives/full"
        ref = np.load(REV / "out/41_audit7/embeddings/ablation_full/merged.npy", mmap_mode="r")
        sub = [(full_dir / f"{aid}.txt").read_text(encoding="utf-8", errors="replace") for aid in ids[:64]]
        v = model.encode(sub, batch_size=batch_size, convert_to_numpy=True,
                         normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        r = np.asarray(ref[:64], dtype=np.float32)
        repro.update({"n_docs": 64, "reference": "out/41_audit7/embeddings/ablation_full/merged.npy[:64]",
                      "min_cosine": float((v * r).sum(1).min()), "max_abs_diff": float(np.abs(v - r).max())})
    except Exception as exc:  # noqa: BLE001
        repro["error_class"] = type(exc).__name__

    tmp = merged_path.with_name(f".tmp_{os.getpid()}_merged.npy")   # np.save needs a .npy suffix
    np.save(tmp, vecs)
    os.replace(tmp, merged_path)
    ids_path.write_text(json.dumps(ids))

    manifest = {
        "arm": ARM_KEY, "arm_id": ARM_ID, "variant": VARIANT, "model_slug": MODEL_SLUG,
        "model_load_path": cfg["load_path"], "model_provenance": prov,
        "pooling": cfg["pooling"], "prefix": cfg["prefix"], "normalize": True,
        "device": device, "device_name": dev_name, "dtype_out": str(vecs.dtype),
        "batch_size": batch_size, "n_docs": len(texts), "dim": int(vecs.shape[1]),
        "enc_sec": round(enc_sec, 2), "docs_per_sec": round(len(texts) / enc_sec, 3),
        "norm_min": float(norms.min()), "norm_max": float(norms.max()),
        "merged_npy_sha256": sha256_file(merged_path),
        "narratives_dir": str(NARRATIVES), "narrative_hash_manifest": str(ARM_DIR / "source_hashes.csv"),
        "narrative_hash_manifest_sha256": sha256_file(ARM_DIR / "source_hashes.csv"),
        "reference_arm_manifest": str(STORED_QWEN3 / ".embed_manifest.json"),
        "repro_check_vs_existing_full_arm": repro,
        "started_utc_iso": now_iso(), **env_record(),
    }
    manifest = embed_common.merge_report_into_manifest(manifest, report)   # re-asserts no truncation
    atomic_write_json(manifest, manifest_path)
    print(json.dumps({k: v for k, v in manifest.items() if k != "model_provenance"}, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["verify", "embed"])
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--limit", type=int, default=None, help="embed smoke test on the first N docs; writes nothing")
    args = ap.parse_args()
    if args.stage == "verify":
        stage_verify()
    else:
        stage_embed(args.device, args.batch_size, args.limit)


if __name__ == "__main__":
    main()
