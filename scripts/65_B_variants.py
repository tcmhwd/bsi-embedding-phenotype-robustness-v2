#!/usr/bin/env python3
"""Task 65 -- Analysis B (FOLLOWUP-2026-09-21-B): fresh `text-embedding-3-small`
embeddings for the two arms that need one.

    B1b  unmodified_reembedded  the byte-identical unmodified corpus of record
                                (out/21_published_variants/narratives/full/)
    B2   no_therapy             out/21_published_variants/narratives/no_therapy/

(B1a `unmodified_seed_only` reuses the pre-existing stored 3-small embedding and
needs no embedding: see 66_B_cluster.py.)

Governing documents (FROZEN, not edited here):
  out/followup_analyses/B_symmetric_perturbation/PREREGISTRATION_B.md
  out/followup_analyses/qc/EXECUTION_PLAN.md   (sections 1, 3, 4, 5 B-1/B-2, 6)

Nothing is re-implemented that already exists:
  * OpenAI client factory, model constants, price, hard cap, shard size and the
    reference partition   <- 49_3small_no_department
  * token-budget batching, retry, shard bookkeeping   <- embed_variants_openai
  * truncation guard, token counter   <- embed_common
  * variant predicate (used ONLY to verify B2's narratives)   <- 21_common
This file adds: the arm table, the byte-identity / predicate verification, the
combined-cost gate, a single-document authentication check, a log scrubber, and
output paths under out/followup_analyses/B_symmetric_perturbation/.

STAGES
------
count_tokens  NO API CALL. Verifies the narratives, writes source_hashes.csv and
              token_report.json (`proceed`, projected cost). The embed stage
              refuses to run unless `proceed` is true.
auth_check    ONE embedding of a fixed, non-patient string, to confirm the
              credential authenticates before any bulk spend. Records the
              exception CLASS only on failure -- never the message.
embed         sharded (SHARD_SIZE), resumable, cost re-checked after every batch.
merge         shards -> merged.npy / merged_ids.json in the stored-partition row
              order, plus (B1b) numeric agreement with the stored embedding.

CREDENTIALS
-----------
Handled exclusively inside `49_3small_no_department._make_client()`:
environment `OPENAI_API_KEY`, else `~/.secrets/openai_api_key`. This file never
reads, prints, logs or writes the key. As a safety net stdout/stderr are wrapped
in a scrubber that redacts anything shaped like an API key, because the shared
retry helper prints `repr(exception)` and an authentication error's repr can
carry a masked key fragment.

Writes ONLY under out/followup_analyses/B_symmetric_perturbation/.
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
s49 = importlib.import_module("49_3small_no_department")
c18 = s49.c18
c21 = s49.c21
evo = s49.evo
embed_common = s49.embed_common

REV = s49.REV
OUT_B = REV / "out/followup_analyses/B_symmetric_perturbation"
EMB_ROOT = OUT_B / "embeddings"
STORED_3SMALL = REV / "out/18_encoder_dec/embeddings/openai-3small"

MODEL_NAME = s49.MODEL_NAME
EMBED_DIM = s49.EMBED_DIM
MAX_SEQ_LENGTH = s49.MAX_SEQ_LENGTH
TOKENIZER_NAME = s49.TOKENIZER_NAME
PRICE_PER_MILLION_TOKENS_USD = s49.PRICE_PER_MILLION_TOKENS_USD
HARD_CAP_USD = s49.HARD_CAP_USD          # 4.00, the 49 gate, unchanged
SHARD_SIZE = s49.SHARD_SIZE              # 2000
REFERENCE = s49.REFERENCE                # 3-small stored whole-corpus partition (k=22, 277 noise)
NARRATIVES = s49.NARRATIVES

# Task-level stop rule: if the two arms' PROJECTED cost exceeds this, stop and report.
PLAN_STOP_USD = 2.00

# arm key -> (arm_id, narrative variant, how the narratives are verified)
ARMS = {
    "B1b": {"arm_id": "B1b_unmodified_reembedded", "variant": "full"},
    "B2": {"arm_id": "B2_no_therapy", "variant": "no_therapy"},
}
AUTH_CHECK_TEXT = "authentication check"   # fixed, non-patient string


# --------------------------------------------------------------------------- utilities
class _Scrub:
    """Redact anything shaped like an API key from a text stream."""
    _pat = re.compile(r"sk-[A-Za-z0-9_\-\*\.]{3,}")

    def __init__(self, stream):
        self._stream = stream

    def write(self, s):
        return self._stream.write(self._pat.sub("sk-[REDACTED]", s))

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def install_scrubber() -> None:
    if not isinstance(sys.stdout, _Scrub):
        sys.stdout = _Scrub(sys.stdout)
    if not isinstance(sys.stderr, _Scrub):
        sys.stderr = _Scrub(sys.stderr)


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


def package_versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scikit-learn", "umap-learn", "hdbscan", "numba",
                "pyarrow", "openai", "tiktoken"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def git_info() -> dict:
    def run(*args) -> str:
        return subprocess.run(["git", "-C", str(REV), *args], capture_output=True,
                              text=True, check=True).stdout.strip()

    dirty = run("status", "--porcelain", "--", "scripts/65_B_variants.py",
                "scripts/66_B_cluster.py", "slurm/65_B_embed.sh", "slurm/66_B_cluster.sh")
    return {"git_commit": run("rev-parse", "HEAD"),
            "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "analysis_scripts_dirty": bool(dirty)}


def env_record() -> dict:
    return {
        "sys_executable": sys.executable,
        "versions": package_versions(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "thread_pins": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS",
                         "OPENBLAS_NUM_THREADS")},
        **git_info(),
    }


def arm_dir(arm: str) -> Path:
    return EMB_ROOT / ARMS[arm]["arm_id"]


def ref_ids() -> list:
    return [int(x) for x in np.load(REFERENCE, allow_pickle=True)["admission_ID"]]


def read_narratives(arm: str):
    """[(admission_ID, text, sha256_of_file_bytes)] in the stored partition's row
    order. Text is decoded exactly as 18c/21b/21a decoded it (utf-8, errors=
    'replace'), so the strings sent are the strings the original 3-small run
    sent. (49's own loader calls read_text() with the locale default.)"""
    src = NARRATIVES / ARMS[arm]["variant"]
    if not src.exists():
        raise SystemExit(f"{src} missing")
    out = []
    for aid in ref_ids():
        p = src / f"{aid}.txt"
        if not p.exists():
            raise SystemExit(f"no narrative for admission {aid} at {p}")
        raw = p.read_bytes()
        out.append((aid, raw.decode("utf-8", errors="replace"), sha256_bytes(raw)))
    return out


def _token_report_path(arm: str) -> Path:
    return arm_dir(arm) / "token_report.json"


def _require_cost_cleared(arm: str) -> dict:
    path = _token_report_path(arm)
    if not path.exists():
        raise SystemExit(f"{path} missing -- run --stage count_tokens --arm {arm} first")
    rep = json.loads(path.read_text())
    if not rep.get("proceed", False):
        raise SystemExit("count_tokens verdict is proceed=False -- refusing to embed")
    return rep


def _ledger_cost(exclude: str | None = None) -> float:
    total = 0.0
    for a in ARMS:
        if a == exclude:
            continue
        m = arm_dir(a) / "manifest.json"
        if m.exists():
            total += json.loads(m.read_text()).get("cost_usd", 0.0)
    return total


# --------------------------------------------------------------------------- count_tokens
def stage_count_tokens(arm: str) -> dict:
    variant = ARMS[arm]["variant"]
    d = arm_dir(arm)
    d.mkdir(parents=True, exist_ok=True)

    triples = read_narratives(arm)
    ids = [a for a, _, _ in triples]
    texts = [t for _, t, _ in triples]
    narr_hash = [h for _, _, h in triples]

    # ---- verification that the text is what the preregistration says it is ----
    meta = c18.load_metadata().set_index("admission_ID")
    n_identical, mismatches = 0, []
    src_hash = []
    for aid, text, h in triples:
        src_path = c18.report_path_for(meta.loc[aid, "newpatient_ID"], aid)
        raw = src_path.read_bytes()
        src_hash.append(sha256_bytes(raw))
        if arm == "B1b":
            ok = (sha256_bytes(raw) == h)
        else:  # B2: the narrative must equal the existing predicate applied to the corpus of record
            src_text = raw.decode("utf-8", errors="replace")
            ok = (c21.apply_variant(src_text, variant) == text)
        n_identical += int(ok)
        if not ok and len(mismatches) < 5:
            mismatches.append(int(aid))
    check_name = ("sha256(narratives/full/<id>.txt) == sha256(reports_pilot_pre/<report>)"
                  if arm == "B1b" else
                  "21_common.apply_variant(reports_pilot_pre text, 'no_therapy') == narratives/no_therapy/<id>.txt")
    verified = (n_identical == len(triples))

    with open(d / "source_hashes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["admission_ID", "sha256_narrative_file", "sha256_source_report"])
        for aid, hn, hs in zip(ids, narr_hash, src_hash):
            w.writerow([aid, hn, hs])

    # ---- tokens and cost (one tokenisation pass) ----
    count_fn = embed_common.openai_token_counter(TOKENIZER_NAME)
    counts = {t: count_fn(t) for t in set(texts)}
    report = embed_common.token_report(texts, lambda t: counts[t], MAX_SEQ_LENGTH, TOKENIZER_NAME)
    total = int(sum(counts[t] for t in texts))
    cost = round(total / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD, 4)

    # ---- cross-checks against numbers already on disk ----
    summary = {}
    with open(REV / "out/21_published_variants/generation_summary.csv") as fh:
        for row in csv.DictReader(fh):
            summary[row["variant"]] = row
    recorded_tokens = int(summary[variant]["total_tokens_projected"])
    crosschecks = {"generation_summary_total_tokens": recorded_tokens}
    must_be_true = {"matches_generation_summary": total == recorded_tokens}
    if arm == "B2":
        # The 3-large no_therapy arm was embedded by 21b_embed_openai.py from
        # c21.NARRATIVES_DIR / "no_therapy"; confirm the same directory, id order and tokens.
        m3l = json.loads((REV / "out/21_published_variants/embeddings/no_therapy/"
                          ".openai_embed_manifest_no_therapy.json").read_text())
        ids3l = json.loads((REV / "out/21_published_variants/embeddings/no_therapy/"
                            "merged_ids.json").read_text())
        crosschecks["3large_no_therapy_tokens_billed"] = int(m3l["total_tokens_billed"])
        must_be_true.update({
            "narratives_dir_is_the_one_21b_read": (NARRATIVES / variant) == (c21.NARRATIVES_DIR / variant),
            "matches_3large_tokens_billed": total == int(m3l["total_tokens_billed"]),
            # 21b merges its shards in ascending admission_ID order, not the stored-partition
            # order, so the comparison that is meaningful here is the SET of documents.
            "3large_no_therapy_id_set_equals_reference": set(int(x) for x in ids3l) == set(ids),
        })
    else:
        stored_ids = json.loads((STORED_3SMALL / "merged_ids.json").read_text())
        stored_manifest_path = STORED_3SMALL / ".openai_embed_manifest_3small.json"
        if stored_manifest_path.exists():
            sm = json.loads(stored_manifest_path.read_text())
            crosschecks["stored_3small_total_tokens_billed"] = int(sm["total_tokens_billed"])
            must_be_true["matches_stored_3small_tokens_billed"] = total == int(sm["total_tokens_billed"])
        must_be_true["stored_3small_row_order_equals_reference"] = [int(x) for x in stored_ids] == ids
    crosschecks.update(must_be_true)
    crosschecks_ok = all(must_be_true.values())

    # ---- combined-cost gate across both arms (task-level stop rule) ----
    other = [a for a in ARMS if a != arm][0]
    other_rep_path = _token_report_path(other)
    other_cost = json.loads(other_rep_path.read_text()).get("projected_cost_usd", 0.0) \
        if other_rep_path.exists() else 0.0
    combined = round(cost + other_cost, 4)

    proceed = bool(verified and crosschecks_ok and report["frac_truncated"] == 0.0
                   and cost <= HARD_CAP_USD and combined <= PLAN_STOP_USD)
    result = {
        "arm": arm, "arm_id": ARMS[arm]["arm_id"], "variant": variant, "model": MODEL_NAME,
        "n_docs": len(texts), "total_tokens": total, "projected_cost_usd": cost,
        "price_per_million_tokens_usd": PRICE_PER_MILLION_TOKENS_USD,
        "hard_cap_usd": HARD_CAP_USD, "plan_stop_usd": PLAN_STOP_USD,
        "combined_projected_cost_usd_known_arms": combined,
        "n_truncated": report["n_truncated"], "frac_truncated": report["frac_truncated"],
        "token_report": report,
        "verification": {"check": check_name, "n_identical": n_identical, "n_docs": len(triples),
                         "all_identical": verified, "first_mismatching_admission_IDs": mismatches},
        "crosschecks": crosschecks, "crosschecks_ok": crosschecks_ok,
        "narratives_dir": str(NARRATIVES / variant),
        "source_hashes_csv": str(d / "source_hashes.csv"),
        "proceed": proceed, "written": now_iso(),
        "api_call_made": False, **env_record(),
    }
    c18.atomic_write_json(result, _token_report_path(arm))
    print(json.dumps({k: v for k, v in result.items() if k not in ("token_report",)}, indent=2), flush=True)
    if not proceed:
        why = []
        if not verified: why.append("narrative verification failed")
        if not crosschecks_ok: why.append("cross-check failed")
        if report["frac_truncated"] > 0: why.append("truncation > 0")
        if cost > HARD_CAP_USD: why.append(f"cost ${cost} > cap ${HARD_CAP_USD}")
        if combined > PLAN_STOP_USD: why.append(f"combined ${combined} > plan stop ${PLAN_STOP_USD}")
        print("STOP: " + "; ".join(why) + " -- must not proceed to embed.", flush=True)
    return result


# --------------------------------------------------------------------------- auth_check
def stage_auth_check() -> None:
    """One embedding of a fixed non-patient string. Failure records the exception
    CLASS (and HTTP status if any) only -- never the message."""
    install_scrubber()
    EMB_ROOT.mkdir(parents=True, exist_ok=True)
    dest = EMB_ROOT / "auth_check.json"
    client = s49._make_client()
    try:
        resp = client.embeddings.create(model=MODEL_NAME, input=[AUTH_CHECK_TEXT])
        dim = len(resp.data[0].embedding)
        rec = {"ok": True, "model": MODEL_NAME, "dim": dim, "n_documents_sent": 1,
               "document": "fixed non-patient string", "tokens_billed": int(resp.usage.total_tokens),
               "written": now_iso()}
    except Exception as exc:  # noqa: BLE001 -- class name only, by design
        rec = {"ok": False, "error_class": type(exc).__name__,
               "http_status": getattr(exc, "status_code", None), "written": now_iso()}
    c18.atomic_write_json(rec, dest)
    print(json.dumps(rec), flush=True)
    if not rec["ok"]:
        raise SystemExit(2)
    if rec["dim"] != EMBED_DIM:
        raise SystemExit(f"unexpected dimension {rec['dim']} != {EMBED_DIM}")


# --------------------------------------------------------------------------- embed
def stage_embed(arm: str) -> None:
    install_scrubber()
    cleared = _require_cost_cleared(arm)
    auth = EMB_ROOT / "auth_check.json"
    if not auth.exists() or not json.loads(auth.read_text()).get("ok", False):
        raise SystemExit(f"{auth} missing or not ok -- run --stage auth_check first")
    d = arm_dir(arm)

    triples = read_narratives(arm)
    recorded = {}
    with open(d / "source_hashes.csv") as fh:
        for row in csv.DictReader(fh):
            recorded[int(row["admission_ID"])] = row["sha256_narrative_file"]
    changed = [a for a, _, h in triples if recorded.get(a) != h]
    if changed:
        raise SystemExit(f"{len(changed)} narratives changed since count_tokens, e.g. {changed[:5]}")

    pairs = [(a, t) for a, t, _ in triples]
    already = evo._already_done_ids(d)
    remaining = [(a, t) for a, t in pairs if a not in already]
    print(f"[65/{arm}] {len(pairs)} total, {len(already)} done, {len(remaining)} remaining", flush=True)
    if not remaining:
        print(f"[65/{arm}] nothing to do", flush=True)
        return

    client = s49._make_client()
    manifest_path = d / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "arm": arm, "arm_id": ARMS[arm]["arm_id"], "variant": ARMS[arm]["variant"],
        "model": MODEL_NAME, "dim": EMBED_DIM, "total_tokens_billed": 0, "n_docs_embedded": 0,
        "cost_usd": 0.0, "price_per_million_tokens_usd": PRICE_PER_MILLION_TOKENS_USD,
        "projected_cost_usd": cleared["projected_cost_usd"], "batches_completed": 0,
        "started": now_iso(),
    }

    count_fn = embed_common.openai_token_counter(TOKENIZER_NAME)
    report = embed_common.token_report([t for _, t in remaining], count_fn, MAX_SEQ_LENGTH, TOKENIZER_NAME)
    embed_common.assert_no_truncation(report)
    print(f"[65/{arm}] truncation guard OK: n_docs={report['n_docs']} "
          f"frac_truncated={report['frac_truncated']:.6f}", flush=True)

    shard_idx = evo._next_shard_index(d)
    for start in range(0, len(remaining), SHARD_SIZE):
        chunk = remaining[start:start + SHARD_SIZE]
        ids, vecs = [], []
        for sub_batch in evo._token_budget_batches(chunk, count_fn):
            ids_b = [aid for aid, _ in sub_batch]
            texts_b = [tx for _, tx in sub_batch]
            v, tok = evo._embed_batch_with_retry(client, texts_b, MODEL_NAME)
            ids.extend(ids_b)
            vecs.extend(v)
            manifest["total_tokens_billed"] += int(tok)
            manifest["batches_completed"] += 1
            manifest["cost_usd"] = manifest["total_tokens_billed"] / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD
            if _ledger_cost(exclude=arm) + manifest["cost_usd"] > HARD_CAP_USD:
                c18.atomic_write_json(manifest, manifest_path)
                raise SystemExit(f"HARD CAP ${HARD_CAP_USD} exceeded -- stopping before the next batch")
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.shape[1] != EMBED_DIM:
            raise SystemExit(f"unexpected dim {arr.shape[1]} != {EMBED_DIM}")
        np.save(d / f"shard_{shard_idx:04d}_of_api.npy", arr)
        (d / f"shard_{shard_idx:04d}_of_api_ids.json").write_text(json.dumps(ids))
        manifest["n_docs_embedded"] += len(ids)
        manifest["last_update"] = now_iso()
        c18.atomic_write_json(manifest, manifest_path)
        print(f"[65/{arm}] shard {shard_idx} n={len(ids)} cost=${manifest['cost_usd']:.4f} "
              f"ledger=${_ledger_cost():.4f}", flush=True)
        shard_idx += 1

    manifest = embed_common.merge_report_into_manifest(manifest, report)
    manifest["completed"] = now_iso()
    c18.atomic_write_json(manifest, manifest_path)


# --------------------------------------------------------------------------- merge
def stage_merge(arm: str) -> None:
    d = arm_dir(arm)
    ids_ref = ref_ids()
    vecs = {}
    for idp in sorted(d.glob("shard_*_of_api_ids.json")):
        vp = idp.with_name(idp.name.replace("_ids.json", ".npy"))
        for a, row in zip(json.loads(idp.read_text()), np.load(vp)):
            vecs[int(a)] = row
    missing = [a for a in ids_ref if a not in vecs]
    if missing:
        raise SystemExit(f"{len(missing)} admissions not embedded, e.g. {missing[:5]}")
    merged = np.stack([vecs[a] for a in ids_ref]).astype(np.float32)
    if merged.shape != (len(ids_ref), EMBED_DIM) or not np.isfinite(merged).all():
        raise SystemExit(f"bad merged array {merged.shape}")
    np.save(d / "merged.npy", merged)
    (d / "merged_ids.json").write_text(json.dumps(ids_ref))
    info = {"arm": arm, "arm_id": ARMS[arm]["arm_id"], "shape": list(merged.shape),
            "dtype": str(merged.dtype), "sha256_merged_npy": sha256_file(d / "merged.npy"),
            "written": now_iso(), **env_record()}
    if arm == "B1b":
        # numbers only: how close is the fresh embedding of identical text to the stored one
        stored = np.load(STORED_3SMALL / "merged.npy")
        cos = np.einsum("ij,ij->i", merged, stored) / (
            np.linalg.norm(merged, axis=1) * np.linalg.norm(stored, axis=1))
        info["vs_stored_3small_embedding"] = {
            "cosine_min": float(cos.min()), "cosine_p1": float(np.percentile(cos, 1)),
            "cosine_median": float(np.median(cos)), "max_abs_elementwise_diff": float(np.abs(merged - stored).max()),
            "n_rows_bitwise_identical": int((merged == stored).all(axis=1).sum())}
    c18.atomic_write_json(info, d / "merge_info.json")
    print(f"[65/{arm}] merged {merged.shape} in reference row order", flush=True)
    if arm == "B1b":
        print(f"[65/{arm}] vs stored: {json.dumps(info['vs_stored_3small_embedding'])}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["count_tokens", "auth_check", "embed", "merge"])
    ap.add_argument("--arm", default=None, choices=list(ARMS))
    args = ap.parse_args()
    if args.stage == "auth_check":
        stage_auth_check()
        return
    arms = [args.arm] if args.arm else list(ARMS)
    for a in arms:
        {"count_tokens": stage_count_tokens, "embed": stage_embed, "merge": stage_merge}[args.stage](a)


if __name__ == "__main__":
    main()
