#!/usr/bin/env python3
"""Re-embed Task 6/7 ablation/template variant narratives via the OpenAI API
(design doc section 1.2(A), author-authorized D5): `text-embedding-3-large`,
3,072-dim, 8,191-token window -- the SAME model and the SAME narratives
(redacted/reformatted subsets of `full`, already transmitted to this API)
that produced the manuscript's published phenotypes. This is the PRIMARY
instrument for Tasks 6/7 (the open model is a replicate arm, handled by
`04_open_embedding.py` for Task 4 only in this pass).

Reuses the narrative files `06_domain_ablation.py`/`07_template_sensitivity.
py` already generated under `out/06_ablation/narratives/<variant>/` and
`out/07_template/narratives/<variant>/` -- 32,329 `.txt` files each, named
by `admission_ID`. Does NOT regenerate them.

Writes shards into the EXACT layout `06_domain_ablation.py::
stage_merge_embeddings` / `07_template_sensitivity.py::stage_merge_embeddings`
already expect (`shard_*_of_*.npy` + matching `_ids.json`), then calls that
task's own (unmodified) merge function to produce `merged.npy` +
`merged_ids.json` under `embeddings/<variant>/` -- so `stage_cluster` in
either module runs completely unchanged against these new vectors.

Before writing anything new, the PRE-EXISTING (truncated, e5-based, invalid
per AUDIT C1) `embeddings/` directory for a task is relabelled to
`embeddings_INVALID_truncated_e5/` (design doc section 4.2: "keep, relabel
... they are the negative control that demonstrates the window effect").
This is idempotent -- safe to re-run.

Cost-saving, honesty-preserving dedup: `out/06_ablation/narratives/full/`
and `out/07_template/narratives/full/` are byte-identical (both are the
untouched original narrative, confirmed by diff before this script was
written) -- `full` is embedded ONCE (under 06_ablation) and the resulting
merged vectors are copied verbatim into 07_template's `embeddings/full/`,
never re-sent to the API. This is not an approximation: it is the same
text, so it is (and must be) the same embedding call.

Batching / checkpointing / resumability (required by the task brief):
- Truncation guard (`embed_common.token_report` + `assert_no_truncation`,
  tiktoken `cl100k_base`, 8,191-token limit) runs over EVERY remaining
  document before any API call for that variant.
- Documents already covered by an existing, fully-written shard (its
  `_ids.json` sidecar present) are skipped on start -- an interrupted run
  never re-pays for completed work.
- Shards are written atomically (tmp + os.replace) in groups of
  `SHARD_SIZE` documents, each internally split into token-budget-aware
  API sub-batches with retry/backoff -- a crash mid-shard loses at most one
  shard's uncommitted sub-batches, not the whole run.
- Token counts billed (from the API response's `usage.total_tokens`) are
  accumulated per variant into `.openai_embed_manifest_<variant>.json`,
  updated after every shard, so cost is auditable incrementally, not only
  at the end.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, List

ROOT = Path("/data/tcmhwd/AI/AST_ai")
REVISION_ROOT = ROOT / "phenotype_revision"

MODEL_NAME = "text-embedding-3-large"
EMBED_DIM = 3072
MAX_SEQ_LENGTH = 8191
TOKENIZER_NAME = "cl100k_base"
PRICE_PER_MILLION_TOKENS_USD = 0.13

# Internal API call size limits (well under OpenAI's per-request caps of
# 2,048 inputs / ~300k tokens for this model) -- kept conservative so a
# single request never risks a 400 from exceeding either cap.
MAX_DOCS_PER_REQUEST = 100
MAX_TOKENS_PER_REQUEST = 250_000
SHARD_SIZE = 2000  # docs per on-disk shard file (crash-resilience granularity)

TASKS = {
    "06_ablation": REVISION_ROOT / "out" / "06_ablation",
    "07_template": REVISION_ROOT / "out" / "07_template",
}


def _atomic_write_json(obj: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, dest)


def relabel_invalid_e5(task: str) -> dict:
    """One-time-per-task, idempotent: move the pre-existing (truncated e5)
    `embeddings/` dir aside so it survives as evidence (design doc section
    4.2) and a fresh `embeddings/` can be built from OpenAI vectors.

    Guarded by a marker file rather than mere directory existence: this is
    called at the start of every `stage_embed_variant` invocation (once per
    variant, potentially many times across a resumed run), so "does
    `embeddings/` exist" is true again as soon as the FIRST variant writes
    its first OpenAI shard -- without the marker, the second variant's call
    would see that populated `embeddings/` dir and wrongly relabel the
    first variant's brand-new, valid work as "INVALID_truncated_e5"."""
    task_out = TASKS[task]
    old = task_out / "embeddings"
    invalid = task_out / "embeddings_INVALID_truncated_e5"
    marker = task_out / ".embeddings_is_openai_migrated"

    if marker.exists():
        return {"task": task, "action": "already_migrated"}

    marker.parent.mkdir(parents=True, exist_ok=True)
    if invalid.exists():
        marker.write_text("embeddings/ already relabelled to embeddings_INVALID_truncated_e5/ previously")
        return {"task": task, "action": "already_relabelled", "dest": str(invalid)}
    if not old.exists():
        marker.write_text("no pre-existing embeddings/ dir found -- nothing to relabel")
        return {"task": task, "action": "nothing_to_relabel"}

    shutil.move(str(old), str(invalid))
    marker.write_text(f"relabelled {old} -> {invalid}")
    return {"task": task, "action": "relabelled", "src": str(old), "dest": str(invalid)}


def _narratives_dir(task: str, variant: str) -> Path:
    return TASKS[task] / "narratives" / variant


def _embeddings_dir(task: str, variant: str) -> Path:
    d = TASKS[task] / "embeddings" / variant
    d.mkdir(parents=True, exist_ok=True)
    return d


def _list_docs(narratives_dir: Path) -> List[Path]:
    return sorted(narratives_dir.glob("*.txt"), key=lambda p: int(p.stem))


def _already_done_ids(emb_dir: Path) -> set:
    done = set()
    for idp in emb_dir.glob("shard_*_of_*_ids.json"):
        try:
            done.update(json.loads(idp.read_text()))
        except (json.JSONDecodeError, OSError):
            continue  # a partial/corrupt sidecar is treated as not-done; its .npy is orphaned but harmless
    return done


def _next_shard_index(emb_dir: Path) -> int:
    existing = list(emb_dir.glob("shard_*_of_api.npy"))
    if not existing:
        return 0
    indices = [int(p.stem.split("_")[1]) for p in existing]
    return max(indices) + 1


def _token_budget_batches(items: List[tuple], count_fn: Callable[[str], int]):
    """items: list of (admission_id, text). Yield sub-batches respecting
    MAX_DOCS_PER_REQUEST and MAX_TOKENS_PER_REQUEST."""
    batch, batch_tokens = [], 0
    for aid, text in items:
        n = count_fn(text)
        if batch and (len(batch) >= MAX_DOCS_PER_REQUEST or batch_tokens + n > MAX_TOKENS_PER_REQUEST):
            yield batch
            batch, batch_tokens = [], 0
        batch.append((aid, text))
        batch_tokens += n
    if batch:
        yield batch


def _embed_batch_with_retry(client, texts: List[str], model: str, max_retries: int = 6):
    import random as _random

    for attempt in range(max_retries):
        try:
            resp = client.embeddings.create(model=model, input=texts)
            vectors = [d.embedding for d in resp.data]
            total_tokens = resp.usage.total_tokens
            return vectors, total_tokens
        except Exception as exc:  # noqa: BLE001 -- broad on purpose: retry any transient API error
            if attempt == max_retries - 1:
                raise
            sleep_s = min(60, (2 ** attempt)) + _random.uniform(0, 1)
            print(f"  API error (attempt {attempt + 1}/{max_retries}): {exc!r} -- retrying in {sleep_s:.1f}s",
                  flush=True)
            time.sleep(sleep_s)


def stage_embed_variant(task: str, variant: str, client=None, dry_run: bool = False) -> dict:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import embed_common
    import numpy as np

    relabel_invalid_e5(task)

    narratives_dir = _narratives_dir(task, variant)
    emb_dir = _embeddings_dir(task, variant)
    files = _list_docs(narratives_dir)
    if not files:
        raise FileNotFoundError(f"no narratives found under {narratives_dir}")

    already_done = _already_done_ids(emb_dir)
    remaining = [(int(p.stem), p) for p in files if int(p.stem) not in already_done]
    print(f"[{task}/{variant}] {len(files)} total, {len(already_done)} already embedded, "
          f"{len(remaining)} remaining", flush=True)

    manifest_path = emb_dir / f".openai_embed_manifest_{variant}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "task": task, "variant": variant, "model": MODEL_NAME, "dim": EMBED_DIM,
        "total_tokens_billed": 0, "n_docs_embedded": 0, "cost_usd": 0.0,
        "price_per_million_tokens_usd": PRICE_PER_MILLION_TOKENS_USD,
        "n_docs_total": len(files), "batches_completed": 0,
    }

    if not remaining:
        print(f"[{task}/{variant}] nothing to do (already complete)", flush=True)
        return manifest

    count_fn = embed_common.openai_token_counter(TOKENIZER_NAME)

    # --- G0: truncation guard over every REMAINING document, before any API
    # call for this variant (hard precondition, applies to the OpenAI arm
    # exactly like the open-model arm). ---
    texts_for_report = [p.read_text(encoding="utf-8", errors="replace") for _, p in remaining]
    report = embed_common.token_report(texts_for_report, count_fn, MAX_SEQ_LENGTH, TOKENIZER_NAME)
    embed_common.assert_no_truncation(report)
    print(f"[{task}/{variant}] truncation guard OK: n_docs={report['n_docs']} "
          f"median={report['n_tokens_median']:.0f} max={report['n_tokens_max']} "
          f"frac_truncated={report['frac_truncated']:.6f}", flush=True)

    if dry_run:
        manifest["dry_run_token_report"] = report
        _atomic_write_json(manifest, manifest_path)
        return manifest

    items = [(aid, text) for (aid, _), text in zip(remaining, texts_for_report)]
    shard_idx = _next_shard_index(emb_dir)

    for shard_start in range(0, len(items), SHARD_SIZE):
        shard_items = items[shard_start:shard_start + SHARD_SIZE]
        shard_ids, shard_vecs = [], []
        for sub_batch in _token_budget_batches(shard_items, count_fn):
            ids_b = [aid for aid, _ in sub_batch]
            texts_b = [t for _, t in sub_batch]
            vectors, total_tokens = _embed_batch_with_retry(client, texts_b, MODEL_NAME)
            shard_ids.extend(ids_b)
            shard_vecs.extend(vectors)
            manifest["total_tokens_billed"] += total_tokens
            manifest["cost_usd"] = manifest["total_tokens_billed"] / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD
            manifest["n_docs_embedded"] += len(ids_b)
            manifest["batches_completed"] += 1
            print(f"[{task}/{variant}] batch of {len(ids_b)} docs, {total_tokens} tokens "
                  f"-- running total {manifest['n_docs_embedded']}/{len(files)} docs, "
                  f"${manifest['cost_usd']:.4f}", flush=True)
            _atomic_write_json(manifest, manifest_path)  # incremental, resumable cost record

        vec_arr = np.asarray(shard_vecs, dtype=np.float32)
        vec_path = emb_dir / f"shard_{shard_idx:04d}_of_api.npy"
        id_path = emb_dir / f"shard_{shard_idx:04d}_of_api_ids.json"
        tmp_vec = vec_path.with_name(f".tmp_{os.getpid()}_{vec_path.name}")
        np.save(tmp_vec, vec_arr)
        os.replace(tmp_vec, vec_path)
        tmp_id = id_path.with_name(f".tmp_{os.getpid()}_{id_path.name}")
        tmp_id.write_text(json.dumps(shard_ids))
        os.replace(tmp_id, id_path)
        print(f"[{task}/{variant}] wrote shard {shard_idx} ({len(shard_ids)} docs) -> {vec_path}", flush=True)
        shard_idx += 1

    manifest = embed_common.merge_report_into_manifest(manifest, report)
    _atomic_write_json(manifest, manifest_path)
    return manifest


def merge_variant(task: str, variant: str) -> dict:
    """Call the task's OWN (unmodified) stage_merge_embeddings, so the
    merged.npy/merged_ids.json this produces are byte-for-byte what that
    module would have written itself."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib

    module_name = {"06_ablation": "06_domain_ablation", "07_template": "07_template_sensitivity"}[task]
    mod = importlib.import_module(module_name)
    return mod.stage_merge_embeddings(variant)


def dedup_full_across_tasks() -> dict:
    """`out/06_ablation/narratives/full/` and `out/07_template/narratives/
    full/` are byte-identical (both are the untouched original narrative).
    Embed `full` once under 06_ablation and copy the merged result into
    07_template's `embeddings/full/` -- never re-sent to the API."""
    src_dir = TASKS["06_ablation"] / "embeddings" / "full"
    dst_dir = TASKS["07_template"] / "embeddings" / "full"
    src_merged, src_ids = src_dir / "merged.npy", src_dir / "merged_ids.json"
    if not src_merged.exists() or not src_ids.exists():
        raise FileNotFoundError(
            f"{src_merged} missing -- embed+merge 06_ablation/full before deduping into 07_template"
        )
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_merged, dst_dir / "merged.npy")
    shutil.copyfile(src_ids, dst_dir / "merged_ids.json")
    dedup_manifest = {
        "task": "07_template", "variant": "full", "action": "deduped_from_06_ablation",
        "src": str(src_dir), "note": (
            "out/06_ablation/narratives/full and out/07_template/narratives/full are "
            "byte-identical (both the untouched original narrative) -- embedded once "
            "under 06_ablation, copied here verbatim, never re-sent to the API."
        ),
    }
    _atomic_write_json(dedup_manifest, dst_dir / ".openai_embed_manifest_full.json")
    return dedup_manifest


def cost_summary() -> dict:
    rows = []
    total_tokens, total_cost = 0, 0.0
    for task, task_out in TASKS.items():
        emb_root = task_out / "embeddings"
        if not emb_root.exists():
            continue
        for variant_dir in sorted(emb_root.glob("*")):
            mpath = variant_dir / f".openai_embed_manifest_{variant_dir.name}.json"
            if not mpath.exists():
                continue
            m = json.loads(mpath.read_text())
            rows.append({
                "task": task, "variant": variant_dir.name,
                "n_docs_embedded": m.get("n_docs_embedded", "n/a (deduped)"),
                "total_tokens_billed": m.get("total_tokens_billed", 0),
                "cost_usd": m.get("cost_usd", 0.0),
            })
            total_tokens += m.get("total_tokens_billed", 0)
            total_cost += m.get("cost_usd", 0.0)
    return {"rows": rows, "total_tokens_billed": total_tokens, "total_cost_usd": total_cost}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["embed", "merge", "dedup_full", "cost_summary"])
    ap.add_argument("--task", choices=list(TASKS))
    ap.add_argument("--variant")
    ap.add_argument("--dry-run", action="store_true",
                     help="run the truncation guard and print token stats; make no API calls")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.stage == "cost_summary":
        print(json.dumps(cost_summary(), indent=2))
        return
    if args.stage == "dedup_full":
        print(json.dumps(dedup_full_across_tasks(), indent=2))
        return
    if not args.task or not args.variant:
        raise SystemExit("--task and --variant are required for --stage embed/merge")
    if args.stage == "embed":
        client = None
        if not args.dry_run:
            from openai import OpenAI
            # Key acquisition order: environment variable -> ~/.secrets/openai_api_key
            # (author-specified, permissions 600). In this environment
            # OPENAI_API_KEY is not exported, and the original research also
            # supplied it directly at run time. The key value is never left
            # in logs/commits/manifests, and is only read from the path
            # below to construct the client.
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                key_path = Path.home() / ".secrets" / "openai_api_key"
                if not key_path.is_file():
                    raise SystemExit(
                        f"Neither the OPENAI_API_KEY environment variable nor {key_path} is set -- "
                        "a key must be provided to run re-embedding."
                    )
                api_key = key_path.read_text().strip()
            client = OpenAI(api_key=api_key)
        manifest = stage_embed_variant(args.task, args.variant, client=client, dry_run=args.dry_run)
        print(json.dumps(manifest, indent=2))
    elif args.stage == "merge":
        print(json.dumps(merge_variant(args.task, args.variant), indent=2))


if __name__ == "__main__":
    main()
