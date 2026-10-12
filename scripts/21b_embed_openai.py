#!/usr/bin/env python3
"""Task 21, Step 2 -- embed every generated narrative variant via the
OpenAI API (`text-embedding-3-large`, matches the model that produced the
published phenotypes -- see `embed_variants_openai.py`'s docstring for why
this, not an open model, is the primary instrument here).

Reuses `embed_variants_openai.py`'s corpus-agnostic batching/retry/shard
helpers (`_already_done_ids`, `_next_shard_index`, `_token_budget_batches`,
`_embed_batch_with_retry`) read-only, exactly as `18c_embed_openai_small.py`
does -- no reimplementation of API-call discipline.

HARD CAP: $80.00 TOTAL ACROSS EVERY VARIANT (task brief), tracked in a
single shared ledger `out/21_published_variants/.cost_ledger.json` (one
entry per variant, updated after every shard) so the check is against the
REAL running total, not a per-variant budget. Refuses to start a variant
whose own `21a` generation manifest is not `status: "ok"` (never embeds a
variant that failed its own truncation/budget gate). Checked before every
API call AND after every shard -- if a shard's actual billed tokens push
the running total over the cap, this raises immediately and does not
attempt the next shard.

Stages: embed (one variant) -> merge (one variant) -> cost_summary (all
variants). `full` MUST be embedded+merged first (every other variant's
comparison-to-unmodified-re-embedding, computed in `21c_cluster_frozen.py`,
depends on it) -- `main()` enforces this ordering when no `--variant` is
given.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c21 = importlib.import_module("21_common")
evo = importlib.import_module("embed_variants_openai")
gen = importlib.import_module("21a_generate_variants")
import embed_common

MODEL_NAME = "text-embedding-3-large"
EMBED_DIM = 3072
MAX_SEQ_LENGTH = 8191
TOKENIZER_NAME = "cl100k_base"
PRICE_PER_MILLION_TOKENS_USD = 0.13
# AUDIT_4 C1 addendum (2026-09-06) -- see the matching comment in
# 21a_generate_variants.py. Raised from 80 to 86 to admit the two decisive-
# control variants (fixed_permutation, shuffle_order_seed2; ~$4.5 each) on
# top of the $73.30 already actually billed for the original 18.
HARD_CAP_USD = 86.0
SHARD_SIZE = 2000

EMB_ROOT = c21.OUT / "embeddings"
LEDGER_PATH = c21.OUT / ".cost_ledger.json"


def _emb_dir(variant: str) -> Path:
    d = EMB_ROOT / variant
    d.mkdir(parents=True, exist_ok=True)
    return d


def _read_ledger() -> dict:
    if LEDGER_PATH.exists():
        return json.loads(LEDGER_PATH.read_text())
    return {"variants": {}, "total_cost_usd": 0.0, "hard_cap_usd": HARD_CAP_USD}


def _write_ledger(ledger: dict) -> None:
    ledger["total_cost_usd"] = sum(v.get("cost_usd", 0.0) for v in ledger["variants"].values())
    c21.atomic_write_json(ledger, LEDGER_PATH)


def _require_generation_ok(variant: str) -> None:
    mpath = c21.OUT / "generation" / variant / "manifest.json"
    if not mpath.exists():
        raise RuntimeError(f"{mpath} missing -- run 21a_generate_variants.py --variant {variant} first")
    m = json.loads(mpath.read_text())
    if m.get("status") != "ok":
        raise RuntimeError(
            f"21a generation manifest for {variant!r} has status={m.get('status')!r}, not 'ok' -- "
            f"refusing to embed (this variant either exceeded the token/truncation gate or the "
            f"$80 generation-time cost PROJECTION; embedding must not proceed)."
        )


def _list_narrative_files(variant: str):
    d = c21.NARRATIVES_DIR / variant
    files = sorted(d.glob("*.txt"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"no narratives under {d}")
    return files


def stage_embed(variant: str, client, limit: int | None = None) -> dict:
    import numpy as np

    _require_generation_ok(variant)
    files = _list_narrative_files(variant)
    if limit is not None:
        files = files[:limit]

    emb_dir = _emb_dir(variant)
    already_done = evo._already_done_ids(emb_dir)
    remaining = [(int(p.stem), p) for p in files if int(p.stem) not in already_done]
    print(f"[21b/{variant}] {len(files)} total, {len(already_done)} already embedded, "
          f"{len(remaining)} remaining", flush=True)

    manifest_path = emb_dir / f".openai_embed_manifest_{variant}.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "variant": variant, "model": MODEL_NAME, "dim": EMBED_DIM,
        "total_tokens_billed": 0, "n_docs_embedded": 0, "cost_usd": 0.0,
        "price_per_million_tokens_usd": PRICE_PER_MILLION_TOKENS_USD,
        "n_docs_total": len(files), "batches_completed": 0,
    }

    if not remaining:
        print(f"[21b/{variant}] nothing to do (already complete)", flush=True)
        return manifest

    count_fn = embed_common.openai_token_counter(TOKENIZER_NAME)

    texts_for_report = [p.read_text(encoding="utf-8", errors="replace") for _, p in remaining]
    report = embed_common.token_report(texts_for_report, count_fn, MAX_SEQ_LENGTH, TOKENIZER_NAME)
    embed_common.assert_no_truncation(report)
    print(f"[21b/{variant}] truncation guard OK: n_docs={report['n_docs']} "
          f"median={report['n_tokens_median']:.0f} max={report['n_tokens_max']} "
          f"frac_truncated={report['frac_truncated']:.6f}", flush=True)

    items = [(aid, text) for (aid, _), text in zip(remaining, texts_for_report)]
    shard_idx = evo._next_shard_index(emb_dir)

    ledger = _read_ledger()
    ledger["variants"].setdefault(variant, {"cost_usd": 0.0})

    for shard_start in range(0, len(items), SHARD_SIZE):
        shard_items = items[shard_start:shard_start + SHARD_SIZE]
        shard_ids, shard_vecs = [], []
        for sub_batch in evo._token_budget_batches(shard_items, count_fn):
            ids_b = [aid for aid, _ in sub_batch]
            texts_b = [t for _, t in sub_batch]
            vectors, total_tokens = evo._embed_batch_with_retry(client, texts_b, MODEL_NAME)
            shard_ids.extend(ids_b)
            shard_vecs.extend(vectors)
            manifest["total_tokens_billed"] += total_tokens
            manifest["cost_usd"] = manifest["total_tokens_billed"] / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD
            manifest["n_docs_embedded"] += len(ids_b)
            manifest["batches_completed"] += 1

            ledger["variants"][variant]["cost_usd"] = manifest["cost_usd"]
            running_total = sum(v.get("cost_usd", 0.0) for v in ledger["variants"].values())
            print(f"[21b/{variant}] batch of {len(ids_b)} docs, {total_tokens} tokens -- "
                  f"running total {manifest['n_docs_embedded']}/{len(files)} docs, "
                  f"variant=${manifest['cost_usd']:.4f} ALL_VARIANTS=${running_total:.4f}", flush=True)
            c21.atomic_write_json(manifest, manifest_path)
            _write_ledger(ledger)

            if running_total > HARD_CAP_USD:
                raise RuntimeError(
                    f"[21b] TOTAL across all variants ${running_total:.4f} exceeded "
                    f"HARD_CAP_USD={HARD_CAP_USD} after a batch in variant={variant!r} -- "
                    "STOPPING immediately. Already-written shards are kept (resumable); "
                    "no further batches/variants will be embedded."
                )

        vec_arr = np.asarray(shard_vecs, dtype=np.float32)
        vec_path = emb_dir / f"shard_{shard_idx:04d}_of_api.npy"
        id_path = emb_dir / f"shard_{shard_idx:04d}_of_api_ids.json"
        tmp_vec = vec_path.with_name(f".tmp_{os.getpid()}_{vec_path.name}")
        np.save(tmp_vec, vec_arr)
        os.replace(tmp_vec, vec_path)
        tmp_id = id_path.with_name(f".tmp_{os.getpid()}_{id_path.name}")
        tmp_id.write_text(json.dumps(shard_ids))
        os.replace(tmp_id, id_path)
        print(f"[21b/{variant}] wrote shard {shard_idx} ({len(shard_ids)} docs) -> {vec_path}", flush=True)
        shard_idx += 1

    manifest = embed_common.merge_report_into_manifest(manifest, report)
    c21.atomic_write_json(manifest, manifest_path)
    return manifest


def stage_merge(variant: str) -> dict:
    import numpy as np

    emb_dir = _emb_dir(variant)
    shard_files = sorted(emb_dir.glob("shard_*_of_api.npy"))
    if not shard_files:
        raise FileNotFoundError(f"no shards under {emb_dir} -- run --stage embed first")

    all_vecs, all_ids = [], []
    for vp in shard_files:
        idp = vp.with_name(vp.stem + "_ids.json")
        ids = json.loads(idp.read_text())
        vecs = np.load(vp)
        if len(ids) != vecs.shape[0]:
            raise RuntimeError(f"{vp}: {vecs.shape[0]} vectors but {len(ids)} ids")
        all_vecs.append(vecs)
        all_ids.extend(ids)

    if len(set(all_ids)) != len(all_ids):
        raise RuntimeError("duplicate admission_ID across shards")

    merged_vecs = np.concatenate(all_vecs, axis=0)
    tmp_vec = emb_dir / f".tmp_{os.getpid()}_merged.npy"
    np.save(tmp_vec, merged_vecs)
    os.replace(tmp_vec, emb_dir / "merged.npy")
    tmp_id = emb_dir / f".tmp_{os.getpid()}_merged_ids.json"
    tmp_id.write_text(json.dumps(all_ids))
    os.replace(tmp_id, emb_dir / "merged_ids.json")

    result = {"variant": variant, "n_docs": len(all_ids), "dim": int(merged_vecs.shape[1]),
              "n_shards": len(shard_files)}
    print(json.dumps(result, indent=2), flush=True)
    return result


def cost_summary() -> dict:
    ledger = _read_ledger()
    return ledger


def _build_client():
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        key_path = Path.home() / ".secrets" / "openai_api_key"
        if not key_path.is_file():
            raise SystemExit(f"no OPENAI_API_KEY and {key_path} not found")
        api_key = key_path.read_text().strip()
    return OpenAI(api_key=api_key)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["embed", "merge", "embed_all", "cost_summary"])
    ap.add_argument("--variant", choices=gen.ORDERED_VARIANTS)
    ap.add_argument("--limit", type=int, default=None)
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    if args.stage == "cost_summary":
        print(json.dumps(cost_summary(), indent=2))
        return
    if args.stage == "merge":
        if not args.variant:
            raise SystemExit("--variant required for --stage merge")
        print(json.dumps(stage_merge(args.variant), indent=2))
        return
    if args.stage == "embed":
        if not args.variant:
            raise SystemExit("--variant required for --stage embed")
        client = _build_client()
        manifest = stage_embed(args.variant, client, limit=args.limit)
        print(json.dumps(manifest, indent=2))
        return
    if args.stage == "embed_all":
        # `full` first (every other variant's re-embedding comparison depends
        # on it), then the rest in the same priority order as generation.
        client = _build_client()
        for variant in gen.ORDERED_VARIANTS:
            t0 = time.time()
            stage_embed(variant, client, limit=args.limit)
            stage_merge(variant)
            print(f"[21b] {variant} embed+merge done in {time.time() - t0:.1f}s", flush=True)
        print(json.dumps(cost_summary(), indent=2), flush=True)


if __name__ == "__main__":
    main()
