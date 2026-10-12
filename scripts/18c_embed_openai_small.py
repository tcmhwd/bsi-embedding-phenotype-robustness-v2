#!/usr/bin/env python3
"""Task 18, Step 1 -- embed the `reports_pilot_pre/` corpus with OpenAI
`text-embedding-3-small` (the third of the three S6 encoders).

Reuses `embed_variants_openai.py`'s generic batching/retry/shard-index
helpers (`_token_budget_batches`, `_embed_batch_with_retry`,
`_already_done_ids`, `_next_shard_index` -- all corpus-agnostic, imported
read-only) so the API-call discipline (token-budget sub-batches, retry with
backoff, atomic shard writes, incremental cost manifest) matches every
other OpenAI-embed stage in this repo. Text loading is NEW
(`18_common.load_texts`, `reports_pilot_pre`, not `reports_all`).

Hard cap: $2.00 total. `--stage count_tokens` MUST be run (and must report
proceed=true) before `--stage embed` will do anything -- mirrors
`12_encoder_sensitivity.py`'s `_require_cost_cleared` guard.

Stages: count_tokens (login-node safe) -> embed (sbatch; API-bound, no
thread pinning needed) -> merge.
"""
import json
import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import importlib
c18 = importlib.import_module("18_common")
evo = importlib.import_module("embed_variants_openai")
import embed_common

MODEL_NAME = "text-embedding-3-small"
EMBED_DIM = 1536
MAX_SEQ_LENGTH = 8191
TOKENIZER_NAME = "cl100k_base"
PRICE_PER_MILLION_TOKENS_USD = 0.02
HARD_CAP_USD = 2.00
SHARD_SIZE = 2000

OUT_DIR = c18.OUT / "embeddings" / "openai-3small"


def _require_step0() -> None:
    step0 = c18.OUT / "step0_validate_corpus.json"
    if not step0.exists() or not json.loads(step0.read_text()).get("proceed", False):
        raise SystemExit(f"{step0} missing or proceed=false -- Step 0 corpus validation must pass first")


def stage_count_tokens() -> dict:
    import tiktoken

    _require_step0()
    frame = c18.load_texts()
    texts = frame["text"].tolist()

    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    count_fn = lambda t: len(enc.encode(t))  # noqa: E731
    report = embed_common.token_report(texts, count_fn, MAX_SEQ_LENGTH, TOKENIZER_NAME)

    total_tokens = int(sum(len(enc.encode(t)) for t in texts))
    projected_cost_usd = round(total_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD, 4)
    proceed = bool(report["frac_truncated"] == 0.0 and projected_cost_usd <= HARD_CAP_USD)

    result = {
        "model": MODEL_NAME, "price_per_million_tokens_usd": PRICE_PER_MILLION_TOKENS_USD,
        "hard_cap_usd": HARD_CAP_USD, "n_docs": len(texts), "total_tokens": total_tokens,
        "projected_cost_usd": projected_cost_usd, "n_truncated": report["n_truncated"],
        "frac_truncated": report["frac_truncated"], "token_report": report, "proceed": proceed,
        "corpus_dir": str(c18.REPORTS_DIR),
    }
    c18.atomic_write_json(result, OUT_DIR / "token_report_3small.json")
    print(json.dumps({k: v for k, v in result.items() if k != "token_report"}, indent=2), flush=True)
    if not proceed:
        reason = "truncation > 0" if report["frac_truncated"] > 0 else \
            f"projected cost ${projected_cost_usd} > cap ${HARD_CAP_USD}"
        print(f"STOP: {reason} -- must not proceed to embed.", flush=True)
    return result


def _require_cost_cleared() -> dict:
    path = OUT_DIR / "token_report_3small.json"
    if not path.exists():
        raise RuntimeError(f"{path} missing -- run --stage count_tokens first")
    report = json.loads(path.read_text())
    if not report.get("proceed", False):
        raise RuntimeError(f"count_tokens verdict is proceed=False -- refusing to embed")
    return report


def stage_embed(client, limit: int = None) -> dict:
    import numpy as np

    _require_step0()
    cost_report = _require_cost_cleared()

    frame = c18.load_texts()
    if limit is not None:
        frame = frame.head(limit)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    already_done = evo._already_done_ids(OUT_DIR)
    all_pairs = [(int(aid), text) for aid, text in zip(frame["admission_ID"], frame["text"])]
    remaining = [(aid, text) for aid, text in all_pairs if aid not in already_done]
    print(f"[18c] {len(all_pairs)} total, {len(already_done)} already embedded, "
          f"{len(remaining)} remaining", flush=True)

    manifest_path = OUT_DIR / ".openai_embed_manifest_3small.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "model": MODEL_NAME, "dim": EMBED_DIM, "total_tokens_billed": 0,
        "n_docs_embedded": 0, "cost_usd": 0.0,
        "price_per_million_tokens_usd": PRICE_PER_MILLION_TOKENS_USD,
        "n_docs_total": len(all_pairs), "batches_completed": 0,
        "count_tokens_projection": {k: cost_report[k] for k in
                                     ("total_tokens", "projected_cost_usd", "hard_cap_usd")},
    }

    if not remaining:
        print("[18c] nothing to do (already complete)", flush=True)
        return manifest

    count_fn = embed_common.openai_token_counter(TOKENIZER_NAME)
    report = embed_common.token_report(
        [t for _, t in remaining], count_fn, MAX_SEQ_LENGTH, TOKENIZER_NAME,
    )
    embed_common.assert_no_truncation(report)
    print(f"[18c] truncation guard OK: n_docs={report['n_docs']} "
          f"frac_truncated={report['frac_truncated']:.6f}", flush=True)

    shard_idx = evo._next_shard_index(OUT_DIR)
    for shard_start in range(0, len(remaining), SHARD_SIZE):
        shard_items = remaining[shard_start:shard_start + SHARD_SIZE]
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
            print(f"[18c] batch of {len(ids_b)} docs, {total_tokens} tokens -- running total "
                  f"{manifest['n_docs_embedded']}/{len(all_pairs)} docs, ${manifest['cost_usd']:.4f}",
                  flush=True)
            c18.atomic_write_json(manifest, manifest_path)
            if manifest["cost_usd"] > HARD_CAP_USD * 1.10:
                raise RuntimeError(
                    f"cost_usd={manifest['cost_usd']:.4f} exceeded 110% of the ${HARD_CAP_USD} "
                    "hard cap mid-run -- stopping immediately."
                )

        vec_arr = np.asarray(shard_vecs, dtype=np.float32)
        vec_path = OUT_DIR / f"shard_{shard_idx:04d}_of_api.npy"
        id_path = OUT_DIR / f"shard_{shard_idx:04d}_of_api_ids.json"
        tmp_vec = vec_path.with_name(f".tmp_{os.getpid()}_{vec_path.name}")
        np.save(tmp_vec, vec_arr)
        os.replace(tmp_vec, vec_path)
        tmp_id = id_path.with_name(f".tmp_{os.getpid()}_{id_path.name}")
        tmp_id.write_text(json.dumps(shard_ids))
        os.replace(tmp_id, id_path)
        print(f"[18c] wrote shard {shard_idx} ({len(shard_ids)} docs) -> {vec_path}", flush=True)
        shard_idx += 1

    manifest = embed_common.merge_report_into_manifest(manifest, report)
    c18.atomic_write_json(manifest, manifest_path)
    return manifest


def stage_merge() -> dict:
    import numpy as np

    shard_files = sorted(OUT_DIR.glob("shard_*_of_api.npy"))
    if not shard_files:
        raise FileNotFoundError(f"no shards under {OUT_DIR} -- run --stage embed first")

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
    tmp_vec = OUT_DIR / f".tmp_{os.getpid()}_merged.npy"
    np.save(tmp_vec, merged_vecs)
    os.replace(tmp_vec, OUT_DIR / "merged.npy")
    tmp_id = OUT_DIR / f".tmp_{os.getpid()}_merged_ids.json"
    tmp_id.write_text(json.dumps(all_ids))
    os.replace(tmp_id, OUT_DIR / "merged_ids.json")

    result = {"n_docs": len(all_ids), "dim": int(merged_vecs.shape[1]), "n_shards": len(shard_files)}
    print(json.dumps(result, indent=2), flush=True)
    return result


def parse_args(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=["count_tokens", "embed", "merge"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "count_tokens":
        stage_count_tokens()
    elif args.stage == "embed":
        client = None
        if not args.dry_run:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                key_path = Path.home() / ".secrets" / "openai_api_key"
                if not key_path.is_file():
                    raise SystemExit(f"no OPENAI_API_KEY and {key_path} not found")
                api_key = key_path.read_text().strip()
            client = OpenAI(api_key=api_key)
        if args.dry_run:
            _require_cost_cleared()
            print("dry-run: cost check cleared, no API calls made.", flush=True)
        else:
            manifest = stage_embed(client, limit=args.limit)
            print(json.dumps(manifest, indent=2), flush=True)
    elif args.stage == "merge":
        stage_merge()


if __name__ == "__main__":
    main()
