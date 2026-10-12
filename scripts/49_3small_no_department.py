#!/usr/bin/env python3
"""Task 49 -- the missing third dominant-field ablation: `text-embedding-3-small`
with its dominant field (treatment department) removed.

WHY THIS EXISTS
---------------
The manuscript's central mechanism is stated for every non-degenerate partition:
each is dominated by one rendered field, and removing that field collapses that
encoder's partition. The intervention had been run for only two of the three
non-degenerate Korean encoders:

    3-large  dominant = index organism        -> no_organism   (21 seeds)  AB-004
    Qwen3    dominant = first antibiotic      -> no_therapy    (11 seeds)  AB-014
    3-small  dominant = treatment department  -> NOT RUN

Internal review of the Phase-2B draft flagged the same gap:
the paper generalises over three encoders from an intervention on two. This
script closes it.

WHAT IT DOES
------------
Stage `count_tokens`  cost gate over the ALREADY-GENERATED `no_department`
                      narratives (out/21_published_variants/narratives/), the
                      same corpus 3-large's ablation used. No new text is
                      rendered; only the encoder changes.
Stage `embed`         embeds those narratives with `text-embedding-3-small`,
                      reusing `embed_variants_openai.py`'s batching/retry/shard
                      helpers read-only, exactly as `18c_embed_openai_small.py`
                      does. Resumable by shard.
Stage `merge`         merges shards to merged.npy / merged_ids.json in the
                      stored-partition row order.
Stage `cluster`       11 UMAP+HDBSCAN fits (seeds 42, 1-10) in the FROZEN
                      environment, matching the seed count of the other two
                      encoders' arms, and scores each against 3-small's own
                      whole-corpus reference partition.

WHY 3-small AND NOT AN OPEN ENCODER: the department-dominated partition IS
3-small's (FD-002: AMI 0.700, margin 0.494, 50/50 fits). No substitute exists.

REFERENCE PARTITION
-------------------
`out/18_encoder_dec/recluster/openai-3small/labels.npz` -- 3-small's own
whole-corpus partition under the configuration of record (k=22, noise 277),
the same object 3-large's arm is scored against for its own encoder. Agreement
uses `both_nonnoise`, as everywhere else.

CONTROL ARM
-----------
`no_organism` is the matched control for 3-small: organism is 3-small's
NON-dominant field (its department AMI is 0.700; organism is far lower), so the
prediction is that `no_organism` leaves 3-small near its unmodified band while
`no_department` collapses it -- the mirror of what 3-large shows. Both arms are
embedded and clustered here so the contrast is within-encoder.

COST
----
`no_department` is 34,358,517 projected tokens and `no_organism` 34,2xx,xxx
(out/21_published_variants/generation_summary.csv). At 3-small's $0.02 per
million that is about $0.69 each, ~$1.40 for both. Hard cap below is $4.00,
checked before the first call and after every shard. The same corpus has
already been sent to this provider with this model
(out/18_encoder_dec/embeddings/openai-3small/), so no new category of data
leaves the institution.

Runs: count_tokens/merge/cluster are light; `embed` is API-bound. `cluster`
MUST use the frozen interpreter.
"""
from __future__ import annotations

import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import argparse
import importlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c18 = importlib.import_module("18_common")
c21 = importlib.import_module("21_common")
evo = importlib.import_module("embed_variants_openai")
import embed_common

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
OUT = REV / "out/49_3small_ablation"

MODEL_NAME = "text-embedding-3-small"
EMBED_DIM = 1536
MAX_SEQ_LENGTH = 8191
TOKENIZER_NAME = "cl100k_base"
PRICE_PER_MILLION_TOKENS_USD = 0.02
HARD_CAP_USD = 4.00
SHARD_SIZE = 2000

# The dominant-field arm and its matched control, in that order.
VARIANTS = ["no_department", "no_organism"]
SEEDS = [42] + list(range(1, 11))

NARRATIVES = REV / "out/21_published_variants/narratives"
REFERENCE = REV / "out/18_encoder_dec/recluster/openai-3small/labels.npz"


def variant_dir(variant: str) -> Path:
    return OUT / "embeddings" / variant


def load_variant_texts(variant: str):
    """(admission_ID, text) for one already-generated variant, in the stored
    partition's row order. Never re-renders: 21a produced these files and this
    task changes only the encoder."""
    src = NARRATIVES / variant
    if not src.exists():
        raise SystemExit(f"{src} missing -- 21a must have generated this variant")
    ref_ids = np.load(REFERENCE, allow_pickle=True)["admission_ID"]
    pairs = []
    for aid in ref_ids:
        # 21a writes one file per admission named "<admission_ID>.txt".
        p = src / f"{int(aid)}.txt"
        if not p.exists():
            raise SystemExit(f"no narrative for admission {aid} at {p}")
        pairs.append((int(aid), p.read_text()))
    return pairs


def stage_count_tokens(variant: str) -> dict:
    import tiktoken

    pairs = load_variant_texts(variant)
    texts = [t for _, t in pairs]
    enc = tiktoken.get_encoding(TOKENIZER_NAME)
    report = embed_common.token_report(texts, lambda t: len(enc.encode(t)),
                                       MAX_SEQ_LENGTH, TOKENIZER_NAME)
    total = int(sum(len(enc.encode(t)) for t in texts))
    cost = round(total / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD, 4)
    proceed = bool(report["frac_truncated"] == 0.0 and cost <= HARD_CAP_USD)
    result = {
        "variant": variant, "model": MODEL_NAME, "n_docs": len(texts),
        "total_tokens": total, "projected_cost_usd": cost,
        "price_per_million_tokens_usd": PRICE_PER_MILLION_TOKENS_USD,
        "hard_cap_usd": HARD_CAP_USD, "n_truncated": report["n_truncated"],
        "frac_truncated": report["frac_truncated"], "token_report": report,
        "proceed": proceed, "narratives_dir": str(NARRATIVES / variant),
    }
    variant_dir(variant).mkdir(parents=True, exist_ok=True)
    c18.atomic_write_json(result, variant_dir(variant) / "token_report.json")
    print(json.dumps({k: v for k, v in result.items() if k != "token_report"}, indent=2), flush=True)
    if not proceed:
        why = "truncation > 0" if report["frac_truncated"] > 0 else f"cost ${cost} > cap ${HARD_CAP_USD}"
        print(f"STOP: {why} -- must not proceed to embed.", flush=True)
    return result


def _require_cost_cleared(variant: str) -> dict:
    path = variant_dir(variant) / "token_report.json"
    if not path.exists():
        raise SystemExit(f"{path} missing -- run --stage count_tokens --variant {variant} first")
    rep = json.loads(path.read_text())
    if not rep.get("proceed", False):
        raise SystemExit("count_tokens verdict is proceed=False -- refusing to embed")
    return rep


def _running_total() -> float:
    total = 0.0
    for v in VARIANTS:
        m = variant_dir(v) / "manifest.json"
        if m.exists():
            total += json.loads(m.read_text()).get("cost_usd", 0.0)
    return total


def _make_client():
    """Same key-acquisition order as every other OpenAI stage in this repo
    (`embed_variants_openai.py`): environment first, then the author-designated
    `~/.secrets/openai_api_key` (mode 600). The key is never logged, never
    written to a manifest and never committed."""
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        key_path = Path.home() / ".secrets" / "openai_api_key"
        if not key_path.is_file():
            raise SystemExit(
                f"neither OPENAI_API_KEY nor {key_path} is available -- "
                "cannot embed without a key")
        api_key = key_path.read_text().strip()
    return OpenAI(api_key=api_key)


def stage_embed(variant: str) -> None:
    _require_cost_cleared(variant)
    d = variant_dir(variant)
    d.mkdir(parents=True, exist_ok=True)

    pairs = load_variant_texts(variant)
    already = evo._already_done_ids(d)
    remaining = [(aid, t) for aid, t in pairs if aid not in already]
    print(f"[49/{variant}] {len(pairs)} total, {len(already)} done, {len(remaining)} remaining",
          flush=True)
    if not remaining:
        print(f"[49/{variant}] nothing to do", flush=True)
        return

    client = _make_client()
    manifest_path = d / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        "variant": variant, "model": MODEL_NAME, "dim": EMBED_DIM,
        "total_tokens_billed": 0, "n_docs_embedded": 0, "cost_usd": 0.0,
        "price_per_million_tokens_usd": PRICE_PER_MILLION_TOKENS_USD,
    }

    # Truncation guard on exactly the documents about to be sent, mirroring
    # 18c: the count_tokens gate ran over the whole variant, this re-checks the
    # remaining subset so a resumed run cannot skip the guard.
    count_fn = embed_common.openai_token_counter(TOKENIZER_NAME)
    report = embed_common.token_report([t for _, t in remaining], count_fn,
                                       MAX_SEQ_LENGTH, TOKENIZER_NAME)
    embed_common.assert_no_truncation(report)
    print(f"[49/{variant}] truncation guard OK: n_docs={report['n_docs']} "
          f"frac_truncated={report['frac_truncated']:.6f}", flush=True)

    shard_idx = evo._next_shard_index(d)
    for start in range(0, len(remaining), SHARD_SIZE):
        chunk = remaining[start:start + SHARD_SIZE]
        ids, vecs = [], []
        # `_token_budget_batches` takes (aid, text) pairs and yields the same
        # shape; ids are collected IN BATCH ORDER so vectors and identifiers
        # cannot drift apart.
        for sub_batch in evo._token_budget_batches(chunk, count_fn):
            ids_b = [aid for aid, _ in sub_batch]
            texts_b = [tx for _, tx in sub_batch]
            v, tok = evo._embed_batch_with_retry(client, texts_b, MODEL_NAME)
            ids.extend(ids_b)
            vecs.extend(v)
            manifest["total_tokens_billed"] += int(tok)
            manifest["cost_usd"] = manifest["total_tokens_billed"] / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD
            # Running total across BOTH variants, taking this variant's live
            # (unwritten) cost rather than its last persisted value.
            others = sum(
                json.loads((variant_dir(v) / "manifest.json").read_text()).get("cost_usd", 0.0)
                for v in VARIANTS
                if v != variant and (variant_dir(v) / "manifest.json").exists()
            )
            if others + manifest["cost_usd"] > HARD_CAP_USD:
                c18.atomic_write_json(manifest, manifest_path)
                raise SystemExit(
                    f"HARD CAP ${HARD_CAP_USD} exceeded "
                    f"(other variants ${others:.4f} + this ${manifest['cost_usd']:.4f}) "
                    "-- stopping before the next batch")
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.shape[1] != EMBED_DIM:
            raise SystemExit(f"unexpected dim {arr.shape[1]} != {EMBED_DIM}")
        np.save(d / f"shard_{shard_idx:04d}_of_api.npy", arr)
        (d / f"shard_{shard_idx:04d}_of_api_ids.json").write_text(json.dumps(ids))
        manifest["n_docs_embedded"] += len(ids)
        c18.atomic_write_json(manifest, manifest_path)
        print(f"[49/{variant}] shard {shard_idx} n={len(ids)} "
              f"cost=${manifest['cost_usd']:.4f} all=${_running_total():.4f}", flush=True)
        shard_idx += 1


def stage_merge(variant: str) -> None:
    d = variant_dir(variant)
    ref_ids = [int(x) for x in np.load(REFERENCE, allow_pickle=True)["admission_ID"]]
    vecs, ids = {}, []
    for idp in sorted(d.glob("shard_*_of_api_ids.json")):
        vp = idp.with_name(idp.name.replace("_ids.json", ".npy"))
        shard_ids = json.loads(idp.read_text())
        arr = np.load(vp)
        for a, row in zip(shard_ids, arr):
            vecs[int(a)] = row
    missing = [a for a in ref_ids if a not in vecs]
    if missing:
        raise SystemExit(f"{len(missing)} admissions not embedded, e.g. {missing[:5]}")
    merged = np.stack([vecs[a] for a in ref_ids]).astype(np.float32)
    np.save(d / "merged.npy", merged)
    (d / "merged_ids.json").write_text(json.dumps(ref_ids))
    print(f"[49/{variant}] merged {merged.shape} in reference row order", flush=True)


def fit(x: np.ndarray, seed: int) -> np.ndarray:
    import hdbscan
    import umap
    # Configuration of record, stated literally rather than imported, so this
    # arm cannot silently drift from the other two encoders' ablation arms.
    coords = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                       metric="euclidean", random_state=seed).fit_transform(x)
    return hdbscan.HDBSCAN(
        min_cluster_size=max(10, len(x) // 200), min_samples=None,
        metric="euclidean", cluster_selection_method="eom",
    ).fit_predict(np.ascontiguousarray(coords, dtype=np.float64)).astype(np.int32)


def agree(a: np.ndarray, b: np.ndarray) -> tuple:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    both = (a != -1) & (b != -1)
    return (float(adjusted_rand_score(a[both], b[both])),
            float(adjusted_mutual_info_score(a[both], b[both])))


def field_ami(labels: np.ndarray, values: np.ndarray) -> float:
    from sklearn.metrics import adjusted_mutual_info_score
    keep = labels != -1
    return float(adjusted_mutual_info_score(labels[keep], values[keep]))


def stage_cluster() -> None:
    import pandas as pd

    payload = np.load(REFERENCE, allow_pickle=True)
    reference = payload["labels"].astype(np.int64)
    ref_ids = [int(x) for x in payload["admission_ID"]]

    ds = pd.read_csv(REV.parent / "manuscript/phenotype_ml_dataset.csv", low_memory=False,
                     usecols=["admission_ID", "treatment_department_eng",
                              "BSI_episode_short_clean"]).drop_duplicates("admission_ID")
    ds = ds.set_index("admission_ID").loc[ref_ids]
    dept = ds["treatment_department_eng"].astype(str).to_numpy()
    org = ds["BSI_episode_short_clean"].astype(str).to_numpy()

    lab_dir = OUT / "labels"
    lab_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "reference": str(REFERENCE),
        "reference_k": int(len(set(reference.tolist()) - {-1})),
        "reference_noise": int((reference == -1).sum()),
        "seeds": SEEDS, "encoder": MODEL_NAME,
        "method": ("UMAP nc2/nn15/md0.1/euclidean/random_state=seed -> HDBSCAN "
                   "mcs=max(10,n//200)=161, ms=None, eom; frozen env; agreement both_nonnoise"),
        "per_fit": {}, "vs_reference": {}, "cross_seed": {},
    }

    for variant in VARIANTS:
        mp = variant_dir(variant) / "merged.npy"
        if not mp.exists():
            raise SystemExit(f"{mp} missing -- run --stage merge --variant {variant}")
        x = np.load(mp)
        labels = {}
        for seed in SEEDS:
            dest = lab_dir / f"{variant}__seed{seed}.npy"
            if dest.exists():
                labels[seed] = np.load(dest)
                print(f"[49/{variant}|{seed}] cached", flush=True)
                continue
            t0 = time.time()
            lab = fit(x, seed)
            tmp = dest.with_name(dest.name + ".tmp.npy")  # np.save appends .npy otherwise
            np.save(tmp, lab)
            tmp.replace(dest)
            labels[seed] = lab
            print(f"[49/{variant}|{seed}] k={len(set(lab.tolist()) - {-1})} "
                  f"noise={(lab == -1).mean():.3f} ({time.time() - t0:.0f}s)", flush=True)

        for seed in SEEDS:
            ari, ami = agree(labels[seed], reference)
            result["per_fit"][f"{variant}|{seed}"] = {
                "k": int(len(set(labels[seed].tolist()) - {-1})),
                "noise": float((labels[seed] == -1).mean()),
                "ari_vs_reference": ari, "ami_vs_reference": ami,
                "ami_department": field_ami(labels[seed], dept),
                "ami_organism": field_ami(labels[seed], org),
            }
        vals = [result["per_fit"][f"{variant}|{s}"]["ari_vs_reference"] for s in SEEDS]
        dep = [result["per_fit"][f"{variant}|{s}"]["ami_department"] for s in SEEDS]
        org_ = [result["per_fit"][f"{variant}|{s}"]["ami_organism"] for s in SEEDS]
        result["vs_reference"][variant] = {
            "n_seeds": len(SEEDS),
            "ari_median": float(np.median(vals)), "ari_min": float(np.min(vals)),
            "ari_max": float(np.max(vals)),
            "ami_department_median": float(np.median(dep)),
            "ami_department_min": float(np.min(dep)), "ami_department_max": float(np.max(dep)),
            "ami_organism_median": float(np.median(org_)),
        }
        pairs = {f"{a}-{b}": agree(labels[a], labels[b])[0]
                 for a, b in itertools.combinations(SEEDS, 2)}
        pv = list(pairs.values())
        result["cross_seed"][variant] = {
            "n_pairs": len(pv), "ari_median": float(np.median(pv)),
            "ari_min": float(np.min(pv)), "ari_max": float(np.max(pv)), "pairs": pairs,
        }

    OUT.mkdir(parents=True, exist_ok=True)
    c18.atomic_write_json(result, OUT / "3small_ablation.json")
    print(f"[49] wrote {OUT / '3small_ablation.json'}", flush=True)
    for v in VARIANTS:
        r = result["vs_reference"][v]
        c = result["cross_seed"][v]
        print(f"  {v:15s} ARIvsRef median {r['ari_median']:+.4f} "
              f"[{r['ari_min']:+.4f}, {r['ari_max']:+.4f}]  "
              f"AMI_dept median {r['ami_department_median']:.4f}  "
              f"cross-seed {c['ari_median']:.4f} [{c['ari_min']:.4f}, {c['ari_max']:.4f}]",
              flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True,
                    choices=["count_tokens", "embed", "merge", "cluster"])
    ap.add_argument("--variant", default=None, choices=VARIANTS + [None])
    args = ap.parse_args()
    if args.stage == "cluster":
        stage_cluster()
        return
    variants = [args.variant] if args.variant else VARIANTS
    for v in variants:
        if args.stage == "count_tokens":
            stage_count_tokens(v)
        elif args.stage == "embed":
            stage_embed(v)
        else:
            stage_merge(v)


if __name__ == "__main__":
    main()
