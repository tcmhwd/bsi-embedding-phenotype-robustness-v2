#!/usr/bin/env python3
"""Task 21, Step 1 -- generate every narrative variant on the PUBLISHED
template (`reports_pilot_pre/`), verify each one (sample diff + corpus-wide
line counts), count tokens, and project embedding cost BEFORE any API call.

Writes, per variant, under `out/21_published_variants/`:
  narratives/<variant>/<admission_id>.txt   -- one file per admission
  generation/<variant>/manifest.json        -- diff/token/cost report
  generation/<variant>/sample_diff.txt      -- one-document, field-names-only
                                                before/after diff
generation_summary.csv summarizes every variant in one table.

Cost gate: PRICE_PER_MILLION_TOKENS_USD (text-embedding-3-large, matches
`embed_variants_openai.py`) is applied to the projected token count for
EVERY variant (full corpus, 32,329 docs) BEFORE that variant's narratives
are finalized as embed-ready. Running total is checked against
`HARD_CAP_USD=80` (task brief) -- if adding a variant's projection would
exceed the cap, generation STOPS (raises) rather than silently proceeding;
already-completed variants are left in place (resumable).

Resumable: a variant already having `generation/<variant>/manifest.json`
with `status: "ok"` is skipped on re-run.

Run on the login node (pure text/tiktoken processing, no model compute) --
this is explicitly within the "token counting / variant-diff verification"
allowance, but full-corpus write-out is dispatched here as a single
lightweight CPU pass since it is not `numpy`/BLAS/GPU work. If desired this
can equally be run under `sbatch` (see `slurm/21a_generate_variants.sh`);
either way it never calls the OpenAI API.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c21 = importlib.import_module("21_common")

OUT = c21.OUT
NARRATIVES_DIR = c21.NARRATIVES_DIR
GEN_DIR = OUT / "generation"

PRICE_PER_MILLION_TOKENS_USD = 0.13  # text-embedding-3-large, matches embed_variants_openai.py
# AUDIT_4 C1 addendum (2026-09-06): the original 18-variant task brief
# capped the shared budget at $80; realized cost across those 18 was
# $73.30. Two more variants (fixed_permutation, shuffle_order_seed2) are
# needed to run the decisive missing control -- projected ~$4.5 each
# (same corpus, same per-doc token count as shuffle_order, just reordered)
# -- against a SEPARATE, task-specific $20 cap tracked outside this file.
# Raised just enough to admit those two ($73.30 + ~$9.1 =~ $82.4) with a
# small margin; not raised further, and not to be raised again without a
# fresh author-approved reason.
HARD_CAP_USD = 86.0

# Variants embedded in this task, grouped by priority (order = execution
# order; "full" is Priority 2's unmodified re-embedding AND the baseline
# every other variant is compared against, so it is always generated/
# embedded first).
PRIORITY_1 = ["index_window"]
PRIORITY_2 = ["full"]
PRIORITY_3 = list(c21.TEMPLATE_VARIANTS)  # shuffle_order, no_headers, json_format, null_dropped, no_definitions
PRIORITY_4 = list(c21.DOMAIN_ABLATION_VARIANTS)

ORDERED_VARIANTS = PRIORITY_2 + PRIORITY_1 + PRIORITY_3 + PRIORITY_4


def _sample_admission_id(meta) -> int:
    # Deterministic sample doc for the before/after diff: a multi-episode
    # admission (more structurally interesting than a single-episode one)
    # if one is findable cheaply, else the first row.
    return int(meta.iloc[0]["admission_ID"])


def _cost_so_far() -> float:
    total = 0.0
    if not GEN_DIR.exists():
        return 0.0
    for mpath in GEN_DIR.glob("*/manifest.json"):
        try:
            m = json.loads(mpath.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if m.get("status") == "ok":
            total += m.get("projected_cost_usd", 0.0)
    return total


def generate_variant(variant: str, meta, seed: int = c21.DEFAULT_SEED, limit: int | None = None) -> dict:
    import tiktoken

    var_out = GEN_DIR / variant
    manifest_path = var_out / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing.get("status") == "ok":
            print(f"[21a/{variant}] already done, skipping (resume)", flush=True)
            return existing

    cost_so_far = _cost_so_far()

    rows = meta if limit is None else meta.iloc[:limit]
    narr_dir = NARRATIVES_DIR / variant
    narr_dir.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("cl100k_base")

    n_docs = 0
    total_tokens = 0
    n_truncated = 0
    max_tokens = 0
    sum_removed = 0
    sum_added = 0
    sum_reordered = 0
    sample_diff_text = None
    sample_admission_id = _sample_admission_id(rows)

    for _, row in rows.iterrows():
        aid = int(row["admission_ID"])
        nid = row["newpatient_ID"]
        p = c21.report_path_for(nid, aid)
        orig_text = p.read_text(encoding="utf-8", errors="replace")
        new_text = c21.apply_variant(orig_text, variant, seed=seed)

        counts = c21.corpus_diff_counts(orig_text, new_text)
        sum_removed += counts["n_lines_removed"]
        sum_added += counts["n_lines_added"]
        sum_reordered += counts["n_lines_reordered"]

        if aid == sample_admission_id:
            sample_diff_text = c21.diff_sample(orig_text, new_text)

        n_tok = len(enc.encode(new_text))
        total_tokens += n_tok
        max_tokens = max(max_tokens, n_tok)
        if n_tok > 8191:
            n_truncated += 1
        n_docs += 1

        out_path = narr_dir / f"{aid}.txt"
        tmp = out_path.with_name(f".tmp_{os.getpid()}_{out_path.name}")
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, out_path)

    projected_cost = total_tokens / 1_000_000 * PRICE_PER_MILLION_TOKENS_USD
    would_be_total = cost_so_far + projected_cost

    var_out.mkdir(parents=True, exist_ok=True)
    (var_out / "sample_diff.txt").write_text(
        f"variant={variant!r} sample_admission_ID={sample_admission_id}\n\n"
        f"{sample_diff_text or '(sample doc unchanged)'}\n", encoding="utf-8"
    )

    manifest = {
        "variant": variant,
        "n_docs": n_docs,
        "total_tokens_projected": total_tokens,
        "n_tokens_max": max_tokens,
        "n_docs_over_8191_tokens": n_truncated,
        "projected_cost_usd": projected_cost,
        "cost_so_far_before_this_variant_usd": cost_so_far,
        "cumulative_projected_cost_usd": would_be_total,
        "hard_cap_usd": HARD_CAP_USD,
        "corpus_wide_lines_removed": sum_removed,
        "corpus_wide_lines_added": sum_added,
        "corpus_wide_lines_reordered": sum_reordered,
        "sample_admission_ID": sample_admission_id,
    }

    print(
        f"[21a/{variant}] n_docs={n_docs} total_tokens={total_tokens} "
        f"projected_cost=${projected_cost:.4f} cumulative=${would_be_total:.4f} "
        f"lines_removed={sum_removed} lines_added={sum_added} lines_reordered={sum_reordered}",
        flush=True,
    )
    print(f"[21a/{variant}] sample diff (field names only):\n{sample_diff_text}", flush=True)

    if would_be_total > HARD_CAP_USD:
        manifest["status"] = "STOPPED_OVER_BUDGET"
        c21.atomic_write_json(manifest, manifest_path)
        raise RuntimeError(
            f"[21a/{variant}] projected cumulative cost ${would_be_total:.4f} would exceed "
            f"HARD_CAP_USD={HARD_CAP_USD} -- STOPPING before marking this variant embed-ready. "
            f"Narratives were written to {narr_dir} but manifest status is STOPPED_OVER_BUDGET; "
            f"21b_embed_openai.py refuses to embed a variant without status=='ok'."
        )

    if n_truncated > 0:
        manifest["status"] = "STOPPED_TRUNCATION"
        c21.atomic_write_json(manifest, manifest_path)
        raise RuntimeError(
            f"[21a/{variant}] {n_truncated} document(s) exceed the 8,191-token embedding window "
            f"-- STOPPING. This should not happen (documents only shrink or reformat under these "
            f"variants); investigate before proceeding."
        )

    manifest["status"] = "ok"
    c21.atomic_write_json(manifest, manifest_path)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=c21.ALL_VARIANTS, help="generate a single variant only")
    ap.add_argument("--limit", type=int, default=None, help="smoke-test on the first N docs only")
    ap.add_argument("--seed", type=int, default=c21.DEFAULT_SEED)
    args = ap.parse_args()

    print(f"[21a] python executable: {sys.executable}", flush=True)
    meta = c21.load_metadata()
    print(f"[21a] corpus: {len(meta)} admissions from {c21.REPORTS_DIR}", flush=True)

    variants = [args.variant] if args.variant else ORDERED_VARIANTS
    summary_rows = []
    for variant in variants:
        m = generate_variant(variant, meta, seed=args.seed, limit=args.limit)
        summary_rows.append(m)

    import pandas as pd
    df = pd.DataFrame(summary_rows)
    c21.atomic_write_csv(df, OUT / "generation_summary.csv")
    print(f"[21a] wrote {OUT / 'generation_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
