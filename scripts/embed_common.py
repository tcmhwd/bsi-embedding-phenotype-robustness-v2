#!/usr/bin/env python3
"""Truncation guard, used everywhere an embedding is produced (Tasks 4, 6, 7, MIMIC).

AUDIT C1: every narrative embedded through `intfloat/multilingual-e5-
large` (512-token window) was silently truncated (min 556 / median 708 /
max 2,776 tokens -- 100% of `full` truncated). No script recorded a token
count or checked truncation before writing an embedding. This module makes
that check a hard, mechanical precondition rather than a discipline: every
embed stage must call `token_report()` before encoding and
`assert_no_truncation()` before writing any `.npy`/vector output, and must
persist the report's fields into its manifest (Mi3) so the check's *result*
is recorded, not just the fact that some check ran.

Deliberately dependency-light at import time: importing this module never
imports torch/sentence-transformers/tiktoken (those are only imported
inside `hf_token_counter`'s tokenizer construction, which the caller does,
and inside `openai_token_counter`), so it stays cheap to import from a
lightweight process (e.g. a CLI arg-parsing entrypoint) exactly like
`06_domain_ablation.py`'s import discipline.

See `.superpowers/sdd/DESIGN_embedding_and_stability.md` section 1.5
("Truncation guard (implement once, import everywhere)") -- this module
implements that spec verbatim: `token_report` and `assert_no_truncation`
have the exact signatures/semantics specified there.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import numpy as np

# Manifest keys every embed-stage manifest must carry once a token_report has
# been merged in (Mi3 / design doc section 1.5's manifest field list, minus
# the environment/library fields each stage already records on its own).
MANIFEST_GUARD_KEYS = (
    "tokenizer_name",
    "max_seq_length",
    "n_tokens_min",
    "n_tokens_median",
    "n_tokens_p90",
    "n_tokens_p99",
    "n_tokens_max",
    "n_truncated",
    "frac_truncated",
)


GATE_ROOT = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision/out/04_open_embed")


def model_slug_for_gate(model_name: str) -> str:
    """Map a model identifier to the slug `04_open_embedding.py` uses under
    `GATE_ROOT/<slug>/`. Known open-model candidates map to their gate
    slug; anything else (e.g. the retired `intfloat/multilingual-e5-large`
    control arm, which never had and never will have a passing gate) maps
    to a readable slug that simply will never have a GATE.md -- which is
    the point: `require_passing_gate` then correctly refuses it."""
    known = {
        "BAAI/bge-m3": "bge-m3",
        "Qwen/Qwen3-Embedding-0.6B": "qwen3-0.6b",
    }
    if model_name in known:
        return known[model_name]
    return model_name.split("/")[-1].lower()


def require_passing_gate(model_slug: str) -> dict:
    """Guard used by any embed stage that would run an OPEN model over
    ablation/template narratives (design doc section 1.6, mechanical guard
    #1): refuses to proceed unless `04_open_embedding.py --stage gate` has
    already written `GATE_ROOT/<model_slug>/GATE.md` with a machine-
    readable verdict of ADOPT or CONDITIONAL. Hard failure, not a warning
    -- a FAIL verdict, or no gate having been run at all, must stop the
    embed stage before any encoding happens."""
    gate_path = GATE_ROOT / model_slug / "GATE.md"
    if not gate_path.exists():
        raise RuntimeError(
            f"no Task 4 gate verdict found for model={model_slug!r} "
            f"(expected {gate_path}) -- run `04_open_embedding.py --stage gate` "
            "first. Per the design doc, no ablation/template embedding may "
            "proceed on an open model without a passing gate (AUDIT C2)."
        )
    concordance_path = GATE_ROOT / model_slug / "concordance.json"
    verdict = json.loads(concordance_path.read_text())["verdict"]
    if verdict not in ("ADOPT", "CONDITIONAL"):
        raise RuntimeError(
            f"Task 4 gate verdict for model={model_slug!r} is {verdict!r} -- "
            "must be ADOPT or CONDITIONAL to proceed. A FAIL verdict means "
            "this model must not be used to embed ablation/template variants "
            "(design doc section 1.6)."
        )
    return {"verdict": verdict, "gate_path": str(gate_path)}


class TruncationError(RuntimeError):
    """Raised by `assert_no_truncation` when `frac_truncated > 0`.

    This is the exception the audit's C1 finding says should have existed
    and never did -- catching it (or letting it propagate) is what makes
    "no embedding may be written unless frac_truncated == 0" enforceable
    rather than aspirational.
    """


def token_report(
    texts: Iterable[str],
    count_fn: Callable[[str], int],
    max_seq_length: int,
    tokenizer_name: str,
) -> Dict:
    """Tokenize every document with `count_fn` and report the length
    distribution against `max_seq_length`.

    `count_fn(text) -> int` must count tokens exactly as the model/tokenizer
    that will actually encode `text` would (see `hf_token_counter` and
    `openai_token_counter` below for the two arms this project uses) --
    this function does no tokenization of its own, so a wrong `count_fn` is
    the only way this guard can be fooled, and that surface is small and
    reviewable.

    Returns a dict with n_docs, n_tokens_{min,median,p90,p99,max},
    n_truncated, frac_truncated, max_seq_length, tokenizer_name -- exactly
    the fields named in the design doc, so this dict can be merged directly
    into a manifest (see `merge_report_into_manifest`).
    """
    texts = list(texts)
    n_docs = len(texts)
    if n_docs == 0:
        raise ValueError("token_report() called with zero documents")

    lengths = np.fromiter((count_fn(t) for t in texts), dtype=np.int64, count=n_docs)
    n_truncated = int((lengths > max_seq_length).sum())

    return {
        "tokenizer_name": tokenizer_name,
        "max_seq_length": int(max_seq_length),
        "n_docs": n_docs,
        "n_tokens_min": int(lengths.min()),
        "n_tokens_median": float(np.median(lengths)),
        "n_tokens_p90": float(np.percentile(lengths, 90)),
        "n_tokens_p99": float(np.percentile(lengths, 99)),
        "n_tokens_max": int(lengths.max()),
        "n_truncated": n_truncated,
        "frac_truncated": float(n_truncated) / n_docs,
    }


def assert_no_truncation(report: Dict) -> None:
    """Raise `TruncationError` if `report["frac_truncated"] > 0`.

    No embedding may be written unless this passes -- call it AFTER
    `token_report()` and BEFORE `model.encode(...)`/`np.save(...)`, in
    every embed stage, no exceptions (design doc section 1.5 / C1's
    correction item 2).
    """
    if report["frac_truncated"] > 0:
        raise TruncationError(
            f"{report['n_truncated']}/{report['n_docs']} documents "
            f"({report['frac_truncated']:.4%}) exceed max_seq_length="
            f"{report['max_seq_length']} tokens under tokenizer "
            f"{report['tokenizer_name']!r} (n_tokens_max={report['n_tokens_max']}). "
            "Truncation must be exactly zero before any embedding is "
            "written -- see AUDIT.md finding C1: this is the check "
            "that was missing when every narrative was silently truncated "
            "under e5-large's 512-token window."
        )


def merge_report_into_manifest(manifest: Dict, report: Dict) -> Dict:
    """Return a new manifest dict with the guard's fields merged in, after
    asserting the report passes. Use this at the point a manifest is about
    to be written, so a manifest can never claim success while omitting
    (or silently failing) the truncation check."""
    assert_no_truncation(report)
    out = dict(manifest)
    for key in MANIFEST_GUARD_KEYS:
        out[key] = report[key]
    return out


# ---------------------------------------------------------------------------
# count_fn constructors -- one per tokenizer family this project uses.
# ---------------------------------------------------------------------------

def hf_token_counter(tokenizer) -> Callable[[str], int]:
    """Build a `count_fn` from a HuggingFace/sentence-transformers tokenizer
    that counts tokens exactly as the model will see them: special tokens
    included (CLS/SEP or equivalent), no truncation applied during the
    count itself (truncation=False) so a document that WOULD be truncated
    is still measured at its true length rather than clipped to
    max_seq_length before we can detect that it needed clipping."""
    def _count(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
    return _count


def openai_token_counter(encoding_name: str = "cl100k_base") -> Callable[[str], int]:
    """Build a `count_fn` for the OpenAI embedding arm. `cl100k_base` is the
    encoding used by `text-embedding-3-large`/`-small` and by
    `text-embedding-ada-002`; the model itself adds no special tokens for
    embeddings inputs, so a raw `enc.encode(text)` length is the exact
    count OpenAI's API will bill and truncate against (limit: 8,191 tokens
    for text-embedding-3-large, per the design doc and OpenAI's docs)."""
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)

    def _count(text: str) -> int:
        return len(enc.encode(text))
    return _count


# ---------------------------------------------------------------------------
# Retro-audit CLI (W0.5): run token_report over every existing narrative
# variant corpus and save a CSV supplement. Does not embed anything.
# ---------------------------------------------------------------------------

def _read_texts(narratives_dir: Path) -> List[str]:
    files = sorted(narratives_dir.glob("*.txt"), key=lambda p: int(p.stem))
    return [p.read_text(encoding="utf-8", errors="replace") for p in files]


def retro_truncation_audit(corpora: Dict[str, Path], models: Dict[str, dict],
                            on_row=None) -> "list[dict]":
    """For each (corpus label -> narratives dir) x (model label -> {count_fn,
    max_seq_length, tokenizer_name}) pair, compute a token_report row.
    `models["count_fn"]` is built lazily per model the first time it's
    needed (tokenizer construction can be slow), then reused across corpora.

    `on_row`, if given, is called with the full `rows` list after every row
    completes -- used to write the CSV incrementally so a SLURM time-limit
    cancellation partway through (16 corpora x 2 tokenizers is not fast)
    does not lose already-computed rows (this is exactly what happened to
    job 586384: cancelled 13/16 rows into the second tokenizer pass, taking
    every completed row with it because the CSV was only written at the
    very end)."""
    rows = []
    tokenizer_cache: Dict[str, Callable[[str], int]] = {}
    for model_label, spec in models.items():
        if model_label not in tokenizer_cache:
            tokenizer_cache[model_label] = spec["build_count_fn"]()
        count_fn = tokenizer_cache[model_label]
        for corpus_label, narratives_dir in corpora.items():
            if not narratives_dir.exists():
                continue
            texts = _read_texts(narratives_dir)
            report = token_report(texts, count_fn, spec["max_seq_length"], spec["tokenizer_name"])
            row = {"corpus": corpus_label, "model": model_label}
            row.update(report)
            rows.append(row)
            print(
                f"[{model_label}] {corpus_label}: n_docs={report['n_docs']} "
                f"median={report['n_tokens_median']:.0f} max={report['n_tokens_max']} "
                f"frac_truncated={report['frac_truncated']:.4%}",
                flush=True,
            )
            if on_row is not None:
                on_row(rows)
    return rows


def write_csv(rows: "list[dict]", dest: Path) -> None:
    if not rows:
        raise ValueError("no rows to write")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    tmp = dest.with_name(f".tmp_{dest.name}")
    with open(tmp, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(dest)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", default="/data/tcmhwd/AI/AST_ai/phenotype_revision/out/06_ablation/TRUNCATION_AUDIT.csv",
    )
    args = ap.parse_args()

    root = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision/out")
    corpora = {}
    for variant_dir in sorted((root / "06_ablation" / "narratives").glob("*")):
        if variant_dir.is_dir():
            corpora[f"06_ablation/{variant_dir.name}"] = variant_dir
    for variant_dir in sorted((root / "07_template" / "narratives").glob("*")):
        if variant_dir.is_dir():
            corpora[f"07_template/{variant_dir.name}"] = variant_dir

    def _build_e5_count_fn():
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
        return hf_token_counter(tok)

    def _build_openai_count_fn():
        return openai_token_counter("cl100k_base")

    models = {
        "intfloat/multilingual-e5-large": {
            "build_count_fn": _build_e5_count_fn,
            "max_seq_length": 512,
            "tokenizer_name": "intfloat/multilingual-e5-large",
        },
        "text-embedding-3-large (cl100k_base)": {
            "build_count_fn": _build_openai_count_fn,
            "max_seq_length": 8191,
            "tokenizer_name": "cl100k_base",
        },
    }

    dest = Path(args.out)
    rows = retro_truncation_audit(corpora, models, on_row=lambda r: write_csv(r, dest))
    print(f"wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
