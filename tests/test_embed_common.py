import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from importlib import import_module

mod = import_module("embed_common")


def _word_count_fn(text: str) -> int:
    """Trivial deterministic count_fn for tests that don't need a real
    tokenizer: one token per whitespace-separated word."""
    return len(text.split())


def test_token_report_basic_stats():
    texts = ["a b c", "a b", "a b c d e"]  # 3, 2, 5 words
    report = mod.token_report(texts, _word_count_fn, max_seq_length=10, tokenizer_name="wordcount")
    assert report["n_docs"] == 3
    assert report["n_tokens_min"] == 2
    assert report["n_tokens_max"] == 5
    assert report["n_truncated"] == 0
    assert report["frac_truncated"] == 0.0
    assert report["max_seq_length"] == 10
    assert report["tokenizer_name"] == "wordcount"


def test_token_report_flags_truncation_fraction():
    texts = ["a " * 3, "a " * 20, "a " * 25]  # 3, 20, 25 tokens
    report = mod.token_report(texts, _word_count_fn, max_seq_length=10, tokenizer_name="wordcount")
    assert report["n_truncated"] == 2
    assert report["frac_truncated"] == pytest.approx(2 / 3)


def test_assert_no_truncation_passes_when_zero():
    report = {"frac_truncated": 0.0, "n_truncated": 0, "n_docs": 5,
              "max_seq_length": 512, "tokenizer_name": "x", "n_tokens_max": 100}
    mod.assert_no_truncation(report)  # must not raise


def test_assert_no_truncation_raises_when_nonzero():
    report = {"frac_truncated": 0.4, "n_truncated": 2, "n_docs": 5,
              "max_seq_length": 512, "tokenizer_name": "x", "n_tokens_max": 900}
    with pytest.raises(mod.TruncationError):
        mod.assert_no_truncation(report)


def test_merge_report_into_manifest_raises_before_writing():
    manifest = {"variant": "full", "model": "x"}
    bad_report = {"frac_truncated": 0.1, "n_truncated": 1, "n_docs": 10,
                  "max_seq_length": 512, "tokenizer_name": "x", "n_tokens_max": 600,
                  "n_tokens_min": 100, "n_tokens_median": 300.0, "n_tokens_p90": 550.0,
                  "n_tokens_p99": 590.0}
    with pytest.raises(mod.TruncationError):
        mod.merge_report_into_manifest(manifest, bad_report)


def test_merge_report_into_manifest_merges_all_guard_keys():
    manifest = {"variant": "full", "model": "x"}
    good_report = {"frac_truncated": 0.0, "n_truncated": 0, "n_docs": 10,
                   "max_seq_length": 512, "tokenizer_name": "x", "n_tokens_max": 400,
                   "n_tokens_min": 50, "n_tokens_median": 200.0, "n_tokens_p90": 350.0,
                   "n_tokens_p99": 390.0}
    merged = mod.merge_report_into_manifest(manifest, good_report)
    assert merged["variant"] == "full"  # original keys preserved
    for key in mod.MANIFEST_GUARD_KEYS:
        assert key in merged and merged[key] == good_report[key]


def test_truncation_guard_rejects_a_document_over_the_window():
    """Required test (design doc Mi4): a synthetic ~600-token document
    against a 512-token window must be caught. Uses the real e5-large
    tokenizer (already cached locally, no network needed) so the count_fn
    reflects what `assert_no_truncation` guards in production, not a toy."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
    count_fn = mod.hf_token_counter(tok)

    # A single word repeated is tokenized as many subword-repeats; verify by
    # construction rather than assuming a token-per-word ratio.
    long_text = "clinical " * 500  # tokenizes to well over 512 for this vocab
    short_text = "clinical finding of note"

    report = mod.token_report(
        [short_text, long_text], count_fn, max_seq_length=512,
        tokenizer_name="intfloat/multilingual-e5-large",
    )
    assert report["n_tokens_max"] > 512, (
        "test setup invariant failed: the synthetic long document must "
        "actually exceed the window for this test to mean anything"
    )
    assert report["n_truncated"] == 1
    assert report["frac_truncated"] == pytest.approx(0.5)
    with pytest.raises(mod.TruncationError):
        mod.assert_no_truncation(report)


def test_openai_token_counter_matches_known_cl100k_count():
    pytest.importorskip("tiktoken")
    count_fn = mod.openai_token_counter("cl100k_base")
    # "hello world" is 2 tokens under cl100k_base (well-known reference value).
    assert count_fn("hello world") == 2
