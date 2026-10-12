#!/usr/bin/env python3
"""Audit 7 (question B) -- build the `no_cluster_ids` narrative variant.

The published narrative's [Summary tokens] block renders two pre-computed
cluster IDs (`admission_cluster_<k>`, `empiric_cluster_<k>`).  The existing
`no_summary_tokens` variant (out/21_published_variants) removes the whole
19-token block, which is too coarse to isolate the two IDs.  This variant
removes ONLY those two lines and leaves every other byte untouched, so a
Qwen3 re-embedding of it answers "does the Korean Qwen3 partition depend on
the rendered cluster-ID tokens" without also removing ICU/sepsis/resistance
flags.

Reads reports_pilot_pre in metadata order (18_common.load_texts) and writes
out/41_audit7/narratives/no_cluster_ids/<admission_ID>.txt.  Asserts that
exactly two lines were removed from every document.
"""
import importlib
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c18 = importlib.import_module("18_common")

OUT = SCRIPT_DIR.parent / "out/41_audit7"
DEST = OUT / "narratives/no_cluster_ids"


def strip_cluster_ids(text: str) -> tuple[str, int]:
    kept, removed = [], 0
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("admission_cluster_") or s.startswith("empiric_cluster_"):
            removed += 1
            continue
        kept.append(line)
    return "\n".join(kept), removed


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    frame = c18.load_texts()
    counts = {}
    for aid, text in zip(frame["admission_ID"], frame["text"]):
        new, removed = strip_cluster_ids(text)
        counts[removed] = counts.get(removed, 0) + 1
        (DEST / f"{int(aid)}.txt").write_text(new, encoding="utf-8")
    manifest = {"n_docs": int(len(frame)), "removed_line_histogram": counts,
                "source": str(c18.REPORTS_DIR), "dest": str(DEST)}
    (OUT / "no_cluster_ids_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    assert set(counts) == {2}, f"expected exactly 2 removed lines per doc, got {counts}"


if __name__ == "__main__":
    main()
