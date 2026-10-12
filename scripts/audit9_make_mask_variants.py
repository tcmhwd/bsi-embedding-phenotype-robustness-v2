#!/usr/bin/env python3
"""Audit 9 -- build two narrative variants that isolate the antibiotic NAME
from the rest of the therapy block, which `no_therapy` removes wholesale
(6 summary tokens + every antibiotic name/duration line + the per-episode
Empiric-treatment sub-block; ~21 lines and ~21% of tokens per document).

  mask_ab_names_all : every `Antibiotic_<k>_name : <drug>` value (in both the
                      [Treatments during admission] block and the per-episode
                      `Empiric treatment` sub-block) is replaced by the constant
                      `masked`.  Durations, efficacy, glucocorticoid lines and
                      all summary tokens are untouched.
  mask_ab1_name_only: only `Antibiotic_1_name` values are masked; every other
                      antibiotic name stays.

If the Qwen3 partition collapses under mask_ab_names_all, the drug names are
causal; if it survives, something else in the block is.  Line and byte counts
are nearly preserved, so document length is not a confound.

Reads out/21_published_variants/narratives/full, writes
out/45_audit9/narratives/<variant>/<admission_ID>.txt.
"""
import json
import re
from pathlib import Path

REV = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
SRC = REV / "out/21_published_variants/narratives/full"
OUT = REV / "out/45_audit9"

PAT = re.compile(r"^(\s*(?:-|o)?\s*Antibiotic_(\d+)_name\s*:\s*)(.*)$")


def mask(text: str, only_first: bool):
    kept, n = [], 0
    for line in text.split("\n"):
        m = PAT.match(line)
        if m and (not only_first or m.group(2) == "1") and m.group(3).strip() != "":
            kept.append(m.group(1) + "masked")
            n += 1
        else:
            kept.append(line)
    return "\n".join(kept), n


def main():
    ids = json.loads((REV / "out/18_encoder_dec/embeddings/qwen3-0.6b/merged_ids.json").read_text())
    manifest = {}
    for variant, only_first in (("mask_ab_names_all", False), ("mask_ab1_name_only", True)):
        dest = OUT / "narratives" / variant
        dest.mkdir(parents=True, exist_ok=True)
        hist = {}
        for aid in ids:
            t = (SRC / f"{aid}.txt").read_text(encoding="utf-8")
            new, n = mask(t, only_first)
            hist[n] = hist.get(n, 0) + 1
            (dest / f"{aid}.txt").write_text(new, encoding="utf-8")
        manifest[variant] = {"n_docs": len(ids), "masked_line_histogram": dict(sorted(hist.items()))}
        print(variant, manifest[variant], flush=True)
    (OUT / "mask_variants_manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
