#!/usr/bin/env python3
"""Task 21 -- shared parsing/variant-generation utilities for the PUBLISHED
narrative template (`reports_pilot_pre/*.report.md`, 32,329 files, the
2025-12-02 snapshot that actually produced the manuscript's phenotypes).

Everything in `06_domain_ablation.py` / `07_template_sensitivity.py` /
`20a_parse_template_fields.py` etc. that ran against `reports_all` (a later,
differently-templated regeneration) is IRRELEVANT here -- read-only, never
imported for its variant-generation logic. This module redefines every
domain-removal / template-format rule from scratch against the actual
`[Section header]` / bullet / nested-indent structure of `reports_pilot_pre`,
verified by reading multiple full documents (single-episode and
multi(2/5/10)-episode) before writing a single regex.

Corpus-agnostic clustering/alignment helpers (UMAP_KW/HDBSCAN_KW/
min_cluster_size_for/agreement/jaccard_for_cluster/atomic_write_*, and the
`reports_pilot_pre`-specific loaders load_metadata/load_texts/
load_published_frame/align_embeddings_to_admission_ids) are REUSED by import
from `18_common` (which itself is corpus-correct already -- Task 18 used
`reports_pilot_pre` from the start) -- not reimplemented, per project
convention ("reuse the project's existing wrapper conventions").

=====================================================================
Template structure (established by reading >10 full documents, single-
and multi-episode, before writing this parser):
=====================================================================

Section headers: bare bracket lines, e.g. `[Bacterial culture]`. Nine
sections, always in this order: [Summary tokens], [Administrative
informations], [Underlying comorbidities], [Treatments during admission],
[Procedures during the admission], [Sofa/qsofa], [Physiological and
biological parameters], [Bacterial culture], [Infection treatment].

[Summary tokens]: ONE PER LINE, format `<category>_<value>` concatenated
with NO delimiter between category and value (e.g. `antiMRSAab_empiric_no`,
`admission_cluster_10`) -- not a `key : value` line like every other
section. Matched against the fixed `SUMMARY_TOKEN_CATEGORIES` list
(longest-prefix-first; verified collision-free over a 2,300-narrative
sample spanning three independent random seeds).

All other sections use `key : value` lines at three possible bullet styles
(`• ` in Administrative informations/Underlying comorbidities, `- `
everywhere else, or no bullet at all -- flat `key : value`, e.g.
Procedures during the admission and the three glucocorticoid lines) plus
two kinds of ANCHOR lines that must never be treated as a removable/
reorderable field:
  - `BSI_episode_num : <id>` -- episode boundary, appears once per episode
    in BOTH [Bacterial culture] and [Infection treatment], in ascending
    Test_date order in both (verified on a sample multi-episode admission:
    5 episodes, identical episode-ID order in both sections).
  - the fixed sentence `For BSI we organise the data in episodes:` (only
    line in the whole corpus that ends in a bare colon with no field
    value -- special-cased by literal match, never field-split).
  - the bare subheader `Antibiotics` inside [Treatments during admission]
    (a label with no colon at all -- falls through as an anchor by the
    generic "no colon -> anchor" rule below, same treatment as blank lines).

Nesting is INDENTATION-based, not marker-based: a top-level item is a
0-indent bullet/flat line; every following line with ANY leading whitespace
belongs to that item as a continuation, regardless of depth or whether it
carries an `o ` marker (`  Antibiotic_1_duration_days : 12` has no `o ` but
is exactly as much a continuation of `- Antibiotic_1_name : ...` as
`  o Antibiotic_1_name : ...` / `    Antibiotic_1_efficacy : NA` are
continuations inside the AST_detailed / Empiric treatment blocks). This is
a deliberately more general rule than `06_domain_ablation.py`'s hardcoded
`BLOCK_BULLET_KEY` set (which only worked because `reports_all` only ever
nested under two specific named bullets) -- `reports_pilot_pre` nests under
every antibiotic-list bullet too, so indentation, not a bullet-name
allowlist, is the only reliable signal.

Edge cases confirmed present in the corpus and handled explicitly:
  - zero antibiotics: `- No recorded antibiotics during admission (NA)`
    (no colon -- kept/dropped as a whole bullet, never key-split).
  - AST not performed: `  o NA (no AST for selected antibiotics)` (no
    colon -- same treatment, one level deeper).
  - duplicate keys within one AST_detailed block (e.g. two `Ampicillin`
    results for a polymicrobial episode) -- never deduplicated, both kept.
  - no Candida/Fungemia content anywhere in this cohort (checked): no
    free-text organism-name scrub is needed, unlike `reports_all`'s
    `_strip_organism_tokens` dictionary pass -- organism identity here
    lives ONLY in two named key lines, both fully dropped by `no_organism`.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

ROOT = Path("/data/tcmhwd/AI/AST_ai")
REV = ROOT / "phenotype_revision"
OUT = REV / "out" / "21_published_variants"
NARRATIVES_DIR = OUT / "narratives"
LOG_DIR = REV / "logs"

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
c18 = importlib.import_module("18_common")

# Re-exported from 18_common (-> 17_common) verbatim -- see module docstring.
REPORTS_DIR = c18.REPORTS_DIR
PUBLISHED_CLUSTER_COL = c18.PUBLISHED_CLUSTER_COL
UMAP_KW = c18.UMAP_KW
HDBSCAN_KW = c18.HDBSCAN_KW
min_cluster_size_for = c18.min_cluster_size_for
jaccard_for_cluster = c18.jaccard_for_cluster
agreement = c18.agreement
atomic_write_json = c18.atomic_write_json
atomic_write_csv = c18.atomic_write_csv
load_metadata = c18.load_metadata
load_texts = c18.load_texts
load_published_frame = c18.load_published_frame
align_embeddings_to_admission_ids = c18.align_embeddings_to_admission_ids
report_path_for = c18.report_path_for

DEFAULT_SEED = 42

# ---------------------------------------------------------------------------
# [Summary tokens] category vocabulary -- verified collision-free over
# 2,300+ narratives across 3 independent random seeds (see module docstring).
# Sorted longest-first so prefix matching is unambiguous even though no
# actual collision was found (defensive).
# ---------------------------------------------------------------------------
SUMMARY_TOKEN_CATEGORIES = sorted([
    "admission_cluster", "antiMRSAab_empiric", "antiVREab_empiric",
    "antipseudoab_empiric", "carbaab_empiric", "cre_colonization",
    "empiric_cluster", "empiric_tt_adapted", "icu",
    "index_bsi_3gcr", "index_bsi_carbar", "index_bsi_ceftriaxoner",
    "index_bsi_crab", "index_bsi_mrsa", "index_bsi_polymicrobial",
    "index_bsi_vre", "sepsis_quicksofa_2", "sepsis_2", "vre_colonization",
], key=len, reverse=True)


def summary_token_category(token_line: str) -> str:
    """`admission_cluster_10` -> `admission_cluster`. Raises if no known
    category prefixes the token -- a template-vocabulary drift should fail
    loudly, not silently pass a token through every variant unclassified."""
    for cat in SUMMARY_TOKEN_CATEGORIES:
        if token_line.startswith(cat):
            return cat
    raise ValueError(f"unrecognized [Summary tokens] category in line {token_line!r}")


# Domain membership of summary-token categories (see module docstring
# rationale in 21a_generate_variants.py's DOMAIN MAPPING section -- restated
# here as the single source of truth both stages import).
SUMMARY_CAT_RESISTANCE = {
    "index_bsi_3gcr", "index_bsi_carbar", "index_bsi_ceftriaxoner",
    "index_bsi_crab", "index_bsi_mrsa", "index_bsi_vre",
    "cre_colonization", "vre_colonization",
}
# Susceptibility-derived, dual-domain flags (author clarification precedent,
# `06_domain_ablation.py` docstring): surfaced as "therapy" but themselves
# AST-derived judgments. In THIS template only 4 of the original 6 keys are
# surfaced as separate summary tokens (`is_*_tt_excessivelylarge` do not
# appear anywhere in reports_pilot_pre -- checked, absent).
SUMMARY_CAT_THERAPY_SUSCEPT_DERIVED = {
    "antiMRSAab_empiric", "antiVREab_empiric", "antipseudoab_empiric", "carbaab_empiric",
}
SUMMARY_CAT_THERAPY = SUMMARY_CAT_THERAPY_SUSCEPT_DERIVED | {"empiric_cluster", "empiric_tt_adapted"}
SUMMARY_CAT_POLYMICROBIAL = {"index_bsi_polymicrobial"}  # dropped only by host_source_severity_only
SUMMARY_CAT_SEVERITY_HOST = {"icu", "sepsis_2", "sepsis_quicksofa_2", "admission_cluster"}

# Line-level (non-summary-token) key sets, keyed by the text before the
# first ":" with bullet markers stripped (see `_line_key`).
ORGANISM_KEYS = {"BSI_episode_short_clean", "Microbes isolated in this episode"}
RESISTANCE_FLAG_KEYS = {"3GCR", "CarbaR", "MRSA", "VRE", "CeftriaxoneR", "CRAB"}
RESISTANCE_LINE_KEYS = RESISTANCE_FLAG_KEYS | {"AST_summary_tokens"}
RESISTANCE_BLOCK_KEYS = {"AST_detailed"}
THERAPY_BLOCK_KEYS = {"Empiric treatment"}
POLYMICROBIAL_LINE_KEY = "BSI_polymicrobial"
PATIENT_ID_KEY = "Patient ID"
DEPARTMENT_KEY = "Treatment departement (this refers to the place where the patient was hospitalized)"
ADMISSION_DATE_KEY = "Admission Date"
TEST_DATE_KEY = "Test_date"

EPISODE_ANCHOR_PREFIX = "BSI_episode_num"
INFECTION_TREATMENT_PREAMBLE = "For BSI we organise the data in episodes:"

BULLET_MARKERS = ("• ", "- ")


def _indent_len(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _is_header(line: str) -> bool:
    return line.startswith("[") and line.rstrip().endswith("]")


def _is_episode_anchor(line: str) -> bool:
    return line.strip().startswith(EPISODE_ANCHOR_PREFIX)


def _line_key(stripped: str) -> str | None:
    """Return the field key for a `key : value` / bulleted `key : value`
    line, stripping one leading bullet marker if present. Returns None for
    lines with no ':' (bare subheaders like `Antibiotics`, no-colon bullets
    like `- No recorded antibiotics during admission (NA)` / `o NA (no AST
    for selected antibiotics)`) -- these are never key-addressable and are
    always kept verbatim by every domain-removal rule (they carry no
    removable "key" to match against, only free text)."""
    body = stripped
    for marker in BULLET_MARKERS:
        if body.startswith(marker):
            body = body[len(marker):]
            break
    else:
        if body.startswith("o "):
            body = body[2:]
    if ":" not in body:
        return None
    return body.split(":", 1)[0].strip()


# ---------------------------------------------------------------------------
# Generic segmentation: anchors (kept in place, never removed/reordered by
# any variant) vs items (one top-level line + every following more-indented
# continuation line, as one atomic unit). Shared by shuffle_order and
# json_format, and used by the domain-removal line-stream pass to decide
# whether a "drop this key" rule also drops the item's continuation lines.
# ---------------------------------------------------------------------------

def segment(text: str) -> List[Tuple[str, List[str]]]:
    lines = text.split("\n")
    segments: List[Tuple[str, List[str]]] = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if stripped == "" or _is_header(line) or _is_episode_anchor(line) \
                or stripped == INFECTION_TREATMENT_PREAMBLE:
            segments.append(("anchor", [line]))
            i += 1
            continue
        if _indent_len(line) > 0:
            # An indented line with no preceding top-level item in this
            # segmentation call is a parser precondition violation.
            raise ValueError(f"orphan indented line with no preceding top-level item: {line!r}")
        item_key = _line_key(stripped)
        if item_key is None and stripped == "Antibiotics":
            segments.append(("anchor", [line]))
            i += 1
            continue
        block = [line]
        i += 1
        while i < n and lines[i].strip() != "" and _indent_len(lines[i]) > 0:
            block.append(lines[i])
            i += 1
        segments.append(("item", block))
    return segments


def content_lines(text: str) -> List[str]:
    """Every non-blank line, flattened -- used to assert content-preservation
    (multiset equality) independent of segmentation/ordering, exactly like
    `07_template_sensitivity.py::_content_lines`."""
    return [ln for ln in text.split("\n") if ln.strip() != ""]


def item_key(item_lines: List[str]) -> str | None:
    return _line_key(item_lines[0].strip())


# ---------------------------------------------------------------------------
# Line-stream domain-removal primitive: single generic pass, mirrors
# `06_domain_ablation.py::_remove_lines` but keyed on this module's
# indentation-based item boundaries instead of a hardcoded block-bullet set.
# ---------------------------------------------------------------------------

def remove_items(text: str, *, drop_keys: set = frozenset()) -> str:
    """Drop every top-level item (its key line + all continuation lines)
    whose key is in `drop_keys`. Anchors are always kept. A `None`-key item
    (no colon) is never dropped by this function -- only named-key drops
    are supported here; whole-episode/whole-block drops use
    `drop_episodes_after_index` / `remove_summary_categories` instead."""
    out: List[str] = []
    for kind, payload in segment(text):
        if kind == "anchor":
            out.extend(payload)
            continue
        key = item_key(payload)
        if key is not None and key in drop_keys:
            continue
        out.extend(payload)
    return "\n".join(out)


def remove_summary_categories(text: str, *, drop_categories: set) -> str:
    """[Summary tokens]-specific removal: drop lines whose category is in
    `drop_categories`. Leaves every other section untouched."""
    lines = text.split("\n")
    out = []
    in_summary = False
    for line in lines:
        if _is_header(line):
            in_summary = (line.strip() == "[Summary tokens]")
            out.append(line)
            continue
        if in_summary and line.strip() != "":
            cat = summary_token_category(line.strip())
            if cat in drop_categories:
                continue
        out.append(line)
    return "\n".join(out)


def parse_episode_ids_in_order(text: str, section_header: str) -> List[str]:
    """Episode ids (as they appear after `BSI_episode_num : `) in the order
    they occur within one named section, e.g. `[Bacterial culture]`."""
    lines = text.split("\n")
    ids: List[str] = []
    in_section = False
    for line in lines:
        if _is_header(line):
            in_section = (line.strip() == f"[{section_header}]")
            continue
        if in_section and _is_episode_anchor(line):
            ids.append(line.split(":", 1)[1].strip())
    return ids


def keep_only_index_episode(text: str) -> str:
    """Priority 1: keep only the index (first-by-Test_date, i.e.
    first-listed -- verified ascending-Test_date order) BSI episode block in
    BOTH [Bacterial culture] and [Infection treatment]; drop every
    subsequent episode's block (its `BSI_episode_num` anchor line AND every
    item that follows it up to the next episode anchor or the next section
    header) entirely."""
    bc_ids = parse_episode_ids_in_order(text, "Bacterial culture")
    it_ids = parse_episode_ids_in_order(text, "Infection treatment")
    if not bc_ids:
        raise ValueError("no BSI_episode_num found in [Bacterial culture] -- template assumption violated")
    if it_ids and it_ids[0] != bc_ids[0]:
        raise ValueError(
            f"index episode mismatch between sections: Bacterial culture={bc_ids[0]!r} "
            f"Infection treatment={it_ids[0]!r}"
        )
    index_id = bc_ids[0]

    lines = text.split("\n")
    out: List[str] = []
    current_section = None
    dropping_episode = False
    for line in lines:
        if _is_header(line):
            current_section = line.strip().strip("[]")
            dropping_episode = False
            out.append(line)
            continue
        if _is_episode_anchor(line):
            this_id = line.split(":", 1)[1].strip()
            dropping_episode = (this_id != index_id)
            if not dropping_episode:
                out.append(line)
            continue
        if dropping_episode:
            continue
        out.append(line)
    return "\n".join(out)


# Summary-token categories that AGGREGATE OVER ALL EPISODES / the whole
# admission (per AUTHOR_DECISIONS D4: `antiXab_empiric_*` are
# `antiXab_empiric==1 AND BSI_episode_X==0` ORed across the admission --
# i.e. can be true/false depending on organisms found in SUBSEQUENT
# episodes) -- dropped by the index-window variant so it does not smuggle
# subsequent-episode information back in through [Summary tokens] while the
# per-episode blocks are being stripped out. `index_bsi_*` are, by
# construction/name, already index-episode-specific and are KEPT.
# `icu`/`sepsis_2`/`sepsis_quicksofa_2` are left UNTOUCHED here: Task 9
# (AUTHOR_DECISIONS) already flagged these as whole-admission-window
# variables, but reconstructing an index-time value for them would require
# re-deriving from raw source data (out of scope for a text transformation,
# and not what this task's "index-window" text ablation is testing) -- this
# is stated explicitly rather than silently left ambiguous.
INDEX_WINDOW_DROP_SUMMARY_CATEGORIES = {
    "admission_cluster", "antiMRSAab_empiric", "antiVREab_empiric",
    "antipseudoab_empiric", "carbaab_empiric", "cre_colonization",
    "vre_colonization", "empiric_cluster", "empiric_tt_adapted",
}


def apply_index_window(text: str) -> str:
    t = keep_only_index_episode(text)
    t = remove_summary_categories(t, drop_categories=INDEX_WINDOW_DROP_SUMMARY_CATEGORIES)
    return t


# ---------------------------------------------------------------------------
# Priority 4 -- domain ablation, redefined on this template.
# ---------------------------------------------------------------------------

def remove_organism(text: str) -> str:
    return remove_items(text, drop_keys=ORGANISM_KEYS)


def remove_resistance(text: str) -> str:
    t = remove_items(text, drop_keys=RESISTANCE_LINE_KEYS | RESISTANCE_BLOCK_KEYS)
    t = remove_summary_categories(t, drop_categories=SUMMARY_CAT_RESISTANCE)
    return t


def remove_resistance_strict(text: str) -> str:
    t = remove_resistance(text)
    t = remove_summary_categories(t, drop_categories=SUMMARY_CAT_THERAPY_SUSCEPT_DERIVED)
    return t


def remove_therapy(text: str) -> str:
    # Antibiotic list in [Treatments during admission]: every top-level item
    # under the bare "Antibiotics" subheader is antibiotic-identity content
    # (drug name+duration bullets, or the "no antibiotics" placeholder) --
    # removed wholesale by section-scoped item drop, NOT by key name (the
    # keys are numbered Antibiotic_1_name/Antibiotic_2_name/... and the
    # placeholder line has no key at all).
    # Section-scoped pass (segment() does not track section boundaries, so
    # it is not used here -- walk raw lines directly).
    lines = text.split("\n")
    out: List[str] = []
    i = 0
    section = None
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if _is_header(line):
            section = stripped.strip("[]")
            out.append(line)
            i += 1
            continue
        if section == "Treatments during admission" and stripped == "Antibiotics":
            out.append(line)
            i += 1
            # Drop only items that are actually antibiotic-identity content:
            # numbered "Antibiotic_N_name" bullets (+ their continuation
            # lines, e.g. "Antibiotic_N_duration_days") or the no-colon
            # placeholder "- No recorded antibiotics during admission (NA)".
            # Stop as soon as a top-level line does NOT match (e.g. the
            # very next line in this template is always
            # "Short_acting_glucocorticoids : ..." with no blank separator
            # -- glucocorticoids are a different, non-therapy domain and
            # must not be swept up by a blank-line/next-header stop rule).
            while i < n and lines[i].strip() != "" and _indent_len(lines[i]) == 0 and not _is_header(lines[i]):
                top_stripped = lines[i].strip()
                k = _line_key(top_stripped)
                is_antibiotic_item = (k is not None and k.startswith("Antibiotic_")) or (
                    k is None and top_stripped.startswith("- ")
                )
                if not is_antibiotic_item:
                    break
                i += 1
                while i < n and lines[i].strip() != "" and _indent_len(lines[i]) > 0:
                    i += 1
            continue
        out.append(line)
        i += 1
    t = "\n".join(out)
    # [Infection treatment] per-episode "Empiric treatment" block.
    t = remove_items(t, drop_keys=THERAPY_BLOCK_KEYS)
    t = remove_summary_categories(t, drop_categories=SUMMARY_CAT_THERAPY)
    return t


def remove_department(text: str) -> str:
    return remove_items(text, drop_keys={DEPARTMENT_KEY})


def remove_year(text: str) -> str:
    """Redact only the 4-digit year in the (single, admission-level)
    `Admission Date` value, leaving month/day -- the closest analogue to
    `reports_all`'s single `Admission year` key. Deliberately does NOT
    touch per-episode `Test_date` lines (those are episode-timeline
    content, not the admission-cohort-year construct `no_year` targets in
    the original design; `no_dates`, new in this template, is the variant
    that removes calendar dates wholesale -- see its own definition)."""
    out = []
    for line in text.split("\n"):
        stripped = line.strip()
        key = _line_key(stripped)
        if key == ADMISSION_DATE_KEY:
            line = re.sub(r"\b\d{4}-\d{2}-\d{2}\b", lambda m: "XXXX" + m.group(0)[4:], line)
        out.append(line)
    return "\n".join(out)


def remove_dates(text: str) -> str:
    """New variant (task Priority 4): remove `Admission Date` and every
    `Test_date` line entirely -- this corpus contains real, unshifted
    calendar dates (the manuscript's Methods incorrectly states dates were
    withheld)."""
    return remove_items(text, drop_keys={ADMISSION_DATE_KEY, TEST_DATE_KEY})


def remove_summary_tokens_block(text: str) -> str:
    """New variant (task Priority 4): drop every line inside [Summary
    tokens] (keep the header itself, consistent with every other
    domain-removal rule which keeps section headers)."""
    return remove_summary_categories(text, drop_categories=set(SUMMARY_TOKEN_CATEGORIES))


def host_source_severity_only(text: str) -> str:
    t = remove_organism(text)
    t = remove_resistance(t)
    t = remove_therapy(t)
    t = remove_department(t)
    # Full admission-date removal (not just year) -- mirrors `reports_all`'s
    # host_source_severity_only, which used the SAME full _remove_year
    # function (there, "Admission year" was a standalone key removed
    # wholesale, not year-redacted-in-place).
    t = remove_items(t, drop_keys={ADMISSION_DATE_KEY})
    t = remove_items(t, drop_keys={PATIENT_ID_KEY, POLYMICROBIAL_LINE_KEY})
    t = remove_summary_categories(t, drop_categories=SUMMARY_CAT_POLYMICROBIAL)
    return t


DOMAIN_ABLATION_VARIANTS: Dict[str, Callable[[str], str]] = {
    "no_organism": remove_organism,
    "no_resistance": remove_resistance,
    "no_therapy": remove_therapy,
    "no_department": remove_department,
    "no_year": remove_year,
    "host_source_severity_only": host_source_severity_only,
    "no_organism_no_resistance": lambda t: remove_resistance(remove_organism(t)),
    "no_resistance_strict": remove_resistance_strict,
    "no_organism_no_resistance_strict": lambda t: remove_resistance_strict(remove_organism(t)),
    "no_dates": remove_dates,
    "no_summary_tokens": remove_summary_tokens_block,
}

# ---------------------------------------------------------------------------
# Priority 3 -- template-format sensitivity, redefined on this template.
# ---------------------------------------------------------------------------

def shuffle_order(text: str, seed: int = DEFAULT_SEED) -> str:
    """Reorder the top-level items WITHIN each run of consecutive items
    bounded by anchors (section headers, episode anchors, the infection-
    treatment preamble sentence, blank lines, the bare `Antibiotics`
    subheader). A block item (its continuation lines) moves as one atomic
    unit -- a continuation line is never separated from its parent.
    Deterministic given `seed` (per-run RNG seeded from a hash of the run's
    own original content, exactly like `07_template_sensitivity.py`)."""
    segments = segment(text)
    out_segments: List[List[str]] = []
    run: List[List[str]] = []

    def flush(run_items, run_index):
        if len(run_items) <= 1:
            return list(run_items)
        key = "\x1e".join("\n".join(item) for item in run_items)
        digest = hashlib.sha256(f"{seed}:{run_index}:{key}".encode()).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        shuffled = list(run_items)
        rng.shuffle(shuffled)
        return shuffled

    run_index = 0
    for kind, payload in segments:
        if kind == "item":
            run.append(payload)
            continue
        for item in flush(run, run_index):
            out_segments.append(item)
        run_index += 1
        run = []
        out_segments.append(payload)
    for item in flush(run, run_index):
        out_segments.append(item)

    return "\n".join(line for block in out_segments for line in block)


# ---------------------------------------------------------------------------
# AUDIT_4 C1 -- the decisive missing control. `shuffle_order` reorders
# each anchor-bounded run with a PER-DOCUMENT, content-hash-seeded RNG (see
# above): document A's [Underlying comorbidities] run and document B's run
# of the same length are, in general, permuted differently, because the RNG
# seed is derived from each run's own text. That confounds two things the
# withdrawn claim ("the embedding responds to order") conflated: (i) does
# reordering AT ALL move the vector, vs (ii) does making that reordering
# INCONSISTENT across documents (so the corpus loses a shared surface form)
# collapse the UMAP+HDBSCAN partition. `fixed_permutation` isolates (i):
# ONE rule, chosen once, with no dependence on document content -- the
# output position of an item within an anchor-bounded run of length n is
# always n-1-(its input position), i.e. reverse the run. Same run
# definition as `shuffle_order` (`segment()`: bounded by section headers,
# episode anchors, the infection-treatment preamble, blank lines, the bare
# `Antibiotics` subheader), same "never move an item across a section or
# episode, keep multi-line items intact" property -- the ONLY difference
# from `shuffle_order` is that this rule is coherent across every document
# instead of independently redrawn per document. Verified on >=3 documents
# in `scripts/26a_verify_fixed_permutation.py` (out/26_controls/): for
# every anchor-bounded run, the output key sequence is exactly the reverse
# of the input key sequence, a pure function of position within the run,
# never of the item's own content or the document it came from.
def fixed_permutation(text: str) -> str:
    """The coherent-transformation counterpart to `shuffle_order`: reverse
    the top-level items within each anchor-bounded run, identically for
    every document (no seed, no content-dependence)."""
    segments = segment(text)
    out_segments: List[List[str]] = []
    run: List[List[str]] = []

    def flush(run_items):
        return list(reversed(run_items))

    for kind, payload in segments:
        if kind == "item":
            run.append(payload)
            continue
        for item in flush(run):
            out_segments.append(item)
        run = []
        out_segments.append(payload)
    for item in flush(run):
        out_segments.append(item)

    return "\n".join(line for block in out_segments for line in block)


# A second, independent RNG seed for `shuffle_order`'s per-document,
# content-hash-seeded permutation (see `shuffle_order`'s own docstring) --
# NOT the clustering seed (42, used throughout for UMAP/HDBSCAN and as
# `shuffle_order`'s own DEFAULT_SEED). Purpose (AUDIT_4 C1): if two
# independent draws of "incoherent, per-document reordering" agree with
# each other (both collapse the partition the same way), the collapse is a
# property of incoherence itself, not of one unlucky seed 42 draw.
SHUFFLE_ORDER_SEED2 = 2


def shuffle_order_seed2(text: str) -> str:
    return shuffle_order(text, seed=SHUFFLE_ORDER_SEED2)


def no_headers(text: str) -> str:
    return "\n".join(line for line in text.split("\n") if not _is_header(line))


def _strip_parens(s: str) -> str:
    out = []
    i, n = 0, len(s)
    while i < n:
        if s[i] == "(":
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if s[j] == "(":
                    depth += 1
                elif s[j] == ")":
                    depth -= 1
                j += 1
            if out and out[-1] == " ":
                out.pop()
            i = j
            continue
        out.append(s[i])
        i += 1
    return "".join(out)


def no_definitions(text: str) -> str:
    """Strip parenthetical glosses from section headers (none contain any
    in this template -- no-op there, kept for symmetry/robustness) and from
    key labels only. Only one key label in this template has a
    parenthetical gloss (`Treatment departement (this refers to ...)`); two
    other parens sightings (`- No recorded antibiotics during admission
    (NA)`, `o NA (no AST for selected antibiotics)`) are no-colon bullets
    with no key part, so they are values/free-text and are left untouched
    (never strip from values, per rule)."""
    out_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if _is_header(line):
            out_lines.append(_strip_parens(line))
            continue
        if ":" in line:
            idx = line.index(":")
            key_part, rest = line[:idx], line[idx:]
            out_lines.append(_strip_parens(key_part) + rest)
            continue
        out_lines.append(line)
    return "\n".join(out_lines)


NULL_MARKERS = {"NA", "nan"}


def null_dropped(text: str) -> str:
    out_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if line.strip() == "" or _is_header(line) or _is_episode_anchor(line) \
                or stripped == INFECTION_TREATMENT_PREAMBLE:
            out_lines.append(line)
            continue
        body = stripped
        for marker in BULLET_MARKERS:
            if body.startswith(marker):
                body = body[len(marker):]
                break
        else:
            if body.startswith("o "):
                body = body[2:]
        if ":" in body:
            _, _, val = body.partition(":")
            if val.strip() in NULL_MARKERS:
                continue
        out_lines.append(line)
    return "\n".join(out_lines)


def parse_narrative(text: str) -> dict:
    """Generic, information-preserving nested structure for `json_format`:
    sections (by header) contain entries; an entry is `{"key","value"}`,
    `{"key","items":[...]}` for a nested block, or `{"episode": id,
    "fields": [...]}`. The `For BSI we organise...` preamble sentence and
    blank lines carry no information and are not represented -- exact
    reconstruction of those is not required for information-equivalence.
    A generic line-shape parser (no key named explicitly), symmetric with
    `segment()` above."""
    lines = text.split("\n")
    sections = []
    current_section = None
    current_episode = None
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()
        if _is_header(line):
            current_section = {"header": stripped.strip("[]"), "entries": []}
            sections.append(current_section)
            current_episode = None
            i += 1
            continue
        if stripped == "" or stripped == INFECTION_TREATMENT_PREAMBLE:
            i += 1
            continue
        if _is_episode_anchor(line):
            current_episode = {"episode": line.split(":", 1)[1].strip(), "fields": []}
            current_section["entries"].append(current_episode)
            i += 1
            continue
        target = current_episode["fields"] if current_episode is not None else current_section["entries"]
        if stripped == "Antibiotics":
            i += 1
            continue  # bare subheader, no information beyond grouping (section already known)
        key = _line_key(stripped)
        if key is None:
            # no-colon bullet/free-text line (e.g. "No recorded antibiotics...")
            target.append({"text": stripped.lstrip("•- ").strip()})
            i += 1
            continue
        _, _, val = stripped.partition(":")
        # strip one leading bullet marker from before the colon check already done via key
        val = val.strip()
        if val == "" and _indent_len(lines[i]) == 0 and i + 1 < n and lines[i + 1].strip() != "" \
                and _indent_len(lines[i + 1]) > _indent_len(lines[i]):
            items = []
            base_indent = _indent_len(lines[i])
            i += 1
            while i < n and lines[i].strip() != "" and _indent_len(lines[i]) > base_indent:
                sub_stripped = lines[i].strip()
                sub_key = _line_key(sub_stripped)
                if sub_key is None:
                    items.append({"text": sub_stripped.lstrip("o ").strip()})
                else:
                    _, _, sub_val = sub_stripped.partition(":")
                    items.append({"key": sub_key, "value": sub_val.strip()})
                i += 1
            target.append({"key": key, "items": items})
            continue
        target.append({"key": key, "value": val})
        i += 1
    return {"sections": sections}


def json_format(text: str) -> str:
    struct = parse_narrative(text)
    return json.dumps(struct, ensure_ascii=False, separators=(",", ":"))


TEMPLATE_VARIANTS: Dict[str, Callable[..., str]] = {
    "shuffle_order": shuffle_order,
    "no_headers": no_headers,
    "json_format": json_format,
    "null_dropped": null_dropped,
    "no_definitions": no_definitions,
    # AUDIT_4 C1 additions -- see the functions' own docstrings above.
    "fixed_permutation": fixed_permutation,
    "shuffle_order_seed2": shuffle_order_seed2,
}

ALL_VARIANTS = ["full", "index_window"] + list(TEMPLATE_VARIANTS) + list(DOMAIN_ABLATION_VARIANTS)


def apply_variant(text: str, variant: str, seed: int = DEFAULT_SEED) -> str:
    if variant == "full":
        return text
    if variant == "index_window":
        return apply_index_window(text)
    if variant in TEMPLATE_VARIANTS:
        fn = TEMPLATE_VARIANTS[variant]
        return fn(text, seed) if variant == "shuffle_order" else fn(text)
    if variant in DOMAIN_ABLATION_VARIANTS:
        return DOMAIN_ABLATION_VARIANTS[variant](text)
    raise ValueError(f"unknown variant {variant!r}; choices are {ALL_VARIANTS}")


# ---------------------------------------------------------------------------
# Verification: field-names-only diff + corpus-wide line-level counts.
# Never prints/returns patient VALUES -- only structural skeletons (bullet
# marker + key, or a length-only placeholder for no-key free text) -- per
# the project's data-sensitivity policy (project configuration notes: no patient-level values
# in logs beyond what a task explicitly requires).
# ---------------------------------------------------------------------------

def redacted_skeleton(text: str) -> List[str]:
    out = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped == "":
            out.append("")
            continue
        if _is_header(line) or _is_episode_anchor(line) or stripped == INFECTION_TREATMENT_PREAMBLE \
                or stripped == "Antibiotics":
            out.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip(" "))]
        marker = ""
        body = stripped
        for m in BULLET_MARKERS:
            if body.startswith(m):
                marker, body = m, body[len(m):]
                break
        else:
            if body.startswith("o "):
                marker, body = "o ", body[2:]
        key = None
        if ":" in body:
            key = body.split(":", 1)[0].strip()
        if key is not None:
            out.append(f"{indent}{marker}{key} : <redacted>")
        else:
            out.append(f"{indent}{marker}<free-text, {len(body)} chars>")
    return out


def diff_sample(orig_text: str, new_text: str) -> str:
    import difflib
    a = redacted_skeleton(orig_text)
    b = redacted_skeleton(new_text)
    if a == b:
        return "(no structural change -- redacted skeleton identical)"
    return "\n".join(difflib.unified_diff(a, b, fromfile="before", tofile="after", lineterm=""))


def corpus_diff_counts(orig_text: str, new_text: str) -> dict:
    """Per-document line-level counts: n_lines_orig/new, n_lines_removed
    (multiset-difference count of content lines present in orig but not in
    new -- 0 for a pure reorder/reformat), n_lines_reordered (only
    meaningful when the content-line multiset is unchanged: count of index
    positions whose content line differs between orig and new)."""
    orig_lines = content_lines(orig_text)
    new_lines = content_lines(new_text)
    import collections
    orig_ct, new_ct = collections.Counter(orig_lines), collections.Counter(new_lines)
    removed = sum((orig_ct - new_ct).values())
    added = sum((new_ct - orig_ct).values())
    reordered = 0
    if removed == 0 and added == 0 and len(orig_lines) == len(new_lines):
        reordered = sum(1 for x, y in zip(orig_lines, new_lines) if x != y)
    return {
        "n_lines_orig": len(orig_lines), "n_lines_new": len(new_lines),
        "n_lines_removed": removed, "n_lines_added": added, "n_lines_reordered": reordered,
    }
