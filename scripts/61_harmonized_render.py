#!/usr/bin/env python3
"""Analysis A (post hoc follow-up), PHASE 1 + PHASE 2 (PREREGISTRATION_A_ADDENDUM_1.md) -- render the three
narrative corpora, QA them, and assemble the author's sign-off package.  NO embedding, NO
clustering, NO API call, NO GPU (tokenizer files only) happens here.

Phase 2 (Addendum 1): AST_detailed and the duplicate `Microbes isolated in this episode` line are removed
from EVERY arm (incl. the random arm); the efficacy lines are dropped in BOTH deterministic arms, so
`harmonized_keep_therapy` differs from the primary arm ONLY by the whole-admission antibiotics block.

Corpus of record: /data/tcmhwd/AI/AST_ai/reports_pilot_pre (32,329 files, read
only).  Its renderer's source no longer exists, so every arm is built by
deterministic LINE SURGERY on it (scripts/21_common.py helpers), exactly as the
20 existing variants were; identical values, identical order, identical syntax,
only the set of present lines changes.

Arms (out/followup_analyses/A_harmonized_template/narratives/<arm>/<admission_ID>.txt):
  korean_mimic_compatible   PRIMARY. Manifest-driven drops (PREREGISTRATION_A D-A-alpha/beta/gamma/delta).
  harmonized_keep_therapy   identical to the primary arm EXCEPT that the whole-admission antibiotics block
                            (its `Antibiotics` subheader and the section header) is retained.
  thickness_matched_random  (1) removes the Addendum-1 lines (AST_detailed, duplicate Microbes line) as in
                            every arm; (2) then removes a random set of the remaining top-level rendered items
                            (numpy default_rng(20260921)) so that the TOTAL per-document removed-line count
                            equals the amended PRIMARY arm's per-document count (as measured by
                            21_common.corpus_diff_counts); section headers / episode anchors are preserved;
                            any per-document shortfall is topped up with single-line items so the corpus-wide
                            total is EXACT.

Every drop operation of the two deterministic arms is DERIVED FROM the reviewed
manifest (harmonized_field_manifest.csv: `mechanism` + `retained_<arm>` columns),
not from constants in this file.  QA additionally recomputes both arms through an
independent reference composition of 21_common helpers and requires byte equality.

Stages:  --stage render | qa | package        (env: bsi_pheno; qa is tokenizer-bound -> sbatch)
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import csv
import datetime as dt
import hashlib
import importlib
import json
import os
import re
import statistics
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
m60 = importlib.import_module("60_harmonized_manifest")
c21 = m60.c21
c18 = m60.c18

REV = m60.REV
A_DIR = m60.A_DIR
REPORTS_DIR = c18.REPORTS_DIR
ARMS_DET = m60.ARMS_DET
ARM_RANDOM = m60.ARM_RANDOM
ALL_ARMS = ARMS_DET + [ARM_RANDOM]
SEED_RANDOM = 20260921
TOKENIZERS = {
    "bge-m3": {"path": str(REV / "model_cache" / "bge-m3"), "max_seq_length": 8192},
    "qwen3-0.6b": {"path": "/home/tcmhwd/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B/snapshots/"
                           "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3", "max_seq_length": 32768},
}
ST, AD, CM, TR, PR, SO, PH, BC, IT = (m60.ST, m60.AD, m60.CM, m60.TR, m60.PR, m60.SO, m60.PH, m60.BC, m60.IT)
ANTIBIOTICS = m60.ANTIBIOTICS_SUBHEADER


def nonblank(text: str) -> list[str]:
    return c21.content_lines(text)


# ----------------------------------------------------------------------------
# Manifest-driven operations
# ----------------------------------------------------------------------------
def load_ops(manifest_csv: Path) -> dict[str, dict]:
    """arm -> ops dict, read from the on-disk (reviewed) manifest and cross-checked
    against the in-module table so the two cannot silently diverge."""
    import pandas as pd
    df = pd.read_csv(manifest_csv, keep_default_na=False)
    by_id = {r["field_id"]: r for r in m60.ROWS}
    if set(df["field_id"]) != set(by_id):
        raise SystemExit("[61] manifest CSV field_ids differ from scripts/60 table -- re-run 60 --stage manifest")
    ops = {}
    for arm in ARMS_DET:
        col = f"retained_{arm}"
        o = dict(drop_summary=set(), drop_items=collections.defaultdict(set), efficacy=False, walk=False, drop_sections=set())
        walk_flags, eff_flags = [], []
        removed_ids = set()
        for _, r in df.iterrows():
            v = r[col]
            exp = m60.arm_retains(by_id[r["field_id"]], arm)
            if (v == "n/a") != (exp is None) or (v in ("yes", "no") and (v == "yes") != bool(exp)):
                raise SystemExit(f"[61] manifest CSV disagrees with scripts/60 table at {r['field_id']} ({arm})")
            if (r["all_arms_removal_addendum1"] == "yes") != bool(by_id[r["field_id"]]["all_arms_removal"]):
                raise SystemExit(f"[61] manifest CSV disagrees with scripts/60 table (all_arms_removal) at {r['field_id']}")
            if v != "no":
                continue
            removed_ids.add(r["field_id"])
            keys = [k for k in r["korean_line_keys"].split(" ;; ") if k]
            mech = r["mechanism"]
            if mech == "summary_category":
                o["drop_summary"].add(keys[0].split("::", 1)[1])
            elif mech == "remove_items":
                for k in keys:
                    sec, key = k.split("::", 1)
                    o["drop_items"][sec].add(key)
            elif mech == "block_continuation":
                pass  # removed together with the parent item (checked below)
            elif mech == "efficacy_subline":
                eff_flags.append(True)
            elif mech == "admission_antibiotic_walk":
                walk_flags.append(r["field_id"])
            elif mech == "section_header":
                o["drop_sections"].add(keys[0].split("::", 1)[1])
            else:
                raise SystemExit(f"[61] row {r['field_id']} is dropped in {arm} but has mechanism {mech!r}")
        o["efficacy"] = bool(eff_flags)
        o["walk"] = bool(walk_flags)
        o["drop_items"] = {k: set(v) for k, v in o["drop_items"].items()}
        # block continuation rows must have their parent removed in the same arm
        if {"B07", "B08", "B09"} & removed_ids and "B06" not in removed_ids:
            raise SystemExit(f"[61] AST_detailed continuation rows removed without the parent item in {arm}")
        ops[arm] = o
    # Addendum-1 items removed from EVERY arm incl. the random arm (rows flagged all_arms_removal_addendum1 == yes)
    allarms = collections.defaultdict(set)
    for _, r in df.iterrows():
        if r["all_arms_removal_addendum1"] == "yes" and r["mechanism"] == "remove_items":
            for k in [k for k in r["korean_line_keys"].split(" ;; ") if k]:
                sec, key = k.split("::", 1)
                allarms[sec].add(key)
    for arm in ARMS_DET:
        for sec, keys in allarms.items():
            if not keys <= ops[arm]["drop_items"].get(sec, set()):
                raise SystemExit(f"[61] Addendum-1 all-arm removal {sec}:{keys} is not removed in {arm}")
    ops["__all_arms__"] = {"drop_items": {k: set(v) for k, v in allarms.items()}}
    return ops


def apply_all_arm_removals(text: str, ops_all: dict) -> str:
    """The Addendum-1 decision 1-2 removals only (used to build the random arm's base)."""
    secs = split_sections(text)
    new = []
    for hdr, body in secs:
        sec = hdr.strip().strip("[]") if hdr else None
        if sec in ops_all["drop_items"]:
            body = c21.remove_items("\n".join(body), drop_keys=set(ops_all["drop_items"][sec])).split("\n")
        new.append((hdr, body))
    return join_sections(new)


def split_sections(text: str):
    secs, cur = [], (None, [])
    for ln in text.split("\n"):
        if c21._is_header(ln):
            secs.append(cur)
            cur = (ln, [])
        else:
            cur[1].append(ln)
    secs.append(cur)
    return secs


def join_sections(secs) -> str:
    out = []
    for hdr, body in secs:
        if hdr is not None:
            out.append(hdr)
        out.extend(body)
    return "\n".join(out)


def walk_drop_admission_antibiotics(body: list[str]) -> list[str]:
    """Verbatim logic of the first half of 21_common.remove_therapy (:428-461), applied to the
    [Treatments during admission] body: drops the numbered Antibiotic_N_name bullets (+ indented
    continuation lines) and the no-colon '- No recorded antibiotics' placeholder under the bare
    'Antibiotics' subheader, stopping at the first non-antibiotic top-level line (glucocorticoids).
    The subsequent `remove_items(THERAPY_BLOCK_KEYS)` (:464) and summary-category drop (:465) are
    NOT applied (PREREGISTRATION_A D-A-gamma).  Equivalence with remove_therapy is asserted in QA."""
    out, i, n = [], 0, len(body)
    while i < n:
        line = body[i]
        if line.strip() == ANTIBIOTICS:
            out.append(line)
            i += 1
            while i < n and body[i].strip() != "" and c21._indent_len(body[i]) == 0 and not c21._is_header(body[i]):
                top = body[i].strip()
                k = c21._line_key(top)
                is_ab = (k is not None and k.startswith("Antibiotic_")) or (k is None and top.startswith("- "))
                if not is_ab:
                    break
                i += 1
                while i < n and body[i].strip() != "" and c21._indent_len(body[i]) > 0:
                    i += 1
            continue
        out.append(line)
        i += 1
    return out


EFF_RE = re.compile(r"^Antibiotic_\d+_efficacy$")


def drop_efficacy_lines(body: list[str]) -> list[str]:
    out = []
    for ln in body:
        if c21._indent_len(ln) > 0:
            k = c21._line_key(ln.strip())
            if k is not None and EFF_RE.match(k):
                continue
        out.append(ln)
    return out


def apply_arm(text: str, ops: dict, ident: str = "") -> str:
    t = text
    if ops["drop_summary"]:
        t = c21.remove_summary_categories(t, drop_categories=set(ops["drop_summary"]))
    secs = split_sections(t)
    new = []
    for hdr, body in secs:
        sec = hdr.strip().strip("[]") if hdr else None
        if sec in ops["drop_items"]:
            body = c21.remove_items("\n".join(body), drop_keys=set(ops["drop_items"][sec])).split("\n")
        if sec == IT and ops["efficacy"]:
            body = drop_efficacy_lines(body)
        if sec == TR and ops["walk"]:
            body = walk_drop_admission_antibiotics(body)
        new.append((hdr, body))
    kept = []
    for hdr, body in new:
        sec = hdr.strip().strip("[]") if hdr else None
        if sec in ops["drop_sections"]:
            left = [ln for ln in body if ln.strip() not in ("", ANTIBIOTICS)]
            if left:
                raise RuntimeError(f"[61] section [{sec}] scheduled for removal is not empty in {ident}: {len(left)} line(s) left")
            continue
        kept.append((hdr, body))
    return join_sections(kept)


# ----------------------------------------------------------------------------
# Independent reference composition (typed from the PREREGISTRATION text, NOT from the manifest)
# ----------------------------------------------------------------------------
_REF_KEYS_DROP_ALWAYS = {
    "Patient ID", "Admission Date", "Test_date", c21.DEPARTMENT_KEY,
    "All ICD10 codes during the admission stay whether infectious or not",
    "Short_acting_glucocorticoids", "Intermediate_acting_glucocorticoids", "Long_acting_glucocorticoids",
    "central_orarterial_line", "urinary_orrenal_catheter", "ventilator",
    "SBP", "DBP", "MBP", "RR", "Body_temperature", "PR", "GCS_2", "Vasopressin_any", "CRP", "AST", "ALT", "pO2_Arterial",
}
_REF_SOFA_KEYS = {"sofa_score_2", "sepsis_2", "quick_sofa_score_2", "sepsis_quicksofa_2"}


def reference_arm(text: str, arm: str) -> str:
    """Compose the arm from 21_common helpers only (remove_dates, remove_department, remove_items,
    remove_summary_categories) plus the same section/efficacy/walk primitives."""
    keep5 = {"icu", "antiMRSAab_empiric", "antiVREab_empiric", "antipseudoab_empiric", "carbaab_empiric"}
    t = c21.remove_summary_categories(text, drop_categories=set(c21.SUMMARY_TOKEN_CATEGORIES) - keep5)
    t = c21.remove_dates(t)
    t = c21.remove_department(t)
    t = c21.remove_items(t, drop_keys=_REF_KEYS_DROP_ALWAYS | _REF_SOFA_KEYS)
    # Addendum 1 decisions 1-2 (every arm): AST_detailed block (21_common.RESISTANCE_BLOCK_KEYS) and the duplicate Microbes line
    t = c21.remove_items(t, drop_keys=set(c21.RESISTANCE_BLOCK_KEYS) | {"Microbes isolated in this episode"})
    secs = split_sections(t)
    out = []
    for hdr, body in secs:
        sec = hdr.strip().strip("[]") if hdr else None
        if sec == IT:
            body = drop_efficacy_lines(body)          # Addendum 1 decision 3: dropped in BOTH arms
        if arm == "korean_mimic_compatible":
            if sec == TR:
                body = walk_drop_admission_antibiotics(body)
        if sec in (PR, SO):
            continue
        if sec == TR and arm == "korean_mimic_compatible":
            continue
        out.append((hdr, body))
    return join_sections(out)


# ----------------------------------------------------------------------------
# Random arm
# ----------------------------------------------------------------------------
def make_random_arm(orig_texts: list[str], budgets: list[int], seed: int = SEED_RANDOM):
    """Deterministic. `orig_texts` are the texts AFTER the Addendum-1 all-arm removals and `budgets` the residual
    per-document budgets. Per-document pass in row order with ONE Generator(PCG64(seed)) stream, then an exact
    corpus-wide top-up. Returns (texts, info)."""
    rng = np.random.default_rng(seed)
    segs_all, removed_all, sizes_all, elig_all = [], [], [], []
    deficit = 0
    for i, text in enumerate(orig_texts):
        segs = c21.segment(text)
        items = [j for j, (k, _) in enumerate(segs) if k == "item"]
        sizes = {j: len(segs[j][1]) for j in items}
        removed: set[int] = set()
        rem = int(budgets[i])
        if rem > 0 and items:
            order = rng.permutation(len(items))
            for o in order:
                if rem == 0:
                    break
                j = items[int(o)]
                if sizes[j] <= rem:
                    removed.add(j)
                    rem -= sizes[j]
        deficit += rem
        segs_all.append(segs)
        removed_all.append(removed)
        sizes_all.append(sizes)
        elig_all.append(items)
    n_topup = 0
    if deficit > 0:
        # exact corpus-wide top-up with size-1 items from randomly chosen documents
        pool = [i for i in range(len(orig_texts)) if any(sizes_all[i][j] == 1 and j not in removed_all[i] for j in elig_all[i])]
        while deficit > 0:
            if not pool:
                raise RuntimeError("[61] random arm: cannot reach the corpus-wide removed-line target")
            pi = int(rng.integers(len(pool)))
            i = pool[pi]
            cands = [j for j in elig_all[i] if sizes_all[i][j] == 1 and j not in removed_all[i]]
            if not cands:
                pool.pop(pi)
                continue
            j = cands[int(rng.integers(len(cands)))]
            removed_all[i].add(j)
            deficit -= 1
            n_topup += 1
            if len(cands) == 1:
                pool.pop(pi)
    texts, per_doc_removed = [], []
    for i in range(len(orig_texts)):
        keep_lines = []
        rmv = 0
        for j, (k, payload) in enumerate(segs_all[i]):
            if j in removed_all[i]:
                rmv += len(payload)
                continue
            keep_lines.extend(payload)
        texts.append("\n".join(keep_lines))
        per_doc_removed.append(rmv)
    info = {"seed": seed, "bit_generator": type(rng.bit_generator).__name__, "numpy": np.__version__,
            "n_topup_lines": n_topup, "n_docs_with_per_doc_budget_met": int(sum(1 for a, b in zip(per_doc_removed, budgets) if a == b)),
            "per_doc_removed": per_doc_removed}
    return texts, info


# ----------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------
def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".tmp_{os.getpid()}_{path.name}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_arm(out_dir: Path, aids, texts) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    real = out_dir.resolve()
    if str(real).startswith(str(REPORTS_DIR.resolve())):
        raise SystemExit("[61] refusing to write inside the corpus of record")

    def _w(a_t):
        a, t = a_t
        p = out_dir / f"{int(a)}.txt"
        try:  # small-file writes are slow on this file system: skip files that are already byte-identical
            if p.exists() and p.read_bytes() == t.encode("utf-8"):
                return
        except OSError:
            pass
        atomic_write_text(p, t)

    with cf.ThreadPoolExecutor(max_workers=m60.n_workers() * 2) as ex:
        list(ex.map(_w, zip(aids, texts)))
    # remove stale files from a previous (e.g. --limit) run
    keep = {f"{int(a)}.txt" for a in aids}
    for p in out_dir.glob("*.txt"):
        if p.name not in keep:
            p.unlink()


def read_arm(out_dir: Path, aids):
    def _r(a):
        b = (out_dir / f"{int(a)}.txt").read_bytes()
        return b.decode("utf-8"), hashlib.sha256(b).hexdigest(), len(b)

    with cf.ThreadPoolExecutor(max_workers=m60.n_workers() * 2) as ex:
        res = list(ex.map(_r, aids))
    return [r[0] for r in res], [r[1] for r in res], [r[2] for r in res]


def stage_render(limit, out_root: Path) -> None:
    manifest_csv = out_root / m60.MANIFEST_CSV.name
    if not manifest_csv.exists():
        raise SystemExit(f"[61] {manifest_csv} missing: run scripts/60_harmonized_manifest.py --stage manifest first")
    mj = json.loads((out_root / m60.MANIFEST_JSON.name).read_text())
    if mj["manifest_csv_sha256"] != m60.sha256_file(manifest_csv):
        raise SystemExit("[61] manifest CSV was modified after the provenance JSON was written (sha256 mismatch)")
    m60.check_prereg_frozen()
    ops = load_ops(manifest_csv)

    aids, nids, texts, _ = m60.read_corpus(limit)
    n = len(texts)
    print(f"[61/render] corpus of record read: {n} docs; writing to {out_root / 'narratives'}", flush=True)

    out = {}
    for arm in ARMS_DET:
        out[arm] = [apply_arm(t, ops[arm], ident=f"{a}/{arm}") for a, t in zip(aids, texts)]
    # budgets for the random arm: per-document n_lines_removed of arm 1 as measured by corpus_diff_counts
    budgets, tot = [], 0
    for t, u in zip(texts, out["korean_mimic_compatible"]):
        cd = c21.corpus_diff_counts(t, u)
        assert cd["n_lines_added"] == 0 and cd["n_lines_removed"] == cd["n_lines_orig"] - cd["n_lines_new"]
        budgets.append(cd["n_lines_removed"])
        tot += cd["n_lines_removed"]
    base = [apply_all_arm_removals(t, ops["__all_arms__"]) for t in texts]
    residual = []
    for t, b, bud in zip(texts, base, budgets):
        pre = len(nonblank(t)) - len(nonblank(b))
        if pre > bud:
            raise SystemExit("[61] Addendum-1 removals exceed a document's primary-arm removal budget")
        residual.append(bud - pre)
    rnd, info = make_random_arm(base, residual)
    info["n_lines_removed_by_addendum1_all_arm_removals"] = int(sum(len(nonblank(t)) - len(nonblank(b)) for t, b in zip(texts, base)))
    info["residual_random_budget_total"] = int(sum(residual))
    out[ARM_RANDOM] = rnd
    tot_rnd = sum(c21.corpus_diff_counts(t, u)["n_lines_removed"] for t, u in zip(texts, rnd))
    if tot_rnd != tot:
        raise SystemExit(f"[61] random arm removed {tot_rnd} lines != arm-1 total {tot}")

    nar = out_root / "narratives"
    for arm in ALL_ARMS:
        write_arm(nar / arm, aids, out[arm])
        print(f"[61/render] wrote {arm}: {n} files", flush=True)
    with open(nar / "row_order.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["embedding_index", "admission_ID", "newpatient_ID"])
        for i, (a, p) in enumerate(zip(aids, nids)):
            w.writerow([i, a, p])
    prov = {
        "stage": "render", "script": "scripts/61_harmonized_render.py", **m60.git_state(["scripts/60_harmonized_manifest.py", "scripts/61_harmonized_render.py"]),
        "sys_executable": sys.executable, "phase": "2 (Addendum 1)",
        "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "limit": limit,
        "n_docs": n, "corpus_of_record": str(REPORTS_DIR), "manifest_csv_sha256": mj["manifest_csv_sha256"],
        "prereg_sha256": mj["prereg_sha256"],
        "script_sha256": {"61_harmonized_render.py": m60.sha256_file(Path(__file__).resolve()),
                          "60_harmonized_manifest.py": m60.sha256_file(Path(m60.__file__).resolve())},
        "addendum_sha256": mj["addendum_sha256"],
        "ops": {a: {"drop_summary": sorted(ops[a]["drop_summary"]),
                    "drop_items": {k: sorted(v) for k, v in ops[a]["drop_items"].items()},
                    "efficacy_lines_dropped": ops[a]["efficacy"], "whole_admission_antibiotic_block_dropped": ops[a]["walk"],
                    "sections_removed_when_empty": sorted(ops[a]["drop_sections"])} for a in ARMS_DET},
        "random_arm": {k: v for k, v in info.items() if k != "per_doc_removed"},
        "random_arm_corpus_wide_lines_removed": tot_rnd, "arm1_corpus_wide_lines_removed": tot,
        "note": "No embedding, clustering, API call or GPU was used.",
    }
    atomic_write_text(out_root / "render_provenance.json", json.dumps(prov, indent=2, default=str))
    print(json.dumps({k: prov[k] for k in ("n_docs", "arm1_corpus_wide_lines_removed", "random_arm_corpus_wide_lines_removed")}, indent=2))


# ----------------------------------------------------------------------------
# QA
# ----------------------------------------------------------------------------
LEAK_PATTERNS = {
    "mortality": r"mortal", "death_died": r"\b(death|died|deceased|dead)\b", "discharge": r"discharg",
    "length_of_stay": r"length[ _-]?of[ _-]?stay|\bLOS\b", "expire": r"expir", "survival": r"surviv",
    "thirty_day": r"30[- ]?day", "dischtime_deathtime": r"dischtime|deathtime|hospital_expire",
}
ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def is_subsequence(small: list[str], big: list[str]) -> bool:
    it = iter(big)
    return all(any(x == y for y in it) for x in small)


def token_lengths(texts: list[str], tok, batch=512):
    out = []
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], add_special_tokens=True, truncation=False)["input_ids"]
        out.extend(len(e) for e in enc)
    return out


def pctl(a, q):
    return float(np.percentile(np.asarray(a, dtype=float), q))


def dist(a):
    return {"median": float(statistics.median(a)), "mean": float(np.mean(a)), "min": float(min(a)), "max": float(max(a)),
            "p5": pctl(a, 5), "p95": pctl(a, 95), "sum": float(np.sum(a))}


def episodes(text: str, section: str):
    blocks, cur, sec = [], None, None
    for ln in text.split("\n"):
        if c21._is_header(ln):
            sec = ln.strip().strip("[]")
            cur = None
            continue
        if sec != section or ln.strip() == "":
            continue
        if c21._is_episode_anchor(ln):
            cur = {"id": ln.split(":", 1)[1].strip()}
            blocks.append(cur)
            continue
        if cur is not None and c21._indent_len(ln) == 0:
            k = c21._line_key(ln.strip())
            if k is not None:
                cur[k] = ln.strip().split(":", 1)[1].strip() if ":" in ln else ""
    return blocks


def derivation_checks(texts):
    flags = ["3GCR", "CarbaR", "CeftriaxoneR", "MRSA", "VRE", "CRAB"]
    tot = collections.Counter()
    for t in texts:
        bc, it = episodes(t, BC), episodes(t, IT)
        it_by = {b["id"]: b for b in it}
        for b in bc:
            tot["episodes"] += 1
            want = " ; ".join(f"{f}_{b.get(f)}" for f in flags)
            tot["ast_summary_tokens_equals_flags"] += int(b.get("AST_summary_tokens") == want)
            ib = it_by.get(b["id"])
            if ib is not None:
                tot["episodes_with_infection_block"] += 1
                tot["microbes_equals_short_clean"] += int(ib.get("Microbes isolated in this episode") == b.get("BSI_episode_short_clean"))
                tot["polymicrobial_yes_iff_short_clean_polymicrobial"] += int(
                    (ib.get("BSI_polymicrobial") == "yes") == (b.get("BSI_episode_short_clean") == "Polymicrobial"))
    eff = collections.Counter()
    for t in texts:
        for ln in t.split("\n"):
            if c21._indent_len(ln) == 4 and "_efficacy" in ln:
                eff[ln.split(":", 1)[1].strip()] += 1
    return {"episode_level_counts": dict(tot), "efficacy_value_counts": dict(eff)}


def stage_qa(limit, out_root: Path) -> None:
    import pandas as pd

    t_start = dt.datetime.now(dt.timezone.utc)
    manifest_csv = out_root / m60.MANIFEST_CSV.name
    mdf = pd.read_csv(manifest_csv, keep_default_na=False)
    ops = load_ops(manifest_csv)
    aids, nids, orig, orig_hash = m60.read_corpus(limit, with_hash=True)
    n = len(orig)
    meta = c18.load_metadata()
    if limit:
        meta = meta.iloc[:limit]
    checks: dict[str, object] = {}
    fails: list[str] = []

    def check(name, ok, detail=None):
        checks[name] = {"pass": bool(ok), **({"detail": detail} if detail is not None else {})}
        if not ok:
            fails.append(name)

    check("corpus_of_record_n_docs", n == (limit or 32329), {"n": n})
    ro = pd.read_csv(out_root / "narratives" / "row_order.csv")
    check("row_order_identical_to_corpus_of_record",
          list(ro["admission_ID"]) == list(meta["admission_ID"]) and list(ro["newpatient_ID"]) == list(meta["newpatient_ID"])
          and list(ro["embedding_index"]) == list(range(n)))

    arm_txt, arm_hash, arm_bytes = {}, {}, {}
    for arm in ALL_ARMS:
        d = out_root / "narratives" / arm
        files = sorted(p.name for p in d.glob("*.txt"))
        check(f"{arm}::file_set_equals_admission_ids", files == sorted(f"{a}.txt" for a in aids), {"n_files": len(files)})
        arm_txt[arm], arm_hash[arm], arm_bytes[arm] = read_arm(d, aids)

    # --- structural checks ----------------------------------------------------
    sk = {}
    for arm in ALL_ARMS:
        ol = [nonblank(t) for t in orig]
        al = [nonblank(t) for t in arm_txt[arm]]
        not_sub = sum(0 if is_subsequence(a, o) else 1 for a, o in zip(al, ol))
        cd = [c21.corpus_diff_counts(o, a) for o, a in zip(orig, arm_txt[arm])]
        added = sum(c["n_lines_added"] for c in cd)
        removed = sum(c["n_lines_removed"] for c in cd)
        dupdocs_before = sum(1 for o in ol if len(set(o)) < len(o))
        dupdocs_after = sum(1 for a in al if len(set(a)) < len(a))
        new_dup = sum(1 for o, a in zip(ol, al) if (len(a) - len(set(a))) > (len(o) - len(set(o))))
        empty_bytes = sum(1 for b in arm_bytes[arm] if b == 0)
        empty_lines = sum(1 for a in al if len(a) == 0)
        content_empty = sum(1 for a in arm_txt[arm] if not any(k == "item" for k, _ in c21.segment(a)))
        check(f"{arm}::order_preserving_subsequence_of_original", not_sub == 0, {"docs_violating": not_sub})
        check(f"{arm}::no_added_lines", added == 0, {"n_lines_added": added})
        check(f"{arm}::no_new_duplicate_lines", new_dup == 0, {"docs_with_more_duplicate_lines_than_original": new_dup,
                                                              "docs_with_any_duplicate_line_before": dupdocs_before,
                                                              "docs_with_any_duplicate_line_after": dupdocs_after})
        check(f"{arm}::no_empty_documents", empty_bytes == 0 and empty_lines == 0, {"zero_byte": empty_bytes, "zero_content_lines": empty_lines})
        checks[f"{arm}::content_empty_documents_info"] = {"docs_with_no_non_anchor_item": content_empty,
                                                          "note": "informational; a document that keeps only headers/anchors"}
        sk[arm] = {"lines_removed_corpus_wide": removed, "cd": cd, "al": al}

    # --- leakage / date regex screens ------------------------------------------
    leak = {}
    base_hits = {k: {i for i, t in enumerate(orig) if re.search(p, t, re.I)} for k, p in LEAK_PATTERNS.items()}
    leak["corpus_of_record_docs_matching"] = {k: len(v) for k, v in base_hits.items()}
    for arm in ALL_ARMS:
        hits = {k: {i for i, t in enumerate(arm_txt[arm]) if re.search(p, t, re.I)} for k, p in LEAK_PATTERNS.items()}
        new = {k: len(hits[k] - base_hits[k]) for k in hits}
        leak[arm] = {"docs_matching": {k: len(v) for k, v in hits.items()}, "new_docs_vs_corpus_of_record": new}
        check(f"{arm}::no_outcome_leakage_tokens_introduced", all(v == 0 for v in new.values()), new)
    checks["leakage_regex_screen"] = {"patterns": LEAK_PATTERNS, "results": leak}

    # --- manifest reconciliation (deterministic arms) --------------------------
    row_keys = {r["field_id"]: r for r in mdf.to_dict("records")}
    orig_ct = collections.Counter(k for t in orig for k in m60.scan_text(t))
    recon = {}
    for arm in ARMS_DET:
        arm_ct = collections.Counter(k for t in arm_txt[arm] for k in m60.scan_text(t))
        bad = []
        pred_removed = 0
        for r in mdf.to_dict("records"):
            if r["retained_" + arm] == "n/a":
                continue
            for k in [x for x in r["korean_line_keys"].split(" ;; ") if x]:
                want = orig_ct[k] if r["retained_" + arm] == "yes" else 0
                if arm_ct[k] != want:
                    bad.append({"row": r["field_id"], "key": k, "orig": orig_ct[k], "arm": arm_ct[k], "expected": want})
                pred_removed += orig_ct[k] - want
        unknown = sorted(k for k in arm_ct if k not in orig_ct)
        check(f"{arm}::manifest_reconciliation_line_counts", not bad and not unknown, {"mismatches": bad[:10], "unknown_keys": unknown[:10]})
        check(f"{arm}::manifest_predicted_removed_equals_observed", pred_removed == sk[arm]["lines_removed_corpus_wide"],
              {"predicted": pred_removed, "observed": sk[arm]["lines_removed_corpus_wide"]})
        recon[arm] = {"arm_line_counts_by_key": dict(arm_ct)}

    # --- explicit programmatic verifications required by the spec --------------
    def per_doc_summary(text):
        lines = text.split("\n")
        sec, toks = None, []
        for ln in lines:
            if c21._is_header(ln):
                sec = ln.strip().strip("[]")
                continue
            if sec == ST and ln.strip():
                toks.append(c21.summary_token_category(ln.strip()))
        return toks

    keep5 = m60.RETAINED_SUMMARY
    drop14 = sorted(set(c21.SUMMARY_TOKEN_CATEGORIES) - set(keep5))
    for arm in ARMS_DET:
        bad5 = bad14 = 0
        for o, a in zip(orig, arm_txt[arm]):
            toks = per_doc_summary(a)
            otoks = per_doc_summary(o)
            if sorted(toks) != sorted(keep5) or toks != [c for c in otoks if c in keep5]:
                bad5 += 1
            if any(c in toks for c in drop14):
                bad14 += 1
        check(f"{arm}::summary_tokens_exactly_the_five_retained_lines_in_original_order", bad5 == 0, {"docs_violating": bad5})
        check(f"{arm}::fourteen_dropped_summary_lines_absent", bad14 == 0, {"dropped_categories": drop14, "docs_violating": bad14})
        has_hdr = sum(1 for a in arm_txt[arm] if a.startswith("[Summary tokens]"))
        check(f"{arm}::summary_tokens_header_present", has_hdr == n, {"docs": has_hdr})
        date_keys = sum(1 for a in arm_txt[arm] for k in m60.scan_text(a) if k.endswith("::Admission Date") or k.endswith("::Test_date"))
        iso = sum(1 for a in arm_txt[arm] if ISO_DATE.search(a))
        check(f"{arm}::no_date_lines_remain", date_keys == 0 and iso == 0, {"date_key_lines": date_keys, "docs_with_iso_date_string": iso})
    # random arm: date lines are NOT dropped by design; report only
    checks[f"{ARM_RANDOM}::date_lines_info"] = {
        "date_key_lines": sum(1 for a in arm_txt[ARM_RANDOM] for k in m60.scan_text(a) if k.endswith("::Admission Date") or k.endswith("::Test_date")),
        "note": "informational; the random arm is field-agnostic"}

    wh = lambda arm: sum(1 for a in arm_txt[arm] for k in m60.scan_text(a) if "::WholeAdm/" in k or k.endswith("<Antibiotics subheader>"))
    ef = lambda arm: sum(1 for a in arm_txt[arm] for k in m60.scan_text(a) if k.endswith("Antibiotic_N_efficacy"))
    orig_wh = sum(orig_ct[k] for k in orig_ct if "::WholeAdm/" in k or k.endswith("<Antibiotics subheader>"))
    orig_ef = orig_ct[f"{IT}::Empiric/Antibiotic_N_efficacy"]
    check("korean_mimic_compatible::whole_admission_antibiotics_block_absent", wh("korean_mimic_compatible") == 0, {"lines": wh("korean_mimic_compatible")})
    check("korean_mimic_compatible::efficacy_lines_absent", ef("korean_mimic_compatible") == 0, {"lines": ef("korean_mimic_compatible")})
    check("korean_mimic_compatible::no_treatments_section_header", not any("[Treatments during admission]" in a for a in arm_txt["korean_mimic_compatible"]))
    check("harmonized_keep_therapy::whole_admission_antibiotics_block_present_unchanged", wh("harmonized_keep_therapy") == orig_wh,
          {"lines": wh("harmonized_keep_therapy"), "orig": orig_wh})
    check("harmonized_keep_therapy::efficacy_lines_absent", ef("harmonized_keep_therapy") == 0,
          {"lines": ef("harmonized_keep_therapy"), "orig": orig_ef})
    check("both_deterministic_arms::efficacy_lines_absent", ef("korean_mimic_compatible") == 0 and ef("harmonized_keep_therapy") == 0,
          {"kmc": ef("korean_mimic_compatible"), "kt": ef("harmonized_keep_therapy"), "orig": orig_ef})
    # --- Addendum 1 decisions 1-2: absent in ALL THREE arms (incl. random)
    ast_lines = lambda arm: sum(1 for a in arm_txt[arm] for k in m60.scan_text(a) if k.startswith(f"{BC}::AST_detailed"))
    mic_lines = lambda arm: sum(1 for a in arm_txt[arm] for k in m60.scan_text(a) if k == f"{IT}::Microbes isolated in this episode")
    orig_ast = sum(v for k, v in orig_ct.items() if k.startswith(f"{BC}::AST_detailed"))
    orig_mic = orig_ct[f"{IT}::Microbes isolated in this episode"]
    for arm in ALL_ARMS:
        check(f"{arm}::AST_detailed_lines_absent", ast_lines(arm) == 0, {"lines": ast_lines(arm), "orig": orig_ast})
        check(f"{arm}::duplicate_Microbes_line_absent", mic_lines(arm) == 0, {"lines": mic_lines(arm), "orig": orig_mic})
        check(f"{arm}::BSI_episode_short_clean_retained_all_episodes",
              sum(1 for a in arm_txt[arm] for k in m60.scan_text(a) if k == f"{BC}::BSI_episode_short_clean") == orig_ct[f"{BC}::BSI_episode_short_clean"]
              if arm != ARM_RANDOM else True)
    ek = lambda arm: sum(1 for a in arm_txt[arm] for k in m60.scan_text(a) if k.endswith("Antibiotic_N_name") and "::Empiric/" in k)
    check("both_harmonized_arms::empiric_antibiotic_names_retained",
          ek("korean_mimic_compatible") == orig_ct[f"{IT}::Empiric/Antibiotic_N_name"] == ek("harmonized_keep_therapy"),
          {"orig": orig_ct[f"{IT}::Empiric/Antibiotic_N_name"], "kmc": ek("korean_mimic_compatible"), "kt": ek("harmonized_keep_therapy")})
    # --- primary vs keep_therapy: differ EXACTLY and ONLY by the whole-admission antibiotics block (Addendum 1 decision 3)
    bad_sub = bad_multi = bad_rev = 0
    diff_lines_total = 0
    for o, a1, a2 in zip(orig, arm_txt["korean_mimic_compatible"], arm_txt["harmonized_keep_therapy"]):
        l1, l2 = nonblank(a1), nonblank(a2)
        if not is_subsequence(l1, l2):
            bad_sub += 1
        ck1, ck2 = collections.Counter(l1), collections.Counter(l2)
        sec, exp = None, collections.Counter()
        for ln in o.split("\n"):
            if ln.strip() == "":
                continue
            if c21._is_header(ln):
                sec = ln.strip().strip("[]")
            k = m60.norm_key(sec, ln)
            if k == f"HEADER::{TR}" or k.endswith("<Antibiotics subheader>") or "::WholeAdm/" in k:
                exp[ln] += 1
        if (ck2 - ck1) != exp:
            bad_multi += 1
        if (ck1 - ck2):
            bad_rev += 1
        diff_lines_total += sum((ck2 - ck1).values())
    check("primary_vs_keep_therapy_differ_exactly_and_only_by_whole_admission_antibiotics_block",
          bad_sub == 0 and bad_multi == 0 and bad_rev == 0,
          {"docs_primary_not_subsequence_of_keep_therapy": bad_sub, "docs_extra_lines_differ_from_expected_block": bad_multi,
           "docs_with_lines_only_in_primary": bad_rev, "total_lines_in_keep_therapy_not_in_primary": diff_lines_total,
           "expected_block_scope": "section header + bare `Antibiotics` subheader + Antibiotic_N_name / duration_days / placeholder lines"})
    diff_kmc_kt = sk["korean_mimic_compatible"]["lines_removed_corpus_wide"] - sk["harmonized_keep_therapy"]["lines_removed_corpus_wide"]
    exp_diff = sum(orig_ct[k] for k in orig_ct if "::WholeAdm/" in k or k.endswith("<Antibiotics subheader>")) + orig_ct[f"HEADER::{TR}"]
    check("harmonized_vs_keep_therapy_line_difference_equals_block_plus_subheader_plus_treatments_header",
          diff_kmc_kt == exp_diff and diff_kmc_kt == diff_lines_total, {"observed": diff_kmc_kt, "expected": exp_diff})

    # --- equivalence of building blocks with 21_common ---------------------------
    eq_bad = ref_bad = split_bad = 0
    for o, a1, a2 in zip(orig, arm_txt["korean_mimic_compatible"], arm_txt["harmonized_keep_therapy"]):
        if join_sections(split_sections(o)) != o:
            split_bad += 1
        secs = split_sections(o)
        walked = join_sections([(h, walk_drop_admission_antibiotics(b) if (h and h.strip() == f"[{TR}]") else b) for h, b in secs])
        mine = c21.remove_summary_categories(c21.remove_items(walked, drop_keys=c21.THERAPY_BLOCK_KEYS), drop_categories=c21.SUMMARY_CAT_THERAPY)
        if mine != c21.remove_therapy(o):
            eq_bad += 1
        if reference_arm(o, "korean_mimic_compatible") != a1 or reference_arm(o, "harmonized_keep_therapy") != a2:
            ref_bad += 1
    check("split_join_roundtrip_identity", split_bad == 0, {"docs_violating": split_bad})
    check("walk_plus_remaining_steps_equals_21_common_remove_therapy", eq_bad == 0, {"docs_violating": eq_bad})
    check("independent_reference_composition_byte_equal_both_deterministic_arms", ref_bad == 0, {"docs_violating": ref_bad})

    # --- random arm ----------------------------------------------------------------
    r_removed = sk[ARM_RANDOM]["lines_removed_corpus_wide"]
    k_removed = sk["korean_mimic_compatible"]["lines_removed_corpus_wide"]
    check("random_arm::corpus_wide_removed_lines_equal_arm1", r_removed == k_removed, {"random": r_removed, "arm1": k_removed})
    per_doc_eq = sum(1 for a, b in zip(sk[ARM_RANDOM]["cd"], sk["korean_mimic_compatible"]["cd"]) if a["n_lines_removed"] == b["n_lines_removed"])
    checks["random_arm::per_document_removed_lines_equal_arm1_info"] = {"docs_equal": per_doc_eq, "docs": n}

    def anchors(text):
        c = collections.Counter()
        for k in m60.scan_text(text):
            if k.startswith("HEADER::") or k.endswith("::BSI_episode_num") or k.endswith("<preamble sentence>") or k.endswith("<Antibiotics subheader>"):
                c[k] += 1
        return c

    anc_bad = sum(1 for o, a in zip(orig, arm_txt[ARM_RANDOM]) if anchors(o) != anchors(a))
    check("random_arm::section_headers_and_episode_anchors_preserved", anc_bad == 0, {"docs_violating": anc_bad})
    budgets = [c["n_lines_removed"] for c in sk["korean_mimic_compatible"]["cd"]]
    ops_all = ops["__all_arms__"]
    base2 = [apply_all_arm_removals(t, ops_all) for t in orig]
    resid2 = [b - (len(nonblank(t)) - len(nonblank(bs))) for b, t, bs in zip(budgets, orig, base2)]
    check("random_arm::residual_budgets_nonnegative", min(resid2) >= 0, {"min_residual": int(min(resid2)), "total_residual": int(sum(resid2))})
    rnd2, info2 = make_random_arm(base2, resid2, SEED_RANDOM)
    # the random arm's per-document budget equals the amended PRIMARY arm's per-document removed-line count
    rem_rnd = [c["n_lines_removed"] for c in sk[ARM_RANDOM]["cd"]]
    n_unequal = sum(1 for a, b in zip(rem_rnd, budgets) if a != b)
    check("random_arm::per_document_budgets_equal_amended_primary_arm",
          n_unequal == 0 if info2["n_topup_lines"] == 0 else n_unequal <= info2["n_topup_lines"],
          {"docs_with_equal_removed_count": n - n_unequal, "docs": n, "n_topup_lines": info2["n_topup_lines"],
           "note": "unequal documents can only be corpus-wide top-up documents; the corpus-wide total is checked separately"})
    h2 = [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in rnd2]
    check("random_arm::deterministic_reproducible_from_seed_20260921", h2 == arm_hash[ARM_RANDOM],
          {"seed": SEED_RANDOM, "bit_generator": info2["bit_generator"], "numpy": info2["numpy"], "n_topup_lines": info2["n_topup_lines"],
           "docs_differing_on_recompute": sum(1 for a, b in zip(h2, arm_hash[ARM_RANDOM]) if a != b)})
    # a different seed must give a different selection (sanity that the seed is actually used)
    rnd3, _ = make_random_arm(base2[:2000], resid2[:2000], SEED_RANDOM + 1)
    check("random_arm::seed_sensitivity_sanity", any(hashlib.sha256(t.encode()).hexdigest() != h for t, h in zip(rnd3, arm_hash[ARM_RANDOM][:2000])))

    # --- per-document sizes and tokens -------------------------------------------
    n_workers = m60.n_workers()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ["RAYON_NUM_THREADS"] = str(n_workers)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tok_report = {}
    doc_stats = pd.DataFrame({"embedding_index": range(n), "admission_ID": aids})
    corp = {"corpus_of_record": orig, **arm_txt}
    for name in ["corpus_of_record"] + ALL_ARMS:
        texts = corp[name]
        doc_stats[f"{name}__n_lines"] = [len(nonblank(t)) for t in texts]
        doc_stats[f"{name}__n_bytes"] = [len(t.encode("utf-8")) for t in texts]
    for tname, spec in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(spec["path"], local_files_only=True)
        # exact-equivalence guard against the pipeline's own counter (embed_common.hf_token_counter)
        ec = importlib.import_module("embed_common")
        cnt = ec.hf_token_counter(tok)
        probe = orig[:40]
        pipe = [cnt(t) for t in probe]
        batched = token_lengths(probe, tok)
        check(f"tokenizer::{tname}::batched_count_equals_pipeline_counter", pipe == batched)
        tok_report[tname] = {"path": spec["path"], "max_seq_length": spec["max_seq_length"], "special_tokens_included": True, "truncation": False}
        for name in ["corpus_of_record"] + ALL_ARMS:
            doc_stats[f"{name}__tokens_{tname}"] = token_lengths(corp[name], tok)
        del tok

    summ = {}
    for name in ["corpus_of_record"] + ALL_ARMS:
        s = {"n_lines": dist(doc_stats[f"{name}__n_lines"]), "n_bytes": dist(doc_stats[f"{name}__n_bytes"])}
        for tname in TOKENIZERS:
            col = doc_stats[f"{name}__tokens_{tname}"]
            s[f"tokens_{tname}"] = dist(col)
            s[f"tokens_{tname}"]["n_over_max_seq_length"] = int((col > TOKENIZERS[tname]["max_seq_length"]).sum())
        summ[name] = s
    for arm in ALL_ARMS:
        rem = doc_stats["corpus_of_record__n_lines"] - doc_stats[f"{arm}__n_lines"]
        summ[arm]["lines_removed_per_doc"] = dist(rem)
        summ[arm]["lines_removed_corpus_wide"] = int(rem.sum())
        summ[arm]["median_lines_per_doc_before_after"] = [summ["corpus_of_record"]["n_lines"]["median"], summ[arm]["n_lines"]["median"]]
        for tname in TOKENIZERS:
            summ[arm][f"median_tokens_{tname}_before_after"] = [summ["corpus_of_record"][f"tokens_{tname}"]["median"],
                                                               summ[arm][f"tokens_{tname}"]["median"]]
        check(f"{arm}::no_truncation_at_model_max_seq_length",
              all(summ[arm][f"tokens_{t}"]["n_over_max_seq_length"] == 0 for t in TOKENIZERS))

    # --- informational: premise of Addendum 1 decision 1 (MIMIC AST_detailed block == a single `o NA` bullet in every doc)
    mimic_dir = REV / "out" / "27_mimic" / "narratives"
    mimic_files = sorted(mimic_dir.glob("*.report.md"))
    n_all_na = 0
    for mp in mimic_files:
        ml = mp.read_text(encoding="utf-8").split("\n")
        try:
            j = next(i for i, x in enumerate(ml) if x.strip() == "AST_detailed :")
        except StopIteration:
            continue
        blk = []
        for x in ml[j + 1:]:
            if x.strip() == "" or not x.startswith(" "):
                break
            blk.append(x.strip())
        n_all_na += int(blk == ["o NA"])
    mimic_ast = {"n_mimic_docs": len(mimic_files), "n_docs_with_block_equal_to_single_o_NA": n_all_na,
                 "premise_holds": bool(len(mimic_files) == 16144 and n_all_na == len(mimic_files)),
                 "source": str(mimic_dir), "note": "informational (Addendum 1 decision 1 cites C_mimic_organism_ablation/render_qa.json for the same fact)"}

    # --- hashes --------------------------------------------------------------------
    rows = []
    for i, a in enumerate(aids):
        rows.append(("corpus_of_record", i, a, f"{nids[i]}_{a}.report.md", orig_hash[i], len(orig[i].encode("utf-8"))))
    for arm in ALL_ARMS:
        for i, a in enumerate(aids):
            rows.append((arm, i, a, f"{a}.txt", arm_hash[arm][i], arm_bytes[arm][i]))
    pd.DataFrame(rows, columns=["arm", "embedding_index", "admission_ID", "file", "sha256", "n_bytes"]).to_csv(out_root / "hashes.csv", index=False)
    doc_stats.to_csv(out_root / "doc_stats.csv", index=False)

    # --- derivation facts on the corpus of record (for the manifest notes) ---------
    deriv = derivation_checks(orig)
    check("derivation::AST_summary_tokens_is_deterministic_function_of_six_flags",
          deriv["episode_level_counts"].get("ast_summary_tokens_equals_flags") == deriv["episode_level_counts"].get("episodes"),
          deriv["episode_level_counts"])

    # 1-doc structural diffs, redacted skeleton only (no values)
    med = doc_stats["corpus_of_record__n_lines"].median()
    sample_idx = int(doc_stats.index[doc_stats["corpus_of_record__n_lines"] == med][0]) if (doc_stats["corpus_of_record__n_lines"] == med).any() else 0
    skeleton = {arm: c21.diff_sample(orig[sample_idx], arm_txt[arm][sample_idx]) for arm in ALL_ARMS}

    qa = {
        "stage": "qa", "script": "scripts/61_harmonized_render.py", **m60.git_state(["scripts/60_harmonized_manifest.py", "scripts/61_harmonized_render.py"]),
        "sys_executable": sys.executable, "phase": "2 (Addendum 1)",
        "mimic_AST_detailed_all_o_NA": mimic_ast,
        "started_utc": t_start.isoformat(timespec="seconds"), "finished_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "limit": limit, "n_docs": n, "corpus_of_record": str(REPORTS_DIR), "manifest_csv_sha256": m60.sha256_file(manifest_csv),
        "script_sha256": {"61_harmonized_render.py": m60.sha256_file(Path(__file__).resolve()),
                          "60_harmonized_manifest.py": m60.sha256_file(Path(m60.__file__).resolve())},
        "all_checks_pass": not fails, "failed_checks": fails, "n_checks": len(checks), "checks": checks,
        "tokenizers": tok_report, "corpus_summaries": summ, "derivation_checks": deriv,
        "lines_removed_by_arm_corpus_wide": {a: summ[a]["lines_removed_corpus_wide"] for a in ALL_ARMS},
        "sample_admission_id_for_structural_diff": int(aids[sample_idx]), "redacted_skeleton_diffs_of_sample": skeleton,
        "no_embedding_clustering_api_or_gpu": True,
    }
    atomic_write_text(out_root / "render_qa.json", json.dumps(qa, indent=1, default=str))
    print(json.dumps({"all_checks_pass": qa["all_checks_pass"], "failed": fails, "n_checks": len(checks),
                      "lines_removed_by_arm": qa["lines_removed_by_arm_corpus_wide"]}, indent=2), flush=True)
    if fails:
        raise SystemExit(f"[61/qa] FAILED checks: {fails}")


# ----------------------------------------------------------------------------
# Review package
# ----------------------------------------------------------------------------
MASK_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
MASK_PID = re.compile(r"(Patient ID : )patient_\d+")


def mask(t: str) -> str:
    return MASK_PID.sub(r"\1patient_<masked>", MASK_DATE.sub("YYYY-MM-DD", t))


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def stage_package(out_root: Path) -> None:
    import pandas as pd
    mdf = pd.read_csv(out_root / m60.MANIFEST_CSV.name, keep_default_na=False)
    mj = json.loads((out_root / m60.MANIFEST_JSON.name).read_text())
    qa = json.loads((out_root / "render_qa.json").read_text())
    rp = json.loads((out_root / "render_provenance.json").read_text())
    ds = pd.read_csv(out_root / "doc_stats.csv")
    ro = pd.read_csv(out_root / "narratives" / "row_order.csv")
    n = len(ro)
    S = qa["corpus_summaries"]
    L = []
    w = L.append
    w("# Analysis A -- review package (Phase 2: manifest and corpora amended by Addendum 1)\n")
    w(f"Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')} from files on disk by `scripts/61_harmonized_render.py --stage package`; "
      f"git `{qa['git_head'][:7]}`{' (dirty)' if qa.get('git_dirty') else ''} (branch `followup-analyses`). This package is produced by the render/QA stage only; that stage performs no embedding, "
      "clustering, API call or GPU work. Nothing below is a result; it describes what was rendered and whether the renderings passed their integrity checks. "
      "The Phase-1 package and corpora are preserved unchanged in `superseded_phase1/`.\n")
    w("Frozen references: `PREREGISTRATION_A.md` (sha256 `%s`, equal to `qc/PREREG_HASHES_AC.txt`), `PREREGISTRATION_A_ADDENDUM_1.md` (sha256 `%s`, equal to `qc/PREREG_HASHES_A_ADDENDUM1.txt`), "
      "`qc/AUTHOR_DECISIONS_2026-09-21.md`.\n" % (mj["prereg_sha256"][:16], mj["addendum_sha256"][:16]))
    w("Files: `harmonized_field_manifest.csv/.json`, `render_qa.json`, `hashes.csv`, `doc_stats.csv`, `narratives/<arm>/<admission_ID>.txt` (row order = `narratives/row_order.csv`, identical to the corpus of record).\n")

    # ---- 1. arms
    w("## 1. The three arms\n")
    w("| arm | what it is | lines removed (corpus-wide) | median lines/doc | median tokens/doc BGE-M3 | median tokens/doc Qwen3 |")
    w("|---|---|---:|---:|---:|---:|")
    cr = S["corpus_of_record"]
    w(f"| corpus of record (`reports_pilot_pre`) | unmodified, 9 sections | 0 | {cr['n_lines']['median']:.0f} | {cr['tokens_bge-m3']['median']:.0f} | {cr['tokens_qwen3-0.6b']['median']:.0f} |")
    desc = {"korean_mimic_compatible": "PRIMARY: MIMIC-renderable field set (D-A-alpha/beta/gamma/delta + Addendum 1)",
            "harmonized_keep_therapy": "companion: identical to the primary arm EXCEPT the whole-admission antibiotics block is kept",
            ARM_RANDOM: "Addendum-1 removals, then random top-level items (seed 20260921) up to the primary arm's per-document removed-line count"}
    for a in ALL_ARMS:
        s = S[a]
        w(f"| `{a}` | {desc[a]} | {s['lines_removed_corpus_wide']:,} | {s['n_lines']['median']:.0f} | {s['tokens_bge-m3']['median']:.0f} | {s['tokens_qwen3-0.6b']['median']:.0f} |")
    w("")
    tot = cr["n_lines"]["sum"]
    w(f"Corpus of record: {n:,} documents, {tot:,.0f} non-blank lines, {cr['n_bytes']['sum'] / 1e6:.1f} MB. Lines are non-blank lines; tokens are counted with the two "
      "tokenizers' own files (special tokens included, no truncation; no model was loaded). Max tokens per document after harmonization: "
      + ", ".join(f"{a}: BGE-M3 {S[a]['tokens_bge-m3']['max']:.0f} / Qwen3 {S[a]['tokens_qwen3-0.6b']['max']:.0f}" for a in ALL_ARMS) + " (limits 8,192 / 32,768; no truncation).\n")
    ar = mj["summary"]["arms"]
    for a in ARMS_DET:
        w(f"- `{a}`: {ar[a]['line_classes_removed']} of {mj['summary']['n_distinct_line_key_classes_observed']} line-key classes removed, "
          f"{ar[a]['line_classes_retained']} retained; sections with removals: {', '.join(ar[a]['sections_with_any_removal'])}; "
          f"section headers removed (Rule R): {', '.join(ar[a]['section_headers_removed']) or 'none'}.")
    w(f"- `{ARM_RANDOM}`: seed {rp['random_arm']['seed']} ({rp['random_arm']['bit_generator']}, numpy {rp['random_arm']['numpy']}); the Addendum-1 removals (AST_detailed, duplicate Microbes line) "
      f"come first ({rp['random_arm']['n_lines_removed_by_addendum1_all_arm_removals']:,} lines), then random items fill the residual budget ({rp['random_arm']['residual_random_budget_total']:,} lines) so that the per-document "
      f"total equals the amended primary arm's removed-line count; top-up lines needed to make the corpus-wide total exact: {rp['random_arm']['n_topup_lines']}; documents whose per-document budget was met exactly: "
      f"{rp['random_arm']['n_docs_with_per_doc_budget_met']:,} / {n:,}. Section headers, episode anchors, the preamble sentence and the bare `Antibiotics` subheader are never removed; "
      "everything else (any top-level rendered item, with its indented continuation lines) is eligible.\n")

    # ---- 2. manifest
    w("## 2. Field manifest (full table: `harmonized_field_manifest.csv`)\n")
    sm = mj["summary"]
    w(f"{sm['n_manifest_rows']} rows: the {sm['n_distinct_line_key_classes_observed']} line-key classes observed in the corpus of record (each covered by exactly one row, "
      "all 63 `raw_fields.parquet` fields assigned) are grouped into "
      f"{sm['n_manifest_rows'] - sum(1 for r in mdf.to_dict('records') if r['row_kind'] == 'mimic_only')} rows, plus {sum(1 for r in mdf.to_dict('records') if r['row_kind'] == 'mimic_only')} rows for MIMIC-only lines with no Korean counterpart. Status counts over rows that exist in the Korean text: "
      + ", ".join(f"{k} {v}" for k, v in sm["harmonization_status_counts_korean_rendered_rows"].items()) + ". "
      "Last column: `S` = Phase-1 flag settled by Addendum 1, `L` = retained as documented limitation (section 5). Retention columns: `Y` retained, `-` removed. Lines = corpus-wide count of that line class in the corpus of record.\n")
    w("| id | Korean field | section | MIMIC equivalent | status | primary | keep_therapy | lines | |")
    w("|---|---|---|---|---|:-:|:-:|---:|:-:|")
    sect_abbr = {ST: "Summary", AD: "Admin", CM: "Comorb.", TR: "Treatments", PR: "Procedures", SO: "SOFA", PH: "Physio/lab", BC: "Bact. culture", IT: "Infect. treatm."}
    short = lambda s, k: (s if len(s) <= k else s[:k - 1] + "…")
    for r in mdf.to_dict("records"):
        if not r["rendered_in_korea"] or r["row_kind"] == "mimic_only":
            continue
        yn = lambda v: "Y" if v == "yes" else ("-" if v == "no" else v)
        w(f"| {r['field_id']} | {short(r['korean_field'], 58)} | {sect_abbr.get(r['korean_section'], r['korean_section'])} | {short(r['mimic_equivalent'], 46)} | "
          f"{r['harmonization_status']} | {yn(r['retained_korean_mimic_compatible'])} | {yn(r['retained_harmonized_keep_therapy'])} | {int(r['n_lines_corpus']):,} | "
          f"{'S' if r['author_confirm'] == 'SETTLED' else ('L' if r['author_confirm'] == 'LIMITATION' else '')} |")
    w("\nMIMIC lines with no Korean line to retain (not added; adding any would be a new computed field):")
    for r in mdf.to_dict("records"):
        if r["row_kind"] == "mimic_only":
            w(f"- {r['field_id']} {r['korean_field']}: {r['mimic_equivalent']} -- {short(r['reason_for_exclusion'], 160)}")
    w("")

    # ---- 3. examples
    w("## 3. Three example documents (full vs `korean_mimic_compatible`)\n")
    w("Chosen by a fixed rule on corpus-of-record non-blank line count: the first document (embedding_index 0), the first document at the median line count, and the longest document "
      "(first occurrence). Patient-ID values and calendar dates are masked in this file only (the corpus files themselves are unchanged); the longest example is cut to its first 70 lines per side.\n")
    lc = ds["corpus_of_record__n_lines"]
    med = lc.median()
    picks = [("first document", 0), ("median-length document", int(ds.index[lc == med][0]) if (lc == med).any() else int((lc - med).abs().idxmin())), ("longest document", int(lc.idxmax()))]
    for label, i in picks:
        a = int(ds.loc[i, "admission_ID"])
        full = (REPORTS_DIR / f"{ro.loc[i, 'newpatient_ID']}_{a}.report.md").read_text(encoding="utf-8", errors="replace")
        kmc = (out_root / "narratives" / "korean_mimic_compatible" / f"{a}.txt").read_text(encoding="utf-8")
        lf, lk = full.rstrip("\n").split("\n"), kmc.rstrip("\n").split("\n")
        cut = ""
        if label == "longest document":
            cut = f"\n… ({len(lf) - 70} more lines)" if len(lf) > 70 else ""
            cut_k = f"\n… ({len(lk) - 70} more lines)" if len(lk) > 70 else ""
            lf, lk = lf[:70], lk[:70]
        else:
            cut_k = ""
        w(f"**{label}** -- embedding_index {i}, admission_ID {a}; non-blank lines {int(lc[i])} -> {int(ds.loc[i, 'korean_mimic_compatible__n_lines'])}; "
          f"tokens BGE-M3 {int(ds.loc[i, 'corpus_of_record__tokens_bge-m3'])} -> {int(ds.loc[i, 'korean_mimic_compatible__tokens_bge-m3'])}, "
          f"Qwen3 {int(ds.loc[i, 'corpus_of_record__tokens_qwen3-0.6b'])} -> {int(ds.loc[i, 'korean_mimic_compatible__tokens_qwen3-0.6b'])}\n")
        w("<table><tr><th>full (corpus of record)</th><th>korean_mimic_compatible</th></tr><tr>"
          f"<td valign=\"top\"><pre>{esc(mask(chr(10).join(lf)))}{esc(cut)}</pre></td>"
          f"<td valign=\"top\"><pre>{esc(mask(chr(10).join(lk)))}{esc(cut_k)}</pre></td></tr></table>\n")

    # ---- 4. QA
    w("## 4. QA results (`render_qa.json`)\n")
    w(f"{qa['n_checks']} checks recorded; **all pass: {qa['all_checks_pass']}**" + ("" if qa["all_checks_pass"] else f" (FAILED: {qa['failed_checks']})") + ".\n")
    w("| check | result |")
    w("|---|---|")
    keyshow = [k for k in qa["checks"] if isinstance(qa["checks"][k], dict) and "pass" in qa["checks"][k]]
    for k in keyshow:
        w(f"| `{k}` | {'PASS' if qa['checks'][k]['pass'] else '**FAIL**'} |")
    w("")
    w("Leakage regex screen (mortality / death / discharge / length-of-stay / expire / survival / 30-day / dischtime): documents matching in the corpus of record: "
      + ", ".join(f"{k} {v}" for k, v in qa["checks"]["leakage_regex_screen"]["results"]["corpus_of_record_docs_matching"].items())
      + "; new matches introduced in any arm: 0 (arms only remove lines; every arm's lines are an order-preserving subsequence of the original).\n")
    dv = qa["derivation_checks"]
    e = dv["episode_level_counts"]
    ma = qa["mimic_AST_detailed_all_o_NA"]
    w(f"Premise of Addendum 1 decision 1, re-checked here on the MIMIC narratives: AST_detailed block equals the single bullet `o NA` in {ma['n_docs_with_block_equal_to_single_o_NA']:,} of {ma['n_mimic_docs']:,} documents (premise holds: {ma['premise_holds']}).\n")
    w("Facts about the corpus of record: "
      f"episodes {e.get('episodes'):,}; `AST_summary_tokens` equals the concatenation of the six flag lines in {e.get('ast_summary_tokens_equals_flags'):,} episodes; "
      f"`Microbes isolated in this episode` equals `BSI_episode_short_clean` in {e.get('microbes_equals_short_clean'):,} of {e.get('episodes_with_infection_block'):,} episodes; "
      f"`BSI_polymicrobial : yes` iff organism is `Polymicrobial` in {e.get('polymicrobial_yes_iff_short_clean_polymicrobial'):,}; "
      f"per-drug efficacy line values {dv['efficacy_value_counts']}.\n")

    # ---- 5. dispositions
    w("## 5. Addendum 1: what changed, what was settled, what remains a limitation\n")
    w("Changes to the frozen text (Addendum 1, section 'Decisions that CHANGE'): (1) `AST_detailed` removed from every arm; (2) duplicate `Microbes isolated in this episode` removed from every arm; "
      "(3) efficacy lines dropped in both deterministic arms, so `harmonized_keep_therapy` differs from the primary arm only by the whole-admission antibiotics block "
      "(verified programmatically: check `primary_vs_keep_therapy_differ_exactly_and_only_by_whole_admission_antibiotics_block`).\n")
    w("| id | field | manifest status | disposition |")
    w("|---|---|---|---|")
    for r in mdf.to_dict("records"):
        if r["author_confirm"] in ("SETTLED", "LIMITATION"):
            w(f"| {r['field_id']} | {short(r['korean_field'], 70)} | {r['harmonization_status']} | {r['author_confirm']}: {r['addendum1_disposition']} |")
    w("")
    w("Documented limitations of Analysis A (non-harmonized differences that remain): Entry mode (`unsafe_to_harmonize`, retained per frozen D-A-delta), Infectious comorbidities, Charlson class, "
      "AST_summary_tokens, BSI_episode_num.\n")
    w("Random-arm construction adopted by Addendum 1: per-document removal budget equals the amended primary arm's per-document removed-line count plus an exact corpus-wide top-up, seed `default_rng(20260921)`, "
      "re-derived after decisions 1-2. Implementation detail worth stating: because decisions 1-2 apply to every arm, the random arm first removes those lines and draws the random items only for the residual budget.\n")
    w("Confirmations still open: none. (Thresholds `frac_k_le_4 <= 0.10` and ARI `>= 0.50` are unchanged, per Addendum 1.)\n")
    (out_root / "REVIEW_PACKAGE.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[61/package] wrote {out_root / 'REVIEW_PACKAGE.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["render", "qa", "package"], required=True)
    ap.add_argument("--limit", type=int, default=None, help="(testing) first N documents only")
    ap.add_argument("--out-root", type=Path, default=A_DIR, help="(testing) override output root")
    args = ap.parse_args()
    if args.limit and args.out_root == A_DIR:
        raise SystemExit("--limit refuses to write into the real output directory; pass --out-root")
    if args.stage == "render":
        stage_render(args.limit, args.out_root)
    elif args.stage == "qa":
        stage_qa(args.limit, args.out_root)
    else:
        stage_package(args.out_root)


if __name__ == "__main__":
    main()
