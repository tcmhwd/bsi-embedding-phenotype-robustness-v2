#!/usr/bin/env python3
"""Task 67 -- Analysis C (FOLLOWUP-2026-09-21-C): MIMIC-IV organism perturbation.
RENDER + QA stage, and the shared helpers used by 67b_C_embed.py / 68_C_cluster.py.

Governing documents (FROZEN, not edited here):
  out/followup_analyses/C_mimic_organism_ablation/PREREGISTRATION_C.md   (sha256 in qc/PREREG_HASHES_AC.txt)
  out/followup_analyses/qc/AUTHOR_DECISIONS_2026-09-21.md
  out/followup_analyses/qc/EXECUTION_PLAN.md  (sections 1, 2, 3, 5 C-1..C-3)

WHAT THIS STAGE DOES
--------------------
Builds the five preregistered arms by LINE-LEVEL SURGERY on the stored MIMIC
narratives of record (out/27_mimic/narratives/*.report.md, 16,144 files) -- the
renderer is NOT re-rendered from a rewritten template.  The five arms remove
exactly these lines of the renderer `27c_mimic_narratives.render_one`:

  unmodified                  none                                   (C0, reference)
  no_organism                 :104                                   (C1, graded companion)
  no_organism_no_resistance   :104 + :105 + :106-107 + :108-112      (C2, PRIMARY organism arm)
  no_charlson                 :78 + :79                              (C3, PRIMARY non-dominant control)
  no_source                   :77                                    (C4, secondary second-shared-field arm)

Line membership is DERIVED, not asserted, in two independent ways that must agree
for every one of the 16,144 documents before anything is written:
  (1) Key matching on the stored text ("Organism summary :", "  o ...", ...), where the
      key -> 27c source line map is obtained by executing `render_one` from an
      ast-instrumented copy of 27c_mimic_narratives.py that records the SOURCE LINE
      NUMBER of every `lines.append(...)` that fires (sentinel row).
  (2) The same instrumented renderer re-rendering every admission from
      admissions_enriched.parquet, which must reproduce the stored text byte-for-byte and
      supplies the 27c line number of every emitted line of every document.

QA (all hard assertions -> non-zero exit, which cancels the `afterok` embedding array):
  identical admission IDs and row order in every arm (= manifest sorted by admission_ID,
  the order 27d embeds in); n = 16,144 and 13,220 unique subjects; unmodified arm
  byte-identical to the stored corpus; each arm == unmodified minus exactly its line set;
  no empty documents; no outcome/leakage terms (mortality/death/discharge/LOS);
  no duplicated lines beyond those already in the unmodified document; edit size per arm;
  sha256 manifest.  Organism-identity leakage: the removed keys, organism names, and
  masked MRSA/VRE/CRAB/3GCR/CarbaR strings are grepped in the RENDERED TEXT of
  no_organism_no_resistance (must be absent) and, for contrast, of no_organism.

Writes ONLY under out/followup_analyses/C_mimic_organism_ablation/ (or --out-root).

Env: /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REV = SCRIPT_DIR.parent
COHORT_DIR = REV / "out/27_mimic/cohort"
STORED_NARR = REV / "out/27_mimic/narratives"
STORED_EMBED = REV / "out/27_mimic/embed"
STORED_LABELS = REV / "out/27_mimic/replicate/labels"
C_ROOT = REV / "out/followup_analyses/C_mimic_organism_ablation"
SRC_27C = SCRIPT_DIR / "27c_mimic_narratives.py"

ENCODERS = ["bge-m3", "qwen3-0.6b"]
SEEDS = [42] + list(range(1, 11))                    # PREREGISTRATION_C.md 3.4
N_EXPECTED = 16144
N_SUBJECTS_EXPECTED = 13220

# Arm -> ranges of 27c_mimic_narratives.py source lines whose emitted output lines are removed.
# Ranges are exactly the preregistration's (section 3.2 table); ranges are resolved to the
# `lines.append` statements they contain by execution, see `trace_renderer`.
ARMS = {
    "unmodified": [],
    "no_organism": [(104, 104)],
    "no_organism_no_resistance": [(104, 104), (105, 105), (106, 107), (108, 112)],
    "no_charlson": [(78, 78), (79, 79)],
    "no_source": [(77, 77)],
}
ARM_ORDER = list(ARMS)
ARM_ROLE = {
    "unmodified": "C0 reference (re-embedding reproduction check)",
    "no_organism": "C1 graded dominant-field removal (Organism summary line only)",
    "no_organism_no_resistance": "C2 PRIMARY organism arm (organism summary + Polymicrobial + 6 resistance flags + AST_detailed block)",
    "no_charlson": "C3 PRIMARY non-dominant control (Charlson category + Charlson score)",
    "no_source": "C4 secondary second-shared-field arm (Source of infection category)",
}

# Outcome / leakage vocabulary that must not appear anywhere in any rendered document.
OUTCOME_RE = re.compile(
    r"mortal|death|died|\bdead\b|expire|deceas|discharge|length of stay|\blos\b|surviv|\balive\b|"
    r"hospital_expire|\bdod\b", re.IGNORECASE)
# Rendered organism identity: the removed keys, organism names.
ORG_KEY_RE = re.compile(
    r"^(Organism summary|Polymicrobial|3GCR|CarbaR|MRSA|VRE|CeftriaxoneR|CRAB|AST_detailed) :", re.MULTILINE)
AST_BULLET_RE = re.compile(r"^  o ", re.MULTILINE)
ORG_NAME_RE = re.compile(
    r"citrobacter|staph|ecoli|e\. ?coli|escherichia|klebsiella|efaecium|efaecalis|enterococc|enterobacter|"
    r"pseudomonas|strepto|serratia|proteus|acinetobacter|stenotrophomonas|morganella|providencia|"
    r"aureus|candida|polymicrobial", re.IGNORECASE)
# The two empiric-coverage spectrum lines legitimately contain the strings MRSA / VRE; mask them
# so the residual grep isolates rendered *culture* resistance-flag text.
SPECTRUM_MASK_RE = re.compile(r"^anti(MRSA|VRE|pseudo)ab_empiric :.*$|^carbaab_empiric :.*$", re.MULTILINE)
RES_FLAG_STR_RE = re.compile(r"MRSA|VRE|CRAB|3GCR|CarbaR|CeftriaxoneR")


# --------------------------------------------------------------------------- shared helpers
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_write_json(obj, dest: Path) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(json.dumps(obj, indent=2, default=str, sort_keys=False))
    os.replace(tmp, dest)


def atomic_write_text(text: str, dest: Path) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(text)
    os.replace(tmp, dest)


def atomic_write_csv(df: pd.DataFrame, dest: Path) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    df.to_csv(tmp, index=False)
    os.replace(tmp, dest)


def atomic_write_parquet(df: pd.DataFrame, dest: Path) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)


# Addendum 1 (out/followup_analyses/C_mimic_organism_ablation/PREREGISTRATION_C_ADDENDUM_1.md, item 2a):
# gross-error detector.  Over the FULL unmodified corpus the minimum cosine between each newly embedded
# vector and the stored (out/27_mimic/embed) vector must be >= 0.99 for both encoders; failure halts the chain.
GROSS_ERROR_COS_MIN = 0.99


def gate_path(root: Path, enc: str) -> Path:
    return Path(root) / f"gross_error_check_{enc}.json"


def require_gate(root: Path, must_pass: list, must_not_have_failed: list = ()) -> None:
    """Halt (SystemExit 4) unless every encoder in `must_pass` has a gross-error JSON with pass=true,
    and no encoder in `must_not_have_failed` has one with pass=false."""
    for enc in must_pass:
        p = gate_path(root, enc)
        if not p.exists():
            raise SystemExit(f"[gate] {p} missing: the unmodified-arm gross-error check for {enc} has not passed; halting")
        g = json.loads(p.read_text())
        if not g.get("pass"):
            raise SystemExit(f"[gate] gross-error check FAILED for {enc} (cosine_min={g.get('cosine_min')}); halting")
    for enc in must_not_have_failed:
        p = gate_path(root, enc)
        if p.exists() and not json.loads(p.read_text()).get("pass"):
            raise SystemExit(f"[gate] gross-error check FAILED for {enc}; halting")


C_FILES = ("scripts/67_C_mimic_variants.py", "scripts/67b_C_embed.py", "scripts/68_C_cluster.py",
           "slurm/67_C_render.sh", "slurm/67_C_embed.sh", "slurm/68_C_cluster.sh", "slurm/68_C_aggregate.sh")


def git_info() -> dict:
    def run(*args) -> str:
        return subprocess.run(["git", "-C", str(REV), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        # dirty flags: (a) any of THIS analysis's own files uncommitted/modified;
        # (b) any tracked file in the repo modified (untracked files ignored).
        "c_analysis_files_dirty": bool(run("status", "--porcelain", "--untracked-files=all", "--",
                                           *[f for f in C_FILES])),
        "tracked_tree_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def package_versions() -> dict:
    out = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "pyarrow", "numba", "umap-learn",
                 "hdbscan", "torch", "transformers", "sentence-transformers", "threadpoolctl"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def env_record() -> dict:
    return {
        "sys_executable": sys.executable,
        "versions": package_versions(),
        "hostname": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "thread_pins": {k: os.environ.get(k) for k in
                        ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        **git_info(),
    }


def load_manifest() -> pd.DataFrame:
    """Cohort manifest in the EXACT row order 27d_mimic_embed.py embeds (sorted admission_ID)."""
    m = pd.read_parquet(COHORT_DIR / "narrative_manifest.parquet").sort_values("admission_ID").reset_index(drop=True)
    assert len(m) == N_EXPECTED, len(m)
    assert m["admission_ID"].is_unique
    assert m["subject_id"].nunique() == N_SUBJECTS_EXPECTED, m["subject_id"].nunique()
    for aid, rf in zip(m["admission_ID"], m["report_file"]):
        assert Path(rf).name == f"{aid}.report.md", (aid, rf)
    return m


def arm_doc_path(root: Path, arm: str, admission_id: str) -> Path:
    return root / "narratives" / arm / f"{admission_id}.report.md"


def load_mimic_fields() -> pd.DataFrame:
    """The MIMIC structured frame used for field attribution, rebuilt exactly as
    scripts/40_field_dominance.py main() (lines 236-249) builds it, then sorted by admission_ID
    (the embedding row order)."""
    keys = ["subject_id", "hadm_id"]
    mimic = pd.read_parquet(COHORT_DIR / "narrative_manifest.parquet")
    for filename, wanted in [
        ("admissions_enriched.parquet", ["age_at_admission", "sex", "polymicrobial", "fungemia"]),
        ("comorbidities.parquet", ["charlson_category", "source_category"]),
        ("ast_detail.parquet", ["BSI_3gcr", "BSI_mrsa", "BSI_vre", "BSI_carbar"]),
        ("empiric_antibiotics.parquet",
         ["antiMRSAab_empiric", "antiVREab_empiric", "antipseudoab_empiric", "carbaab_empiric"]),
    ]:
        table = pd.read_parquet(COHORT_DIR / filename)
        mimic = mimic.merge(table[keys + wanted], on=keys, how="left", validate="one_to_one")
    return mimic.sort_values("admission_ID").reset_index(drop=True)


# --------------------------------------------------------------------------- renderer tracing
def trace_renderer():
    """Compile an ast-instrumented copy of 27c_mimic_narratives.py in which every
    `lines.append(X)` statement is preceded by `_TRACE.append(<source line number>)`.
    Executing `render_one` from it yields, for every emitted line, the 27c source line that
    emitted it.  The original file is not modified; the instrumented copy is never written."""
    src = SRC_27C.read_text()
    tree = ast.parse(src)

    class T(ast.NodeTransformer):
        def visit_Expr(self, node):
            self.generic_visit(node)
            v = node.value
            if (isinstance(v, ast.Call) and isinstance(v.func, ast.Attribute) and v.func.attr == "append"
                    and isinstance(v.func.value, ast.Name) and v.func.value.id == "lines"):
                rec = ast.Expr(value=ast.Call(
                    func=ast.Attribute(value=ast.Name(id="_TRACE", ctx=ast.Load()), attr="append", ctx=ast.Load()),
                    args=[ast.Constant(value=node.lineno)], keywords=[]))
                ast.copy_location(rec, node)
                return [rec, node]
            return node

    tree = T().visit(tree)
    ast.fix_missing_locations(tree)
    ns = {"__name__": "c27_traced", "__file__": str(SRC_27C), "_TRACE": []}
    exec(compile(tree, str(SRC_27C), "exec"), ns)          # noqa: S102 -- own repo source
    return ns


def traced_render(ns, row) -> tuple[str, list[int]]:
    ns["_TRACE"].clear()
    text = ns["render_one"](row)
    trace = list(ns["_TRACE"])
    return text, trace


def append_linenos_in_ranges(ns_src_lines: list[int], ranges: list[tuple[int, int]]) -> set[int]:
    return {ln for ln in ns_src_lines if any(lo <= ln <= hi for lo, hi in ranges)}


def line_key(line: str) -> str | None:
    """Structural key of one rendered line: None for blank, '[Section]' for headers,
    '  o' for AST bullets, else the text before ' :'."""
    s = line.rstrip("\n")
    if s == "":
        return None
    if s.startswith("[") and s.endswith("]"):
        return s
    if s.startswith("  o "):
        return "  o"
    if " :" in s:
        k = s.split(" :")[0]
        return re.sub(r"^Antibiotic_\d+_name$", "Antibiotic_N_name", k)
    raise ValueError(f"unparseable line {s!r}")


def build_key_to_linenos(ns) -> tuple[dict, list[dict]]:
    """Sentinel render -> {line key: set(27c source line numbers that emit it)} and the
    ordered evidence table (output line, 27c line)."""
    import pandas as _pd
    row = _pd.Series({
        "age_at_admission": "SENT", "sex": "SENT", "entry_mode": "SENT", "fungemia": "SENT",
        "source_category": "SENT", "charlson_category": "SENT", "charlson_score": "SENT", "icu": "SENT",
        "WBC_Count": "SENT", "T_Bilirubin": "SENT", "Creatinine": "SENT", "Hemoglobin": "SENT",
        "Albumin": "SENT", "PLT_Count": "SENT",
        "antiMRSAab_empiric": "SENT", "antiVREab_empiric": "SENT", "antipseudoab_empiric": "SENT",
        "carbaab_empiric": "SENT",
        "index_ts": _pd.Timestamp("2000-01-03"), "admittime": _pd.Timestamp("2000-01-01"),
        "organism_category": "SENT", "polymicrobial": "SENT",
        "BSI_3gcr": "SENT", "BSI_carbar": "SENT", "BSI_mrsa": "SENT", "BSI_vre": "SENT",
        "BSI_ceftriaxoner": "SENT", "BSI_crab": "SENT",
        "ast_detail": ["a", "b"], "n_bsi_episodes": "SENT", "empiric_antibiotics": ["x", "y"],
    })
    text, trace = traced_render(ns, row)
    lines = text.split("\n")[:-1] if text.endswith("\n") else text.split("\n")
    assert len(lines) == len(trace), (len(lines), len(trace))
    k2l: dict = {}
    table = []
    for ln_text, ln_src in zip(lines, trace):
        k = line_key(ln_text)
        table.append({"rendered_line": ln_text, "key": k, "27c_source_line": ln_src})
        if k is not None:
            k2l.setdefault(k, set()).add(ln_src)
    # keys emitted by an alternative branch (single-antibiotic NA, empty AST) are not needed:
    # no arm removes them.
    return k2l, table


def lines_to_remove_by_key(lines: list[str], remove_keys: set) -> list[int]:
    return [i for i, l in enumerate(lines) if line_key(l) in remove_keys]


# --------------------------------------------------------------------------- render stage
def stage_render(out_root: Path, limit: int | None = None) -> None:
    man = load_manifest()
    if limit:
        man = man.iloc[:limit].copy()
    ids = man["admission_ID"].tolist()
    n = len(ids)
    print(f"[67/render] n_docs={n} out_root={out_root}", flush=True)

    ns = trace_renderer()
    k2l, sentinel_table = build_key_to_linenos(ns)
    all_append_linenos = sorted({ln for s in k2l.values() for ln in s})
    src_lines = SRC_27C.read_text().split("\n")

    # ---- arm -> 27c append lines -> keys (re-derived from the renderer, then compared with the prereg)
    arm_linenos, arm_keys, arm_evidence = {}, {}, {}
    for arm, ranges in ARMS.items():
        lns = append_linenos_in_ranges(all_append_linenos, ranges)
        # a key is removed iff ALL source lines that can emit it are in the arm's set
        keys = {k for k, s in k2l.items() if s and s <= lns}
        partial = {k for k, s in k2l.items() if s & lns and not s <= lns}
        assert not partial, f"{arm}: key emitted by both removed and kept source lines: {partial}"
        arm_linenos[arm], arm_keys[arm] = sorted(lns), keys
        arm_evidence[arm] = {
            "prereg_27c_ranges": [f":{lo}" if lo == hi else f":{lo}-{hi}" for lo, hi in ranges],
            "resolved_27c_lines_that_append": sorted(lns),
            "removed_line_keys": sorted(keys),
            "source_text_of_resolved_lines": {str(l): src_lines[l - 1].strip() for l in sorted(lns)},
        }
    # the preregistration's claims, checked against the derivation
    assert arm_linenos["no_organism"] == [104]
    assert arm_linenos["no_source"] == [77]
    assert arm_linenos["no_charlson"] == [78, 79]
    assert arm_linenos["no_organism_no_resistance"] == [104, 105, 107, 110, 112], arm_linenos["no_organism_no_resistance"]
    assert arm_keys["no_organism"] == {"Organism summary"}
    assert arm_keys["no_source"] == {"Source of infection category"}
    assert arm_keys["no_charlson"] == {"Charlson category", "Charlson score"}
    assert arm_keys["no_organism_no_resistance"] == {
        "Organism summary", "Polymicrobial", "3GCR", "CarbaR", "MRSA", "VRE", "CeftriaxoneR", "CRAB",
        "AST_detailed", "  o"}, arm_keys["no_organism_no_resistance"]

    # ---- (2) independent derivation: re-render every admission with the traced renderer
    adm = pd.read_parquet(COHORT_DIR / "admissions_enriched.parquet")
    adm["admittime"] = pd.to_datetime(adm["admittime"])
    adm["index_ts"] = pd.to_datetime(adm["index_ts"])
    adm["admission_ID"] = "mimic_" + adm["subject_id"].astype(str) + "_" + adm["hadm_id"].astype(str)
    adm = adm.set_index("admission_ID")
    assert set(ids) <= set(adm.index)

    arm_lineset = {a: set(v) for a, v in arm_linenos.items()}
    stored, arm_text = {}, {a: {} for a in ARM_ORDER}
    stats = {a: [] for a in ARM_ORDER}                 # per-doc records
    n_render_mismatch = 0
    n_index_mismatch = 0
    line_count_before = []
    for i, aid in enumerate(ids):
        raw = Path(man["report_file"].iloc[i]).read_bytes()
        text = raw.decode("utf-8")
        stored[aid] = raw
        lines = text.splitlines(keepends=True)
        assert "".join(lines) == text
        line_count_before.append(len(lines))
        # (2) traced re-render must reproduce the stored text and carry a 27c line number per line
        rtext, rtrace = traced_render(ns, adm.loc[aid])
        if rtext != text:
            n_render_mismatch += 1
        rlines = rtext.splitlines(keepends=True)
        for arm in ARM_ORDER:
            rm_key = set(lines_to_remove_by_key(lines, arm_keys[arm]))                  # (1)
            rm_trace = {j for j, ln in enumerate(rtrace) if ln in arm_lineset[arm]}       # (2)
            if rm_key != rm_trace:
                n_index_mismatch += 1
            kept = [l for j, l in enumerate(lines) if j not in rm_key]
            arm_text[arm][aid] = "".join(kept)
            stats[arm].append({
                "admission_ID": aid, "n_lines_before": len(lines), "n_lines_after": len(kept),
                "n_lines_removed": len(rm_key), "n_chars_before": len(text),
                "n_chars_after": len("".join(kept)),
                "removed_keys_present": sorted({line_key(lines[j]) for j in rm_key}),
            })
    assert n_render_mismatch == 0, f"{n_render_mismatch} stored narratives differ from the re-render by render_one"
    assert n_index_mismatch == 0, f"{n_index_mismatch} (doc,arm) removal sets differ between key-match and 27c line trace"
    print("[67/render] stored corpus reproduced by traced renderer; key-match == 27c-line-trace removal sets "
          f"for all {n} docs x {len(ARM_ORDER)} arms", flush=True)

    # ---- every removable key occurs exactly once per stored doc (AST bullets >= 1)
    expected_once = ["Organism summary", "Polymicrobial", "3GCR", "CarbaR", "MRSA", "VRE", "CeftriaxoneR", "CRAB",
                     "AST_detailed", "Charlson category", "Charlson score", "Source of infection category"]
    for aid in ids:
        c = Counter(line_key(l) for l in stored[aid].decode("utf-8").splitlines(keepends=True))
        for k in expected_once:
            assert c.get(k, 0) == 1, (aid, k, c.get(k, 0))
        assert c.get("  o", 0) >= 1, aid

    # ---- structural QA per arm
    qa_arms = {}
    for arm in ARM_ORDER:
        st = pd.DataFrame(stats[arm])
        texts = [arm_text[arm][a] for a in ids]
        empty = sum(1 for t in texts if not t.strip())
        dup_arm = dup_unmod = doc_extra_dup = 0
        for aid, t in zip(ids, texts):
            def ndup(s):
                nb = [l for l in s.split("\n") if l.strip()]
                return sum(v - 1 for v in Counter(nb).values() if v > 1)
            da, du = ndup(t), ndup(stored[aid].decode("utf-8"))
            dup_arm += da
            dup_unmod += du
            doc_extra_dup += int(da > du)
        outcome_hits = sum(1 for t in texts if OUTCOME_RE.search(t))
        # arm text must equal unmodified minus exactly the arm's lines (subsequence property)
        for aid in ids:
            u = stored[aid].decode("utf-8").splitlines(keepends=True)
            a = arm_text[arm][aid].splitlines(keepends=True)
            it = iter(u)
            assert all(any(x == y for y in it) for x in a), (arm, aid)
        if arm == "unmodified":
            assert all(arm_text[arm][a].encode("utf-8") == stored[a] for a in ids), "unmodified != stored corpus"
        qa_arms[arm] = {
            "role": ARM_ROLE[arm],
            "n_docs": n, "n_empty_documents": empty,
            "n_docs_with_outcome_or_leakage_term": outcome_hits,
            "duplicated_nonblank_lines_total": dup_arm, "duplicated_nonblank_lines_total_in_unmodified": dup_unmod,
            "n_docs_with_more_duplicates_than_unmodified": doc_extra_dup,
            "lines_per_doc_before": _dist(st["n_lines_before"]),
            "lines_per_doc_after": _dist(st["n_lines_after"]),
            "lines_removed_per_doc": _dist(st["n_lines_removed"]),
            "chars_per_doc_before": _dist(st["n_chars_before"]),
            "chars_per_doc_after": _dist(st["n_chars_after"]),
            "chars_removed_per_doc": _dist(st["n_chars_before"] - st["n_chars_after"]),
            "fraction_of_lines_removed_median": float((st["n_lines_removed"] / st["n_lines_before"]).median()),
            "27c_line_evidence": arm_evidence[arm],
        }
        assert empty == 0 and outcome_hits == 0 and doc_extra_dup == 0, (arm, empty, outcome_hits, doc_extra_dup)

    # ---- organism-identity leakage grep on the RENDERED TEXT of each arm
    leak = {}
    for arm in ARM_ORDER:
        texts = [arm_text[arm][a] for a in ids]
        masked = [SPECTRUM_MASK_RE.sub("", t) for t in texts]
        leak[arm] = {
            "docs_with_organism_or_resistance_or_AST_key_line": sum(1 for t in texts if ORG_KEY_RE.search(t)),
            "docs_with_AST_bullet_line": sum(1 for t in texts if AST_BULLET_RE.search(t)),
            "docs_with_organism_name_token": sum(1 for t in texts if ORG_NAME_RE.search(t)),
            "docs_with_MRSA_VRE_CRAB_3GCR_CarbaR_string_after_masking_empiric_spectrum_lines":
                sum(1 for t in masked if RES_FLAG_STR_RE.search(t)),
            "docs_with_MRSA_VRE_CRAB_3GCR_CarbaR_string_unmasked": sum(1 for t in texts if RES_FLAG_STR_RE.search(t)),
            "docs_with_resistance_flag_line_equal_yes": sum(
                1 for t in texts if re.search(r"^(3GCR|CarbaR|MRSA|VRE|CeftriaxoneR|CRAB) : yes$", t, re.MULTILINE)),
        }
    for k in ("docs_with_organism_or_resistance_or_AST_key_line", "docs_with_AST_bullet_line",
              "docs_with_organism_name_token",
              "docs_with_MRSA_VRE_CRAB_3GCR_CarbaR_string_after_masking_empiric_spectrum_lines"):
        assert leak["no_organism_no_resistance"][k] == 0, ("LEAK in no_organism_no_resistance", k,
                                                           leak["no_organism_no_resistance"][k])
    # the contrast the preregistration (H-C3) relies on: no_organism still renders organism-defining lines
    assert leak["no_organism"]["docs_with_organism_or_resistance_or_AST_key_line"] == n
    assert leak["no_organism"]["docs_with_AST_bullet_line"] == n
    print("[67/render] leakage grep: no_organism_no_resistance clean; QA assertions passed", flush=True)

    # ---- separability / residual organism-correlated fields, from STRUCTURED data only
    frame = load_mimic_fields()
    if limit:
        frame = frame[frame["admission_ID"].isin(ids)].reset_index(drop=True)
    resid = residual_organism_correlates(frame)

    # ---- write arms + hashes + manifest
    rows, arm_manifest = [], {}
    for arm in ARM_ORDER:
        d = out_root / "narratives" / arm
        d.mkdir(parents=True, exist_ok=True)
        h_all = hashlib.sha256()
        for aid in ids:
            b = arm_text[arm][aid].encode("utf-8")
            p = arm_doc_path(out_root, arm, aid)
            # write-if-different, direct write: on this filesystem create+rename of ~80k small files
            # dominated the run time.  A partial file left by a killed run differs from `b` and is
            # rewritten on re-run; every consumer re-verifies each file against hashes.csv anyway.
            if not (p.exists() and p.read_bytes() == b):
                p.write_bytes(b)
            hh = sha256_bytes(b)
            h_all.update(f"{aid}\t{hh}\n".encode())
            rows.append({"arm": arm, "admission_ID": aid, "sha256": hh, "n_bytes": len(b)})
        arm_manifest[arm] = {"dir": str(d), "n_docs": n, "corpus_sha256": h_all.hexdigest(),
                             "role": ARM_ROLE[arm]}
    hdf = pd.DataFrame(rows)
    atomic_write_csv(hdf, out_root / "narratives" / "hashes.csv")
    # verify what was written
    for arm in ARM_ORDER:
        chk = sum(1 for aid in ids if arm_doc_path(out_root, arm, aid).read_bytes() ==
                  arm_text[arm][aid].encode("utf-8"))
        assert chk == n, (arm, chk)

    rec = env_record()
    qa = {
        "task": "67_C_mimic_variants render+QA", "prereg": "PREREGISTRATION_C.md",
        "written_before_any_embedding_or_clustering_result": True,
        "n_docs": n, "n_unique_subjects": int(man["subject_id"].nunique()),
        "row_order": "narrative_manifest.parquet sorted by admission_ID (27d_mimic_embed.py:81)",
        "admission_id_order_identical_across_arms": True,
        "unmodified_arm_byte_identical_to_stored_corpus": True,
        "stored_corpus_reproduced_by_instrumented_render_one": {"n_docs": n, "n_mismatch": n_render_mismatch},
        "removal_set_key_match_vs_27c_line_trace": {"n_doc_arm_pairs": n * len(ARM_ORDER), "n_disagreements": n_index_mismatch},
        "renderer_source_line_map_from_sentinel_render": sentinel_table,
        "arms": qa_arms,
        "organism_identity_leakage_grep_of_rendered_text": leak,
        "residual_organism_correlated_fields_structured_data": resid,
        "outcome_regex": OUTCOME_RE.pattern,
        "organism_name_regex": ORG_NAME_RE.pattern,
        "edit_sizes_not_size_matched_note": "Arms are NOT size-matched: see lines_removed_per_doc per arm.",
        "arm_manifest": arm_manifest,
        **rec,
    }
    atomic_write_json(qa, out_root / "render_qa.json")
    atomic_write_json({"arms": arm_manifest, "n_docs": n, "hashes_csv": str(out_root / "narratives" / "hashes.csv"),
                       **rec}, out_root / "render_manifest.json")
    print(f"[67/render] wrote {out_root / 'render_qa.json'}", flush=True)
    for arm in ARM_ORDER:
        e = qa_arms[arm]["lines_removed_per_doc"]
        print(f"[67/render] {arm:28s} lines removed/doc min={e['min']} median={e['median']} max={e['max']} "
              f"corpus_sha256={arm_manifest[arm]['corpus_sha256'][:16]}", flush=True)


def _dist(s) -> dict:
    s = pd.Series(s)
    return {"min": float(s.min()), "median": float(s.median()), "mean": round(float(s.mean()), 4),
            "max": float(s.max())}


def residual_organism_correlates(frame: pd.DataFrame) -> dict:
    """AMI between organism_category and every MIMIC field, from the STRUCTURED cohort data
    (no perturbation result involved), with whether the field is still rendered in
    no_organism_no_resistance.  Documents what organism information the arm does NOT remove."""
    from sklearn.metrics import adjusted_mutual_info_score
    s40 = importlib.import_module("40_field_dominance")
    fields = s40.build_fields(frame, s40.MIMIC_CATEGORICAL, s40.MIMIC_CONTINUOUS)
    org = frame["organism_category"].astype(str).to_numpy()
    rendered_key = {
        "organism_category": "Organism summary", "care_unit": None, "admit_period": None,
        "n_bsi_episodes": "Number of BSI episodes this admission", "sex": "Sex", "icu": "icu",
        "charlson_category": "Charlson category", "source_category": "Source of infection category",
        "BSI_3gcr": "3GCR", "BSI_mrsa": "MRSA", "BSI_vre": "VRE", "BSI_carbar": "CarbaR",
        "polymicrobial": "Polymicrobial", "fungemia": "Fungemia",
        "antiMRSAab_empiric": "antiMRSAab_empiric", "antiVREab_empiric": "antiVREab_empiric",
        "antipseudoab_empiric": "antipseudoab_empiric", "carbaab_empiric": "carbaab_empiric",
        "age_at_admission (decile)": "Age",
    }
    removed_c2 = {"Organism summary", "Polymicrobial", "3GCR", "CarbaR", "MRSA", "VRE", "CeftriaxoneR", "CRAB",
                  "AST_detailed"}
    out = {}
    for name, vals in fields.items():
        key = rendered_key.get(name)
        out[name] = {
            "ami_with_organism_category": round(float(adjusted_mutual_info_score(org, vals.to_numpy())), 6),
            "rendered_line_key": key,
            "rendered_in_unmodified": key is not None,
            "rendered_in_no_organism_no_resistance": key is not None and key not in removed_c2,
        }
    ct = pd.crosstab(frame["organism_category"], frame["fungemia"].astype(str))
    return {
        "note": ("Structured-data descriptive check made BEFORE any perturbation result. Fields still rendered "
                 "in no_organism_no_resistance that correlate with organism (fungemia flag, empiric spectrum "
                 "flags, antibiotic names) are NOT removed by the preregistered arm."),
        "per_field": out,
        "organism_x_fungemia_counts": {str(o): {str(c): int(v) for c, v in row.items()} for o, row in ct.iterrows()},
        "antibiotic_name_lines_rendered": "Antibiotic_N_name lines (27c:117-123) are rendered in every arm "
                                          "and are not organism-identity lines",
    }


# --------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["render"], required=True)
    ap.add_argument("--out-root", default=str(C_ROOT))
    ap.add_argument("--limit", type=int, default=None, help="smoke-test only: first N admissions")
    args = ap.parse_args()
    if args.stage == "render":
        stage_render(Path(args.out_root), args.limit)


if __name__ == "__main__":
    main()
