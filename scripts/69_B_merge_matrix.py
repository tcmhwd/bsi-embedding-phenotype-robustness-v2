#!/usr/bin/env python3
"""Analysis B, supporting deliverable -- the completed Figure-3 perturbation matrix.

Merges, into ONE new table, the arms that already sit in Figure 3 with the arms added by Analysis B:

  existing : out/33_figures/ablation.csv                                   (120 rows; the file Figure 3 is drawn from)
  new      : out/followup_analyses/B_symmetric_perturbation/perturbation_matrix.csv      (3-small B1a / B1b / B2; 33 rows)
             out/followup_analyses/B_symmetric_perturbation/perturbation_matrix_B3.csv   (Qwen3 no_department; 11 rows)

Writes (never overwrites an input):
  out/followup_analyses/B_symmetric_perturbation/perturbation_matrix_completed.csv
  out/followup_analyses/B_symmetric_perturbation/perturbation_matrix_completed_summary.md      (numbers only)
  out/followup_analyses/B_symmetric_perturbation/perturbation_matrix_completed_provenance.json (hashes, counts, git)

Nothing is recomputed: every ARI / k value is copied from its source row.  The harmonisation is
  * encoder labels are those of ablation.csv ("OpenAI 3-large", "OpenAI 3-small", "Qwen3-0.6B");
  * `role` is assigned from an explicit (encoder, variant) lookup that reproduces the role assignment used to
    colour Figure 3 in scripts/36_figures_v4.R (role_xcat), with the Qwen3 first-antibiotic mask split out as its own role
    ("targeted mask") -- ablation.csv's own `arm` label is kept verbatim in `arm_label_in_source` because it is
    encoder-blind (it labels 3-large `no_therapy` "dominant block removed" although that is a non-dominant control for 3-large);
  * `source` = "existing" | "new".
Columns absent from ablation.csv (AMI, noise fraction, ...) are left empty for the existing rows -- nothing is imputed.

Env: /data/tcmhwd/miniconda3/envs/bsi_pheno/bin/python3   (pandas / numpy only; no clustering, no embedding)
"""
import os

for _key in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REV = Path(__file__).resolve().parent.parent
B_DIR = REV / "out/followup_analyses/B_symmetric_perturbation"
EXISTING = REV / "out/33_figures/ablation.csv"
NEW_FILES = [B_DIR / "perturbation_matrix.csv", B_DIR / "perturbation_matrix_B3.csv"]
OUT_CSV = B_DIR / "perturbation_matrix_completed.csv"
OUT_MD = B_DIR / "perturbation_matrix_completed_summary.md"
OUT_JSON = B_DIR / "perturbation_matrix_completed_provenance.json"
THIS_FILES = ["scripts/69_B_merge_matrix.py", "slurm/69_B_merge_matrix.sh"]

ROLES = ("unmodified", "dominant-field perturbation", "non-dominant control", "targeted mask")
# (encoder, variant) -> role.  Mirrors 36_figures_v4.R::role_xcat; the mask arm is its own role here.
ROLE_LOOKUP = {
    ("OpenAI 3-large", "full"): "unmodified",
    ("OpenAI 3-large", "no_department"): "non-dominant control",
    ("OpenAI 3-large", "no_therapy"): "non-dominant control",
    ("OpenAI 3-large", "no_organism"): "dominant-field perturbation",
    ("OpenAI 3-small", "full"): "unmodified",                        # NEW (B1a, B1b)
    ("OpenAI 3-small", "no_organism"): "non-dominant control",
    ("OpenAI 3-small", "no_department"): "dominant-field perturbation",
    ("OpenAI 3-small", "no_therapy"): "non-dominant control",        # NEW (B2)
    ("Qwen3-0.6B", "full"): "unmodified",
    ("Qwen3-0.6B", "no_organism"): "non-dominant control",
    ("Qwen3-0.6B", "no_therapy"): "dominant-field perturbation",
    ("Qwen3-0.6B", "no_department"): "non-dominant control",         # NEW (B3)
    ("Qwen3-0.6B", "mask_ab1_name_only"): "targeted mask",
}
# What the new files' own `arm` column must say for the role above (consistency check, not an input to `role`).
NEW_ARM_EXPECTED = {"non-dominant control": "other field removed", "unmodified": "unmodified"}

COLS = ["encoder", "dominant_field", "arm_id", "variant", "role", "role_basis", "source", "source_file",
        "arm_label_in_source", "seed", "k", "ari_vs_reference", "ami_vs_reference", "noise_fraction",
        "largest_cluster_share", "degenerate_k_lt_5", "n", "n_both_nonnoise", "agreement_convention",
        "reference", "reference_k", "reference_noise", "dominant_field_name", "dominant_field_ami",
        "organism_field_name", "organism_ami", "top_field", "top_field_ami", "runner_up_field",
        "dominance_margin", "edit_lines_removed_total", "edit_lines_removed_per_doc", "embedding_source",
        "source_git_commit"]
ROLE_BASIS = "encoder x variant lookup reproducing 36_figures_v4.R::role_xcat (mask arm split out)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict:
    def run(*args) -> str:
        return subprocess.run(["git", "-C", str(REV), *args], capture_output=True, text=True, check=True).stdout.strip()

    st = run("status", "--porcelain", "--", *THIS_FILES)
    return {"git_commit": run("rev-parse", "HEAD"), "git_branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": bool(st), "git_dirty_detail": st.splitlines()[:20], "git_dirty_scope": THIS_FILES}


def atomic_write(text: str, dest: Path) -> None:
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(text)
    os.replace(tmp, dest)


def fmt(x, nd=4) -> str:
    return "NA" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main() -> None:
    for p in [EXISTING, *NEW_FILES]:
        if not p.exists():
            raise SystemExit(f"[69] missing input {p}")
    for p in (OUT_CSV, OUT_MD, OUT_JSON):
        if p.exists():
            print(f"[69] note: {p.name} exists and will be replaced atomically (it is an output of this script)", flush=True)

    ex = pd.read_csv(EXISTING)
    new_parts = [pd.read_csv(p) for p in NEW_FILES]
    n_in = {"existing": len(ex), **{p.name: len(d) for p, d in zip(NEW_FILES, new_parts)}}

    # ------------------------------------------------------------------ existing rows
    need = {"encoder", "dominant_field", "variant", "seed", "k", "ari_vs_reference", "arm"}
    assert need <= set(ex.columns), f"ablation.csv lacks {need - set(ex.columns)}"
    ex_h = pd.DataFrame({
        "encoder": ex["encoder"], "dominant_field": ex["dominant_field"],
        "arm_id": "existing_" + ex["variant"], "variant": ex["variant"],
        "source": "existing", "source_file": "out/33_figures/ablation.csv",
        "arm_label_in_source": ex["arm"], "seed": ex["seed"], "k": ex["k"],
        "ari_vs_reference": ex["ari_vs_reference"],
        "agreement_convention": "both_nonnoise",            # 33_figure_data.py::fig3_ablation docstring: "Every ARI here is noise-excluded"
    })

    # ------------------------------------------------------------------ new rows
    nw = pd.concat(new_parts, ignore_index=True)
    assert set(nw["analysis"]) == {"B"}
    assert (nw["agreement_convention"] == "both_nonnoise").all()
    nw_h = nw.rename(columns={"arm": "arm_label_in_source", "git_commit": "source_git_commit"}).copy()
    nw_h["source"] = "new"
    nw_h["source_file"] = np.concatenate([[f"out/followup_analyses/B_symmetric_perturbation/{p.name}"] * len(d)
                                          for p, d in zip(NEW_FILES, new_parts)])
    nw_h = nw_h.drop(columns=["analysis", "role"])          # verbose free-text role of the source is replaced by `role`

    # encoder-label harmonisation is by construction: refuse anything that is not an ablation.csv label
    enc_ok = set(ex["encoder"])
    bad_enc = sorted(set(nw_h["encoder"]) - enc_ok)
    if bad_enc:
        raise SystemExit(f"[69] encoder labels not in ablation.csv naming: {bad_enc}")
    dom_map = ex.drop_duplicates("encoder").set_index("encoder")["dominant_field"].to_dict()
    mism = nw_h[nw_h["dominant_field"] != nw_h["encoder"].map(dom_map)]
    if len(mism):
        raise SystemExit(f"[69] dominant_field naming differs from ablation.csv in {len(mism)} new rows")

    merged = pd.concat([ex_h, nw_h], ignore_index=True)
    merged["role"] = [ROLE_LOOKUP.get((e, v)) for e, v in zip(merged["encoder"], merged["variant"])]
    if merged["role"].isna().any():
        raise SystemExit("[69] unmapped (encoder, variant): "
                         + str(merged.loc[merged["role"].isna(), ["encoder", "variant"]].drop_duplicates().values.tolist()))
    merged["role_basis"] = ROLE_BASIS
    # consistency of the new files' own arm label with the assigned role
    nrows = merged[merged["source"] == "new"]
    bad_role = nrows[[NEW_ARM_EXPECTED.get(r) != a for r, a in zip(nrows["role"], nrows["arm_label_in_source"])]]
    if len(bad_role):
        raise SystemExit(f"[69] {len(bad_role)} new rows whose source `arm` label disagrees with the assigned role")

    merged["seed"] = merged["seed"].astype(int)
    merged["k"] = merged["k"].astype(int)
    for c in COLS:
        if c not in merged.columns:
            merged[c] = np.nan
    merged = merged[COLS]
    enc_order = {"OpenAI 3-large": 0, "OpenAI 3-small": 1, "Qwen3-0.6B": 2}
    role_order = {r: i for i, r in enumerate(ROLES)}
    merged = merged.assign(_e=merged["encoder"].map(enc_order), _r=merged["role"].map(role_order))
    merged = merged.sort_values(["_e", "_r", "variant", "arm_id", "seed"], kind="stable").drop(columns=["_e", "_r"]).reset_index(drop=True)

    # ------------------------------------------------------------------ integrity checks
    checks = {}
    checks["rows_input_existing"] = n_in["existing"]
    checks["rows_input_new"] = int(sum(len(d) for d in new_parts))
    checks["rows_expected_merged"] = checks["rows_input_existing"] + checks["rows_input_new"]
    checks["rows_merged"] = int(len(merged))
    checks["rows_match"] = checks["rows_merged"] == checks["rows_expected_merged"]
    checks["rows_by_source"] = merged["source"].value_counts().to_dict()
    dup = merged.duplicated(["encoder", "arm_id", "seed"]).sum()
    checks["duplicate_encoder_arm_seed_keys"] = int(dup)
    checks["encoder_labels"] = sorted(merged["encoder"].unique())
    checks["encoder_labels_subset_of_ablation_csv"] = set(merged["encoder"]) <= enc_ok
    # value-level copy check: every existing row's (encoder, variant, seed) -> ari must be identical to the source
    src = ex.set_index(["encoder", "variant", "seed"])["ari_vs_reference"]
    got = merged[merged["source"] == "existing"].set_index(["encoder", "variant", "seed"])["ari_vs_reference"]
    checks["existing_ari_max_abs_diff_vs_source"] = float((got.sort_index() - src.sort_index()).abs().max())
    srcn = nw.set_index(["encoder", "arm_id", "seed"])["ari_vs_reference"]
    gotn = merged[merged["source"] == "new"].set_index(["encoder", "arm_id", "seed"])["ari_vs_reference"]
    checks["new_ari_max_abs_diff_vs_source"] = float((gotn.sort_index() - srcn.sort_index()).abs().max())
    checks["seeds_per_arm"] = {f"{e}|{a}": sorted(g["seed"].tolist()) for (e, a), g in merged.groupby(["encoder", "arm_id"])}
    checks["n_seeds_per_arm"] = {k: len(v) for k, v in checks["seeds_per_arm"].items()}
    checks["role_counts_rows"] = merged["role"].value_counts().to_dict()
    checks["source_arm_label_vs_role_disagreements_existing"] = (
        merged[merged["source"] == "existing"].groupby(["encoder", "variant", "arm_label_in_source", "role"]).size()
        .reset_index(name="rows").to_dict("records"))
    assert checks["rows_match"] and dup == 0
    assert checks["existing_ari_max_abs_diff_vs_source"] == 0.0 and checks["new_ari_max_abs_diff_vs_source"] == 0.0

    # ------------------------------------------------------------------ summary (numbers only)
    lines = ["# perturbation_matrix_completed -- summary (numbers only)", "",
             f"rows merged = {len(merged)} (existing {checks['rows_by_source'].get('existing', 0)}, new {checks['rows_by_source'].get('new', 0)}); "
             f"inputs: ablation.csv {n_in['existing']}, perturbation_matrix.csv {n_in['perturbation_matrix.csv']}, "
             f"perturbation_matrix_B3.csv {n_in['perturbation_matrix_B3.csv']}",
             "ARI = adjusted Rand index, both-non-noise convention, against each encoder's own stored reference partition. "
             "Noise = fraction of rows labelled -1 (recorded for new arms only; NA where the source file has no such column).", "",
             "| encoder | role | arm_id | variant | source | n seeds | ARI median | ARI min | ARI max | k median | noise median |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (e, r, a), g in merged.groupby(["encoder", "role", "arm_id"], sort=False):
        nz = g["noise_fraction"].astype(float)
        lines.append(f"| {e} | {r} | {a} | {g['variant'].iloc[0]} | {g['source'].iloc[0]} | {len(g)} | "
                     f"{fmt(g['ari_vs_reference'].median())} | {fmt(g['ari_vs_reference'].min())} | {fmt(g['ari_vs_reference'].max())} | "
                     f"{fmt(float(g['k'].median()), 1)} | {fmt(nz.median() if nz.notna().any() else None)} |")
    lines += ["", "seeds per arm: " + "; ".join(f"{k} n={v}" for k, v in checks["n_seeds_per_arm"].items()), ""]
    atomic_write("\n".join(lines) + "\n", OUT_MD)

    tmp = OUT_CSV.with_name(f".tmp_{os.getpid()}_{OUT_CSV.name}")
    merged.to_csv(tmp, index=False)
    os.replace(tmp, OUT_CSV)

    prov = {"written": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "script": "scripts/69_B_merge_matrix.py", "sys_executable": sys.executable, "python": platform.python_version(),
            "pandas": pd.__version__, "numpy": np.__version__, **git_info(),
            "inputs": {str(p.relative_to(REV)): {"sha256": sha256(p), "rows": n} for p, n in
                       zip([EXISTING, *NEW_FILES], [n_in["existing"], *[len(d) for d in new_parts]])},
            "outputs": {str(p.relative_to(REV)): {"sha256": sha256(p)} for p in (OUT_CSV, OUT_MD)},
            "checks": checks, "columns": COLS, "role_lookup": {f"{k[0]}|{k[1]}": v for k, v in ROLE_LOOKUP.items()}}
    atomic_write(json.dumps(prov, indent=2, default=str), OUT_JSON)
    print(json.dumps({"rows_merged": checks["rows_merged"], "rows_by_source": checks["rows_by_source"],
                      "rows_match": checks["rows_match"], "n_seeds_per_arm": checks["n_seeds_per_arm"],
                      "outputs": [str(OUT_CSV), str(OUT_MD), str(OUT_JSON)]}, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
