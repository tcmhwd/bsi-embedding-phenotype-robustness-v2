# Representation dependence in EHR embedding-based phenotyping

Analysis code for the manuscript *"Clinical phenotypes derived from electronic
health record embeddings depend on representation choice"* (bloodstream
infection cohorts, South Korea two-hospital corpus and a MIMIC-IV comparator),
manuscript1009.

This is **v2** of this code release. An earlier v1 release of this repository
was withdrawn after a curation error that included content that should not
have been public; both the GitHub repository and the Zenodo record for v1
have been fully deleted. v2 is a smaller, curated subset
of the private analysis repository containing only the code that computes a
number, figure panel, or table cell actually reported in manuscript1009,
plus the shared library modules and inputs those producer scripts require to
run. It does not contain patient-level data, embeddings, or any credentialed
dataset.

## What is here

- `scripts/` (117 top-level files, plus `scripts/R/house_style.R` in a
  subdirectory — 118 total) — the subset of the private analysis repository's
  scripts that computes a reported manuscript1009 value, plus the shared
  library modules those scripts import and the input files they require.
  This is **not** a full pipeline from raw data to narrative text: see
  "Narrative serialisation" below for what is specifically excluded and why.
  What it does cover: the reconstruction and seed/thread-sensitivity checks
  on the originally derived reference partition; the six-axis stress-testing
  framework (S1-S6) and artifact screen; the E. coli within-organism
  repetition; structured-data clustering-free comparators; cross-encoder
  concordance; the MIMIC-IV comparator pipeline; field-alignment (AMI)
  analyses; the dominant-field perturbation / targeted-ablation analyses
  (Analyses A-D); the 90-day mortality nearest-neighbour retrieval analysis;
  Table 1 assembly; and the figure-data and figure-drawing scripts for every
  panel of Figures 1-4 and Supplementary Figures 1-5.
- `scripts/R/house_style.R` — the shared figure house-style library the two
  figure-drawing scripts `source()`. See "Figures" below for where to put it.
- `aux_scripts/` (8 files) — load-bearing code that, in the private working
  repository, lived under its `out/` working-output tree rather than under
  `scripts/`. Only one of these four scripts (`a_variant_geometry.py`) is
  loaded this way — at runtime, via `importlib.util.spec_from_file_location`,
  by the kept producer script `26b_variant_geometry_all.py`. The other three
  (`b_variant_seeds.py`, `d_gower_index_only.py`, and
  `aux_scripts/12_stack_check/c1_cluster.py`) are not imported by any other
  script; each runs standalone via its own SLURM wrapper in the same
  directory. See "Release edits" below.
- `slurm/` (78 files) — SLURM submission wrappers for the reported runs.
  They encode the resource requests, conda environment, and run order for
  the scripts they wrap; they are not required to re-run a script
  interactively, only to reproduce how the reported run was originally
  executed. Not every kept script has a wrapper — see "Run order" below for
  the scripts that ran directly as light steps with no SLURM submission.
- `tests/` (12 files + `conftest.py`) — unit tests for the core library
  modules (`01_build_analysis_frame.py` through `08_transport.py`,
  `17e_aggregate_metrics.py`, `18f_findings.py`, `embed_common.py`,
  `embed_variants_openai.py`). Most are self-contained unit tests against
  synthetic fixtures; `test_01_frame.py` is an integration test against the
  real analysis frame and is skipped automatically when that restricted data
  is not present (see "Running the tests" below).
- `environments/` — exported conda package lists for the four environments
  used (see "Environments" below — **this distinction matters and caused a
  real reproducibility incident during this project**).
- `env_constraints.txt` — the two version pins (`torch`, `numpy`) that were
  hand-constrained for the general analysis environment.
- `REPRODUCIBILITY_MAP.md` — the item -> script -> output -> environment
  cross-reference, the condensed deviations-from-pre-specification account,
  and the known provenance gaps. Read this first if you are trying to
  connect a manuscript number back to the code that produced it.

## What is deliberately not here

- Raw/intermediate outputs (`out/`), model caches, logs, and rendered figure
  files. They are either too large, derived entirely from the scripts here,
  or not eligible for redistribution.
- **Manuscript-production tooling**: the explainer/HTML-dossier builders,
  the DOCX/Source-Data conversion and relocation scripts, and word-count
  scripts. No reported number is computed by any of these; they only
  reformat already-computed values for journal submission. The journal's
  own Source Data files ship with the article itself.
- **Superseded analysis lines**: an earlier pre-reconstruction analysis line
  (run against a from-scratch "A0" refit before the reference partition was
  reconstructed bit-for-bit) and superseded versions of the figure scripts
  (only the final `..._v8_...R` scripts behind the published figures are
  kept).
- **Internal post hoc QA and dead-end re-checks** whose findings were not
  carried into manuscript1009 (several were relocated into the journal's
  Source Data or dropped during revision).
- **A test fixture containing real-patient-derived narrative text.** Two
  modules in the private repository (an earlier ablation/template-sensitivity
  analysis line) and their tests contained a verbatim de-identified case
  narrative used as a test fixture. Those modules compute no manuscript1009
  value and are dropped in their entirety along with their tests, so no
  synthetic replacement fixture is needed in this release. All 12 kept
  tests were independently re-checked and contain no real-looking patient or
  episode identifier and no narrative text (synthetic IDs such as
  `f"A{i:04d}"` and category labels such as `"Ecoli"` only).
- The internal stress-testing pre-specification document. It is cited
  throughout the kept code and would be the single most useful document for
  a reviewer, but is withheld from this release pending the authors'
  separate archival deposition of it (the manuscript states it will be
  cited there by DOI once deposited).
- `AUTHOR_DECISIONS.md` (an internal revision-log file) — not shipped.

The repository targets **verification of reported results**, not a
general-purpose re-run of the whole analysis from raw clinical records.

## Narrative serialisation is not reproducible from this repository

The South Korea cohort's de-identified case-narrative text (the corpus of
record consumed by every embedding script, referred to in the code as
`reports_pilot_pre/`, dated 2025-12-02) was produced by an earlier pipeline
stage whose renderer source code was not retained — only a compiled
`.pyc` file remains, which is not usable to regenerate narratives from raw
fields and is not included here. The original `text-embedding-3-large`
embedding of that corpus, and the original UMAP/HDBSCAN derivation of the
16-cluster reference partition, also came from that earlier pipeline.
`16_reproduce_original_env.py` re-derives the reference partition
bit-for-bit **from the already-embedded, stored vectors** (ARI = AMI =
1.000 against the originally derived labels) — it does not regenerate the
narratives or the embeddings themselves. Everything downstream of the
stored embeddings (clustering, stress-testing, perturbation, retrieval,
figures) is reproducible from this repository; narrative generation from
raw structured fields is not.

## Environments — read this before trying to reproduce anything

**Four** separate conda environments were used (the v1 release's README
stated two and was wrong about which analyses ran under which one):

- **F — the frozen environment**, `/home/programs/anaconda3/envs/gpt`
  (`umap-learn==0.5.9.post2`, `numpy==2.2.6`, `scipy==1.15.x`,
  `hdbscan==0.8.39` — see `environments/conda_frozen_reconstruction_export.txt`).
  Required for exact reconstruction of the originally derived 16-cluster
  reference partition: a *different* environment with the same informal
  name but `umap-learn==0.5.7` and `numpy==1.26.4` exists on the originating
  cluster and silently produces a different clustering solution (k=8
  instead of k=16) from the same stored embeddings and the same seed — this
  cost about three months of failed reproduction attempts during this
  project before being diagnosed. That other environment is **F' below** —
  it is not a reproduction hazard to avoid, it is itself the deliberately-run
  comparison arm for one specific reported result (ST1a's "software-stack-only
  change" row); see F' for which. **F is also required for every analysis
  that refits UMAP/HDBSCAN for a reported clustering comparison** — not only
  the reconstruction itself — specifically: the S1-S5 and S3 stress-testing
  battery, the E. coli within-organism repetition of S1/S2/S4/S5, the
  one-hot structured-data clustering comparator, every variant/perturbation
  clustering fit (the dominant-field perturbation analyses, Qwen3/3-large/
  3-small ablation arms, Analysis B/B3), and `18e_recluster_frozen.py`'s
  per-encoder whole-corpus reference partitions.
- **F' — the non-frozen `gpt` environment**, `/data/tcmhwd/miniconda3/envs/gpt`
  (`umap-learn==0.5.7`, `numpy==1.26.4` — see
  `environments/conda_gpt_nonfrozen_export.txt`). This is the *different*
  environment referenced in F's entry above: same informal name (`gpt`), same
  originating cluster, but a different conda prefix and a different
  software stack that silently clusters the same stored embeddings into
  k=8 instead of k=16. It is shipped here, distinct from F's export, because
  it is not merely a historical hazard — it is the environment that actually
  produced ST1a's "Software-stack-only change" row (k=8, ARI≈0.487) in
  `REPRODUCIBILITY_MAP.md` (array tasks 21-22 of
  `aux_scripts/12_stack_check/c1_array.sh`); every other task in that
  wrapper runs under **B** (`bsi_pheno`). Do not confuse F and F' — they
  share a name and nothing else.
- **B — `bsi_pheno`** (`torch==2.5.1`, `numpy==1.26.4`; see
  `environments/conda_bsi_pheno_export.txt`). Used for embedding generation
  (OpenAI and local BGE-M3/Qwen3), the MIMIC-IV pipeline, the cross-encoder
  *matched-fit* comparisons (`audit6_matched_encoders.py`,
  `27e_mimic_replicate.py`, `63_harmonized_matched.py`, `68_C_cluster.py`,
  `73_D_cluster.py` — these do not need bit-for-bit reconstruction fidelity
  against the original partition, only an internally consistent new fit),
  field-alignment analyses, the mortality retrieval analysis, and all
  figure-data assembly steps.
- **R — `rnaseq_r`** (see `environments/conda_rnaseq_r_export.txt`). Used
  only by the two figure-drawing scripts, `85_v8_figures_main.R` and
  `86_v8_figure3_supp456.R`.

`REPRODUCIBILITY_MAP.md`'s item table gives the environment for every
producer script. A few scripts are dual-use depending on the `--stage`
argument they are invoked with (e.g. `49_3small_no_department.py` and
`65_B_variants.py` run in B for `--stage embed` and in F for
`--stage cluster`); this is noted in the map.

## Figures

`85_v8_figures_main.R` and `86_v8_figure3_supp456.R` both call
`source("~/R/house_style.R")` — **do not edit that line**; instead copy
`scripts/R/house_style.R` from this repository to `~/R/house_style.R` on
the machine that will run them. Both scripts also require a fontconfig file
at `out/followup_analyses/fonts/fonts.conf` pointing at a bundled Liberation
Sans / Mono substitute (metric-compatible with Arial / Courier New on a
compute node that has no Arial installed) — this fontconfig file and font
bundle are not shipped with this repository; see `REPRODUCIBILITY_MAP.md`
G4. Both scripts save PNG (300 dpi) and SVG for every panel.

## Run order

By analysis block (matching the groupings in `REPRODUCIBILITY_MAP.md`'s item
table); within a block, run in the listed order.

1. **Frame**: `01_build_analysis_frame.py` (no wrapper; light pandas step).
2. **Reference-partition reconstruction and seed sensitivity**:
   `16_reproduce_original_env.sh` -> `16_reproduce_original_env_array.sh` ->
   `16_thread_sensitivity_array.sh` -> `aux_scripts/12_stack_check/c1_array.sh`
   -> `16_hdbscan_sweep.py` (no wrapper) -> `16_summarize_sweep.py` (no
   wrapper) -> `20e_task3_seed_basins.py` (no wrapper).
3. **Stress-testing S1-S5, S3**: `17a_s1_seed_array.sh`,
   `17b_s2_bootstrap_array.sh`, `29_n7_s4s5_transport_fixed.sh` (the
   reported S4/S5 run — not the superseded `17c_s4s5_transport_array.sh`,
   which is not in this release), `48_s3_config_sensitivity.sh`, then
   `17d_token_lookup.py` (no wrapper) and `17e_aggregate_metrics.py` (no
   wrapper; see `REPRODUCIBILITY_MAP.md` G1 for the rerun-invocation gap).
4. **S6 / encoders**: `convert_bge_m3_safetensors.sh` ->
   `18a_step0_validate_corpus.py` (no wrapper; gates `18b`/`18c`) ->
   `18b_embed_open_gpu.sh`, `18c_embed_openai_small.sh` ->
   `18d_s6_knn.sh` -> `18e_recluster_frozen.sh` -> `18f_findings.py` (no
   wrapper) -> `20d_task2_extended_lookup.py` (no wrapper) ->
   `28b_s6_three_readings.py` (no wrapper, via `slurm/`-less run).
5. **One-hot structured comparator**: `20a_parse_template_fields.py` (no
   wrapper) -> `20b_recluster_onehot.sh` -> `20c_task1_report.py` (no
   wrapper).
6. **E. coli repetition**: `22b_ecoli_s1_array.sh`, `22c_ecoli_s2_array.sh`,
   `22d_ecoli_s4s5_array.sh` -> `22e_ecoli_aggregate.py` (no wrapper).
7. **Artifact screen / Methods value**: `22f_artifact_screen.py` (no
   wrapper) and `78_provenance_checks.sh`.
8. **Structured-data comparators (ST4)**: `22g0_gower_precompute.sh` ->
   `22g1_point_fast_array.sh`/`22g1_point_kproto.sh` ->
   `22g2_bootstrap_array.sh` -> `22g3_assemble_table.py` (no wrapper) ->
   `22h_published_baseline_rows.py` (no wrapper).
9. **Cross-encoder concordance**: `audit6_matched_encoders.sh` ->
   `audit6_concordance_whole.py` (no wrapper) ->
   `audit7_dominance_checks.sh` -> `73_D_cluster.sh`.
10. **MIMIC-IV pipeline**: `27a_mimic_cohort.sh` ->
    `27b_mimic_abx.sh`/`27b_mimic_ast.sh`/`27b_mimic_comorbid.sh`/
    `27b_mimic_labs.sh`/`27b_mimic_merge.sh` -> `27c_mimic_narratives.sh` ->
    `27d_mimic_embed_bge_m3.sh`/`27d_mimic_embed_qwen3.sh` ->
    `27e_mimic_replicate.sh`.
11. **Field alignment**: `40_field_dominance.py` (no wrapper; imported by
    several later scripts) and `audit7_dominance_checks.sh` (block 9).
12. **Dominant-field perturbation / Analyses B, B3, D**:
    `21a_generate_variants.sh` -> `21b_embed_openai.sh` ->
    `21c_cluster_frozen.sh` -> `26f_seed_sweep_remaining.sh` (plus
    `aux_scripts/25_audit4/b.sh` for the first six variants) ->
    `26g_aggregate.py` (no wrapper) -> `26b_variant_geometry_all.sh` (runs
    `aux_scripts/25_audit4/a_variant_geometry.py` at runtime) ->
    `45_3large_seeds.sh` -> `audit10_no_organism_seeds.py` (no wrapper) ->
    `audit7_make_variant.py` (no wrapper) -> `audit7_embed_qwen3.sh` ->
    `audit7_cluster_variants.sh` -> `44_ablation_cluster.sh` ->
    `47_qwen3_extra_seeds.sh` -> `audit9_mask_embed.sh` ->
    `audit9_cluster_mask.sh` -> `audit9_knn_purity.sh` ->
    `49_3small_embed.sh`/`49_3small_cluster.sh` -> `65_B_embed.sh` ->
    `65b_B3_embed.sh` -> `66_B_cluster.sh` -> `66b_B3_cluster.sh` ->
    `69_B_merge_matrix.sh` -> `73_D_cluster.sh` (block 9).
13. **Analysis A (MIMIC-compatible restriction)**:
    `60_61_harmonized_A_phase2_render.sh` -> `62_harmonized_A_embed.sh` ->
    `63_harmonized_A_matched.sh` -> `64_harmonized_A_aggregate.sh` ->
    `70_A_rescore_korea_full.sh` -> `76_mimic_doc_stats.sh`.
14. **Analysis C (MIMIC organism perturbation)**: `67_C_render.sh` ->
    `67_C_embed.sh` -> `68_C_cluster.sh` -> `68_C_aggregate.sh`.
15. **Mortality retrieval**: `23b_retrieval_array.sh`,
    `23c_run_anchors.sh` -> `23e_assemble_table.py` (no wrapper) ->
    `23g_neighborhood_balance_array.sh` -> `23h_assemble_neighborhood_balance.py`
    (no wrapper) -> `24c_arm3_sofa_anchor.sh` -> `26e_sofa_stratified.sh`
    -> `28a_openpool_query_mask.sh` -> `50_retrieval_bootstrap.sh` ->
    `50b_baseline_bootstrap.sh`.
16. **Table 1**: `50_cohort_table.py` (no wrapper).
17. **Figure data and figures**: `33_figure_data.py` (no wrapper) ->
    `36_figure_data_v4.py` (no wrapper) -> `71_followup_figure_data.sh` ->
    `74_v6_figure_data.sh` -> `75_fix_runnerup.sh` ->
    `80_v7_figure3_supp456.py` (no wrapper; verified byte-copy
    `tables_v6/` -> `tables_v7/`) -> `85_v8_figures_main.sh` ->
    `86_v8_figure3_supp456.sh`.

## Release edits

Five documented edits were made when curating this release from the private
analysis repository. No reported numeric value is affected by any of them.

- **E1 — follow-up namespace renamed.** The private repository's working
  namespace for the post hoc Analyses A-D (an internal codename used as a
  directory component, a git branch name, and two internal analysis-ID
  strings) has been renamed throughout the kept scripts and wrappers to
  `out/followup_analyses/...` for public release — the original internal
  name does not appear anywhere in this repository. Sub-folder names under
  that namespace
  (`A_harmonized_template/`, `B_symmetric_perturbation/`,
  `C_mimic_organism_ablation/`, `D_alt_pipeline_optional/`, `qc/`,
  `tables*/`, `figures*/`, `fonts/`, `logs/`) are unchanged. **Consequence:**
  paths in this release no longer match the on-disk layout of the private
  repository; reproducing a run inside that private environment would need
  a local symlink from the old path to the new one (outside the scope of
  this public release).
- **E2 — comments, docstrings and user-facing messages translated to
  English.** No code path, number, threshold, identifier, or control flow
  was changed. Comment-only mentions that identified a specific prior
  peer-review round by reviewer letter and comment number, or that named
  the prior submission journal, were also generalised to "(prior review
  comment)" / "the authors' earlier supplementary material" in the same
  pass, to avoid disclosing this manuscript's submission history to a new
  journal's reviewers.
- **E3 — test import paths made portable.** All 12 kept tests previously
  hardcoded `sys.path.insert(0, "<private-repo-path>/scripts")`; this is
  replaced by a portable `SCRIPTS_DIR = Path(__file__).resolve().parent.parent
  / "scripts"` computed at test time (also provided via a new
  `tests/conftest.py`), including inside the three tests that embed the
  path a second time in a `subprocess.run([python, "-c", ...])` code string.
- **E4 — a manuscript-package-copy step was removed from two figure
  wrappers.** `85_v8_figures_main.sh` and `86_v8_figure3_supp456.sh`
  originally also copied their rendered figures, under a renumbered
  filename, into an internal manuscript-production package directory.
  That block has been removed from both wrappers in this release (it is
  manuscript-production packaging, not analysis, and it referenced an
  internal folder name); both wrappers still run the figure script itself
  and still verify that every expected output file was written.
- **E5 — references to specific internal code-review/audit tools
  generalised.** Comments, docstrings and one data file
  (`prespec_thresholds.json`) that named specific internal code-review or
  audit tooling by name have been reworded to generic terms ("internal
  review", "an internal audit pass") in this release; no threshold, code
  path, or logic was changed. The source directory
  `aux_scripts/12_codex_c1/` (named after one such tool) was renamed to
  `aux_scripts/12_stack_check/` for the same reason, and every reference to
  that directory elsewhere in this repository (this README, the run order
  above, `REPRODUCIBILITY_MAP.md`) was updated to match. This rename is
  shipped-source-location only: internal output-path strings baked into the
  scripts themselves (e.g. `out/12_codex_c1/...`, which is not shipped with
  this repository — see `REPRODUCIBILITY_MAP.md` G3) were left unedited,
  consistent with this repository's general policy of shipping code paths
  unmodified except for these documented release edits.

## Paths

Scripts and SLURM wrappers in this repository were written for the
originating HPC environment and contain absolute paths rooted at that
cluster's filesystem layout. They are included as-written (not
parameterised) so that the exact logic used to produce the reported results
is preserved unmodified — **except** for the five edits listed above (E1-E5),
none of which changes any code path, logic, or reported value. Reproducing a
run elsewhere requires adjusting these paths to point at your own copy of
the (not included) input data.

Code comments throughout this release also cite internal project documents
that are not shipped with this repository (internal design/decision/
provenance Markdown files, and a handful of superseded script filenames
dropped from this release) — these are historical cross-references left in
place for context, not links to anything included here or retrievable
elsewhere. Relatedly, `71_followup_figure_data.py`, `74_v6_figure_data.py`,
and `80_v7_figure3_supp456.py` each hardcode a `THIS_FILES`-style list of
sibling filenames (including some figure-drawing/wrapper scripts) that is
passed to `git status --porcelain --` for their own provenance record; some
of those listed filenames were dropped from this v2 release, so the
provenance JSON these three scripts produce will reference paths that do
not exist here. This is not a crash — `git status` tolerates unmatched
pathspecs — just a disclosed provenance-record artifact.

## API keys

Scripts that call the OpenAI API (`text-embedding-3-large`,
`text-embedding-3-small`) read the key from the `OPENAI_API_KEY` environment
variable. If that variable is not set, `embed_variants_openai.py` (and the
scripts built on it — `18c_embed_openai_small.py`, `21b_embed_openai.py`,
`49_3small_no_department.py`, `65_B_variants.py`) **fall back to reading a
key from `~/.secrets/openai_api_key`** (expected file permissions 600). No
key value of any kind is hard-coded or stored anywhere in this repository.
The two locally deployable encoders (BGE-M3, Qwen3-Embedding-0.6B) require
no external API call.

## Running the tests

```
cd scripts/R && cp house_style.R ~/R/house_style.R   # if you intend to run the figure scripts
cd tests && pytest -q
```

`conftest.py` and each test's own `SCRIPTS_DIR` computation add `scripts/`
to the Python path automatically — no environment variable or installation
step is required to import the modules under test. `test_01_frame.py` is an
integration test against the real analysis frame and the embedding shards;
it is skipped automatically (`pytest.mark.skipif`) when that restricted
patient-level data is not present on the machine running the tests. Several
tests fit UMAP on synthetic matrices of a few thousand rows — this is light
enough for a laptop but should still not be run on a shared interactive
login node under this project's HPC policy; submit it as a small batch job.

## Data availability

This repository does not include the South Korea BSI cohort (institutional,
not publicly redistributable) or MIMIC-IV (credentialed access via
PhysioNet, subject to its own data use agreement; this analysis did not
transmit MIMIC-IV patient-level records to any external API, which is why
only the two locally deployable encoders were evaluated on that cohort).
See the manuscript's Data availability statement for details.

## Citation

If you use this code, please cite the manuscript (citation to be added once
the DOI is assigned) and the archived release of this repository:

> tcmhwd. (2026). *bsi-embedding-phenotype-robustness* (v2.0.0) [Code].
> Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX <!-- TODO: fill in the new
> Zenodo DOI once v2 is archived. The v1 release and its Zenodo record have
> been deleted and should not be cited; this v2 release supersedes it. -->

## License

MIT License — see `LICENSE`.
