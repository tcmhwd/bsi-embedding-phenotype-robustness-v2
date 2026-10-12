# Reproducibility map

This file replaces the earlier `ANALYSIS_PROVENANCE.md`. That file used the
v10/v11 internal supplementary-numbering scheme (Supplementary Notes 1-2,
"Option B", `REVISION_MAP_v11.md`, `out/39_audit6`, `out/41_audit7`, an
internal Source Data 4 build log, an `[[AUTHOR ACTION ...]]` bracket, and a
private-repository commit range) and is not meaningful against the current
manuscript (manuscript1009) or this repository's contents, so it is not
shipped. Everything in this file uses **manuscript1009 numbering**
(Table 1-2, Fig. 1-4, Supplementary Fig. 1-5, Supplementary Table 1-9).

See `README.md` for what each top-level directory contains, the run order,
and the environment setup. This file is the item -> script -> output ->
environment cross-reference, the record of where the released code departs
from the internal pre-specification, and the list of known provenance gaps.

---

## (i) Manuscript item -> producing script -> output -> environment

Legend: **F** = frozen environment (`/home/programs/anaconda3/envs/gpt`,
umap 0.5.9.post2) · **F'** = non-frozen `gpt` environment
(`/data/tcmhwd/miniconda3/envs/gpt`, umap 0.5.7 -- a different environment
with the same informal name as F; see README.md's Environments section) ·
**B** = `bsi_pheno` · **R** = `rnaseq_r`. Output paths
are given relative to `out/` in the working repository layout, with the
released follow-up namespace (`out/followup_analyses/...`, renamed from an
internal codename used in the private repository -- see "Release edits"
in `README.md`). Paths under `out/` are not shipped with this repository
(see G3 below); they are the paths a full rerun would reproduce.

| Manuscript item | Producing script(s) | Output | Env |
|---|---|---|---|
| Study design: "32,329 admissions among 27,083 patients" | `01_build_analysis_frame.py` | `01_frame/unit_of_analysis_report.json` | B |
| Table 1 (both cohorts) | `50_cohort_table.py` | `50_cohort_table/table1_cohort.csv` + `.md` | B |
| ST1a "Exact reconstruction" (ARI = AMI = 1.000, k=16, 1,792 noise) | `16_reproduce_original_env.py` | `16_c1_sweep/result_original_env_seed42.json` | F |
| ST1a "Corpus-only change" (seeds 1-5, 42) | `16_reproduce_original_env_multi.py` | `16_c1_sweep/result_{stem}.json` (one file per seed, `stem` encodes source+seed) | F |
| ST1a "Thread sensitivity" (threads 1/4/8/24) | `16_thread_sensitivity_original_env.py` | `16_c1_sweep/result_thread_sensitivity_threads{N}.json` | F |
| ST1a "Software-stack-only change" (k=8, ARI 0.487) and "Stack x corpus" | `aux_scripts/12_stack_check/c1_cluster.py` | `12_codex_c1/result_{source}_seed{N}_{sklearnver}.json` + `labels_{source}_seed{N}_{sklearnver}.npz` | F' (both reported arms run under the non-frozen `gpt` environment -- array tasks 21 and 22 of `c1_array.sh` respectively; every other task in that same wrapper, which is not itself one of these two reported manuscript items, runs under B) |
| ST1a "HDBSCAN parameter grid" (752 combinations) | `16_hdbscan_sweep.py`, summarised by `16_summarize_sweep.py` | `16_c1_sweep/sweep_results.jsonl` | F / B |
| ST1b/c (per-seed k/basin; pairwise ARI by basin) | `20e_task3_seed_basins.py` | `20_onehot/task3_per_seed.csv`, `task3_k8_basin_merge_*.csv`, `task3_pairwise_ari.csv` (and sibling `task3_*` files in the same directory) | F |
| T2, ST2a S1-S5 pass/fail matrix | `17a_s1_seed_thread.py`, `17b_s2_bootstrap.py`, `17c_s4s5_transport.py` (fixed rerun), `48_s3_config_sensitivity.py`, aggregated by `17e_aggregate_metrics.py` | `29_fixes/17e_rerun_verify/survival_matrix.csv` | F (S1/S2/S4/S5/S3 fits) / B (`17e` aggregation) |
| Res: "13 of 16 clusters ... F1 >= 0.60" (S6c token lookup); SF1 | `17d_token_lookup.py` | `17_correct_env/token_lookup/g4_perclass_published.json` + `.csv` | B |
| ST2a "Artifact label" (T2 stage 4); hospital-onset fraction 0.737 at 2-day cutoff | `22f_artifact_screen.py` | `22_published_objects/task_b_artifact_screen/artifact_screen_matrix.csv` + `SUMMARY.json` | B |
| Meth: hospital-onset fraction 0.728 at 3-day cutoff | `78_provenance_checks.py` | `followup_analyses/qc/provenance_checks_table3_0693_table2_0728.json` | B |
| Discussion limitations: Spearman size-stability, "none survived Holm correction" | `32_size_stability.py` | `32_size_stability/size_stability.json` | B |
| S6 whole-corpus partitions (BGE-M3/Qwen3 reference partitions for every perturbation arm); ST5c | `18a_step0_validate_corpus.py` (gate) -> `18b_embed_open.py`, `18c_embed_openai_small.py` -> `18d_s6_knn.py` -> `18e_recluster_frozen.py` -> `18f_findings.py` | `18_encoder_dec/s6_matrix.csv` | B (embed/knn/findings) / F (`18e` reclustering) |
| ST2b (S6 under three lookup readings); S6 column of ST2a/T2 | `20d_task2_extended_lookup.py`, `28b_s6_three_readings.py` | `28_openpool/s6_three_readings.csv` + `_summary.json` | B |
| ST3 (E. coli S1/S2/S4/S5 10x4 matrix) | `22b_ecoli_s1_seed.py`, `22c_ecoli_s2_bootstrap.py`, `22d_ecoli_s4s5_transport.py`, aggregated by `22e_ecoli_aggregate.py` | `22_published_objects/task_a_ecoli/survival_matrix_s1s2s4s5.csv` + `SUMMARY.json` | F (fits) / B (aggregation) |
| ST4 (structured comparators); Res: "ARI 0.037-0.070" (one-hot) | `20a_parse_template_fields.py`, `20b_recluster_onehot.py`, `20c_task1_report.py`, `22g0_gower_precompute.py`, `22g1_comparator_point.py`, `22g2_comparator_bootstrap.py`, `22g3_assemble_table.py`, `22h_published_baseline_rows.py` | `22_published_objects/task_c_comparators/comparison_table.csv` | F (`20b` reclustering) / B (everything else) |
| F1a/c, Res "within-encoder stability"; cross-encoder ARI 0.003-0.049 | `audit6_matched_encoders.py`, `audit6_concordance_whole.py` | `39_audit6/matched_encoders.json`, `39_audit6/concordance_whole.json` | B |
| Discussion limitations: 20-D UMAP cross-encoder ARI medians 0.049/0.015/0.001 | `73_D_cluster.py` | `followup_analyses/D_alt_pipeline_optional/d_alt_summary.json`, `d_alt_*.csv` | B |
| T1 MIMIC columns; F1b; F2; ST9 | `27a_mimic_cohort.py`, `27b_mimic_enrich.py`, `27c_mimic_narratives.py`, `27d_mimic_embed.py`, `27e_mimic_replicate.py` | `27_mimic/cohort/admissions_enriched.parquet`, `27_mimic/replicate/` | B |
| F2a/b, ST5 field alignment (margin > 0.10 rule; 0.653/0.700/0.947; MIMIC 0.759/0.696; 0.571 vs 0.803) | `40_field_dominance.py`, `audit7_dominance_checks.py` | `40_field_dominance/`, `41_audit7/dominance_checks.json` | B |
| F3a/b, ST7, SF4 (every South Korea variant; Qwen3/3-large/3-small arms) | `21a_generate_variants.py`, `21b_embed_openai.py`, `21c_cluster_frozen.py`, `26g_aggregate.py`, `26b_variant_geometry_all.py` (runs `aux_scripts/25_audit4/a_variant_geometry.py`), `45_3large_ablation_seeds.py`, `audit10_no_organism_seeds.py`, `audit7_make_variant.py`, `audit7_embed_qwen3.py`, `audit7_cluster_variants.py`, `44_ablation_cluster.py`, `47_qwen3_extra_seeds.py`, `audit9_make_mask_variants.py`, `audit9_embed_qwen3.py`, `audit9_cluster_mask.py`, `audit9_knn_purity.py`, `49_3small_no_department.py`, `65_B_variants.py`, `65b_B3_embed_qwen3.py`, `66_B_cluster.py`, `66b_B3_cluster.py`, `69_B_merge_matrix.py` | `followup_analyses/B_symmetric_perturbation/`, `25_audit4/` | F (clustering fits: `21c`, `45`, `audit10`, `audit7_cluster_variants`, `44`, `47`, `audit9_cluster_mask`, `66`, `66b`, and `49`/`65` when run with `--stage cluster`) / B (embeddings, variant generation, aggregation, and `49`/`65` when run with `--stage embed`) |
| Res/SF2/SF3/ST6, Analysis A (MIMIC-compatible restriction) | `60_harmonized_manifest.py`, `61_harmonized_render.py`, `62_harmonized_embed.py`, `63_harmonized_matched.py`, `64_harmonized_fields.py`, `70_A_rescore_korea_full.py`, `76_mimic_doc_stats.py` | `followup_analyses/A_harmonized_template/` | B |
| SF5, Analysis C (MIMIC organism perturbation; AMI 0.804 -> 0.555) | `67_C_mimic_variants.py`, `67b_C_embed.py`, `68_C_cluster.py` | `followup_analyses/C_mimic_organism_ablation/` | B |
| F4a/b, ST8, Abstract 90-day retrieval discrimination | `23b_run_retrieval.py`, `23c_run_anchors.py`, `23e_assemble_table.py`, `23g_run_neighborhood_balance.py`, `23h_assemble_neighborhood_balance.py`, `24c_arm3_sofa_anchor.py`, `26e_sofa_stratified.py` (gate), `28a_openpool_query_mask.py`, `50_retrieval_bootstrap.py`, `50b_baseline_bootstrap.py` | `23_retrieval/comparison_table.csv`, `26_controls/sofa_stratified.json` | B |
| ST8b note: dropping four organ-support fields, Gower 0.794/0.786 | `aux_scripts/25_audit4/d_gower_index_only.py` | `25_audit4/d_gower_index_only.json` | F |
| Figure data assembly (all tidy CSVs behind F1-F4, SF1, SF4) | `33_figure_data.py`, `36_figure_data_v4.py`, `71_followup_figure_data.py`, `74_v6_figure_data.py`, `75_fix_runnerup.py`, `80_v7_figure3_supp456.py` (byte-copy) | `33_figures/`, `tables_v6/`, `tables_v7/` | B |
| Fig. 1, Fig. 2, Fig. 4, SF1, SF4 | `85_v8_figures_main.R` | `followup_analyses/figures_v8/fig1_agreement.*`, `fig2_field_dominance.*`, `fig5_retrieval.*`, `figS1_variant_ranks.*`, `fig4_token_lookup.*` | R |
| Fig. 3, SF2, SF3, SF5 | `86_v8_figure3_supp456.R` | `followup_analyses/figures_v8/Figure_3.*`, `Supplementary_Figure_4.*`, `Supplementary_Figure_5.*`, `Supplementary_Figure_6.*` | R |

**Figure/table filename map (manuscript1009 numbering).** The output base
filenames predate manuscript1009's current numbering and are not renamed in
the code (changing them would be a code edit beyond this release's scope).
From `85_v8_figures_main.R`: `fig1_agreement` = **Figure 1**,
`fig2_field_dominance` = **Figure 2**, `fig5_retrieval` = **Figure 4**,
`fig4_token_lookup` = **Supplementary Figure 1**, `figS1_variant_ranks` =
**Supplementary Figure 4**, `fig6_size_stability` = not displayed in the
manuscript (referenced only as a Discussion limitation). From
`86_v8_figure3_supp456.R`, the on-disk saved basenames are `Figure_3`,
`Supplementary_Figure_4`, `Supplementary_Figure_5`, `Supplementary_Figure_6`
(the R object names in the script itself -- `fig3`, `figSF4`, `figSF5`,
`figSF6` -- differ from what is actually written to disk; use the saved
basenames, not the object names, when looking for the files). Mapped to
manuscript1009 numbering: `Figure_3` = **Figure 3**, `Supplementary_Figure_4`
= **Supplementary Figure 2**, `Supplementary_Figure_5` =
**Supplementary Figure 3**, `Supplementary_Figure_6` =
**Supplementary Figure 5**.

**Shared library modules** (`02_recluster.py`, `02b_partition_clustering.py`,
`03_stability.py`, `04_open_embedding.py`, `05_baseline_comparators.py`,
`05c_comparison_table.py`, `08_transport.py`, `17_common.py`,
`18_common.py`, `21_common.py`, `22a_ecoli_common.py`,
`22g_comparator_common.py`, `23a_retrieval_common.py`,
`23f_neighborhood_balance_common.py`, `embed_common.py`,
`embed_variants_openai.py`, `audit6_token_checks.py`,
`38_token_recoverability.py`, `prespec_thresholds.json`) compute no
manuscript number directly; they are imported by the producer scripts
above and run in whichever environment the importing script runs in (most
are dual-use: `02_recluster.fit`-style code paths execute in F when called
from an F-environment producer such as `17c`/`22d`, and in B when called
from a B-environment producer such as `22g1`).

---

## (ii) Deviations from the internal stress-testing pre-specification

The stress-testing pre-specification document itself
(`.superpowers/sdd/PRESPEC_stress_criteria.md`) is not shipped in this
repository (see "What is deliberately not here" in `README.md`); the
manuscript states it will be cited by DOI once archived. The kept code
implements a design that deviates from that pre-specification in five
respects. A reviewer reading the code without this list could misread the
implemented design as the pre-specified one, so it is disclosed here even
though the pre-spec document itself is absent:

- **D1.** The stress battery (S1-S6) and the artifact screen are run
  against the **stored, already-published cluster partition**
  (`phenotype_cluster`), not against a freshly refit "A0" baseline
  partition fit from scratch inside this project. An earlier analysis line
  (scripts numbered in the 02/06/07/11-15 range, not kept in this release)
  did attempt an A0-refit design; it is superseded and dropped because no
  manuscript1009 value is derived from it.
- **D2.** S1 (seed/thread sensitivity) tests thread counts **{1, 24, 2}**,
  not a wider thread sweep. The 2-thread arm was added after the 1-vs-24
  comparison alone was judged insufficient to characterise the
  sensitivity.
- **D3.** S3 (clustering-configuration sensitivity) was **added after** the
  original four-axis (S1/S2/S4/S5) design, once an artifact-screen
  reviewer comment raised configuration sensitivity as a separate concern;
  it is not part of the original four-axis pre-specification.
- **D4.** There is **no "A1" arm** (a second alternative baseline
  partition) in the released design. The pre-specification's A1 concept
  was dropped before implementation.
- **D5.** The E. coli within-organism repetition (ST3) repeats only
  **four** axes (S1, S2, S4, S5) — **not** S3 or S6. S3
  (configuration sensitivity) and S6 (encoder recoverability) were judged
  not to generalise usefully to the smaller E. coli subset and were not
  run there.

Rationale for disclosing this here rather than silently matching the code
to the pre-spec: the code is correct and internally consistent as written,
and this is the design actually used to produce every reported S1-S6/ST2/ST3
value — but a reader who assumes it follows an unseen pre-specification
document verbatim would draw the wrong conclusion about why these specific
choices (stored partition, 1/24/2 threads, S3 added later, no A1, E. coli
without S3/S6) were made. This condenses the relevant part of the v1
`ANALYSIS_PROVENANCE.md` §9 deviations account, with the internal
`[[AUTHOR ACTION ...]]` bracket and the private-repository commit range
removed (neither is meaningful in a public release with fresh git history).

---

## (iii) Known provenance gaps

See also "Release edits" (E1-E5) and the environment-table note above.

- **G1. `17e_aggregate_metrics.py` rerun invocation is not recorded.** The
  reported S1/S2/S4/S5 pass/fail matrix is
  `out/29_fixes/17e_rerun_verify/survival_matrix.csv`, built from the fixed
  S4/S5 transport outputs in `out/29_fixes/17c_s4s5_transport/` (the
  corrected rerun via `slurm/29_n7_s4s5_transport_fixed.sh`, not the
  superseded pre-fix `slurm/17c_s4s5_transport_array.sh`, which is not
  included in this release). `17e_aggregate_metrics.py` hardcodes
  `S4S5_DIR = OUT / "s4s5_transport"` and takes no path argument, and no
  surviving wrapper records how the script was pointed at the corrected
  output location for the reported run. This is disclosed rather than
  silently fixed: the script was not changed to add a path argument
  (that would be a new, unreported edit); the aggregation is understood to
  have been re-run by hand against the corrected transport outputs, but
  the exact invocation is not recorded.
- **G2. The original South Korea narrative renderer is not included and is
  not reproducible from this repository.** See README.md's "Narrative
  serialisation" note.
- **G3. Code that lived under `out/` in the working repository, not under
  `scripts/`.** Several load-bearing scripts (`aux_scripts/25_audit4/`,
  `aux_scripts/12_stack_check/`) were originally one-off analysis files kept
  under the project's `out/` working-output tree rather than its `scripts/`
  tree. They are copied here verbatim into `aux_scripts/`. Only one of them
  (`a_variant_geometry.py`) is actually executed at runtime by a kept
  producer script (`spec_from_file_location`, from `26b_variant_geometry_all.py`);
  the other three run via their own SLURM wrapper instead (see README.md's
  "What is here" note on `aux_scripts/`). Their internal path references
  (`out/25_audit4/...`, `out/12_codex_c1/...`) are left unedited, consistent
  with this repository's general policy of shipping code unmodified except
  for the five documented release edits (E1-E5) -- this is why
  `aux_scripts/12_stack_check/c1_cluster.py`'s own output path and
  `16_hdbscan_sweep.py`'s cached-input path still read `out/12_codex_c1/...`
  even though the shipped source directory was renamed to
  `aux_scripts/12_stack_check/` for this release (see README.md's
  "Release edits" E5). The figure house-style file
  (`scripts/R/house_style.R`) similarly lived outside the private
  repository's own `scripts/` tree, at `~/R/`, because it is a cross-project
  shared file; see README.md for where to place it.
- **G4. Fonts (Arial) and the fontconfig file for figure rendering are not
  shipped.** `85_v8_figures_main.R`/`86_v8_figure3_supp456.R` require
  `FONTCONFIG_FILE=out/followup_analyses/fonts/fonts.conf` pointing at a
  bundled Liberation Sans/Mono substitute (metric-compatible with
  Arial/Courier New on a compute node that has no Arial installed); this
  fontconfig file and font bundle are not part of this release. See
  README.md for what to provide.
- **G5. A number of light steps ran directly with no SLURM wrapper.** See
  "Run order" in README.md for the full list; this is expected, not an
  omission, for scripts whose own runtime is a few seconds to a few
  minutes of pandas/numpy work.
- **G6. `audit7_dominance_checks.py`'s section 5 runs a classifier-ablation
  check (via `audit6_token_checks.py` -> `38_token_recoverability.py`)
  whose own output is not reported anywhere in manuscript1009.** It is kept
  in this release only because `audit7_dominance_checks.py`'s main()
  function executes it as a later section and the script needs to run to
  completion; the script's actually-reported output
  (`dominance_checks.json`, feeding F2b) is written before section 5 runs.
