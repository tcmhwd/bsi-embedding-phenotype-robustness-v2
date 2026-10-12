#!/usr/bin/env Rscript
# Manuscript package v8 figures: Figure 3, Supplementary Figures 4, 5, 6 -- TARGETED POLISH
# pass on top of the v7 restructure below. NEW, additive script -- does NOT edit
# scripts/80_v7_figure3_supp456.R (or scripts/74_v6_figures.R) in place; those remain frozen
# "before" records. Reads the SAME out/followup_analyses/tables_v7/*.csv v7 already reads (no
# data prep re-run -- v7's own finding was that no data prep was needed for a style pass, and
# the underlying CSVs have not changed since; verified by timestamp before writing this
# script, no scripts/86_..._data.py written). No embedding/clustering/API job, no numeric
# value changed anywhere in this script. v8 changes (full detail + before/after text in
# embedding_v8/CHANGELOG_FIGURES_v8.md):
#  Figure 3b  (right, Qwen3 15-NN purity panel) PRIORITY fix -- x-axis category labels
#             shortened further (Unmodified / Organism removed / Department removed /
#             Therapy removed / First-antibiotic masked / All-antibiotics masked, verified
#             arm-for-arm against knn_readout_B3.csv / figB_knn.csv, same six arms as v7, none
#             invented/dropped) and enlarged; panel-b widths rebalanced so the right sub-panel
#             gets more room (was 1 : 1.7 = ~37% : 63%; now ~42% : 58%, within the task's
#             40-45% / 55-60% target) instead of shrinking the left sub-panel's font.
#  Figure 3a  no change -- confirmed "Not run" boxes remain absent (v7 already removed them)
#             and no explanatory prose was re-added to the plotting area.
#  Suppl. 5   dominant-field outline colour changed from HS_PAL$red (#C0504D) to HS_PAL$ink
#             (#16150F, linewidth 1.1 -> 1.4) to match Figure 2a's own outline convention
#             (Figure 2a's outline was switched from red to near-black in v7 after failing a
#             WCAG/CVD colour-blind-safety check against the same AMI fill gradient this
#             panel uses -- SF5 was not touched by that v7 check and still carried the old
#             red, which this pass now brings into line). "n/s" grey text and grey
#             not-rendered tiles are UNCHANGED (that grey is an intentional neutral-category
#             marker, not the thing being fixed).
#  Suppl. 6   x-axis arm labels shortened further (line-count suffix and detailed edit
#             description moved OUT of the axis into the changelog, per the task; arm names
#             verified unchanged against mimic_ablation*.csv) and the freed space used to
#             enlarge the axis-text font.
#  Figure 4, Supplementary 4 are otherwise byte-for-byte the same drawing code as
#  scripts/80_v7_figure3_supp456.R (re-run for consistency / diff-check only; SF4 already had
#  a shape encoding for BGE-M3 vs Qwen3-0.6B from v7, verified present, not re-added).
#
# ---------------------------------------------------------------------------
# v7 header (unchanged, retained for provenance):
# Manuscript package v7 figures: Figure 3 (major restructure) and Supplementary Figures 4, 5, 6 (declutter/restructure).
# PURE RESTYLE round: source data is identical to v6 (out/followup_analyses/tables_v7/, a straight copy of tables_v6/ made
# by scripts/80_v7_figure3_supp456.py -- nothing is recomputed, no number changes anywhere in this script). Modelled on
# scripts/74_v6_figures.R (not edited in place; that script and out/followup_analyses/figures_v6/ / tables_v6/ are untouched).
#
# v7 changes vs v6 (full detail + before/after text in submission_build_v7/CHANGELOG_FIGB.md):
#  Figure 3   (A) removed the "Not run" dashed placeholder + category for the dominant-field value-masked arm from the
#                 text-3-large and text-3-small facets (that arm was run only for Qwen3-0.6B -- verified against
#                 out/followup_analyses/B_symmetric_perturbation/perturbation_matrix_completed.csv); each facet now has
#                 only the x categories that were actually run.
#             (B) grey -> near-black (HS_PAL$ink) for meaningful in-panel text (dominant-field subtitles, chance-level
#                 annotation); panel-a subtitle REMOVED (duplicated the legend) -- see changelog for the removed wording.
#             (C) panel-b right (Qwen3 15-NN purity) x-axis labels enlarged to match panel-b left and reworded to
#                 2-line-max concise labels, arm-for-arm identical to before (verified against
#                 out/followup_analyses/B_symmetric_perturbation/knn_readout_B3.csv and tables_v7/figB_knn.csv).
#             (D) CRITICAL LABEL CHECK (verified, not a canvas change): confirmed dominant_field_display_key ==
#                 "first_antibiotic" for Qwen3-0.6B / korea_full_template in tables_v7/figA_dominance_summary.csv
#                 (top_ami_median 0.947226...) and in out/33_figures/field_dominance.csv (is_dominant row, ami
#                 0.947226...). DOM_FIELD below already read "first antibiotic named" for Qwen3-0.6B in the v6 script
#                 and does so again here -- NO CANVAS TEXT WAS WRONG, so none was changed. The "dominant field is the
#                 therapy block" mislabel this item warned about is present in the Fig. 3 LEGEND TEXT (.md), not on
#                 any canvas produced by this script; see CHANGELOG_FIGB.md (this script cannot fix it -- .md files
#                 are out of scope for this task).
#             (E) provenance shapes (circle = original arm, diamond = added post hoc) kept unchanged; already concise.
#  Suppl. 4   near-black text for "median [min, max]" / "pairs" columns and their headers; pA1's explanatory subtitle
#             removed (duplicated in the legend) and the freed space used to enlarge those two columns; panel c gains
#             a shape (in addition to colour) encoding for BGE-M3 vs Qwen3-0.6B. Grey/blue open-vs-filled pair-set
#             encoding (panel a/b) is UNCHANGED -- that grey is the pair-set meaning, not text.
#  Suppl. 5   panel a's long explanatory paragraph removed (verbatim in changelog; the field-set caveat it carried is
#             already stated at length in the external caption/footnote -- quoted in the changelog, not deleted);
#             panel b's subtitle shortened (kept short, because the "not the AMI of the field named beside it" point
#             is a flagged misreading-risk fix, not ordinary clutter) and recoloured near-black; condition headers
#             abbreviated (mapping in changelog, to be reused verbatim by the caption); freed space used to enlarge
#             cell numbers / axis text; "not rendered" / "not scored" legend keys kept unchanged.
#  Suppl. 6   long grey explanatory subtitles (panels a, c) removed (verbatim in changelog; content is already in the
#             external caption); x-axis arm labels shortened to 2 lines and enlarged; BGE-M3 vs Qwen3-0.6B gains a
#             shape encoding (in addition to colour) in panels a/b.
#
# House style: shared ggplot2 figure conventions (Arial). Conventions unchanged from 74_v6_figures.R: ARI /
# AMI are both-non-noise; ranges are OBSERVED ranges across fits / pairs / seeds and are not confidence intervals.

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  source("~/R/house_style.R")
})

cat("[86] R ", as.character(getRversion()), "; ggplot2 ", as.character(packageVersion("ggplot2")), "; patchwork ",
    as.character(packageVersion("patchwork")), "; ragg ", as.character(packageVersion("ragg")), "; svglite ", as.character(packageVersion("svglite")), "\n", sep = "")
.arial_path <- systemfonts::match_fonts("Arial")$path
cat("[86] Arial resolves to: ", .arial_path, "\n[86] mono resolves to: ", systemfonts::match_fonts("mono")$path, "\n", sep = "")
if (!grepl("Liberation|Arial", .arial_path))
  stop("Arial does not resolve to Arial/Liberation Sans (got ", .arial_path, "); refusing to draw with a substitute font")

REV    <- "/data/tcmhwd/AI/AST_ai/phenotype_revision"
TABLES <- file.path(REV, "out/followup_analyses/tables_v7")  # v8: unchanged data, still v7's tables
STAGE  <- file.path(REV, "out/followup_analyses/figures_v8")  # v8: own stage dir (figures_v7 is
                                                            # shared with scripts/80_v7_figure3_supp456.R,
                                                            # whose wrapper wipes it with `rm -f` on
                                                            # every run -- reusing it here would race)
dir.create(STAGE, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------- Sizing (as 36_figures_v4.R / 74_v6_figures.R)
WIDTH_IN  <- 7.3
PRINT_MM  <- 180
SCALE     <- PRINT_MM / (WIDTH_IN * 25.4)
BASE_SIZE <- 10
MM <- function(final_pt) final_pt / (2.8346 * SCALE)          # geom_text/annotate `size` (mm) that renders as final_pt AT FINAL PRINT SIZE
FPT <- function(final_pt) rel(final_pt / (BASE_SIZE * SCALE)) # theme element_text `size` (rel to BASE_SIZE) that renders as final_pt at final print
fmt_num  <- function(x) gsub("-", "−", sprintf("%.3f", x), fixed = TRUE)
fmt_cell <- function(x) { v <- round(x, 3); ifelse(v == 0, "0.000", gsub("-", "−", sprintf("%.3f", v), fixed = TRUE)) }   # true minus; a rounded-to-zero negative prints 0.000
NOT_RENDERED <- "#B9BDC0"
ORANGE <- HS_PAL$groups[2]
GREEN  <- HS_PAL$groups[3]
INK    <- HS_PAL$ink            # near-black (#16150F); house-style override target for meaningful annotation text (was HS_PAL$gray in v6)
truthy <- function(x) x %in% c(TRUE, "True", "TRUE", "true")

# ---------------------------------------------------------------- SVG font declaration
# NOTE (v7, diverges from 74_v6_figures.R): ~/R/house_style.R's own hs_save_vector() now rewrites the SVG's sans
# font-family itself, to  Arial, "Liberation Sans", Helvetica, sans-serif  (its on.exit block; the "Helvetica"
# fallback and the switch from single- to double-quoted attribute text were added to that shared file on 2026-09-22,
# after 74_v6_figures.R was last run -- see house_style.R's own comment on the single-quote SVG-corruption incident).
# house_style.R is shared across projects and is read-only here, so SANS_STACK below is kept in sync with its
# CURRENT output rather than hand-rolling a different stack; MONO_STACK is unaffected (house_style.R's regex does not
# touch "Liberation Mono") and this script still performs that substitution itself, exactly as in 74_v6_figures.R.
SANS_STACK <- 'font-family: Arial, "Liberation Sans", Helvetica, sans-serif'
MONO_STACK <- 'font-family: "DejaVu Sans Mono", "Liberation Mono", "Courier New", monospace'
fix_svg_fonts <- function(path) {
  n <- file.info(path)$size
  x <- readChar(path, n, useBytes = TRUE)
  x <- gsub('font-family: "Liberation Sans"', SANS_STACK, x, fixed = TRUE, useBytes = TRUE)
  x <- gsub('font-family: Arial, "Liberation Sans", Helvetica, sans-serif', SANS_STACK, x, fixed = TRUE, useBytes = TRUE)
  x <- gsub('font-family: "Liberation Mono"', MONO_STACK, x, fixed = TRUE, useBytes = TRUE)
  writeChar(x, path, eos = NULL, useBytes = TRUE)
  fam <- regmatches(x, gregexpr("font-family: [^;']*", x, useBytes = TRUE))[[1]]
  tab <- table(fam)
  cat("[86]   svg font-family declarations in ", basename(path), ": ", paste0(names(tab), " x", as.integer(tab), collapse = " | "), "\n", sep = "")
  ok <- fam %in% c(SANS_STACK, MONO_STACK)
  if (!all(ok)) stop("unexpected font-family in ", path, ": ", paste(unique(fam[!ok]), collapse = "; "))
  invisible(path)
}
save_fig <- function(p, stem, height) {
  hs_save(p, file.path(STAGE, paste0(stem, ".png")), width = WIDTH_IN, height = height, dpi = 300)
  hs_save_vector(p, file.path(STAGE, paste0(stem, ".svg")), width = WIDTH_IN, height = height)
  fix_svg_fonts(file.path(STAGE, paste0(stem, ".svg")))
  cat("[86] written ", stem, " (.png, .svg), ", WIDTH_IN, " x ", height, " in\n", sep = "")
}

# ---------------------------------------------------------------- Display label maps (raw identifier -> canvas label)
LABEL_LOG <- data.frame(kind = character(), raw_id = character(), display = character(), stringsAsFactors = FALSE)
map_log <- function(kind, raw, disp) {
  disp1 <- gsub("\n", " / ", disp, fixed = TRUE)
  LABEL_LOG <<- rbind(LABEL_LOG, data.frame(kind = kind, raw_id = raw, display = disp1, stringsAsFactors = FALSE))
}
no_raw <- function(x, what) {           # audit rule: no underscore-bearing identifier may reach a canvas
  bad <- x[grepl("_", x, fixed = TRUE) & !is.na(x)]
  if (length(bad)) stop("raw identifier(s) on canvas (", what, "): ", paste(unique(bad), collapse = ", "))
  x
}

# Analysis A conditions -- FULL form, unchanged from v6, used by Supplementary Fig. 4 only (SF4 condition labels were
# not in scope for this round's changes)
COND_ORDER <- c("korea_full_template", "korean_mimic_compatible", "harmonized_keep_therapy", "thickness_matched_random", "mimic_iv")
COND_DISP <- c(korea_full_template = "Korea full template",
               korean_mimic_compatible = "Korea MIMIC-compatible\nfield set",
               harmonized_keep_therapy = "Korea MIMIC-compatible\nfield set, therapy kept",
               thickness_matched_random = "Korea random line removal\n(matched line budget)",
               mimic_iv = "MIMIC-IV")
for (i in COND_ORDER) map_log("SF4 condition (row / axis, unchanged v6 form)", i, COND_DISP[[i]])
no_raw(COND_DISP, "COND_DISP")
cond_axis <- function(id) { stopifnot(all(id %in% names(COND_DISP))); unname(COND_DISP[id]) }

# Analysis A conditions -- SHORT form, v7 only, used by Supplementary Fig. 5 (both panel a's column strip and panel
# b's y axis -- task requirement: "these are SHORTER than the existing display_label_map_v6.csv strings ...use the
# new short forms in the figure and write the mapping (old->new) into your changelog so the legend-writing pass uses
# the SAME short forms in the caption/legend text -- they must match exactly"). SF4 keeps the full COND_DISP form
# above; only SF5 uses these.
COND_SHORT <- c(korea_full_template = "Korea full",
                korean_mimic_compatible = "Korea MIMIC-compatible",
                harmonized_keep_therapy = "Korea MIMIC-compatible\n+ therapy",
                thickness_matched_random = "Korea random\nline removal",
                mimic_iv = "MIMIC-IV")
COND_SHORT_STRIP <- c(korea_full_template = "Korea\nfull",
                      korean_mimic_compatible = "Korea MIMIC-\ncompatible",
                      harmonized_keep_therapy = "Korea MIMIC-\ncompatible\n+ therapy",
                      thickness_matched_random = "Korea random\nline removal",
                      mimic_iv = "MIMIC-IV")
for (i in COND_ORDER) {
  map_log("SF5 condition, v7 abbreviated form (old->new mapping for the caption)", paste0(i, " | old: ", gsub("\n", " ", COND_DISP[[i]])), COND_SHORT[[i]])
}
no_raw(COND_SHORT, "COND_SHORT"); no_raw(COND_SHORT_STRIP, "COND_SHORT_STRIP")

# clinical fields (SF5a rows / SF5b text) -- unchanged from v6
FIELD_LABELS <- c(
  admission_cluster = "Pre-computed admission cluster", empiric_cluster = "Pre-computed empiric-treatment cluster",
  antibiotic_set = "Antibiotic regimen", first_antibiotic = "First antibiotic named", n_antibiotic_lines = "Number of antibiotics",
  antipseudoab_empiric = "Empiric antipseudomonal therapy", carbaab_empiric = "Empiric carbapenem therapy",
  antiMRSAab_empiric = "Empiric anti-MRSA therapy", antiVREab_empiric = "Empiric anti-VRE therapy",
  index_organism = "Index organism", polymicrobial = "Polymicrobial infection", source_category = "Source of infection",
  charlson_category = "Charlson category", charlson_class = "Charlson class",
  comorbid_infectious = "Infectious comorbidities (ICD-10)", entry_mode = "Entry mode", icu = "ICU stay",
  index_bsi_mrsa = "Index-episode MRSA", index_bsi_3gcr = "Index-episode 3GCR", index_bsi_vre = "Index-episode VRE",
  index_bsi_carbar = "Index-episode carbapenem resistance", index_bsi_ceftriaxoner = "Index-episode ceftriaxone resistance",
  index_bsi_crab = "Index-episode CRAB", index_bsi_polymicrobial = "Index-episode polymicrobial",
  BSI_3gcr = "Any-episode 3GCR", BSI_carbar = "Any-episode carbapenem resistance", BSI_mrsa = "Any-episode MRSA", BSI_vre = "Any-episode VRE",
  n_bsi_episodes = "Number of BSI episodes", treatment_department_eng = "Treatment department", hospital_code = "Hospital code",
  admit_year = "Admission year", admit_period = "Admission period", sex = "Sex", care_unit = "Care unit", fungemia = "Fungemia",
  ventilator = "Ventilator", central_orarterial_line = "Central or arterial line", urinary_orrenal_catheter = "Urinary or renal catheter")
gloss <- function(v) {
  out <- ifelse(v %in% names(FIELD_LABELS), unname(FIELD_LABELS[v]), v)
  for (i in which(!is.na(v) & v %in% names(FIELD_LABELS))) map_log("clinical field", v[i], out[i])
  out
}

# =================================================================== Supplementary Figure 4: concordance
conc  <- read.csv(file.path(TABLES, "figA_concordance.csv"))
degen <- read.csv(file.path(TABLES, "figA_degeneracy.csv"))
cond_levels <- c(rev(cond_axis(COND_ORDER)), "")     # last level = top row of the discrete axis: carries the column headers
conc$cond <- factor(cond_axis(conc$condition), levels = cond_levels)
PS_LEV <- c("All matched pairs", "Pairs with k ≥ 5 in both partitions (non-degenerate subset)")
conc$pair_set_lab <- factor(ifelse(conc$pair_set == "all_pairs", PS_LEV[1], PS_LEV[2]), levels = PS_LEV)
conc$stat_txt <- sprintf("%s [%s, %s]", fmt_num(conc$median), fmt_num(conc$min), fmt_num(conc$max))
conc$n_txt <- ifelse(conc$pair_set == "all_pairs", sprintf("n = %d", conc$n_pairs), sprintf("n = %d of %d", conc$n_pairs, conc$n_pairs_total))
DODGE <- position_dodge(width = 0.62)

# v7: "median [min, max]" / "pairs" values and headers -> near-black (was HS_PAL$gray for n_txt + both headers); both
# columns enlarged (6.6-6.8pt final -> 7.4-7.6pt final) using the room freed by removing pA1's subtitle below.
# xmax/x_col2 for each call site (below) are widened by the same ratio as the font size increase, so the two columns
# keep clearance at the larger size (a first pass at +30% text with unchanged positions overlapped col1 into col2;
# fixed by scaling positions together with size, verified by re-rendering and reading the PNG -- see CHANGELOG_FIGB.md).
conc_panel <- function(metric, title, xlab, xmax, breaks, x_col1, x_col2) {
  d <- conc[conc$metric == metric, ]
  hdr <- data.frame(cond = factor("", levels = cond_levels))
  ggplot(d, aes(y = cond, colour = pair_set_lab, group = pair_set_lab)) +
    geom_vline(xintercept = 0, linetype = "dashed", linewidth = 0.4, colour = HS_PAL$gray) +
    geom_linerange(aes(xmin = min, xmax = max, linewidth = pair_set_lab), position = DODGE, orientation = "y") +
    geom_point(aes(x = median, shape = pair_set_lab, fill = pair_set_lab), position = DODGE, size = 2.6, stroke = 0.8) +
    geom_text(aes(x = x_col1, label = stat_txt), position = DODGE, hjust = 0, size = MM(7.6), family = "mono", colour = INK, show.legend = FALSE) +
    geom_text(aes(x = x_col2, label = n_txt), position = DODGE, hjust = 0, size = MM(7.4), family = "mono", colour = INK, show.legend = FALSE) +
    geom_text(data = hdr, aes(x = x_col1, y = cond), label = "median [min, max]", inherit.aes = FALSE, hjust = 0,
              size = MM(7.4), family = "mono", colour = INK) +
    geom_text(data = hdr, aes(x = x_col2, y = cond), label = "pairs", inherit.aes = FALSE, hjust = 0,
              size = MM(7.4), family = "mono", colour = INK) +
    scale_colour_manual(values = c(HS_PAL$blue, "grey35"), name = NULL) +
    scale_fill_manual(values = c(HS_PAL$blue, "white"), name = NULL) +
    scale_shape_manual(values = c(19, 21), name = NULL) +
    scale_linewidth_manual(values = c(0.9, 0.5), guide = "none") +
    scale_x_continuous(limits = c(-0.05, xmax), breaks = breaks, expand = c(0, 0)) +
    scale_y_discrete(drop = FALSE) +
    labs(title = title, x = xlab, y = NULL) +
    theme_house(base_size = BASE_SIZE) +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
          legend.position = "bottom", legend.direction = "vertical", axis.text.y = element_text(lineheight = 0.9))
}
# v7: pA1's explanatory subtitle REMOVED -- it duplicated the caption ("Points are medians and lines the observed
# range across matched pairs — not confidence intervals, and not independent replication.", 02_Supplementary_Information.md).
# Removed wording (verbatim, was on pA1 only): "Points: median; lines: observed range across matched pairs (not a confidence interval)."
pA1 <- conc_panel("ARI", "a) Cross-encoder agreement: adjusted Rand index (ARI)",
                  "ARI, BGE-M3 vs Qwen3-0.6B (both-non-noise convention)", 1.18, seq(0, 0.5, 0.1), 0.58, 0.93)
pA2 <- conc_panel("AMI", "b) Cross-encoder agreement: adjusted mutual information (AMI)",
                  "AMI, BGE-M3 vs Qwen3-0.6B (both-non-noise convention)", 1.72, seq(0, 1, 0.25), 0.98, 1.44)

degen$cond <- factor(cond_axis(degen$condition), levels = rev(cond_axis(COND_ORDER)))
degen$ind_lab <- factor(degen$indicator, levels = c("k", "frac_k_le_4", "largest_share"),
                        labels = c("Median k, non-noise clusters\n(line: observed range across fits)",
                                   "Share of fits with k ≤ 4",
                                   "Largest cluster, share of all rows\n(median; line: observed range)"))
degen$enc <- factor(degen$encoder_label, levels = c("BGE-M3", "Qwen3-0.6B"))
degen$txt <- ifelse(degen$indicator == "k", sprintf("%.1f", degen$value), sprintf("%.2f", degen$value))
lim  <- data.frame(ind_lab = factor(levels(degen$ind_lab), levels = levels(degen$ind_lab)), x = c(92, 1.55, 1.55))
lim0 <- data.frame(ind_lab = lim$ind_lab, x = c(-3, -0.04, -0.04))
DODGE2 <- position_dodge(width = 0.6)
DEG_BREAKS <- function(l) if (l[2] > 10) c(0, 20, 40, 60) else c(0, 0.25, 0.5, 0.75, 1)
# v7: panel c gains a shape encoding for encoder (BGE-M3 vs Qwen3-0.6B), in addition to colour, so the two remain
# distinguishable without colour (task requirement for SF4 panel c).
ENC_SHAPE <- c("BGE-M3" = 16, "Qwen3-0.6B" = 17)
pA3 <- ggplot(degen, aes(y = cond, colour = enc, group = enc)) +
  geom_blank(data = lim, aes(x = x), inherit.aes = FALSE) +
  geom_blank(data = lim0, aes(x = x), inherit.aes = FALSE) +
  geom_linerange(data = degen[!is.na(degen$lo), ], aes(xmin = lo, xmax = hi), position = DODGE2, orientation = "y", linewidth = 0.6) +
  geom_point(aes(x = value, shape = enc), position = DODGE2, size = 2.5) +
  geom_text(aes(x = ifelse(is.na(hi), value, hi), label = txt), position = DODGE2, hjust = -0.35, size = MM(6.8),
            family = "mono", colour = HS_PAL$ink, show.legend = FALSE) +
  facet_wrap(~ind_lab, nrow = 1, scales = "free_x") +
  scale_colour_manual(values = c("BGE-M3" = ORANGE, "Qwen3-0.6B" = GREEN), name = NULL) +
  scale_shape_manual(values = ENC_SHAPE, name = NULL) +
  scale_x_continuous(expand = c(0, 0), breaks = DEG_BREAKS) +
  labs(title = "c) Degeneracy indicators per encoder (matched fits; MIMIC-IV: seeds)", x = "Indicator value (see panel headings)", y = NULL) +
  theme_house(base_size = BASE_SIZE) +
  theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(), legend.position = "bottom",
        strip.text = element_text(size = rel(0.72), lineheight = 0.95), axis.text.y = element_text(lineheight = 0.9),
        axis.text.x = element_text(size = rel(0.75)), panel.spacing.x = unit(0.9, "lines"))
figSF4 <- (pA1 / pA2 / pA3) + patchwork::plot_layout(heights = c(1.15, 1.15, 1.0), guides = "collect") & theme(legend.position = "bottom")
save_fig(figSF4, "Supplementary_Figure_4", 10.4)

# =================================================================== Supplementary Figure 5: field alignment
cells <- read.csv(file.path(TABLES, "figA_alignment_cells.csv"))
dsum  <- read.csv(file.path(TABLES, "figA_dominance_summary.csv"))
cells$is_dominant <- truthy(cells$is_dominant)
cells$cond <- factor(cells$condition, levels = COND_ORDER, labels = unname(COND_SHORT_STRIP[COND_ORDER]))  # v7: short strip labels
cells$enc_x <- factor(cells$encoder_label, levels = c("BGE-M3", "Qwen3-0.6B"))
# display filter of the legend: a ROW is drawn when the field reaches AMI >= 0.08 in at least one partition; no value inside a drawn row is hidden
keep <- unique(cells$field_display_key[cells$status == "scored" & !is.na(cells$ami_median) & cells$ami_median >= 0.08])
full <- cells[cells$field_display_key %in% keep, ]
full$field_txt <- no_raw(gloss(full$field_display_key), "SF5a rows")
full$field_disp <- factor(full$field_txt, levels = rev(sort(unique(full$field_txt))))
full$txt_colour <- ifelse(full$status == "scored" & full$ami_median > 0.55, "white", HS_PAL$ink)
full$outline <- full$status == "scored" & full$is_dominant & !is.na(full$dominance_margin_median) & full$dominance_margin_median > 0.10
sc <- full[full$status == "scored", ]
nr <- full[full$status == "not_rendered", ]
ns <- full[full$status == "not_scored", ]
cat("[86] SF5a: ", length(keep), " displayed rows; ", nrow(sc), " scored cells (all labelled), ", nrow(nr), " not-rendered, ", nrow(ns), " not-scored (n/s)\n", sep = "")

legend_layers <- list(); shape_vals <- c(); shape_fill <- c()
if (nrow(nr) > 0) {
  legend_layers <- c(legend_layers, list(geom_point(data = nr[1, , drop = FALSE], aes(shape = "Not rendered in this condition"), x = -Inf, y = -Inf, size = 0, stroke = 0, na.rm = TRUE)))
  shape_vals <- c(shape_vals, "Not rendered in this condition" = 22); shape_fill <- c(shape_fill, NOT_RENDERED)
}
if (nrow(ns) > 0) {
  legend_layers <- c(legend_layers, list(geom_point(data = ns[1, , drop = FALSE], aes(shape = "n/s: not scored (outside the stored 29-field set)"), x = -Inf, y = -Inf, size = 0, stroke = 0, na.rm = TRUE)))
  shape_vals <- c(shape_vals, "n/s: not scored (outside the stored 29-field set)" = 22); shape_fill <- c(shape_fill, "white")
}
# v7: the long explanatory paragraph is REMOVED from this panel (verbatim wording, kept for the legend-writing pass,
# is in CHANGELOG_FIGB.md); only a short title remains. The field-set caveat that paragraph carried
# ("Candidate field sets differ by column...") is NOT lost -- it is already stated at length in the external caption
# (02_Supplementary_Information.md, Supplementary Fig. 5 legend: "The candidate field sets are not identical across
# columns..."), quoted in the changelog. Cell numbers, "n/s" text and axis/strip text are enlarged with the freed space
# (each encoder column is only ~0.29in wide -- a first pass at MM(9.0) overflowed a tile's own column into its
# neighbour; MM(6.8), a modest +10% over v6's MM(6.2), is the largest size that stayed inside one column at this
# panel width, verified by re-rendering and reading the PNG -- see CHANGELOG_FIGB.md).
p_heat <- ggplot(full, aes(x = enc_x, y = field_disp)) +
  geom_tile(data = sc, aes(fill = ami_median), colour = "white", linewidth = 1.0) +
  geom_tile(data = nr, fill = NOT_RENDERED, colour = "white", linewidth = 1.0) +
  geom_tile(data = ns, fill = "white", colour = HS_PAL$edge, linewidth = 0.6) +
  geom_text(data = sc, aes(label = fmt_cell(ami_median), colour = txt_colour), size = MM(6.8), family = "mono") +
  legend_layers +
  geom_text(data = ns, label = "n/s", size = MM(6.8), family = "mono", colour = HS_PAL$gray) +   # "not scored" is a neutral category: grey kept intentionally
  scale_colour_identity() +
  # v8: dominant-field outline HS_PAL$red (#C0504D) -> HS_PAL$ink (#16150F), linewidth
  # 1.1 -> 1.4 -- matches Figure 2a's own dominant-field outline exactly (Figure 2a switched
  # from the same HS_PAL$red to HS_PAL$ink at the same linewidth in v7 after failing a
  # WCAG-contrast + colour-vision-deficiency check against the shared AMI fill gradient this
  # panel also uses; see scripts/79_v7_figures_main.R's `dominant_outline_col`/`_lwd`
  # derivation). SF5 was not covered by that check and still carried the old red before this
  # fix. Presence/absence of the outline (which cells get it) is unchanged -- still driven by
  # `full$outline` (is_dominant & margin > 0.10), same as before; only colour/width changed.
  geom_tile(data = full[full$outline, ], colour = HS_PAL$ink, fill = NA, linewidth = 1.4) +
  scale_fill_gradient(low = HS_PAL$mid, high = HS_PAL$blue, limits = c(0, 1), breaks = seq(0, 1, 0.25), name = "AMI", oob = scales::squish) +
  scale_shape_manual(values = shape_vals, breaks = names(shape_vals), name = NULL,
                     guide = guide_legend(position = "bottom", override.aes = list(size = 4, stroke = 0.4, colour = "grey40", fill = shape_fill))) +
  facet_grid(~cond, scales = "free_x", space = "free_x") +
  labs(title = "a) Median AMI between each partition and each rendered field",
       x = "Encoder-derived partition", y = "Rendered clinical field") +
  theme_house(base_size = BASE_SIZE) +
  theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(), legend.position = "right",
        strip.text = element_text(face = "bold", size = rel(0.85), lineheight = 0.95),
        axis.text.x = element_text(angle = 45, hjust = 1, size = rel(0.85)), axis.text.y = element_text(size = rel(0.85)),
        panel.spacing.x = unit(0.5, "lines"))

# ---- panel b: median per-fit MAXIMUM AMI (audit F3 / v6 open item 2) ----
dsum$cond <- factor(unname(COND_SHORT[dsum$condition]), levels = c(rev(unname(COND_SHORT[COND_ORDER])), ""))  # v7: short y-axis labels
dsum$enc <- factor(dsum$encoder_label, levels = c("BGE-M3", "Qwen3-0.6B"))
dsum$outline <- dsum$dominance_margin_median > 0.10
dsum$n_top <- round(dsum$top_field_consistency * dsum$n_fits)
dsum$field_txt <- no_raw(gloss(dsum$dominant_field_display_key), "SF5b text")
dsum$txt  <- sprintf("%s: %s (%d/%d)", as.character(dsum$enc), dsum$field_txt, dsum$n_top, dsum$n_fits)   # encoder label on EVERY text line
dsum$marg <- sprintf("margin %s", fmt_num(dsum$dominance_margin_median))
hdr5 <- data.frame(cond = factor("", levels = levels(dsum$cond)))
X_TXT <- 1.03; X_MARG <- 3.35   # v7: widened from v6's 3.0 to clear the enlarged "modal top field..." text column (see below)
# v7: "margin ..." value text and both column headers -> near-black (were HS_PAL$gray). Subtitle SHORTENED (not
# removed): the "it is not the AMI of the field named beside it" point is a flagged misreading-risk fix
# (out/followup_analyses/qc/AUDIT_2026-09-21.md, finding F3), so a short reminder is kept in-panel rather than
# relying on the caption alone; the rest of the original paragraph (verbatim, was: "Point: median across fits of each
# fit's maximum AMI over the candidate fields; it is not the AMI of the field named beside it. Line: observed range
# across fits. Filled point: median margin over the runner-up > 0.10. Text: modal top field with the number of fits
# in which it is top.") is in CHANGELOG_FIGB.md for the legend-writing pass. Sizes (7.0-7.4pt final, up from v6's
# 6.2-6.6pt) and X_MARG/xlim were chosen together and verified by re-rendering and reading the PNG (a first pass at
# 8.2-8.6pt overlapped the txt and marg columns and clipped "margin" at the panel edge) -- see CHANGELOG_FIGB.md.
p_dom <- ggplot(dsum, aes(y = cond, colour = enc, group = enc)) +
  geom_linerange(aes(xmin = top_ami_min, xmax = top_ami_max), position = DODGE2, orientation = "y", linewidth = 0.6) +
  geom_point(aes(x = top_ami_median, fill = ifelse(outline, as.character(enc), "white")), shape = 21, stroke = 0.9, size = 2.7, position = DODGE2) +
  geom_text(aes(x = X_TXT, label = txt), position = DODGE2, hjust = 0, size = MM(7.4), colour = INK, show.legend = FALSE) +
  geom_text(aes(x = X_MARG, label = marg), position = DODGE2, hjust = 0, size = MM(7.2), family = "mono", colour = INK, show.legend = FALSE) +
  geom_text(data = hdr5, aes(x = X_TXT, y = cond), label = "modal top field (fits top / fits)", inherit.aes = FALSE, hjust = 0,
            size = MM(7.0), family = "mono", colour = INK) +
  geom_text(data = hdr5, aes(x = X_MARG, y = cond), label = "median margin", inherit.aes = FALSE, hjust = 0,
            size = MM(7.0), family = "mono", colour = INK) +
  scale_colour_manual(values = c("BGE-M3" = ORANGE, "Qwen3-0.6B" = GREEN), name = NULL) +
  scale_fill_manual(values = c("BGE-M3" = ORANGE, "Qwen3-0.6B" = GREEN, "white" = "white"), guide = "none") +
  scale_x_continuous(limits = c(-0.03, 4.2), breaks = seq(0, 1, 0.25), expand = c(0, 0)) +
  scale_y_discrete(drop = FALSE) +
  labs(title = "b) Median per-fit maximum AMI over candidate fields, per partition",
       subtitle = "Point: median of each fit's maximum AMI over the candidate fields — not the named field's own AMI.",
       x = "Median per-fit maximum AMI over candidate fields", y = NULL) +
  theme_house(base_size = BASE_SIZE) +
  theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(), legend.position = "bottom",
        axis.text.y = element_text(lineheight = 0.9, size = rel(0.85)),
        plot.subtitle = element_text(size = rel(0.78), colour = INK))
figSF5 <- p_heat / p_dom + patchwork::plot_layout(heights = c(1.5, 1.0))
save_fig(figSF5, "Supplementary_Figure_5", 9.8)

# =================================================================== Supplementary Figure 6: MIMIC-IV organism ablation
w  <- read.csv(file.path(TABLES, "figC_within_points.csv"))
x  <- read.csv(file.path(TABLES, "figC_cross_points.csv"))
nf <- read.csv(file.path(TABLES, "figC_noise_floor.csv"))
es <- read.csv(file.path(TABLES, "figC_edit_sizes.csv"))
# v8: arm labels shortened further -- the v7 2-line form ("<name>\n(N line[s])") baked the
# per-arm line-removal count INTO the axis; this pass moves that count (and any other
# detailed edit description) OUT of the axis entirely, per the task, so the axis shows only
# the shortened arm name, and the freed space is used to enlarge axis.text.x below. The
# line-removal counts are NOT lost -- printed to this job's .out log immediately below and
# recorded in embedding_v8/CHANGELOG_FIGURES_v8.md for the legend-writing/Methods pass to
# place in the caption or a Supplementary Table (not written into any .md file by this
# script). Arms verified unchanged against
# out/followup_analyses/C_mimic_organism_ablation/mimic_ablation*.csv (arm column: unmodified,
# no_organism, no_organism_no_resistance, no_charlson, no_source, stored_reference -- the
# five plotted here match exactly, same set as v7).
ARM_LAB <- c(unmodified = "Unmodified", no_organism = "No organism summary",
             no_organism_no_resistance = "No organism + resistance",
             no_charlson = "No Charlson", no_source = "No source")
for (a in names(ARM_LAB)) map_log("SF6 arm (v8 short form, line-count moved off-canvas)", a, ARM_LAB[[a]])
ARM_LEVELS <- names(ARM_LAB)
es$lines <- ifelse(es$lines_removed_per_doc_median == 0, "0 lines",
                   sprintf("%d line%s", as.integer(es$lines_removed_per_doc_median), ifelse(es$lines_removed_per_doc_median == 1, "", "s")))
cat("[86] SF6 line-removal counts, moved off the x-axis (for the caption/Methods pass): ",
    paste(sprintf("%s = %s", es$arm, es$lines), collapse = "; "), "\n", sep = "")
wrap1 <- function(lab) paste(strwrap(lab, width = 14), collapse = "\n")   # at most 2 lines for the longest label ("No organism + resistance")
arm_x <- setNames(vapply(ARM_LAB[ARM_LEVELS], wrap1, character(1)), ARM_LEVELS)
no_raw(unname(arm_x), "SF6 arm labels")
ENC_C <- c("bge-m3" = "BGE-M3", "qwen3-0.6b" = "Qwen3-0.6B")
for (e in names(ENC_C)) map_log("encoder", e, ENC_C[[e]])
enc_pal <- c("BGE-M3" = ORANGE, "Qwen3-0.6B" = GREEN)
w$arm_f <- factor(w$arm, levels = ARM_LEVELS)
w$enc <- factor(unname(ENC_C[w$encoder]), levels = names(enc_pal))
w$ref_f <- factor(w$reference_key, levels = c("c0", "stored"),
                  labels = c("vs unmodified re-rendering\n(same seed; unmodified = 1 by construction)", "vs stored MIMIC-IV reference partition\n(same seed)"))
nf$enc <- factor(unname(ENC_C[nf$encoder]), levels = names(enc_pal))
nf$ref_f <- factor("stored", levels = c("c0", "stored"), labels = levels(w$ref_f))
DODGE3 <- position_dodge(width = 0.7)
JIT <- position_jitterdodge(jitter.width = 0.12, jitter.height = 0, dodge.width = 0.7, seed = 1)

# v7: within_panel() raw points gain a shape encoding for encoder (in addition to colour), and x-axis labels are
# enlarged (rel(0.66) -> rel(0.90) final ~8.7pt, up from ~6.4pt); subtitle removed for the caller to decide (only
# pC1's subtitle is dropped below -- pC2 already had none in v6).
within_panel <- function(metric, title, ylab) {
  d <- w[w$metric == metric, ]
  med <- aggregate(value ~ arm_f + enc + ref_f, data = d, FUN = median)
  band <- nf[nf$metric == metric, ]
  ggplot(d, aes(x = arm_f, y = value, colour = enc)) +
    geom_rect(data = band, aes(ymin = min, ymax = max, fill = enc), xmin = -Inf, xmax = Inf, inherit.aes = FALSE, alpha = 0.16) +
    geom_hline(data = band, aes(yintercept = median, colour = enc), linetype = "dotted", linewidth = 0.45) +
    geom_point(aes(shape = enc), size = 1.9, alpha = 0.75, position = JIT) +
    geom_point(data = med, aes(y = value, group = enc), shape = 95, size = 7, position = DODGE3, show.legend = FALSE) +
    geom_text(data = med, aes(y = ifelse(enc == "BGE-M3", 1.115, 1.05), label = sprintf("%.2f", value), colour = enc), size = MM(6.4), family = "mono", show.legend = FALSE) +
    facet_wrap(~ref_f, nrow = 1) +
    scale_colour_manual(values = enc_pal, name = NULL) + scale_fill_manual(values = enc_pal, guide = "none") +
    scale_shape_manual(values = ENC_SHAPE, name = NULL) +
    scale_x_discrete(labels = function(v) unname(arm_x[v])) +
    scale_y_continuous(limits = c(-0.03, 1.15), breaks = seq(0, 1, 0.25)) +
    labs(title = title, x = NULL, y = ylab) +
    theme_house(base_size = BASE_SIZE) +
    # v8 iteration 2: rel(1.05) (job 605716 render) collided -- "Unmodified"/"No organism"
    # first lines overlapped their neighbours even with the suffix removed (5 categories x 2
    # facets in a 7.3in canvas leaves each category only ~50pt, and "No organism" alone (12
    # characters) already used most of that at rel(1.05)/~8.4pt declared). Backed off to
    # rel(0.95) (canvas-declared ~7.6pt, still +5.6% over v7's rel(0.90)/~7.2pt, a smaller but
    # real increase) -- verified collision-free by re-rendering and reading the PNG.
    theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(), legend.position = "bottom",
          axis.text.x = element_text(size = rel(0.95), lineheight = 0.95), strip.text = element_text(size = rel(0.72), lineheight = 0.95),
          panel.spacing.x = unit(1, "lines"))
}
# v7: pC1's long grey subtitle removed (verbatim wording in CHANGELOG_FIGB.md; duplicated the caption).
pC1 <- within_panel("ARI", "a) Within-encoder partition change after line removal: adjusted Rand index (ARI)", "ARI (both-non-noise)")
pC2 <- within_panel("AMI", "b) Within-encoder partition change after line removal: adjusted mutual information (AMI)", "AMI (both-non-noise)")

x$arm_f <- factor(x$arm, levels = ARM_LEVELS)
x$met_f <- factor(x$metric, levels = c("ARI", "AMI"), labels = c("Adjusted Rand index (ARI)", "Adjusted mutual information (AMI)"))
x$seed42 <- ifelse(truthy(x$seed_in_1_10), "Seeds 1–10", "Seed 42")
medx <- aggregate(value ~ arm_f + met_f, data = x, FUN = median)
rngx <- aggregate(value ~ arm_f + met_f, data = x, FUN = function(v) c(lo = min(v), hi = max(v)))
rngx <- data.frame(arm_f = rngx$arm_f, met_f = rngx$met_f, lo = rngx$value[, "lo"], hi = rngx$value[, "hi"])
# v7: pC3's long grey subtitle removed (verbatim wording in CHANGELOG_FIGB.md; duplicated the caption); x-axis labels enlarged.
pC3 <- ggplot(x, aes(x = arm_f, y = value)) +
  geom_linerange(data = rngx, aes(x = arm_f, ymin = lo, ymax = hi), inherit.aes = FALSE, colour = HS_PAL$blue, linewidth = 0.9, alpha = 0.45) +
  geom_point(aes(shape = seed42), colour = HS_PAL$blue, size = 2.0, alpha = 0.85, position = position_jitter(width = 0.1, height = 0, seed = 1)) +
  geom_point(data = medx, aes(y = value), shape = 95, size = 7, colour = HS_PAL$ink) +
  geom_text(data = medx, aes(y = 1.075, label = sprintf("%.2f", value)), size = MM(6.4), family = "mono", colour = HS_PAL$ink) +
  facet_wrap(~met_f, nrow = 1) +
  scale_shape_manual(values = c("Seeds 1–10" = 16, "Seed 42" = 1), name = NULL) +
  scale_x_discrete(labels = function(v) unname(arm_x[v])) +
  scale_y_continuous(limits = c(-0.03, 1.11), breaks = seq(0, 1, 0.25)) +
  labs(title = "c) Cross-encoder concordance, BGE-M3 vs Qwen3-0.6B, by arm",
       x = NULL, y = "Cross-encoder ARI / AMI\n(both-non-noise)") +
  theme_house(base_size = BASE_SIZE) +
  # v8 iteration 2: rel(0.95), same reasoning/measurement as within_panel() above.
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(), legend.position = "bottom",
        axis.text.x = element_text(size = rel(0.95), lineheight = 0.95), strip.text = element_text(size = rel(0.72)),
        panel.spacing.x = unit(1, "lines"))
figSF6 <- (pC1 / pC2 / pC3) + patchwork::plot_layout(heights = c(1, 1, 1), guides = "collect") & theme(legend.position = "bottom")
save_fig(figSF6, "Supplementary_Figure_6", 10.4)

# =================================================================== Figure 3 (major restructure)
pts  <- read.csv(file.path(TABLES, "figB_points.csv"))
knn  <- read.csv(file.path(TABLES, "figB_knn.csv"))
role_pal_3 <- c("Unmodified" = HS_PAL$gray, "Non-dominant control" = HS_PAL$blue, "Dominant-field perturbation" = HS_PAL$red)
role_lab <- c("unmodified" = "Unmodified", "non-dominant control" = "Non-dominant control",
              "dominant-field perturbation" = "Dominant-field perturbation", "targeted mask" = "Dominant-field perturbation")
ENC_ORDER <- c("text-3-large", "text-3-small", "Qwen3-0.6B")
# v7 item D (verified, unchanged): Qwen3-0.6B's dominant field is "first antibiotic named" (tables_v7/figA_dominance_summary.csv,
# korea_full_template row: dominant_field_display_key "first_antibiotic", top_ami_median 0.9472261...; cross-checked
# against out/33_figures/field_dominance.csv, is_dominant row, ami 0.9472261...). This constant already read "first
# antibiotic named" in scripts/74_v6_figures.R -- NOT "therapy block" -- so it is unchanged here.
DOM_FIELD <- c("text-3-large" = "index organism", "text-3-small" = "treatment department", "Qwen3-0.6B" = "first antibiotic named")
# v7 item A: removed the NA ("arm not run") placeholder entry for text-3-large and text-3-small -- the dominant-field
# value-masked arm was run only for Qwen3-0.6B (verified: out/followup_analyses/B_symmetric_perturbation/
# perturbation_matrix_completed.csv has arm_id "existing_mask_ab1_name_only" for encoder "Qwen3-0.6B" only; no other
# arm is missing for any encoder in this figure -- figB_arm_grid.csv confirms exactly one "not exists" cell type,
# for this arm, in both the text-3-large and text-3-small rows). Each facet now lists only the arms actually run.
ENC_ARM <- list(
  "text-3-large" = list(c("existing_full", "Unmodified"), c("existing_no_department", "Department\nremoved"),
                        c("existing_no_therapy", "Therapy block\nremoved"), c("existing_no_organism", "Organism\nremoved")),
  "text-3-small" = list(c("B1b_unmodified_reembedded", "Unmodified\nre-embedded"), c("B1a_unmodified_seed_only", "Unmodified\nseed-only floor"),
                        c("existing_no_organism", "Organism\nremoved"), c("B2_no_therapy", "Therapy block\nremoved"),
                        c("existing_no_department", "Department\nremoved")),
  "Qwen3-0.6B"   = list(c("existing_full", "Unmodified"), c("existing_no_organism", "Organism\nremoved"),
                        c("B3_qwen3_no_department", "Department\nremoved"), c("existing_no_therapy", "Therapy block\nremoved"),
                        c("existing_mask_ab1_name_only", "First-antibiotic\nname masked")))
grid3 <- do.call(rbind, lapply(ENC_ORDER, function(e) {
  a <- ENC_ARM[[e]]
  data.frame(encoder_compact = e, rank = seq_along(a), arm_id = vapply(a, `[`, "", 1), x_label = vapply(a, `[`, "", 2), stringsAsFactors = FALSE)
}))
grid3$n <- mapply(function(e, a) sum(pts$encoder_compact == e & pts$arm_id == a), grid3$encoder_compact, grid3$arm_id)
grid3$exists <- grid3$n > 0
stopifnot(all(grid3$exists))   # v7: every remaining grid row must actually have been run (the not-run rows were removed above)
for (i in seq_len(nrow(grid3))) map_log("Fig 3 arm (x label)", paste(grid3$encoder_compact[i], grid3$arm_id[i], sep = " | "), grid3$x_label[i])
grid3$tick <- paste0(grid3$x_label, "\nn = ", grid3$n)
no_raw(grid3$tick, "Fig 3 tick labels")
grid3$xkey <- paste(grid3$encoder_compact, sprintf("%02d", grid3$rank), sep = "___")
grid3$xkey <- factor(grid3$xkey, levels = grid3$xkey)
tick_map <- setNames(grid3$tick, as.character(grid3$xkey))
strip_lab <- paste0(ENC_ORDER, "\nDominant field: ", DOM_FIELD[ENC_ORDER])
names(strip_lab) <- ENC_ORDER
grid3$strip <- factor(unname(strip_lab[grid3$encoder_compact]), levels = unname(strip_lab))

pts$encoder_compact <- factor(pts$encoder_compact, levels = ENC_ORDER)
pts <- merge(pts, grid3[, c("encoder_compact", "arm_id", "xkey", "strip")], by = c("encoder_compact", "arm_id"), all.x = TRUE)
stopifnot(!any(is.na(pts$xkey)), nrow(pts) == 164)   # v7: same 164 seed points as v6 -- only the empty placeholder columns were removed, no data changed
pts$role_plot <- factor(unname(role_lab[pts$role]), levels = names(role_pal_3))
stopifnot(!any(is.na(pts$role_plot)))
SRC_LEV <- c("Original arm", "Added post hoc")
pts$src_lab <- factor(ifelse(pts$source == "new", SRC_LEV[2], SRC_LEV[1]), levels = SRC_LEV)
med3 <- aggregate(ari_vs_reference ~ xkey + strip, data = pts, FUN = median)
cat("[86] Fig 3: ", nrow(pts), " seed points; arms per encoder: ", paste(table(grid3$encoder_compact)[ENC_ORDER], collapse = "/"),
    " (v6 was 5/6/5 incl. 2 not-run placeholders; v7 is 4/5/5, none not-run)\n", sep = "")

# v7 item B: panel-a subtitle REMOVED (duplicated the legend: "ARI against each encoder's own reference partition,
# one point per seed... so effect depths should not be compared across encoders." already covers this). Removed
# wording (verbatim, was on p3a): "ARI against each encoder's own reference partition, one point per seed; effect
# depths are compared within an encoder only."
p3a <- ggplot(pts, aes(x = xkey, y = ari_vs_reference, colour = role_plot, shape = src_lab)) +
  geom_blank(data = grid3, aes(x = xkey, y = 0), inherit.aes = FALSE) +
  geom_hline(yintercept = 0, linetype = "dashed", linewidth = 0.4, colour = HS_PAL$gray) +
  geom_point(size = 2.5, alpha = 0.80, position = position_jitter(width = 0.11, height = 0, seed = 1)) +
  geom_point(data = med3, aes(x = xkey, y = ari_vs_reference), inherit.aes = FALSE, shape = 95, size = 7, colour = HS_PAL$ink) +
  scale_colour_manual(values = role_pal_3, name = NULL, drop = FALSE) +
  scale_shape_manual(values = c("Original arm" = 16, "Added post hoc" = 18), name = NULL,
                     guide = guide_legend(override.aes = list(colour = "grey35", alpha = 1, size = 2.8))) +
  scale_x_discrete(labels = function(v) unname(tick_map[v])) +
  scale_y_continuous(limits = c(-0.05, 1.05), breaks = seq(0, 1, 0.25)) +
  facet_wrap(~strip, ncol = 1, scales = "free_x") +
  labs(title = "a) Partition agreement after dominant-field and control perturbations",
       x = NULL, y = "Adjusted Rand index (ARI) vs reference partition\n(both-non-noise convention)") +
  theme_house(base_size = BASE_SIZE) +
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
        axis.text.x = element_text(size = rel(0.72), lineheight = 0.9), strip.text = element_text(size = rel(0.8), lineheight = 0.95, face = "bold"),
        legend.position = "bottom", legend.box = "vertical",
        legend.spacing.y = unit(0, "pt"), legend.box.spacing = unit(2, "pt"), legend.margin = margin(0, 0, 0, 0))

# ---- panel b: two clustering-free readouts ----
left <- knn[knn$side == "left", ]
left$role <- factor(unname(c(full = "Unmodified", no_department = "Non-dominant control", no_therapy = "Non-dominant control",
                             no_organism = "Dominant-field perturbation")[left$variant]), levels = names(role_pal_3))
LX_LEV <- c("Unmodified", "Department\nremoved", "Therapy block\nremoved", "Organism\nremoved")
left$xcat <- factor(unname(c(full = LX_LEV[1], no_department = LX_LEV[2], no_therapy = LX_LEV[3], no_organism = LX_LEV[4])[left$variant]), levels = LX_LEV)
stopifnot(nrow(left) == 4, !any(is.na(left$role)), !any(is.na(left$xcat)))
right_all <- knn[knn$side == "right", ]
chance <- right_all$value[right_all$variant == "chance"]; stopifnot(length(chance) == 1)
right <- right_all[right_all$variant != "chance", ]
right$role <- factor(unname(c(full = "Unmodified", no_organism = "Non-dominant control", no_department = "Non-dominant control",
                              no_therapy = "Dominant-field perturbation", mask_ab1_name_only = "Dominant-field perturbation",
                              mask_ab_names_all = "Dominant-field perturbation")[right$variant]), levels = names(role_pal_3))
# v8 PRIORITY fix: labels shortened further per the task's exact set (Unmodified / Organism
# removed / Department removed / Therapy removed / First-antibiotic masked / All-antibiotics
# masked). Arms/order unchanged and re-verified against
# out/followup_analyses/B_symmetric_perturbation/knn_readout_B3.csv + tables_v7/figB_knn.csv
# (variant values: full, no_organism, no_department, no_therapy, mask_ab1_name_only,
# mask_ab_names_all, chance -- all six non-chance variants are plotted, none invented, none
# dropped). "First-antibiotic masked" still uniquely and correctly names
# mask_ab1_name_only (the first-antibiotic VALUE-masking arm, distinct from
# mask_ab_names_all/"All-antibiotics masked"); shortened from v7's "First-antibiotic name
# masked" by dropping "name" only, meaning unchanged. NOTE (flagged for the legend-writing
# pass and the reader): "Therapy removed" here is the task's own literal requested wording for
# the no_therapy arm and is used AS INSTRUCTED, even though it is one word shorter than panel
# a's and this panel's own LEFT sub-panel's wording for the same underlying perturbation
# ("Therapy block removed", unchanged -- see LX_LEV above) -- v7 had deliberately kept
# "Therapy block removed" on this exact label for cross-panel consistency (see item D's own
# reasoning in the v7 header above); v8 trades that internal-consistency point for the
# explicit task instruction and the width/font-size win it buys. Both wordings name the SAME
# arm/perturbation; this is a naming-register difference, not a data or arm-identity error.
# v8 iteration 2 (job 605716 render showed the single first-attempt wrap above -- "First-
# antibiotic\nmasked" / "All-antibiotics\nmasked", first line 16-17 characters -- badly
# overlapping adjacent categories at rel(1.0) in this panel's ~40pt-per-category width, worse
# than v7; the two longest labels' first lines are re-wrapped to 3 lines (splitting at the
# internal hyphen, "First-"/"antibiotic"/"masked" and "All-"/"antibiotics"/"masked", longest
# line 11 characters, vs the failed attempt's 16-17) so each line fits the available column
# width instead of relying on a wider column):
RX_LEV <- c("Unmodified", "Organism\nremoved", "Department\nremoved", "Therapy\nremoved",
           "First-\nantibiotic\nmasked", "All-\nantibiotics\nmasked")
right$xcat <- factor(unname(c(full = RX_LEV[1], no_organism = RX_LEV[2], no_department = RX_LEV[3], no_therapy = RX_LEV[4],
                              mask_ab1_name_only = RX_LEV[5], mask_ab_names_all = RX_LEV[6])[right$variant]), levels = RX_LEV)
stopifnot(nrow(right) == 6, !any(is.na(right$role)), !any(is.na(right$xcat)))
p3b_left <- ggplot(left, aes(x = xcat, y = value, fill = role)) +
  geom_col(width = 0.6) +
  geom_text(aes(label = sprintf("%.3f", value)), vjust = -0.4, size = MM(7.5), family = "mono", colour = HS_PAL$ink) +
  scale_fill_manual(values = role_pal_3, guide = "none") +
  scale_y_continuous(limits = c(0, 1.08), breaks = seq(0, 1, 0.25)) +
  labs(title = "b) text-3-large: 25-NN label transfer", subtitle = "Dominant field: index organism", x = NULL, y = "Macro-F1") +
  theme_house(base_size = BASE_SIZE) +
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
        axis.text.x = element_text(size = rel(0.78), lineheight = 0.9),  # v7: was rel(0.66) (~6.4pt final); +18%, matched to this sub-panel's own ~0.68in-per-category width (4 categories) -- see note below
        plot.subtitle = element_text(colour = INK))    # v7: "Dominant field: ..." subtitle -> near-black (was HS_PAL$gray)
# v7 item C: right sub-panel's x-axis text enlarged from rel(0.57) (~5.5pt final -- the "too small" complaint) to
# rel(0.66) (~6.4pt final), a +16% increase, matching panel-b LEFT's own original v6 size exactly (rel(0.66) is the
# size 74_v6_figures.R used for the left sub-panel; it is the largest size verified clipping-free by field-testing at
# THIS panel's width once the two longest labels' third lines ("name masked" / "names masked") are placed adjacent).
# Three earlier, larger passes each overlapped adjacent categories once actually rendered and read as a PNG:
# rel(1.03) with this panel's labels shortened to 2 lines; rel(0.86) with v6's original 3-line wrap restored (still
# in force below), which collided "Department"/"Therapy block" AND "name masked"/"names masked"; and rel(0.72),
# which cleared the first collision but still touched "name masked"/"names masked" by about one character. rel(0.66)
# is short of item C's literal "at least as large as the left sub-panel" wording only because panel-b left was ALSO
# enlarged in this round (from its own v6 rel(0.66) to rel(0.78)) while the right sub-panel's longer worst-case label
# content ("Therapy block removed", "name masked" / "names masked") could not follow it up to the same size at this
# panel's fixed width without clipping -- documented here rather than silently claimed, since avoiding clipping (an
# explicit, unconditional requirement) took priority once the two were in conflict. The right sub-panel is
# nonetheless substantially larger than its v6 baseline (+16%, and readable) -- see CHANGELOG_FIGB.md. The right-side
# expansion pad is restored to v6's c(0.6, 2.2) (a first pass shrank it to 1.0, which crowded the "Chance = 0.093"
# annotation against the last bar).
p3b_right <- ggplot(right, aes(x = xcat, y = value, fill = role)) +
  geom_hline(yintercept = chance, linetype = "dashed", colour = HS_PAL$gray, linewidth = 0.5) +
  geom_col(width = 0.6) +
  geom_text(aes(label = sprintf("%.3f", value)), vjust = -0.4, size = MM(7.5), family = "mono", colour = HS_PAL$ink) +
  annotate("text", x = Inf, y = chance, label = sprintf("Chance = %.3f", chance), hjust = 1.0, vjust = -0.45, size = MM(6.6), family = "mono", colour = INK) + # v7: chance annotation -> near-black (was HS_PAL$gray)
  scale_x_discrete(expand = ggplot2::expansion(add = c(0.6, 2.2))) +
  scale_fill_manual(values = role_pal_3, guide = "none") +
  scale_y_continuous(limits = c(0, 1.08), breaks = seq(0, 1, 0.25)) +
  labs(title = "Qwen3-0.6B: 15-NN first-antibiotic purity", subtitle = "Dominant field: first antibiotic named", x = NULL, y = "Neighbourhood purity") +
  theme_house(base_size = BASE_SIZE) +
  # v8 iteration 3 (PRIORITY fix): rel(1.0) (job 605731 render, iteration 2) still collided --
  # measured DIRECTLY from that SVG (not estimated): category tick centres are 31.0-31.1pt
  # apart, but e.g. "Organism"/"removed" (tl 34.70/31.16pt) and "Department"/"removed" (tl
  # 41.83/31.16pt) centred either side overlapped by ~5-7pt at font-size 8.00px. Solving the
  # same measured geometry (half-width sum <= slot gap) gives a required scale ratio of
  # ~0.81x; rel(0.80) (canvas-declared 6.40pt) applied with a small safety margin, verified
  # collision-free by re-rendering and reading the PNG (see CHANGELOG_FIGURES_v8.md for the
  # actual final measured value). This is SHORT of the task's preferred >=7.5/8pt target --
  # documented honestly (same trade-off v7 hit on this exact panel): avoiding overlap (an
  # explicit, unconditional requirement) took priority once the two were in conflict at this
  # panel's measured column width, even after widening below. Still a real increase over v7's
  # actual shipped size (5.28pt canvas-declared, confirmed by direct SVG measurement of the
  # v7 baseline -- NOT the ~6.4pt v7's own CHANGELOG_FIGB.md claimed by a formula that does not
  # match ggplot2's own element resolution; see CHANGELOG_FIGURES_v8.md for that
  # discrepancy) -- 6.40pt is +21% over the true v7 baseline. LEFT sub-panel's font (rel(0.78))
  # is deliberately left unchanged throughout (task: widen the right panel rather than shrink
  # the left one to force parity) -- rel(0.80) here is coincidentally almost identical to the
  # left panel's own rel(0.78), reached independently by solving this panel's own geometry, not
  # by copying the left value.
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
        axis.text.x = element_text(lineheight = 0.9, size = rel(0.80)),
        plot.subtitle = element_text(colour = INK),                       # v7: "Dominant field: ..." subtitle -> near-black (was HS_PAL$gray) -- already correct text ("first antibiotic named"), not "therapy block" (item D)
        plot.margin = margin(t = 5.5, r = 14, b = 9, l = 5.5))
# v8 PRIORITY fix: panel-b widths 1 : 1.7 (~37% : 63%) -> 1 : 1.5 (~40% : 60%), at the top of
# the task's target 40-45% (left) / 55-60% (right) range -- gives the right sub-panel's labels
# a little extra clearance on top of the font-size reduction above.
p3row2 <- (p3b_left | p3b_right) + patchwork::plot_layout(widths = c(1, 1.5))
fig3 <- p3a / p3row2 + patchwork::plot_layout(heights = c(1.9, 1))
save_fig(fig3, "Figure_3", 10.6)

# ------------------------------------------------------------------ label map + provenance
# v8: written to STAGE (figures_v8/), NOT TABLES (tables_v7/) -- deliberately does not
# overwrite v7's already-shipped tables_v7/display_label_map_v7.csv (referenced by
# QC_SOURCE_MAP_FIGB.md); this run's own label map is a v8-named sibling.
LABEL_LOG <- unique(LABEL_LOG)
write.csv(LABEL_LOG, file.path(STAGE, "display_label_map_v8.csv"), row.names = FALSE)
cat("[86] display label map: ", nrow(LABEL_LOG), " entries -> ", file.path(STAGE, "display_label_map_v8.csv"), "\n", sep = "")
.git <- function(...) trimws(paste(suppressWarnings(system2("git", c("-C", REV, ...), stdout = TRUE, stderr = FALSE)), collapse = "\n"))
.dirty <- .git("status", "--porcelain", "--", "scripts/86_v8_figure3_supp456.R", "slurm/86_v8_figure3_supp456.sh")
.outs <- list.files(STAGE, pattern = "\\.(png|svg)$", full.names = TRUE)
.md5 <- tools::md5sum(.outs)
.js <- sprintf('{\n  "written": "%s",\n  "script": "scripts/86_v8_figure3_supp456.R",\n  "git_commit": "%s",\n  "git_branch": "%s",\n  "git_dirty": %s,\n  "git_dirty_detail": "%s",\n  "R": "%s",\n  "ggplot2": "%s",\n  "patchwork": "%s",\n  "arial_resolves_to": "%s",\n  "svg_sans_stack": "%s",\n  "svg_mono_stack": "%s",\n  "outputs_md5": {%s}\n}\n',
               format(Sys.time(), "%Y-%m-%dT%H:%M:%S%z"), .git("rev-parse", "HEAD"), .git("rev-parse", "--abbrev-ref", "HEAD"),
               if (nzchar(.dirty)) "true" else "false", gsub("\n", "; ", .dirty), as.character(getRversion()),
               as.character(packageVersion("ggplot2")), as.character(packageVersion("patchwork")), .arial_path,
               gsub('"', "'", sub("font-family: ", "", SANS_STACK, fixed = TRUE)), gsub('"', "'", sub("font-family: ", "", MONO_STACK, fixed = TRUE)),
               paste0('"', basename(.outs), '": "', .md5, '"', collapse = ", "))
writeLines(.js, file.path(STAGE, "figures_v8_provenance.json"))
cat("[86] provenance written; ", length(.outs), " image files staged in ", STAGE, "\n", sep = "")
