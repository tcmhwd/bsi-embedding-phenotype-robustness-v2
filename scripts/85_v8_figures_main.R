#!/usr/bin/env Rscript
# Paper 1 figures -- V8 TARGETED POLISH PASS (2026-09-22), built on the v7
# readability pass below. NEW, additive script -- does NOT edit
# scripts/79_v7_figures_main.R (or 36/33/34/35_figures*.R, or
# 74_v6_figures.R) in place; those remain frozen "before" records. v8 changes
# on top of v7 (see embedding_v8/CHANGELOG_FIGURES_v8.md for full detail):
#   - Fig. 1c: a few points more space between the panel title and the
#     rotated y-axis title (plot.title bottom margin), plus
#     patchwork::free(..., side = "l") on panel c only so its panel isn't
#     forced to inherit panels a/b's much wider left margin (their long
#     categorical row labels) -- this is the fix for the "title crowds the
#     y-axis label" / "excess blank space on the left" report AND is the
#     same defect as the cosmetic 1-line overlap flagged at the end of v7.
#     No data, tick value, or axis-title wording touched.
#   - Fig. 2a: South Korea/MIMIC-IV x-axis tick labels shortened to compact
#     forms (3-large/3-small/BGE-M3<dagger>/Qwen3-0.6B and BGE-M3/Qwen3-0.6B)
#     to fix a labelled collision with the axis title; angle reverted
#     0->30->0 (horizontal) now that the compact forms fit; one in-canvas
#     caption-style dagger footnote added to panel a (same mechanism Fig. 4a
#     already used for its own dagger footnote in v7: labs(caption=...) +
#     plot.caption coloured HS_PAL$ink). No cell colour/value or dominant-
#     field marking touched.
#   - Figure 1, 2, 4 and Supplementary 1-3 are otherwise byte-for-byte the
#     same drawing code as scripts/79_v7_figures_main.R (re-run for
#     consistency / diff-check only, no other change) -- see v7's own header
#     note immediately below for that pass's own history.
#
# ---------------------------------------------------------------------------
# v7 header (unchanged, retained for provenance):
# Paper 1 figures -- V7 READABILITY / PRESENTATION PASS (2026-09-22).
#
# This is a NEW, additive script. It does NOT edit scripts/36_figures_v4.R
# (or 33/34/35_figures*.R, or 74_v6_figures.R) in place -- those remain the
# frozen "before" record for their respective revisions. It reads the SAME
# tidy CSVs already on disk under out/33_figures/ (agreement.csv,
# encoder_degeneracy.csv, field_dominance.csv, conditional_agreement.csv,
# retrieval_ladder_v4.csv, retrieval_bootstrap_contrasts.csv, variants.csv,
# token_lookup.csv, size_stability.csv) -- NO new computation, NO new
# statistics, NO embedding/clustering/API job. Every number this script
# prints is copied verbatim (same read.csv() column, same sprintf()) from
# scripts/36_figures_v4.R; only STYLE changed. See QC_SOURCE_MAP_FIGA.md
# and CHANGELOG_FIGA.md (submission_build_v7/) for the itemised list and
# the verification evidence.
#
# Scope: Figure 1, Figure 2, Figure 4, Supplementary Figure 1, 2, 3 only.
# Figure 3 / Supplementary Figure 4-6 are out of scope for this pass (built
# separately by scripts/74_v6_figures.R; another agent is revising those) --
# this script does not read ablation_path / the retrieval_ladder.csv (v3,
# 5-rung) / size_stability inputs Figure 3 uses, and never writes
# fig3_ablation.* or figS4-6.
#
# v7 changes on top of the v4/v6 rendering (readability/presentation pass
# only -- see the task's exact change list, reproduced per-figure below):
#   - Header/explanatory annotation text (e.g. "ARI median [observed
#     range]", "AMI median", "AMI: unconditional -> within-organism",
#     "ΔAUROC [95% bootstrap CI]", the Fig. 4 dagger footnote,
#     "Unmodified seed range", "Pre-specified lookup F1 ...", "Spearman
#     rho [95% bootstrap CI]", "Holm-adjusted P") recoloured from
#     HS_PAL$gray (#8A8578, theme_house()'s default caption/subtitle grey)
#     to HS_PAL$ink (#16150F, near-black) -- these carry SEMANTIC content,
#     not decoration. Numeric data-value annotations in this source were
#     already HS_PAL$ink (verified against the v6 SVGs before touching
#     anything -- see QC_SOURCE_MAP_FIGA.md); no separate "numeric" edit
#     was required beyond the header/explanatory strings above.
#   - Decorative/categorical grey is explicitly KEPT: grid lines, axis
#     guide lines, the Fig. 1c reference axis, the Fig. 2 "not rendered in
#     this cohort" tiles/legend swatch (categorical, not decorative), the
#     SF1 reference-band FILL and its dotted boundary lines, and the Fig. 4
#     comparator DATA POINTS (colour = "Comparator" role, not text).
#   - Fig. 1c y-axis label: "Largest cluster (% of corpus)" ->
#     "Largest-cluster share (%)" (more concise, same meaning; no numbers
#     touched).
#   - Fig. 2a dominant-field outline colour: checked for WCAG contrast and
#     simulated colour-vision-deficiency (deutan/protan/tritan) legibility
#     against the AMI fill gradient it sits on; kept HS_PAL$red only if it
#     passes (see the printed [v7 fig2 QC] block in this job's .out log for
#     the actual numbers, and CHANGELOG_FIGA.md for the verdict).
#   - Text sizing raised throughout so FINAL PRINTED size (after the
#     working-canvas -> PRINT_MM=180mm scale-down, i.e. multiplied by
#     SCALE, exactly as the v4/v6 script's own MM()/BASE_SIZE mechanism
#     already did) lands at: panel titles ~12-14pt, axis titles ~11-12pt,
#     tick labels ~10-11pt, annotation/legend text >=10pt, nothing <9pt.
#     A first attempt (this pass, render 604960/604963) raised BASE_SIZE and
#     every MM(x) annotation target on the ORIGINAL 7.3in canvas and widened
#     the fixed-x annotation-column data-coordinates (X_F1_AMI etc.) to make
#     room -- rendering that revealed a real bug: widening a DATA-coordinate
#     xlim on a FIXED-width canvas shrinks inches-per-data-unit for the
#     WHOLE panel (including the pre-existing 0-1 axis ticks), so the tick
#     labels collided worse than before (visual QC read of the PNG, not
#     just the SVG text list, is what caught this). Fixed (render 604967+)
#     by leaving every fixed-x annotation-column constant at its EXACT v4
#     value and instead growing the WORKING CANVAS (WIDTH_IN 7.3 -> 11.0in,
#     GROWTH = 1.507) with BASE_SIZE now COMPUTED from SCALE
#     (TARGET_AXIS_TITLE_PT / (0.85 * SCALE)) rather than a hardcoded
#     constant, so the calibration is self-consistent at any WIDTH_IN;
#     MM(10) needed no change since it already divides by SCALE. Every
#     figure's save height grown by the same GROWTH factor for matching
#     vertical breathing room. A THIRD render (604971) still showed the
#     fixed-x annotation columns colliding for four panels (Fig. 1 a/b, Fig.
#     2b, SF2, SF3): measuring their SVGs' own textLength/tick-spacing
#     (exact, not estimated) and solving the panel's own layout equation
#     (panel_px = axis_pad*px_per_unit + column_text_px + gap/margin_px,
#     px_per_unit = panel_px/xlim_range, panel_px confirmed CONSTANT across
#     renders) showed hitting the full 10pt annotation target on these four
#     panels needs an xlim span of roughly 3x-7x the v4 original -- not a
#     legible design at ANY canvas width, since WIDTH_IN cancels out of that
#     ratio. These four panels' bracketed-CI mono columns therefore use
#     MM(9) (the task's own explicit floor, "nothing below 9pt"), solved at
#     that floor for each panel's own measured geometry, with matching
#     xlim/column-position widening and reduced tick count. Fig. 4b and SF1
#     (each carrying only ONE such column, not two side by side) DO reach
#     the full MM(10) target with a modest (~20%) xlim increase -- solved
#     the same way, confirmed feasible. See QC_SOURCE_MAP_FIGA.md /
#     CHANGELOG_FIGA.md for the full per-panel numbers and all four render
#     attempts.
#   - Colour-blind safety: two colour-only distinctions that carry a
#     scientifically important reading and had no other visual encoding
#     were given an additional SHAPE encoding: Fig. 1's
#     Cross-encoder-vs-Within-encoder role (panels a/b), Fig. 1c's
#     Degenerate-vs-Non-degenerate status, and SF1's three-role grammar
#     (Unmodified reference / Order control / Other exploratory variant).
#     Fig. 2b's Unconditional/Within-organism-strata pair and Fig. 4's
#     pool status (unrestricted/past-admissions-only) already carried a
#     shape encoding in the v4/v6 source and were left alone. Fig. 4's
#     Structured/Embedding/Comparator colour grouping was NOT given an
#     additional shape: every row already carries an explicit text label
#     naming its category (e.g. "SOFA score", "Embedding retrieval
#     (BGE-M3, k = 25)"), so colour there is redundant reinforcement, not
#     the sole carrier of the distinction, and the shape channel in that
#     panel is already occupied by pool status -- overloading it risked a
#     confusing double-meaning for a cosmetic-only pass.
#
# No CSV under out/33_figures/ is read differently, modified, or
# regenerated by this script. No `write.csv()` call exists in this file.

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  source("~/R/house_style.R")
})

REV  <- "/data/tcmhwd/AI/AST_ai/phenotype_revision"
DATA <- file.path(REV, "out/33_figures")
FIGS <- file.path(REV, "out/followup_analyses/figures_v8")  # v8: own stage dir, matches slurm/85_v8_figures_main.sh's STAGE;
                                                          # deliberately NOT figures_v7 (that dir is shared with
                                                          # scripts/79_v7_figures_main.R and its wrapper wipes it with
                                                          # `rm -f *.png *.svg` on every run -- reusing it here would
                                                          # race/clobber a v7 rebuild running concurrently)
dir.create(FIGS, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------- Sizing
# v7 fix (after a first render showed x-axis tick labels and fixed-x annotation columns
# colliding, QC/79 render 604963->604960 log + visual read of the PNGs): widening the
# DATA-coordinate xlim/column positions on a FIXED-width canvas is backwards -- it shrinks
# inches-per-data-unit for the WHOLE panel (including the pre-existing 0-1 axis ticks),
# making the collision worse, not better. The correct lever is the physical canvas
# (WIDTH_IN): grow it, and let BASE_SIZE/MM() (which both divide by SCALE, i.e. already
# self-correct for canvas width) keep the FINAL PRINTED size at the same target regardless
# of how wide the working canvas is. Every fixed-x annotation-column constant below
# (X_F1_AMI, Fig. 2b/4b/SF1/SF2/SF3's column x/limits) is therefore left at its EXACT v4
# value -- only WIDTH_IN, BASE_SIZE and each figure's save-height changed.
GROWTH    <- 1.507                        # WIDTH_IN growth factor (7.3in -> 11.0in)
WIDTH_IN  <- 11.0                         # v7: 7.3 -> 11.0in working canvas (still shrinks to
                                           # the SAME 180mm/~7.087in print width; SCALE below
                                           # just does more of the work now)
PRINT_MM  <- 180                          # unchanged -- target journal print width
SCALE     <- PRINT_MM / (WIDTH_IN * 25.4) # unchanged formula, now ~0.644
TARGET_AXIS_TITLE_PT <- 11.5              # v7: mid of the requested 11-12pt axis-title range
BASE_SIZE <- TARGET_AXIS_TITLE_PT / (0.85 * SCALE)  # v7: computed, not hardcoded -- theme_house()'s
                                           # axis.title uses rel(0.85) of base_size, so solving for
                                           # BASE_SIZE here pins axis.title (and, via the theme's
                                           # OTHER fixed rel() ratios, plot.title/axis.text/
                                           # strip.text/legend.text together) at the printed target
                                           # regardless of WIDTH_IN -- see header note for the
                                           # verified final-pt numbers (title ~13.5, axis.title
                                           # ~11.5, axis.text ~10.8, all within the requested ranges)
MM <- function(final_pt) final_pt / (2.8346 * SCALE)  # unchanged formula (already self-corrects for SCALE)

INK <- HS_PAL$ink  # near-black ink used for every recoloured semantic annotation below

# ---------------------------------------------------------------- Formatting (unchanged from v4)
fmt_num  <- function(x) gsub("-", "−", sprintf("%.3f", x), fixed = TRUE)
fmt_num2 <- function(x) gsub("-", "−", sprintf("%.2f", x), fixed = TRUE)

X_F1_ARI <- 1.02   # unchanged from v4 -- deliberately NOT widened (see Sizing note above)
# v7 Fig. 1 panels a/b are the ONE genuinely infeasible case in this pass: they carry TWO
# side-by-side bracketed-CI mono columns (ARI, AMI) next to already-long row labels
# ("Between encoders (text-3-small vs text-3-large)"), and solving the panel's own layout
# equation -- panel_px = axis_pad*px_per_unit + col1_text_px + gap_px + col2_text_px +
# margin_px, with px_per_unit = panel_px/xlim_range -- for a FIXED panel_px (confirmed
# constant at 334.6px across two actual renders: 133.85*2.5 = 97.0*3.45 = 334.6) shows that
# hitting the full 10pt annotation target requires xlim_range ~= 17.9 (the 0-1 ARI axis
# would occupy under 6% of the panel) -- not a legible design at any canvas width, per the
# Sizing note's algebra (WIDTH_IN cancels out of this ratio). Two widening attempts
# (604966, 604971 -- see QC_SOURCE_MAP_FIGA.md) confirmed the collision empirically before
# this was solved analytically. Per the task's own "annotation text >=10pt WHEREVER
# POSSIBLE" / "nothing below 9pt" rule, panels a/b's annotation columns use MM(9) (the hard
# floor, not 10) -- the smallest compromise that still complies -- solving the SAME layout
# equation at t=9 for the panel's own X_F1_AMI/xlim (below) instead of guessing.
X_F1_AMI <- 5.25   # v7: 1.97 -> 5.25 -- see the MM(9) note above; solved from the same
                    # measured panel_px=334.6 and this render's actual textLength values


ORANGE <- "#D98A4A"  # HS_PAL$groups[2] -- unchanged
NOT_RENDERED <- "#B9BDC0"   # unchanged -- categorical grey, deliberately kept (task instruction)

# ---- Fig. 2 dominant-field outline: WCAG contrast + colour-vision-deficiency check ----
# No data dependency -- pure colour math against the fixed palette endpoints, so it is
# computed once up top and only the resulting `dominant_outline_col`/`_lwd` are used below.
.srgb_lin <- function(x) { x <- x / 255; ifelse(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055)^2.4) }
.rel_lum  <- function(hex) {
  rgb <- grDevices::col2rgb(hex)[, 1]
  lin <- .srgb_lin(rgb)
  0.2126 * lin[1] + 0.7152 * lin[2] + 0.0722 * lin[3]
}
.contrast <- function(hex1, hex2) {
  L1 <- .rel_lum(hex1); L2 <- .rel_lum(hex2)
  (max(L1, L2) + 0.05) / (min(L1, L2) + 0.05)
}
fill_lo  <- HS_PAL$mid   # AMI ~ 0 end of the Fig. 2a gradient (scale_fill_gradient low)
fill_hi  <- HS_PAL$blue  # AMI ~ 1 end of the gradient (scale_fill_gradient high)
fill_mid <- grDevices::colorRampPalette(c(HS_PAL$mid, HS_PAL$blue))(3)[2]  # AMI ~ 0.5
wcag_min <- 3.0  # WCAG 1.4.11 non-text (graphical-object) contrast minimum

normal_contrasts <- c(lo = .contrast(HS_PAL$red, fill_lo),
                      mid = .contrast(HS_PAL$red, fill_mid),
                      hi = .contrast(HS_PAL$red, fill_hi))
cat(sprintf("[v7 fig2 QC] dominant-outline %s vs fill -- normal-vision WCAG contrast: lo=%.2f mid=%.2f hi=%.2f (min required %.1f)\n",
            HS_PAL$red, normal_contrasts["lo"], normal_contrasts["mid"], normal_contrasts["hi"], wcag_min))

cvd_min <- NA_real_
if (hs_has("colorspace")) {
  for (cvd_fn in list(deutan = colorspace::deutan, protan = colorspace::protan, tritan = colorspace::tritan)) {
    sim <- cvd_fn(c(outline = HS_PAL$red, lo = fill_lo, mid = fill_mid, hi = fill_hi))
    c_lo  <- .contrast(sim["outline"], sim["lo"])
    c_mid <- .contrast(sim["outline"], sim["mid"])
    c_hi  <- .contrast(sim["outline"], sim["hi"])
    cvd_min <- min(c(cvd_min, c_lo, c_mid, c_hi), na.rm = TRUE)
    cat(sprintf("[v7 fig2 QC] %s simulation -- outline vs fill contrast: lo=%.2f mid=%.2f hi=%.2f\n",
                deparse(substitute(cvd_fn)), c_lo, c_mid, c_hi))
  }
} else {
  cat("[v7 fig2 QC] colorspace package not installed in this env -- CVD simulation skipped; ",
      "verdict based on normal-vision WCAG contrast only (red vs blue is not a red-green pair, ",
      "the failure mode CVD simulation would catch)\n", sep = "")
}
outline_fails <- (min(normal_contrasts) < wcag_min) || (!is.na(cvd_min) && cvd_min < wcag_min)
if (outline_fails) {
  dominant_outline_col <- HS_PAL$ink
  dominant_outline_lwd <- 1.4
  cat("[v7 fig2 QC] VERDICT: dominant-field outline FAILED the contrast/CVD check -- switched HS_PAL$red -> HS_PAL$ink and widened 1.1 -> 1.4\n")
} else {
  dominant_outline_col <- HS_PAL$red
  dominant_outline_lwd <- 1.1
  cat("[v7 fig2 QC] VERDICT: dominant-field outline PASSED the contrast/CVD check -- kept HS_PAL$red at linewidth 1.1 (unchanged)\n")
}

# =================================================================== Figure 1
# "Fig. 1 | High within-encoder reproducibility does not imply cross-encoder
# concordance." Panel a: South Korea, non-degenerate pairs only. Panel b:
# MIMIC-IV (positive control). Panel c: which Korean encoder fits are
# degenerate, and why BGE-M3 vs Qwen3 is excluded from panel a rather than
# shown as the cross-encoder headline.
f1 <- read.csv(file.path(DATA, "agreement.csv"))

LBL_F1 <- c(
  "Qwen3-0.6B"         = "Qwen3-0.6B",
  "3-large"            = "text-3-large",
  "3-small"            = "text-3-small",
  "Qwen3 vs 3-small"   = "Qwen3-0.6B vs text-3-small",
  "Qwen3 vs 3-large"   = "Qwen3-0.6B vs text-3-large",
  "3-small vs 3-large" = "text-3-small vs text-3-large",
  "BGE-M3"             = "BGE-M3",
  "BGE-M3 vs Qwen3"    = "BGE-M3 vs Qwen3-0.6B")
stopifnot(all(f1$label %in% names(LBL_F1)))
f1$label <- unname(LBL_F1[f1$label])

fA <- f1[f1$panel == "A", ]
fA$role <- ifelse(fA$comparison == "Cross-encoder", "Cross-encoder", "Within-encoder (cross-seed)")
fA$row_label <- ifelse(fA$comparison == "Cross-encoder",
                        paste0("Between encoders (", fA$label, ")"),
                        paste0("Between seeds (", fA$label, ")"))
row_order_A <- c("Between seeds (Qwen3-0.6B)", "Between seeds (text-3-large)",
                 "Between seeds (text-3-small)",
                 "Between encoders (Qwen3-0.6B vs text-3-small)",
                 "Between encoders (Qwen3-0.6B vs text-3-large)",
                 "Between encoders (text-3-small vs text-3-large)")
fA$annot_ari <- sprintf("%s [%s, %s]", fmt_num(fA$estimate), fmt_num(fA$lo), fmt_num(fA$hi))
fA$annot_ami <- fmt_num(fA$ami)

fB <- f1[f1$panel == "B", ]
fB$role <- ifelse(fB$comparison == "Cross-encoder", "Cross-encoder", "Within-encoder (cross-seed)")
fB$row_label <- ifelse(fB$comparison == "Cross-encoder",
                        paste0("Between encoders (", fB$label, ")"),
                        paste0("Between seeds (", fB$label, ")"))
row_order_B <- c("Between seeds (BGE-M3)", "Between seeds (Qwen3-0.6B)",
                 "Between encoders (BGE-M3 vs Qwen3-0.6B)")
fB$annot_ari <- sprintf("%s [%s, %s]", fmt_num(fB$estimate), fmt_num(fB$lo), fmt_num(fB$hi))
fB$annot_ami <- fmt_num(fB$ami)

role_pal_1 <- c("Cross-encoder" = HS_PAL$blue, "Within-encoder (cross-seed)" = ORANGE)
# v7: colour-blind safety -- role was colour-only (shape was fixed = 16 for both roles) even
# though every row's text label already states "Between encoders"/"Between seeds". Adding a
# distinct shape lets a reader who cannot use colour at all still tell the two point clouds
# apart at a glance, not just by re-reading each row label.
role_shp_1 <- c("Cross-encoder" = 17, "Within-encoder (cross-seed)" = 16)

.panel_ab <- function(d, title, row_levels) {
  d$row_label <- factor(as.character(d$row_label), levels = c(row_levels, ""))
  stopifnot(!any(is.na(d$row_label)))
  ggplot(d, aes(y = row_label)) +
    geom_vline(xintercept = 0, colour = HS_PAL$edge, linewidth = 0.3) +
    geom_segment(aes(x = lo, xend = hi, yend = row_label, colour = role),
                 linewidth = 2.2, lineend = "round") +
    geom_point(aes(x = estimate, colour = role, shape = role), size = 2.9) +
    # v7: MM(9), not MM(10) -- see the X_F1_AMI note above (the one annotation column in
    # this whole pass that cannot reach 10pt within a legible layout; 9pt is the task's own
    # explicit floor, applied here deliberately, not an oversight).
    geom_text(aes(x = X_F1_ARI, label = annot_ari), hjust = 0, size = MM(9),
              family = "mono", colour = INK) +
    geom_text(aes(x = X_F1_AMI, label = annot_ami), hjust = 0, size = MM(9),
              family = "mono", colour = INK) +
    # v7 (a): header colour grey -> near-black -- these state what the two result
    # columns ARE (units/definition), a semantic label, not decoration.
    annotate("text", x = X_F1_ARI, y = "", label = "ARI median\n[observed range]",
             hjust = 0, vjust = 0.5, size = MM(9), family = "mono",
             colour = INK, lineheight = 0.95) +
    annotate("text", x = X_F1_AMI, y = "", label = "AMI median\n ",
             hjust = 0, vjust = 0.5, size = MM(9), family = "mono",
             colour = INK, lineheight = 0.95) +
    scale_y_discrete(drop = FALSE) +
    scale_colour_manual(values = role_pal_1, name = NULL) +
    scale_shape_manual(values = role_shp_1, name = NULL) +
    # v7: only the axis extremes (was seq(0,1,0.25), 5 ticks in v4) -- at this xlim range
    # (needed for the MM(9) annotation columns above) the 0-1 ARI axis occupies a small
    # fraction of the panel, so even 3 ticks would crowd; the 2 endpoint ticks stay legible.
    scale_x_continuous(breaks = c(0, 1)) +
    coord_cartesian(xlim = c(-0.05, 7.40), clip = "off") +
    labs(title = title, x = "Adjusted Rand index (ARI)", y = NULL) +
    theme_house(base_size = BASE_SIZE) +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
          legend.position = "bottom")
}
p1a <- .panel_ab(fA, "a) South Korea (n-matched, n = 16,144)", rev(row_order_A))
p1b <- .panel_ab(fB, "b) MIMIC-IV (n = 16,144)", rev(row_order_B))

f1c <- read.csv(file.path(DATA, "encoder_degeneracy.csv"))
stopifnot(all(f1c$label %in% names(LBL_F1)))
f1c$label <- unname(LBL_F1[f1c$label])
f1c$label <- factor(f1c$label,
                    levels = c("BGE-M3", "Qwen3-0.6B", "text-3-small", "text-3-large"))
f1c$status <- ifelse(f1c$degenerate, "Degenerate", "Non-degenerate")
# v7: colour-blind safety -- Degenerate vs Non-degenerate is THE finding of this panel and
# was colour-only (fixed shape = 16 for both); no axis label states "degenerate" anywhere
# else on the canvas. Triangle vs circle carries the distinction without colour.
status_shp_1c <- c("Degenerate" = 17, "Non-degenerate" = 16)

p1c <- ggplot(f1c, aes(x = label, y = largest_share_median_refits, colour = status, shape = status)) +
  geom_hline(yintercept = 0, colour = HS_PAL$gray, linewidth = 0.3) +
  geom_segment(aes(xend = label, y = 0, yend = largest_share_median_refits), linewidth = 3.0,
               lineend = "round", alpha = 0.35) +
  geom_point(size = 3.2) +
  geom_text(aes(label = sprintf("%.1f%%\n(k = %.0f)", largest_share_median_refits * 100, k_median_refits)),
            vjust = -0.25, hjust = 0.5, size = MM(10), family = "mono", colour = INK,
            lineheight = 0.95) +
  scale_colour_manual(values = c("Degenerate" = HS_PAL$gray, "Non-degenerate" = HS_PAL$blue), name = NULL) +
  scale_shape_manual(values = status_shp_1c, name = NULL) +
  scale_x_discrete(expand = ggplot2::expansion(add = c(0.7, 0.7))) +
  scale_y_continuous(limits = c(0, 1.42), breaks = seq(0, 1, 0.25),
                     labels = scales::percent_format(accuracy = 1)) +
  # v7 (c): more concise, equally precise axis label (no number touched).
  labs(title = "c) South Korea encoder degeneracy", x = NULL,
       y = "Largest-cluster share (%)") +
  theme_house(base_size = BASE_SIZE) +
  # v7: x-axis category labels ("BGE-M3", "text-3-small", "text-3-large", ...) overlapped
  # at the larger axis.text target (same fix as Fig. 2a's facet x-axis, confirmed by
  # reading the rendered PNG).
  # v8: plot.title given extra bottom margin -- at the larger BASE_SIZE the bold panel
  # title ("c) South Korea encoder degeneracy") sat close enough to the top of the rotated
  # y-axis title ("Largest-cluster share (%)") to visually crowd/overlap by about a line;
  # this is the SAME defect flagged as a known cosmetic 1-line overlap at the end of the v7
  # cycle. Y-axis title text is untouched (still exactly "Largest-cluster share (%)").
  theme(panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
        legend.position = "bottom",
        axis.text.x = element_text(angle = 20, hjust = 1, vjust = 1),
        # v8: margin(b=8) (canvas pt) measured (job 605715 SVG) as NOT enough -- the title's
        # own glyph box (baseline y=767.64, font-size 21px, so top edge ~y=750.8) still
        # overlapped the rotated y-axis title's top run-end (~y=758, well inside the title's
        # vertical span). Raised to margin(b=30) canvas pt (~19pt at this figure's SCALE
        # ~0.644) to clear a real measured gap, not a guessed one -- re-verified after
        # re-rendering.
        plot.title = element_text(margin = margin(b = 30)))

# v8: free panel c's LEFT side from patchwork's cross-panel width alignment. Panels a/b
# carry long categorical row labels ("Between encoders (text-3-small vs text-3-large)") that
# force a wide left margin; by default patchwork unions that margin across the whole stacked
# column, so panel c -- which has no such row labels (its y axis is a plain 0-100% scale) --
# inherited the same wide left margin, showing up as excess blank space to the left of its
# own plotting area / a right-shifted panel. patchwork::free(type = "panel") releases just
# that one panel's width from the cross-panel union without touching any other panel's
# alignment, any data, or any tick value -- a layout-only fix (patchwork >= 1.1.2 API).
p1 <- (p1a / p1b / patchwork::free(p1c, type = "panel", side = "l")) +
  patchwork::plot_layout(heights = c(2.2, 1.35, 1.3))

hs_save(p1, file.path(FIGS, "fig1_agreement.png"), width = WIDTH_IN, height = 14.8, dpi = 300)
hs_save_vector(p1, file.path(FIGS, "fig1_agreement.svg"), width = WIDTH_IN, height = 14.8)
cat("[v8 fig1] written\n")

# =================================================================== Figure 2
# "Fig. 2 | Encoder-derived partitions align with different rendered clinical
# fields." Panel a: field x partition AMI heatmap, grouped by cohort. Panel
# b: cross-encoder AMI before/after conditioning on organism, non-degenerate
# headline pairs only.
f5_path <- file.path(DATA, "field_dominance.csv")
f5b_path <- file.path(DATA, "conditional_agreement.csv")

FIELD_LABELS <- c(
  admission_cluster = "Pre-computed admission cluster",
  antibiotic_set = "Antibiotic regimen",
  antipseudoab_empiric = "Empiric antipseudomonal therapy",
  BSI_episode_short_clean = "Index organism",
  carbaab_empiric = "Empiric carbapenem therapy",
  charlson_category_withcancermerged = "Charlson category (cancer merged)",
  charlson_category = "Charlson category",
  empiric_cluster = "Pre-computed empiric-treatment cluster",
  first_antibiotic = "First antibiotic named",
  n_antibiotic_lines = "Number of antibiotics",
  n_bsi_episodes = "Number of BSI episodes",
  organism_category = "Organism category",
  polymicrobial = "Polymicrobial infection",
  source_category = "Source of infection",
  treatment_department_eng = "Treatment department"
)
gloss <- function(v) ifelse(v %in% names(FIELD_LABELS), unname(FIELD_LABELS[v]), v)

if (file.exists(f5_path)) {
  f5 <- read.csv(f5_path)
  f5$is_dominant <- f5$is_dominant %in% c(TRUE, "True", "TRUE", "true")

  ENC_SHORT <- c("openai-3large" = "text-3-large", "openai-3small" = "text-3-small",
                 "bge-m3" = "BGE-M3", "qwen3-0.6b" = "Qwen3-0.6B")
  f5$encoder_short <- unname(ENC_SHORT[f5$encoder])
  f5$cohort_disp <- factor(ifelse(f5$cohort == "Korea", "South Korea", "MIMIC-IV"),
                           levels = c("South Korea", "MIMIC-IV"))
  # v8: PRIORITY fix -- x-axis tick labels under panel a collided with each other and with
  # the axis title (worst offenders: "text-3-small", "BGE-M3\n(degenerate)", "Qwen3-0.6B").
  # Replaced with compact forms, panel-a-local only (Fig. 1/2/4's OWN encoder-name spelling
  # elsewhere, e.g. row labels/legends, is untouched): South Korea "text-3-large"/
  # "text-3-small" -> "3-large"/"3-small" (drop the shared "text-" prefix, unambiguous within
  # this panel since every South Korea column is an OpenAI/BGE/Qwen encoder), "BGE-M3
  # (degenerate)" -> "BGE-M3†" (dagger explained by an in-canvas caption, see below,
  # cross-referencing Fig. 1c's own degenerate/non-degenerate finding), "Qwen3-0.6B"
  # unchanged. MIMIC-IV "BGE-M3"/"Qwen3-0.6B" unchanged (verified against the actual encoders
  # present per cohort in field_dominance.csv: South Korea has all 4 encoders, MIMIC-IV has
  # only bge-m3/qwen3-0.6b -- nothing reordered or dropped). No cell colour/value or
  # dominant-field marking touched.
  ENC_COMPACT_KR <- c("openai-3large" = "3-large", "openai-3small" = "3-small",
                      "bge-m3" = "BGE-M3†", "qwen3-0.6b" = "Qwen3-0.6B")
  f5$x_disp <- ifelse(f5$cohort == "Korea", unname(ENC_COMPACT_KR[f5$encoder]), f5$encoder_short)
  kr_levels <- paste("South Korea", unname(ENC_COMPACT_KR[c("openai-3large", "openai-3small",
                                                             "bge-m3", "qwen3-0.6b")]))
  mi_levels <- paste("MIMIC-IV", c("BGE-M3", "Qwen3-0.6B"))
  f5$col_key <- factor(paste(f5$cohort_disp, f5$x_disp), levels = c(kr_levels, mi_levels))
  stopifnot(!any(is.na(f5$col_key)))

  keep_fields <- unique(f5$field[f5$ami >= 0.08])
  combos <- unique(f5[f5$field %in% keep_fields, c("cohort_disp", "col_key", "x_disp")])
  grid <- merge(combos, data.frame(field = keep_fields), by = NULL)
  full <- merge(grid, f5[f5$field %in% keep_fields,
                         c("cohort_disp", "col_key", "field", "ami", "is_dominant", "margin")],
               by = c("cohort_disp", "col_key", "field"), all.x = TRUE)
  full$rendered <- !is.na(full$ami)
  full$txt_colour <- ifelse(full$rendered & full$ami > 0.55, "white", HS_PAL$ink)
  field_levels <- rev(sort(unique(gloss(keep_fields))))
  full$field_disp <- factor(gloss(full$field), levels = field_levels)

  not_rendered_df <- subset(full, !rendered)
  p2a <- ggplot(full, aes(x = col_key, y = field_disp)) +
    geom_tile(data = subset(full, rendered), aes(fill = ami), colour = "white", linewidth = 1.2) +
    # v7 (b): "not rendered" tiles stay exactly NOT_RENDERED grey -- this grey is
    # CATEGORICAL (a third data state, "cell does not exist for this cohort"), not a
    # decorative default, so it is explicitly left unchanged per the task instruction.
    geom_tile(data = not_rendered_df, fill = NOT_RENDERED, colour = "white", linewidth = 1.2) +
    geom_point(data = not_rendered_df[1, , drop = FALSE],
               aes(shape = "Not rendered in this cohort"), x = -Inf, y = -Inf,
               size = 0, stroke = 0, na.rm = TRUE) +
    geom_text(data = subset(full, rendered & ami >= 0.08), aes(label = sprintf("%.2f", ami), colour = txt_colour),
              size = MM(10), family = "mono") +
    scale_colour_identity() +
    # v7 (c): dominant-field outline colour/width set above by the WCAG + CVD-simulation
    # check (dominant_outline_col / dominant_outline_lwd) rather than hardcoded HS_PAL$red.
    geom_tile(data = subset(full, rendered & is_dominant & margin > 0.10),
              colour = dominant_outline_col, fill = NA, linewidth = dominant_outline_lwd) +
    scale_fill_gradient(low = HS_PAL$mid, high = HS_PAL$blue, limits = c(0, 1),
                        breaks = seq(0, 1, 0.25), name = "AMI") +
    scale_shape_manual(values = c("Not rendered in this cohort" = 22), name = NULL,
                       guide = guide_legend(position = "bottom",
                                            override.aes = list(size = 4, stroke = 0.4,
                                                                 fill = NOT_RENDERED,
                                                                 colour = "grey40"))) +
    scale_x_discrete(labels = function(v) sub("^(South Korea|MIMIC-IV) ", "", v)) +
    facet_grid(~cohort_disp, scales = "free_x", space = "free_x") +
    labs(title = "a) Field alignment differs across encoder-derived partitions",
         x = "Encoder-derived partition", y = "Rendered clinical field",
         # v8: in-canvas caption-style footnote explaining the "†" mark on the
         # compact "BGE-M3†" tick label above -- same mechanism Fig. 4a already used in
         # v7 for its own dagger footnote (labs(caption=...) + plot.caption recoloured
         # HS_PAL$ink, since this defines what the symbol MEANS, i.e. semantic, not
         # decorative). Placed in-canvas per the task's preference; nothing added to any
         # .md legend file. Wrapped onto two lines (same words) -- the one-line form measured
         # textLength 442.36pt starting at x=356.23pt on a 792pt-wide canvas, i.e. running
         # ~7pt past the canvas edge (confirmed in the SVG, same class of overflow Fig. 4a's
         # own dagger caption had in v7).
         caption = "† Degenerate partition under the operational criterion\nin Fig. 1c.") +
    theme_house(base_size = BASE_SIZE) +
    # v7: x-axis category labels ("text-3-large", "BGE-M3\n(degenerate)", ...) overlapped
    # at the larger axis.text target -- confirmed by comparing against the v6 PNG, where
    # the SAME labels at the smaller v6 font did not overlap (a genuine consequence of the
    # larger tick-label target, not a pre-existing defect). Angling them (standard fix for
    # a crowded categorical axis) resolves it without shrinking the font or touching any
    # label's text/data.
    # v8: horizontal (angle = 0) tried FIRST per the task's stated preference, with the
    # shortened label forms above -- rendered and read as a PNG (job 605715): the tick labels
    # still visibly ran into each other and into the neighbouring facet ("3-large3-small
    # BGE-M3†Qwen3-0.6BGE-M3 Qwen3-0.6B"), because each column's own width here is set by
    # the number of rows (facet_grid(space="free_x")) and by the wide row-label/legend area
    # this panel also carries, not by tick-label length -- so even single-word compact labels
    # do not fit horizontally at this column width. Reverted to angle = 30 (NOT 45, per the
    # task's "only rotate if every other fix fails" -- this is the SAME angle v7 already
    # verified collision-free for the OLDER, longer label set at this same column width, so it
    # is certain to stay clean for these shorter ones; a smaller angle was not re-tried given
    # the horizontal attempt's margin of overlap was large, not marginal).
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          legend.position = "right",
          strip.text = element_text(face = "bold"),
          axis.text.x = element_text(lineheight = 0.9, angle = 30, hjust = 1, vjust = 1),
          plot.caption = element_text(colour = INK, hjust = 0))

  if (file.exists(f5b_path)) {
    f5b <- read.csv(f5b_path)
    f5b$involves_bge_m3 <- f5b$involves_bge_m3 %in% c(TRUE, "True", "TRUE", "true")
    keep_pairs <- f5b[!(f5b$cohort == "Korea" & f5b$involves_bge_m3), ]
    pretty <- function(p) {
      p <- gsub("bge-m3", "BGE-M3", p, fixed = TRUE)
      p <- gsub("qwen3-0.6b", "Qwen3-0.6B", p, fixed = TRUE)
      p <- gsub("openai-3small", "text-3-small", p, fixed = TRUE)
      p <- gsub("openai-3large", "text-3-large", p, fixed = TRUE)
      p
    }
    keep_pairs$cohort_disp <- ifelse(keep_pairs$cohort == "Korea", "South Korea", "MIMIC-IV")
    keep_pairs$pair_label <- paste0(keep_pairs$cohort_disp, ": ", pretty(keep_pairs$pair))
    ord <- order(keep_pairs$cohort == "Korea", -keep_pairs$ami_within_organism)
    pair_levels <- c(rev(keep_pairs$pair_label[ord]), "")
    keep_pairs$pair_label <- factor(keep_pairs$pair_label, levels = pair_levels)
    keep_pairs$annot <- ifelse(
      is.na(keep_pairs$ami_within_lo),
      sprintf("%s → %s", fmt_num(keep_pairs$ami_unconditional), fmt_num(keep_pairs$ami_within_organism)),
      sprintf("%s → %s [%s, %s]", fmt_num(keep_pairs$ami_unconditional), fmt_num(keep_pairs$ami_within_organism),
              fmt_num(keep_pairs$ami_within_lo), fmt_num(keep_pairs$ami_within_hi))
    )

    long <- rbind(
      data.frame(pair_label = keep_pairs$pair_label, x = keep_pairs$ami_unconditional, type = "Unconditional"),
      data.frame(pair_label = keep_pairs$pair_label, x = keep_pairs$ami_within_organism, type = "Within organism strata")
    )
    long$type <- factor(long$type, levels = c("Unconditional", "Within organism strata"))
    long$pair_label <- factor(as.character(long$pair_label), levels = pair_levels)
    stopifnot(!any(is.na(long$pair_label)))

    p2b <- ggplot() +
      geom_segment(data = keep_pairs, aes(x = ami_unconditional, xend = ami_within_organism,
                                          y = pair_label, yend = pair_label),
                   colour = HS_PAL$gray, linewidth = 0.8) +
      geom_errorbarh(data = keep_pairs, aes(xmin = ami_within_lo, xmax = ami_within_hi, y = pair_label),
                      height = 0, linewidth = 0.6, colour = HS_PAL$blue, alpha = 0.55, na.rm = TRUE) +
      geom_point(data = long, aes(x = x, y = pair_label, shape = type), size = 2.9,
                colour = HS_PAL$blue, fill = "white") +
      # v7: MM(9), not MM(10) -- same infeasible-at-10pt case as Fig. 1's panels a/b (a
      # single wide bracketed column, "0.803 -> 0.571 [0.512, 0.612]" = 29 chars, next to
      # long row labels); solving the same layout equation at t=10 required xlim ~= 6.5 (an
      # empirically confirmed clipped/overlapping render, job 604971) -- 9pt (the task's own
      # floor) is the compliant compromise, same as Fig. 1.
      geom_text(data = keep_pairs, aes(x = 1.05, y = pair_label, label = annot),
                hjust = 0, size = MM(9), family = "mono", colour = INK) +
      # v7 (a): header colour grey -> near-black -- states what the numbers ARE
      # (a semantic label), not a decorative caption.
      annotate("text", x = 1.05, y = "",
               label = "AMI: unconditional →\nwithin-organism",
               hjust = 0, vjust = 0.5, size = MM(9), family = "mono",
               colour = INK, lineheight = 0.95) +
      scale_y_discrete(drop = FALSE) +
      scale_shape_manual(values = c("Unconditional" = 21, "Within organism strata" = 19), name = NULL) +
      # v7: 1.9 -> 2.9, fewer ticks (was seq(0,1,0.25)) -- both solved from the MM(9) fit
      # (see note above); at this xlim range 0.25-spaced ticks would again crowd.
      scale_x_continuous(limits = c(0, 2.9), breaks = seq(0, 1, 0.5), expand = c(0, 0)) +
      coord_cartesian(clip = "off") +  # v7: the widest row sits close to the panel's own
                                        # xlim edge (a PNG/SVG font-metric discrepancy
                                        # clipped it under clip="on") -- matches the
                                        # clip="off" already used for every other panel here
      labs(title = "b) Cross-encoder concordance before and after conditioning on organism",
           x = "Cross-encoder adjusted mutual information (AMI)", y = NULL) +
      theme_house(base_size = BASE_SIZE) +
      theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
            legend.position = "bottom")

    p2 <- p2a / p2b + patchwork::plot_layout(heights = c(2.2, 1.7))
    hs_save(p2, file.path(FIGS, "fig2_field_dominance.png"), width = WIDTH_IN, height = 15.7, dpi = 300)
    hs_save_vector(p2, file.path(FIGS, "fig2_field_dominance.svg"), width = WIDTH_IN, height = 15.7)
    cat("[v8 fig2] written (2 panels)\n")
  } else {
    hs_save(p2a, file.path(FIGS, "fig2_field_dominance.png"), width = WIDTH_IN, height = 9.6, dpi = 300)
    hs_save_vector(p2a, file.path(FIGS, "fig2_field_dominance.svg"), width = WIDTH_IN, height = 9.6)
    cat("[v8 fig2] written (panel a only)\n")
  }
} else {
  cat("[v8 fig2] skipped -- ", f5_path, " not on disk\n", sep = "")
}

# =================================================================== Figure 4
# "Fig. 4 | Structured nearest-neighbour retrieval showed higher 90-day
# mortality discrimination than embedding retrieval in any of three encoders,
# and than the cluster labels." Panel a: seven-rung retrieval ladder. Panel
# b: paired patient-level bootstrap DeltaAUROC contrasts (Gower minus each
# comparator).
GRP_COL_4 <- c("Structured retrieval" = ORANGE, "Embedding retrieval" = HS_PAL$blue, "Comparator" = HS_PAL$gray)

f4a_path <- file.path(DATA, "retrieval_ladder_v4.csv")
f4b_path <- file.path(DATA, "retrieval_bootstrap_contrasts.csv")
if (file.exists(f4a_path) && file.exists(f4b_path)) {
  f4a <- read.csv(f4a_path)
  LBL_F4A <- c(
    "Structured retrieval (Gower, k = 25)"  = "Structured retrieval (Gower, k = 25)",
    "Embedding retrieval (3-small, k = 25)" = "Embedding retrieval (text-3-small, k = 25)",
    "Embedding retrieval (BGE-M3, k = 25)"  = "Embedding retrieval (BGE-M3, k = 25)",
    "Embedding retrieval (3-large, k = 25)" = "Embedding retrieval (text-3-large, k = 25)",
    "SOFA"                                  = "SOFA score",
    "Charlson"                              = "Charlson index",
    "Originally derived cluster labels"     = "Originally derived cluster labels")
  stopifnot(all(f4a$label %in% names(LBL_F4A)))
  f4a$label <- unname(LBL_F4A[f4a$label])
  f4a$label2_txt <- ifelse(f4a$is_original_representation %in% c(TRUE, "True", "TRUE", "true"),
                           paste0(f4a$label, "†"), f4a$label)
  rung_order <- f4a$label2_txt[order(f4a$rung)]
  f4a$label2 <- factor(f4a$label2_txt, levels = rev(unique(rung_order)))
  f4a$ci_low  <- f4a$estimate - 1.96 * f4a$sd
  f4a$ci_high <- f4a$estimate + 1.96 * f4a$sd
  unrestricted <- f4a[f4a$variant == "unrestricted", ]
  past         <- f4a[f4a$variant == "past_only", ]
  stopifnot(nrow(past) > 0)

  POOL_SHAPES <- c("Unrestricted pool" = 16, "Past-admissions-only pool" = 21)

  p4a <- ggplot() +
    geom_vline(xintercept = seq(0.6, 0.8, 0.05), colour = HS_PAL$edge, linetype = "dotted", linewidth = 0.3) +
    geom_errorbarh(data = unrestricted, aes(xmin = ci_low, xmax = ci_high, y = label2, colour = group),
                   height = 0, linewidth = 0.7, alpha = 0.55) +
    geom_segment(data = past, aes(x = estimate, xend = unrestricted$estimate[match(label2, unrestricted$label2)],
                                  y = label2, yend = label2),
                 colour = HS_PAL$gray, linewidth = 0.5, linetype = "dashed") +
    geom_point(data = unrestricted, aes(x = estimate, y = label2, colour = group), shape = 16, size = 3.2) +
    geom_point(data = past, aes(x = estimate, y = label2), shape = 21, size = 2.4, stroke = 0.8,
               fill = "white", colour = HS_PAL$gray) +
    geom_text(data = unrestricted, aes(x = estimate, y = label2, label = sprintf("%.3f", estimate)),
              vjust = -1.1, size = MM(10), family = "mono", colour = INK) +
    # v7 (a): past-admissions-only AUROC labels grey -> near-black -- these are numeric
    # data-value annotations, not decoration (the task's explicit example).
    geom_text(data = past, aes(x = estimate, y = label2, label = sprintf("%.3f", estimate)),
              vjust = 2.0, size = MM(10), family = "mono", colour = INK) +
    geom_point(data = unrestricted[1, , drop = FALSE], aes(shape = "Unrestricted pool"),
               x = -Inf, y = -Inf, size = 0, stroke = 0, na.rm = TRUE) +
    geom_point(data = past[1, , drop = FALSE], aes(shape = "Past-admissions-only pool"),
               x = -Inf, y = -Inf, size = 0, stroke = 0, na.rm = TRUE) +
    scale_colour_manual(values = GRP_COL_4, name = NULL,
                        guide = guide_legend(order = 1, override.aes = list(size = 3.2, shape = 16))) +
    scale_shape_manual(values = POOL_SHAPES, name = NULL,
                       breaks = c("Unrestricted pool", "Past-admissions-only pool"),
                       guide = guide_legend(order = 2,
                                            override.aes = list(size = 3.0, stroke = 0.8,
                                                                colour = HS_PAL$ink, fill = "white"))) +
    scale_x_continuous(limits = c(0.58, 0.85), breaks = seq(0.6, 0.85, 0.05)) +
    labs(title = "a) 90-day mortality discrimination across representations",
         x = "90-day mortality AUROC", y = NULL,
         # v7: manual line break -- at the larger caption font this one-line sentence (89
         # chars) ran past the canvas edge (measured: textLength 609.66px starting at
         # x=340.56pt on a 792pt-wide canvas, i.e. genuinely clipped, confirmed in the SVG,
         # not just the PNG preview). Same words, wrapped to two lines; no text changed.
         caption = "† the embedding representation from which the\noriginally derived partition was obtained") +
    theme_house(base_size = BASE_SIZE) +
    # v7 (c): the dagger explanatory caption line grey -> near-black -- theme_house()'s
    # default plot.caption colour is HS_PAL$gray, which is right for a decorative
    # caption but wrong here (this caption defines what "†" MEANS, i.e. semantic).
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          legend.position = "bottom", legend.box = "vertical",
          legend.spacing.y = unit(1, "pt"), legend.margin = margin(t = 0, b = 0),
          plot.caption = element_text(colour = INK, hjust = 0))

  f4b <- read.csv(f4b_path)
  LBL_F4B <- c(
    "Embedding retrieval (3-small)"     = "Embedding retrieval (text-3-small)",
    "Embedding retrieval (BGE-M3)"      = "Embedding retrieval (BGE-M3)",
    "Embedding retrieval (3-large)"     = "Embedding retrieval (text-3-large)",
    "SOFA"                              = "SOFA score",
    "Charlson"                          = "Charlson index",
    "Originally derived cluster labels"  = "Originally derived cluster labels")
  stopifnot(all(f4b$comparator_label %in% names(LBL_F4B)))
  f4b$comparator_label <- unname(LBL_F4B[f4b$comparator_label])
  f4b$row <- factor(f4b$comparator_label,
                    levels = c(rev(f4b$comparator_label[order(f4b$order)]), ""))
  stopifnot(!any(is.na(f4b$row)))

  f4b$label_txt <- sprintf("%.3f [%.3f, %.3f]", f4b$delta, f4b$ci_low, f4b$ci_high)
  X_F4B <- 0.235

  p4b <- ggplot(f4b, aes(y = row, colour = comparator_group)) +
    geom_vline(xintercept = 0, colour = HS_PAL$ink, linewidth = 0.5) +
    geom_errorbarh(aes(xmin = ci_low, xmax = ci_high), height = 0,
                   linewidth = 0.9, alpha = 0.60) +
    geom_point(aes(x = delta), size = 3.0, shape = 16) +
    geom_text(aes(x = X_F4B, label = label_txt), hjust = 0, vjust = 0.5,
              size = MM(10), family = "mono", colour = INK) +
    # v7 (b): "ΔAUROC [95% bootstrap CI]" header grey -> near-black.
    annotate("text", x = X_F4B, y = "", label = "ΔAUROC\n[95% bootstrap CI]",
             hjust = 0, vjust = 0.5, size = MM(10), family = "mono",
             colour = INK, lineheight = 0.95) +
    scale_y_discrete(drop = FALSE) +
    # v7 (d): comparator DATA POINTS keep HS_PAL$gray via GRP_COL_4 -- that grey IS the
    # comparator category (data encoding), not decoration; left unchanged as instructed.
    scale_colour_manual(values = GRP_COL_4, name = NULL, guide = "none") +
    # v7: 0.40 -> 0.48 (measured widest row "0.188 [0.180, 0.195]", 186.88px, was already
    # being clipped by the old limit)
    scale_x_continuous(limits = c(-0.01, 0.52), breaks = seq(0, 0.20, 0.05)) +
    coord_cartesian(clip = "off") +  # v7: safety margin is thin here too (see Fig. 2b note)
    labs(title = "b) Gower structured retrieval minus comparator, ΔAUROC",
         x = "ΔAUROC (Gower − comparator), patient-level bootstrap",
         y = NULL) +
    theme_house(base_size = BASE_SIZE) +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank())
  # v7 (e): panel b's bootstrap CIs (ci_low/ci_high, both from f4b as read off disk) are
  # untouched -- geom_errorbarh() reads the same columns unchanged from the same CSV.

  p4 <- p4a / p4b + patchwork::plot_layout(heights = c(1.15, 1.0))
  hs_save(p4, file.path(FIGS, "fig5_retrieval.png"), width = WIDTH_IN, height = 14.5, dpi = 300)
  hs_save_vector(p4, file.path(FIGS, "fig5_retrieval.svg"), width = WIDTH_IN, height = 14.5)
  cat("[v8 fig4] written (2 panels, 7 rungs / 6 contrasts)\n")
} else {
  cat("[v8 fig4] skipped -- ", f4a_path, " or ", f4b_path, " not on disk\n", sep = "")
}

# ============================================================ Supplementary 1
# "Exploratory template perturbations relative to the unmodified arm's own
# seed range." EXPLORATORY -- rankings are not effect sizes.
sf1_path <- file.path(DATA, "variants.csv")
if (file.exists(sf1_path)) {
  sf1 <- read.csv(sf1_path)
  band <- sf1[sf1$variant == "full", ]
  band_lo <- band$full_band_lo[1]; band_hi <- band$full_band_hi[1]

  SAFE_LABELS <- c(
    full = "Unmodified (reference)", no_headers = "Headers removed",
    no_summary_tokens = "Summary-token block removed", no_year = "Admission year removed",
    no_department = "Treatment department removed", no_therapy = "Therapy block removed",
    no_organism = "Organism removed", fixed_permutation = "Fixed line reordering",
    shuffle_order = "Random line-order shuffle", shuffle_order_seed2 = "Random line-order shuffle (seed 2)",
    no_dates = "Dates removed", json_format = "JSON-format rendering",
    null_dropped = "Null-valued lines dropped", no_definitions = "Definitions removed",
    host_source_severity_only = "Host/source/severity only",
    no_resistance                    = "Resistance information removed",
    no_resistance_strict             = "Resistance information removed (strict)",
    index_window                     = "Index-episode window only",
    no_organism_no_resistance        = "Organism and resistance removed",
    no_organism_no_resistance_strict = "Organism and resistance removed (strict)"
  )
  stopifnot(all(sf1$variant %in% names(SAFE_LABELS)))
  sf1$display <- ifelse(sf1$variant %in% names(SAFE_LABELS), unname(SAFE_LABELS[sf1$variant]), sf1$variant)

  sf1$label <- factor(sf1$display,
                      levels = c(sf1$display[order(sf1$ari_noise_excl_median)], ""))

  sf1$role_v4 <- ifelse(sf1$variant == "full", "Unmodified reference",
                 ifelse(sf1$variant %in% c("fixed_permutation", "shuffle_order", "shuffle_order_seed2"),
                        "Order control", "Other exploratory template variant"))
  role_pal_sf1 <- c("Unmodified reference"               = HS_PAL$ink,
                    "Order control"                      = HS_PAL$blue,
                    "Other exploratory template variant" = HS_PAL$gray)
  # v7: colour-blind safety -- this three-role grammar was colour-only (fixed shape = 21
  # for all roles). "Order control" in particular has no other textual cue that it is a
  # control (its row labels are plain manipulation names, e.g. "Fixed line reordering").
  # Kept all three as OPEN/white-centre shapes (square/triangle/circle) to preserve the
  # original point's visual family while adding the distinction.
  role_shp_sf1 <- c("Unmodified reference"               = 22,
                    "Order control"                      = 24,
                    "Other exploratory template variant" = 21)
  sf1$role_v4 <- factor(sf1$role_v4, levels = names(role_pal_sf1))
  sf1$annot <- sprintf("%s [%s–%s]", fmt_num(sf1$ari_noise_excl_median),
                       fmt_num(sf1$ari_noise_excl_min), fmt_num(sf1$ari_noise_excl_max))

  y_band_label <- match("Random line-order shuffle", levels(sf1$label))
  stopifnot(length(y_band_label) == 1L, !is.na(y_band_label))

  psf1 <- ggplot(sf1, aes(y = label)) +
    # v7: reference-band FILL stays light/unchanged (task: "keep the reference-band
    # shading light") -- only the text label on it is recoloured below.
    annotate("rect", xmin = band_lo, xmax = band_hi, ymin = -Inf, ymax = Inf, fill = HS_PAL$mid, alpha = 0.75) +
    geom_vline(xintercept = c(band_lo, band_hi), colour = HS_PAL$gray, linetype = "dotted", linewidth = 0.35) +
    # v7: "Unmodified seed range" band LABEL grey -> near-black (this is the text label
    # naming the shaded region, not the shaded region's fill itself -- the fill above is
    # untouched).
    annotate("text", x = (band_lo + band_hi) / 2, y = y_band_label, label = "Unmodified seed range",
             hjust = 0.5, vjust = 0.5, size = MM(10), family = hs_font(),
             colour = INK) +
    geom_segment(aes(x = ari_noise_excl_min, xend = ari_noise_excl_max, yend = label, colour = role_v4),
                 linewidth = 2.4, lineend = "round") +
    geom_point(aes(x = ari_noise_excl_median, colour = role_v4, shape = role_v4), size = 2.2, fill = "white", stroke = 1.0) +
    geom_text(aes(x = 1.02, label = annot), hjust = 0, size = MM(10), family = "mono", colour = INK) +
    # v7: "ARI median [observed range]" header grey -> near-black.
    annotate("text", x = 1.02, y = "", label = "ARI median\n[observed range]",
             hjust = 0, vjust = 0.5, size = MM(10), family = "mono",
             colour = INK, lineheight = 0.95) +
    scale_y_discrete(drop = FALSE) +
    # v7: ncol = 1 (was single-row default; nrow = 2 was still measured 2.7pt from the
    # canvas edge -- one item per row removes ALL horizontal crowding regardless of text
    # length) -- "Other exploratory template variant" (35 chars) at the larger legend-text
    # target ran past the canvas edge in a single row.
    scale_colour_manual(values = role_pal_sf1, name = NULL,
                        guide = guide_legend(ncol = 1)) +
    scale_shape_manual(values = role_shp_sf1, name = NULL,
                       guide = guide_legend(ncol = 1)) +
    # v7: 1.42 -> 1.62 (measured widest row "0.601 [0.479–0.983]", 177.53px, was already
    # being clipped by the old limit)
    # v7: 1.42 -> 1.75 -- SF1's single column (no second AMI-style column competing) fits the
    # full MM(10) target with a modest xlim increase (solved layout equation: T needs >=1.74;
    # 1.75 kept here with a small safety margin -- confirmed feasible, unlike Fig. 1/2b/SF2).
    scale_x_continuous(limits = c(0, 1.85), breaks = seq(0, 1, 0.25), expand = c(0, 0)) +
    coord_cartesian(clip = "off") +
    labs(title = "Exploratory template perturbations relative to the unmodified
arm's seed range",  # v7: wrapped, was clipped at this canvas/font size
         x = "Adjusted Rand index (ARI) vs originally derived partition",
         y = NULL) +
    theme_house(base_size = BASE_SIZE) +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(), legend.position = "bottom")

  hs_save(psf1, file.path(FIGS, "figS1_variant_ranks.png"), width = WIDTH_IN, height = 13.6, dpi = 300)
  hs_save_vector(psf1, file.path(FIGS, "figS1_variant_ranks.svg"), width = WIDTH_IN, height = 13.6)
  cat("[v8 sf1] written\n")
} else {
  cat("[v8 sf1] skipped -- ", sf1_path, " not on disk\n", sep = "")
}

# ============================================================ Supplementary 2
# "Token recoverability and artifact screening across the sixteen originally
# derived clusters." NOTE: this plot draws LOOKUP F1, not encoder F1.
f2_path <- file.path(DATA, "token_lookup.csv")
if (file.exists(f2_path)) {
  f2 <- read.csv(f2_path)
  SCREEN_SHORT <- c("organism"                    = "Organism-defined",
                    "no artifact label"           = "No pre-specified screen label",
                    "year/era-defined"            = "Year/era-defined",
                    "timing (long stay)"          = "Timing-defined (long stay)",
                    "timing-defined (short stay)" = "Timing-defined (short stay)")
  f2$screen_raw <- ifelse(f2$screen_label == "(no label at all)", "no artifact label", f2$screen_label)
  stopifnot(all(f2$screen_raw %in% names(SCREEN_SHORT)))
  f2$screen <- ifelse(f2$screen_raw %in% names(SCREEN_SHORT), unname(SCREEN_SHORT[f2$screen_raw]), f2$screen_raw)
  f2$row_label <- paste0(f2$label, " — ", f2$screen)
  f2$row_label <- factor(f2$row_label, levels = c(rev(f2$row_label[order(f2$f1_prespec)]), ""))
  f2$annot <- ifelse(f2$gain >= 0.30,
                     sprintf("%s → %s", fmt_num2(f2$f1_prespec), fmt_num2(f2$f1_extended)),
                     fmt_num2(f2$f1_prespec))

  p2sf <- ggplot(f2, aes(y = row_label)) +
    geom_segment(aes(x = 0, xend = 1, yend = row_label), colour = HS_PAL$track, linewidth = 2.4, lineend = "round") +
    geom_segment(aes(x = f1_prespec, xend = f1_extended, yend = row_label), colour = HS_PAL$gray, linewidth = 0.9) +
    geom_point(aes(x = f1_extended, shape = "Exploratory extended lookup"),
               size = 2.4, stroke = 0.7, fill = "white", colour = HS_PAL$gray) +
    geom_point(aes(x = f1_prespec, colour = screen, shape = "Pre-specified lookup"), size = 2.9) +
    # v7: MM(9), not MM(10) -- same infeasible-at-10pt case as Fig. 1/2b: the header's own
    # second line, "(→ extended, where it differs)" (280.31px), is wider than any data value
    # and next to long row labels; solving the layout equation at t=10 required xlim ~= 3.5
    # (an empirically confirmed collision, job 604971). 9pt is the task's own floor.
    geom_text(aes(x = 1.05, label = annot), hjust = 0, size = MM(9), family = "mono", colour = INK) +
    # v7: "Pre-specified lookup F1 (→ extended, where it differs)" header grey ->
    # near-black. Wording unchanged ("Pre-specified", the project's established term --
    # not touched here since we only touch colour/size).
    annotate("text", x = 1.05, y = "",
             label = "Pre-specified lookup F1\n(→ extended, where it differs)",
             hjust = 0, vjust = 0.5, size = MM(9), family = "mono",
             colour = INK, lineheight = 0.95) +
    scale_y_discrete(drop = FALSE) +
    scale_colour_manual(values = hs_colors(length(unique(f2$screen))), name = NULL) +
    scale_shape_manual(values = c("Pre-specified lookup" = 16, "Exploratory extended lookup" = 21), name = NULL) +
    # v7: 1.75 -> 2.9, fewer ticks (was seq(0,1,0.25)) -- both solved from the MM(9) fit
    # (see note above).
    scale_x_continuous(limits = c(0, 2.9), breaks = seq(0, 1, 0.5), expand = c(0, 0)) +
    coord_cartesian(clip = "off") +  # v7: safety margin is thin here too (see Fig. 2b note)
    # v7: ncol = 1 (was nrow = 2 default, then nrow = 3 -- still measured "Timing-defined
    # (short stay)" clipped past the canvas edge) -- one item per row removes ALL horizontal
    # crowding regardless of text length, for the 5-category colour legend.
    guides(colour = guide_legend(ncol = 1, order = 1), shape = guide_legend(order = 2)) +
    labs(title = "Token recoverability and artifact screening across the sixteen
originally derived clusters",  # v7: wrapped, was clipped
         x = "Per-cluster one-vs-rest lookup F1", y = NULL) +
    theme_house(base_size = BASE_SIZE) +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
          legend.position = "bottom", legend.box = "vertical", legend.margin = margin(t = 0, b = 0))

  hs_save(p2sf, file.path(FIGS, "fig4_token_lookup.png"), width = WIDTH_IN, height = 11.6, dpi = 300)
  hs_save_vector(p2sf, file.path(FIGS, "fig4_token_lookup.svg"), width = WIDTH_IN, height = 11.6)
  cat("[v8 sf2] written\n")
} else {
  cat("[v8 sf2] skipped -- ", f2_path, " not on disk\n", sep = "")
}

# ============================================================ Supplementary 3
# "Size-stability correlations were positive, but none remained significant
# after Holm correction." INDETERMINATE, not "no association" -- n=16, low
# power.
f6_path <- file.path(DATA, "size_stability.csv")
if (file.exists(f6_path)) {
  f6 <- read.csv(f6_path)
  f6$label[f6$column == "frac_retained"] <- "S2 fraction not dissolved"
  f6$label[f6$column == "J_seed_p05"]    <- "S1 seed Jaccard (5th percentile)"
  f6$label2 <- ifelse(f6$n == max(f6$n), f6$label, sprintf("%s (n = %d)", f6$label, f6$n))
  f6$label2 <- factor(f6$label2, levels = c(rev(f6$label2[order(f6$rho)]), ""))
  f6$annot_rho <- sprintf("%s [%s, %s]", fmt_num(f6$rho), fmt_num(f6$lo), fmt_num(f6$hi))
  f6$annot_p   <- ifelse(is.na(f6$p), "—",
                  ifelse(f6$p < 0.001, "<0.001", sprintf("%.3f", f6$p)))
  # v7: same infeasible-at-10pt case as Fig. 1/2b/SF2 (two side-by-side bracketed columns,
  # like Fig. 1's ARI/AMI); solving the layout equation at t=10 required xlim span ~= 8.9
  # (an empirically confirmed collision, job 604971) -- 9pt is the task's own floor, and
  # X_RHO/X_P/xlim below are solved at t=9 for this panel's own measured geometry.
  X_RHO <- 1.05
  X_P   <- 3.95   # v7: 2.35 -> 3.95, solved at MM(9) from this panel's own measured
                   # px-per-unit (103.56px/unit) and textLength values

  p6 <- ggplot(f6, aes(y = label2)) +
    geom_vline(xintercept = 0, colour = HS_PAL$gray, linewidth = 0.4) +
    geom_segment(aes(x = lo, xend = hi, yend = label2), colour = HS_PAL$blue, linewidth = 2.0, lineend = "round") +
    geom_point(aes(x = rho), shape = 16, size = 2.7, colour = HS_PAL$ink) +
    geom_text(aes(x = X_RHO, label = annot_rho), hjust = 0, size = MM(9),
              family = "mono", colour = INK) +
    geom_text(aes(x = X_P, label = annot_p), hjust = 0, size = MM(9),
              family = "mono", colour = INK) +
    # v7: "Spearman ρ [95% bootstrap CI]" header grey -> near-black.
    annotate("text", x = X_RHO, y = "", label = "Spearman ρ\n[95% bootstrap CI]",
             hjust = 0, vjust = 0.5, size = MM(9), family = "mono",
             colour = INK, lineheight = 0.95) +
    # v7: "Holm-adjusted P" header grey -> near-black.
    annotate("text", x = X_P, y = "", label = "Holm-adjusted\nP",
             hjust = 0, vjust = 0.5, size = MM(9), family = "mono",
             colour = INK, lineheight = 0.95) +
    scale_y_discrete(drop = FALSE) +
    coord_cartesian(xlim = c(-1, 5.8), clip = "off") +  # v7: 3.3 -> 5.8, solved at MM(9)
    # v7: fewer ticks (was seq(-1,1,0.5)) -- at this xlim range 0.5-spaced ticks would crowd.
    scale_x_continuous(breaks = c(-1, 0, 1)) +
    labs(title = "Size–stability correlations were positive, but none remained
significant after Holm correction",  # v7: wrapped, was clipped
         x = "Spearman ρ between cluster size and stability metric", y = NULL) +
    theme_house(base_size = BASE_SIZE) +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank())

  hs_save(p6, file.path(FIGS, "fig6_size_stability.png"), width = WIDTH_IN, height = 7.4, dpi = 300)
  hs_save_vector(p6, file.path(FIGS, "fig6_size_stability.svg"), width = WIDTH_IN, height = 7.4)
  cat("[v8 sf3] written\n")
} else {
  cat("[v8 sf3] skipped -- ", f6_path, " not on disk\n", sep = "")
}

cat("[v8 figures] wrote to ", FIGS, "\n", sep = "")
