# -*- coding: utf-8 -*-
# ─────────────────────────────────────────────────────────────────────────────
#  house_style.R — shared ggplot2 figure style for all projects
#
#  Usage:  source("~/R/house_style.R")
#
#  Conventions (finalized 2026-09-04):
#    1. Figures are drawn in R / ggplot2.
#    2. Font is Arial.
#    3. Where possible, statistical values (R²·p·ρ·n) are included in the figure itself.
#
#  The palette and figure types inherit the house style established in the PD cohort project.
# ─────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(ggplot2); library(ragg); library(systemfonts); library(scales)
})

#' Check whether an optional package is installed. Falls back to a base implementation if not.
hs_has <- function(pkg) requireNamespace(pkg, quietly = TRUE)

# ── Palette ───────────────────────────────────────────────────────────────────
HS_PAL <- list(
  red   = "#C0504D",   # positive (+) direction / emphasis
  blue  = "#4A7FC1",   # negative (−) direction
  cream = "#F1F0E9",   # background tone
  ink   = "#16150F",   # body text
  gray  = "#8A8578",   # secondary text (stat labels, captions)
  edge  = "#DAD6C9",   # axis lines / gridlines
  track = "#E6E2D6",   # ranked bar track
  mid   = "#EFE9DD",   # diverging scale midpoint / confidence band
  groups = c("#4A7FC1", "#D98A4A", "#4AAE8A", "#B25FA8",
             "#C0504D", "#7E9BC4", "#A38B6B", "#5F8A7A")
)

#' Group color vector. Cycles through n colors.
hs_colors <- function(n) rep(HS_PAL$groups, length.out = n)

# ── Fonts ─────────────────────────────────────────────────────────────────────
#' Body font.
#' This machine has had actual Arial files since 2026-09-22 (~/.fonts/Arial*.ttf, 4 variants).
#' On machines without it, fontconfig substitutes Liberation Sans (metric-compatible, same
#' character widths), so "Arial" can be used as-is -- the layout is identical either way.
#' @param cjk If TRUE, use a font for Korean (Hangul). Arial has no Hangul glyphs. However,
#'   ragg does glyph-level fallback, so leaving this FALSE usually does not break Korean text
#'   either -- when the proportion of Korean text is large, specifying this explicitly makes
#'   rendering more reliable.
hs_font <- function(cjk = FALSE) {
  if (cjk) {
    fams <- unique(systemfonts::system_fonts()$family)
    for (f in c("Noto Sans CJK KR", "Malgun Gothic", "NanumGothic", "AppleGothic")) {
      if (f %in% fams) return(f)
    }
    warning("Could not find a Korean font -- Korean text in the figure may render as tofu boxes (□)")
  }
  "Arial"
}

# ── Theme ─────────────────────────────────────────────────────────────────────
theme_house <- function(base_size = 11, cjk = FALSE) {
  theme_minimal(base_size = base_size, base_family = hs_font(cjk)) +
    theme(
      text             = element_text(colour = HS_PAL$ink),
      plot.title       = element_text(face = "bold", hjust = 0, size = rel(1.0)),
      plot.subtitle    = element_text(colour = HS_PAL$gray, size = rel(0.85)),
      plot.caption     = element_text(colour = HS_PAL$gray, size = rel(0.75), hjust = 0),
      axis.title       = element_text(size = rel(0.85)),
      axis.text        = element_text(colour = HS_PAL$ink, size = rel(0.8)),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(colour = HS_PAL$edge, linewidth = 0.3, linetype = "dotted"),
      panel.border     = element_blank(),
      axis.line        = element_line(colour = HS_PAL$edge, linewidth = 0.4),
      strip.text       = element_text(face = "bold", hjust = 0, size = rel(0.9)),
      legend.position  = "bottom",
      legend.title     = element_blank(),
      legend.key.size  = unit(0.8, "lines"),
      plot.title.position = "plot"
    )
}

# ── Statistical helpers ────────────────────────────────────────────────────────────────
hs_fmt_p <- function(p) {
  ifelse(is.na(p), "p = —", ifelse(p < 0.001, "p < 0.001", sprintf("p = %.3f", p)))
}

#' Regression/correlation summary -- for scatter plots. n, R², Spearman ρ, p.
#' @note R² is only the explanatory power of the linear fit -- it does not mean "the two
#'   methods agree." For method comparison, report Bland-Altman (bias/limits of agreement)
#'   alongside it.
hs_stats_fit <- function(x, y) {
  ok <- is.finite(x) & is.finite(y); x <- x[ok]; y <- y[ok]
  if (length(x) < 3) return(list(n = length(x), label = sprintf("n = %d", length(x))))
  fit <- stats::lm(y ~ x); s <- summary(fit)
  pv  <- stats::pf(s$fstatistic[1], s$fstatistic[2], s$fstatistic[3], lower.tail = FALSE)
  rho <- suppressWarnings(stats::cor(x, y, method = "spearman"))
  list(n = length(x), r2 = s$r.squared, slope = unname(coef(fit)[2]),
       intercept = unname(coef(fit)[1]), p = unname(pv), rho = rho,
       label = sprintf("n = %d\nR² = %.3f\nSpearman ρ = %.2f\n%s",
                       length(x), s$r.squared, rho, hs_fmt_p(unname(pv))))
}

#' Between-group comparison -- Mann-Whitney for 2 groups, Kruskal-Wallis for 3+ groups.
hs_stats_group <- function(value, group) {
  ok <- is.finite(value) & !is.na(group)
  value <- value[ok]; group <- droplevels(as.factor(group[ok]))
  if (nlevels(group) < 2 || length(value) < 4) return(list(label = "", p = NA_real_))
  if (nlevels(group) == 2) {
    tt <- suppressWarnings(stats::wilcox.test(value ~ group)); nm <- "Mann–Whitney"
  } else {
    tt <- stats::kruskal.test(value ~ group); nm <- "Kruskal–Wallis"
  }
  list(p = unname(tt$p.value), test = nm,
       label = sprintf("%s %s", nm, hs_fmt_p(unname(tt$p.value))))
}

#' Paired before/after comparison -- Wilcoxon signed-rank.
hs_stats_paired <- function(before, after) {
  ok <- is.finite(before) & is.finite(after)
  b <- before[ok]; a <- after[ok]
  if (length(b) < 5) return(list(label = sprintf("n = %d (insufficient)", length(b)), p = NA_real_))
  d <- a - b
  pv <- if (any(d != 0)) suppressWarnings(stats::wilcox.test(b, a, paired = TRUE)$p.value) else NA_real_
  list(n = length(b), delta_median = stats::median(d), p = pv,
       label = sprintf("n = %d pairs\nΔ median = %+.2f\nWilcoxon %s",
                       length(b), stats::median(d), hs_fmt_p(pv)))
}

#' Place a statistical label in a panel corner. Pinned at ±Inf, independent of axis range.
#' @param corner "tl"/"tr"/"bl"/"br"
hs_annotate <- function(label, corner = "tl", size = 3, cjk = FALSE) {
  if (is.null(label) || !nzchar(label)) return(NULL)
  pos <- switch(corner,
    tl = list(x = -Inf, y =  Inf, h = -0.08, v =  1.15),
    tr = list(x =  Inf, y =  Inf, h =  1.08, v =  1.15),
    bl = list(x = -Inf, y = -Inf, h = -0.08, v = -0.25),
    br = list(x =  Inf, y = -Inf, h =  1.08, v = -0.25))
  annotate("text", x = pos$x, y = pos$y, label = label, hjust = pos$h, vjust = pos$v,
           size = size, colour = HS_PAL$gray, family = hs_font(cjk), lineheight = 1.1)
}

# ── Figure types ────────────────────────────────────────────────────────────────

#' (1) boxplot + jitter -- the default form for between-group comparison.
#' Attaches per-group n to the x-axis labels and puts the test result inside the panel.
#' @param direction_hint Direction note to append to the title (e.g. "↓ = higher value means more abundant")
hs_box <- function(df, group, value, title = NULL, ylab = value,
                   direction_hint = NULL, stats = TRUE, cjk = FALSE) {
  d <- df[is.finite(df[[value]]) & !is.na(df[[group]]), c(group, value)]
  names(d) <- c("g", "v"); d$g <- droplevels(as.factor(d$g))
  st  <- if (stats) hs_stats_group(d$v, d$g) else list(label = "")
  cnt <- tapply(d$v, d$g, length)
  levels(d$g) <- sprintf("%s\nn=%d", levels(d$g), cnt[levels(d$g)])
  ttl <- if (is.null(direction_hint)) title else paste0(title, " (", direction_hint, ")")
  ggplot(d, aes(g, v, fill = g, colour = g)) +
    geom_boxplot(width = 0.55, outlier.shape = NA, alpha = 0.20, linewidth = 0.6) +
    geom_jitter(width = 0.16, size = 1.1, alpha = 0.75, stroke = 0) +
    scale_fill_manual(values = hs_colors(nlevels(d$g))) +
    scale_colour_manual(values = hs_colors(nlevels(d$g))) +
    hs_annotate(st$label, cjk = cjk) +
    labs(title = ttl, x = NULL, y = ylab) +
    theme_house(cjk = cjk) +
    theme(legend.position = "none", panel.grid.major.x = element_blank())
}

#' (2) Scatter plot + regression line + R²/p.
#' @param identity If TRUE, also draw the identity line (y=x) as a dashed line (for method comparison)
hs_scatter <- function(df, x, y, colour = NULL, title = NULL,
                       identity = FALSE, stats = TRUE, cjk = FALSE) {
  d  <- df[is.finite(df[[x]]) & is.finite(df[[y]]), ]
  st <- if (stats) hs_stats_fit(d[[x]], d[[y]]) else list(label = "")
  p  <- ggplot(d, aes(.data[[x]], .data[[y]]))
  if (!is.null(colour)) p <- p + aes(colour = .data[[colour]])
  if (identity) p <- p + geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                                     colour = HS_PAL$gray, linewidth = 0.4)
  p + geom_point(size = 1.3, alpha = 0.7, stroke = 0) +
    geom_smooth(method = "lm", formula = y ~ x, se = TRUE,
                colour = HS_PAL$red, fill = HS_PAL$mid, linewidth = 0.6) +
    scale_colour_manual(values = hs_colors(64), na.value = HS_PAL$gray) +
    hs_annotate(st$label, cjk = cjk) +
    labs(title = title, x = x, y = y) + theme_house(cjk = cjk)
}

#' (3) Sectioned heatmap -- variable x metric correlation.
#' @param d long-format: row_label, col_label, rho, p, section
hs_heatmap <- function(d, title = NULL, sig = 0.05, legend_name = "Spearman ρ", cjk = FALSE) {
  d$row_label <- factor(d$row_label, levels = rev(unique(d$row_label)))
  d$col_label <- factor(d$col_label, levels = unique(d$col_label))
  ggplot(d, aes(col_label, row_label, fill = rho)) +
    geom_tile(colour = "white", linewidth = 1.2) +
    geom_point(data = subset(d, !is.na(p) & p < sig), size = 1.1, colour = HS_PAL$ink) +
    scale_fill_gradient2(low = HS_PAL$blue, mid = HS_PAL$mid, high = HS_PAL$red,
                         midpoint = 0, limits = c(-1, 1), na.value = "white",
                         name = legend_name) +
    { if ("section" %in% names(d))
        facet_grid(rows = vars(section), scales = "free_y", space = "free_y", switch = "y") } +
    labs(title = title, x = NULL, y = NULL,
         caption = sprintf("● p < %.2f (uncorrected)", sig)) +
    theme_house(cjk = cjk) +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          axis.line = element_blank(), axis.text.x = element_text(face = "bold"),
          strip.placement = "outside",
          strip.text.y.left = element_text(angle = 0, hjust = 0, colour = HS_PAL$gray),
          legend.title = element_text(size = rel(0.8)),
          legend.key.width = unit(2.2, "lines"))
}

#' (4) ranked effect bar -- shows at a glance which variable has the strongest association.
#' @param d label, rho, p (+ optional sublabel)
hs_ranked <- function(d, title = NULL, xlim = c(-1, 1), cjk = FALSE) {
  d <- d[order(d$p), ]
  d$label <- factor(d$label, levels = rev(d$label))
  d$dir   <- ifelse(d$rho >= 0, "pos", "neg")
  pad <- diff(xlim) * 0.22
  ggplot(d, aes(y = label)) +
    geom_segment(aes(x = xlim[1], xend = xlim[2], yend = label),
                 colour = HS_PAL$track, linewidth = 2.6, lineend = "round") +
    geom_segment(aes(x = 0, xend = rho, yend = label, colour = dir),
                 linewidth = 2.6, lineend = "round") +
    geom_text(aes(x = xlim[2] + pad * 0.12, label = hs_fmt_p(p)), hjust = 0,
              size = 2.7, colour = HS_PAL$ink, family = "mono") +
    scale_colour_manual(values = c(pos = HS_PAL$red, neg = HS_PAL$blue), guide = "none") +
    scale_x_continuous(limits = c(xlim[1], xlim[2] + pad), expand = c(0, 0)) +
    labs(title = title, x = NULL, y = NULL) +
    theme_house(cjk = cjk) +
    theme(panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
          axis.line = element_blank(), axis.text.x = element_blank(),
          axis.ticks = element_blank())
}

#' (5) Longitudinal spaghetti plot -- before/after comparison connecting the same subject with a line.
#' @param time_levels Order of the time points. If not given, factor() will **sort alphabetically**,
#'   causing accidents like B/M/E becoming B/E/M -- always specify this when the time points have an order.
hs_paired <- function(df, id, time, value, group = NULL, title = NULL,
                      ylab = value, time_levels = NULL, stats = TRUE, cjk = FALSE) {
  d <- df[is.finite(df[[value]]), ]
  d$.id <- d[[id]]
  d$.t  <- if (is.null(time_levels)) factor(d[[time]]) else factor(d[[time]], levels = time_levels)
  d$.v  <- d[[value]]
  lab <- ""
  if (stats && nlevels(d$.t) >= 2) {
    w  <- stats::reshape(d[, c(".id", ".t", ".v")], idvar = ".id", timevar = ".t",
                         direction = "wide")
    st <- hs_stats_paired(w[[2]], w[[ncol(w)]])
    lab <- st$label
  }
  p <- ggplot(d, aes(.t, .v))
  if (!is.null(group)) p <- p + aes(colour = .data[[group]])
  p + geom_line(aes(group = .id), colour = HS_PAL$gray, linewidth = 0.4, alpha = 0.45) +
    geom_point(size = 1.4, alpha = 0.75, stroke = 0) +
    stat_summary(fun = stats::median, geom = "crossbar", width = 0.45,
                 colour = HS_PAL$ink, linewidth = 0.4) +
    scale_colour_manual(values = hs_colors(64), na.value = HS_PAL$gray) +
    hs_annotate(lab, cjk = cjk) +
    labs(title = title, x = NULL, y = ylab) + theme_house(cjk = cjk) +
    theme(panel.grid.major.x = element_blank())
}

#' (6) forest plot -- point estimate + confidence interval, against a reference line.
#' Draws the effect size (ref=0 for a difference, ref=1 for a ratio/OR/HR) and the 95% CI
#' on each row, with the p-value attached in monospace on the right (house style's
#' statistical-notation convention, the same practice as hs_ranked). Significance is
#' double-encoded not just by color but also by point shape (filled = significant / open
#' circle = not significant) -- so it is still distinguishable in black-and-white print.
#' @param d label, estimate, ci_low, ci_high (+ optional p, group, panel columns)
#' @param ref Reference-line position (difference=0, ratio/OR/HR=1)
#' @param log_x If TRUE, use a log scale on the x-axis (for ratio metrics like OR/HR)
#' @param order "effect" (default, ascending by effect size) | "label" (alphabetical by label) |
#'   "none" (keep the input order as-is -- when passing already-sorted data, or when you want
#'   to keep the same label order across facets)
#' @param group Column name to split point color by (e.g. primary vs. secondary contrast,
#'   predictor-group distinction). If omitted, falls back to the same convention as hs_ranked:
#'   automatic positive (RED) / negative (BLUE) coloring relative to the reference line.
#' @param panel Column name to split by with facet_grid(rows=) (for multiple forest panels,
#'   e.g. stacking several stratified subgroups into one figure). The label order is shared
#'   across the whole patch.
#' @param show_p Whether to attach the p-value in monospace on the right (default TRUE; turned
#'   off automatically if there is no p column)
hs_forest <- function(d, title = NULL, xlab = NULL, ref = 0, log_x = FALSE,
                      order = c("effect", "label", "none"), group = NULL, panel = NULL,
                      show_p = TRUE, cjk = FALSE) {
  order <- match.arg(order)
  stopifnot(all(c("label", "estimate", "ci_low", "ci_high") %in% names(d)))
  d <- as.data.frame(d)
  if (!("p" %in% names(d))) { d$p <- NA_real_; show_p <- FALSE }
  d$.sig <- ifelse(is.na(d$ci_low) | is.na(d$ci_high), NA,
                   !(d$ci_low <= ref & d$ci_high >= ref))
  d$.grp <- if (!is.null(group)) as.factor(d[[group]]) else
    factor(ifelse(d$estimate >= ref, "pos", "neg"), levels = c("pos", "neg"))

  ord <- switch(order,
    effect = order(d$estimate),
    label  = order(as.character(d$label)),
    none   = seq_len(nrow(d)))
  d <- d[ord, ]
  d$label <- factor(d$label, levels = unique(d$label))

  pal <- if (!is.null(group)) stats::setNames(hs_colors(nlevels(d$.grp)), levels(d$.grp))
         else c(pos = HS_PAL$red, neg = HS_PAL$blue)

  # right-hand margin for the p column -- computed dynamically from the data range (log axis uses a multiplicative margin)
  rng_lo <- min(c(d$ci_low, d$estimate, ref), na.rm = TRUE)
  rng_hi <- max(c(d$ci_high, d$estimate, ref), na.rm = TRUE)
  if (log_x) {
    l_lo <- log10(max(rng_lo, 1e-6)); l_hi <- log10(max(rng_hi, 1e-6))
    span <- max(l_hi - l_lo, 1e-6)
    d$.p_x   <- 10^(log10(pmax(d$ci_high, ref)) + span * 0.06)
    xlim_hi  <- if (show_p) 10^(l_hi + span * 0.32) else NA
  } else {
    span <- max(rng_hi - rng_lo, 1e-6)
    d$.p_x  <- pmax(d$ci_high, ref) + span * 0.06
    xlim_hi <- if (show_p) rng_hi + span * 0.32 else NA
  }

  p_obj <- ggplot(d, aes(x = estimate, y = label)) +
    geom_vline(xintercept = ref, colour = HS_PAL$gray, linetype = "dashed", linewidth = 0.4) +
    geom_errorbarh(aes(xmin = ci_low, xmax = ci_high, colour = .grp), height = 0, linewidth = 0.7) +
    geom_point(aes(colour = .grp, shape = .sig), size = 2.4, stroke = 1.0, fill = "white") +
    scale_colour_manual(values = pal, na.value = HS_PAL$gray,
                        guide = if (!is.null(group)) "legend" else "none") +
    scale_shape_manual(values = c(`TRUE` = 16, `FALSE` = 21), na.value = 4, guide = "none") +
    labs(title = title, x = xlab, y = NULL, colour = NULL) +
    theme_house(cjk = cjk) +
    theme(panel.grid.major.y = element_blank())

  if (show_p) {
    p_obj <- p_obj + geom_text(aes(x = .p_x, label = hs_fmt_p(p)), hjust = 0,
                                size = 2.7, colour = HS_PAL$ink, family = "mono")
  }
  if (log_x) {
    p_obj <- p_obj + scale_x_log10(limits = c(NA, if (is.na(xlim_hi)) NA else xlim_hi))
  } else if (!is.na(xlim_hi)) {
    p_obj <- p_obj + scale_x_continuous(limits = c(NA, xlim_hi))
  }
  if (!is.null(panel)) {
    p_obj <- p_obj + facet_grid(rows = vars(.data[[panel]]), scales = "free_y", space = "free_y")
  }
  p_obj
}

# ── Optional-package convenience layer ──────────────────────────────────────────────────

#' Add the regression equation + R² + p to a scatter plot, as a formula. Auto-recomputed per facet (ggpmisc).
hs_poly_eq <- function(size = 3) {
  if (!hs_has("ggpmisc")) { warning("ggpmisc not installed -- use hs_annotate(hs_stats_fit(...)) instead"); return(NULL) }
  ggpmisc::stat_poly_eq(
    mapping = aes(label = paste(after_stat(eq.label), after_stat(rr.label),
                                after_stat(p.value.label), sep = "~~~")),
    formula = y ~ x, parse = TRUE, size = size, colour = HS_PAL$gray,
    label.x = "left", label.y = "top")
}

#' Between-group significance brackets (ggpubr). If comparisons is not given, shows only the overall test's p.
hs_signif <- function(comparisons = NULL, size = 3, method = "wilcox.test") {
  if (!hs_has("ggpubr")) { warning("ggpubr not installed -- use hs_annotate(hs_stats_group(...)) instead"); return(NULL) }
  if (is.null(comparisons)) {
    ggpubr::stat_compare_means(size = size, colour = HS_PAL$gray)
  } else {
    ggpubr::stat_compare_means(comparisons = comparisons, method = method, size = size,
                               colour = HS_PAL$ink, tip.length = 0.01, bracket.size = 0.3)
  }
}

# ── Saving ─────────────────────────────────────────────────────────────────────
#' Save with ragg::agg_png. More accurate font handling than base png(), with glyph fallback.
hs_save <- function(plot, path, width = 8, height = 5, dpi = 200) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  ragg::agg_png(path, width = width, height = height, units = "in",
                res = dpi, background = "white")
  print(plot); invisible(grDevices::dev.off())
  path
}

#' Vector output (for manuscript submission). svglite or pdf.
#'
#' The extension and the content must not disagree. The previous version silently fell back
#' to cairo_pdf when svglite was unavailable, while **keeping the filename as .svg** -- so the
#' file the author opened to edit was actually a PDF. This actually happened on 2026-09-14,
#' and all 11 figures were delivered in that state. Since the extension must say what is
#' actually inside the file, if .svg is requested and svglite is unavailable, this does not
#' fall back -- it **stops**.
hs_save_vector <- function(plot, path, width = 8, height = 5) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  if (grepl("\\.svg$", path)) {
    if (!hs_has("svglite"))
      stop("hs_save_vector(): svglite is not installed, so '", basename(path),
           "' cannot be written as SVG. Falling back to cairo_pdf under a .svg ",
           "name would hand the author a file that does not open in an SVG ",
           "editor. Install svglite, or ask for a .pdf path.")
    svglite::svglite(path, width = width, height = height)
    on.exit({
      # Font stack for submission. The name written into the SVG puts Arial first. On this
      # machine the actual Arial font exists, so it renders as-is; on a machine without it,
      # it falls back to Liberation Sans, which is metric-compatible so character widths and
      # line breaks are identical.
      #
      # WARNING: **Do not use single quotes.** svglite wraps the style attribute as
      #    style='...'. Putting 'Liberation Sans' inside that breaks the attribute off right
      #    there, corrupting the whole SVG (measured directly on 2026-09-22 -- every figure's
      #    SVG became not well-formed). Double quotes inside a single-quoted attribute are
      #    legal, so double quotes are used instead.
      if (file.exists(path)) {
        txt <- readLines(path, warn = FALSE)
        txt <- gsub('font-family: *"?(Liberation Sans|DejaVu Sans|Arial)"?',
                    'font-family: Arial, "Liberation Sans", Helvetica, sans-serif',
                    txt)
        writeLines(txt, path)
        # After rewriting, **always parse it to check.** This substitution has broken SVGs before.
        ok <- tryCatch({ xml2::read_xml(path); TRUE }, error = function(e) conditionMessage(e))
        if (!isTRUE(ok))
          stop("hs_save_vector(): the SVG broke after the font substitution -- ", basename(path), ": ", ok)
      }
    }, add = TRUE)
  } else {
    grDevices::cairo_pdf(path, width = width, height = height)
  }
  print(plot); invisible(grDevices::dev.off())
  path
}
