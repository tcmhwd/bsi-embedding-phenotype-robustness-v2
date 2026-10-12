#!/usr/bin/env python3
"""Assemble tidy CSVs for the Paper 1 figures.

CSVs are named for their CONTENT, not their figure number: figure order is a
presentation decision that has already changed once, and a rename cascade
every time it changes is how a stale reference gets left behind.

Separated from the plotting (`scripts/33_figures.R`) so that every number a
figure draws is on disk as a readable row first.  The explainer page cites
these CSVs, and nothing is retyped from a JSON into a plot by hand.

One symmetry repair happens here rather than in the caption.  The Korean
cross-encoder agreement is summarised over 50 values (10 subsample draws x 5
seeds, from `out/30_structure/`), but the MIMIC cross-encoder number on disk
is a single seed-42 comparison.  Drawing a range beside a point would invite
reading the point as more certain, so the MIMIC cross-encoder agreement is
recomputed here across the same 10 clustering seeds whose labels Task 27e
already wrote.  Both sides then carry a range over seeds, and the range is
labelled as a range -- never as a confidence interval, which none of these
are.
"""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

ROOT = Path("/data/tcmhwd/AI/AST_ai/phenotype_revision")
STRUCTURE = ROOT / "out/30_structure"
MIMIC_LABELS = ROOT / "out/27_mimic/replicate/labels"
MIMIC_REPLICATE = ROOT / "out/27_mimic/replicate"
SCREEN = ROOT / "out/22_published_objects/task_b_artifact_screen/artifact_screen_matrix.csv"
SURVIVAL = ROOT / "out/29_fixes/17e_rerun_verify/survival_matrix.csv"
OPENPOOL = ROOT / "out/28_openpool/sofa_stratified_openpool.json"
S6_READINGS = ROOT / "out/28_openpool/s6_three_readings.csv"
SIZE_STAB = ROOT / "out/32_size_stability/size_stability.json"
DOMINANCE = ROOT / "out/40_field_dominance/field_dominance.json"
ABLATION = ROOT / "out/44_ablation/ablation.json"
# 11-seed supersession of the four Qwen3 arms (task 47, 2026-09-08). When
# present it REPLACES the 4-seed sources below for Qwen3; those remain on
# disk and remain correct at 4 seeds.
QWEN3_SEEDS = ROOT / "out/47_qwen3_seeds/qwen3_extra_seeds.json"
# 3-small dominant-field ablation (task 49, 2026-09-08) -- the third and last
# non-degenerate Korean encoder. Scored against 3-small's STORED partition
# because no unmodified re-embedding arm exists for it; the within-encoder
# control (no_organism) carries the same re-embedding difference, which is
# what makes the pair interpretable.
SMALL_ABLATION = ROOT / "out/49_3small_ablation/3small_ablation.json"
MASK_ABLATION = ROOT / "out/45_audit9/mask_ablation.json"
THREE_LARGE_SEEDS = ROOT / "out/45_3large_seeds/three_large_seeds.json"
AUDIT4_SEEDS = ROOT / "out/25_audit4/seeds"
NO_ORGANISM_EXTRA_SEEDS = ROOT / "out/46_audit10/no_organism_extra_seeds.json"
PUB_VARIANTS = ROOT / "out/21_published_variants/cluster"
# 2026-09-08 planning freeze (05_FIGURE_CLAIM_MAP.md): panel-redesign sources.
MATCHED_ENCODERS = ROOT / "out/39_audit6/matched_encoders.json"
CONCORDANCE_WHOLE = ROOT / "out/39_audit6/concordance_whole.json"
DOMINANCE_CHECKS = ROOT / "out/41_audit7/dominance_checks.json"
KNN_PURITY = ROOT / "out/45_audit9/knn_purity.json"
VARIANT_GEOMETRY_ALL20 = ROOT / "out/26_controls/a_variant_geometry_all20.json"
RETRIEVAL_COMPARISON = ROOT / "out/23_retrieval/comparison_table.csv"
OUT = ROOT / "out/33_figures"

SEEDS = list(range(1, 11))
ENCODERS = ("bge-m3", "qwen3-0.6b")


def ari_noise_excluded(a: np.ndarray, b: np.ndarray) -> float:
    both = (a != -1) & (b != -1)
    return float(adjusted_rand_score(a[both], b[both]))


def mimic_labels(encoder: str, seed: int) -> np.ndarray:
    table = pd.read_parquet(MIMIC_LABELS / f"{encoder}_seed{seed}.parquet")
    return table.sort_values("admission_ID")["cluster"].to_numpy()


def fig1_agreement() -> pd.DataFrame:
    """Cross-seed and cross-encoder agreement, both cohorts, same metric.

    2026-09-08 planning freeze (05_FIGURE_CLAIM_MAP.md section "Figure 1"):
    Panel A (Korea) headline pair used to be BGE-M3 vs Qwen3, whose ARI is
    forced near zero by BGE-M3's own near-degenerate partition (k median 3,
    largest cluster 97.4% of the corpus on Korean matched refits) rather
    than by genuine cross-encoder disagreement -- a reviewer who checks
    cluster counts invalidates that pair in the first paragraph. Panel A
    now carries the three NON-degenerate Korean pairs (Qwen3, 3-small,
    3-large -- none of which collapses to a handful of clusters) plus their
    own three cross-seed rows; BGE-M3 moves to Panel C (fig1c_degeneracy)
    as the excluded/degenerate case, flagged with its reason rather than
    silently dropped. Panel B (MIMIC-IV) is unchanged: neither MIMIC
    encoder is degenerate (largest cluster 8.8% / 6.4%), so BGE-M3 vs
    Qwen3 there is a valid pair and stays as-is.

    No new computation: every number below is read from
    `out/39_audit6/matched_encoders.json` (job 588... audit 6, n-matched
    Korean refits, 10 subsample draws x 5 seeds = 50 fits/encoder) --
    `cross_seed_raw_pairs[encoder].median_of_draw_medians_ari` /
    `.ari.{min,max}` / `.ami.median` for cross-seed rows, and
    `cross_encoder_seed_matched[pair].{ari,ami}.median/min/max` for
    cross-encoder rows (the three non-BGE pairs are ALWAYS non-degenerate
    for both partitions, so the file's top-level n=50 statistics already
    equal its own `nondegenerate_both_k_ge_5` subset for these three pairs
    -- verified 2026-09-08, no separate filtering needed). This replaces
    the previous per-draw reconstruction from `out/30_structure/
    matched_draw*.json`, which only ever stored BGE-M3/Qwen3 (not
    3-small/3-large) and is now used solely as the MIMIC cross-encoder
    seed source below.

    AMI is carried alongside ARI because ARI is bounded low by a granularity
    mismatch: BGE-M3 returns 2-4 clusters on the Korean corpus where Qwen3
    returns 23-25, and ARI penalises that split even under perfect nesting.
    AMI does not, so a low ARI with a low AMI means genuine disagreement
    whereas a low ARI with a high AMI would mean nesting.
    """
    rows = []
    matched = json.loads(MATCHED_ENCODERS.read_text())

    KOREA_NONDEGENERATE = ["qwen3-0.6b", "openai-3large", "openai-3small"]
    KOREA_PAIR_LABEL = {
        "qwen3-0.6b vs openai-3small": "Qwen3 vs 3-small",
        "qwen3-0.6b vs openai-3large": "Qwen3 vs 3-large",
        "openai-3small vs openai-3large": "3-small vs 3-large",
    }
    for encoder in KOREA_NONDEGENERATE:
        block = matched["cross_seed_raw_pairs"][encoder]
        rows.append(dict(
            panel="A", cohort="Korea (n-matched)", comparison="Cross-seed", encoder=encoder,
            estimate=float(block["median_of_draw_medians_ari"]),
            lo=float(block["ari"]["min"]), hi=float(block["ari"]["max"]),
            ami=float(block["ami"]["median"]), n_values=block["ari"]["n"],
            n_unit="raw pairs (range exact; point = median of 10 draw medians)",
        ))
    for pair, label in KOREA_PAIR_LABEL.items():
        block = matched["cross_encoder_seed_matched"][pair]
        rows.append(dict(
            panel="A", cohort="Korea (n-matched)", comparison="Cross-encoder", encoder=label,
            estimate=float(block["ari"]["median"]), lo=float(block["ari"]["min"]),
            hi=float(block["ari"]["max"]), ami=float(block["ami"]["median"]),
            n_values=block["ari"]["n"], n_unit="raw pairs (10 draws x 5 seeds)",
        ))

    # MIMIC-IV: unchanged from the pre-freeze version. Both encoders
    # partition properly there (largest cluster 8.8% / 6.4%), so BGE-M3 vs
    # Qwen3 is a valid, non-degenerate pair in this cohort and stays put.
    for encoder in ENCODERS:
        payload = json.loads((MIMIC_REPLICATE / f"cross_seed_{encoder}.json").read_text())
        block = payload["ari_noise_excluded"]
        # 27e stored only ARI for cross-seed, so AMI is recomputed from the
        # same stored labels rather than left blank next to a Korean AMI.
        ami_pairs = []
        for a_seed, b_seed in itertools.combinations(SEEDS, 2):
            a, b = mimic_labels(encoder, a_seed), mimic_labels(encoder, b_seed)
            both = (a != -1) & (b != -1)
            ami_pairs.append(float(adjusted_mutual_info_score(a[both], b[both])))
        rows.append(dict(
            panel="B", cohort="MIMIC-IV", comparison="Cross-seed", encoder=encoder,
            estimate=block["median"], lo=block["min"], hi=block["max"],
            ami=float(np.median(ami_pairs)), n_values=payload.get("n_pairs"),
            n_unit="raw pairs",
        ))

    values_ari, values_ami = [], []
    for seed in SEEDS:
        a, b = mimic_labels(ENCODERS[0], seed), mimic_labels(ENCODERS[1], seed)
        both = (a != -1) & (b != -1)
        values_ari.append(float(adjusted_rand_score(a[both], b[both])))
        values_ami.append(float(adjusted_mutual_info_score(a[both], b[both])))
    rows.append(dict(
        panel="B", cohort="MIMIC-IV", comparison="Cross-encoder", encoder="BGE-M3 vs Qwen3",
        estimate=float(np.median(values_ari)), lo=float(np.min(values_ari)),
        hi=float(np.max(values_ari)), ami=float(np.median(values_ami)),
        n_values=len(values_ari), n_unit="raw pairs",
    ))

    frame = pd.DataFrame(rows)
    frame["label"] = frame["encoder"].replace({
        "bge-m3": "BGE-M3", "qwen3-0.6b": "Qwen3-0.6B", "openai-3large": "3-large",
        "openai-3small": "3-small",
    })
    return frame


def fig1c_degeneracy() -> pd.DataFrame:
    """Figure 1 Panel C: which Korean encoder fits are degenerate, and why
    the BGE-M3 vs Qwen3 pair is excluded from Panel A rather than shown as
    the headline cross-encoder disagreement.

    No new computation -- read from `out/39_audit6/matched_encoders.json`
    (`k_per_fit`, `largest_cluster_share`: median over the same 50
    n-matched refits as fig1_agreement, plus `frac_k_le_4`) and
    `out/39_audit6/concordance_whole.json` (`korea_full_corpus.partitions`:
    the single whole-corpus fit, n=32,329). The BGE-M3 vs Qwen3 pair's ARI
    (median 0.001, from `matched_encoders.json`'s top-level
    `cross_encoder_seed_matched`, n=50 raw pairs) is carried here, flagged
    as excluded-with-reason, rather than in Panel A's non-degenerate set.
    """
    matched = json.loads(MATCHED_ENCODERS.read_text())
    whole = json.loads(CONCORDANCE_WHOLE.read_text())["korea_full_corpus"]["partitions"]
    whole_key = {"bge-m3": "bge-m3", "qwen3-0.6b": "qwen3-0.6b",
                 "openai-3small": "openai-3small", "openai-3large": "openai-3large-published"}
    rows = []
    for encoder in ("bge-m3", "qwen3-0.6b", "openai-3small", "openai-3large"):
        k_block = matched["k_per_fit"][encoder]
        share_block = matched["largest_cluster_share"][encoder]
        w = whole[whole_key[encoder]]
        n_fits = share_block["n"]
        rows.append(dict(
            encoder=encoder,
            label={"bge-m3": "BGE-M3", "qwen3-0.6b": "Qwen3-0.6B",
                   "openai-3small": "3-small", "openai-3large": "3-large"}[encoder],
            k_median_refits=k_block["median"], n_fits=n_fits,
            frac_k_le_4=k_block["frac_k_le_4"],
            n_k_le_4=int(round(k_block["frac_k_le_4"] * n_fits)),
            largest_share_median_refits=share_block["median"],
            k_wholecorpus=w["k"], largest_share_wholecorpus=w["largest_share"],
            degenerate=(encoder == "bge-m3"),
        ))
    frame = pd.DataFrame(rows)

    pair = matched["cross_encoder_seed_matched"]["bge-m3 vs qwen3-0.6b"]
    excluded = pd.DataFrame([dict(
        pair_label="BGE-M3 vs Qwen3-0.6B (excluded from panel A)",
        ari_median=pair["ari"]["median"], ari_min=pair["ari"]["min"], ari_max=pair["ari"]["max"],
        ami_median=pair["ami"]["median"], n_values=pair["ari"]["n"],
        reason="BGE-M3's Korean partition is itself near-degenerate (k median 3, "
               "largest cluster 97.4% of the corpus); its near-zero ARI against any "
               "other encoder reflects a near-degenerate partition, not only disagreement.",
    )])
    return frame, excluded


def fig1b_cluster_sizes() -> pd.DataFrame:
    """Why the Korean cross-encoder ARI is near zero: BGE-M3 barely partitions.

    Without this panel a reader takes ARI 0.001 as 'two encoders disagree
    about structure'.  It is closer to 'one encoder found almost no
    structure to disagree about' -- BGE-M3 puts 88.8% of the Korean corpus
    in a single cluster.  Reporting the agreement without the size
    distribution overstates what the comparison shows.
    """
    rows = []
    for cohort, base, encoders in [
        ("Korea", ROOT / "out/18_encoder_dec/recluster", ["bge-m3", "qwen3-0.6b", "openai-3small"]),
        ("MIMIC-IV", None, ENCODERS),
    ]:
        for encoder in encoders:
            if cohort == "Korea":
                payload = np.load(base / encoder / "labels.npz", allow_pickle=True)
                key = [k for k in payload.files if "label" in k.lower()][0]
                labels = payload[key]
            else:
                labels = mimic_labels(encoder, 42)
            counts = pd.Series(labels[labels != -1]).value_counts()
            rows.append(dict(
                cohort=cohort, encoder=encoder, n=int(len(labels)),
                k=int(len(counts)), noise_frac=float((labels == -1).mean()),
                largest_cluster=int(counts.iloc[0]),
                largest_share=float(counts.iloc[0] / len(labels)),
                top5="; ".join(str(int(v)) for v in counts.head(5)),
            ))
    return pd.DataFrame(rows)


def fig2_token_lookup() -> pd.DataFrame:
    """Per-published-cluster token lookup F1 under two lookup definitions.

    The pre-specified lookup keys on organism / hospital / year / department
    and recovers 13 of 16 clusters outright.  It does NOT contain the episode
    count -- which the manuscript itself names as the single largest
    contributor to cluster assignment -- so drawing only that lookup would
    let three clusters look like genuine structure.  The exploratory lookup
    adds episode count, ICU and ventilation, and is drawn beside it, because
    the distance between the two points is the paper's actual argument.

    Both columns come from `out/28_openpool/s6_three_readings.csv`; reading 2
    (narrative-only) is used as the exploratory arm rather than reading 3,
    which also admits `day_band` -- derived from length of stay, never
    rendered in the narrative, and outcome-adjacent.
    """
    survival = pd.read_csv(SURVIVAL)
    screen = pd.read_csv(SCREEN)
    readings = pd.read_csv(S6_READINGS)

    frame = survival[["cluster", "size", "ALL_AXES_pass"]].merge(
        screen[["cluster", "labels", "organism_defined", "timing_long_stay"]],
        on="cluster", validate="one_to_one",
    ).merge(
        readings[["cluster", "lookup_f1_reading1_prespec",
                  "lookup_f1_reading2_narrative_exploratory"]],
        on="cluster", validate="one_to_one",
    ).rename(columns={
        "lookup_f1_reading1_prespec": "f1_prespec",
        "lookup_f1_reading2_narrative_exploratory": "f1_extended",
    })

    frame["label"] = frame.apply(
        lambda r: f"C{int(r['cluster'])} (n={int(r['size'])})", axis=1
    )
    frame["screen_label"] = frame["labels"].str.replace(
        r"timing-defined \(long stay / post-index\)", "timing (long stay)", regex=True
    ).str.replace("organism-defined", "organism", regex=False)
    frame["gain"] = frame["f1_extended"] - frame["f1_prespec"]
    return frame.sort_values("f1_prespec", ascending=False)


def fig3_retrieval() -> pd.DataFrame:
    """Open-pool retrieval AUROC by SOFA-computability stratum."""
    payload = json.loads(OPENPOOL.read_text())
    rows = []
    for stratum, block in payload["strata"].items():
        for key, entry in block["retrieval_k25_openpool"].items():
            rows.append(dict(
                stratum=stratum, n=block["n"], event_rate=block["event_rate"],
                representation=entry["representation"], variant=entry["variant"],
                estimate=entry["auroc_mean"], sd=entry["auroc_std"],
                lo=entry["auroc_mean"] - 1.96 * entry["auroc_std"],
                hi=entry["auroc_mean"] + 1.96 * entry["auroc_std"],
                n_folds=entry["n_folds_scored"],
            ))
    frame = pd.DataFrame(rows)
    frame["label"] = (
        frame["representation"].replace({"gower": "Structured (Gower)",
                                          "openai-3large": "Embedding (3-large)"})
        + frame["variant"].map({"unrestricted": "", "past_only": ", past cases only"})
    )
    frame["stratum_label"] = frame["stratum"].map({
        "sofa_computable": "SOFA computable (n=3,793)",
        "sofa_not_computable": "SOFA not computable (n=28,536)",
    })
    return frame


def fig4a_retrieval_ladder() -> pd.DataFrame:
    """Figure 4 Panel A: single-axis AUROC ladder from the cluster labels
    the phenotypes were meant to support, up through the two retrieval
    representations, for 90-day mortality across the whole cohort.

    No new computation -- five rows selected from
    `out/23_retrieval/comparison_table.csv` (k=25, `unrestricted`), plus the
    matching `past_only` row for the two retrieval rungs drawn as an open
    point on the same line (05_FIGURE_CLAIM_MAP.md section "Figure 4",
    panel A spec)."""
    df = pd.read_csv(RETRIEVAL_COMPARISON)

    def pick(repr_prefix: str, variant_prefix: str, k=None) -> pd.Series:
        sub = df[df["representation"].str.startswith(repr_prefix) &
                  df["variant"].str.startswith(variant_prefix)]
        if k is not None:
            sub = sub[sub["k"] == k]
        assert len(sub) == 1, f"expected 1 row for {repr_prefix}/{variant_prefix}/{k}, got {len(sub)}"
        return sub.iloc[0]

    rungs = [
        dict(rung=1, label="Cluster labels (originally derived, k=16)",
             row=pick("published_16", "n/a")),
        dict(rung=2, label="Charlson (alone)", row=pick("Charlson", "n/a")),
        dict(rung=3, label="SOFA (alone)", row=pick("SOFA", "n/a")),
        dict(rung=4, label="Embedding retrieval (3-large, k=25)",
             row=pick("openai-3large", "unrestricted", 25.0),
             past_row=pick("openai-3large", "past_only", 25.0)),
        dict(rung=5, label="Structured retrieval (Gower, k=25)",
             row=pick("gower", "unrestricted", 25.0),
             past_row=pick("gower", "past_only", 25.0)),
    ]
    rows = []
    for r in rungs:
        row = r["row"]
        rows.append(dict(
            rung=r["rung"], label=r["label"], variant="unrestricted",
            estimate=float(row["auroc_mean"]), sd=float(row["auroc_std"]),
            n_folds=row["n_folds_scored"],
        ))
        if "past_row" in r:
            pr = r["past_row"]
            rows.append(dict(
                rung=r["rung"], label=r["label"], variant="past_only",
                estimate=float(pr["auroc_mean"]), sd=float(pr["auroc_std"]),
                n_folds=pr["n_folds_scored"],
            ))
    return pd.DataFrame(rows)


def fig4_size_stability() -> pd.DataFrame | None:
    # Audit finding D1 (2026-09-08): function name and CLI log labels below
    # predate the manuscript's final figure numbering and were never
    # updated -- this feeds `size_stability.csv`, drawn by 33_figures.R
    # into `figures/fig6_size_stability.{png,svg}`, displayed/cited as
    # Supplementary Figure S3, not "Figure 4". Kept unrenamed (the name is
    # internal-only; the output CSV name, `size_stability.csv`, has always
    # been generic and unaffected by this).
    if not SIZE_STAB.exists():
        return None
    payload = json.loads(SIZE_STAB.read_text())
    rows = []
    for column, block in payload["spearman_size_vs_axis"].items():
        if block.get("rho") is None:
            continue
        rows.append(dict(
            column=column, label=block["axis_label"], rho=block["rho"],
            lo=block["ci95_bootstrap"][0], hi=block["ci95_bootstrap"][1],
            p=block["p_holm"], p_nominal=block["p_permutation_two_sided"],
            significant_after_holm=block["significant_after_holm"], n=block["n"],
        ))
    return pd.DataFrame(rows)


def fig5_field_dominance() -> pd.DataFrame | None:
    """Which rendered template field dominates each encoder's geometry.

    Audit finding D1 (2026-09-08): function name and CLI log labels below
    predate the manuscript's final figure numbering -- this feeds
    `field_dominance.csv`, drawn by 33_figures.R into
    `figures/fig2_field_dominance.{png,svg}`, displayed/cited as body
    Figure 2, not "Figure 5". Kept unrenamed (the output CSV name,
    `field_dominance.csv`, has always been generic and unaffected).

    This is the paper's mechanism: MIMIC's two encoders are both dominated
    by the organism line and therefore agree with each other, while the
    Korean encoders are dominated by different fields (organism for
    3-large, treatment department for 3-small) and therefore agree with
    nothing.  Values are medians over every stored fit -- 50 per Korean
    encoder (10 subsample draws x 5 seeds), 10 per MIMIC encoder -- so a
    cell is not one lucky seed.
    """
    if not DOMINANCE.exists():
        return None
    payload = json.loads(DOMINANCE.read_text())
    rows = []
    for cohort_key, cohort_label in [("korea_matched_refits", "Korea"),
                                      ("mimic", "MIMIC-IV")]:
        for encoder, block in payload["cohorts"][cohort_key].items():
            for field, value in block["per_field_ami_median"].items():
                rows.append(dict(
                    cohort=cohort_label, encoder=encoder, field=field, ami=value,
                    is_dominant=(field == block["modal_dominant_field"]),
                    modal_share=block["modal_share"],
                    top_ami=block["top_ami_median"],
                    margin=block["margin_over_runner_up_median"],
                    n_fits=block["n_fits"],
                ))
    return pd.DataFrame(rows)


def fig2b_conditional_agreement() -> pd.DataFrame:
    """Figure 2 Panel B: cross-encoder agreement AFTER conditioning on the
    dominant field (organism), both cohorts -- "how much agreement is left
    once you control for the dominant field?" This completes Figure 2's
    if-and-only-if structure (P1), which previously lived only in prose in
    the figure's own subtitle.

    No new computation -- read from `out/41_audit7/dominance_checks.json`'s
    `korea_conditional_agreement` / `mimic_conditional_agreement` blocks
    (EN-012), `organism` stratum only (the stratum the plan's headline
    numbers use; `department` and `organism_x_department` strata are in the
    same source file but not plotted here). MIMIC has only one non-
    degenerate pair (BGE-M3 vs Qwen3); Korea has six pairs total, three of
    which involve BGE-M3 (degenerate, per fig1c_degeneracy) and are kept
    here for completeness but flagged.
    """
    payload = json.loads(DOMINANCE_CHECKS.read_text())
    rows = []
    for cohort_key, cohort_label in [("korea_conditional_agreement", "Korea"),
                                      ("mimic_conditional_agreement", "MIMIC-IV")]:
        for pair, strata in payload[cohort_key].items():
            block = strata.get("organism")
            if block is None:
                continue
            rows.append(dict(
                cohort=cohort_label, pair=pair,
                involves_bge_m3=("bge-m3" in pair),
                ami_within_organism=block["ami_within_median"],
                ami_within_lo=(block.get("ami_within_range") or [None, None])[0],
                ami_within_hi=(block.get("ami_within_range") or [None, None])[1],
                ami_unconditional=block["ami_unconditional_median"],
                n_fits=block["n_fits"], n_strata_median=block["n_strata_median"],
            ))
    return pd.DataFrame(rows)


def fig3_ablation() -> pd.DataFrame | None:
    """Removing the field an encoder is built on destroys its partition.

    Three arms per encoder, in increasing strength of evidence:

      block removed   -- the whole rendered block carrying the field.  This
                         is where the analysis started, and it is the weak
                         arm: `no_therapy` deletes 21.3 lines per document
                         (688,342 corpus-wide / 32,329 docs, AB-009), so a
                         collapse could be a text-volume effect rather than
                         a field effect.
      other field     -- the control.  `no_organism` is a 2.26-line
                         edit that destroys 3-large and leaves Qwen3 intact,
                         which is what makes the pair a cross-over.
      value masked    -- the decisive arm, added after the block arm was
                         found to be confounded with edit size.  Only the
                         drug-name VALUES are replaced with the literal
                         "masked"; durations, efficacy lines, summary tokens
                         and steroid lines all stay, so document length is
                         essentially unchanged.  (An earlier draft quoted
                         "2,599 vs 2,591 bytes" here; that pair appears in no
                         output file and was removed -- see planning/
                         01_FINAL_EVIDENCE_AUDIT.md section 2-D.)
                         If the partition still collapses,
                         it collapsed because of the field, not the bulk.

    Every ARI here is noise-excluded.  The first version of this figure
    mixed conventions -- Qwen3 noise-excluded against 3-large all-row --
    which is not visible to a reader and changes 3-large's numbers by up to
    0.07.
    """
    if not ABLATION.exists():
        return None
    rows = []

    ARM = {"full": "unmodified", "no_organism": "other field removed",
           "no_therapy": "dominant block removed",
           "mask_ab_names_all": "field value masked",
           "mask_ab1_name_only": "field value masked"}

    # Qwen3: 11 seeds if task 47 has run, else the original 4-seed sources.
    # The asymmetry with 3-large's 11 (21) seeds was the most obvious question
    # a reviewer could ask of this figure, and the four-seed no_therapy arm was
    # also the sole support for the field-takeover reading; both are closed by
    # task 47, which refit the original four seeds as a reproduction control
    # (20 fits, 0 mismatch) before adding seven more.
    PLOTTED_QWEN3 = ["full", "no_organism", "no_therapy", "mask_ab1_name_only"]
    if QWEN3_SEEDS.exists():
        payload = json.loads(QWEN3_SEEDS.read_text())
        for key, block in payload["per_fit"].items():
            variant, seed = key.split("|")
            if variant not in PLOTTED_QWEN3:
                continue  # mask_ab_names_all is the stricter mask arm, kept out of the panel
            rows.append(dict(
                encoder="Qwen3-0.6B", dominant_field="first antibiotic",
                variant=variant, seed=int(seed), k=block["k"],
                ari_vs_reference=block["ari_vs_stored"], arm=ARM[variant]))
    else:
        for source, path in [("block", ABLATION), ("mask", MASK_ABLATION)]:
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            for key, block in payload["per_fit"].items():
                variant, seed = key.split("|")
                if source == "mask" and variant != "mask_ab1_name_only":
                    continue  # the stricter of the two mask arms
                rows.append(dict(
                    encoder="Qwen3-0.6B", dominant_field="first antibiotic",
                    variant=variant, seed=int(seed), k=block["k"],
                    ari_vs_reference=block.get("ari_vs_stored",
                                               block.get("ari_noise_excl")),
                    arm=ARM[variant]))

    # 3-small: its dominant field is the treatment department, so no_department
    # is the destructive arm and no_organism the encoder-specific non-dominant-field
    # control (a crossed negative-control edit, NOT size-matched) -- the mirror of
    # 3-large. Arm labels use the same vocabulary as the other two encoders so the
    # facet reads on one axis.
    if SMALL_ABLATION.exists():
        payload = json.loads(SMALL_ABLATION.read_text())
        ARM_3S = {"no_department": "dominant field removed",
                  "no_organism": "other field removed"}
        for key, block in payload["per_fit"].items():
            variant, seed = key.split("|")
            rows.append(dict(
                encoder="OpenAI 3-small", dominant_field="treatment department",
                variant=variant, seed=int(seed), k=block["k"],
                ari_vs_reference=block["ari_vs_reference"], arm=ARM_3S[variant]))

    # 3-large, four seeds per variant.  The first version of this figure
    # used the single seed-42 fit stored with each variant, which turned out
    # to be the lucky seed: the unmodified arm ranges 0.576-0.981 across
    # seeds because 3-large falls into the k=10 attractor on two of four.
    # Reporting 0.981 alone would have understated the control arms' spread
    # while leaving the destructive arm (flat 0.0081) unchanged.
    ARM_3L = {"full": "unmodified", "no_department": "other field removed",
              "no_therapy": "dominant block removed",
              "no_organism": "dominant field removed"}

    # Eleven seeds, not four.  A ten-seed sweep of these same four variant
    # embeddings already existed in out/25_audit4/seeds/ and was never looked
    # for; the four-seed rerun re-derived three of them and added seed 42.
    # Pooling all eleven matters because the CONTROL arms are seed-fragile:
    # on four seeds `no_department` looked like 0.950, on eleven it is 0.742.
    # The destructive arm barely moves (0.008, one escape at 0.095).
    seen = set()
    if THREE_LARGE_SEEDS.exists():
        payload = json.loads(THREE_LARGE_SEEDS.read_text())
        for key, block in payload["per_fit"].items():
            variant, seed = key.split("|")
            seen.add((variant, int(seed)))
            rows.append(dict(
                encoder="OpenAI 3-large", dominant_field="organism",
                variant=variant, seed=int(seed), k=block["k"],
                ari_vs_reference=block["ari_noise_excl"], arm=ARM_3L[variant]))
    for variant in ARM_3L:
        for path in sorted(AUDIT4_SEEDS.glob(f"{variant}_seed*.json")):
            block = json.loads(path.read_text())
            seed = int(block["seed"])
            if (variant, seed) in seen:
                continue  # already carried from the 4-seed rerun
            seen.add((variant, seed))
            rows.append(dict(
                encoder="OpenAI 3-large", dominant_field="organism",
                variant=variant, seed=seed, k=block["k"],
                ari_vs_reference=block["agreement_vs_published"]["ari_noise_excl"],
                arm=ARM_3L[variant]))

    # Ten MORE seeds for no_organism/3-large specifically (11 -> 21 total).
    # 05_FIGURE_CLAIM_MAP.md section "Figure 3" panel A asks for the
    # per-arm seed count to be printed in-panel, and 21 (not 11) is the
    # correct count for this one arm -- out/46_audit10/
    # no_organism_extra_seeds.json (audit 10) already exists and was never
    # wired into this assembly. Seed 6 is stored in both this file and
    # AUDIT4_SEEDS above; the `seen` set (now updated in that loop too)
    # de-duplicates it rather than double-counting.
    if NO_ORGANISM_EXTRA_SEEDS.exists():
        for block in json.loads(NO_ORGANISM_EXTRA_SEEDS.read_text()):
            seed = int(block["seed"])
            if ("no_organism", seed) in seen:
                continue
            seen.add(("no_organism", seed))
            rows.append(dict(
                encoder="OpenAI 3-large", dominant_field="organism",
                variant="no_organism", seed=seed, k=block["k"],
                ari_vs_reference=block["ari_noise_excl"], arm=ARM_3L["no_organism"]))

    frame = pd.DataFrame(rows)
    order = ["unmodified", "other field removed", "dominant block removed",
             "dominant field removed", "field value masked"]
    frame["arm"] = pd.Categorical(frame["arm"], categories=order, ordered=True)
    return frame


def fig3b_knn_readouts() -> pd.DataFrame:
    """Figure 3 Panel B: two clustering-free nearest-neighbour readouts
    of the same ablation arms as panel A -- promoted from the former
    supplementary S9 panel 2 (05_FIGURE_CLAIM_MAP.md section "Figure 3").

    Sub-panel 1 (3-large): 25-NN label transfer macro-F1 (out/25_audit4/
    a_variant_geometry.py K_KNN=25, StratifiedGroupKFold(5, patient_ID,
    seed 42), noise excluded) against the
    originally derived cluster labels, from
    `out/26_controls/a_variant_geometry_all20.json`'s
    `<variant>.label_transfer_macro_f1` (four of its twenty variants: full,
    no_department, no_therapy, no_organism).

    Sub-panel 2 (Qwen3): 15-NN neighbourhood purity (K_GRAPH=15, raw graph,
    no cross-validation) against the MEASURED
    first-antibiotic field value (not a cluster label), from
    `out/45_audit9/knn_purity.json`'s
    `arms.<arm>.purity_first_antibiotic`, alongside the chance match rate
    for that field.

    The two sub-panels use DIFFERENT ground truths (cluster label vs.
    measured field value) and are not on a shared scale -- do not compare
    absolute values across them (05 section "Figure 3", legend requirement;
    also ST4/S9's footnote in `51_package_items.py`).

    No new computation.
    """
    geometry = json.loads(VARIANT_GEOMETRY_ALL20.read_text())
    variants_3large = [
        ("full", "unmodified"), ("no_department", "other field removed"),
        ("no_therapy", "dominant block removed"), ("no_organism", "dominant field removed"),
    ]
    rows = []
    for variant, arm in variants_3large:
        rows.append(dict(
            subpanel="3-large label transfer (cluster labels)",
            variant=variant, arm=arm,
            value=geometry[variant]["label_transfer_macro_f1"],
            metric="k=25 label-transfer macro F1 (patient-grouped 5-fold CV) vs originally derived cluster label",
        ))

    knn = json.loads(KNN_PURITY.read_text())
    arm_map = {
        "stored_full": ("full", "unmodified"),
        "no_organism": ("no_organism", "other field removed"),
        "no_therapy": ("no_therapy", "dominant block removed"),
        "mask_ab1_name_only": ("mask_ab1_name_only", "field value masked (ab1 only)"),
        "mask_ab_names_all": ("mask_ab_names_all", "field value masked (all)"),
    }
    for key, (variant, arm) in arm_map.items():
        rows.append(dict(
            subpanel="Qwen3 first-antibiotic neighbourhood purity (measured field)",
            variant=variant, arm=arm,
            value=knn["arms"][key]["purity_first_antibiotic"],
            metric="k=15 neighbourhood purity vs measured first-antibiotic value",
        ))
    rows.append(dict(
        subpanel="Qwen3 first-antibiotic neighbourhood purity (measured field)",
        variant="chance", arm="chance level",
        value=knn["chance_match_rate"]["first_antibiotic"],
        metric="chance match rate (majority-class baseline)",
    ))
    return pd.DataFrame(rows)


def sf1_variant_ranks() -> pd.DataFrame:
    """SF1: the twenty narrative-template variants ranked by clustering
    agreement, with the unmodified `full` arm's own across-seed range
    carried alongside so the figure can draw it as a background band.

    No new computation. Columns are selected/renamed from
    `out/26_controls/part2_multiseed_table.csv` (task 26, part 2 -- ten
    clustering seeds per variant), and `ari_all_row_*` is dropped in favour
    of `ari_noise_excl_*` only (audit 9 finding F1: mixing the two
    conventions moved a figure's values by up to 0.07). The one derived
    column, `inside_full_band`, is a same-row-set boolean comparison of
    each variant's own [min, max] against the `full` row's own [min, max]
    -- not a new statistic, just a membership check on values already in
    this file.

    This is the data behind i8's prose point (`35_explainer_items.py`):
    variant RANKINGS are not effect sizes, because most variants' entire
    seed range sits inside the unmodified arm's own seed-to-seed spread.
    """
    df = pd.read_csv(ROOT / "out/26_controls/part2_multiseed_table.csv")
    full = df.loc[df["variant"] == "full"].iloc[0]
    band_lo, band_hi = float(full["ari_noise_excl_min"]), float(full["ari_noise_excl_max"])

    keep = ["variant", "n_seeds", "k_median", "k_min", "k_max",
            "ari_noise_excl_median", "ari_noise_excl_min", "ari_noise_excl_max"]
    frame = df[keep].copy()
    frame["full_band_lo"] = band_lo
    frame["full_band_hi"] = band_hi
    frame["inside_full_band"] = (
        (frame["ari_noise_excl_min"] >= band_lo) & (frame["ari_noise_excl_max"] <= band_hi)
    )

    def role(row: pd.Series) -> str:
        if row["variant"] == "full":
            return "unmodified (reference)"
        if row["variant"] in ("fixed_permutation", "shuffle_order_seed2"):
            return "order control"
        return ("template variant -- inside reference band" if row["inside_full_band"]
                else "template variant -- outside reference band")

    frame["role"] = frame.apply(role, axis=1)
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    f1 = fig1_agreement()
    f1.to_csv(OUT / "agreement.csv", index=False)
    print("[fig1] agreement\n", f1.to_string(index=False), "\n", flush=True)

    f1c, f1c_excl = fig1c_degeneracy()
    f1c.to_csv(OUT / "encoder_degeneracy.csv", index=False)
    f1c_excl.to_csv(OUT / "encoder_degeneracy_excluded_pair.csv", index=False)
    print("[fig1c] degeneracy\n", f1c.to_string(index=False), "\n",
          f1c_excl.to_string(index=False), "\n", flush=True)

    f2 = fig2_token_lookup()
    f2.to_csv(OUT / "token_lookup.csv", index=False)
    print("[fig2] token lookup:", len(f2), "clusters", flush=True)

    f1b = fig1b_cluster_sizes()
    f1b.to_csv(OUT / "cluster_sizes.csv", index=False)
    print("[fig1b] cluster sizes\n", f1b.to_string(index=False), "\n", flush=True)

    f3 = fig3_retrieval()
    f3.to_csv(OUT / "retrieval.csv", index=False)
    print("[fig3] retrieval:", len(f3), "rows", flush=True)

    f4a = fig4a_retrieval_ladder()
    f4a.to_csv(OUT / "retrieval_ladder.csv", index=False)
    print("[fig4a] retrieval ladder\n", f4a.to_string(index=False), "\n", flush=True)

    f3 = fig3_ablation()
    if f3 is None:
        print("[fig3] ablation not on disk yet -- skipped")
    else:
        f3.to_csv(OUT / "ablation.csv", index=False)
        summary = (f3.groupby(["encoder", "arm"], observed=True)
                     .agg(ari=("ari_vs_reference", "median"), k=("k", "median"),
                          n_seeds=("seed", "nunique"))
                     .reset_index())
        print("[fig3] ablation\n", summary.to_string(index=False), "\n", flush=True)

    f3b = fig3b_knn_readouts()
    f3b.to_csv(OUT / "knn_readouts.csv", index=False)
    print("[fig3b] knn readouts\n", f3b.to_string(index=False), "\n", flush=True)

    f5 = fig5_field_dominance()
    if f5 is None:
        print("[fig2 field dominance] not on disk yet -- skipped")
    else:
        f5.to_csv(OUT / "field_dominance.csv", index=False)
        wide = f5.pivot_table(index=["cohort", "encoder"], columns="field", values="ami")
        top = f5.loc[f5.groupby(["cohort", "encoder"])["ami"].idxmax()]
        print("[fig2 field dominance] dominant field per partition\n",
              top[["cohort", "encoder", "field", "ami", "margin", "modal_share"]]
              .to_string(index=False), "\n", flush=True)

    f2b = fig2b_conditional_agreement()
    f2b.to_csv(OUT / "conditional_agreement.csv", index=False)
    print("[fig2b] conditional agreement\n", f2b.to_string(index=False), "\n", flush=True)

    f4 = fig4_size_stability()
    if f4 is None:
        print("[fig6 size-stability] not on disk yet (job 588116) -- skipped")
    else:
        f4.to_csv(OUT / "size_stability.csv", index=False)
        print("[fig6 size-stability]\n", f4.to_string(index=False), flush=True)

    fsf1 = sf1_variant_ranks()
    fsf1.to_csv(OUT / "variants.csv", index=False)
    n_inside = int(fsf1["inside_full_band"].sum())
    print(f"[sf1] variants: {len(fsf1)} rows, {n_inside} fall inside the unmodified "
          "arm's own seed range\n", fsf1.to_string(index=False), "\n", flush=True)

    print(f"\n[figure_data] wrote {OUT}")


if __name__ == "__main__":
    main()
