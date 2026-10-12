#!/bin/bash
#SBATCH --job-name=bsi_85_v8figs
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/85_v8_figures_main_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/85_v8_figures_main_%j.err
set -euo pipefail
# Re-renders Figure 1, 2, 4 and Supplementary Figure 1, 2, 3 (PNG 300 dpi + SVG each) from
# the already-written out/33_figures/*.csv -- no new computation, no embedding/clustering/API
# job, no numeric value changes. STAGE is out/followup_analyses/figures_v8/.
# Interpreter/resources: conda env rnaseq_r; the script is single-threaded (ggplot2/ragg/
# svglite do not read SLURM_CPUS_PER_TASK), the second core is headroom only. No PNG->image
# conversion tools (absent on compute nodes).
# Fonts: compute nodes have no Arial. FONTCONFIG_FILE adds the bundled Liberation Sans / Mono
# (metric-compatible with Arial / Courier New) so "Arial" resolves to LiberationSans-Regular.ttf;
# house_style.R's hs_save_vector() refuses to draw otherwise and rewrites the SVG font-family
# to Arial, "Liberation Sans", sans-serif.
# Release note: the original wrapper also renumbered the staged files into
# Figure_N/Supplementary_Figure_N and copied them into an internal manuscript-production
# package directory after this point. That renumbering/copy step has been removed from this
# public wrapper (manuscript packaging, not analysis); only the Rscript call and the
# staged-artefact existence check are kept. The current manuscript1009 figure/table numbering
# for these base filenames is given in README.md.
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export FONTCONFIG_FILE=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/fonts/fonts.conf
STAGE="/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/figures_v8"
echo "[wrapper] host=$(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK:-unset}"
mkdir -p "${STAGE}"
rm -f "${STAGE}"/fig1_agreement.* "${STAGE}"/fig2_field_dominance.* "${STAGE}"/fig5_retrieval.* \
      "${STAGE}"/figS1_variant_ranks.* "${STAGE}"/fig4_token_lookup.* "${STAGE}"/fig6_size_stability.* \
      2>/dev/null || true
/data/tcmhwd/miniconda3/envs/rnaseq_r/bin/Rscript scripts/85_v8_figures_main.R
# staged-artefact existence check
SRC_FIGS=(fig1_agreement fig2_field_dominance fig5_retrieval figS1_variant_ranks fig4_token_lookup fig6_size_stability)
for src in "${SRC_FIGS[@]}"; do
  for ext in png svg; do
    test -s "${STAGE}/${src}.${ext}" || { echo "[wrapper] MISSING or empty: ${STAGE}/${src}.${ext}"; exit 3; }
  done
done
echo "[wrapper] staged artefacts: $(ls "${STAGE}"/fig1_agreement.* "${STAGE}"/fig2_field_dominance.* "${STAGE}"/fig5_retrieval.* "${STAGE}"/figS1_variant_ranks.* "${STAGE}"/fig4_token_lookup.* "${STAGE}"/fig6_size_stability.* 2>/dev/null | wc -l) image files"
echo "[wrapper] done"
