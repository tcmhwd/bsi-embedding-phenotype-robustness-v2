#!/bin/bash
#SBATCH --job-name=bsi_86_v8figs
#SBATCH --partition=cpu_partition
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/86_v8_figure3_supp456_%j.out
#SBATCH --error=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/logs/86_v8_figure3_supp456_%j.err
set -euo pipefail
# Draws Figure 3 + Supplementary Figures 4-6 from scripts/86_v8_figure3_supp456.R. Does NOT
# run any .py data-prep step; it reads the already-written tables_v7/*.csv (no new
# computation, no numeric value changes). STAGE is out/followup_analyses/figures_v8/.
# Resources/threading/font handling: conda env rnaseq_r, R is single-threaded
# (ggplot2/ragg/svglite do not read SLURM_CPUS_PER_TASK; the second core is headroom only);
# FONTCONFIG_FILE adds bundled Liberation Sans/Mono so "Arial" resolves correctly on compute
# nodes (no Arial there); no PNG->image conversion tools on compute nodes.
# Release note: the original wrapper also copied the staged figures into an internal
# manuscript-production package directory after this point. That copy step has been removed
# from this public wrapper (manuscript packaging, not analysis); only the Rscript call and the
# staged-artefact existence check are kept.
cd "/data/tcmhwd/AI/AST_ai/phenotype_revision"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export FONTCONFIG_FILE=/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/fonts/fonts.conf
STAGE="/data/tcmhwd/AI/AST_ai/phenotype_revision/out/followup_analyses/figures_v8"
echo "[wrapper] host=$(hostname) job=${SLURM_JOB_ID} cpus=${SLURM_CPUS_PER_TASK:-unset}"
mkdir -p "${STAGE}"
rm -f "${STAGE}"/Figure_3.* "${STAGE}"/Supplementary_Figure_4.* "${STAGE}"/Supplementary_Figure_5.* \
      "${STAGE}"/Supplementary_Figure_6.* 2>/dev/null || true
/data/tcmhwd/miniconda3/envs/rnaseq_r/bin/Rscript scripts/86_v8_figure3_supp456.R
# staged-artefact existence check
for f in Figure_3 Supplementary_Figure_4 Supplementary_Figure_5 Supplementary_Figure_6; do
  for ext in png svg; do
    test -s "${STAGE}/${f}.${ext}" || { echo "[wrapper] MISSING or empty: ${STAGE}/${f}.${ext}"; exit 3; }
  done
done
echo "[wrapper] staged artefacts: $(ls "${STAGE}"/Figure_3.* "${STAGE}"/Supplementary_Figure_4.* "${STAGE}"/Supplementary_Figure_5.* "${STAGE}"/Supplementary_Figure_6.* 2>/dev/null | wc -l) image files"
echo "[wrapper] done"
