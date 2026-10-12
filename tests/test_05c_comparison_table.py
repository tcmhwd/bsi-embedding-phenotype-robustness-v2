import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from importlib import import_module

mod = import_module("05c_comparison_table")
comparators = import_module("05_baseline_comparators")
stability_mod = import_module("03_stability")

SCRIPT_PATH = Path(SCRIPTS_DIR) / "05c_comparison_table.py"

# Global Constraints (precisely scoped per the brief): these four must never
# appear anywhere in this script -- not even inside evaluate_prognostic_separation,
# which only needs 90d_mortality.
OTHER_BANNED_OUTCOME_COLS = {
    "30d_mortality", "in_hospital_mortality", "death_date", "discharge_date",
}


def _non_docstring_string_literals(source: str) -> list:
    """All string-literal constants in `source` EXCLUDING module/class/
    function docstrings -- i.e. strings that are actually used as values in
    code (usecols=[...], comparisons, dict keys, ...), not prose that merely
    talks about a column name. Used to check the Outcome-blind constraint
    precisely: mentioning a banned column's name in an explanatory docstring
    is fine, using it as a real value anywhere is not.
    """
    tree = ast.parse(source)
    docstring_ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                docstring_ids.add(id(node.body[0].value))
    return [
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstring_ids
    ]


def test_other_banned_outcome_cols_never_used_as_code_literals():
    """Outcome-blind constraint, precisely scoped: 30d_mortality,
    in_hospital_mortality, death_date, discharge_date must never be used as
    an actual value anywhere in 05c_comparison_table.py's code (docstring
    prose describing the constraint itself is fine and expected -- the
    module docstring's "Outcome-blind constraint" section names them deliberately).
    """
    literals = _non_docstring_string_literals(SCRIPT_PATH.read_text())
    for banned in OTHER_BANNED_OUTCOME_COLS:
        assert banned not in literals, f"banned outcome column {banned!r} used as a code literal in 05c_comparison_table.py"


def test_mortality_col_name_only_used_as_a_value_inside_evaluate_prognostic_separation():
    """`MORTALITY_COL`'s value ("90d_mortality") -- and the `MORTALITY_COL`
    identifier itself -- must only be referenced (as an actual value/name
    use, not docstring prose) inside `evaluate_prognostic_separation` or at
    its own module-level assignment. Other dict keys such as
    "auroc_90d_mortality_mean" legitimately contain the substring
    "90d_mortality" for self-documentation and are not flagged (this checks
    exact-value/exact-name usage, not substring occurrence).
    """
    source = SCRIPT_PATH.read_text()
    tree = ast.parse(source)

    func_node = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "evaluate_prognostic_separation"
    )
    func_lines = set(range(func_node.lineno, func_node.end_lineno + 1))

    const_assign = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MORTALITY_COL" for t in n.targets
        )
    )
    const_lines = set(range(const_assign.lineno, const_assign.end_lineno + 1))
    allowed_lines = func_lines | const_lines

    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "MORTALITY_COL" and node.lineno not in allowed_lines:
            offenders.append(node.lineno)
        if (isinstance(node, ast.Constant) and node.value == "90d_mortality"
                and node.lineno not in allowed_lines):
            offenders.append(node.lineno)
    assert not offenders, f"MORTALITY_COL / '90d_mortality' used outside evaluate_prognostic_separation at lines {offenders}"


# ---------------------------------------------------------------------------
# Axis 4: prognostic separation (AUROC)
# ---------------------------------------------------------------------------

def _separable_mortality_csv(tmp_path, n_per_cluster=100, seed=0):
    """Two clusters with obviously different mortality rates (5% vs 95%) --
    a cluster-only classifier should achieve a high AUROC.
    """
    rng = np.random.default_rng(seed)
    admission_ids = [f"A{i:04d}" for i in range(2 * n_per_cluster)]
    labels = np.repeat([0, 1], n_per_cluster)
    mortality = np.concatenate([
        rng.binomial(1, 0.05, size=n_per_cluster),
        rng.binomial(1, 0.95, size=n_per_cluster),
    ])
    df = pd.DataFrame({
        "admission_ID": admission_ids,
        "90d_mortality": mortality,
        # a couple of unrelated columns, to confirm usecols restricts the read
        "sex": rng.choice(["M", "F"], size=2 * n_per_cluster),
    })
    path = tmp_path / "synthetic_mortality.csv"
    df.to_csv(path, index=False)
    return path, admission_ids, labels


def test_evaluate_prognostic_separation_high_auroc_on_separable_data(tmp_path):
    dataset_path, admission_ids, labels = _separable_mortality_csv(tmp_path)
    mean_auc, std_auc = mod.evaluate_prognostic_separation(
        labels, admission_ids, dataset_path=dataset_path, seed=42,
    )
    assert mean_auc > 0.9, f"expected near-perfect separation, got mean AUROC={mean_auc}"
    assert std_auc >= 0.0


def test_evaluate_prognostic_separation_low_auroc_when_uninformative(tmp_path):
    rng = np.random.default_rng(1)
    n = 200
    admission_ids = [f"B{i:04d}" for i in range(n)]
    labels = rng.integers(0, 3, size=n)  # cluster assignment uncorrelated with outcome
    mortality = rng.binomial(1, 0.3, size=n)
    df = pd.DataFrame({"admission_ID": admission_ids, "90d_mortality": mortality})
    path = tmp_path / "uninformative_mortality.csv"
    df.to_csv(path, index=False)

    mean_auc, _ = mod.evaluate_prognostic_separation(labels, admission_ids, dataset_path=path, seed=42)
    assert 0.3 < mean_auc < 0.7, f"expected near-chance AUROC for uninformative clusters, got {mean_auc}"


# ---------------------------------------------------------------------------
# Axis 5: top-3 characteristic features
# ---------------------------------------------------------------------------

def _known_ground_truth_matrix(n_per_cluster=80, seed=0):
    """2 clusters. `distinguishing_numeric` and `distinguishing_cat` are
    strongly different by cluster; `noise_numeric`/`noise_cat` are drawn
    identically regardless of cluster (should not outrank the true signals).
    """
    rng = np.random.default_rng(seed)
    n = 2 * n_per_cluster
    labels = np.repeat([0, 1], n_per_cluster)
    df = pd.DataFrame({
        "distinguishing_numeric": np.concatenate([
            rng.normal(0.0, 1.0, n_per_cluster), rng.normal(50.0, 1.0, n_per_cluster),
        ]),
        "noise_numeric": rng.normal(10.0, 1.0, n),
        "distinguishing_cat": np.concatenate([
            np.full(n_per_cluster, "A"), np.full(n_per_cluster, "B"),
        ]),
        "noise_cat": rng.choice(["X", "Y", "Z"], size=n),
    })
    df["distinguishing_cat"] = df["distinguishing_cat"].astype("category")
    df["noise_cat"] = df["noise_cat"].astype("category")
    categorical_cols = ["distinguishing_cat", "noise_cat"]
    return df, categorical_cols, labels


def test_characteristic_features_picks_known_distinguishing_variables():
    matrix, categorical_cols, labels = _known_ground_truth_matrix()
    top3 = mod.characteristic_features_per_cluster(matrix, categorical_cols, labels, top_n=3)

    assert set(top3["cluster"]) == {0, 1}
    for cl in (0, 1):
        cluster_vars = set(top3.loc[top3["cluster"] == cl, "variable"])
        assert "distinguishing_numeric" in cluster_vars
        assert "distinguishing_cat" in cluster_vars
        # rank 1 should be one of the two true signals, not a noise column
        rank1_var = top3.loc[(top3["cluster"] == cl) & (top3["rank"] == 1), "variable"].iloc[0]
        assert rank1_var in {"distinguishing_numeric", "distinguishing_cat"}

    # the distinguishing categorical variable's recorded level should match
    # the cluster it's most associated with
    row_cl0 = top3[(top3["cluster"] == 0) & (top3["variable"] == "distinguishing_cat")].iloc[0]
    row_cl1 = top3[(top3["cluster"] == 1) & (top3["variable"] == "distinguishing_cat")].iloc[0]
    assert row_cl0["level"] == "A"
    assert row_cl1["level"] == "B"


def test_characteristic_features_output_shape():
    matrix, categorical_cols, labels = _known_ground_truth_matrix()
    top3 = mod.characteristic_features_per_cluster(matrix, categorical_cols, labels, top_n=3)
    assert set(top3.columns) == {"cluster", "rank", "variable", "level", "smd"}
    for cl in (0, 1):
        assert len(top3[top3["cluster"] == cl]) <= 3


# ---------------------------------------------------------------------------
# Bootstrap generalization (03_stability.bootstrap_stability_generic)
# ---------------------------------------------------------------------------

def test_bootstrap_stability_generic_still_works_for_umap_hdbscan_case():
    """Task 3's original UMAP+HDBSCAN use case must still work after
    generalizing bootstrap_stability into bootstrap_stability_generic.
    """
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(600, 16)).astype(np.float32)
    base = np.repeat([0, 1, 2], 200)
    ari, jac = stability_mod.bootstrap_stability(
        emb, base, {"n_components": 0, "n_neighbors": 15, "min_cluster_size": 30},
        n_boot=3, seed=1,
    )
    assert len(ari) == 3
    assert set(jac["cluster"]) == {0, 1, 2}


def test_bootstrap_stability_generic_works_for_a_structured_method():
    """The same generalized bootstrap must work for at least one of the five
    new structured-data methods (fit_kprototypes chosen: fast at this tiny
    synthetic scale, unlike at full scale)."""
    matrix, categorical_cols, _ = _known_ground_truth_matrix()
    base_labels = comparators.fit_kprototypes(matrix, categorical_cols, k=2, seed=42)

    def fit_fn(sub_matrix, seed):
        return comparators.fit_kprototypes(sub_matrix, categorical_cols, k=2, seed=seed)

    ari, jac = stability_mod.bootstrap_stability_generic(
        matrix, base_labels, fit_fn, n_boot=2, seed=1,
    )
    assert len(ari) == 2
    assert set(jac["cluster"]) == set(base_labels.tolist()) - {-1}


def test_bootstrap_ari_mean_wrapper():
    matrix, categorical_cols, _ = _known_ground_truth_matrix()
    base_labels = comparators.fit_famd_kmeans(matrix, categorical_cols, k=2, seed=42)
    mean_ari = mod.bootstrap_ari_mean(
        matrix, categorical_cols, comparators.fit_famd_kmeans, k=2,
        base_labels=base_labels, n_boot=2, seed=1,
    )
    assert isinstance(mean_ari, float)
    assert -1.0 <= mean_ari <= 1.0


def test_bootstrap_ari_mean_returns_nan_for_zero_n_boot():
    matrix, categorical_cols, _ = _known_ground_truth_matrix()
    base_labels = comparators.fit_famd_kmeans(matrix, categorical_cols, k=2, seed=42)
    mean_ari = mod.bootstrap_ari_mean(
        matrix, categorical_cols, comparators.fit_famd_kmeans, k=2,
        base_labels=base_labels, n_boot=0, seed=1,
    )
    assert np.isnan(mean_ari)


# ---------------------------------------------------------------------------
# Gower-matrix-reuse (CRITICAL performance fix): precomputed_distance must
# give identical results to the honest from-scratch computation.
# ---------------------------------------------------------------------------

def test_fit_gower_pam_precomputed_distance_matches_default_path():
    matrix, categorical_cols, _truth = _two_cluster_categorical_fixture()
    default_labels = comparators.fit_gower_pam(matrix, categorical_cols, k=2, seed=42)

    D = comparators.compute_gower_matrix(matrix, categorical_cols)
    reused_labels = comparators.fit_gower_pam(
        matrix, categorical_cols, k=2, seed=42, precomputed_distance=D,
    )
    assert np.array_equal(default_labels, reused_labels), (
        "fit_gower_pam(precomputed_distance=...) must produce identical labels "
        "to the default (recompute-every-time) path on the same data/seed"
    )


def test_compute_gower_matrix_reused_across_multiple_k_values():
    """The core 5c performance fix: compute the distance matrix once, reuse
    it across several k values, each call giving a result matching the
    from-scratch equivalent for that k."""
    matrix, categorical_cols, _truth = _two_cluster_categorical_fixture()
    D = comparators.compute_gower_matrix(matrix, categorical_cols)

    for k in (2, 3):
        reused = comparators.fit_gower_pam(matrix, categorical_cols, k=k, seed=42, precomputed_distance=D)
        fresh = comparators.fit_gower_pam(matrix, categorical_cols, k=k, seed=42)
        assert np.array_equal(reused, fresh)


def _two_cluster_categorical_fixture(seed=0, n_per=60):
    rng = np.random.default_rng(seed)
    rows = []
    for g in range(2):
        group = np.full(n_per, f"g{g}")
        site = rng.choice([f"site{i}" for i in range(10)], size=n_per)
        rows.append(pd.DataFrame({"group": group, "site": site}))
    df = pd.concat(rows, ignore_index=True)
    df["group"] = df["group"].astype("category")
    df["site"] = df["site"].astype("category")
    categorical_cols = ["group", "site"]
    truth = np.repeat([0, 1], n_per)
    return df, categorical_cols, truth


# ---------------------------------------------------------------------------
# ari_against_reference
# ---------------------------------------------------------------------------

def test_ari_against_reference_perfect_agreement():
    admission_ids = [f"A{i}" for i in range(20)]
    labels = np.array([0] * 10 + [1] * 10)
    ref_df = pd.DataFrame({"admission_ID": admission_ids, "ref": labels})
    ari = mod.ari_against_reference(admission_ids, labels, ref_df, "ref")
    assert ari == pytest.approx(1.0)


def test_ari_against_reference_robust_to_reference_row_order():
    admission_ids = [f"A{i}" for i in range(20)]
    labels = np.array([0] * 10 + [1] * 10)
    ref_df = pd.DataFrame({"admission_ID": admission_ids, "ref": labels}).sample(
        frac=1.0, random_state=0,
    )
    ari = mod.ari_against_reference(admission_ids, labels, ref_df, "ref")
    assert ari == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# End-to-end pipeline wiring on small synthetic data (build_comparison_row)
# ---------------------------------------------------------------------------

def test_build_comparison_row_end_to_end_on_synthetic_data(tmp_path):
    matrix, categorical_cols, labels = _known_ground_truth_matrix(n_per_cluster=40)
    admission_ids = [f"A{i:04d}" for i in range(len(matrix))]

    dataset_df = pd.DataFrame({
        "admission_ID": admission_ids,
        "90d_mortality": np.concatenate([
            np.random.default_rng(0).binomial(1, 0.1, 40),
            np.random.default_rng(1).binomial(1, 0.6, 40),
        ]),
    })
    dataset_path = tmp_path / "synthetic_dataset.csv"
    dataset_df.to_csv(dataset_path, index=False)

    embedding_labels_df = pd.DataFrame({"admission_ID": admission_ids, "label": labels})
    published_labels_df = pd.DataFrame({
        "admission_ID": admission_ids,
        "phenotype_cluster": labels,
    })
    imputed = comparators.impute_structured_matrix(matrix, categorical_cols)
    out_dir = tmp_path / "out"

    row = mod.build_comparison_row(
        "famd_kmeans", comparators.fit_famd_kmeans, matrix, categorical_cols, k=2,
        admission_ids=admission_ids, embedding_labels_df=embedding_labels_df,
        published_labels_df=published_labels_df, imputed_matrix=imputed, out_dir=out_dir,
        dataset_path=dataset_path, seed=42, n_boot=2,
    )

    expected_keys = {
        "method", "k", "n_clusters_actual", "bootstrap_ari_mean", "n_boot_used",
        "ari_vs_embedding", "ari_vs_published_paper", "auroc_90d_mortality_mean",
        "auroc_90d_mortality_std", "top_features_path", "fit_wall_time_sec",
        "bootstrap_wall_time_sec",
    }
    assert set(row) == expected_keys
    assert row["method"] == "famd_kmeans"
    assert row["k"] == 2
    assert row["n_clusters_actual"] >= 1
    assert not np.isnan(row["bootstrap_ari_mean"])  # n_boot=2 > 0
    assert (out_dir / row["top_features_path"]).exists()


def test_build_diagnosis_markdown_does_not_declare_a_winner_and_includes_caveat():
    rows_df = pd.DataFrame([
        {"method": "kprototypes", "k": 16, "n_clusters_actual": 16,
         "bootstrap_ari_mean": 0.5, "ari_vs_embedding": 0.2, "ari_vs_published_paper": 0.3,
         "auroc_90d_mortality_mean": 0.6, "auroc_90d_mortality_std": 0.02},
        {"method": "gower_pam", "k": 16, "n_clusters_actual": 16,
         "bootstrap_ari_mean": 0.4, "ari_vs_embedding": 0.25, "ari_vs_published_paper": 0.28,
         "auroc_90d_mortality_mean": 0.58, "auroc_90d_mortality_std": 0.03},
    ])
    text = mod.build_diagnosis_markdown(rows_df, "nc10_nn15_mcs200", [16, 32], 16, mod.N_BOOT)
    assert "superior" in text
    # These would only appear if the document declared an actual winner
    # (as opposed to the disclaimer's negated mention of "superiority").
    for banned_phrase in ("is the best method", "outperforms all", "embedding wins", "kprototypes is better", "gower_pam is better"):
        assert banned_phrase not in text
