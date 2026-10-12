#!/usr/bin/env python3
"""Task 4 -- the open-embedding concordance gate that was never run
(AUDIT C2; design doc `.superpowers/sdd/DESIGN_embedding_and_stability.md`
section 1.6, which this script implements verbatim).

Answers: does an open, long-context embedding model recover the SAME
phenotype structure as the proprietary `text-embedding-3-large` embedding
that produced the manuscript's published clusters -- on the exact same
32,329 UNMODIFIED narratives, same rows, same order as `load_frame()`?

This gate must produce a verdict (ADOPT / CONDITIONAL / FAIL) BEFORE any
ablation/template variant is embedded with the open model, and G0
(truncation) is a hard precondition: if it fails, no other metric is
computed (design doc section 1.6, "Fail -> model rejected, no other metric
is computed or reported").

Model candidates, in priority order (design doc section 1.2(B)):
  1. BAAI/bge-m3       -- 8,192-token window, unblocked via a local
                          .bin -> .safetensors conversion (see
                          `convert_bge_m3_safetensors.py`); loaded from the
                          project-local `model_cache/bge-m3/` directory, NOT
                          the shared HF hub cache.
  2. Qwen/Qwen3-Embedding-0.6B -- 32k window, safetensors on the hub
                          natively, zero-friction fallback.

Never imports numpy/pandas/torch/sentence-transformers/umap/hdbscan/sklearn
at module top level -- only inside stage functions, after `main()` has set
that stage's thread env vars (same discipline as `06_domain_ablation.py` /
`07_template_sensitivity.py`, for the same reason: embedding is thread-
count-independent so wants many threads, UMAP+HDBSCAN clustering is
thread-count-DEPENDENT even with a fixed seed so must be pinned to 1).

Outcome-blind (Global Constraints): `30d_mortality`, `90d_mortality`,
`in_hospital_mortality`, `death_date`, `discharge_date` are never read here.
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path("/data/tcmhwd/AI/AST_ai")
REVISION_ROOT = ROOT / "phenotype_revision"
OUT = REVISION_ROOT / "out" / "04_open_embed"
REFERENCE_LABELS_PATH = REVISION_ROOT / "out" / "02_recluster" / "labels_nc20_nn30_mcs100.parquet"
ORGANISM_CHANCE_LEVEL = 0.256  # E. coli share, AUDIT C2/M7 -- recomputed live below, quoted here as the audit's reference value

# The reference clustering config: IDENTICAL to Tasks 6/7's fixed config
# (nc20_nn30_mcs100 -- n_components=20, n_neighbors=30, min_cluster_size=100,
# defaults min_dist=0.0/metric=cosine/min_samples=10), which is exactly the
# config that produced `out/02_recluster/labels_nc20_nn30_mcs100.parquet`
# (the OpenAI-embedding reference this gate compares against).
CLUSTER_CONFIG = dict(n_components=20, n_neighbors=30, min_cluster_size=100)

MODEL_CONFIGS = {
    "bge-m3": {
        "load_path": str(REVISION_ROOT / "model_cache" / "bge-m3"),
        "max_seq_length": 8192,
        "dim": 1024,
        "pooling": "cls",
        "prefix": "",
        "tokenizer_name": "BAAI/bge-m3 (local safetensors conversion)",
        "model_revision": None,  # filled from model_cache/bge-m3/CONVERSION_PROVENANCE.json
    },
    "qwen3-0.6b": {
        "load_path": "Qwen/Qwen3-Embedding-0.6B",
        "max_seq_length": 32768,
        "dim": 1024,
        "pooling": "last_token",
        "prefix": "",
        "tokenizer_name": "Qwen/Qwen3-Embedding-0.6B",
        "model_revision": None,  # filled from the loaded model's resolved snapshot
    },
}

DEFAULT_BATCH_SIZE = {"cpu": 16, "cuda": 128}


def _resolve_bge_m3_revision() -> str:
    prov_path = Path(MODEL_CONFIGS["bge-m3"]["load_path"]) / "CONVERSION_PROVENANCE.json"
    if prov_path.exists():
        return json.loads(prov_path.read_text())["source_revision"]
    return "unknown (CONVERSION_PROVENANCE.json missing)"


def _atomic_write_json(obj: dict, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, dest)


def _atomic_write_text(text: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".tmp_{os.getpid()}_{dest.name}")
    tmp.write_text(text)
    os.replace(tmp, dest)


def _model_dir(model_slug: str) -> Path:
    return OUT / model_slug


def require_passing_gate(model_slug: str) -> dict:
    """Thin wrapper over `embed_common.require_passing_gate` (the shared
    implementation every embed stage -- this script and 06/07's
    `stage_embed` -- imports), kept here too so callers of THIS module
    don't need to know the guard actually lives in `embed_common`. Uses
    `_model_dir`/`OUT` (this module's own path constants) so a test that
    monkeypatches `04_open_embedding.OUT` still works; production code
    should prefer `embed_common.require_passing_gate` directly."""
    gate_path = _model_dir(model_slug) / "GATE.md"
    if not gate_path.exists():
        raise RuntimeError(
            f"no Task 4 gate verdict found for model={model_slug!r} "
            f"(expected {gate_path}) -- run `--stage gate` first. "
            "Per the design doc, no ablation/template embedding may proceed "
            "on an open model without a passing gate (AUDIT C2)."
        )
    concordance_path = _model_dir(model_slug) / "concordance.json"
    verdict = json.loads(concordance_path.read_text())["verdict"]
    if verdict not in ("ADOPT", "CONDITIONAL"):
        raise RuntimeError(
            f"Task 4 gate verdict for model={model_slug!r} is {verdict!r} -- "
            "must be ADOPT or CONDITIONAL to proceed. A FAIL verdict means "
            "this model must not be used to embed ablation/template variants "
            "(design doc section 1.6)."
        )
    return {"verdict": verdict, "gate_path": str(gate_path)}


# ---------------------------------------------------------------------------
# Stage: embed -- HIGH thread count on CPU (embedding is thread-count-
# independent for e5; not yet re-verified for this model here, see
# floors_reembed). Deferred heavy imports.
# ---------------------------------------------------------------------------

def _load_admission_and_text():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib
    frame_mod = importlib.import_module("01_build_analysis_frame")
    frame = frame_mod.load_frame(verify_reports=True)
    frame = frame.sort_values("admission_ID").reset_index(drop=True)
    return frame


def stage_embed(model_slug: str, device: str = "cpu", batch_size: int = None,
                 limit: int = None, subsample_seed: int = None) -> dict:
    import time

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import embed_common

    cfg = MODEL_CONFIGS[model_slug]
    if model_slug == "bge-m3":
        cfg["model_revision"] = _resolve_bge_m3_revision()

    frame = _load_admission_and_text()
    if subsample_seed is not None and limit is not None:
        frame = frame.sample(n=limit, random_state=subsample_seed).sort_values("admission_ID").reset_index(drop=True)
    elif limit is not None:
        frame = frame.head(limit)

    texts = [open(p, encoding="utf-8", errors="replace").read() for p in frame["report_file"]]
    admission_ids = frame["admission_ID"].tolist()

    # --- G0: truncation guard, BEFORE encoding (hard precondition). ---
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["load_path"])
    count_fn = embed_common.hf_token_counter(tokenizer)
    report = embed_common.token_report(
        texts, count_fn, cfg["max_seq_length"], cfg["tokenizer_name"],
    )
    embed_common.assert_no_truncation(report)  # raises loudly if any doc would be truncated

    if batch_size is None:
        batch_size = DEFAULT_BATCH_SIZE[device]

    # Same measured device-loading constraint as 06/07: direct device="cuda"
    # segfaults silently (jobs 586270/586271/586273); load on CPU then
    # .to("cuda") (verified job 586275).
    model = SentenceTransformer(cfg["load_path"], device="cpu")
    model.max_seq_length = cfg["max_seq_length"]  # set EXPLICITLY, never rely on default
    if device == "cuda":
        model = model.to("cuda")

    t0 = time.time()
    vecs = model.encode(
        texts, batch_size=batch_size, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=False,
    )
    enc_sec = time.time() - t0

    out_dir = _model_dir(model_slug) / f"embeddings_{device}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "merged" if limit is None else f"subsample{limit}_seed{subsample_seed}"
    vec_path = out_dir / f"{tag}.npy"
    id_path = out_dir / f"{tag}_ids.json"
    tmp_vec = vec_path.with_name(f".tmp_{os.getpid()}_{vec_path.name}")
    np.save(tmp_vec, vecs.astype(np.float32))
    os.replace(tmp_vec, vec_path)
    tmp_id = id_path.with_name(f".tmp_{os.getpid()}_{id_path.name}")
    tmp_id.write_text(json.dumps(admission_ids))
    os.replace(tmp_id, id_path)

    manifest = {
        "model_slug": model_slug, "model_load_path": cfg["load_path"],
        "model_revision": cfg["model_revision"], "pooling": cfg["pooling"],
        "prefix": cfg["prefix"], "normalize": True, "device": device,
        "batch_size": batch_size, "n_docs": len(texts), "dim": int(vecs.shape[1]),
        "enc_sec": round(enc_sec, 2),
        "docs_per_sec": round(len(texts) / enc_sec, 3) if enc_sec > 0 else None,
        "torch_threads": torch.get_num_threads(),
        "env_threads": {
            k: os.environ.get(k) for k in
            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "torch_version": torch.__version__,
    }
    import transformers as _tf
    import sentence_transformers as _st
    manifest["transformers_version"] = _tf.__version__
    manifest["sentence_transformers_version"] = _st.__version__
    manifest = embed_common.merge_report_into_manifest(manifest, report)  # asserts no truncation again, defense-in-depth

    manifest_path = out_dir / f".embed_manifest_{tag}.json"
    _atomic_write_json(manifest, manifest_path)
    print(json.dumps({k: v for k, v in manifest.items() if k != "env_threads"}, indent=2), flush=True)
    return manifest


# ---------------------------------------------------------------------------
# Stage: floors -- F_seed, F_reembed, F_device (design doc section 1.6 Step 0).
# Cluster-stage thread discipline: pinned to 1.
# ---------------------------------------------------------------------------

def _load_merged(model_slug: str, device: str, tag: str = "merged"):
    import json as _json

    import numpy as np

    emb_dir = _model_dir(model_slug) / f"embeddings_{device}"
    vecs = np.load(emb_dir / f"{tag}.npy")
    ids = np.asarray(_json.loads((emb_dir / f"{tag}_ids.json").read_text()))
    return vecs, ids


def stage_floors_seed(model_slug: str, device: str = "cpu", seeds=range(1, 21)) -> dict:
    import itertools

    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib
    recl = importlib.import_module("02_recluster")

    vecs, ids = _load_merged(model_slug, device)
    all_labels = {}
    for seed in seeds:
        r = recl.run_config(vecs, seed=seed, **CLUSTER_CONFIG)
        all_labels[seed] = r["labels"]
        print(f"F_seed: seed={seed} n_clusters={r['n_clusters']} noise_frac={r['noise_frac']:.4f}", flush=True)

    pairs = list(itertools.combinations(seeds, 2))
    aris = [adjusted_rand_score(all_labels[a], all_labels[b]) for a, b in pairs]
    aris = np.asarray(aris)

    result = {
        "model_slug": model_slug, "device": device, "n_seeds": len(list(seeds)),
        "n_pairs": len(pairs), "ari_mean": float(aris.mean()), "ari_p05": float(np.percentile(aris, 5)),
        "ari_min": float(aris.min()), "ari_max": float(aris.max()),
        "cluster_config": CLUSTER_CONFIG,
    }
    _atomic_write_json(result, _model_dir(model_slug) / f".F_seed_{device}.json")
    print(json.dumps(result, indent=2), flush=True)
    return result


def stage_floors_reembed(model_slug: str, device: str = "cpu", limit: int = 2000, subsample_seed: int = 42) -> dict:
    """F_reembed: re-run the embed stage twice on an IDENTICAL subsample
    (same model/device/batch size) and assert bitwise-identical vectors.
    A subsample (not the full 32,329) is used purely for wall-time economy
    -- see `check_embedding_thread_determinism.py`'s identical 100-doc
    precedent for the analogous e5 check; the claim being tested
    ("same code + same weights + same device -> same floats") does not
    depend on corpus size."""
    import numpy as np

    m1 = stage_embed(model_slug, device=device, limit=limit, subsample_seed=subsample_seed)
    v1, ids1 = _load_merged(model_slug, device, tag=f"subsample{limit}_seed{subsample_seed}")

    # Re-run into a distinct tag so the first run's output isn't overwritten
    # before the diff, then compare.
    out_dir = _model_dir(model_slug) / f"embeddings_{device}"
    (out_dir / f"subsample{limit}_seed{subsample_seed}.npy").rename(
        out_dir / f"subsample{limit}_seed{subsample_seed}_run1.npy"
    )
    (out_dir / f"subsample{limit}_seed{subsample_seed}_ids.json").rename(
        out_dir / f"subsample{limit}_seed{subsample_seed}_run1_ids.json"
    )
    v1 = np.load(out_dir / f"subsample{limit}_seed{subsample_seed}_run1.npy")

    stage_embed(model_slug, device=device, limit=limit, subsample_seed=subsample_seed)
    v2, ids2 = _load_merged(model_slug, device, tag=f"subsample{limit}_seed{subsample_seed}")

    assert list(ids1) == list(ids2), "F_reembed: row order differs between runs"
    max_abs_diff = float(np.max(np.abs(v1.astype(np.float64) - v2.astype(np.float64))))
    result = {
        "model_slug": model_slug, "device": device, "n_docs": len(ids2),
        "max_abs_diff": max_abs_diff, "bitwise_identical": bool(max_abs_diff == 0.0),
    }
    _atomic_write_json(result, _model_dir(model_slug) / f".F_reembed_{device}.json")
    print(json.dumps(result, indent=2), flush=True)
    if max_abs_diff != 0.0:
        print(
            "WARNING: F_reembed is NOT exactly 1.0-equivalent (max_abs_diff "
            f"= {max_abs_diff}) -- per the design doc this means the pipeline "
            "is nondeterministic at a level nothing else can be interpreted "
            "through. Reported, not hidden.",
            flush=True,
        )
    return result


def stage_floors_device(model_slug: str) -> dict:
    """F_device: CPU vs GPU embedding of the SAME text, clustered with the
    identical config -- requires embeddings_cpu/merged.npy AND
    embeddings_cuda/merged.npy to already exist (the GPU pass is queued
    behind the author's ColabFold GPU array via --dependency chaining, see
    slurm/04_open_embed_gpu.sh)."""
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib
    recl = importlib.import_module("02_recluster")

    cpu_dir = _model_dir(model_slug) / "embeddings_cpu"
    cuda_dir = _model_dir(model_slug) / "embeddings_cuda"
    if not (cpu_dir / "merged.npy").exists() or not (cuda_dir / "merged.npy").exists():
        result = {
            "model_slug": model_slug, "status": "PENDING",
            "reason": "embeddings_cpu/merged.npy and/or embeddings_cuda/merged.npy "
                      "not yet written -- GPU embed job queued behind the ColabFold "
                      "array, see slurm/04_open_embed_gpu.sh",
        }
        _atomic_write_json(result, _model_dir(model_slug) / ".F_device.json")
        print(json.dumps(result, indent=2), flush=True)
        return result

    v_cpu, ids_cpu = _load_merged(model_slug, "cpu")
    v_cuda, ids_cuda = _load_merged(model_slug, "cuda")
    assert list(ids_cpu) == list(ids_cuda), "F_device: row order differs between cpu and cuda embeddings"

    max_abs_diff = float(np.max(np.abs(v_cpu.astype(np.float64) - v_cuda.astype(np.float64))))
    r_cpu = recl.run_config(v_cpu, seed=42, **CLUSTER_CONFIG)
    r_cuda = recl.run_config(v_cuda, seed=42, **CLUSTER_CONFIG)
    ari = adjusted_rand_score(r_cpu["labels"], r_cuda["labels"])

    result = {
        "model_slug": model_slug, "status": "OK", "max_abs_diff": max_abs_diff,
        "n_clusters_cpu": r_cpu["n_clusters"], "n_clusters_cuda": r_cuda["n_clusters"],
        "noise_frac_cpu": r_cpu["noise_frac"], "noise_frac_cuda": r_cuda["noise_frac"],
        "ari_cpu_vs_cuda": float(ari), "cluster_config": CLUSTER_CONFIG,
    }
    _atomic_write_json(result, _model_dir(model_slug) / ".F_device.json")
    print(json.dumps(result, indent=2), flush=True)
    return result


# ---------------------------------------------------------------------------
# Stage: gate -- G0-G4, verdict. Cluster-stage thread discipline (pinned 1).
# ---------------------------------------------------------------------------

def _linear_cka(X, Y) -> float:
    """Standard linear CKA via the cross-covariance Frobenius-norm identity
    -- works for X (n x d1), Y (n x d2) with d1 != d2."""
    import numpy as np

    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    hsic = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    norm_x = np.linalg.norm(Xc.T @ Xc, ord="fro")
    norm_y = np.linalg.norm(Yc.T @ Yc, ord="fro")
    if norm_x == 0 or norm_y == 0:
        return float("nan")
    return float(hsic / (norm_x * norm_y))


def _mutual_knn_overlap(X, Y, k: int) -> float:
    """Mean over rows of |N_X(i) ∩ N_Y(i)| / k, cosine metric, self excluded."""
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    nn_x = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=1).fit(X)
    nn_y = NearestNeighbors(n_neighbors=k + 1, metric="cosine", algorithm="brute", n_jobs=1).fit(Y)
    _, idx_x = nn_x.kneighbors(X)
    _, idx_y = nn_y.kneighbors(Y)
    idx_x = idx_x[:, 1:]  # drop self (distance 0, always first)
    idx_y = idx_y[:, 1:]

    overlaps = np.empty(X.shape[0], dtype=np.float64)
    for i in range(X.shape[0]):
        overlaps[i] = len(set(idx_x[i].tolist()) & set(idx_y[i].tolist())) / k
    return float(overlaps.mean())


def _organism_purity_ways(labels, organisms, chance_level: float) -> dict:
    import numpy as np

    labels = np.asarray(labels)
    organisms = np.asarray(organisms, dtype=object)
    is_poly = organisms == "Polymicrobial"

    def _purity(mask):
        lab, org = labels[mask], organisms[mask]
        if len(lab) == 0:
            return float("nan")
        total_weighted, total_n = 0.0, 0
        for cl in np.unique(lab):
            sel = lab == cl
            n = int(sel.sum())
            _, counts = np.unique(org[sel], return_counts=True)
            total_weighted += n * (counts.max() / n)
            total_n += n
        return float(total_weighted / total_n)

    all_rows = np.ones(len(labels), dtype=bool)
    not_noise = labels != -1
    not_poly = ~is_poly

    return {
        "incl_noise_incl_poly": _purity(all_rows),
        "excl_noise_incl_poly": _purity(not_noise),
        "incl_noise_excl_poly": _purity(all_rows & not_poly),
        "excl_noise_excl_poly": _purity(not_noise & not_poly),
        "chance_level": chance_level,
    }


def _label_transfer_scores(X, y, groups, k: int = 25, n_splits: int = 5, seed: int = 42) -> dict:
    import numpy as np
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.neighbors import KNeighborsClassifier

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_true_all, y_pred_all = [], []
    for train_idx, test_idx in skf.split(X, y, groups):
        clf = KNeighborsClassifier(n_neighbors=k, metric="cosine", n_jobs=1)
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        y_true_all.append(y[test_idx])
        y_pred_all.append(pred)
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    return {
        "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
        "n_rows": int(len(y_true_all)),
        "n_classes": int(len(np.unique(y))),
        "k": k, "n_splits": n_splits,
    }


def stage_gate(model_slug: str, device: str = "cpu", seed: int = 42) -> dict:
    import numpy as np
    import pandas as pd
    from scipy.stats import spearmanr
    from sklearn.metrics import (adjusted_mutual_info_score,
                                  adjusted_rand_score, v_measure_score)
    from sklearn.metrics.pairwise import cosine_distances

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib
    recl = importlib.import_module("02_recluster")
    frame_mod = importlib.import_module("01_build_analysis_frame")
    import embed_common

    model_dir = _model_dir(model_slug)
    embed_manifest_path = model_dir / f"embeddings_{device}" / ".embed_manifest_merged.json"
    if not embed_manifest_path.exists():
        raise FileNotFoundError(
            f"{embed_manifest_path} missing -- run `--stage embed --device {device}` first"
        )
    embed_manifest = json.loads(embed_manifest_path.read_text())

    # --- G0: truncation, hard precondition. If it fails, STOP -- no other
    # metric is computed or reported (design doc section 1.6). ---
    g0_pass = embed_manifest["frac_truncated"] == 0.0
    if not g0_pass:
        concordance = {
            "model_slug": model_slug, "device": device,
            "verdict": "FAIL", "verdict_reason": "G0 (truncation) failed",
            "G0_pass": False, "frac_truncated": embed_manifest["frac_truncated"],
        }
        _atomic_write_json(concordance, model_dir / "concordance.json")
        gate_md = (
            f"# Task 4 gate -- {model_slug} ({device})\n\n"
            f"**VERDICT: FAIL**\n\n"
            f"G0 (truncation) failed: frac_truncated="
            f"{embed_manifest['frac_truncated']:.4%}. Per the design doc, "
            "no other metric is computed when G0 fails.\n"
        )
        _atomic_write_text(gate_md, model_dir / "GATE.md")
        print(gate_md)
        return concordance

    # --- Load open embedding, OpenAI reference embedding, published labels,
    # OpenAI-same-config reference clustering -- all row-aligned on
    # admission_ID via an inner join. ---
    open_vecs, open_ids = _load_merged(model_slug, device)
    open_df = pd.DataFrame({"admission_ID": open_ids, "row": np.arange(len(open_ids))})

    frame = frame_mod.load_frame(verify_reports=False)
    oai_vecs_full = frame_mod.load_embeddings(frame)
    oai_df = pd.DataFrame({"admission_ID": frame["admission_ID"].to_numpy(), "row": np.arange(len(frame))})

    reference_labels = pd.read_parquet(REFERENCE_LABELS_PATH).rename(columns={"label": "label_openai_ref"})

    organism_lookup = frame.set_index("admission_ID") if "BSI_episode_short_clean" not in frame.columns else None
    # BSI_episode_short_clean is not in FRAME_COLS (01_build_analysis_frame),
    # so read it directly and outcome-blind, mirroring 06_domain_ablation's
    # `_organism_lookup` (same two columns only).
    import csv as _csv
    organism_map = {}
    with open(ROOT / "manuscript" / "phenotype_ml_dataset.csv", newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            aid = row.get("admission_ID")
            if aid:
                organism_map[int(aid)] = row.get("BSI_episode_short_clean")

    merged = (
        open_df.merge(oai_df, on="admission_ID", suffixes=("_open", "_oai"))
        .merge(frame[["admission_ID", "patient_ID", "phenotype_cluster"]], on="admission_ID")
        .merge(reference_labels, on="admission_ID")
    )
    merged["organism"] = merged["admission_ID"].map(organism_map)
    n_rows = len(merged)
    print(f"gate: {n_rows} rows aligned across open/openai/published/reference", flush=True)

    X_open = open_vecs[merged["row_open"].to_numpy()]
    X_oai = oai_vecs_full[merged["row_oai"].to_numpy()]
    published = merged["phenotype_cluster"].to_numpy()
    reference_labels_arr = merged["label_openai_ref"].to_numpy()
    patient_ids = merged["patient_ID"].to_numpy()
    organisms = merged["organism"].to_numpy()

    # --- Cluster the open embedding with the IDENTICAL config used for the
    # OpenAI reference (nc20_nn30_mcs100). ---
    r_open = recl.run_config(X_open, seed=seed, **CLUSTER_CONFIG)
    open_labels = r_open["labels"]

    def _ari_pair(a, b, exclude_noise_either):
        a, b = np.asarray(a), np.asarray(b)
        if exclude_noise_either:
            mask = (a != -1) & (b != -1)
            a, b = a[mask], b[mask]
        return float(adjusted_rand_score(a, b))

    def _g1(reference, ref_name):
        return {
            f"ari_all_vs_{ref_name}": _ari_pair(open_labels, reference, False),
            f"ari_excluding_noise_vs_{ref_name}": _ari_pair(open_labels, reference, True),
            f"ami_vs_{ref_name}": float(adjusted_mutual_info_score(open_labels, reference)),
            f"v_measure_vs_{ref_name}": float(v_measure_score(open_labels, reference)),
        }

    g1 = {}
    g1.update(_g1(reference_labels_arr, "openai_same_config"))
    g1.update(_g1(published, "published"))
    g1["n_clusters_open"] = r_open["n_clusters"]
    g1["n_clusters_openai_same_config"] = int(len(set(reference_labels_arr.tolist()) - {-1}))
    g1["n_clusters_published"] = int(len(set(published.tolist()) - {-1}))
    g1["noise_frac_open"] = float((open_labels == -1).mean())
    g1["noise_frac_openai_same_config"] = float((reference_labels_arr == -1).mean())
    g1["noise_frac_published"] = float((published == -1).mean())

    # --- G2: organism purity, 4 ways, open vs OpenAI-same-config reference,
    # both next to chance level. ---
    chance_level = float(pd.Series(organisms).value_counts(normalize=True).max())
    g2_open = _organism_purity_ways(open_labels, organisms, chance_level)
    g2_reference = _organism_purity_ways(reference_labels_arr, organisms, chance_level)
    reference_usable = g2_reference["excl_noise_incl_poly"] >= 0.60
    g2 = {"open": g2_open, "openai_same_config_reference": g2_reference,
          "reference_purity_usable": reference_usable}

    # --- G3: geometry agreement, independent of HDBSCAN. ---
    g3 = {
        "mutual_knn_overlap_k10": _mutual_knn_overlap(X_open, X_oai, 10),
        "mutual_knn_overlap_k50": _mutual_knn_overlap(X_open, X_oai, 50),
    }
    rng = np.random.default_rng(seed)
    sub_idx = rng.choice(n_rows, size=min(5000, n_rows), replace=False)
    sub_idx.sort()
    d_open = cosine_distances(X_open[sub_idx])
    d_oai = cosine_distances(X_oai[sub_idx])
    iu = np.triu_indices_from(d_open, k=1)
    rho, pval = spearmanr(d_open[iu], d_oai[iu])
    g3["mantel_spearman_rho_5000subsample"] = float(rho)
    g3["mantel_spearman_pvalue_5000subsample"] = float(pval)
    g3["linear_cka_5000subsample"] = _linear_cka(X_open[sub_idx], X_oai[sub_idx])

    # --- G4: label transfer -- kNN predicting published 16 labels (noise
    # excluded: -1 is not a phenotype), StratifiedGroupKFold by patient_ID. ---
    keep = published != -1
    g4_open = _label_transfer_scores(X_open[keep], published[keep], patient_ids[keep], seed=seed)
    g4_oai = _label_transfer_scores(X_oai[keep], published[keep], patient_ids[keep], seed=seed)
    g4 = {"open": g4_open, "openai_ceiling": g4_oai,
          "macro_f1_gap_from_ceiling": float(g4_oai["macro_f1"] - g4_open["macro_f1"])}

    # --- Verdict rule (design doc section 1.6, written before the numbers
    # are read into the branch below -- the thresholds are quoted verbatim
    # from the design doc's table, not re-derived here). ---
    ari_ref = g1["ari_excluding_noise_vs_openai_same_config"]
    g3a_k50 = g3["mutual_knn_overlap_k50"]
    f1_gap = g4["macro_f1_gap_from_ceiling"]

    if ari_ref >= 0.60 and g3a_k50 >= 0.50 and f1_gap <= 0.10:
        verdict, reason = "ADOPT", (
            f"ari_excluding_noise={ari_ref:.4f} >= 0.60, G3a(k=50)={g3a_k50:.4f} >= 0.50, "
            f"G4 macro-F1 gap={f1_gap:.4f} <= 0.10"
        )
    elif 0.40 <= ari_ref < 0.60:
        verdict, reason = "CONDITIONAL", f"0.40 <= ari_excluding_noise={ari_ref:.4f} < 0.60"
    elif ari_ref < 0.40 and g3a_k50 >= 0.50 and f1_gap <= 0.15:
        verdict, reason = "CONDITIONAL", (
            f"ari_excluding_noise={ari_ref:.4f} < 0.40 but G3a(k=50)={g3a_k50:.4f} >= 0.50 "
            f"and G4 gap={f1_gap:.4f} <= 0.15 (geometry agrees, HDBSCAN bifurcates differently -- M6)"
        )
    else:
        verdict, reason = "FAIL", (
            f"ari_excluding_noise={ari_ref:.4f} < 0.40, G3a(k=50)={g3a_k50:.4f}, "
            f"G4 gap={f1_gap:.4f} -- none of the ADOPT/CONDITIONAL conditions met"
        )

    concordance = {
        "model_slug": model_slug, "device": device, "seed": seed,
        "n_rows": n_rows, "cluster_config": CLUSTER_CONFIG,
        "G0_pass": True, "frac_truncated": embed_manifest["frac_truncated"],
        "G1": g1, "G2": g2, "G3": g3, "G4": g4,
        "verdict": verdict, "verdict_reason": reason,
        "embed_manifest": {k: v for k, v in embed_manifest.items() if k != "env_threads"},
    }
    _atomic_write_json(concordance, model_dir / "concordance.json")

    labels_df = pd.DataFrame({
        "admission_ID": merged["admission_ID"].to_numpy(),
        "label_open": open_labels,
        "label_openai_same_config": reference_labels_arr,
        "phenotype_cluster": published,
    })
    tmp = model_dir / f".tmp_{os.getpid()}_labels.parquet"
    labels_df.to_parquet(tmp, index=False)
    os.replace(tmp, model_dir / "labels.parquet")

    gate_md = _gate_markdown(model_slug, device, concordance)
    _atomic_write_text(gate_md, model_dir / "GATE.md")
    print(gate_md)
    return concordance


def _gate_markdown(model_slug: str, device: str, c: dict) -> str:
    g1, g2, g3, g4 = c["G1"], c["G2"], c["G3"], c["G4"]
    lines = [
        f"# Task 4 gate -- {model_slug} ({device})",
        "",
        f"**VERDICT: {c['verdict']}**",
        "",
        f"Reason: {c['verdict_reason']}",
        "",
        f"n_rows={c['n_rows']}, cluster_config={c['cluster_config']}, seed={c['seed']}",
        "",
        "## G0 -- truncation (hard precondition)",
        f"frac_truncated = {c['frac_truncated']:.6f} -- {'PASS' if c['G0_pass'] else 'FAIL'}",
        "",
        "## G1 -- partition agreement",
        f"vs OpenAI (same config nc20_nn30_mcs100): ari_all={g1['ari_all_vs_openai_same_config']:.4f}, "
        f"ari_excluding_noise={g1['ari_excluding_noise_vs_openai_same_config']:.4f}, "
        f"ami={g1['ami_vs_openai_same_config']:.4f}, v_measure={g1['v_measure_vs_openai_same_config']:.4f}",
        f"vs published 16: ari_all={g1['ari_all_vs_published']:.4f}, "
        f"ari_excluding_noise={g1['ari_excluding_noise_vs_published']:.4f}, "
        f"ami={g1['ami_vs_published']:.4f}, v_measure={g1['v_measure_vs_published']:.4f}",
        f"n_clusters: open={g1['n_clusters_open']}, openai_same_config={g1['n_clusters_openai_same_config']}, "
        f"published={g1['n_clusters_published']}",
        f"noise_frac: open={g1['noise_frac_open']:.4f}, openai_same_config={g1['noise_frac_openai_same_config']:.4f}, "
        f"published={g1['noise_frac_published']:.4f}",
        "",
        "## G2 -- organism purity (4 ways, open vs OpenAI-same-config reference)",
        f"reference (OpenAI same config) purity usable (>=0.60 excl-noise-incl-poly)? {g2['reference_purity_usable']}",
        f"chance level = {g2['open']['chance_level']:.4f}",
        f"open:      {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in g2['open'].items()})}",
        f"reference: {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in g2['openai_same_config_reference'].items()})}",
        "",
        "## G3 -- geometry agreement (independent of HDBSCAN)",
        f"mutual-kNN overlap: k=10 -> {g3['mutual_knn_overlap_k10']:.4f}, k=50 -> {g3['mutual_knn_overlap_k50']:.4f}",
        f"Mantel/Spearman rho (5,000-row subsample) = {g3['mantel_spearman_rho_5000subsample']:.4f} "
        f"(p={g3['mantel_spearman_pvalue_5000subsample']:.2e})",
        f"linear CKA (same subsample) = {g3['linear_cka_5000subsample']:.4f}",
        "",
        "## G4 -- label transfer (kNN k=25 cosine, StratifiedGroupKFold(5, groups=patient_ID), published 16 labels, noise excluded)",
        f"open:           macro_f1={g4['open']['macro_f1']:.4f}, balanced_accuracy={g4['open']['balanced_accuracy']:.4f} "
        f"(n_rows={g4['open']['n_rows']}, n_classes={g4['open']['n_classes']})",
        f"OpenAI ceiling: macro_f1={g4['openai_ceiling']['macro_f1']:.4f}, "
        f"balanced_accuracy={g4['openai_ceiling']['balanced_accuracy']:.4f}",
        f"macro-F1 gap from ceiling = {g4['macro_f1_gap_from_ceiling']:.4f}",
        "",
        "## Verdict rule (design doc section 1.6, pre-registered)",
        "| verdict | condition |",
        "|---|---|",
        "| ADOPT | G0 pass AND ari_excluding_noise>=0.60 AND G3a(k=50)>=0.50 AND G4 macro-F1 gap<=0.10 |",
        "| CONDITIONAL | 0.40<=ARI<0.60, OR ARI<0.40 with G3a>=0.50 and G4 gap<=0.15 |",
        "| FAIL | ARI<0.40 AND G3a<0.50 AND G4 gap>0.15 |",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True,
                    choices=["embed", "floors_seed", "floors_reembed", "floors_device", "gate"])
    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS))
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--subsample-seed", type=int, default=42)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()

    # floors_reembed also calls stage_embed twice internally, so it must get
    # the same treatment as embed. It previously fell through to the else
    # branch and had its threads pinned to 1, and running 2,000 records x 2
    # on a single core cost job 586381 nearly 4 hours on this stage alone
    # (~20 minutes at 128 threads). Job 586200 already verified that the
    # embedding is bit-for-bit identical regardless of thread count, so
    # there is no reason to pin this to 1 thread -- the proposition being
    # tested ("same code + same weights + same device -> same floating-point
    # output") is independent of thread count. floors_seed/gate, which
    # involves clustering, is still kept at 1 thread (Task 2: thread count
    # changes the number of clusters).
    if args.stage in ("embed", "floors_reembed"):
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    else:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["NUMBA_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"

    OUT.mkdir(parents=True, exist_ok=True)

    if args.stage == "embed":
        stage_embed(args.model, device=args.device, batch_size=args.batch_size,
                    limit=args.limit, subsample_seed=(args.subsample_seed if args.limit else None))
    elif args.stage == "floors_seed":
        stage_floors_seed(args.model, device=args.device)
    elif args.stage == "floors_reembed":
        stage_floors_reembed(args.model, device=args.device, limit=args.limit or 2000,
                              subsample_seed=args.subsample_seed)
    elif args.stage == "floors_device":
        stage_floors_device(args.model)
    elif args.stage == "gate":
        stage_gate(args.model, device=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
