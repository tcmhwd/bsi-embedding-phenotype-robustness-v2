import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
sys.path.insert(0, SCRIPTS_DIR)
from importlib import import_module

mod = import_module("embed_variants_openai")


class FakeOpenAIClient:
    """Mimics the OpenAI SDK's `client.embeddings.create(...)` surface just
    enough for `_embed_batch_with_retry`: returns one fixed-length fake
    vector per input text and a `usage.total_tokens` count. Never touches
    the network."""

    def __init__(self, dim=8, fail_first_n_calls=0):
        self.dim = dim
        self.calls = []
        self.fail_first_n_calls = fail_first_n_calls
        self._call_count = 0
        self.embeddings = self  # so `client.embeddings.create(...)` resolves to self.create

    def create(self, model, input):
        self._call_count += 1
        if self._call_count <= self.fail_first_n_calls:
            raise RuntimeError("simulated transient API error")
        self.calls.append(list(input))
        data = [_Emb([float(i)] * self.dim) for i in range(len(input))]
        # a trivial deterministic token count: one token per word, matching
        # nothing real but sufficient to exercise the accounting path
        total_tokens = sum(len(t.split()) for t in input)
        return _Resp(data, total_tokens)


class _Emb:
    def __init__(self, vec):
        self.embedding = vec


class _Resp:
    def __init__(self, data, total_tokens):
        self.data = data
        self.usage = _Usage(total_tokens)


class _Usage:
    def __init__(self, total_tokens):
        self.total_tokens = total_tokens


@pytest.fixture
def fake_tasks(tmp_path, monkeypatch):
    tasks = {
        "06_ablation": tmp_path / "06_ablation",
        "07_template": tmp_path / "07_template",
    }
    monkeypatch.setattr(mod, "TASKS", tasks)
    return tasks


def _write_narratives(narratives_dir: Path, admission_ids, text_fn=lambda aid: f"case {aid} short narrative"):
    narratives_dir.mkdir(parents=True, exist_ok=True)
    for aid in admission_ids:
        (narratives_dir / f"{aid}.txt").write_text(text_fn(aid))


def test_relabel_invalid_e5_moves_existing_dir_once(fake_tasks):
    task_out = fake_tasks["06_ablation"]
    old_emb = task_out / "embeddings"
    old_emb.mkdir(parents=True)
    (old_emb / "marker.txt").write_text("old e5 vectors")

    result = mod.relabel_invalid_e5("06_ablation")
    assert result["action"] == "relabelled"
    assert not old_emb.exists()
    assert (task_out / "embeddings_INVALID_truncated_e5" / "marker.txt").exists()

    # idempotent: second call is a no-op, does not raise, and (critically)
    # does NOT re-relabel anything newly written into embeddings/ since
    # (e.g. a resumed run's own OpenAI shards for another variant)
    (task_out / "embeddings").mkdir(parents=True, exist_ok=True)
    (task_out / "embeddings" / "shard_0000_of_api.npy").write_text("not really an npy, just a marker")
    result2 = mod.relabel_invalid_e5("06_ablation")
    assert result2["action"] == "already_migrated"
    assert (task_out / "embeddings" / "shard_0000_of_api.npy").exists()


def test_relabel_invalid_e5_nothing_to_relabel_when_no_prior_embeddings(fake_tasks):
    result = mod.relabel_invalid_e5("07_template")
    assert result["action"] == "nothing_to_relabel"


def test_dry_run_reports_tokens_and_makes_no_api_calls(fake_tasks):
    narratives_dir = fake_tasks["06_ablation"] / "narratives" / "full"
    _write_narratives(narratives_dir, [1, 2, 3])

    client = FakeOpenAIClient()
    manifest = mod.stage_embed_variant("06_ablation", "full", client=client, dry_run=True)

    assert client.calls == []  # no API call made
    assert manifest["dry_run_token_report"]["n_docs"] == 3
    assert manifest["dry_run_token_report"]["frac_truncated"] == 0.0


def test_stage_embed_variant_writes_shards_and_is_resumable(fake_tasks, monkeypatch):
    monkeypatch.setattr(mod, "SHARD_SIZE", 2)  # force multiple shards for 5 docs
    narratives_dir = fake_tasks["06_ablation"] / "narratives" / "full"
    _write_narratives(narratives_dir, [10, 20, 30, 40, 50])

    client = FakeOpenAIClient(dim=4)
    manifest = mod.stage_embed_variant("06_ablation", "full", client=client)

    emb_dir = fake_tasks["06_ablation"] / "embeddings" / "full"
    shard_files = sorted(emb_dir.glob("shard_*_of_api.npy"))
    assert len(shard_files) == 3  # ceil(5/2)
    all_ids = mod._already_done_ids(emb_dir)
    assert all_ids == {10, 20, 30, 40, 50}
    assert manifest["n_docs_embedded"] == 5
    assert manifest["total_tokens_billed"] > 0
    assert manifest["cost_usd"] == pytest.approx(
        manifest["total_tokens_billed"] / 1_000_000 * mod.PRICE_PER_MILLION_TOKENS_USD
    )
    assert manifest["frac_truncated"] == 0.0  # guard fields merged in

    n_calls_first_run = len(client.calls)

    # Resume: a second call with a client that would raise if asked to
    # embed anything must do nothing (everything already done).
    class RaisingClient:
        embeddings = None

        def __init__(self):
            self.embeddings = self

        def create(self, model, input):
            raise AssertionError("must not be called -- all docs already embedded")

    manifest2 = mod.stage_embed_variant("06_ablation", "full", client=RaisingClient())
    assert manifest2["n_docs_embedded"] == 5  # unchanged, no new work
    assert n_calls_first_run > 0


def test_stage_embed_variant_rejects_a_document_over_the_window(fake_tasks):
    narratives_dir = fake_tasks["06_ablation"] / "narratives" / "full"
    # cl100k_base: "word " repeated many times comfortably exceeds 8,191 tokens.
    _write_narratives(
        narratives_dir, [1, 2],
        text_fn=lambda aid: "clinical finding" if aid == 1 else "clinical finding " * 6000,
    )

    class RaisingClient:
        def __init__(self):
            self.embeddings = self

        def create(self, model, input):
            raise AssertionError("must not be called -- truncation guard must fire first")

    with pytest.raises(Exception) as excinfo:
        mod.stage_embed_variant("06_ablation", "full", client=RaisingClient())
    sys.path.insert(0, SCRIPTS_DIR)
    embed_common = import_module("embed_common")
    assert isinstance(excinfo.value, embed_common.TruncationError)


def test_dedup_full_across_tasks_copies_without_reembedding(fake_tasks):
    src_dir = fake_tasks["06_ablation"] / "embeddings" / "full"
    src_dir.mkdir(parents=True)
    vecs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    np.save(src_dir / "merged.npy", vecs)
    (src_dir / "merged_ids.json").write_text(json.dumps([10, 20]))

    result = mod.dedup_full_across_tasks()
    assert result["action"] == "deduped_from_06_ablation"

    dst_dir = fake_tasks["07_template"] / "embeddings" / "full"
    assert np.array_equal(np.load(dst_dir / "merged.npy"), vecs)
    assert json.loads((dst_dir / "merged_ids.json").read_text()) == [10, 20]


def test_dedup_full_raises_if_source_not_yet_embedded(fake_tasks):
    with pytest.raises(FileNotFoundError):
        mod.dedup_full_across_tasks()


def test_token_budget_batches_respects_doc_and_token_caps(monkeypatch):
    monkeypatch.setattr(mod, "MAX_DOCS_PER_REQUEST", 2)
    monkeypatch.setattr(mod, "MAX_TOKENS_PER_REQUEST", 1000)

    def count_fn(text):
        return len(text.split())

    items = [(1, "a b c"), (2, "d e f"), (3, "g h i"), (4, "j k l")]
    batches = list(mod._token_budget_batches(items, count_fn))
    assert len(batches) == 2
    assert [aid for aid, _ in batches[0]] == [1, 2]
    assert [aid for aid, _ in batches[1]] == [3, 4]


def test_token_budget_batches_splits_on_token_budget(monkeypatch):
    monkeypatch.setattr(mod, "MAX_DOCS_PER_REQUEST", 100)
    monkeypatch.setattr(mod, "MAX_TOKENS_PER_REQUEST", 5)

    def count_fn(text):
        return len(text.split())

    items = [(1, "a b c"), (2, "d e f")]  # 3 + 3 tokens, budget 5 -> must split
    batches = list(mod._token_budget_batches(items, count_fn))
    assert len(batches) == 2
