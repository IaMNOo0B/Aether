import json
from urllib.error import HTTPError

import pytest

from aether.embeddings import EmbeddingUnavailable, OllamaEmbeddingProvider
from aether.memory import MemoryManager
from aether.storage import Store


class FakeResponse:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


class FakeEmbedding:
    model = "fake-embedding-v1"

    def embed(self, text):
        value = text.casefold()
        if any(term in value for term in ("cat", "feline", "pet")):
            return [1.0, 0.0]
        if any(term in value for term in ("finance", "accounting", "ledger")):
            return [0.0, 1.0]
        return [0.0, 0.0]


class FailingEmbedding:
    model = "unavailable-model"

    def embed(self, text):
        raise EmbeddingUnavailable("embedding server offline")


def test_ollama_embedding_provider_reads_modern_embed_response(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"embeddings": [[0.1, 0.2, 0.3]]})

    monkeypatch.setattr("aether.embeddings.urlopen", fake_urlopen)
    provider = OllamaEmbeddingProvider("http://localhost:11434", "nomic-embed-text")
    assert provider.embed("test sentence") == [0.1, 0.2, 0.3]
    assert captured["url"] == "http://localhost:11434/api/embed"
    assert captured["payload"]["model"] == "nomic-embed-text"


def test_ollama_embedding_provider_falls_back_to_legacy_endpoint(monkeypatch):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if request.full_url.endswith("/api/embed"):
            raise HTTPError(request.full_url, 404, "not found", {}, None)
        return FakeResponse({"embedding": [0.4, 0.5]})

    monkeypatch.setattr("aether.embeddings.urlopen", fake_urlopen)
    provider = OllamaEmbeddingProvider("http://localhost:11434", "legacy-model")
    assert provider.embed("test") == [0.4, 0.5]
    assert calls == [
        "http://localhost:11434/api/embed",
        "http://localhost:11434/api/embeddings",
    ]


def test_ollama_embedding_provider_rejects_invalid_vectors(monkeypatch):
    monkeypatch.setattr(
        "aether.embeddings.urlopen",
        lambda *args, **kwargs: FakeResponse({"embeddings": [["bad", 0.2]]}),
    )
    provider = OllamaEmbeddingProvider("http://localhost:11434", "model")
    with pytest.raises(EmbeddingUnavailable, match="non-numeric"):
        provider.embed("text")


def test_hybrid_memory_finds_semantically_related_text_without_shared_terms(tmp_path):
    manager = MemoryManager(
        Store(tmp_path / "memory.sqlite3"),
        embedding_provider=FakeEmbedding(),
        semantic_weight=0.8,
    )
    manager.remember("AETHER stores pictures of domestic cats")
    manager.remember("Quarterly finance accounting ledger")

    results = manager.search("feline animal")

    assert results[0]["content"] == "AETHER stores pictures of domestic cats"
    assert results[0]["semantic_similarity"] == 1.0


def test_memory_index_backfills_records_created_without_embeddings(tmp_path):
    store = Store(tmp_path / "memory.sqlite3")
    lexical = MemoryManager(store)
    first = lexical.remember("A cat sleeps on the sofa")
    second = lexical.remember("Finance ledger is balanced")
    hybrid = MemoryManager(store, embedding_provider=FakeEmbedding())

    result = hybrid.index_pending(limit=10)

    assert result == {"indexed": 2, "remaining": 0, "error": None}
    assert store.get_memory_embedding(first, "fake-embedding-v1") == [1.0, 0.0]
    assert store.get_memory_embedding(second, "fake-embedding-v1") == [0.0, 1.0]


def test_embedding_failure_keeps_memory_and_uses_lexical_fallback(tmp_path):
    manager = MemoryManager(
        Store(tmp_path / "memory.sqlite3"),
        embedding_provider=FailingEmbedding(),
        semantic_weight=0.8,
    )
    manager.remember("The AETHER project uses local Python memory")

    results = manager.search("AETHER Python memory")

    assert len(results) == 1
    assert results[0]["content"].startswith("The AETHER")
    assert results[0]["semantic_similarity"] is None


def test_memory_correction_indexes_the_new_record_when_available(tmp_path):
    store = Store(tmp_path / "memory.sqlite3")
    manager = MemoryManager(store, embedding_provider=FakeEmbedding())
    old_id = manager.remember("A pet is a cat")
    new_id = manager.correct(old_id, "A feline is a domestic cat", "More precise wording")

    assert store.get_memory_embedding(old_id, "fake-embedding-v1") is not None
    assert store.get_memory_embedding(new_id, "fake-embedding-v1") == [1.0, 0.0]
    assert [item["id"] for item in manager.search("feline")] == [new_id]
