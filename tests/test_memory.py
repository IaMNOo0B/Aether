import pytest

from aether.memory import MemoryManager
from aether.storage import Store


def test_memory_retrieval_ranks_relevant_documents(tmp_path):
    manager = MemoryManager(Store(tmp_path / "memory.sqlite3"))
    manager.remember(
        "AETHER uses Windows and Python 3.12 for local orchestration",
        tags=["windows", "python"], kind="semantic",
    )
    manager.remember(
        "The user prefers short-form video editing and storytelling",
        tags=["video", "editing"], kind="episodic",
    )
    manager.remember(
        "Use pytest to run the Python test suite",
        tags=["python", "testing"], kind="procedural",
    )

    results = manager.search("Python test suite", limit=3)
    assert results[0]["content"] == "Use pytest to run the Python test suite"
    assert results[0]["kind"] == "procedural"
    assert results[0]["relevance"] > 0


def test_memory_correction_preserves_old_record_and_supersedes_it(tmp_path):
    store = Store(tmp_path / "memory.sqlite3")
    manager = MemoryManager(store)
    old_id = manager.remember(
        "The default model is llama2", source="user", kind="semantic",
        status="user-provided", tags=["model"],
    )

    new_id = manager.correct(
        old_id, "The configured model is phi3:mini-4k-q4_0",
        "The current AETHER settings file specifies this model",
    )

    old = store.get_memory(old_id)
    new = store.get_memory(new_id)
    assert old["status"] == "superseded"
    assert old["metadata"]["superseded_by_reason"].startswith("The current")
    assert new["status"] == "user-corrected"
    assert new["metadata"]["supersedes_memory_id"] == old_id
    assert [item["id"] for item in manager.search("configured model")] == [new_id]


def test_memory_validates_kind_and_confidence(tmp_path):
    manager = MemoryManager(Store(tmp_path / "memory.sqlite3"))
    with pytest.raises(ValueError, match="memory kind"):
        manager.remember("Test", kind="imaginary")
    with pytest.raises(ValueError, match="confidence"):
        manager.remember("Test", confidence=1.5)


def test_memory_search_does_not_return_unrelated_records(tmp_path):
    manager = MemoryManager(Store(tmp_path / "memory.sqlite3"))
    manager.remember("AETHER runs on Windows")
    manager.remember("The best pizza has a thin crust")
    assert manager.search("quantum chemistry") == []
