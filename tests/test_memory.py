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

def test_superseded_memory_cannot_be_corrected_twice(tmp_path):
    manager = MemoryManager(Store(tmp_path / "memory.sqlite3"))
    original = manager.remember("Old fact")
    manager.correct(original, "Corrected fact", "The original was inaccurate")
    with pytest.raises(ValueError, match="already been superseded"):
        manager.correct(original, "Another version", "Do not fork a correction chain")



def test_memory_versions_return_full_ordered_correction_chain(tmp_path):
    store = Store(tmp_path / "memory.sqlite3")
    manager = MemoryManager(store)
    first = manager.remember("The model is alpha")
    second = manager.correct(first, "The model is beta", "Updated configuration")
    third = manager.correct(second, "The model is gamma", "Newer configuration")

    versions = manager.versions(first)
    assert [item["id"] for item in versions] == [first, second, third]
    assert [item["id"] for item in manager.versions(third)] == [first, second, third]


def test_memory_conflicts_only_flag_explicit_fact_metadata(tmp_path):
    manager = MemoryManager(Store(tmp_path / "memory.sqlite3"))
    left = manager.remember(
        "The deployment notes say one thing",
        metadata={"facts": [{"subject": "AETHER", "predicate": "default model", "object": "alpha"}]},
    )
    right = manager.remember(
        "A separate note says something else",
        metadata={"facts": [{"subject": "aether", "predicate": "default model", "object": "beta"}]},
    )
    conflicts = manager.find_contradictions(right)
    assert len(conflicts) == 1
    assert {conflicts[0]["left_memory_id"], conflicts[0]["right_memory_id"]} == {left, right}
    assert conflicts[0]["resolution"].startswith("unresolved")


def test_memory_conflicts_ignore_superseded_versions(tmp_path):
    manager = MemoryManager(Store(tmp_path / "memory.sqlite3"))
    old = manager.remember(
        "Old model note",
        metadata={"facts": [{"subject": "AETHER", "predicate": "model", "object": "alpha"}]},
    )
    manager.remember(
        "Current model note",
        metadata={"facts": [{"subject": "AETHER", "predicate": "model", "object": "beta"}]},
    )
    manager.correct(old, "Corrected model note", "Old note was inaccurate")
    assert manager.find_contradictions() == []
