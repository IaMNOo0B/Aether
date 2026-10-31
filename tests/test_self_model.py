from aether.self_model import SelfModel


def test_identity_and_history_survive_restart(tmp_path):
    database = tmp_path / "aether.sqlite3"
    model = SelfModel(database)
    first = model.snapshot()
    model.update_profile(version="0.4.0", objectives=["Improve recovery"])
    restarted = SelfModel(database)
    current = restarted.snapshot()

    assert current["identity_id"] == first["identity_id"]
    assert current["version"] == "0.4.0"
    assert current["objectives"] == ["Improve recovery"]
    assert len(restarted.history()) == 2


def test_capability_is_not_verified_without_evidence(tmp_path):
    model = SelfModel(tmp_path / "aether.sqlite3")
    capability = model.register_capability(
        "memory.hybrid", "Hybrid memory retrieval", "Keyword plus semantic retrieval"
    )
    assert capability["state"] == "planned"
    assert capability["evidence"] == []

    verified = model.record_capability_result(
        "memory.hybrid", "verified", "tests/test_memory.py::test_hybrid_retrieval passed"
    )
    assert verified["state"] == "verified"
    assert len(verified["evidence"]) == 1


def test_capability_result_requires_evidence(tmp_path):
    import pytest

    model = SelfModel(tmp_path / "aether.sqlite3")
    model.register_capability("voice", "Voice", "Speech input")
    with pytest.raises(ValueError, match="evidence"):
        model.record_capability_result("voice", "verified", " ")
