import pytest

from aether.goals import GoalDiscovery
from aether.self_model import SelfModel
from aether.storage import Store


def test_goal_discovery_persists_capability_gaps_without_executing_them(tmp_path):
    database = tmp_path / "goals.sqlite3"
    store = Store(database)
    model = SelfModel(database)
    model.register_capability(
        "vision.ocr", "Vision OCR", "Extract text from images", state="implemented_unverified"
    )
    discovery = GoalDiscovery(store, model)

    first = discovery.discover()
    second = discovery.discover()

    assert len(first) == 1
    assert first[0]["objective"].startswith("Evaluate capability")
    assert first[0]["status"] == "proposed"
    assert first[0]["id"] == second[0]["id"]
    assert store.list_schedules() == []


def test_discovered_goal_requires_explicit_approval_before_scheduling(tmp_path):
    database = tmp_path / "goals.sqlite3"
    store = Store(database)
    model = SelfModel(database)
    model.register_capability("audio.asr", "Audio ASR", "Speech recognition", state="planned")
    discovery = GoalDiscovery(store, model)
    candidate = discovery.discover()[0]

    with pytest.raises(PermissionError, match="explicitly approved"):
        discovery.schedule(candidate["id"], interval_seconds=60)

    discovery.approve(candidate["id"])
    schedule_id = discovery.schedule(candidate["id"], interval_seconds=60, max_runs=1)

    assert schedule_id == f"discovered-{candidate['id']}"
    assert store.list_schedules()[0]["id"] == schedule_id
    assert store.list_goal_candidates()[0]["status"] == "scheduled"


def test_goal_discovery_rejects_unbounded_limits_and_terminal_review(tmp_path):
    database = tmp_path / "goals.sqlite3"
    store = Store(database)
    discovery = GoalDiscovery(store)
    with pytest.raises(ValueError):
        discovery.discover(limit=1000)
    with pytest.raises(ValueError, match="already-reviewed"):
        store.set_goal_candidate_status("missing", "approved")
