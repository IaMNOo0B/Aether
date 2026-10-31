import pytest

from aether.storage import Store
from aether.tasks import TaskManager


def test_task_lifecycle_writes_ordered_audit_events(tmp_path):
    manager = TaskManager(Store(tmp_path / "events.sqlite3"))
    task = manager.create("Build and verify a report", {"kind": "test"})
    manager.transition(task.id, "running")
    manager.checkpoint(
        task.id, {"kind": "test", "current_step": 1},
        {"step": 1, "result": "source retrieved"},
    )
    manager.transition(task.id, "completed", {"verified": True})

    events = manager.events(task.id)
    assert [event["event_type"] for event in events] == [
        "created", "transition", "checkpoint", "transition"
    ]
    assert [event["sequence"] for event in events] == sorted(
        event["sequence"] for event in events
    )
    assert events[1]["from_status"] == "queued"
    assert events[1]["to_status"] == "running"
    assert events[2]["payload"]["step"] == 1
    assert events[-1]["to_status"] == "completed"


def test_interrupted_task_is_marked_failed_without_automatic_replay(tmp_path):
    manager = TaskManager(Store(tmp_path / "recovery.sqlite3"))
    task = manager.create("Write an external artifact", {"kind": "autonomy"})
    manager.transition(task.id, "running")
    manager.checkpoint(task.id, {
        "kind": "autonomy",
        "plan": [{"tool": "workspace.write", "args": {"path": "report.txt"}}],
        "current_step": 1,
        "receipts": [],
    })

    # Simulate a process restart by constructing a new Store/TaskManager instance.
    restarted = TaskManager(Store(tmp_path / "recovery.sqlite3"))
    recovered = restarted.recover_interrupted()
    assert recovered == [task.id]
    saved = restarted.store.get_task(task.id)
    assert saved["status"] == "failed"
    assert saved["state"]["recovery"]["status"] == "interrupted"
    assert saved["state"]["recovery"]["policy"] == "manual_review_required"
    assert "not replayed" in saved["state"]["recovery"]["reason"]
    assert restarted.recover_interrupted() == []
    assert restarted.events(task.id)[-1]["event_type"] == "interrupted_recovery"


def test_checkpoint_only_allowed_while_running(tmp_path):
    manager = TaskManager(Store(tmp_path / "checkpoint.sqlite3"))
    task = manager.create("Do not checkpoint queued work")
    with pytest.raises(ValueError, match="Only running tasks"):
        manager.checkpoint(task.id, {"step": 1})


def test_unknown_task_event_lookup_fails(tmp_path):
    manager = TaskManager(Store(tmp_path / "unknown.sqlite3"))
    with pytest.raises(KeyError, match="Unknown task"):
        manager.events("missing-task")
