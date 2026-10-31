import pytest

from aether.storage import Store
from aether.tasks import TaskManager


def test_task_lifecycle_persists_and_transitions(tmp_path):
    manager = TaskManager(Store(tmp_path / "tasks.sqlite3"))
    task = manager.create("Investigate provider failure", {"attempt": 0})
    assert task.status == "queued"
    running = manager.transition(task.id, "running", {"attempt": 1})
    assert running.status == "running"
    completed = manager.transition(task.id, "completed", {"verified": True})
    assert completed.status == "completed"
    assert manager.store.get_task(task.id)["state"] == {"verified": True}


def test_task_manager_rejects_invalid_transition(tmp_path):
    manager = TaskManager(Store(tmp_path / "tasks.sqlite3"))
    task = manager.create("Do not skip states")
    with pytest.raises(ValueError):
        manager.transition(task.id, "completed")
