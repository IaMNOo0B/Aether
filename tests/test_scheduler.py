import json

from aether.autonomy import AutonomyRunner
from aether.config import (
    AutonomySettings, LimitSettings, ResearchSettings, SchedulerSettings,
)
from aether.scheduler import AutonomyScheduler
from aether.storage import Store


class FakeProvider:
    def __init__(self, response):
        self.response = response

    def chat(self, messages):
        return self.response


def test_scheduler_runs_persisted_goal_with_read_only_tools(tmp_path):
    store = Store(tmp_path / "scheduler.sqlite3")
    store.create_schedule(
        "read-only-goal", "List the workspace",
        interval_seconds=10, now_epoch=0,
    )
    runner = AutonomyRunner(
        FakeProvider(json.dumps({"steps": [
            {"purpose": "Inspect workspace", "tool": "workspace.list", "args": {"path": "."}}
        ]})),
        store, AutonomySettings(enabled=True, allow_read_only=True),
        LimitSettings(), ResearchSettings(), tmp_path,
    )
    scheduler = AutonomyScheduler(
        store, runner, SchedulerSettings(enabled=True, max_tasks_per_tick=1)
    )

    result = scheduler.run_once()

    assert len(result["enqueued"]) == 1
    assert result["results"][0]["status"] == "completed"
    task = store.get_task(result["enqueued"][0])
    assert task["status"] == "completed"


def test_scheduler_never_approves_workspace_writes(tmp_path):
    store = Store(tmp_path / "scheduler.sqlite3")
    store.create_schedule(
        "write-goal", "Try to write a file",
        interval_seconds=10, now_epoch=0,
    )
    runner = AutonomyRunner(
        FakeProvider(json.dumps({"steps": [
            {"purpose": "Write file", "tool": "workspace.write",
             "args": {"path": "should-not-exist.txt", "content": "blocked"}}
        ]})),
        store,
        AutonomySettings(
            enabled=True, allow_read_only=True, allow_workspace_writes=True
        ),
        LimitSettings(), ResearchSettings(), tmp_path,
    )
    scheduler = AutonomyScheduler(
        store, runner, SchedulerSettings(enabled=True, max_tasks_per_tick=1)
    )

    result = scheduler.run_once()

    assert result["results"][0]["status"] == "cancelled"
    assert not (tmp_path / "workspace" / "should-not-exist.txt").exists()


def test_disabled_scheduler_does_not_enqueue_or_execute(tmp_path):
    store = Store(tmp_path / "scheduler.sqlite3")
    scheduler = AutonomyScheduler(
        store, object(), SchedulerSettings(enabled=False)
    )
    assert scheduler.run_once() == {
        "status": "disabled", "enqueued": [], "results": []
    }
