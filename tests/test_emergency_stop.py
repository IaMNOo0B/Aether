import json

import pytest

from aether.autonomy import AutonomyRunner, EmergencyStop
from aether.config import AutonomySettings, LimitSettings, ResearchSettings, SchedulerSettings
from aether.scheduler import AutonomyScheduler
from aether.storage import Store


class FakeProvider:
    def __init__(self, response=None):
        self.response = response

    def chat(self, messages):
        if self.response is None:
            raise AssertionError("Provider must not be called after emergency stop")
        return self.response


def make_runner(tmp_path, store, response=None):
    return AutonomyRunner(
        FakeProvider(response), store,
        AutonomySettings(enabled=True, allow_read_only=True),
        LimitSettings(), ResearchSettings(), tmp_path,
    )


def test_emergency_stop_is_persistent_and_audited(tmp_path):
    database = tmp_path / "controls.sqlite3"
    store = Store(database)
    assert store.emergency_stop_enabled() is False

    store.set_emergency_stop(True, "operator pressed stop")
    assert Store(database).emergency_stop_enabled() is True
    store.set_emergency_stop(False, "operator resumed")
    assert store.emergency_stop_enabled() is False

    events = store.system_control_events()
    assert [event["enabled"] for event in reversed(events)] == [1, 0]
    assert events[0]["reason"] == "operator resumed"


def test_emergency_stop_cancels_queued_task_without_calling_model(tmp_path):
    store = Store(tmp_path / "controls.sqlite3")
    runner = make_runner(tmp_path, store)
    task = runner.enqueue("Do a harmless read-only task")
    store.set_emergency_stop(True, "test stop")

    result = runner.run_task(task.id)

    assert result["status"] == "cancelled"
    assert store.get_task(task.id)["status"] == "cancelled"
    assert store.get_task(task.id)["state"]["emergency_stop"] is True


def test_emergency_stop_blocks_new_autonomy_tasks(tmp_path):
    store = Store(tmp_path / "controls.sqlite3")
    runner = make_runner(tmp_path, store)
    store.set_emergency_stop(True)
    with pytest.raises(EmergencyStop, match="emergency stop is active"):
        runner.enqueue("Do not enqueue this")


def test_emergency_stop_between_steps_stops_remaining_tools(tmp_path):
    store = Store(tmp_path / "controls.sqlite3")
    plan = json.dumps({"steps": [
        {"purpose": "First read", "tool": "workspace.list", "args": {"path": "."}},
        {"purpose": "Second read", "tool": "workspace.list", "args": {"path": "."}},
    ]})
    runner = make_runner(tmp_path, store, plan)
    task = runner.enqueue("Run two read-only steps")
    original_execute = runner.gateway.execute
    calls = {"count": 0}

    def execute_then_stop(tool, args, approval=None):
        calls["count"] += 1
        result = original_execute(tool, args, approval)
        if calls["count"] == 1:
            store.set_emergency_stop(True, "stop between tool calls")
        return result

    runner.gateway.execute = execute_then_stop
    result = runner.run_task(task.id)

    assert result["status"] == "cancelled"
    assert len(result["receipts"]) == 1
    assert store.get_task(task.id)["state"]["emergency_stop"] is True


def test_scheduler_does_not_enqueue_when_emergency_stop_is_active(tmp_path):
    store = Store(tmp_path / "controls.sqlite3")
    store.create_schedule(
        "scheduled", "Read a public page", interval_seconds=5, now_epoch=0
    )
    store.set_emergency_stop(True, "stop scheduler")
    scheduler = AutonomyScheduler(
        store, object(), SchedulerSettings(enabled=True)
    )

    result = scheduler.run_once()

    assert result["status"] == "emergency_stop"
    assert result["enqueued"] == []
    assert store.list_tasks(status="queued") == []

def test_tool_gateway_itself_checks_emergency_stop(tmp_path):
    store = Store(tmp_path / "controls.sqlite3")
    runner = make_runner(tmp_path, store)
    store.set_emergency_stop(True, "block direct gateway access")

    with pytest.raises(EmergencyStop, match="tool execution is blocked"):
        runner.gateway.execute("workspace.list", {"path": "."})

def test_emergency_stop_while_waiting_for_approval_blocks_write(tmp_path):
    store = Store(tmp_path / "controls.sqlite3")
    plan = json.dumps({"steps": [
        {"purpose": "Write file", "tool": "workspace.write",
         "args": {"path": "must-not-exist.txt", "content": "blocked"}}
    ]})
    runner = AutonomyRunner(
        FakeProvider(plan), store,
        AutonomySettings(
            enabled=True, allow_read_only=True, allow_workspace_writes=True
        ),
        LimitSettings(), ResearchSettings(), tmp_path,
    )
    task = runner.enqueue("Write a test file")

    def approve_then_stop(tool, purpose, args):
        store.set_emergency_stop(True, "stop pressed during approval")
        return True

    result = runner.run_task(task.id, approve_then_stop)

    assert result["status"] == "cancelled"
    assert store.get_task(task.id)["state"]["emergency_stop"] is True
    assert not (tmp_path / "workspace" / "must-not-exist.txt").exists()
