import json

from aether.autonomy import AutonomyRunner
from aether.config import AutonomySettings, LimitSettings, ResearchSettings
from aether.storage import Store


class FakeProvider:
    def __init__(self, response):
        self.response = response

    def chat(self, messages):
        return self.response


def test_autonomy_executes_approved_workspace_tools_and_persists_receipts(tmp_path):
    response = json.dumps({"steps": [
        {"purpose": "Create a reversible note", "tool": "workspace.write",
         "args": {"path": "notes/result.txt", "content": "research complete"}},
        {"purpose": "Read back the note", "tool": "workspace.read",
         "args": {"path": "notes/result.txt"}},
    ]})
    store = Store(tmp_path / "aether.sqlite3")
    policy = AutonomySettings(
        enabled=True, allow_read_only=True, allow_workspace_writes=True,
        allow_code_execution=False, workspace="workspace",
    )
    runner = AutonomyRunner(
        FakeProvider(response), store, policy, LimitSettings(), ResearchSettings(), tmp_path
    )
    result = runner.run("Write a short note", lambda *args: True)
    assert result["status"] == "completed"
    assert len(result["receipts"]) == 2
    assert result["receipts"][1]["result"]["content"] == "research complete"
    assert (tmp_path / "workspace" / "notes" / "result.txt").read_text() == "research complete"
    assert store.get_task(result["task_id"])["status"] == "completed"


def test_autonomy_declined_write_is_cancelled_without_side_effect(tmp_path):
    response = json.dumps({"steps": [
        {"purpose": "Write a file", "tool": "workspace.write",
         "args": {"path": "blocked.txt", "content": "no"}},
    ]})
    store = Store(tmp_path / "aether.sqlite3")
    policy = AutonomySettings(enabled=True, allow_workspace_writes=True)
    runner = AutonomyRunner(
        FakeProvider(response), store, policy, LimitSettings(), ResearchSettings(), tmp_path
    )
    result = runner.run("Write a file", lambda *args: False)
    assert result["status"] == "cancelled"
    assert not (tmp_path / "workspace" / "blocked.txt").exists()


def test_autonomy_rejects_unknown_model_generated_tools(tmp_path):
    response = json.dumps({"steps": [
        {"purpose": "Run arbitrary command", "tool": "shell.exec", "args": {"command": "whoami"}},
    ]})
    store = Store(tmp_path / "aether.sqlite3")
    policy = AutonomySettings(enabled=True)
    runner = AutonomyRunner(
        FakeProvider(response), store, policy, LimitSettings(), ResearchSettings(), tmp_path
    )
    try:
        runner.run("Execute a command")
    except PermissionError:
        pass
    else:
        raise AssertionError("Unknown tool should be denied")
