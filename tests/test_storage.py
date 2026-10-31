import json

from aether.storage import Store


def test_messages_persist_and_return_in_order(tmp_path):
    store = Store(tmp_path / "test.sqlite3")
    store.add_message("user", "hello")
    store.add_message("assistant", "hi")
    assert store.recent_messages() == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


def test_memory_and_task_persist(tmp_path):
    store = Store(tmp_path / "test.sqlite3")
    store.add_memory("Windows is the target", source="user")
    memories = store.search_memories("Windows target")
    assert memories[0]["content"] == "Windows is the target"

    store.save_task("task-1", "bootstrap", "in_progress", {"step": 2})
    with store._connect() as db:
        row = db.execute("SELECT * FROM tasks WHERE id = ?", ("task-1",)).fetchone()
    assert row["status"] == "in_progress"
    assert json.loads(row["state_json"]) == {"step": 2}


def test_invalid_message_role_is_rejected(tmp_path):
    store = Store(tmp_path / "test.sqlite3")
    try:
        store.add_message("tool", "not a supported role")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid role should be rejected")
