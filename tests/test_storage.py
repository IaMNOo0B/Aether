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



def test_database_backup_and_restore_preserve_memory_and_messages(tmp_path):
    original = Store(tmp_path / "original.sqlite3")
    original.add_message("user", "keep this conversation")
    original.add_memory("Keep this memory", source="user", status="user-provided")
    backup = original.backup_to(tmp_path / "backups" / "aether.sqlite3")

    target = Store(tmp_path / "target.sqlite3")
    target.add_memory("This should be replaced")
    target.restore_from(backup)

    assert target.recent_messages() == [
        {"role": "user", "content": "keep this conversation"}
    ]
    assert [item["content"] for item in target.list_memories()] == ["Keep this memory"]


def test_restore_rejects_invalid_backup_without_changing_live_database(tmp_path):
    store = Store(tmp_path / "live.sqlite3")
    store.add_memory("Do not lose this")
    invalid = tmp_path / "not-a-database.sqlite3"
    invalid.write_text("not sqlite", encoding="utf-8")

    try:
        store.restore_from(invalid)
    except Exception:
        pass
    else:
        raise AssertionError("Invalid restore source should be rejected")

    assert store.list_memories()[0]["content"] == "Do not lose this"


def test_backup_cannot_overwrite_live_database(tmp_path):
    store = Store(tmp_path / "live.sqlite3")
    try:
        store.backup_to(store.path)
    except ValueError:
        pass
    else:
        raise AssertionError("Live database must not be accepted as backup target")
