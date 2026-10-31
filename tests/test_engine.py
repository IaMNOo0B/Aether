from pathlib import Path

from aether.config import load_settings
from aether.engine import AetherEngine
from aether.storage import Store


def test_engine_persists_exchange_when_model_is_disabled(tmp_path):
    settings = load_settings(Path("config/settings.toml"))
    store = Store(tmp_path / "aether.sqlite3")
    engine = AetherEngine(settings, store)

    answer = engine.respond("hello")

    assert "language model is disabled" in answer
    history = store.recent_messages()
    assert history[-2] == {"role": "user", "content": "hello"}
    assert history[-1]["role"] == "assistant"
    tasks = store.list_tasks(limit=1)
    assert tasks[0]["status"] == "completed"
    assert tasks[0]["state"]["response_persisted"] is True
