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

def test_engine_self_model_does_not_overclaim_capability_verification(tmp_path):
    settings = load_settings(Path("config/settings.toml"))
    store = Store(tmp_path / "aether.sqlite3")
    engine = AetherEngine(settings, store)

    assert engine.self_model.path == store.path
    capabilities = engine.self_model.snapshot()["capabilities"]
    assert len(capabilities) >= 10
    assert all(item["state"] == "implemented_unverified" for item in capabilities)
    assert all(item["evidence"] == [] for item in capabilities)

def test_engine_cancels_model_call_when_resource_budget_blocks(tmp_path, monkeypatch):
    from dataclasses import replace
    from aether.resources import ResourceLimitExceeded

    settings = load_settings(Path("config/settings.toml"))
    settings = replace(
        settings,
        models=replace(settings.models, enabled=True),
    )
    store = Store(tmp_path / "aether.sqlite3")
    engine = AetherEngine(settings, store)

    def blocked(*args, **kwargs):
        raise ResourceLimitExceeded("RAM budget exceeded")

    monkeypatch.setattr(engine.resources, "check", blocked)
    answer = engine.respond("hello")

    assert "resource budget blocked it" in answer
    assert store.list_tasks(limit=1)[0]["status"] == "cancelled"
