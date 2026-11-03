import pytest
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


def test_engine_uses_explicit_independent_provider_routes_for_parallel_agents(tmp_path):
    from dataclasses import replace
    from aether.config import ProviderEndpointSettings
    from aether.providers import ModelDescriptor

    settings = load_settings(Path("config/settings.toml"))
    fallbacks = tuple(
        ProviderEndpointSettings(
            provider="openai_compatible",
            base_url=f"https://specialist-{index}.example.invalid/v1",
            model=f"model-{index}",
            api_key_env="",
        )
        for index in range(1, 4)
    )
    models = replace(
        settings.models,
        provider="openai_compatible",
        enabled=True,
        base_url="https://primary.example.invalid/v1",
        model="researcher-model",
        fallbacks=fallbacks,
    )
    agents = replace(
        settings.agents,
        parallel_enabled=True,
        researcher_provider="openai_compatible",
        planner_provider="fallback_1",
        critic_provider="fallback_2",
        verifier_provider="fallback_3",
    )
    settings = replace(settings, models=models, agents=agents)
    store = Store(tmp_path / "parallel.sqlite3")
    engine = AetherEngine(settings, store)

    class FakeProvider:
        def __init__(self, label):
            self.label = label

        def chat(self, messages):
            return f"{self.label} output"

        def is_available(self):
            return True

    for key in ("openai_compatible", "fallback_1", "fallback_2", "fallback_3"):
        engine.registry._providers[key] = FakeProvider(key)

    outputs = engine.run_parallel_agents("Review the system")
    assert [item.role for item in outputs] == [
        "researcher", "planner", "critic", "independent_verifier"
    ]
    assert len(store.list_agent_runs(limit=10)) == 4


def test_engine_blocks_parallel_roles_that_would_load_multiple_local_models(tmp_path):
    from dataclasses import replace
    from aether.config import ProviderEndpointSettings

    settings = load_settings(Path("config/settings.toml"))
    fallbacks = tuple(
        ProviderEndpointSettings(
            provider="ollama", base_url="http://127.0.0.1:11434",
            model=f"local-{index}", api_key_env="",
        )
        for index in range(1, 4)
    )
    settings = replace(
        settings,
        models=replace(settings.models, enabled=True, fallbacks=fallbacks),
        agents=replace(
            settings.agents, parallel_enabled=True,
            researcher_provider="ollama", planner_provider="fallback_1",
            critic_provider="fallback_2", verifier_provider="fallback_3",
        ),
    )
    engine = AetherEngine(settings, Store(tmp_path / "local-parallel.sqlite3"))
    with pytest.raises(ValueError, match="at most one local"):
        engine.run_parallel_agents("Test VRAM guard")
