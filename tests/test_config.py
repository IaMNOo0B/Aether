from pathlib import Path

import pytest

from aether.config import load_settings


def test_load_settings_from_project_config():
    settings = load_settings(Path("config/settings.toml"))
    assert settings.app.name == "AETHER"
    assert settings.models.provider == "ollama"
    assert settings.models.enabled is False
    assert settings.models.api_key_env == "AETHER_API_KEY"


def test_load_settings_accepts_openai_compatible_provider(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text(
        """
[app]
name = "AETHER"
[models]
provider = "openai_compatible"
enabled = true
base_url = "http://127.0.0.1:1234/v1"
model = "local-model"
api_key_env = ""
[storage]
database = "data/aether.sqlite3"
[limits]
max_history_messages = 8
max_prompt_characters = 12000
max_autonomy_steps = 5
max_research_results = 8
tool_timeout_seconds = 60
""",
        encoding="utf-8",
    )

    settings = load_settings(config)
    assert settings.models.provider == "openai_compatible"
    assert settings.models.api_key_env == ""


def test_load_settings_rejects_invalid_api_key_environment_name(tmp_path):
    source = Path("config/settings.toml").read_text(encoding="utf-8")
    config = tmp_path / "settings.toml"
    config.write_text(source.replace(
        'api_key_env = "AETHER_API_KEY"',
        'api_key_env = "NOT VALID"',
    ), encoding="utf-8")

    with pytest.raises(ValueError, match="contain only letters"):
        load_settings(config)

def test_load_settings_parses_provider_fallbacks(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text(
        """
[app]
name = "AETHER"
[models]
provider = "ollama"
enabled = true
base_url = "http://127.0.0.1:11434"
model = "local-model"
[[models.fallbacks]]
provider = "openai_compatible"
base_url = "https://api.example.invalid/v1"
model = "cloud-model"
api_key_env = "AETHER_CLOUD_KEY"
[storage]
database = "data/aether.sqlite3"
[limits]
max_history_messages = 8
max_prompt_characters = 12000
max_autonomy_steps = 5
max_research_results = 8
tool_timeout_seconds = 60
""",
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert len(settings.models.fallbacks) == 1
    assert settings.models.fallbacks[0].provider == "openai_compatible"
    assert settings.models.fallbacks[0].model == "cloud-model"

def test_scheduler_defaults_to_disabled(tmp_path):
    settings = load_settings(Path("config/settings.toml"))
    assert settings.scheduler.enabled is False
    assert settings.scheduler.max_tasks_per_tick == 1


def test_scheduler_requires_autonomy_and_model_enabled(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text(
        """
[app]
name = "AETHER"
[models]
provider = "ollama"
enabled = false
base_url = "http://127.0.0.1:11434"
model = "local-model"
[storage]
database = "data/aether.sqlite3"
[limits]
max_history_messages = 8
max_prompt_characters = 12000
max_autonomy_steps = 5
max_research_results = 8
tool_timeout_seconds = 60
[scheduler]
enabled = true
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires both autonomy.enabled and models.enabled"):
        load_settings(config)

def test_semantic_memory_configuration_defaults_to_disabled():
    settings = load_settings(Path("config/settings.toml"))
    assert settings.memory.semantic_enabled is False
    assert settings.memory.embedding_provider == "ollama"
    assert settings.memory.embedding_model == "nomic-embed-text"


def test_semantic_memory_configuration_validates_weight_and_batch_size(tmp_path):
    source = Path("config/settings.toml").read_text(encoding="utf-8")
    config = tmp_path / "settings.toml"
    config.write_text(
        source.replace("semantic_weight = 0.45", "semantic_weight = 1.5"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="semantic_weight"):
        load_settings(config)

def test_resource_budget_defaults_and_limits_are_loaded():
    settings = load_settings(Path("config/settings.toml"))
    assert settings.resources.enabled is True
    assert settings.resources.max_cpu_percent == 95.0
    assert settings.resources.min_free_vram_mb == 512


def test_resource_budget_rejects_invalid_percentages(tmp_path):
    source = Path("config/settings.toml").read_text(encoding="utf-8")
    config = tmp_path / "settings.toml"
    config.write_text(
        source.replace("max_cpu_percent = 95.0", "max_cpu_percent = 150.0"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_cpu_percent"):
        load_settings(config)


def test_parallel_agent_routes_are_disabled_by_default():
    settings = load_settings(Path("config/settings.toml"))
    assert settings.agents.parallel_enabled is False
    assert settings.agents.max_workers == 3


def test_parallel_agent_routes_require_four_distinct_provider_keys(tmp_path):
    config = tmp_path / "settings.toml"
    config.write_text(
        """
[app]
name = "AETHER"
[models]
provider = "ollama"
enabled = true
base_url = "http://127.0.0.1:11434"
model = "local-model"
[[models.fallbacks]]
provider = "openai_compatible"
base_url = "https://one.example.invalid/v1"
model = "one"
[[models.fallbacks]]
provider = "openai_compatible"
base_url = "https://two.example.invalid/v1"
model = "two"
[[models.fallbacks]]
provider = "openai_compatible"
base_url = "https://three.example.invalid/v1"
model = "three"
[storage]
database = "data/aether.sqlite3"
[limits]
max_history_messages = 8
max_prompt_characters = 12000
max_autonomy_steps = 5
max_research_results = 8
tool_timeout_seconds = 60
[agents]
parallel_enabled = true
researcher_provider = "ollama"
planner_provider = "fallback_1"
critic_provider = "fallback_2"
verifier_provider = "fallback_3"
max_workers = 3
""",
        encoding="utf-8",
    )
    settings = load_settings(config)
    assert settings.agents.parallel_enabled is True
    assert settings.agents.verifier_provider == "fallback_3"


def test_parallel_agent_routes_reject_duplicate_provider_keys(tmp_path):
    source = Path("config/settings.toml").read_text(encoding="utf-8")
    source = source.replace(
        'parallel_enabled = false', 'parallel_enabled = true'
    ).replace(
        'researcher_provider = ""', 'researcher_provider = "ollama"'
    ).replace(
        'planner_provider = ""', 'planner_provider = "ollama"'
    ).replace(
        'critic_provider = ""', 'critic_provider = "fallback_1"'
    ).replace(
        'verifier_provider = ""', 'verifier_provider = "fallback_2"'
    )
    config = tmp_path / "settings.toml"
    config.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="four distinct"):
        load_settings(config)
