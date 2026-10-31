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
