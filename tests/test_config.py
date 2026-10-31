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
