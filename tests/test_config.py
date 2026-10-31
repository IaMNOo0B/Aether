from pathlib import Path

from aether.config import load_settings


def test_load_settings_from_project_config():
    settings = load_settings(Path("config/settings.toml"))
    assert settings.app.name == "AETHER"
    assert settings.models.provider == "ollama"
    assert settings.models.enabled is False
