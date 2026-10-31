"""Validated application configuration using Python's standard library."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
import tomllib


@dataclass(frozen=True)
class AppSettings:
    name: str = "AETHER"
    environment: str = "local"
    log_level: str = "INFO"


@dataclass(frozen=True)
class ModelSettings:
    provider: str = "ollama"
    enabled: bool = False
    base_url: str = "http://127.0.0.1:11434"
    model: str = "phi3:mini-4k-q4_0"
    timeout_seconds: int = 120


@dataclass(frozen=True)
class StorageSettings:
    database: str = "data/aether.sqlite3"


@dataclass(frozen=True)
class LimitSettings:
    max_history_messages: int = 8
    max_prompt_characters: int = 12000


@dataclass(frozen=True)
class Settings:
    app: AppSettings
    models: ModelSettings
    storage: StorageSettings
    limits: LimitSettings


def load_settings(path: Path | str = "config/settings.toml") -> Settings:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration not found: {config_path}. Run from the AETHER project folder."
        )

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    try:
        app = AppSettings(**raw.get("app", {}))
        models = ModelSettings(**raw.get("models", {}))
        storage = StorageSettings(**raw.get("storage", {}))
        limits = LimitSettings(**raw.get("limits", {}))
    except TypeError as exc:
        raise ValueError(f"Unknown or invalid configuration field: {exc}") from exc

    if not app.name.strip():
        raise ValueError("app.name cannot be empty")
    if app.log_level.upper() not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        raise ValueError("app.log_level must be CRITICAL, ERROR, WARNING, INFO, or DEBUG")
    if models.provider.strip().lower() != "ollama":
        raise ValueError(f"Unsupported model provider: {models.provider}")
    parsed_url = urlparse(models.base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("models.base_url must be a valid HTTP or HTTPS URL")
    if not models.model.strip():
        raise ValueError("models.model cannot be empty")
    if models.timeout_seconds < 1:
        raise ValueError("models.timeout_seconds must be positive")
    if limits.max_history_messages < 1 or limits.max_prompt_characters < 256:
        raise ValueError("History and prompt limits are too small")
    if not storage.database.strip():
        raise ValueError("storage.database cannot be empty")

    return Settings(app=app, models=models, storage=storage, limits=limits)
