"""Validated application configuration using Python's standard library."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    model: str = "qwen2.5:7b-instruct-q4_K_M"
    timeout_seconds: int = 120


@dataclass(frozen=True)
class StorageSettings:
    database: str = "data/aether.sqlite3"


@dataclass(frozen=True)
class LimitSettings:
    max_history_messages: int = 12
    max_prompt_characters: int = 24000


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

    app = AppSettings(**raw.get("app", {}))
    models = ModelSettings(**raw.get("models", {}))
    storage = StorageSettings(**raw.get("storage", {}))
    limits = LimitSettings(**raw.get("limits", {}))

    if models.timeout_seconds < 1:
        raise ValueError("models.timeout_seconds must be positive")
    if limits.max_history_messages < 1 or limits.max_prompt_characters < 100:
        raise ValueError("History and prompt limits are too small")

    return Settings(app=app, models=models, storage=storage, limits=limits)
