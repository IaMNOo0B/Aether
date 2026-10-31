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
    max_autonomy_steps: int = 5
    max_research_results: int = 8
    tool_timeout_seconds: int = 60


@dataclass(frozen=True)
class ResearchSettings:
    enabled: bool = True
    timeout_seconds: int = 15
    max_pages: int = 5


@dataclass(frozen=True)
class AutonomySettings:
    enabled: bool = False
    allow_read_only: bool = True
    allow_workspace_writes: bool = False
    allow_code_execution: bool = False
    workspace: str = "workspace"


@dataclass(frozen=True)
class DesktopSettings:
    enabled: bool = False
    allow_input: bool = False
    allowed_apps: tuple[str, ...] = ("notepad.exe", "calc.exe")


@dataclass(frozen=True)
class VoiceSettings:
    enabled: bool = False
    wake_word: str = "jarvis"
    vosk_model_path: str = "models/vosk"
    speak_responses: bool = True


@dataclass(frozen=True)
class Settings:
    app: AppSettings
    models: ModelSettings
    storage: StorageSettings
    limits: LimitSettings
    research: ResearchSettings = ResearchSettings()
    autonomy: AutonomySettings = AutonomySettings()
    desktop: DesktopSettings = DesktopSettings()
    voice: VoiceSettings = VoiceSettings()


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
        research = ResearchSettings(**raw.get("research", {}))
        autonomy = AutonomySettings(**raw.get("autonomy", {}))
        desktop_raw = dict(raw.get("desktop", {}))
        if "allowed_apps" in desktop_raw:
            desktop_raw["allowed_apps"] = tuple(desktop_raw["allowed_apps"])
        desktop = DesktopSettings(**desktop_raw)
        voice = VoiceSettings(**raw.get("voice", {}))
    except (TypeError, ValueError) as exc:
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
    if not models.model.strip() or models.timeout_seconds < 1:
        raise ValueError("Model name and timeout must be valid")
    if (limits.max_history_messages < 1 or limits.max_prompt_characters < 256
            or limits.max_autonomy_steps < 1 or limits.max_research_results < 1
            or limits.tool_timeout_seconds < 1):
        raise ValueError("Configured limits must be positive and prompt budget at least 256")
    if not storage.database.strip():
        raise ValueError("storage.database cannot be empty")
    if research.timeout_seconds < 1 or research.max_pages < 1:
        raise ValueError("Research limits must be positive")
    if not autonomy.workspace.strip():
        raise ValueError("autonomy.workspace cannot be empty")
    if not desktop.allowed_apps:
        raise ValueError("desktop.allowed_apps must contain at least one application")
    if not voice.wake_word.strip() or not voice.vosk_model_path.strip():
        raise ValueError("Voice wake word and Vosk model path cannot be empty")

    return Settings(
        app=app, models=models, storage=storage, limits=limits,
        research=research, autonomy=autonomy, desktop=desktop, voice=voice,
    )
