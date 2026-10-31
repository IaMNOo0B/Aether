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
class ProviderEndpointSettings:
    provider: str
    base_url: str
    model: str
    timeout_seconds: int = 120
    api_key_env: str = "AETHER_API_KEY"


@dataclass(frozen=True)
class ModelSettings:
    provider: str = "ollama"
    enabled: bool = False
    base_url: str = "http://127.0.0.1:11434"
    model: str = "phi3:mini-4k-q4_0"
    timeout_seconds: int = 120
    api_key_env: str = "AETHER_API_KEY"
    fallbacks: tuple[ProviderEndpointSettings, ...] = ()


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
class MemorySettings:
    semantic_enabled: bool = False
    embedding_provider: str = "ollama"
    embedding_base_url: str = "http://127.0.0.1:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_timeout_seconds: int = 30
    semantic_weight: float = 0.45
    index_batch_size: int = 32


@dataclass(frozen=True)
class SchedulerSettings:
    enabled: bool = False
    poll_interval_seconds: int = 30
    max_tasks_per_tick: int = 1


@dataclass(frozen=True)
@dataclass(frozen=True)
class ResourceSettings:
    enabled: bool = True
    max_cpu_percent: float = 95.0
    max_ram_percent: float = 92.0
    min_free_ram_mb: int = 512
    max_gpu_utilization_percent: float = 98.0
    min_free_vram_mb: int = 512
    require_gpu_telemetry: bool = False
    cpu_sample_seconds: float = 0.1


class Settings:
    app: AppSettings
    models: ModelSettings
    storage: StorageSettings
    limits: LimitSettings
    research: ResearchSettings = ResearchSettings()
    autonomy: AutonomySettings = AutonomySettings()
    desktop: DesktopSettings = DesktopSettings()
    voice: VoiceSettings = VoiceSettings()
    scheduler: SchedulerSettings = SchedulerSettings()
    memory: MemorySettings = MemorySettings()
    resources: ResourceSettings = ResourceSettings()


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
        model_raw = dict(raw.get("models", {}))
        fallback_raw = model_raw.pop("fallbacks", [])
        if not isinstance(fallback_raw, list):
            raise ValueError("models.fallbacks must be an array of provider tables")
        fallbacks = tuple(ProviderEndpointSettings(**item) for item in fallback_raw)
        models = ModelSettings(**model_raw, fallbacks=fallbacks)
        storage = StorageSettings(**raw.get("storage", {}))
        limits = LimitSettings(**raw.get("limits", {}))
        research = ResearchSettings(**raw.get("research", {}))
        autonomy = AutonomySettings(**raw.get("autonomy", {}))
        desktop_raw = dict(raw.get("desktop", {}))
        if "allowed_apps" in desktop_raw:
            desktop_raw["allowed_apps"] = tuple(desktop_raw["allowed_apps"])
        desktop = DesktopSettings(**desktop_raw)
        voice = VoiceSettings(**raw.get("voice", {}))
        scheduler = SchedulerSettings(**raw.get("scheduler", {}))
        memory = MemorySettings(**raw.get("memory", {}))
        resources = ResourceSettings(**raw.get("resources", {}))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unknown or invalid configuration field: {exc}") from exc

    if not app.name.strip():
        raise ValueError("app.name cannot be empty")
    if app.log_level.upper() not in {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}:
        raise ValueError("app.log_level must be CRITICAL, ERROR, WARNING, INFO, or DEBUG")
    if models.provider.strip().lower() not in {"ollama", "openai_compatible"}:
        raise ValueError(f"Unsupported model provider: {models.provider}")
    if not isinstance(models.api_key_env, str):
        raise ValueError("models.api_key_env must be an environment variable name")
    if models.api_key_env and not models.api_key_env.replace("_", "").isalnum():
        raise ValueError("models.api_key_env must contain only letters, numbers and underscores")
    parsed_url = urlparse(models.base_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("models.base_url must be a valid HTTP or HTTPS URL")
    if not models.model.strip() or models.timeout_seconds < 1:
        raise ValueError("Model name and timeout must be valid")

    for index, fallback in enumerate(models.fallbacks, start=1):
        if fallback.provider.strip().lower() not in {"ollama", "openai_compatible"}:
            raise ValueError(f"Unsupported fallback provider #{index}: {fallback.provider}")
        fallback_url = urlparse(fallback.base_url)
        if fallback_url.scheme not in {"http", "https"} or not fallback_url.netloc:
            raise ValueError(f"Fallback provider #{index} has an invalid base_url")
        if not fallback.model.strip() or fallback.timeout_seconds < 1:
            raise ValueError(f"Fallback provider #{index} model and timeout must be valid")
        if (not isinstance(fallback.api_key_env, str)
                or (fallback.api_key_env and not fallback.api_key_env.replace("_", "").isalnum())):
            raise ValueError(f"Fallback provider #{index} has an invalid api_key_env")
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
    if not 0 <= resources.max_cpu_percent <= 100:
        raise ValueError("Resource max_cpu_percent must be between zero and 100")
    if not 0 <= resources.max_ram_percent <= 100:
        raise ValueError("Resource max_ram_percent must be between zero and 100")
    if resources.min_free_ram_mb < 0 or resources.min_free_vram_mb < 0:
        raise ValueError("Resource free-memory limits cannot be negative")
    if not 0 <= resources.max_gpu_utilization_percent <= 100:
        raise ValueError("Resource max_gpu_utilization_percent must be between zero and 100")
    if not 0.05 <= resources.cpu_sample_seconds <= 1.0:
        raise ValueError("Resource cpu_sample_seconds must be between 0.05 and 1.0")
    if memory.embedding_provider.strip().lower() != "ollama":
        raise ValueError(f"Unsupported embedding provider: {memory.embedding_provider}")
    embedding_url = urlparse(memory.embedding_base_url)
    if embedding_url.scheme not in {"http", "https"} or not embedding_url.netloc:
        raise ValueError("Memory embedding_base_url must be an absolute HTTP or HTTPS URL")
    if not memory.embedding_model.strip() or memory.embedding_timeout_seconds < 1:
        raise ValueError("Memory embedding model and timeout must be valid")
    if not 0 <= memory.semantic_weight <= 1:
        raise ValueError("Memory semantic_weight must be between zero and one")
    if not 1 <= memory.index_batch_size <= 1000:
        raise ValueError("Memory index_batch_size must be between 1 and 1000")
    if scheduler.poll_interval_seconds < 5 or not 1 <= scheduler.max_tasks_per_tick <= 10:
        raise ValueError("Scheduler poll interval must be >= 5 seconds and max_tasks_per_tick 1..10")
    if scheduler.enabled and (not autonomy.enabled or not models.enabled):
        raise ValueError("Scheduler requires both autonomy.enabled and models.enabled")
    if not voice.wake_word.strip() or not voice.vosk_model_path.strip():
        raise ValueError("Voice wake word and Vosk model path cannot be empty")

    return Settings(
        app=app, models=models, storage=storage, limits=limits,
        research=research, autonomy=autonomy, desktop=desktop, voice=voice,
        scheduler=scheduler, memory=memory, resources=resources,
    )
