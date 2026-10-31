"""Provider contracts and registry for replaceable language-model backends."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from aether.ollama import ModelUnavailable, OllamaClient


class ChatProvider(Protocol):
    """Minimal contract required by the orchestration layer."""

    name: str

    def is_available(self) -> bool: ...
    def chat(self, messages: list[dict[str, str]]) -> str: ...


@dataclass(frozen=True)
class ModelDescriptor:
    provider: str
    model: str
    enabled: bool = True


class ProviderRegistry:
    """Explicit registry; providers are never silently downloaded or substituted."""

    def __init__(self) -> None:
        self._providers: dict[str, ChatProvider] = {}

    def register(self, name: str, provider: ChatProvider) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("Provider name cannot be empty")
        if key in self._providers:
            raise ValueError(f"Provider already registered: {key}")
        self._providers[key] = provider

    def get(self, name: str) -> ChatProvider:
        key = name.strip().lower()
        try:
            return self._providers[key]
        except KeyError as exc:
            raise ModelUnavailable(f"No provider registered for: {key}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))


def build_registry(base_url: str, model: str, timeout_seconds: int) -> ProviderRegistry:
    registry = ProviderRegistry()
    registry.register("ollama", OllamaProvider(base_url, model, timeout_seconds))
    return registry


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout_seconds: int = 120):
        self.model = model
        self.client = OllamaClient(base_url, model, timeout_seconds)

    def is_available(self) -> bool:
        return self.client.is_available()

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self.client.chat(messages)
