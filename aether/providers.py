"""Provider contracts and replaceable language-model backends."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str, model: str, timeout_seconds: int = 120):
        self.model = model
        self.client = OllamaClient(base_url, model, timeout_seconds)

    def is_available(self) -> bool:
        return self.client.is_available()

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self.client.chat(messages)


class OpenAICompatibleProvider:
    """OpenAI Chat Completions-compatible API adapter.

    Secrets are read from the process environment at call time and are never
    persisted in settings or included in raised error messages.
    """

    name = "openai_compatible"

    def __init__(
        self, base_url: str, model: str, timeout_seconds: int = 120,
        api_key_env: str = "AETHER_API_KEY",
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.api_key_env = api_key_env

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env, "").strip() if self.api_key_env else ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def is_available(self) -> bool:
        request = Request(
            f"{self.base_url}/models", headers=self._headers(), method="GET"
        )
        try:
            with urlopen(request, timeout=min(self.timeout_seconds, 5)) as response:
                return 200 <= response.status < 300
        except (OSError, URLError, HTTPError, TimeoutError):
            return False

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
        }).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers=self._headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            # Do not echo request headers or API keys into logs/errors.
            try:
                detail = exc.read(1500).decode("utf-8", errors="replace")
            except OSError:
                detail = "response body unavailable"
            raise ModelUnavailable(
                f"OpenAI-compatible provider returned HTTP {exc.code}: {detail[:1500]}"
            ) from exc
        except (OSError, URLError, TimeoutError) as exc:
            raise ModelUnavailable(
                f"OpenAI-compatible provider request failed: {type(exc).__name__}"
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise ModelUnavailable(
                f"OpenAI-compatible provider returned invalid JSON: {type(exc).__name__}"
            ) from exc

        try:
            choices = result["choices"]
            message = choices[0]["message"]
            response_content = message["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelUnavailable("OpenAI-compatible provider response has no assistant message") from exc

        if isinstance(response_content, list):
            response_content = "".join(
                item.get("text", "") for item in response_content if isinstance(item, dict)
            )
        if not isinstance(response_content, str) or not response_content.strip():
            raise ModelUnavailable("OpenAI-compatible provider returned an empty assistant response")
        return response_content.strip()


def build_registry(
    base_url: str, model: str, timeout_seconds: int,
    provider: str = "ollama", api_key_env: str = "AETHER_API_KEY",
) -> ProviderRegistry:
    registry = ProviderRegistry()
    selected = provider.strip().lower()
    if selected == "ollama":
        registry.register("ollama", OllamaProvider(base_url, model, timeout_seconds))
    elif selected == "openai_compatible":
        registry.register(
            "openai_compatible",
            OpenAICompatibleProvider(base_url, model, timeout_seconds, api_key_env),
        )
    else:
        raise ModelUnavailable(f"Unsupported model provider: {selected}")
    return registry
