"""Provider contracts and replaceable language-model backends."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from aether.config import ProviderEndpointSettings
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

    @property
    def supports_vision(self) -> bool:
        model = self.model.casefold()
        return any(token in model for token in (
            "llava", "llama3.2-vision", "qwen2.5vl", "qwen2.5-vl",
            "qwen2-vl", "qwen3-vl", "gemma3", "moondream", "minicpm-v",
        ))

    def chat_with_images(self, prompt: str, images: list[dict[str, str]]) -> str:
        if not self.supports_vision:
            raise ModelUnavailable(
                f"Configured Ollama model '{self.model}' is not recognized as a vision model"
            )
        return self.client.chat_with_images(
            prompt, [item["data"] for item in images]
        )


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

    @property
    def supports_vision(self) -> bool:
        # OpenAI-compatible servers may expose vision-capable models; the endpoint
        # remains the authority and will reject unsupported model/payload pairs.
        return True

    def chat_with_images(self, prompt: str, images: list[dict[str, str]]) -> str:
        if not prompt.strip() or not images:
            raise ValueError("Image analysis requires a prompt and at least one image")
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for item in images:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{item['mime_type']};base64,{item['data']}"
                },
            })
        return self._request_chat([{"role": "user", "content": content}])

    def chat(self, messages: list[dict[str, str]]) -> str:
        return self._request_chat(messages)

    def _request_chat(self, messages: list[dict[str, Any]]) -> str:
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
            response_content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelUnavailable("OpenAI-compatible provider response has no assistant message") from exc
        if isinstance(response_content, list):
            response_content = "".join(
                item.get("text", "") for item in response_content if isinstance(item, dict)
            )
        if not isinstance(response_content, str) or not response_content.strip():
            raise ModelUnavailable("OpenAI-compatible provider returned an empty assistant response")
        return response_content.strip()


class ProviderChain:
    """Ordered provider failover; only transport/provider failures trigger fallback."""

    name = "provider_chain"

    def __init__(self, providers: list[tuple[str, ChatProvider]]):
        if not providers:
            raise ValueError("Provider chain requires at least one configured provider")
        self.providers = tuple(providers)
        self.last_provider: str | None = None

    def is_available(self) -> bool:
        for _, provider in self.providers:
            try:
                if provider.is_available():
                    return True
            except Exception:
                continue
        return False

    def chat(self, messages: list[dict[str, str]]) -> str:
        failures = []
        for label, provider in self.providers:
            try:
                answer = provider.chat(messages)
                self.last_provider = label
                return answer
            except ModelUnavailable as exc:
                failures.append(f"{label}: {str(exc)[:400]}")
        self.last_provider = None
        raise ModelUnavailable(
            "All configured model providers failed. " + " | ".join(failures)
        )

    def chat_with_images(self, prompt: str, images: list[dict[str, str]]) -> str:
        failures = []
        for label, provider in self.providers:
            if not getattr(provider, "supports_vision", False):
                continue
            method = getattr(provider, "chat_with_images", None)
            if not callable(method):
                continue
            try:
                answer = method(prompt, images)
                self.last_provider = label
                return answer
            except ModelUnavailable as exc:
                failures.append(f"{label}: {str(exc)[:400]}")
        self.last_provider = None
        detail = " | ".join(failures) if failures else "No configured provider advertises image support"
        raise ModelUnavailable("Image analysis unavailable. " + detail)


def _make_provider(
    provider: str, base_url: str, model: str, timeout_seconds: int, api_key_env: str,
) -> ChatProvider:
    selected = provider.strip().lower()
    if selected == "ollama":
        return OllamaProvider(base_url, model, timeout_seconds)
    if selected == "openai_compatible":
        return OpenAICompatibleProvider(base_url, model, timeout_seconds, api_key_env)
    raise ModelUnavailable(f"Unsupported model provider: {selected}")


def build_registry(
    base_url: str, model: str, timeout_seconds: int,
    provider: str = "ollama", api_key_env: str = "AETHER_API_KEY",
    fallbacks: tuple[ProviderEndpointSettings, ...] = (),
) -> ProviderRegistry:
    registry = ProviderRegistry()
    selected = provider.strip().lower()
    registry.register(
        selected, _make_provider(selected, base_url, model, timeout_seconds, api_key_env)
    )
    for index, fallback in enumerate(fallbacks, start=1):
        registry.register(
            f"fallback_{index}",
            _make_provider(
                fallback.provider, fallback.base_url, fallback.model,
                fallback.timeout_seconds, fallback.api_key_env,
            ),
        )
    return registry
