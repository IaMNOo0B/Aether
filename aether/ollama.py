"""Small Ollama chat client. No model downloads or background installs are triggered."""
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ModelUnavailable(RuntimeError):
    pass


class TransientModelError(ModelUnavailable):
    """Provider failure that may succeed when retried after a short delay."""



class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout_seconds: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def is_available(self) -> bool:
        request = Request(f"{self.base_url}/api/tags", method="GET")
        try:
            with urlopen(request, timeout=3) as response:
                return response.status == 200
        except (OSError, URLError, HTTPError):
            return False

    def chat(self, messages: list[dict[str, str]]) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
        }).encode("utf-8")
        request = Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            error_type = TransientModelError if exc.code == 429 or 500 <= exc.code < 600 else ModelUnavailable
            raise error_type(f"Ollama returned HTTP {exc.code}") from exc
        except (OSError, URLError, TimeoutError) as exc:
            raise TransientModelError(f"Ollama request failed: {type(exc).__name__}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ModelUnavailable(f"Ollama returned an invalid response: {exc}") from exc

        message = result.get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ModelUnavailable("Ollama returned an empty assistant response")
        return content.strip()
