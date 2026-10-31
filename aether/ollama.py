"""Small Ollama chat client. No model downloads or background installs are triggered."""
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ModelUnavailable(RuntimeError):
    pass


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
        except (OSError, URLError, HTTPError, TimeoutError) as exc:
            raise ModelUnavailable(f"Ollama request failed: {exc}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
            raise ModelUnavailable(f"Ollama returned an invalid response: {exc}") from exc

        message = result.get("message", {})
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ModelUnavailable("Ollama returned an empty assistant response")
        return content.strip()
