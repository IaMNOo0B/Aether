"""Optional local Ollama embedding adapter for semantic memory."""
from __future__ import annotations

from math import isfinite
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json


class EmbeddingUnavailable(RuntimeError):
    pass


class EmbeddingProvider(Protocol):
    model: str

    def embed(self, text: str) -> list[float]: ...


class OllamaEmbeddingProvider:
    """Use Ollama's /api/embed, with compatibility fallback to /api/embeddings."""

    def __init__(
        self, base_url: str, model: str, timeout_seconds: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def _post(self, path: str, payload: dict) -> dict:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    def embed(self, text: str) -> list[float]:
        value = text.strip()
        if not value:
            raise ValueError("Cannot embed empty text")
        try:
            try:
                result = self._post("/api/embed", {"model": self.model, "input": value})
                if not isinstance(result, dict):
                    raise EmbeddingUnavailable("Ollama embedding response must be a JSON object")
                vectors = result.get("embeddings")
                if isinstance(vectors, list) and vectors:
                    vector = vectors[0] if isinstance(vectors[0], list) else vectors
                else:
                    if not isinstance(result, dict):
                    raise EmbeddingUnavailable("Ollama embedding response must be a JSON object")
                vector = result.get("embedding")
            except HTTPError as exc:
                if exc.code != 404:
                    raise
                result = self._post(
                    "/api/embeddings", {"model": self.model, "prompt": value}
                )
                vector = result.get("embedding")
        except HTTPError as exc:
            try:
                detail = exc.read(1000).decode("utf-8", errors="replace")
            except OSError:
                detail = "response body unavailable"
            raise EmbeddingUnavailable(
                f"Ollama embedding endpoint returned HTTP {exc.code}: {detail[:1000]}"
            ) from exc
        except (OSError, URLError, TimeoutError) as exc:
            raise EmbeddingUnavailable(
                f"Ollama embedding request failed: {type(exc).__name__}"
            ) from exc
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
            raise EmbeddingUnavailable(
                f"Ollama embedding endpoint returned invalid JSON: {type(exc).__name__}"
            ) from exc

        if not isinstance(vector, list) or not vector:
            raise EmbeddingUnavailable("Ollama returned no embedding vector")
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float))
            or not isfinite(float(item))
            for item in vector
        ):
            raise EmbeddingUnavailable("Ollama returned a non-numeric embedding vector")
        return [float(item) for item in vector]
