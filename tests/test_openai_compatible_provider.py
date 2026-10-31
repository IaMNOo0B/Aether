import json

import pytest

from aether.ollama import ModelUnavailable
from aether.providers import OpenAICompatibleProvider, build_registry


class FakeResponse:
    status = 200

    def __init__(self, payload):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return self.payload


def test_openai_compatible_provider_sends_model_messages_and_bearer_key(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = dict(request.header_items())
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return FakeResponse({"choices": [{"message": {"content": "  response  "}}]})

    monkeypatch.setenv("AETHER_TEST_KEY", "secret-test-value")
    monkeypatch.setattr("aether.providers.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        "https://example.invalid/v1", "test-model", 17, "AETHER_TEST_KEY"
    )

    answer = provider.chat([{"role": "user", "content": "hello"}])

    assert answer == "response"
    assert captured["url"] == "https://example.invalid/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-test-value"
    assert captured["payload"]["model"] == "test-model"
    assert captured["payload"]["messages"] == [{"role": "user", "content": "hello"}]
    assert captured["timeout"] == 17


def test_openai_compatible_provider_does_not_require_key_for_local_endpoints(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured.update(dict(request.header_items()))
        return FakeResponse({"choices": [{"message": {"content": "local"}}]})

    monkeypatch.delenv("AETHER_TEST_KEY", raising=False)
    monkeypatch.setattr("aether.providers.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        "http://127.0.0.1:1234/v1", "local-model", 10, "AETHER_TEST_KEY"
    )

    assert provider.chat([{"role": "user", "content": "hello"}]) == "local"
    assert "Authorization" not in captured


def test_openai_compatible_provider_rejects_empty_response(monkeypatch):
    monkeypatch.setattr(
        "aether.providers.urlopen",
        lambda *args, **kwargs: FakeResponse({"choices": [{"message": {"content": ""}}]}),
    )
    provider = OpenAICompatibleProvider("http://localhost:1234/v1", "model")
    with pytest.raises(ModelUnavailable, match="empty assistant response"):
        provider.chat([{"role": "user", "content": "hello"}])


def test_registry_builds_configured_openai_compatible_provider():
    registry = build_registry(
        "http://127.0.0.1:1234/v1", "local-model", 30,
        provider="openai_compatible", api_key_env="",
    )
    assert registry.names() == ("openai_compatible",)
    assert registry.get("openai_compatible").model == "local-model"


def test_registry_rejects_unsupported_provider():
    with pytest.raises(ModelUnavailable, match="Unsupported model provider"):
        build_registry("http://localhost", "model", 30, provider="unknown")
