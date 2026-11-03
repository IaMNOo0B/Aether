import pytest

from aether.ollama import ModelUnavailable
from aether.providers import ProviderRegistry, build_registry


class FakeProvider:
    name = "fake"

    def is_available(self):
        return True

    def chat(self, messages):
        return "ok"


def test_registry_registers_and_resolves_provider():
    registry = ProviderRegistry()
    provider = FakeProvider()
    registry.register("FAKE", provider)
    assert registry.get("fake") is provider
    assert registry.names() == ("fake",)


def test_registry_rejects_duplicate_and_unknown_provider():
    registry = ProviderRegistry()
    registry.register("fake", FakeProvider())
    with pytest.raises(ValueError):
        registry.register("fake", FakeProvider())
    with pytest.raises(ModelUnavailable):
        registry.get("missing")


class FailingProvider:
    name = "failing"

    def is_available(self):
        return False

    def chat(self, messages):
        raise ModelUnavailable("provider offline")


def test_provider_chain_falls_back_and_records_successful_provider():
    from aether.providers import ProviderChain

    chain = ProviderChain([
        ("primary", FailingProvider()),
        ("fallback_1", FakeProvider()),
    ])
    assert chain.chat([{"role": "user", "content": "hello"}]) == "ok"
    assert chain.last_provider == "fallback_1"


def test_provider_chain_reports_all_provider_failures():
    from aether.providers import ProviderChain

    chain = ProviderChain([
        ("primary", FailingProvider()),
        ("fallback_1", FailingProvider()),
    ])
    with pytest.raises(ModelUnavailable, match="All configured model providers failed"):
        chain.chat([{"role": "user", "content": "hello"}])


def test_registry_registers_fallback_endpoint():
    from aether.config import ProviderEndpointSettings

    registry = build_registry(
        "http://127.0.0.1:11434", "local-model", 30,
        provider="ollama",
        fallbacks=(ProviderEndpointSettings(
            provider="openai_compatible",
            base_url="http://127.0.0.1:1234/v1",
            model="fallback-model",
            api_key_env="",
        ),),
    )
    assert registry.names() == ("fallback_1", "ollama")


def test_provider_chain_retries_with_bounded_exponential_backoff_before_fallback():
    from aether.providers import ProviderChain

    class FlakyProvider:
        name = "flaky"

        def __init__(self):
            self.calls = 0

        def is_available(self):
            return True

        def chat(self, messages):
            self.calls += 1
            if self.calls < 3:
                raise ModelUnavailable("temporary outage")
            return "recovered"

    delays = []
    flaky = FlakyProvider()
    chain = ProviderChain(
        [("primary", flaky), ("fallback", FakeProvider())],
        max_attempts=3, backoff_seconds=0.2, sleep=delays.append,
    )
    assert chain.chat([{"role": "user", "content": "hello"}]) == "recovered"
    assert flaky.calls == 3
    assert delays == [0.2, 0.4]
    assert chain.last_provider == "primary"
    assert chain.last_attempts["primary"] == 3


def test_provider_chain_validates_retry_limits():
    from aether.providers import ProviderChain

    with pytest.raises(ValueError, match="max_attempts"):
        ProviderChain([("primary", FakeProvider())], max_attempts=0)
    with pytest.raises(ValueError, match="backoff_seconds"):
        ProviderChain([("primary", FakeProvider())], backoff_seconds=11)


def test_model_descriptor_exposes_explicit_capabilities_and_privacy():
    from aether.providers import ModelDescriptor

    descriptor = ModelDescriptor(
        "ollama", "phi3:mini", capabilities=("chat", "reasoning"),
        context_window=4096, privacy="local",
    )
    assert descriptor.supports_vision is False
    assert descriptor.privacy == "local"
    with pytest.raises(ValueError):
        ModelDescriptor("ollama", "phi3:mini", context_window=0)


def test_registry_exposes_model_descriptor_for_each_configured_provider():
    registry = build_registry("http://127.0.0.1:11434", "phi3:mini", 30)
    descriptor = registry.descriptor("ollama")
    assert descriptor.model == "phi3:mini"
    assert descriptor.privacy == "local"
    assert "ollama" in {item.provider for item in registry.descriptors()}


def test_loopback_openai_compatible_model_metadata_is_local():
    registry = build_registry(
        "http://127.0.0.1:1234/v1", "local-vision", 30,
        provider="openai_compatible",
    )
    assert registry.descriptor("openai_compatible").privacy == "local"
