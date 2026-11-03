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


def test_provider_chain_retries_only_transient_failures_with_bounded_backoff():
    from aether.ollama import TransientModelError
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
                raise TransientModelError("temporary overload")
            return "recovered"

    delays = []
    provider = FlakyProvider()
    chain = ProviderChain(
        [("primary", provider)], max_retries=2,
        backoff_seconds=0.1, sleeper=delays.append,
    )
    assert chain.chat([{"role": "user", "content": "hello"}]) == "recovered"
    assert provider.calls == 3
    assert delays == [0.1, 0.2]


def test_provider_chain_does_not_retry_permanent_errors():
    from aether.providers import ProviderChain

    class PermanentFailure(FailingProvider):
        def __init__(self):
            self.calls = 0

        def chat(self, messages):
            self.calls += 1
            raise ModelUnavailable("invalid model name")

    provider = PermanentFailure()
    chain = ProviderChain(
        [("primary", provider)], max_retries=3, sleeper=lambda delay: None,
    )
    with pytest.raises(ModelUnavailable):
        chain.chat([{"role": "user", "content": "hello"}])
    assert provider.calls == 1
