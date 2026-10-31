import pytest

from aether.ollama import ModelUnavailable
from aether.providers import ProviderRegistry


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
