import json

from aether.providers import OpenAICompatibleProvider, ProviderChain, OllamaProvider


def test_ollama_vision_support_is_limited_to_recognized_model_families():
    assert OllamaProvider("http://127.0.0.1:11434", "qwen2.5vl:7b").supports_vision
    assert OllamaProvider("http://127.0.0.1:11434", "llava:7b").supports_vision
    assert not OllamaProvider("http://127.0.0.1:11434", "phi3:mini").supports_vision


def test_provider_chain_routes_images_only_to_vision_capable_provider():
    class TextOnly:
        supports_vision = False

    class Vision:
        supports_vision = True

        def chat_with_images(self, prompt, images):
            assert prompt == "Describe"
            assert images[0]["mime_type"] == "image/png"
            return "Image response"

    chain = ProviderChain([("text-only", TextOnly()), ("vision", Vision())])
    result = chain.chat_with_images(
        "Describe", [{"mime_type": "image/png", "data": "aGVsbG8="}]
    )
    assert result == "Image response"
    assert chain.last_provider == "vision"


def test_openai_compatible_provider_builds_multimodal_chat_payload(monkeypatch):
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"Visible result"}}]}'

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("aether.providers.urlopen", fake_urlopen)
    provider = OpenAICompatibleProvider(
        "http://127.0.0.1:1234/v1", "vision-model"
    )
    result = provider.chat_with_images(
        "Describe this",
        [{"mime_type": "image/png", "data": "aGVsbG8="}],
    )

    assert result == "Visible result"
    content = captured["payload"]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "Describe this"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,aGVsbG8="
