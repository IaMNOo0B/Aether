import base64
import pytest

from aether.multimodal import analyze_image, detect_image_mime, load_image_payload
from aether.ollama import ModelUnavailable


PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def test_detect_supported_image_signatures_and_reject_svg():
    assert detect_image_mime(PNG_HEADER) == "image/png"
    assert detect_image_mime(b"\xff\xd8\xffdata") == "image/jpeg"
    assert detect_image_mime(b"GIF89a-data") == "image/gif"
    assert detect_image_mime(b"RIFFxxxxWEBPdata") == "image/webp"
    with pytest.raises(ValueError, match="SVG is rejected"):
        detect_image_mime(b"<svg></svg>")


def test_load_image_payload_checks_size_and_records_hash(tmp_path):
    path = tmp_path / "sample.png"
    path.write_bytes(PNG_HEADER + b"image-data")
    payload = load_image_payload(path)

    assert payload["mime_type"] == "image/png"
    assert base64.b64decode(payload["data"]) == PNG_HEADER + b"image-data"
    assert len(payload["sha256"]) == 64
    with pytest.raises(ValueError, match="safety limit"):
        load_image_payload(path, max_bytes=4)


def test_image_analysis_requires_provider_capability_and_prompt(tmp_path):
    path = tmp_path / "sample.png"
    path.write_bytes(PNG_HEADER + b"image-data")
    with pytest.raises(ModelUnavailable, match="does not support"):
        analyze_image(object(), path, "Describe this")
    with pytest.raises(ValueError, match="prompt"):
        analyze_image(object(), path, " ")


def test_image_analysis_passes_payload_to_provider(tmp_path):
    path = tmp_path / "sample.png"
    path.write_bytes(PNG_HEADER + b"image-data")

    class FakeProvider:
        def chat_with_images(self, prompt, images):
            assert prompt == "Describe this"
            assert images[0]["mime_type"] == "image/png"
            return "A visible image."

    assert analyze_image(FakeProvider(), path, " Describe this ") == "A visible image."
