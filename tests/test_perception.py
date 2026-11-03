from pathlib import Path
import subprocess

import pytest

from aether.perception import FFmpegMediaProcessor, OpenAICompatibleVisionProvider


def test_vision_adapter_rejects_invalid_image_and_remote_http():
    with pytest.raises(ValueError, match="HTTPS"):
        OpenAICompatibleVisionProvider("http://example.com/v1", "vision-model")
    provider = OpenAICompatibleVisionProvider("https://api.example.com/v1", "vision-model")
    with pytest.raises(ValueError, match="PNG, JPEG and WebP"):
        provider.analyze_image(b"image", "image/gif", "Describe this")
    with pytest.raises(ValueError, match="MIME type"):
        provider.analyze_image(b"not-png", "image/png", "Describe this")
    with pytest.raises(ValueError, match="size limit"):
        OpenAICompatibleVisionProvider(
            "https://api.example.com/v1", "vision-model", max_image_bytes=1
        ).analyze_image(b"too large", "image/png", "Describe this")


def test_vision_adapter_uses_local_privacy_metadata_for_loopback():
    provider = OpenAICompatibleVisionProvider(
        "http://127.0.0.1:1234/v1", "local-vision"
    )
    assert provider.privacy == "local"
    assert provider.model_id == "local-vision"


def test_ffmpeg_frame_extraction_is_shell_free_and_bounded(tmp_path, monkeypatch):
    import aether.perception as perception

    source = tmp_path / "video.mp4"
    source.write_bytes(b"fake-media")
    destination = tmp_path / "frames"
    observed = {}

    def fake_run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        destination.mkdir(exist_ok=True)
        (destination / "0001.jpg").write_bytes(b"frame")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(perception.subprocess, "run", fake_run)
    frames = FFmpegMediaProcessor().extract_frames(
        source, destination, interval_seconds=1, max_frames=5
    )
    assert len(frames) == 1
    assert observed["shell"] is False
    assert "-frames:v" in observed["command"]
    assert observed["command"][observed["command"].index("-frames:v") + 1] == "5"


def test_ffmpeg_rejects_unsafe_limits_and_missing_source(tmp_path):
    processor = FFmpegMediaProcessor()
    with pytest.raises(ValueError, match="interval"):
        processor.extract_frames(tmp_path / "missing.mp4", tmp_path / "frames", interval_seconds=0.1)
    with pytest.raises(FileNotFoundError):
        processor.extract_audio(tmp_path / "missing.mp4", tmp_path / "audio.wav")
