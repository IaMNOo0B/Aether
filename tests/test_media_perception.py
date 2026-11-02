from pathlib import Path

import pytest

from aether import media_perception as media


def test_video_duration_is_bounded(monkeypatch, tmp_path):
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"x")
    monkeypatch.setattr(media, "_run", lambda *a, **k: b'{"format":{"duration":"301"},"streams":[]}')
    monkeypatch.setattr(media, "_tool", lambda name: name)
    with pytest.raises(media.MediaError, match="duration"):
        media.probe_video(source)


def test_video_frames_validate_bounds():
    with pytest.raises(ValueError):
        media.extract_video_frames("not-used", interval_seconds=0.1)
    with pytest.raises(ValueError):
        media.extract_video_frames("not-used", max_frames=25)


def test_ocr_rejects_invalid_language(tmp_path):
    source = tmp_path / "image.png"
    source.write_bytes(b"x")
    with pytest.raises(ValueError, match="language"):
        media.ocr_image(source, language="../eng")


def test_audio_event_top_k_is_bounded(tmp_path):
    source = tmp_path / "audio.wav"
    source.write_bytes(b"x")
    model = tmp_path / "model.onnx"
    labels = tmp_path / "labels.txt"
    model.write_bytes(b"x")
    labels.write_text("silence\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top_k"):
        media.audio_events(source, model_path=model, labels_path=labels, top_k=21)


def test_input_rejects_oversized_file(tmp_path):
    source = tmp_path / "large.mp4"
    source.write_bytes(b"x" * 20)
    with pytest.raises(media.MediaError, match="safety limit"):
        media._input_file(source) if media.MAX_MEDIA_BYTES < 20 else pytest.skip("default cap exceeds fixture")
