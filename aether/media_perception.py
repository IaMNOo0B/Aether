"""Optional, bounded video/OCR/audio-event perception for Phase 5.

No media is uploaded. External executables and model files are operator-installed.
Video frames are sampled with FFmpeg; OCR uses Tesseract; audio events use a
locally supplied YAMNet-compatible ONNX model and label map.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Sequence

MAX_MEDIA_BYTES = 512 * 1024 * 1024
MAX_DURATION_SECONDS = 300.0
MAX_FRAMES = 24
MAX_AUDIO_SECONDS = 30.0
SAMPLE_RATE = 16_000


class MediaError(RuntimeError):
    """A media operation could not be completed safely."""


def _input_file(value: str | Path) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not path.is_file():
        raise MediaError("Media input must be a regular file")
    if path.stat().st_size < 1 or path.stat().st_size > MAX_MEDIA_BYTES:
        raise MediaError(f"Media file must be between 1 byte and {MAX_MEDIA_BYTES} bytes")
    return path


def _tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise MediaError(f"Required executable '{name}' was not found on PATH")
    return found


def _run(command: Sequence[str], *, timeout: int = 60, max_output: int = 8_000_000) -> bytes:
    try:
        completed = subprocess.run(
            list(command), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=timeout, check=False,
            shell=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"Media command timed out after {timeout}s") from exc
    except OSError as exc:
        raise MediaError(f"Could not start media command: {exc}") from exc
    if completed.returncode:
        detail = completed.stderr.decode("utf-8", errors="replace")[-1200:]
        raise MediaError(f"Media command failed ({completed.returncode}): {detail}")
    if len(completed.stdout) > max_output:
        raise MediaError("Media command output exceeded the configured safety limit")
    return completed.stdout


def probe_video(path: str | Path) -> dict[str, Any]:
    source = _input_file(path)
    raw = _run([
        _tool("ffprobe"), "-v", "error", "-show_entries",
        "format=duration,format_name,size:stream=codec_type,codec_name,width,height,r_frame_rate",
        "-of", "json", str(source),
    ])
    try:
        info = json.loads(raw)
        duration = float(info.get("format", {}).get("duration", 0))
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise MediaError("ffprobe returned invalid media metadata") from exc
    if not math.isfinite(duration) or duration <= 0 or duration > MAX_DURATION_SECONDS:
        raise MediaError(f"Video duration must be greater than 0 and at most {MAX_DURATION_SECONDS}s")
    return {"path": str(source), "duration_seconds": duration,
            "format": info.get("format", {}), "streams": info.get("streams", [])}


def extract_video_frames(
    path: str | Path, *, interval_seconds: float = 5.0, max_frames: int = 12,
) -> list[Path]:
    if isinstance(interval_seconds, bool) or not isinstance(interval_seconds, (int, float)) or not math.isfinite(interval_seconds) or not 0.5 <= interval_seconds <= 60:
        raise ValueError("interval_seconds must be between 0.5 and 60")
    if isinstance(max_frames, bool) or not isinstance(max_frames, int) or not 1 <= max_frames <= MAX_FRAMES:
        raise ValueError(f"max_frames must be between 1 and {MAX_FRAMES}")
    source = _input_file(path)
    metadata = probe_video(source)
    interval = max(interval_seconds, metadata["duration_seconds"] / max_frames)
    directory = Path(tempfile.mkdtemp(prefix="aether-media-"))
    pattern = directory / "frame-%03d.jpg"
    _run([
        _tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-vf", f"fps=1/{interval:.6f},scale='min(1280,iw)':-2",
        "-frames:v", str(max_frames), "-q:v", "4", str(pattern),
    ], timeout=90, max_output=1024)
    frames = sorted(directory.glob("frame-*.jpg"))
    if not frames:
        shutil.rmtree(directory, ignore_errors=True)
        raise MediaError("FFmpeg did not produce any video frames")
    return frames


def ocr_image(path: str | Path, *, language: str = "eng", timeout: int = 30) -> dict[str, Any]:
    source = _input_file(path)
    if not language or len(language) > 32 or not language.replace("+", "").isalnum():
        raise ValueError("Invalid Tesseract language code")
    raw = _run([_tool("tesseract"), str(source), "stdout", "-l", language],
               timeout=timeout, max_output=1_000_000)
    return {"path": str(source), "language": language,
            "text": raw.decode("utf-8", errors="replace").strip()}


def analyze_video(
    provider: Any, path: str | Path, prompt: str, *,
    interval_seconds: float = 5.0, max_frames: int = 8,
) -> dict[str, Any]:
    """Describe sampled frames with a configured local/consented vision provider."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A non-empty video-analysis prompt is required")
    from aether.multimodal import analyze_image
    frames = extract_video_frames(path, interval_seconds=interval_seconds, max_frames=max_frames)
    try:
        observations = []
        for index, frame in enumerate(frames):
            observations.append({
                "frame": index,
                "observation": analyze_image(provider, frame, prompt.strip()),
            })
        return {"video": str(Path(path).expanduser().resolve()),
                "sampled_frames": len(observations), "observations": observations,
                "warning": "Only sampled frames were analyzed; motion between frames may be missed."}
    finally:
        if frames:
            shutil.rmtree(frames[0].parent, ignore_errors=True)


def ocr_video(
    path: str | Path, *, interval_seconds: float = 5.0,
    max_frames: int = 12, language: str = "eng",
) -> dict[str, Any]:
    frames = extract_video_frames(path, interval_seconds=interval_seconds, max_frames=max_frames)
    try:
        results = []
        for index, frame in enumerate(frames):
            results.append({"frame": index, "file": frame.name, **ocr_image(frame, language=language)})
        return {"video": str(Path(path).expanduser().resolve()), "frames": results}
    finally:
        if frames:
            shutil.rmtree(frames[0].parent, ignore_errors=True)


def audio_events(
    path: str | Path, *, model_path: str | Path, labels_path: str | Path,
    top_k: int = 5,
) -> dict[str, Any]:
    """Run a local YAMNet-compatible ONNX classifier; no model is downloaded."""
    source = _input_file(path)
    model = Path(model_path).expanduser().resolve(strict=True)
    labels_file = Path(labels_path).expanduser().resolve(strict=True)
    if not model.is_file() or not labels_file.is_file():
        raise MediaError("ONNX model and label map must be regular local files")
    if model.stat().st_size > 512 * 1024 * 1024 or labels_file.stat().st_size > 2_000_000:
        raise MediaError("Audio model or label map exceeds the safety limit")
    if isinstance(top_k, bool) or not 1 <= top_k <= 20:
        raise ValueError("top_k must be between 1 and 20")
    decoded = _run([
        _tool("ffmpeg"), "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", str(source), "-t", str(MAX_AUDIO_SECONDS), "-vn", "-ac", "1",
        "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1",
    ], timeout=90, max_output=SAMPLE_RATE * 4 * int(MAX_AUDIO_SECONDS + 1))
    if not decoded or len(decoded) % 4:
        raise MediaError("FFmpeg returned an invalid or empty float32 audio stream")
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise MediaError("Install the optional 'media' dependencies: numpy and onnxruntime") from exc
    waveform = np.frombuffer(decoded, dtype="<f4").copy()
    if not np.isfinite(waveform).all():
        raise MediaError("Decoded audio contains non-finite samples")
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise MediaError("Expected a single-input YAMNet-compatible ONNX model")
    input_shape = inputs[0].shape
    if len(input_shape) == 1:
        tensor = waveform.astype(np.float32, copy=False)
    elif len(input_shape) == 2:
        tensor = waveform.reshape(1, -1).astype(np.float32, copy=False)
    else:
        raise MediaError("Expected a 1-D waveform or 2-D batched waveform input")
    scores = session.run(None, {inputs[0].name: tensor})[0]
    scores = np.asarray(scores)
    if scores.ndim == 2:
        scores = scores.mean(axis=0)
    elif scores.ndim == 1:
        pass
    else:
        raise MediaError("ONNX model returned an unsupported score tensor")
    labels = [line.strip() for line in labels_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(labels) != scores.shape[-1]:
        raise MediaError("Label count does not match the ONNX model output dimension")
    order = np.argsort(scores)[::-1][:top_k]
    predictions = [{"label": labels[int(i)], "score": float(scores[int(i)])} for i in order]
    return {"path": str(source), "model": str(model), "sample_rate": SAMPLE_RATE,
            "analyzed_seconds": len(waveform) / SAMPLE_RATE, "top_events": predictions,
            "warning": "Predictions are model estimates, not verified facts."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded local video/OCR/audio-event perception.")
    sub = parser.add_subparsers(dest="command", required=True)
    video = sub.add_parser("video"); video.add_argument("--input", required=True)
    video.add_argument("--interval", type=float, default=5.0); video.add_argument("--frames", type=int, default=12)
    video.add_argument("--ocr", action="store_true", help="Run Tesseract OCR on sampled frames")
    video.add_argument("--language", default="eng")
    ocr = sub.add_parser("ocr"); ocr.add_argument("--input", required=True); ocr.add_argument("--language", default="eng")
    events = sub.add_parser("audio-events"); events.add_argument("--input", required=True)
    events.add_argument("--model", required=True); events.add_argument("--labels", required=True)
    events.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args(argv)
    if args.command == "video":
        result = probe_video(args.input)
        if args.ocr:
            result["ocr"] = ocr_video(args.input, interval_seconds=args.interval,
                                      max_frames=args.frames, language=args.language)
        else:
            frames = extract_video_frames(args.input, interval_seconds=args.interval,
                                          max_frames=args.frames)
            try:
                result["sampled_frames"] = len(frames)
                result["frame_files"] = [frame.name for frame in frames]
            finally:
                if frames:
                    shutil.rmtree(frames[0].parent, ignore_errors=True)
            result["note"] = "Use the Python analyze_video(provider, ...) API for frame-level vision-model descriptions."
    elif args.command == "ocr":
        result = ocr_image(args.input, language=args.language)
    else:
        result = audio_events(args.input, model_path=args.model, labels_path=args.labels, top_k=args.top_k)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
