"""Optional multimodal adapters with bounded media processing.

No model or binary is downloaded automatically. OCR, speech recognition, FFmpeg,
and vision-model inference are explicit optional integrations. Energy segmentation
is not semantic audio-event recognition.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import json
import shutil
import subprocess
import wave


MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_AUDIO_BYTES = 100 * 1024 * 1024
MAX_VIDEO_BYTES = 500 * 1024 * 1024


class VisionProvider(Protocol):
    def analyze_image(self, image_path: str, prompt: str) -> str: ...


@dataclass(frozen=True)
class MediaResult:
    modality: str
    status: str
    data: dict[str, Any]
    limitations: tuple[str, ...] = ()


def _bounded_file(path: str | Path, maximum: int) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size > maximum:
        raise ValueError(f"Media file exceeds the configured {maximum}-byte limit")
    return source


class ImagePerception:
    def __init__(self, vision_provider: VisionProvider | None = None):
        self.vision_provider = vision_provider

    def analyze(self, image_path: str | Path, prompt: str = "Describe the visible content.") -> MediaResult:
        source = _bounded_file(image_path, MAX_IMAGE_BYTES)
        suffix = source.suffix.casefold()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            raise ValueError("Unsupported image type")
        if self.vision_provider is not None:
            output = self.vision_provider.analyze_image(str(source), prompt[:1000])
            return MediaResult("image", "analyzed", {"provider_output": output},
                               ("Vision output is model-generated and must be independently checked.",))
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Install the optional vision extra with Pillow and pytesseract, plus the Tesseract executable.") from exc
        with Image.open(source) as image:
            text = pytesseract.image_to_string(image)
            width, height = image.size
        return MediaResult("image", "ocr_only", {"width": width, "height": height, "text": text},
                           ("OCR extracts text; it does not perform general visual understanding.",))


class AudioPerception:
    def __init__(self, vosk_model_path: str | Path | None = None):
        self.vosk_model_path = Path(vosk_model_path) if vosk_model_path else None

    def inspect_wav(self, audio_path: str | Path) -> MediaResult:
        source = _bounded_file(audio_path, MAX_AUDIO_BYTES)
        if source.suffix.casefold() != ".wav":
            raise ValueError("Audio inspection currently accepts WAV files only")
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.getnframes()
            duration = frames / max(rate, 1)
        return MediaResult("audio", "metadata_only", {
            "sample_rate": rate, "channels": channels, "sample_width_bytes": sample_width,
            "frames": frames, "duration_seconds": duration,
        }, ("Metadata inspection is not speech recognition or semantic event classification.",))

    def transcribe_wav(self, audio_path: str | Path) -> MediaResult:
        source = _bounded_file(audio_path, MAX_AUDIO_BYTES)
        if self.vosk_model_path is None or not self.vosk_model_path.is_dir():
            raise RuntimeError("A local Vosk model directory must be configured before transcription")
        try:
            import vosk
        except ImportError as exc:
            raise RuntimeError("Install the optional voice dependencies before transcription") from exc
        with wave.open(str(source), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise ValueError("Vosk input must be mono 16-bit PCM WAV")
            recognizer = vosk.KaldiRecognizer(vosk.Model(str(self.vosk_model_path)), handle.getframerate())
            chunks = []
            while True:
                data = handle.readframes(4000)
                if not data:
                    break
                if recognizer.AcceptWaveform(data):
                    value = json.loads(recognizer.Result())
                    if value.get("text"):
                        chunks.append(value["text"])
            final = json.loads(recognizer.FinalResult())
            if final.get("text"):
                chunks.append(final["text"])
        return MediaResult("audio", "transcribed", {"transcript": " ".join(chunks)},
                           ("Speech transcription does not identify non-speech audio events.",))


class VideoPerception:
    def __init__(self, *, max_frames: int = 24, timeout_seconds: int = 90):
        if not 1 <= max_frames <= 120 or timeout_seconds < 1:
            raise ValueError("Video processing limits are invalid")
        self.max_frames = max_frames
        self.timeout_seconds = timeout_seconds

    def extract_frames(self, video_path: str | Path, output_dir: str | Path) -> MediaResult:
        source = _bounded_file(video_path, MAX_VIDEO_BYTES)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg is not installed; no binary will be downloaded automatically")
        destination = Path(output_dir).expanduser().resolve()
        destination.mkdir(parents=True, exist_ok=True)
        # Extract a bounded sample; -frames:v caps total frames regardless of duration.
        pattern = destination / "frame-%04d.jpg"
        command = [
            ffmpeg, "-nostdin", "-v", "error", "-i", str(source),
            "-vf", f"fps=1/5,scale='min(1280,iw)':-2",
            "-frames:v", str(self.max_frames), "-q:v", "4", str(pattern),
        ]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=self.timeout_seconds, check=False, shell=False)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Video frame extraction timed out") from exc
        if result.returncode:
            raise RuntimeError(f"FFmpeg frame extraction failed: {result.stderr[-2000:]}")
        frames = sorted(destination.glob("frame-*.jpg"))
        return MediaResult("video", "frames_extracted", {
            "frame_count": len(frames), "frames": [str(path) for path in frames],
        }, ("Frame extraction is not scene understanding, temporal reasoning, or audio-video alignment.",))


class PerceptionSuite:
    def __init__(self, *, vision_provider: VisionProvider | None = None,
                 vosk_model_path: str | Path | None = None):
        self.image = ImagePerception(vision_provider)
        self.audio = AudioPerception(vosk_model_path)
        self.video = VideoPerception()

    def capabilities(self) -> dict[str, Any]:
        return {
            "image": {"vision_model": self.image.vision_provider is not None,
                      "ocr": self._available("pytesseract") and self._available("PIL")},
            "audio": {"vosk": self._available("vosk"),
                      "configured_model": bool(self.audio.vosk_model_path and self.audio.vosk_model_path.is_dir())},
            "video": {"ffmpeg": shutil.which("ffmpeg") is not None,
                      "max_frames": self.video.max_frames},
            "limitations": [
                "Audio event understanding requires a separately validated event-classification model.",
                "Video understanding requires a vision provider to interpret extracted frames.",
                "No optional dependency or model is downloaded automatically.",
            ],
        }

    @staticmethod
    def _available(module: str) -> bool:
        try:
            __import__(module)
            return True
        except ImportError:
            return False
