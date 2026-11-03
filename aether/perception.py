"""Bounded optional image/audio/video adapters; no model or binary downloads."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import json, shutil, subprocess, wave

MAX_IMAGE_BYTES, MAX_AUDIO_BYTES, MAX_VIDEO_BYTES = 20*1024*1024, 100*1024*1024, 500*1024*1024


class VisionProvider(Protocol):
    def analyze_image(self, image_path: str, prompt: str) -> str: ...


@dataclass(frozen=True)
class MediaResult:
    modality: str
    status: str
    data: dict[str, Any]
    limitations: tuple[str, ...] = ()


def _bounded(path: str | Path, maximum: int) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.stat().st_size > maximum:
        raise ValueError(f"Media file exceeds {maximum}-byte limit")
    return source


class ImagePerception:
    def __init__(self, vision_provider: VisionProvider | None = None):
        self.vision_provider = vision_provider

    def analyze(self, path: str | Path, prompt: str = "Describe the visible content.") -> MediaResult:
        source = _bounded(path, MAX_IMAGE_BYTES)
        if source.suffix.casefold() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            raise ValueError("Unsupported image type")
        if self.vision_provider:
            return MediaResult("image", "analyzed",
                {"provider_output": self.vision_provider.analyze_image(str(source), prompt[:1000])},
                ("Vision output is model-generated and needs independent verification.",))
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Install Pillow and pytesseract plus the Tesseract executable for OCR.") from exc
        with Image.open(source) as image:
            text = pytesseract.image_to_string(image)
            width, height = image.size
        return MediaResult("image", "ocr_only", {"width": width, "height": height, "text": text},
                           ("OCR is not general visual understanding.",))


class AudioPerception:
    def __init__(self, vosk_model_path: str | Path | None = None):
        self.vosk_model_path = Path(vosk_model_path) if vosk_model_path else None

    def inspect_wav(self, path: str | Path) -> MediaResult:
        source = _bounded(path, MAX_AUDIO_BYTES)
        if source.suffix.casefold() != ".wav":
            raise ValueError("Audio inspection currently accepts WAV files only")
        with wave.open(str(source), "rb") as handle:
            channels, width, rate, frames = handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()
        return MediaResult("audio", "metadata_only", {
            "sample_rate": rate, "channels": channels, "sample_width_bytes": width,
            "frames": frames, "duration_seconds": frames / max(rate, 1)},
            ("Metadata inspection is not speech recognition or event classification.",))

    def transcribe_wav(self, path: str | Path) -> MediaResult:
        source = _bounded(path, MAX_AUDIO_BYTES)
        if not self.vosk_model_path or not self.vosk_model_path.is_dir():
            raise RuntimeError("Configure a local Vosk model directory before transcription")
        try:
            import vosk
        except ImportError as exc:
            raise RuntimeError("Install optional Vosk voice dependencies first") from exc
        with wave.open(str(source), "rb") as handle:
            if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
                raise ValueError("Vosk requires mono 16-bit PCM WAV")
            recognizer = vosk.KaldiRecognizer(vosk.Model(str(self.vosk_model_path)), handle.getframerate())
            chunks = []
            while data := handle.readframes(4000):
                if recognizer.AcceptWaveform(data):
                    text = json.loads(recognizer.Result()).get("text", "")
                    if text: chunks.append(text)
            text = json.loads(recognizer.FinalResult()).get("text", "")
            if text: chunks.append(text)
        return MediaResult("audio", "transcribed", {"transcript": " ".join(chunks)},
                           ("Speech transcription does not classify non-speech audio events.",))


class VideoPerception:
    def __init__(self, max_frames: int = 24, timeout_seconds: int = 90):
        if not 1 <= max_frames <= 120 or timeout_seconds < 1:
            raise ValueError("Video processing limits are invalid")
        self.max_frames, self.timeout_seconds = max_frames, timeout_seconds

    def extract_frames(self, path: str | Path, output_dir: str | Path) -> MediaResult:
        source = _bounded(path, MAX_VIDEO_BYTES)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("FFmpeg is not installed; no binary will be downloaded")
        target = Path(output_dir).expanduser().resolve()
        target.mkdir(parents=True, exist_ok=True)
        command = [ffmpeg, "-nostdin", "-v", "error", "-i", str(source),
                   "-vf", "fps=1/5,scale='min(1280,iw)':-2", "-frames:v",
                   str(self.max_frames), "-q:v", "4", str(target / "frame-%04d.jpg")]
        result = subprocess.run(command, capture_output=True, text=True,
                                timeout=self.timeout_seconds, check=False, shell=False)
        if result.returncode:
            raise RuntimeError(f"FFmpeg failed: {result.stderr[-2000:]}")
        frames = sorted(target.glob("frame-*.jpg"))
        return MediaResult("video", "frames_extracted",
            {"frame_count": len(frames), "frames": [str(item) for item in frames]},
            ("Frame extraction is not scene understanding or audio-video alignment.",))


class PerceptionSuite:
    def __init__(self, *, vision_provider: VisionProvider | None = None,
                 vosk_model_path: str | Path | None = None):
        self.image = ImagePerception(vision_provider)
        self.audio = AudioPerception(vosk_model_path)
        self.video = VideoPerception()

    @staticmethod
    def _available(module: str) -> bool:
        try: __import__(module); return True
        except ImportError: return False

    def capabilities(self) -> dict:
        return {
            "image": {"vision_model": self.image.vision_provider is not None,
                      "ocr": self._available("pytesseract") and self._available("PIL")},
            "audio": {"vosk": self._available("vosk"),
                      "configured_model": bool(self.audio.vosk_model_path and self.audio.vosk_model_path.is_dir())},
            "video": {"ffmpeg": shutil.which("ffmpeg") is not None, "max_frames": self.video.max_frames},
            "limitations": ["Audio event understanding requires a validated event classifier.",
                            "Video understanding requires a vision model to interpret extracted frames.",
                            "Optional packages and models are never downloaded automatically."],
        }
