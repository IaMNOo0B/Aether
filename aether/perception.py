"""Optional image, audio and video perception adapters.

Vision calls are opt-in and remote when configured with a hosted endpoint.
Video/audio extraction uses an explicit ffmpeg executable and never a shell.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import base64
import json
import os
import subprocess
import wave
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from aether.ollama import ModelUnavailable


class VisionProvider(Protocol):
    model_id: str
    privacy: str

    def analyze_image(self, image: bytes, mime_type: str, prompt: str) -> str: ...


@dataclass(frozen=True)
class ImageAnalysis:
    model_id: str
    privacy: str
    image_sha256: str
    response: str


class OpenAICompatibleVisionProvider:
    """Opt-in OpenAI-compatible image analysis adapter."""

    ALLOWED_MIME = {"image/png", "image/jpeg", "image/webp"}

    def __init__(
        self, base_url: str, model: str, api_key_env: str = "AETHER_VISION_API_KEY",
        timeout_seconds: int = 60, max_image_bytes: int = 8_000_000,
    ):
        if not base_url.startswith(("https://", "http://127.0.0.1", "http://localhost")):
            raise ValueError("Vision endpoint must use HTTPS or loopback HTTP")
        if not model.strip() or timeout_seconds < 1 or not 1 <= max_image_bytes <= 20_000_000:
            raise ValueError("Invalid vision model, timeout or image size limit")
        self.base_url = base_url.rstrip("/")
        self.model_id = model.strip()
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.max_image_bytes = max_image_bytes
        self.privacy = "remote" if not self.base_url.startswith(("http://127.0.0.1", "http://localhost")) else "local"

    def analyze_image(self, image: bytes, mime_type: str, prompt: str) -> str:
        if not isinstance(image, bytes) or not image or len(image) > self.max_image_bytes:
            raise ValueError("Image bytes must be non-empty and within the configured size limit")
        if mime_type not in self.ALLOWED_MIME:
            raise ValueError("Only PNG, JPEG and WebP images are supported")
        if not prompt.strip():
            raise ValueError("Image analysis prompt is required")
        encoded = base64.b64encode(image).decode("ascii")
        payload = {
            "model": self.model_id,
            "stream": False,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt[:4000]},
                    {"type": "image_url", "image_url": {
                        "url": f"data:{mime_type};base64,{encoded}"
                    }},
                ],
            }],
        }
        headers = {"Content-Type": "application/json"}
        key = os.environ.get(self.api_key_env, "").strip() if self.api_key_env else ""
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                result = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise ModelUnavailable(f"Vision endpoint returned HTTP {exc.code}") from exc
        except (OSError, URLError, TimeoutError) as exc:
            raise ModelUnavailable(f"Vision endpoint unavailable: {type(exc).__name__}") from exc
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelUnavailable("Vision endpoint returned invalid JSON") from exc
        try:
            text = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelUnavailable("Vision response has no assistant content") from exc
        if isinstance(text, list):
            text = "".join(item.get("text", "") for item in text if isinstance(item, dict))
        if not isinstance(text, str) or not text.strip():
            raise ModelUnavailable("Vision endpoint returned an empty response")
        return text.strip()


class FFmpegMediaProcessor:
    """Bounded, shell-free frame/audio extraction for local media files."""

    def __init__(self, ffmpeg: str = "ffmpeg", timeout_seconds: int = 120):
        self.ffmpeg = ffmpeg
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _validate_source(source: str | Path) -> Path:
        path = Path(source).resolve()
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("Media source must be an existing regular file")
        if path.stat().st_size > 1_000_000_000:
            raise ValueError("Media source exceeds the 1 GB processing limit")
        return path

    def extract_frames(
        self, source: str | Path, destination: str | Path, *,
        interval_seconds: float = 2.0, max_frames: int = 60,
    ) -> list[Path]:
        if not 0.25 <= interval_seconds <= 60:
            raise ValueError("Frame interval must be between 0.25 and 60 seconds")
        if not 1 <= max_frames <= 120:
            raise ValueError("max_frames must be between 1 and 120")
        input_path = self._validate_source(source)
        output_dir = Path(destination).resolve()
        if output_dir == input_path.parent or input_path in output_dir.parents:
            raise ValueError("Frame output must not overwrite or contain the source media")
        output_dir.mkdir(parents=True, exist_ok=True)
        pattern = output_dir / "%04d.jpg"
        command = [
            self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(input_path), "-vf", f"fps=1/{interval_seconds}",
            "-frames:v", str(max_frames), "-q:v", "3", str(pattern),
        ]
        self._run(command)
        frames = sorted(output_dir.glob("*.jpg"))
        if not frames:
            raise RuntimeError("ffmpeg completed without producing video frames")
        if len(frames) > max_frames:
            raise RuntimeError("ffmpeg produced more frames than the configured limit")
        return frames

    def extract_audio(self, source: str | Path, destination: str | Path) -> Path:
        input_path = self._validate_source(source)
        output_path = Path(destination).resolve()
        if output_path == input_path or input_path in output_path.parents:
            raise ValueError("Audio output must not overwrite or be inside the source media")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._run([
            self.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(input_path), "-vn", "-ac", "1", "-ar", "16000",
            "-f", "wav", str(output_path),
        ])
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError("ffmpeg did not produce an audio file")
        return output_path

    def _run(self, command: list[str]) -> None:
        try:
            result = subprocess.run(
                command, capture_output=True, text=True,
                timeout=self.timeout_seconds, check=False, shell=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("ffmpeg is not installed or not on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("Media processing exceeded its time limit") from exc
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed with exit code {result.returncode}: {result.stderr[-1000:]}")


class VoskFileTranscriber:
    """Offline transcription of extracted 16 kHz mono PCM WAV files."""

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)
        self._model: Any = None

    def transcribe(self, wav_path: str | Path) -> dict[str, Any]:
        source = Path(wav_path).resolve()
        if not source.is_file() or source.stat().st_size > 250_000_000:
            raise ValueError("WAV source must exist and be no larger than 250 MB")
        try:
            import vosk
        except ImportError as exc:
            raise RuntimeError('Install optional voice dependencies with pip install -e ".[voice]"') from exc
        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Vosk model directory not found: {self.model_path}")
        if self._model is None:
            self._model = vosk.Model(str(self.model_path))
        with wave.open(str(source), "rb") as audio:
            if audio.getnchannels() != 1 or audio.getframerate() != 16000 or audio.getsampwidth() != 2:
                raise ValueError("WAV must be 16 kHz, mono, 16-bit PCM; use FFmpegMediaProcessor.extract_audio")
            recognizer = vosk.KaldiRecognizer(self._model, 16000)
            recognizer.SetWords(True)
            chunks = []
            while True:
                data = audio.readframes(8000)
                if not data:
                    break
                if recognizer.AcceptWaveform(data):
                    chunks.append(json.loads(recognizer.Result()))
            chunks.append(json.loads(recognizer.FinalResult()))
        text = " ".join(item.get("text", "").strip() for item in chunks if item.get("text"))
        return {
            "text": text,
            "chunks": chunks,
            "audio_sha256": sha256(source.read_bytes()).hexdigest(),
            "model_path": str(self.model_path),
            "limitations": "Offline ASR output requires review; transcription confidence is not semantic correctness.",
        }
