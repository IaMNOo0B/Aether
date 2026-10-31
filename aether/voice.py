"""Push-to-talk offline voice interface using optional Vosk, PyAudio, and pyttsx3."""
from __future__ import annotations

from pathlib import Path
from array import array
import json
import math
import time


class VoiceUnavailable(RuntimeError):
    pass


class VoiceInterface:
    def __init__(
        self, enabled: bool, model_path: str, wake_word: str = "jarvis",
        speak_responses: bool = True,
    ):
        self.enabled = enabled
        self.model_path = Path(model_path)
        self.wake_word = wake_word.casefold().strip()
        self.speak_responses = speak_responses
        self._model = None
        self._speaker = None

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise VoiceUnavailable("Voice is disabled in config/settings.toml")

    def listen_once(self, timeout: int = 8, phrase_time_limit: int = 12) -> str:
        self._require_enabled()
        if not self.model_path.is_dir():
            raise VoiceUnavailable(
                f"Vosk model not found at {self.model_path}. Download an offline Vosk model "
                "and set voice.vosk_model_path."
            )
        try:
            import pyaudio
            from vosk import KaldiRecognizer, Model
        except ImportError as exc:
            raise VoiceUnavailable(
                'Install the optional voice extra: python -m pip install -e ".[voice]"'
            ) from exc
        if self._model is None:
            self._model = Model(str(self.model_path))

        audio = pyaudio.PyAudio()
        stream = None
        recognizer = KaldiRecognizer(self._model, 16000)
        try:
            stream = audio.open(
                format=pyaudio.paInt16, channels=1, rate=16000,
                input=True, frames_per_buffer=4000,
            )
            stream.start_stream()
            started = time.monotonic()
            voice_started = None
            while True:
                now = time.monotonic()
                if voice_started is None and now - started >= timeout:
                    break
                if voice_started is not None and now - voice_started >= phrase_time_limit:
                    break
                data = stream.read(4000, exception_on_overflow=False)
                samples = array("h")
                samples.frombytes(data)
                rms = math.sqrt(sum(sample * sample for sample in samples) / max(1, len(samples)))
                if rms >= 300 and voice_started is None:
                    voice_started = time.monotonic()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = str(result.get("text", "")).strip()
                    if text:
                        return self._wake_filter(text)
            final = json.loads(recognizer.FinalResult())
            return self._wake_filter(str(final.get("text", "")).strip())
        except (OSError, RuntimeError) as exc:
            raise VoiceUnavailable(f"Microphone or offline speech recognition failed: {exc}") from exc
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            audio.terminate()

    def _wake_filter(self, text: str) -> str:
        if not text:
            return ""
        if self.wake_word:
            lowered = text.casefold()
            if lowered.startswith("hey " + self.wake_word):
                text = text[4 + len(self.wake_word):]
            elif lowered.startswith(self.wake_word):
                text = text[len(self.wake_word):]
            else:
                return ""
            text = text.lstrip(" ,.:")
        return text

    def speak(self, text: str) -> None:
        self._require_enabled()
        if not self.speak_responses:
            return
        try:
            import pyttsx3
        except ImportError as exc:
            raise VoiceUnavailable(
                'Install the optional voice extra: python -m pip install -e ".[voice]"'
            ) from exc
        if self._speaker is None:
            self._speaker = pyttsx3.init()
            self._speaker.setProperty("rate", 175)
        self._speaker.say(text[:4000])
        self._speaker.runAndWait()
