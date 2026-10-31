"""Push-to-talk offline voice interface using optional Vosk and pyttsx3 packages."""
from __future__ import annotations

from pathlib import Path


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
            import speech_recognition as sr
            from vosk import Model
        except ImportError as exc:
            raise VoiceUnavailable(
                "Install the optional voice extra: python -m pip install -e ".[voice]""
            ) from exc
        if self._model is None:
            self._model = Model(str(self.model_path))
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as microphone:
                recognizer.adjust_for_ambient_noise(microphone, duration=0.5)
                audio = recognizer.listen(
                    microphone, timeout=timeout, phrase_time_limit=phrase_time_limit
                )
            transcript = recognizer.recognize_vosk(audio, model=self._model)
        except (OSError, RuntimeError, ImportError) as exc:
            raise VoiceUnavailable(f"Microphone or offline speech recognition failed: {exc}") from exc
        except Exception as exc:
            if type(exc).__name__ in {"WaitTimeoutError", "UnknownValueError", "RequestError"}:
                raise VoiceUnavailable(f"Speech was not recognized: {exc}") from exc
            raise
        import json
        try:
            text = str(json.loads(transcript).get("text", "")).strip()
        except (json.JSONDecodeError, AttributeError, TypeError):
            text = transcript.strip()
        if not text:
            return ""
        if self.wake_word:
            if not text.casefold().startswith(self.wake_word):
                return ""
            text = text[len(self.wake_word):].lstrip(" ,.:")
        return text

    def speak(self, text: str) -> None:
        self._require_enabled()
        if not self.speak_responses:
            return
        try:
            import pyttsx3
        except ImportError as exc:
            raise VoiceUnavailable(
                "Install the optional voice extra: python -m pip install -e ".[voice]""
            ) from exc
        if self._speaker is None:
            self._speaker = pyttsx3.init()
            self._speaker.setProperty("rate", 175)
        self._speaker.say(text[:4000])
        self._speaker.runAndWait()
