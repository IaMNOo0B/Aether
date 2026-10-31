import pytest

from aether.desktop import DesktopController, DesktopPermissionError
from aether.voice import VoiceInterface, VoiceUnavailable


def test_desktop_controller_is_disabled_by_default():
    desktop = DesktopController(False, ("notepad.exe",))
    with pytest.raises(DesktopPermissionError):
        desktop.open_app("notepad.exe")


def test_desktop_controller_rejects_non_allowlisted_and_shell_apps():
    desktop = DesktopController(True, ("notepad.exe", "powershell.exe"))
    with pytest.raises(DesktopPermissionError):
        desktop.open_app("powershell.exe")
    with pytest.raises(DesktopPermissionError):
        desktop.open_app("cmd.exe")


def test_voice_interface_is_disabled_by_default():
    voice = VoiceInterface(False, "models/vosk")
    with pytest.raises(VoiceUnavailable):
        voice.listen_once()


def test_voice_wake_filter_accepts_hey_jarvis_but_rejects_other_speech():
    voice = VoiceInterface(True, "models/vosk", wake_word="jarvis")
    assert voice._wake_filter("Hey Jarvis open notepad") == "open notepad"
    assert voice._wake_filter("Jarvis what time is it") == "what time is it"
    assert voice._wake_filter("open notepad") == ""
