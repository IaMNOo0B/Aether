"""Opt-in Windows desktop interaction with a strict application allowlist."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Any


class DesktopPermissionError(PermissionError):
    pass


class DesktopController:
    BLOCKED_APPS = {
        "cmd.exe", "powershell.exe", "pwsh.exe", "wscript.exe", "cscript.exe",
        "regedit.exe", "taskmgr.exe", "diskpart.exe", "format.com", "shutdown.exe",
    }

    def __init__(
        self, enabled: bool, allowed_apps: tuple[str, ...], allow_input: bool = False
    ):
        self.enabled = enabled
        self.allowed_apps = {Path(item).name.casefold() for item in allowed_apps}
        self.allow_input = allow_input

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise DesktopPermissionError("Desktop control is disabled in config/settings.toml")

    def open_app(self, app: str) -> dict[str, Any]:
        self._require_enabled()
        name = Path(app.strip().strip('"')).name.casefold()
        if not name or name not in self.allowed_apps or name in self.BLOCKED_APPS:
            raise DesktopPermissionError(f"Application is not allowlisted: {name or app}")
        executable = shutil.which(name)
        if executable is None:
            raise FileNotFoundError(f"Allowlisted application is not on PATH: {name}")
        process = subprocess.Popen([executable], shell=False, close_fds=True)
        return {"application": name, "pid": process.pid, "started": True}

    def _input_backend(self):
        self._require_enabled()
        if not self.allow_input:
            raise DesktopPermissionError("Desktop keyboard/mouse input is disabled")
        try:
            import pyautogui
        except ImportError as exc:
            raise RuntimeError(
                "Install the optional desktop extra: python -m pip install -e ".[desktop]""
            ) from exc
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.15
        return pyautogui

    def click(self, x: int, y: int, approved: bool = False) -> dict[str, Any]:
        if not approved:
            raise DesktopPermissionError("A human approval is required before mouse input")
        gui = self._input_backend()
        width, height = gui.size()
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Click coordinates are outside the current screen")
        gui.click(x=x, y=y)
        return {"clicked": True, "x": x, "y": y}

    def type_text(self, text: str, approved: bool = False) -> dict[str, Any]:
        if not approved:
            raise DesktopPermissionError("A human approval is required before keyboard input")
        if len(text) > 2000:
            raise ValueError("Desktop text input is limited to 2000 characters")
        gui = self._input_backend()
        gui.write(text, interval=0.01)
        return {"typed_characters": len(text)}

    def press_key(self, key: str, approved: bool = False) -> dict[str, Any]:
        if not approved:
            raise DesktopPermissionError("A human approval is required before keyboard input")
        allowed = {"enter", "tab", "esc", "escape", "space", "backspace", "delete"}
        normalized = key.casefold().strip()
        if normalized not in allowed:
            raise ValueError(f"Key is not in the safe single-key allowlist: {normalized}")
        self._input_backend().press(normalized)
        return {"pressed": normalized}

    def screenshot(self, destination: str | Path, approved: bool = False) -> Path:
        if not approved:
            raise DesktopPermissionError("A human approval is required before capturing the screen")
        gui = self._input_backend()
        path = Path(destination).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        gui.screenshot().save(path)
        return path
