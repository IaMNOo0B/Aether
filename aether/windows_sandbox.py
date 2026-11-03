"""Optional Windows Sandbox execution for reviewed candidate workspaces.

The host folder is copied into a disposable staging directory. The guest sees
that input read-only, a separate output folder read-write, and networking,
clipboard, audio/video input and vGPU are disabled. A guest run.cmd must write
its integer exit code to C:\\AetherOutput\\exit-code.txt before shutdown.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copytree, ignore_patterns, which
import os
import subprocess
import tempfile
from xml.sax.saxutils import escape


class SandboxUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SandboxRun:
    status: str
    exit_code: int | None
    config_path: str
    output_directory: str
    limitation: str


class WindowsSandboxRunner:
    def __init__(self, timeout_seconds: int = 900):
        if not 30 <= timeout_seconds <= 3600:
            raise ValueError("Sandbox timeout must be between 30 and 3600 seconds")
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def executable() -> Path | None:
        candidate = which("WindowsSandbox.exe")
        if candidate:
            return Path(candidate)
        windows = os.environ.get("WINDIR")
        if windows:
            path = Path(windows) / "System32" / "WindowsSandbox.exe"
            if path.is_file():
                return path
        return None

    @staticmethod
    def available() -> bool:
        return os.name == "nt" and WindowsSandboxRunner.executable() is not None

    def prepare(
        self,
        source_directory: str | Path,
        staging_root: str | Path,
        *,
        memory_mb: int = 4096,
    ) -> tuple[Path, Path]:
        if not 512 <= memory_mb <= 8192:
            raise ValueError("Windows Sandbox memory must be between 512 and 8192 MB")
        source = Path(source_directory).resolve()
        if not source.is_dir() or source.is_symlink():
            raise ValueError("Sandbox source must be a regular directory")
        stage = Path(staging_root).resolve()
        if stage == source or source in stage.parents or stage in source.parents:
            raise ValueError("Sandbox staging root must be separate from the source tree")
        stage.mkdir(parents=True, exist_ok=True)
        input_dir = stage / "input"
        output_dir = stage / "output"
        if input_dir.exists() or output_dir.exists():
            raise FileExistsError("Sandbox staging root must be empty for each run")
        copytree(
            source, input_dir,
            ignore=ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache",
                                   "data", "logs", "workspace", ".aether", "node_modules"),
        )
        if not (input_dir / "run.cmd").is_file():
            raise FileNotFoundError("Sandbox candidate must provide a run.cmd guest entry point")
        output_dir.mkdir()
        bootstrap = input_dir / "aether-sandbox-bootstrap.cmd"
        bootstrap.write_text(
            "@echo off\r\n"
            "cd /d C:\\AetherInput\r\n"
            "call C:\\AetherInput\\run.cmd\r\n"
            "set AETHER_EXIT=%ERRORLEVEL%\r\n"
            "echo %AETHER_EXIT%>C:\\AetherOutput\\exit-code.txt\r\n"
            "shutdown /s /t 0\r\n",
            encoding="utf-8",
        )
        config = stage / "aether-sandbox.wsb"
        input_xml = escape(str(input_dir))
        output_xml = escape(str(output_dir))
        guest_command = escape(
            "cmd.exe /d /c C:\\AetherInput\\aether-sandbox-bootstrap.cmd"
        )
        config.write_text(
            "<Configuration>\n"
            "  <VGpu>Disable</VGpu>\n"
            "  <Networking>Disable</Networking>\n"
            "  <ClipboardRedirection>Disable</ClipboardRedirection>\n"
            "  <PrinterRedirection>Disable</PrinterRedirection>\n"
            "  <AudioInput>Disable</AudioInput>\n"
            "  <VideoInput>Disable</VideoInput>\n"
            "  <ProtectedClient>Enable</ProtectedClient>\n"
            f"  <MemoryInMB>{memory_mb}</MemoryInMB>\n"
            "  <MappedFolders>\n"
            f"    <MappedFolder><HostFolder>{input_xml}</HostFolder><SandboxFolder>C:\\AetherInput</SandboxFolder><ReadOnly>true</ReadOnly></MappedFolder>\n"
            f"    <MappedFolder><HostFolder>{output_xml}</HostFolder><SandboxFolder>C:\\AetherOutput</SandboxFolder><ReadOnly>false</ReadOnly></MappedFolder>\n"
            "  </MappedFolders>\n"
            f"  <LogonCommand><Command>{guest_command}</Command></LogonCommand>\n"
            "</Configuration>\n",
            encoding="utf-8",
        )
        return config, output_dir

    def run(self, config_path: str | Path, output_directory: str | Path) -> SandboxRun:
        if os.name != "nt":
            raise SandboxUnavailable("Windows Sandbox is available only on Windows 10/11 Pro or Enterprise with the feature enabled")
        executable = self.executable()
        if executable is None:
            raise SandboxUnavailable("Windows Sandbox feature is not installed or enabled")
        config = Path(config_path).resolve()
        output = Path(output_directory).resolve()
        if not config.is_file() or config.suffix.casefold() != ".wsb":
            raise ValueError("A valid .wsb configuration file is required")
        try:
            completed = subprocess.run(
                [str(executable), str(config)],
                capture_output=True, text=True,
                timeout=self.timeout_seconds, check=False, shell=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxRun(
                "timeout", None, str(config), str(output),
                "Sandbox GUI process exceeded timeout; guest side effects remain isolated to mapped output.",
            )
        except OSError as exc:
            raise SandboxUnavailable(f"Could not launch Windows Sandbox: {type(exc).__name__}") from exc
        result_file = output / "exit-code.txt"
        if completed.returncode != 0 or not result_file.is_file():
            return SandboxRun(
                "unknown", None, str(config), str(output),
                "Host launcher exit status does not prove guest command success; guest exit-code file is missing.",
            )
        try:
            exit_code = int(result_file.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return SandboxRun(
                "unknown", None, str(config), str(output),
                "Guest exit-code file is unreadable; no success claim is made.",
            )
        return SandboxRun(
            "completed" if exit_code == 0 else "failed", exit_code,
            str(config), str(output),
            "Windows Sandbox isolation is optional and OS-dependent; host hardware and feature availability require local validation.",
        )
