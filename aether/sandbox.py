"""Fail-closed Docker candidate sandbox; no network, dropped capabilities and resource limits."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import shutil, subprocess, tempfile


class SandboxUnavailable(RuntimeError):
    pass


class SandboxRejected(ValueError):
    pass


class DockerSandbox:
    def __init__(self, project_root: str | Path, *, image: str = "aether-python-test:3.12",
                 memory_limit: str = "2g", cpu_limit: str = "2.0",
                 pids_limit: int = 128, timeout_seconds: int = 600,
                 runner=subprocess.run):
        self.root = Path(project_root).resolve()
        self.image, self.memory_limit, self.cpu_limit = image, memory_limit, cpu_limit
        self.pids_limit, self.timeout_seconds, self.runner = pids_limit, timeout_seconds, runner
        if not image or any(char.isspace() for char in image):
            raise ValueError("Docker image must be a valid image reference")
        if pids_limit < 16 or timeout_seconds < 1:
            raise ValueError("Sandbox limits are invalid")

    def availability(self) -> dict:
        docker = shutil.which("docker")
        if not docker:
            return {"available": False, "reason": "Docker CLI is not installed"}
        try:
            daemon = self.runner([docker, "info", "--format", "{{.ServerVersion}}"],
                                 capture_output=True, text=True, timeout=10, check=False, shell=False)
            if daemon.returncode:
                return {"available": False, "reason": "Docker daemon is unavailable"}
            image = self.runner([docker, "image", "inspect", self.image],
                                capture_output=True, text=True, timeout=10, check=False, shell=False)
            if image.returncode:
                return {"available": False, "reason": f"Image {self.image!r} is not present locally"}
            return {"available": True, "docker": docker, "server": daemon.stdout.strip(), "image": self.image}
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"available": False, "reason": str(exc)}

    def run_tests(self, candidate_root: str | Path, *, approved: bool = False,
                  command: Sequence[str] = ("python", "-m", "pytest", "-q", "-p", "no:cacheprovider")) -> dict:
        if not approved:
            raise PermissionError("OS-isolated candidate execution requires explicit approval")
        source = Path(candidate_root).resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        if source == self.root or self.root not in source.parents:
            raise SandboxRejected("Candidate must be a child directory of the project root")
        if not command or any(not isinstance(part, str) or not part for part in command):
            raise SandboxRejected("Sandbox command must be a non-empty argument list")
        state = self.availability()
        if not state["available"]:
            raise SandboxUnavailable(state["reason"])
        with tempfile.TemporaryDirectory(prefix="aether-sandbox-") as temp:
            workspace = Path(temp) / "candidate"
            shutil.copytree(source, workspace, ignore=shutil.ignore_patterns(
                ".git", ".venv", "__pycache__", ".pytest_cache", "data", "logs", ".aether"))
            argv = [
                state["docker"], "run", "--rm", "--network", "none", "--read-only",
                "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", str(self.pids_limit), "--memory", self.memory_limit,
                "--cpus", self.cpu_limit, "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
                "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "PYTEST_ADDOPTS=-p no:cacheprovider",
                "-v", f"{workspace}:/workspace:rw", "-w", "/workspace",
                self.image, *command,
            ]
            try:
                result = self.runner(argv, capture_output=True, text=True,
                                     timeout=self.timeout_seconds, check=False, shell=False)
                return {"status": "passed" if result.returncode == 0 else "failed",
                        "return_code": result.returncode, "stdout": result.stdout[-12000:],
                        "stderr": result.stderr[-8000:],
                        "isolation": {"network": "none", "root": "read-only",
                                      "capabilities": "dropped-all", "no_new_privileges": True,
                                      "memory": self.memory_limit, "cpus": self.cpu_limit,
                                      "pids": self.pids_limit, "image": self.image}}
            except subprocess.TimeoutExpired:
                return {"status": "timed_out", "timeout_seconds": self.timeout_seconds}
