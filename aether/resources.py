"""Cross-platform resource telemetry and configurable execution guard."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import ctypes
import os
import subprocess
import time


class ResourceLimitExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class ResourcePolicy:
    enabled: bool = True
    max_cpu_percent: float = 95.0
    max_ram_percent: float = 92.0
    min_free_ram_mb: int = 512
    max_gpu_utilization_percent: float = 98.0
    min_free_vram_mb: int = 512
    require_gpu_telemetry: bool = False
    cpu_sample_seconds: float = 0.1


class ResourceMonitor:
    """Read CPU/RAM telemetry using OS APIs and GPU telemetry through nvidia-smi."""

    @staticmethod
    def _filetime_value(value: Any) -> int:
        return (int(value.dwHighDateTime) << 32) | int(value.dwLowDateTime)

    def _cpu_counters(self) -> tuple[int, int] | None:
        if os.name == "nt":
            class FILETIME(ctypes.Structure):
                _fields_ = [
                    ("dwLowDateTime", ctypes.c_uint32),
                    ("dwHighDateTime", ctypes.c_uint32),
                ]

            idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
            if not ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
            ):
                return None
            idle_value = self._filetime_value(idle)
            total_value = self._filetime_value(kernel) + self._filetime_value(user)
            return idle_value, total_value
        stat = Path("/proc/stat")
        if stat.exists():
            fields = stat.read_text(encoding="utf-8").splitlines()[0].split()
            values = [int(value) for value in fields[1:]]
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            return idle, sum(values)
        return None

    def cpu_percent(self, interval: float = 0.1) -> float | None:
        before = self._cpu_counters()
        if before is None:
            return None
        time.sleep(max(0.05, min(interval, 1.0)))
        after = self._cpu_counters()
        if after is None:
            return None
        total_delta = after[1] - before[1]
        idle_delta = after[0] - before[0]
        if total_delta <= 0:
            return None
        return max(0.0, min(100.0, 100.0 * (1 - idle_delta / total_delta)))

    def memory(self) -> dict[str, float | int] | None:
        if os.name == "nt":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MEMORYSTATUSEX()
            status.dwLength = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None
            total = int(status.ullTotalPhys)
            available = int(status.ullAvailPhys)
        else:
            meminfo = Path("/proc/meminfo")
            if not meminfo.exists():
                return None
            values = {}
            for line in meminfo.read_text(encoding="utf-8").splitlines():
                key, _, value = line.partition(":")
                if key in {"MemTotal", "MemAvailable"}:
                    values[key] = int(value.strip().split()[0]) * 1024
            total = values.get("MemTotal", 0)
            available = values.get("MemAvailable", 0)
            if total <= 0:
                return None
        used = max(0, total - available)
        return {
            "total_mb": round(total / 1024 / 1024, 1),
            "free_mb": round(available / 1024 / 1024, 1),
            "used_percent": round(100 * used / total, 1) if total else 0.0,
        }

    @staticmethod
    def gpu() -> dict[str, float] | None:
        command = [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
        kwargs: dict[str, Any] = {
            "capture_output": True, "text": True, "timeout": 2.5, "check": False,
        }
        if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            result = subprocess.run(command, **kwargs)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if result.returncode != 0:
            return None
        try:
            fields = [float(item.strip()) for item in result.stdout.strip().split(",")]
        except ValueError:
            return None
        if len(fields) != 4:
            return None
        total, used, free, utilization = fields
        return {
            "total_mb": total, "used_mb": used,
            "free_mb": free, "utilization_percent": utilization,
        }

    def snapshot(self, cpu_sample_seconds: float = 0.1) -> dict[str, Any]:
        return {
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "cpu_percent": self.cpu_percent(cpu_sample_seconds),
            "ram": self.memory(),
            "gpu": self.gpu(),
        }


class ResourceGuard:
    def __init__(self, policy: ResourcePolicy, monitor: ResourceMonitor | None = None):
        self.policy = policy
        self.monitor = monitor or ResourceMonitor()

    def snapshot(self) -> dict[str, Any]:
        return self.monitor.snapshot(self.policy.cpu_sample_seconds)

    def check(self, operation: str, *, gpu_required: bool = False) -> dict[str, Any]:
        if not self.policy.enabled:
            return {"disabled": True}
        snapshot = self.snapshot()
        cpu = snapshot.get("cpu_percent")
        if cpu is not None and cpu > self.policy.max_cpu_percent:
            raise ResourceLimitExceeded(
                f"{operation} blocked: CPU usage {cpu:.1f}% exceeds "
                f"{self.policy.max_cpu_percent:.1f}% budget"
            )
        ram = snapshot.get("ram")
        if ram:
            if ram["used_percent"] > self.policy.max_ram_percent:
                raise ResourceLimitExceeded(
                    f"{operation} blocked: RAM usage {ram['used_percent']:.1f}% exceeds "
                    f"{self.policy.max_ram_percent:.1f}% budget"
                )
            if ram["free_mb"] < self.policy.min_free_ram_mb:
                raise ResourceLimitExceeded(
                    f"{operation} blocked: only {ram['free_mb']:.0f} MB RAM is free"
                )
        if gpu_required:
            gpu = snapshot.get("gpu")
            if gpu is None:
                if self.policy.require_gpu_telemetry:
                    raise ResourceLimitExceeded(
                        f"{operation} blocked: GPU telemetry is required but unavailable"
                    )
            else:
                if gpu["free_mb"] < self.policy.min_free_vram_mb:
                    raise ResourceLimitExceeded(
                        f"{operation} blocked: only {gpu['free_mb']:.0f} MB VRAM is free"
                    )
                if gpu["utilization_percent"] > self.policy.max_gpu_utilization_percent:
                    raise ResourceLimitExceeded(
                        f"{operation} blocked: GPU utilization "
                        f"{gpu['utilization_percent']:.1f}% exceeds budget"
                    )
        return snapshot
