import pytest

from aether.resources import (
    ResourceGuard, ResourceLimitExceeded, ResourceMonitor, ResourcePolicy,
)


class FakeMonitor:
    def __init__(self, snapshot):
        self.value = snapshot
        self.calls = 0

    def snapshot(self, interval):
        self.calls += 1
        return self.value


def sample(cpu=20, ram_percent=40, ram_free=4096, gpu=None):
    return {
        "cpu_percent": cpu,
        "ram": {"used_percent": ram_percent, "free_mb": ram_free, "total_mb": 8192},
        "gpu": gpu,
    }


def test_resource_guard_accepts_usage_within_configured_budgets():
    monitor = FakeMonitor(sample())
    guard = ResourceGuard(ResourcePolicy(), monitor)
    result = guard.check("test operation")
    assert result["cpu_percent"] == 20
    assert monitor.calls == 1


def test_resource_guard_blocks_cpu_over_budget():
    guard = ResourceGuard(
        ResourcePolicy(max_cpu_percent=70), FakeMonitor(sample(cpu=85))
    )
    with pytest.raises(ResourceLimitExceeded, match="CPU usage"):
        guard.check("model inference")


def test_resource_guard_blocks_ram_pressure_and_low_free_memory():
    guard = ResourceGuard(
        ResourcePolicy(max_ram_percent=80, min_free_ram_mb=1024),
        FakeMonitor(sample(ram_percent=85, ram_free=2048)),
    )
    with pytest.raises(ResourceLimitExceeded, match="RAM usage"):
        guard.check("code tests")

    guard = ResourceGuard(
        ResourcePolicy(max_ram_percent=95, min_free_ram_mb=1024),
        FakeMonitor(sample(ram_percent=50, ram_free=256)),
    )
    with pytest.raises(ResourceLimitExceeded, match="only 256 MB RAM"):
        guard.check("code tests")


def test_resource_guard_checks_vram_only_for_gpu_required_operations():
    gpu = {
        "total_mb": 6144, "used_mb": 5900,
        "free_mb": 244, "utilization_percent": 70,
    }
    guard = ResourceGuard(
        ResourcePolicy(min_free_vram_mb=512), FakeMonitor(sample(gpu=gpu))
    )
    guard.check("CPU-only file listing", gpu_required=False)
    with pytest.raises(ResourceLimitExceeded, match="only 244 MB VRAM"):
        guard.check("local model inference", gpu_required=True)


def test_gpu_telemetry_can_be_required_or_optional():
    optional = ResourceGuard(
        ResourcePolicy(require_gpu_telemetry=False),
        FakeMonitor(sample(gpu=None)),
    )
    optional.check("local model inference", gpu_required=True)

    required = ResourceGuard(
        ResourcePolicy(require_gpu_telemetry=True),
        FakeMonitor(sample(gpu=None)),
    )
    with pytest.raises(ResourceLimitExceeded, match="GPU telemetry"):
        required.check("local model inference", gpu_required=True)


def test_disabled_guard_does_not_collect_telemetry():
    monitor = FakeMonitor(sample(cpu=100, ram_percent=100, ram_free=0))
    guard = ResourceGuard(ResourcePolicy(enabled=False), monitor)
    assert guard.check("disabled operation") == {"disabled": True}
    assert monitor.calls == 0


def test_nvidia_smi_gpu_telemetry_is_parsed(monkeypatch):
    class Result:
        returncode = 0
        stdout = "6144, 2048, 4096, 65\n"

    monkeypatch.setattr("aether.resources.subprocess.run", lambda *a, **k: Result())
    result = ResourceMonitor.gpu()
    assert result == {
        "total_mb": 6144.0, "used_mb": 2048.0,
        "free_mb": 4096.0, "utilization_percent": 65.0,
    }


def test_gpu_telemetry_returns_unknown_when_nvidia_smi_is_unavailable(monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi not found")

    monkeypatch.setattr("aether.resources.subprocess.run", unavailable)
    assert ResourceMonitor.gpu() is None
