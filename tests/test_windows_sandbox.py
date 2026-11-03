from pathlib import Path

import pytest

from aether.windows_sandbox import WindowsSandboxRunner


def test_windows_sandbox_prepares_network_disabled_read_only_input(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.cmd").write_text("@echo off\necho 0>C:\\AetherOutput\\exit-code.txt\n", encoding="utf-8")
    (source / "project.py").write_text("print('candidate')\n", encoding="utf-8")
    config, output = WindowsSandboxRunner().prepare(
        source, tmp_path / "stage", memory_mb=2048
    )
    xml = config.read_text(encoding="utf-8")
    assert "<Networking>Disable</Networking>" in xml
    assert "<VGpu>Disable</VGpu>" in xml
    assert "<ReadOnly>true</ReadOnly>" in xml
    assert "<ReadOnly>false</ReadOnly>" in xml
    assert "<MemoryInMB>2048</MemoryInMB>" in xml
    assert (config.parent / "input" / "aether-sandbox-bootstrap.cmd").is_file()
    assert output.is_dir()


def test_windows_sandbox_requires_guest_entrypoint_and_separate_staging(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(FileNotFoundError, match="run.cmd"):
        WindowsSandboxRunner().prepare(source, tmp_path / "stage")
    (source / "run.cmd").write_text("@echo off", encoding="utf-8")
    with pytest.raises(ValueError, match="separate"):
        WindowsSandboxRunner().prepare(source, source / "stage")


def test_windows_sandbox_memory_limits_are_bounded(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.cmd").write_text("@echo off", encoding="utf-8")
    with pytest.raises(ValueError, match="memory"):
        WindowsSandboxRunner().prepare(source, tmp_path / "stage", memory_mb=16384)
