import json

import pytest

from aether.audit import AuditIntegrityError, SignedAuditLog
from aether.security_scan import scan_python_sources


def test_signed_audit_log_appends_and_verifies_hash_chain(tmp_path):
    path = tmp_path / "logs" / "audit.jsonl"
    log = SignedAuditLog(path, b"k" * 32)
    first = log.append("task.started", {"task_id": "task-1"})
    second = log.append("task.completed", {"task_id": "task-1", "status": "verified"})
    result = log.verify()
    assert result["valid"] is True
    assert result["entries"] == 2
    assert first["sequence"] == 1
    assert second["sequence"] == 2


def test_signed_audit_log_detects_tampering_and_wrong_key(tmp_path):
    path = tmp_path / "audit.jsonl"
    SignedAuditLog(path, b"k" * 32).append("decision", {"approved": False})
    with pytest.raises(AuditIntegrityError, match="signature mismatch"):
        SignedAuditLog(path, b"x" * 32).verify()
    content = path.read_text(encoding="utf-8").replace('"approved": false', '"approved": true')
    path.write_text(content, encoding="utf-8")
    with pytest.raises(AuditIntegrityError):
        SignedAuditLog(path, b"k" * 32).verify()


def test_signed_audit_log_rejects_short_key(tmp_path):
    with pytest.raises(ValueError, match="32 bytes"):
        SignedAuditLog(tmp_path / "audit.jsonl", b"short")


def test_static_security_scanner_detects_high_risk_python_patterns(tmp_path):
    source = tmp_path / "unsafe.py"
    source.write_text(
        "import subprocess\nimport pickle\n"
        "def run(value):\n"
        "    eval(value)\n"
        "    subprocess.run(value, shell=True)\n"
        "    pickle.loads(value)\n",
        encoding="utf-8",
    )
    report = scan_python_sources([tmp_path])
    rules = {item["rule"] for item in report["findings"]}
    assert report["files_scanned"] == 1
    assert {"dynamic-code-execution", "shell-true", "unsafe-deserialization"} <= rules
