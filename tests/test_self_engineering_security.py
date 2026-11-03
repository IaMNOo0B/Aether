from aether.audit import SignedAuditLog
from aether.self_engineering import SelfEngineer


class PatchProvider:
    def __init__(self, patch):
        self.patch = patch

    def chat(self, messages):
        return self.patch


def test_self_engineering_security_scan_blocks_high_risk_candidate_before_tests(tmp_path):
    project = tmp_path / "project"
    (project / "aether").mkdir(parents=True)
    (project / "tests").mkdir()
    (project / "aether" / "safe.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    patch = (
        "diff --git a/aether/safe.py b/aether/safe.py\n"
        "--- a/aether/safe.py\n"
        "+++ b/aether/safe.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = eval('1')\n"
    )
    audit = SignedAuditLog(project / "data" / "audit.jsonl", b"a" * 32)
    engineer = SelfEngineer(project, PatchProvider(patch), audit_log=audit)
    experiment = engineer.propose("Test the candidate security gate")
    result = engineer.evaluate(experiment["id"], approved=True)

    assert result["status"] == "security_review_required"
    assert result["tests_passed"] is False
    assert any(item["rule"] == "dynamic-code-execution"
               for item in result["security_scan"]["findings"])
    verified = audit.verify()
    assert verified["entries"] == 3


def test_self_engineering_audit_is_optional_and_candidate_remains_unintegrated(tmp_path):
    engineer = SelfEngineer(tmp_path, PatchProvider("NO_SAFE_PATCH"))
    assert engineer.audit_log is None
