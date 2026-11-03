import subprocess

import pytest

from aether.self_engineering import SelfEngineer


class IntegrationPatchProvider:
    def __init__(self, patch):
        self.patch = patch

    def chat(self, messages):
        return self.patch


def _integration_project(tmp_path):
    project = tmp_path / "project"
    (project / "aether").mkdir(parents=True)
    (project / "tests").mkdir()
    (project / "aether" / "safe.py").write_text("VALUE = 1\n", encoding="utf-8")
    (project / "tests" / "test_safe.py").write_text(
        "from aether.safe import VALUE\ndef test_value():\n    assert VALUE == 2\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "--quiet"], cwd=project, check=True, shell=False)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=project, check=True, shell=False)
    subprocess.run(["git", "config", "user.name", "AETHER test"], cwd=project, check=True, shell=False)
    subprocess.run(["git", "add", "."], cwd=project, check=True, shell=False)
    subprocess.run(["git", "commit", "-m", "baseline", "--quiet"], cwd=project, check=True, shell=False)
    return project


def _integration_patch():
    return (
        "diff --git a/aether/safe.py b/aether/safe.py\n"
        "--- a/aether/safe.py\n"
        "+++ b/aether/safe.py\n"
        "@@ -1 +1 @@\n"
        "-VALUE = 1\n"
        "+VALUE = 2\n"
    )


def test_self_engineer_integrates_only_validated_candidate_after_confirmation(tmp_path):
    project = _integration_project(tmp_path)
    engineer = SelfEngineer(project, IntegrationPatchProvider(_integration_patch()))
    proposal = engineer.propose("Fix the verified constant")
    evaluated = engineer.evaluate(proposal["id"], approved=True)
    assert evaluated["status"] == "validated_candidate"
    with pytest.raises(PermissionError, match="INTEGRATE"):
        engineer.integrate(proposal["id"], approved=True, confirmation="YES")
    integrated = engineer.integrate(
        proposal["id"], approved=True, confirmation="INTEGRATE"
    )
    assert integrated["integrated"] is True
    assert integrated["status"] == "integrated"
    assert (project / "aether" / "safe.py").read_text(encoding="utf-8") == "VALUE = 2\n"


def test_self_engineer_refuses_stale_production_file_at_integration(tmp_path):
    project = _integration_project(tmp_path)
    engineer = SelfEngineer(project, IntegrationPatchProvider(_integration_patch()))
    proposal = engineer.propose("Fix the verified constant")
    evaluated = engineer.evaluate(proposal["id"], approved=True)
    assert evaluated["status"] == "validated_candidate"
    (project / "aether" / "safe.py").write_text("VALUE = 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed since proposal"):
        engineer.integrate(proposal["id"], approved=True, confirmation="INTEGRATE")
    assert (project / "aether" / "safe.py").read_text(encoding="utf-8") == "VALUE = 3\n"
