import pytest

from aether.self_engineering import PatchRejected, SelfEngineer


class FakeProvider:
    def chat(self, messages):
        return "NO_SAFE_PATCH"


def test_self_engineer_rejects_security_policy_changes(tmp_path):
    engineer = SelfEngineer(tmp_path, FakeProvider())
    patch = (
        "diff --git a/config/settings.toml b/config/settings.toml\n"
        "--- a/config/settings.toml\n"
        "+++ b/config/settings.toml\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    with pytest.raises(PatchRejected, match="protected"):
        engineer.validate_patch_text(patch)


def test_self_engineer_rejects_path_traversal(tmp_path):
    engineer = SelfEngineer(tmp_path, FakeProvider())
    patch = (
        "diff --git a/../outside.py b/../outside.py\n"
        "--- a/../outside.py\n"
        "+++ b/../outside.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    with pytest.raises(PatchRejected, match="Unsafe"):
        engineer.validate_patch_text(patch)


def test_self_engineer_requires_approval_before_running_generated_code(tmp_path):
    engineer = SelfEngineer(tmp_path, FakeProvider())
    with pytest.raises(PermissionError, match="explicit approval"):
        engineer.evaluate("not-a-real-experiment", approved=False)
