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


def test_self_engineer_rejects_drive_qualified_patch_paths(tmp_path):
    engineer = SelfEngineer(tmp_path, FakeProvider())
    patch = (
        "diff --git a/C:/outside.py b/C:/outside.py\n"
        "--- a/C:/outside.py\n"
        "+++ b/C:/outside.py\n"
        "@@ -1 +1 @@\n-a\n+b\n"
    )
    with pytest.raises(PatchRejected, match="Absolute"):
        engineer.validate_patch_text(patch)

def test_self_engineer_rejects_windows_backslash_traversal(tmp_path):
    engineer = SelfEngineer(tmp_path, FakeProvider())
    patch = (
        "diff --git a/..\\\\outside.py b/..\\\\outside.py\\n"
        "--- a/..\\\\outside.py\\n"
        "+++ b/..\\\\outside.py\\n"
        "@@ -1 +1 @@\\n-a\\n+b\\n"
    )
    with pytest.raises(PatchRejected, match="Unsafe"):
        engineer.validate_patch_text(patch)


def test_self_engineer_rejects_symlink_or_file_mode_changes(tmp_path):
    engineer = SelfEngineer(tmp_path, FakeProvider())
    patch = (
        "diff --git a/payload.py b/payload.py\\n"
        "new file mode 120000\\n"
        "--- /dev/null\\n"
        "+++ b/payload.py\\n"
        "@@ -0,0 +1 @@\\n+target\\n"
    )
    with pytest.raises(PatchRejected, match="File mode"):
        engineer.validate_patch_text(patch)
