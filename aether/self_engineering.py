"""Controlled self-engineering: inspect, propose, and test patches without auto-integration."""
from __future__ import annotations

from pathlib import Path, PureWindowsPath
from typing import Any
from uuid import uuid4
import json
import shutil
import subprocess
import sys

from aether.providers import ChatProvider


class PatchRejected(ValueError):
    pass


class SelfEngineer:
    PROTECTED_PATHS = {
        ".github/workflows", "config/settings.toml", "aether/autonomy.py",
        "aether/desktop.py", "aether/voice.py", "aether/self_engineering.py",
        "aether/config.py", "main.py",
        ".gitignore", "INSTALL.bat", "START_AETHER.bat",
    }

    def __init__(
        self, project_root: str | Path, provider: ChatProvider, timeout: int = 300,
        audit_log: Any | None = None,
    ):
        self.root = Path(project_root).resolve()
        self.provider = provider
        self.timeout = timeout
        self.audit_log = audit_log
        self.experiments = self.root / ".aether" / "experiments"
        self.experiments.mkdir(parents=True, exist_ok=True)

    def _audit(self, event_type: str, details: dict[str, Any]) -> None:
        if self.audit_log is not None:
            self.audit_log.append(event_type, details)

    def inspect(self, max_files: int = 40) -> dict[str, Any]:
        ignored = {".git", ".venv", "__pycache__", ".pytest_cache", "data", "logs",
                   "workspace", ".aether", "node_modules"}
        files = []
        for path in sorted(self.root.rglob("*.py")):
            if any(part in ignored for part in path.parts):
                continue
            try:
                if path.stat().st_size > 100_000:
                    continue
                files.append({
                    "path": str(path.relative_to(self.root)),
                    "content": path.read_text(encoding="utf-8")[:4000],
                })
            except (OSError, UnicodeDecodeError):
                continue
            if len(files) >= max(1, min(max_files, 80)):
                break
        return {"root": str(self.root), "python_files": files}

    def propose(self, objective: str) -> dict[str, Any]:
        if not objective.strip():
            raise ValueError("Self-engineering objective cannot be empty")
        snapshot = self.inspect(max_files=12)
        context = "\n\n".join(
            "FILE: " + item["path"] + "\n" + item["content"]
            for item in snapshot["python_files"]
        )
        response = self.provider.chat([
            {
                "role": "system",
                "content": (
                    "You are a cautious software engineer. Return a unified git diff only. "
                    "Use paths relative to the repository. Do not change security policy, "
                    "permissions, startup scripts, or protected runtime controls. Do not add "
                    "dependencies. Make the smallest coherent change and include tests. "
                    "Do not claim tests passed. If the requested change is unsafe or context "
                    "is insufficient, return the exact text NO_SAFE_PATCH."
                ),
            },
            {"role": "user", "content": (
                f"Improvement objective: {objective}\n\n"
                "Current source snapshot (read-only reference):\n" + context[:10000]
            )},
        ])
        if response.strip() == "NO_SAFE_PATCH":
            raise PatchRejected("The model declined to propose a safe patch")
        patch = response.strip()
        if not patch.startswith("diff --git ") and "diff --git " in patch:
            patch = patch[patch.index("diff --git "):]
        self.validate_patch_text(patch)
        experiment_id = uuid4().hex
        directory = self.experiments / experiment_id
        directory.mkdir(parents=True)
        patch_path = directory / "proposal.patch"
        patch_path.write_text(patch + "\n", encoding="utf-8")
        metadata = {
            "id": experiment_id, "objective": objective,
            "patch": str(patch_path.relative_to(self.root)),
            "status": "proposed", "integrated": False,
        }
        (directory / "experiment.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        self._audit("self_engineering.patch_proposed", {
            "experiment_id": experiment_id,
            "objective": objective[:500],
            "patch_sha256": __import__("hashlib").sha256(patch.encode("utf-8")).hexdigest(),
        })
        return metadata

    def validate_patch_text(self, patch: str) -> None:
        if not patch.startswith("diff --git ") or len(patch) > 200_000:
            raise PatchRejected("Patch must be a unified git diff under 200 KB")
        if "GIT binary patch" in patch or chr(0) in patch:
            raise PatchRejected("Binary patches are not accepted")
        for line in patch.splitlines():
            if line.startswith(("old mode ", "new mode ", "new file mode ", "deleted file mode ")):
                raise PatchRejected("File mode and symlink changes are not accepted")
            if not (line.startswith("--- ") or line.startswith("+++ ")):
                continue
            raw_path = line[4:].split("\t", 1)[0]
            if raw_path == "/dev/null":
                continue
            if (raw_path.startswith(("/", "\\")) or "\\" in raw_path
                    or ".." in Path(raw_path).parts
                    or ".." in PureWindowsPath(raw_path).parts):
                raise PatchRejected(f"Unsafe patch path: {raw_path}")
            normalized = raw_path[2:] if raw_path.startswith(("a/", "b/")) else raw_path
            normalized_path = Path(normalized)
            windows_path = PureWindowsPath(normalized)
            if (normalized_path.is_absolute() or normalized_path.drive
                    or windows_path.is_absolute() or windows_path.drive):
                raise PatchRejected(f"Absolute patch path is not allowed: {raw_path}")
            if any(
                normalized == protected or normalized.startswith(protected + "/")
                for protected in self.PROTECTED_PATHS
            ):
                raise PatchRejected(f"Patch touches protected path: {normalized}")
            if Path(normalized).suffix.casefold() not in {".py", ".md", ".txt"}:
                raise PatchRejected(f"File type is not allowed in a patch: {normalized}")

    def evaluate(self, experiment_id: str, approved: bool = False) -> dict[str, Any]:
        if not approved:
            raise PermissionError(
                "Testing generated code requires explicit approval. Tests execute code and "
                "this copy-based workspace is not an OS security sandbox."
            )
        directory = (self.experiments / experiment_id).resolve()
        if self.experiments.resolve() not in directory.parents:
            raise ValueError("Experiment path is outside the experiment directory")
        metadata_path = directory / "experiment.json"
        patch_path = directory / "proposal.patch"
        if not metadata_path.is_file() or not patch_path.is_file():
            raise FileNotFoundError(f"Unknown experiment: {experiment_id}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("status") != "proposed":
            raise ValueError("Only proposed experiments can be evaluated")

        patch = patch_path.read_text(encoding="utf-8")
        self.validate_patch_text(patch)
        self._audit("self_engineering.evaluation_started", {
            "experiment_id": experiment_id, "approved": True,
        })
        baseline = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=self.root, capture_output=True, text=True,
            timeout=self.timeout, check=False, shell=False,
        )
        if baseline.returncode != 0:
            metadata.update({"status": "rejected", "reason": "baseline tests failed",
                             "baseline": baseline.stdout[-6000:] + baseline.stderr[-3000:]})
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            self._audit("self_engineering.baseline_rejected", {
                "experiment_id": experiment_id, "reason": "baseline tests failed",
            })
            return metadata

        worktree = directory / "worktree"
        ignored = shutil.ignore_patterns(
            ".git", ".venv", "__pycache__", ".pytest_cache", "data", "logs",
            "workspace", ".aether", "node_modules",
        )
        shutil.copytree(self.root, worktree, ignore=ignored)
        initialized = subprocess.run(
            ["git", "init", "--quiet"], cwd=worktree, capture_output=True, text=True,
            timeout=30, check=False, shell=False,
        )
        if initialized.returncode != 0:
            metadata.update({"status": "rejected", "reason": "could not initialize candidate worktree",
                             "patch_error": initialized.stderr[-4000:]})
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return metadata
        check = subprocess.run(
            ["git", "apply", "--check", str(patch_path)],
            cwd=worktree, capture_output=True, text=True,
            timeout=30, check=False, shell=False,
        )
        if check.returncode != 0:
            metadata.update({"status": "rejected", "reason": "patch did not apply",
                             "patch_error": check.stderr[-4000:]})
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return metadata
        applied = subprocess.run(
            ["git", "apply", str(patch_path)],
            cwd=worktree, capture_output=True, text=True,
            timeout=30, check=False, shell=False,
        )
        if applied.returncode != 0:
            metadata.update({"status": "rejected", "reason": "patch application failed",
                             "patch_error": applied.stderr[-4000:]})
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return metadata

        from aether.security_scan import scan_python_sources
        security_report = scan_python_sources([worktree / "aether"])
        high_findings = [
            item for item in security_report["findings"] if item["severity"] == "high"
        ]
        metadata["security_scan"] = security_report
        if high_findings:
            metadata.update({
                "status": "security_review_required",
                "integrated": False,
                "baseline_passed": True,
                "tests_passed": False,
                "reason": "Static security scanner found high-severity patterns; candidate tests were not executed.",
            })
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            self._audit("self_engineering.security_review_required", {
                "experiment_id": experiment_id,
                "finding_count": len(high_findings),
                "rules": sorted({item["rule"] for item in high_findings}),
            })
            return metadata

        compile_result = subprocess.run(
            [sys.executable, "-m", "compileall", "-q", "aether"],
            cwd=worktree, capture_output=True, text=True,
            timeout=60, check=False, shell=False,
        )
        tests = subprocess.run(
            [sys.executable, "-m", "pytest", "-q"],
            cwd=worktree, capture_output=True, text=True,
            timeout=self.timeout, check=False, shell=False,
        )
        passed = compile_result.returncode == 0 and tests.returncode == 0
        metadata.update({
            "status": "validated_candidate" if passed else "rejected",
            "integrated": False,
            "baseline_passed": True,
            "compile_passed": compile_result.returncode == 0,
            "tests_passed": tests.returncode == 0,
            "test_output": tests.stdout[-8000:] + tests.stderr[-4000:],
            "security_scan": security_report,
            "worktree": str(worktree.relative_to(self.root)),
            "note": "Candidate only. No production files were changed.",
        })
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._audit("self_engineering.candidate_evaluated", {
            "experiment_id": experiment_id,
            "status": metadata["status"],
            "compile_passed": metadata["compile_passed"],
            "tests_passed": metadata["tests_passed"],
            "security_finding_count": security_report["finding_count"],
        })
        return metadata

    def rollback(self, experiment_id: str) -> None:
        directory = (self.experiments / experiment_id).resolve()
        if self.experiments.resolve() not in directory.parents:
            raise ValueError("Experiment path is outside the experiment directory")
        worktree = directory / "worktree"
        if worktree.exists():
            shutil.rmtree(worktree)
        self._audit("self_engineering.candidate_rolled_back", {"experiment_id": experiment_id})
