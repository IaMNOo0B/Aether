"""Static security checks for Python source; findings require human triage."""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class SecurityFinding:
    path: str
    line: int
    rule: str
    severity: str
    message: str


_SECRET = re.compile(
    r"(?i)(?:api[_-]?key|secret|token|password)\\s*[:=]\\s*['\"][A-Za-z0-9_./+=-]{16,}['\"]"
)


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: str):
        self.path = path
        self.findings: list[SecurityFinding] = []

    def add(self, node: ast.AST, rule: str, severity: str, message: str) -> None:
        self.findings.append(SecurityFinding(self.path, getattr(node, "lineno", 1), rule, severity, message))

    def visit_Call(self, node: ast.Call) -> None:
        name = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
        if name in {"eval", "exec"}:
            self.add(node, "dynamic-code-execution", "high", f"Call to {name}() requires strict review.")
        if name in {"os.system", "os.popen"}:
            self.add(node, "shell-execution", "high", f"Call to {name}() bypasses structured process arguments.")
        if name.startswith("subprocess."):
            if any(keyword.arg == "shell" and isinstance(keyword.value, ast.Constant)
                   and keyword.value.value is True for keyword in node.keywords):
                self.add(node, "shell-true", "high", "subprocess call enables shell=True.")
        if name in {"pickle.load", "pickle.loads"}:
            self.add(node, "unsafe-deserialization", "high", "Pickle can execute code while loading untrusted data.")
        if name == "yaml.load" and not any(
            keyword.arg == "Loader" for keyword in node.keywords
        ):
            self.add(node, "unsafe-yaml-loader", "medium", "yaml.load without an explicit safe loader requires review.")
        self.generic_visit(node)


def scan_python_sources(paths: Iterable[str | Path]) -> dict:
    findings: list[SecurityFinding] = []
    scanned = 0
    for raw_path in paths:
        path = Path(raw_path)
        candidates = [path] if path.is_file() else sorted(path.rglob("*.py")) if path.is_dir() else []
        for candidate in candidates:
            if candidate.is_symlink() or candidate.stat().st_size > 2_000_000:
                continue
            try:
                source = candidate.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            scanned += 1
            relative = candidate.as_posix()
            for line_number, line in enumerate(source.splitlines(), start=1):
                if _SECRET.search(line):
                    findings.append(SecurityFinding(
                        relative, line_number, "possible-hardcoded-secret", "high",
                        "Possible hardcoded credential; verify and rotate if genuine.",
                    ))
            try:
                tree = ast.parse(source, filename=relative)
            except SyntaxError as exc:
                findings.append(SecurityFinding(
                    relative, exc.lineno or 1, "syntax-error", "medium",
                    "Python source could not be parsed by the static scanner.",
                ))
                continue
            visitor = _Visitor(relative)
            visitor.visit(tree)
            findings.extend(visitor.findings)
    return {
        "scanner": "aether-python-static-security-v1",
        "files_scanned": scanned,
        "finding_count": len(findings),
        "findings": [asdict(item) for item in findings],
        "limitations": [
            "Static checks can miss vulnerabilities and can produce false positives.",
            "This is not a substitute for dependency auditing, dynamic testing or OS sandboxing.",
        ],
    }
