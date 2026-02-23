"""Validation layer — enforces language isolation, import rules, and output limits."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

from rnnr.config import (
    DEFAULT_MAX_FILES,
    DEFAULT_MAX_LINES,
    ModeProfile,
    LanguageFamily,
)


@dataclass
class Violation:
    rule: str
    detail: str


@dataclass
class ValidationResult:
    valid: bool
    violations: list[Violation] = field(default_factory=list)

    def add(self, rule: str, detail: str) -> None:
        self.violations.append(Violation(rule=rule, detail=detail))
        self.valid = False


class Validator:
    """Inspects LLM-generated code and enforces curriculum constraints."""

    def __init__(
        self,
        profile: ModeProfile,
        max_files: int = DEFAULT_MAX_FILES,
        max_lines: int = DEFAULT_MAX_LINES,
    ) -> None:
        self.profile = profile
        self.max_files = max_files
        self.max_lines = max_lines

    def validate(self, code: str, filename: str) -> ValidationResult:
        """Run all validation checks on a single code artifact."""
        result = ValidationResult(valid=True)
        self._check_line_count(code, result)
        self._check_language_isolation(code, filename, result)
        if self.profile.language == LanguageFamily.PYTHON:
            self._check_python_imports(code, result)
        self._check_forbidden_patterns(code, result)
        return result

    def validate_batch(self, files: dict[str, str]) -> ValidationResult:
        """Validate a set of {filename: code} pairs."""
        result = ValidationResult(valid=True)
        if len(files) > self.max_files:
            result.add(
                "max_files",
                f"Generated {len(files)} files, limit is {self.max_files}",
            )
        for fname, code in files.items():
            r = self.validate(code, fname)
            if not r.valid:
                result.valid = False
                result.violations.extend(r.violations)
        return result

    # ── Individual checks ───────────────────────────────────────────

    def _check_line_count(self, code: str, result: ValidationResult) -> None:
        count = code.count("\n") + 1
        if count > self.max_lines:
            result.add("max_lines", f"{count} lines exceeds limit of {self.max_lines}")

    def _check_language_isolation(
        self, code: str, filename: str, result: ValidationResult
    ) -> None:
        lang = self.profile.language

        if lang == LanguageFamily.PYTHON:
            # Detect HTML/JS in Python mode
            if _contains_html(code):
                result.add("language_isolation", "HTML detected in Python mode")
            if _contains_js_pattern(code):
                result.add("language_isolation", "JavaScript detected in Python mode")

        elif lang == LanguageFamily.WEB:
            # Detect Python in web mode
            if _contains_python(code) and not filename.endswith((".html", ".js", ".css")):
                result.add("language_isolation", "Python detected in web mode")

    def _check_python_imports(self, code: str, result: ValidationResult) -> None:
        imports = _extract_python_imports(code)
        allowed = self.profile.allowed_imports
        for imp in imports:
            top_level = imp.split(".")[0]
            if imp not in allowed and top_level not in allowed:
                result.add("forbidden_import", f"Import '{imp}' not allowed in this mode")

    def _check_forbidden_patterns(self, code: str, result: ValidationResult) -> None:
        # No external API calls
        if re.search(r"https?://", code):
            result.add("no_external_api", "URL/API reference detected")
        # No package installation
        if re.search(r"(pip|npm|yarn)\s+install", code):
            result.add("no_install", "Package installation detected")
        # No os/system commands
        if re.search(r"\bos\.system\s*\(", code):
            result.add("no_os_system", "os.system() call detected")
        if re.search(r"\bsubprocess\.", code):
            result.add("no_subprocess", "subprocess usage detected")


def correction_prompt(violations: list[Violation], mode_name: str) -> str:
    """Build a correction prompt to re-steer the LLM after a violation."""
    details = "\n".join(f"  - [{v.rule}] {v.detail}" for v in violations)
    return (
        f"Your previous response violated these rules:\n{details}\n\n"
        f"You are in '{mode_name}' mode. Please regenerate your response "
        f"strictly following all constraints. Do NOT mix languages. "
        f"Do NOT use forbidden imports or APIs."
    )


# ── Heuristic detectors ────────────────────────────────────────────

def _contains_html(code: str) -> bool:
    return bool(re.search(r"<(!DOCTYPE|html|head|body|div|script|style)\b", code, re.I))


def _contains_js_pattern(code: str) -> bool:
    js_patterns = [
        r"\bdocument\.(getElementById|querySelector|createElement)\b",
        r"\bconsole\.log\b",
        r"\bwindow\.\b",
        r"\baddEventListener\b",
        r"\bfunction\s+\w+\s*\(",  # too broad alone, combine with others
    ]
    matches = sum(1 for p in js_patterns if re.search(p, code))
    return matches >= 2


def _contains_python(code: str) -> bool:
    py_patterns = [
        r"^def\s+\w+\s*\(", r"^class\s+\w+", r"^import\s+\w+",
        r"^from\s+\w+\s+import", r"\bprint\s*\(",
    ]
    matches = sum(1 for p in py_patterns if re.search(p, code, re.M))
    return matches >= 2


def _extract_python_imports(code: str) -> set[str]:
    """Extract top-level import names from Python source."""
    imports: set[str] = set()
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fallback: regex extraction
        for m in re.finditer(r"^\s*import\s+([\w.]+)", code, re.M):
            imports.add(m.group(1))
        for m in re.finditer(r"^\s*from\s+([\w.]+)\s+import", code, re.M):
            imports.add(m.group(1))
        return imports

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports
