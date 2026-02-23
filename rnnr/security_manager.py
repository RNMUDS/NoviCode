"""Security manager — blocks dangerous commands and path traversal."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from rnnr.config import WORKING_DIR


@dataclass(frozen=True)
class SecurityVerdict:
    allowed: bool
    reason: str = ""


# Shell patterns that are always blocked
_BLOCKED_COMMANDS: list[re.Pattern] = [
    re.compile(r"\bsudo\b"),
    re.compile(r"\bchmod\b"),
    re.compile(r"\bchown\b"),
    re.compile(r"\bdd\b\s"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"/dev/"),
    re.compile(r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|--recursive)\b.*(/|\s)"),
    re.compile(r"\brm\s+-rf\s+/"),
    re.compile(r"\bcurl\b.*\|\s*\bbash\b"),
    re.compile(r"\bwget\b.*\|\s*\bbash\b"),
    re.compile(r"\bpip\s+install\b"),
    re.compile(r"\bpip3\s+install\b"),
    re.compile(r"\bnpm\s+install\b"),
    re.compile(r"\byarn\s+add\b"),
    re.compile(r"\bcurl\b"),
    re.compile(r"\bwget\b"),
    re.compile(r"\bnc\b\s"),
    re.compile(r"\bnetcat\b"),
    re.compile(r"\bssh\b"),
    re.compile(r"\bscp\b"),
    re.compile(r"\brsync\b"),
    re.compile(r"\btelnet\b"),
    re.compile(r"\bnmap\b"),
    re.compile(r"\biptables\b"),
    re.compile(r"\bsystemctl\b"),
    re.compile(r"\bservice\b"),
    re.compile(r"\bkill\b"),
    re.compile(r"\bkillall\b"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bmount\b"),
    re.compile(r"\bumount\b"),
    re.compile(r"\bfdisk\b"),
    re.compile(r"\bparted\b"),
    re.compile(r"\bdocker\b"),
    re.compile(r"\bpodman\b"),
]

# Python imports that are never allowed (network/system)
_BLOCKED_PYTHON_IMPORTS: set[str] = {
    "subprocess", "os.system", "shutil", "socket", "http", "urllib",
    "requests", "httpx", "aiohttp", "flask", "django", "fastapi",
    "paramiko", "fabric", "boto3", "botocore", "google.cloud",
    "azure", "ftplib", "smtplib", "imaplib", "poplib",
    "ctypes", "cffi", "multiprocessing",
    "webbrowser", "antigravity",
}


class SecurityManager:
    """Validates commands and file paths against security policy."""

    def __init__(self, working_dir: str | None = None) -> None:
        self.working_dir = os.path.realpath(working_dir or WORKING_DIR)

    def check_command(self, command: str) -> SecurityVerdict:
        """Check a shell command against the blocklist."""
        for pattern in _BLOCKED_COMMANDS:
            if pattern.search(command):
                return SecurityVerdict(
                    allowed=False,
                    reason=f"Blocked command pattern: {pattern.pattern}",
                )
        return SecurityVerdict(allowed=True)

    def check_path(self, path: str) -> SecurityVerdict:
        """Ensure path is within the working directory (no traversal)."""
        real = os.path.realpath(path)
        if not real.startswith(self.working_dir):
            return SecurityVerdict(
                allowed=False,
                reason=f"Path escapes working directory: {real}",
            )
        # block symlink traversal
        if os.path.islink(path):
            target = os.path.realpath(path)
            if not target.startswith(self.working_dir):
                return SecurityVerdict(
                    allowed=False,
                    reason=f"Symlink points outside working directory: {target}",
                )
        return SecurityVerdict(allowed=True)

    def check_python_imports(self, imports: set[str]) -> SecurityVerdict:
        """Check if any import is in the global blocklist."""
        blocked = imports & _BLOCKED_PYTHON_IMPORTS
        if blocked:
            return SecurityVerdict(
                allowed=False,
                reason=f"Blocked imports: {', '.join(sorted(blocked))}",
            )
        return SecurityVerdict(allowed=True)
