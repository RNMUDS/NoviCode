"""Glob tool — finds files matching glob patterns."""

from __future__ import annotations

import glob as _glob
import os
from rnnr.security_manager import SecurityManager


class GlobTool:
    name = "glob"

    def __init__(self, security: SecurityManager, working_dir: str) -> None:
        self.security = security
        self.working_dir = working_dir

    def execute(self, arguments: dict) -> dict:
        pattern = arguments.get("pattern", "")
        if not pattern:
            return {"error": "No pattern provided"}

        if not os.path.isabs(pattern):
            pattern = os.path.join(self.working_dir, pattern)

        # Verify the base is within working dir
        base = pattern.split("*")[0]
        if base:
            verdict = self.security.check_path(base)
            if not verdict.allowed:
                return {"error": f"Blocked: {verdict.reason}"}

        matches = sorted(_glob.glob(pattern, recursive=True))[:100]
        return {"files": matches, "count": len(matches)}
