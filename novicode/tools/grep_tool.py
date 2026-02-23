"""Grep tool — searches file contents by regex."""

from __future__ import annotations

import os
import re
from novicode.security_manager import SecurityManager


class GrepTool:
    name = "grep"

    def __init__(self, security: SecurityManager, working_dir: str) -> None:
        self.security = security
        self.working_dir = working_dir

    def execute(self, arguments: dict) -> dict:
        pattern = arguments.get("pattern", "")
        search_path = arguments.get("path", self.working_dir)
        if not pattern:
            return {"error": "No pattern provided"}

        if not os.path.isabs(search_path):
            search_path = os.path.join(self.working_dir, search_path)

        verdict = self.security.check_path(search_path)
        if not verdict.allowed:
            return {"error": f"Blocked: {verdict.reason}"}

        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return {"error": f"Invalid regex: {exc}"}

        matches: list[dict] = []
        max_matches = 50

        if os.path.isfile(search_path):
            matches.extend(self._search_file(search_path, regex))
        else:
            for root, _, files in os.walk(search_path):
                for fname in files:
                    fp = os.path.join(root, fname)
                    matches.extend(self._search_file(fp, regex))
                    if len(matches) >= max_matches:
                        return {"matches": matches[:max_matches], "truncated": True}

        return {"matches": matches, "truncated": False}

    def _search_file(self, filepath: str, regex: re.Pattern) -> list[dict]:
        results = []
        try:
            with open(filepath, "r", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if regex.search(line):
                        results.append({
                            "file": filepath,
                            "line": i,
                            "text": line.rstrip()[:200],
                        })
        except Exception:
            pass
        return results
