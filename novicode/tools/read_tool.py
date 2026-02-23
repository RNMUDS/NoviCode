"""Read tool — reads file content within security boundaries."""

from __future__ import annotations

import os
from novicode.security_manager import SecurityManager


class ReadTool:
    name = "read"

    def __init__(self, security: SecurityManager, working_dir: str) -> None:
        self.security = security
        self.working_dir = working_dir

    def execute(self, arguments: dict) -> dict:
        path = arguments.get("path", "")
        if not path:
            return {"error": "No path provided"}

        full_path = os.path.join(self.working_dir, path) if not os.path.isabs(path) else path
        verdict = self.security.check_path(full_path)
        if not verdict.allowed:
            return {"error": f"Blocked: {verdict.reason}"}

        try:
            with open(full_path, "r") as f:
                content = f.read()
            if len(content) > 50000:
                content = content[:50000] + "\n... (truncated)"
            return {"content": content}
        except FileNotFoundError:
            return {"error": f"File not found: {path}"}
        except Exception as exc:
            return {"error": str(exc)}
