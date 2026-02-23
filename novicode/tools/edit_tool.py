"""Edit tool — performs string replacement in files."""

from __future__ import annotations

import os
from novicode.security_manager import SecurityManager
from novicode.policy_engine import PolicyEngine


class EditTool:
    name = "edit"

    def __init__(
        self, security: SecurityManager, policy: PolicyEngine, working_dir: str
    ) -> None:
        self.security = security
        self.policy = policy
        self.working_dir = working_dir

    def execute(self, arguments: dict) -> dict:
        path = arguments.get("path", "")
        old_string = arguments.get("old_string", "")
        new_string = arguments.get("new_string", "")
        if not path:
            return {"error": "No path provided"}

        full_path = os.path.join(self.working_dir, path) if not os.path.isabs(path) else path

        verdict = self.security.check_path(full_path)
        if not verdict.allowed:
            return {"error": f"Blocked: {verdict.reason}"}

        ext_verdict = self.policy.check_file_extension(full_path)
        if not ext_verdict.allowed:
            return {"error": f"Policy violation: {ext_verdict.reason}"}

        try:
            with open(full_path, "r") as f:
                content = f.read()
        except FileNotFoundError:
            return {"error": f"File not found: {path}"}

        if old_string not in content:
            return {"error": "old_string not found in file"}

        new_content = content.replace(old_string, new_string, 1)
        with open(full_path, "w") as f:
            f.write(new_content)
        return {"status": "ok", "path": full_path}
