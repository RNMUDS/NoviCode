"""Write tool — writes files within security boundaries and policy constraints."""

from __future__ import annotations

import os
from novicode.security_manager import SecurityManager
from novicode.policy_engine import PolicyEngine


class WriteTool:
    name = "write"

    def __init__(
        self, security: SecurityManager, policy: PolicyEngine, working_dir: str
    ) -> None:
        self.security = security
        self.policy = policy
        self.working_dir = working_dir

    def execute(self, arguments: dict) -> dict:
        path = arguments.get("path", "")
        content = arguments.get("content", "")
        if not path:
            return {"error": "No path provided"}

        full_path = os.path.join(self.working_dir, path) if not os.path.isabs(path) else path

        # Security check
        verdict = self.security.check_path(full_path)
        if not verdict.allowed:
            return {"error": f"Blocked: {verdict.reason}"}

        # Policy: file extension check
        ext_verdict = self.policy.check_file_extension(full_path)
        if not ext_verdict.allowed:
            return {"error": f"Policy violation: {ext_verdict.reason}"}

        try:
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w") as f:
                f.write(content)
            return {"status": "ok", "path": full_path, "bytes": len(content)}
        except Exception as exc:
            return {"error": str(exc)}
