"""Bash tool — executes shell commands within security constraints."""

from __future__ import annotations

import subprocess
from rnnr.security_manager import SecurityManager, SecurityVerdict


class BashTool:
    """Runs a shell command after security validation."""

    name = "bash"

    def __init__(self, security: SecurityManager, working_dir: str) -> None:
        self.security = security
        self.working_dir = working_dir

    def execute(self, arguments: dict) -> dict:
        command = arguments.get("command", "")
        if not command:
            return {"error": "No command provided"}

        verdict = self.security.check_command(command)
        if not verdict.allowed:
            return {"error": f"Blocked: {verdict.reason}"}

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=self.working_dir,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            # Truncate long output
            if len(output) > 10000:
                output = output[:10000] + "\n... (truncated)"
            return {"output": output, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out (30s limit)"}
        except Exception as exc:
            return {"error": str(exc)}
