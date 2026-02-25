"""Bash tool — executes shell commands within security constraints."""

from __future__ import annotations

import re
import subprocess
import sys

from novicode.config import Mode
from novicode.security_manager import SecurityManager

# Pattern: `python <script>.py` (with optional flags before the script)
_PYTHON_SCRIPT_RE = re.compile(
    r"^python3?\s+\S+\.py(\s|$)"
)

# How long to wait for a py5 sketch before declaring the window is open
_PY5_STARTUP_TIMEOUT = 3


class BashTool:
    """Runs a shell command after security validation."""

    name = "bash"

    def __init__(
        self,
        security: SecurityManager,
        working_dir: str,
        mode: Mode | None = None,
    ) -> None:
        self.security = security
        self.working_dir = working_dir
        self.mode = mode

    # ── py5 detection ─────────────────────────────────────────────

    def _is_py5_script_command(self, command: str) -> bool:
        """Return True if *command* looks like ``python script.py``
        and we are in py5 mode."""
        if self.mode != Mode.PY5:
            return False
        return bool(_PYTHON_SCRIPT_RE.match(command.strip()))

    # ── py5 non-blocking execution ────────────────────────────────

    def _run_py5_script(self, command: str) -> dict:
        """Run a py5 sketch with Popen (non-blocking).

        * If the process exits within *_PY5_STARTUP_TIMEOUT* seconds the
          output / error is captured and returned.
        * If it is still running after the timeout we assume the sketch
          window opened successfully.
        """
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=self.working_dir,
        )
        try:
            stdout, stderr = proc.communicate(timeout=_PY5_STARTUP_TIMEOUT)
            # Process exited quickly — likely an error
            output = stdout
            if stderr:
                output += f"\nSTDERR:\n{stderr}"

            # Auto-install py5 if missing
            if "No module named 'py5'" in (stderr or ""):
                return self._handle_py5_missing(command)

            return {"output": output, "returncode": proc.returncode}
        except subprocess.TimeoutExpired:
            # Still running → window is open
            return {"output": "スケッチウィンドウが開きました。", "returncode": 0}

    # ── py5 auto-install ──────────────────────────────────────────

    def _handle_py5_missing(self, original_command: str) -> dict:
        """Auto-install py5 when it is not found, then retry."""
        install = subprocess.run(
            [sys.executable, "-m", "pip", "install", "py5"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if install.returncode != 0:
            return {
                "error": "py5 のインストールに失敗しました。\n" + install.stderr,
                "returncode": 1,
            }
        # Retry the original command
        return self._run_py5_script(original_command)

    # ── execute ───────────────────────────────────────────────────

    def execute(self, arguments: dict) -> dict:
        command = arguments.get("command", "")
        if not command:
            return {"error": "No command provided"}

        verdict = self.security.check_command(command)
        if not verdict.allowed:
            return {"error": f"Blocked: {verdict.reason}"}

        # py5 mode: non-blocking window execution
        if self._is_py5_script_command(command):
            return self._run_py5_script(command)

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
