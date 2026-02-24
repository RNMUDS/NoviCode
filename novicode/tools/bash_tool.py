"""Bash tool — executes shell commands within security constraints."""

from __future__ import annotations

import re
import subprocess

from novicode.config import Mode
from novicode.security_manager import SecurityManager

# Pattern: `python <script>.py` (with optional flags before the script)
_PYTHON_SCRIPT_RE = re.compile(
    r"^(python3?\s+)(\S+\.py)(.*)$"
)


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

    # ── py5 command rewriting ──────────────────────────────────────

    def _rewrite_for_py5(self, command: str) -> str:
        """In py5 mode, rewrite ``python script.py`` to use py5_runner."""
        if self.mode != Mode.PY5:
            return command
        m = _PYTHON_SCRIPT_RE.match(command.strip())
        if m:
            script = m.group(2)
            rest = m.group(3)
            return f"python -m novicode.py5_runner {script}{rest}"
        return command

    # ── image display ─────────────────────────────────────────────

    @staticmethod
    def _show_inline_images(output: str) -> str:
        """Detect ``__PY5_OUTPUT__:`` markers and display images inline."""
        from novicode.py5_runner import OUTPUT_MARKER
        from novicode.imgcat import display_image

        clean_lines: list[str] = []
        for line in output.splitlines():
            if line.startswith(OUTPUT_MARKER):
                path = line[len(OUTPUT_MARKER):].strip()
                display_image(path)
            else:
                clean_lines.append(line)
        return "\n".join(clean_lines)

    # ── execute ───────────────────────────────────────────────────

    def execute(self, arguments: dict) -> dict:
        command = arguments.get("command", "")
        if not command:
            return {"error": "No command provided"}

        verdict = self.security.check_command(command)
        if not verdict.allowed:
            return {"error": f"Blocked: {verdict.reason}"}

        command = self._rewrite_for_py5(command)

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

            # Display inline images for py5 output
            if self.mode == Mode.PY5 and "__PY5_OUTPUT__:" in output:
                output = self._show_inline_images(output)

            return {"output": output, "returncode": result.returncode}
        except subprocess.TimeoutExpired:
            return {"error": "Command timed out (30s limit)"}
        except Exception as exc:
            return {"error": str(exc)}
