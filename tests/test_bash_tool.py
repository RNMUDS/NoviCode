"""Tests for bash_tool — especially py5 window execution."""

from __future__ import annotations

import subprocess
import tempfile
from unittest.mock import patch, MagicMock

import pytest

from novicode.config import Mode
from novicode.security_manager import SecurityManager
from novicode.tools.bash_tool import BashTool, _PY5_STARTUP_TIMEOUT


@pytest.fixture
def tmpworkdir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def py5_tool(tmpworkdir):
    """BashTool in py5 mode."""
    sec = SecurityManager(tmpworkdir)
    return BashTool(sec, tmpworkdir, mode=Mode.PY5)


@pytest.fixture
def plain_tool(tmpworkdir):
    """BashTool without a mode (plain Python)."""
    sec = SecurityManager(tmpworkdir)
    return BashTool(sec, tmpworkdir)


# ── py5 detection ─────────────────────────────────────────────────

class TestPy5Detection:
    def test_detects_python_script(self, py5_tool):
        assert py5_tool._is_py5_script_command("python sketch.py")

    def test_detects_python3_script(self, py5_tool):
        assert py5_tool._is_py5_script_command("python3 sketch.py")

    def test_ignores_non_script(self, py5_tool):
        assert not py5_tool._is_py5_script_command("ls -la")

    def test_ignores_when_not_py5_mode(self, plain_tool):
        assert not plain_tool._is_py5_script_command("python sketch.py")


# ── py5 window execution (Popen) ─────────────────────────────────

class TestPy5WindowExecution:
    def test_window_opened_on_timeout(self, py5_tool):
        """When the process doesn't exit within timeout, report window opened."""
        mock_proc = MagicMock()
        mock_proc.communicate.side_effect = subprocess.TimeoutExpired(
            cmd="python sketch.py", timeout=_PY5_STARTUP_TIMEOUT
        )
        mock_proc.stderr = MagicMock()

        with patch("novicode.tools.bash_tool.subprocess.Popen", return_value=mock_proc):
            result = py5_tool.execute({"command": "python sketch.py"})

        assert result["returncode"] == 0
        assert "スケッチウィンドウが開きました" in result["output"]
        mock_proc.stderr.close.assert_called_once()

    def test_immediate_error_captured(self, py5_tool):
        """When the process exits quickly with an error, capture stderr."""
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (None, "SyntaxError: invalid syntax")
        mock_proc.returncode = 1

        with patch("novicode.tools.bash_tool.subprocess.Popen", return_value=mock_proc):
            result = py5_tool.execute({"command": "python sketch.py"})

        assert result["returncode"] == 1
        assert "SyntaxError" in result["output"]

    def test_normal_exit_no_stderr(self, py5_tool):
        """Normal exit with no stderr → empty output (stdout goes to DEVNULL)."""
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (None, "")
        mock_proc.returncode = 0

        with patch("novicode.tools.bash_tool.subprocess.Popen", return_value=mock_proc):
            result = py5_tool.execute({"command": "python sketch.py"})

        assert result["returncode"] == 0
        assert result["output"] == ""


# ── py5 auto-install ──────────────────────────────────────────────

class TestPy5AutoInstall:
    def test_auto_install_on_missing_module(self, py5_tool):
        """If py5 is not installed, auto-install and retry."""
        # First call: process exits quickly with ModuleNotFoundError
        mock_proc_fail = MagicMock()
        mock_proc_fail.communicate.return_value = (
            None,
            "ModuleNotFoundError: No module named 'py5'",
        )
        mock_proc_fail.returncode = 1

        # Retry call: process hangs (window opened)
        mock_proc_ok = MagicMock()
        mock_proc_ok.communicate.side_effect = subprocess.TimeoutExpired(
            cmd="python sketch.py", timeout=_PY5_STARTUP_TIMEOUT
        )
        mock_proc_ok.stderr = MagicMock()

        # pip install succeeds
        mock_install = MagicMock()
        mock_install.returncode = 0
        mock_install.stderr = ""

        with patch(
            "novicode.tools.bash_tool.subprocess.Popen",
            side_effect=[mock_proc_fail, mock_proc_ok],
        ), patch(
            "novicode.tools.bash_tool.subprocess.run",
            return_value=mock_install,
        ):
            result = py5_tool.execute({"command": "python sketch.py"})

        assert result["returncode"] == 0
        assert "スケッチウィンドウが開きました" in result["output"]

    def test_auto_install_failure(self, py5_tool):
        """If pip install fails, return error."""
        mock_proc = MagicMock()
        mock_proc.communicate.return_value = (
            None,
            "ModuleNotFoundError: No module named 'py5'",
        )
        mock_proc.returncode = 1

        mock_install = MagicMock()
        mock_install.returncode = 1
        mock_install.stderr = "ERROR: Could not install py5"

        with patch(
            "novicode.tools.bash_tool.subprocess.Popen",
            return_value=mock_proc,
        ), patch(
            "novicode.tools.bash_tool.subprocess.run",
            return_value=mock_install,
        ):
            result = py5_tool.execute({"command": "python sketch.py"})

        assert "error" in result
        assert "インストールに失敗" in result["error"]


# ── non-py5 execution (unchanged behaviour) ───────────────────────

class TestNonPy5Execution:
    def test_plain_echo(self, plain_tool):
        result = plain_tool.execute({"command": "echo hello"})
        assert "hello" in result["output"]
        assert result["returncode"] == 0

    def test_blocked_command(self, plain_tool):
        result = plain_tool.execute({"command": "sudo rm -rf /"})
        assert "error" in result

    def test_empty_command(self, plain_tool):
        result = plain_tool.execute({"command": ""})
        assert "error" in result

    def test_no_command_key(self, plain_tool):
        result = plain_tool.execute({})
        assert "error" in result

    def test_timeout_handling(self, plain_tool):
        result = plain_tool.execute({"command": "sleep 60"})
        assert "error" in result

    def test_nonzero_returncode(self, plain_tool):
        result = plain_tool.execute({"command": "python3 -c 'exit(1)'"})
        assert result.get("returncode") == 1


# ── py5 mode does NOT rewrite non-python commands ─────────────────

class TestPy5ModeNonScript:
    def test_ls_in_py5_mode(self, py5_tool):
        """Non-python commands in py5 mode should use normal execution."""
        result = py5_tool.execute({"command": "echo test"})
        assert "test" in result["output"]
        assert result["returncode"] == 0
