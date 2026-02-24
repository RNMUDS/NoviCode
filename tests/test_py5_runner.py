"""Tests for novicode.py5_runner — AST helpers and command rewriting."""

from __future__ import annotations

import pytest

from novicode.py5_runner import _has_function, _extract_size, has_draw


# ── _has_function ──────────────────────────────────────────────────

class TestHasFunction:
    def test_finds_setup(self):
        src = "def setup():\n    pass\n"
        assert _has_function(src, "setup") is True

    def test_finds_draw(self):
        src = "def draw():\n    pass\n"
        assert _has_function(src, "draw") is True

    def test_missing_function(self):
        src = "x = 1\n"
        assert _has_function(src, "draw") is False

    def test_nested_function_not_matched(self):
        src = "def setup():\n    def draw():\n        pass\n"
        # draw is nested, not top-level
        assert _has_function(src, "draw") is False

    def test_syntax_error_returns_false(self):
        assert _has_function("def :", "setup") is False


# ── _extract_size ──────────────────────────────────────────────────

class TestExtractSize:
    def test_py5_size(self):
        src = "def setup():\n    py5.size(800, 600)\n"
        assert _extract_size(src) == (800, 600)

    def test_bare_size(self):
        src = "def setup():\n    size(640, 480)\n"
        assert _extract_size(src) == (640, 480)

    def test_default_when_no_size(self):
        src = "def setup():\n    pass\n"
        assert _extract_size(src) == (400, 400)

    def test_default_on_syntax_error(self):
        assert _extract_size("def :") == (400, 400)

    def test_float_values_converted(self):
        src = "size(300.0, 200.0)\n"
        assert _extract_size(src) == (300, 200)


# ── has_draw ───────────────────────────────────────────────────────

class TestHasDraw:
    def test_static_sketch(self):
        src = "def setup():\n    py5.size(400, 400)\n    py5.rect(10, 10, 50, 50)\n"
        assert has_draw(src) is False

    def test_animation_sketch(self):
        src = (
            "def setup():\n    py5.size(400, 400)\n\n"
            "def draw():\n    py5.ellipse(200, 200, 50, 50)\n"
        )
        assert has_draw(src) is True


# ── BashTool py5 rewriting ─────────────────────────────────────────

class TestBashToolPy5Rewrite:
    def test_rewrite_in_py5_mode(self):
        from novicode.config import Mode
        from novicode.tools.bash_tool import BashTool
        from unittest.mock import MagicMock

        security = MagicMock()
        tool = BashTool(security, "/tmp", mode=Mode.PY5)
        assert tool._rewrite_for_py5("python sketch.py") == (
            "python -m novicode.py5_runner sketch.py"
        )

    def test_no_rewrite_in_python_mode(self):
        from novicode.config import Mode
        from novicode.tools.bash_tool import BashTool
        from unittest.mock import MagicMock

        security = MagicMock()
        tool = BashTool(security, "/tmp", mode=Mode.PYTHON_BASIC)
        assert tool._rewrite_for_py5("python sketch.py") == "python sketch.py"

    def test_rewrite_python3(self):
        from novicode.config import Mode
        from novicode.tools.bash_tool import BashTool
        from unittest.mock import MagicMock

        security = MagicMock()
        tool = BashTool(security, "/tmp", mode=Mode.PY5)
        assert tool._rewrite_for_py5("python3 my_sketch.py") == (
            "python -m novicode.py5_runner my_sketch.py"
        )

    def test_no_rewrite_non_python(self):
        from novicode.config import Mode
        from novicode.tools.bash_tool import BashTool
        from unittest.mock import MagicMock

        security = MagicMock()
        tool = BashTool(security, "/tmp", mode=Mode.PY5)
        assert tool._rewrite_for_py5("ls -la") == "ls -la"
