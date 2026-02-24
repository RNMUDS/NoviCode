"""Tests for novicode.imgcat — inline image display."""

from __future__ import annotations

import os
import tempfile
from unittest import mock

import pytest

from novicode.imgcat import is_iterm2, display_image


class TestIsIterm2:
    def test_true_when_env_set(self):
        with mock.patch.dict(os.environ, {"TERM_PROGRAM": "iTerm2"}):
            assert is_iterm2() is True

    def test_false_when_env_missing(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            assert is_iterm2() is False

    def test_false_when_other_terminal(self):
        with mock.patch.dict(os.environ, {"TERM_PROGRAM": "Apple_Terminal"}):
            assert is_iterm2() is False


class TestDisplayImage:
    def test_missing_file_prints_error(self, capsys):
        display_image("/nonexistent/file.png")
        captured = capsys.readouterr()
        assert "image not found" in captured.err

    def test_fallback_prints_path(self, tmp_path, capsys):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n")
        with mock.patch.dict(os.environ, {"TERM_PROGRAM": "other"}):
            display_image(str(img))
        captured = capsys.readouterr()
        assert "image saved" in captured.out
        assert str(img) in captured.out

    def test_iterm2_outputs_osc_sequence(self, tmp_path, capsys):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n")
        with mock.patch.dict(os.environ, {"TERM_PROGRAM": "iTerm2"}):
            display_image(str(img))
        captured = capsys.readouterr()
        assert "\033]1337;File=" in captured.out
        assert "inline=1" in captured.out
