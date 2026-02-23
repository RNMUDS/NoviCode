"""Tests for config module."""

import pytest
from rnnr.config import (
    validate_model,
    auto_select_model,
    Mode,
    build_mode_profile,
    LanguageFamily,
    SUPPORTED_MODELS,
)


def test_validate_model_accepts_supported():
    assert validate_model("qwen3:8b") == "qwen3:8b"
    assert validate_model("qwen3-coder:30b") == "qwen3-coder:30b"


def test_validate_model_rejects_unsupported():
    with pytest.raises(ValueError, match="Unsupported model"):
        validate_model("gpt-4")


def test_validate_model_auto():
    result = validate_model("auto")
    assert result in SUPPORTED_MODELS


def test_auto_select_returns_supported():
    model = auto_select_model()
    assert model in SUPPORTED_MODELS


def test_build_mode_profile_python():
    profile = build_mode_profile(Mode.PYTHON_BASIC)
    assert profile.language == LanguageFamily.PYTHON
    assert ".py" in profile.allowed_extensions
    assert ".html" not in profile.allowed_extensions
    assert "bash" in profile.allowed_tools


def test_build_mode_profile_web():
    profile = build_mode_profile(Mode.AFRAME)
    assert profile.language == LanguageFamily.WEB
    assert ".html" in profile.allowed_extensions
    assert ".py" not in profile.allowed_extensions
    assert "bash" not in profile.allowed_tools


def test_all_modes_have_profiles():
    for mode in Mode:
        profile = build_mode_profile(mode)
        assert profile.mode == mode
        assert profile.system_prompt
        assert profile.allowed_extensions
