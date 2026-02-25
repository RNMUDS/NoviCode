"""Tests for config module."""

import json
from unittest.mock import patch, MagicMock

import pytest
from novicode.config import (
    validate_model,
    list_ollama_models,
    Mode,
    build_mode_profile,
    LanguageFamily,
)


# ── validate_model ──────────────────────────────────────────────────

def test_validate_model_auto():
    assert validate_model("auto") == "auto"


def test_validate_model_passthrough():
    assert validate_model("gpt-oss-swallow:20b-rl") == "gpt-oss-swallow:20b-rl"
    assert validate_model("qwen3:8b") == "qwen3:8b"
    assert validate_model("my-custom-model:latest") == "my-custom-model:latest"


# ── list_ollama_models ──────────────────────────────────────────────

def test_list_ollama_models_success():
    fake_response = json.dumps({
        "models": [
            {"name": "qwen3:8b", "size": 5268045824, "modified_at": "2025-01-01T00:00:00Z"},
            {"name": "llama3:latest", "size": 4100000000, "modified_at": "2025-01-02T00:00:00Z"},
        ]
    }).encode()

    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_response
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        models = list_ollama_models("http://localhost:11434")

    assert len(models) == 2
    assert models[0]["name"] == "qwen3:8b"
    assert models[0]["size"] == 5268045824
    assert models[1]["name"] == "llama3:latest"


def test_list_ollama_models_connection_failure():
    with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
        models = list_ollama_models("http://localhost:11434")
    assert models == []


def test_list_ollama_models_empty():
    fake_response = json.dumps({"models": []}).encode()

    mock_resp = MagicMock()
    mock_resp.read.return_value = fake_response
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)

    with patch("urllib.request.urlopen", return_value=mock_resp):
        models = list_ollama_models("http://localhost:11434")

    assert models == []


# ── build_mode_profile ──────────────────────────────────────────────

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
