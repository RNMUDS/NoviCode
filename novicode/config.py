"""Configuration constants and mode definitions for NoviCode."""

from __future__ import annotations

import json
import os
import platform
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet


# ── Ollama model discovery ──────────────────────────────────────────

def list_ollama_models(base_url: str | None = None) -> list[dict]:
    """Fetch installed models from Ollama's /api/tags endpoint.

    Returns a list of dicts with keys: name, size, modified_at.
    Returns an empty list on connection failure.
    """
    url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
    try:
        req = urllib.request.Request(f"{url}/api/tags")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return []

    models = []
    for m in data.get("models", []):
        models.append({
            "name": m.get("name", ""),
            "size": m.get("size", 0),
            "modified_at": m.get("modified_at", ""),
        })
    return models


RAM_THRESHOLD_GB = 32  # boundary for auto-selection


def get_system_ram_gb() -> float:
    """Return total physical RAM in GB (cross-platform)."""
    try:
        if platform.system() == "Darwin":
            import subprocess

            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
            return int(out.strip()) / (1024**3)
        else:
            mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            return mem / (1024**3)
    except Exception:
        return 16.0  # conservative fallback


def validate_model(name: str) -> str:
    """Validate and return the model name.

    - ``"auto"`` is returned as-is (caller handles interactive selection).
    - Any other string is accepted as a model name.
    """
    return name


# ── Modes ───────────────────────────────────────────────────────────

class Mode(str, Enum):
    PYTHON_BASIC = "python_basic"
    PY5 = "py5"
    SKLEARN = "sklearn"
    PANDAS = "pandas"
    WEB_BASIC = "web_basic"
    AFRAME = "aframe"
    THREEJS = "threejs"


class LanguageFamily(str, Enum):
    PYTHON = "python"
    WEB = "web"  # HTML + JavaScript


MODE_LANGUAGE: dict[Mode, LanguageFamily] = {
    Mode.PYTHON_BASIC: LanguageFamily.PYTHON,
    Mode.PY5: LanguageFamily.PYTHON,
    Mode.SKLEARN: LanguageFamily.PYTHON,
    Mode.PANDAS: LanguageFamily.PYTHON,
    Mode.WEB_BASIC: LanguageFamily.WEB,
    Mode.AFRAME: LanguageFamily.WEB,
    Mode.THREEJS: LanguageFamily.WEB,
}


# ── Allowed imports per mode ────────────────────────────────────────

ALLOWED_IMPORTS: dict[Mode, FrozenSet[str]] = {
    Mode.PYTHON_BASIC: frozenset({
        "math", "random", "string", "collections", "itertools",
        "functools", "operator", "copy", "pprint", "typing",
        "dataclasses", "enum", "json", "csv", "datetime", "re",
        "os.path", "pathlib", "textwrap", "decimal", "fractions",
        "statistics", "abc", "contextlib", "io", "struct",
    }),
    Mode.PY5: frozenset({
        "math", "random", "py5", "typing", "dataclasses", "enum",
        "collections", "itertools", "functools", "copy", "json",
    }),
    Mode.SKLEARN: frozenset({
        "math", "random", "numpy", "sklearn", "typing", "dataclasses",
        "collections", "itertools", "functools", "copy", "json",
        "csv", "pathlib", "os.path", "statistics", "warnings",
    }),
    Mode.PANDAS: frozenset({
        "math", "random", "numpy", "pandas", "matplotlib", "seaborn",
        "typing", "dataclasses", "collections", "itertools", "functools",
        "copy", "json", "csv", "pathlib", "os.path", "statistics",
        "warnings", "io",
    }),
    Mode.WEB_BASIC: frozenset(),  # no Python imports — web mode
    Mode.AFRAME: frozenset(),     # no Python imports — web mode
    Mode.THREEJS: frozenset(),    # no Python imports — web mode
}


# ── Allowed file extensions per language family ─────────────────────

ALLOWED_EXTENSIONS: dict[LanguageFamily, FrozenSet[str]] = {
    LanguageFamily.PYTHON: frozenset({".py"}),
    LanguageFamily.WEB: frozenset({".html", ".js", ".css"}),
}


# ── System prompts per mode ─────────────────────────────────────────

@dataclass(frozen=True)
class ModeProfile:
    mode: Mode
    language: LanguageFamily
    system_prompt: str
    allowed_imports: FrozenSet[str]
    allowed_extensions: FrozenSet[str]
    allowed_tools: FrozenSet[str]


def _python_tools() -> FrozenSet[str]:
    return frozenset({"bash", "read", "write", "edit", "grep", "glob"})


def _web_tools() -> FrozenSet[str]:
    return frozenset({"read", "write", "edit", "grep", "glob"})


_SYSTEM_PROMPTS: dict[Mode, str] = {
    Mode.PYTHON_BASIC: (
        "You are a Python tutor. Generate ONLY Python code. "
        "Use only the Python standard library. "
        "Do NOT generate HTML, JavaScript, or CSS. "
        "Focus on fundamentals: variables, loops, functions, classes, data structures, algorithms."
    ),
    Mode.PY5: (
        "You are a creative-coding tutor using Py5 (Processing for Python). "
        "Generate ONLY Python code. import py5 してから py5.size(), py5.rect() のように py5. プレフィックスで呼び出す。"
        "Always define a setup() function. Use py5.size() to set canvas size. "
        "For static images, put all drawing code in setup(). "
        "For animations, define both setup() and draw(). "
        "スクリプトの最後に必ず py5.run_sketch() を書く。"
        "Do NOT call py5.save() or py5.exit_sketch(). "
        "Focus on geometry, animation, color, and interactive sketches."
    ),
    Mode.SKLEARN: (
        "You are a machine-learning tutor using scikit-learn. "
        "Generate ONLY Python code. Allowed libraries: numpy, sklearn. "
        "Do NOT generate HTML, JavaScript, or CSS. "
        "Focus on statistics, classification, regression, clustering, and evaluation."
    ),
    Mode.PANDAS: (
        "You are a data-analysis tutor using pandas, matplotlib, and seaborn. "
        "Generate ONLY Python code. "
        "Do NOT generate HTML, JavaScript, or CSS. "
        "Focus on data loading, cleaning, tables, charts, and exploratory analysis."
    ),
    Mode.WEB_BASIC: (
        "You are a web development tutor. "
        "Generate HTML, CSS, and JavaScript code for web applications. "
        "Do NOT generate Python code. "
        "Focus on DOM manipulation, events, forms, and responsive design."
    ),
    Mode.AFRAME: (
        "You are a WebXR tutor using A-Frame. "
        "Generate ONLY HTML and JavaScript code using the A-Frame framework. "
        "Do NOT generate Python code. "
        "Focus on 3D scenes, entities, components, and immersive experiences."
    ),
    Mode.THREEJS: (
        "You are a 3D graphics tutor using Three.js. "
        "Generate ONLY HTML and JavaScript code using Three.js. "
        "Do NOT generate Python code. "
        "Focus on scenes, cameras, renderers, meshes, lights, and animation loops."
    ),
}


def build_mode_profile(mode: Mode, level: str = "beginner") -> ModeProfile:
    """Build a mode profile. The ``level`` parameter is stored for reference
    but does not alter which imports or tools are available."""
    lang = MODE_LANGUAGE[mode]
    return ModeProfile(
        mode=mode,
        language=lang,
        system_prompt=_SYSTEM_PROMPTS[mode],
        allowed_imports=ALLOWED_IMPORTS[mode],
        allowed_extensions=ALLOWED_EXTENSIONS[lang],
        allowed_tools=_python_tools() if lang == LanguageFamily.PYTHON else _web_tools(),
    )


# ── Defaults ────────────────────────────────────────────────────────

DEFAULT_MAX_ITERATIONS = 50
DEFAULT_MAX_FILES = 1
DEFAULT_MAX_LINES = 50
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
WORKING_DIR = os.environ.get("NOVICODE_WORKDIR", os.getcwd())

# Scope description for refusal messages
SCOPE_DESCRIPTION = """NoviCode supports ONLY these domains:
  1. Python fundamentals
  2. Py5 (Processing-style geometry & animation)
  3. scikit-learn (statistics & ML basics)
  4. pandas + matplotlib + seaborn (data analysis)
  5. HTML + CSS + JavaScript (Web basics)
  6. HTML + JavaScript with A-Frame (WebXR 3D)
  7. HTML + JavaScript with Three.js (3D graphics)

Requests outside these domains cannot be fulfilled."""
