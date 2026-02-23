"""Configuration constants and mode definitions for NoviCode."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet


# ── Supported models ────────────────────────────────────────────────

SUPPORTED_MODELS: dict[str, dict] = {
    "qwen3:8b": {
        "min_ram_gb": 8,
        "context_length": 8192,
        "description": "Qwen3 8B — lightweight, suitable for ≤32 GB RAM",
    },
    "qwen3-coder:30b": {
        "min_ram_gb": 32,
        "context_length": 16384,
        "description": "Qwen3-Coder 30B — full capacity, requires ≥32 GB RAM",
    },
}

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


def auto_select_model() -> str:
    """Pick the best supported model based on available RAM."""
    ram = get_system_ram_gb()
    if ram >= RAM_THRESHOLD_GB:
        return "qwen3-coder:30b"
    return "qwen3:8b"


def validate_model(name: str) -> str:
    """Validate and return the canonical model name, or raise."""
    if name == "auto":
        return auto_select_model()
    if name not in SUPPORTED_MODELS:
        allowed = ", ".join(sorted(SUPPORTED_MODELS))
        raise ValueError(
            f"Unsupported model '{name}'. Allowed models: {allowed}"
        )
    return name


# ── Modes ───────────────────────────────────────────────────────────

class Mode(str, Enum):
    PYTHON_BASIC = "python_basic"
    PY5 = "py5"
    SKLEARN = "sklearn"
    PANDAS = "pandas"
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
    Mode.AFRAME: frozenset(),   # no Python imports — web mode
    Mode.THREEJS: frozenset(),  # no Python imports — web mode
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
        "Generate ONLY Python code using the py5 library. "
        "Do NOT generate HTML, JavaScript, or CSS. "
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
DEFAULT_MAX_LINES = 300
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
WORKING_DIR = os.environ.get("NOVICODE_WORKDIR", os.getcwd())

# Scope description for refusal messages
SCOPE_DESCRIPTION = """NoviCode supports ONLY these domains:
  1. Python fundamentals
  2. Py5 (Processing-style geometry & animation)
  3. scikit-learn (statistics & ML basics)
  4. pandas + matplotlib + seaborn (data analysis)
  5. HTML + JavaScript with A-Frame (WebXR 3D)
  6. HTML + JavaScript with Three.js (3D graphics)

Requests outside these domains cannot be fulfilled."""
