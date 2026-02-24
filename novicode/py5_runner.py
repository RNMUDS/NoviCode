"""Headless py5 runner — renders sketches to PNG without opening a window.

Usage::

    python -m novicode.py5_runner sketch.py

Outputs a marker line to stdout::

    __PY5_OUTPUT__:/path/to/output.png
"""

from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path

OUTPUT_MARKER = "__PY5_OUTPUT__:"


# ── AST helpers ────────────────────────────────────────────────────────

def _has_function(source: str, name: str) -> bool:
    """Return True if *source* defines a top-level function called *name*."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.FunctionDef) and node.name == name
        for node in ast.iter_child_nodes(tree)
    )


def _extract_size(source: str) -> tuple[int, int]:
    """Extract (width, height) from a ``py5.size(w, h)`` or ``size(w, h)`` call.

    Falls back to (400, 400) if not found.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 400, 400

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Match py5.size(...) or size(...)
        func = node.func
        is_size = False
        if isinstance(func, ast.Attribute) and func.attr == "size":
            is_size = True
        elif isinstance(func, ast.Name) and func.id == "size":
            is_size = True
        if is_size and len(node.args) >= 2:
            try:
                w = ast.literal_eval(node.args[0])
                h = ast.literal_eval(node.args[1])
                if isinstance(w, (int, float)) and isinstance(h, (int, float)):
                    return int(w), int(h)
            except (ValueError, TypeError):
                pass
    return 400, 400


def has_draw(source: str) -> bool:
    """Return True if sketch defines a ``draw()`` function (animation)."""
    return _has_function(source, "draw")


# ── Runner ─────────────────────────────────────────────────────────────

def run_sketch(sketch_path: str, output_path: str | None = None) -> str:
    """Execute a py5 sketch headlessly and save the result as PNG.

    Parameters
    ----------
    sketch_path:
        Path to a ``.py`` file containing ``setup()`` and optionally ``draw()``.
    output_path:
        Where to save the PNG.  Defaults to ``<sketch_dir>/output.png``.

    Returns
    -------
    str
        Absolute path to the saved PNG.
    """
    import py5  # imported here so tests can run without py5 installed

    sketch_path = os.path.abspath(sketch_path)
    source = Path(sketch_path).read_text(encoding="utf-8")

    if output_path is None:
        output_path = os.path.join(os.path.dirname(sketch_path), "output.png")

    width, height = _extract_size(source)

    # Build a namespace that exposes py5 functions directly
    ns: dict = {"py5": py5, "__name__": "__main__"}

    # Inject all public py5 names so bare `size(...)`, `rect(...)` etc. work
    for attr in dir(py5):
        if not attr.startswith("_"):
            ns[attr] = getattr(py5, attr)

    # Execute the sketch source to define setup/draw in *ns*
    exec(compile(source, sketch_path, "exec"), ns)  # noqa: S102

    setup_fn = ns.get("setup")
    draw_fn = ns.get("draw")

    if draw_fn is not None:
        # Animation sketch — render a few frames, keep the last
        num_frames = 3

        def _setup_anim():
            py5.size(width, height)
            if setup_fn:
                setup_fn()

        def _draw_anim():
            draw_fn()

        frames = py5.render_frame_sequence(
            _setup_anim, _draw_anim, num_frames, width=width, height=height
        )
        if frames:
            frames[-1].save(output_path)
    else:
        # Static sketch — single frame
        def _draw_static():
            if setup_fn:
                setup_fn()

        img = py5.render_frame(_draw_static, width=width, height=height)
        img.save(output_path)

    return os.path.abspath(output_path)


# ── CLI entry point ────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m novicode.py5_runner <sketch.py>", file=sys.stderr)
        sys.exit(1)

    sketch_path = sys.argv[1]
    if not os.path.isfile(sketch_path):
        print(f"File not found: {sketch_path}", file=sys.stderr)
        sys.exit(1)

    output = run_sketch(sketch_path)
    print(f"{OUTPUT_MARKER}{output}")


if __name__ == "__main__":
    main()
