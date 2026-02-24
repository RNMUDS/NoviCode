"""Inline image display for iTerm2 (OSC 1337) with fallback."""

from __future__ import annotations

import base64
import os
import sys


def is_iterm2() -> bool:
    """Return True if running inside iTerm2."""
    return os.environ.get("TERM_PROGRAM") == "iTerm2"


def display_image(path: str, *, width: str = "auto", height: str = "auto") -> None:
    """Display an image inline in the terminal.

    Uses iTerm2's OSC 1337 protocol when available, otherwise prints
    the file path as a fallback.
    """
    if not os.path.isfile(path):
        print(f"  [image not found: {path}]", file=sys.stderr)
        return

    if not is_iterm2():
        print(f"  [image saved: {path}]")
        return

    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")

    size = os.path.getsize(path)
    name_b64 = base64.b64encode(os.path.basename(path).encode()).decode("ascii")

    # OSC 1337 ; File=[args]:base64data ST
    params = f"name={name_b64};size={size};inline=1;width={width};height={height}"
    seq = f"\033]1337;File={params}:{data}\a"
    sys.stdout.write(seq)
    sys.stdout.write("\n")
    sys.stdout.flush()
