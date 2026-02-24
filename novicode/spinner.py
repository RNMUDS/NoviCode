"""Terminal spinner for showing progress during LLM / tool execution."""

from __future__ import annotations

import sys
import threading
import time

_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_INTERVAL = 0.08  # seconds between frames


class Spinner:
    """Thread-based braille-dot spinner that writes to *stderr*.

    * ``start(msg)`` — begin spinning (or update message if already running).
    * ``update(msg)`` — alias for ``start(msg)`` while running.
    * ``stop()`` — clear the spinner line and stop the thread.

    The spinner is a no-op when *stderr* is not a TTY.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._msg: str = ""
        self._running = False
        self._thread: threading.Thread | None = None
        self._is_tty: bool = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()

    # ── public API ────────────────────────────────────────────

    def start(self, msg: str = "処理中...") -> None:
        """Start the spinner with *msg*, or update the message if already running."""
        with self._lock:
            self._msg = msg
            if self._running:
                return  # already spinning — just update message
            self._running = True
        if not self._is_tty:
            return
        t = threading.Thread(target=self._spin, daemon=True)
        self._thread = t
        t.start()

    def update(self, msg: str) -> None:
        """Update the spinner message (no-op if not running)."""
        with self._lock:
            self._msg = msg

    def stop(self) -> None:
        """Stop the spinner and clear the line."""
        with self._lock:
            if not self._running:
                return
            self._running = False
        t = self._thread
        if t is not None:
            t.join(timeout=1.0)
            self._thread = None
        if self._is_tty:
            sys.stderr.write("\033[2K\r")
            sys.stderr.flush()

    # ── internals ─────────────────────────────────────────────

    def _spin(self) -> None:
        idx = 0
        while True:
            with self._lock:
                if not self._running:
                    break
                msg = self._msg
            frame = _FRAMES[idx % len(_FRAMES)]
            sys.stderr.write(f"\033[2K\r  {frame} {msg}")
            sys.stderr.flush()
            idx += 1
            time.sleep(_INTERVAL)
