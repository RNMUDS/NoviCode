"""Raw terminal input reader with kitty keyboard protocol support.

Provides multi-line editing with:
- Enter: newline
- Shift+Enter: send (kitty protocol ``\\x1b[13;2u``) — if supported
- Enter on empty line: send (fallback for terminals without kitty)
- Ctrl+D: send (always works)
- ESC: exit (bare ESC with no follow-up within 50 ms)
- Backspace: delete character
- Ctrl+C: raise KeyboardInterrupt
"""

from __future__ import annotations

import os
import select
import sys
import termios
import tty
import unicodedata
from dataclasses import dataclass


@dataclass
class InputResult:
    """Return value from :meth:`InputReader.read_input`."""
    text: str
    action: str  # "send" | "exit"


# ── ANSI helpers ─────────────────────────────────────────────────────

_GREEN = "\033[38;2;118;185;0m"
_BOLD = "\033[1m"
_DIM = "\033[90m"
_RESET = "\033[0m"

_KITTY_ENABLE = "\x1b[>1u"    # Push mode 1 (disambiguate)
_KITTY_DISABLE = "\x1b[<u"    # Pop keyboard mode
_KITTY_QUERY = "\x1b[?u"      # Query current keyboard mode


def _write(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.flush()


def _char_width(ch: str) -> int:
    """Return the terminal display width of a single character."""
    if len(ch) != 1:
        return sum(_char_width(c) for c in ch)
    cat = unicodedata.east_asian_width(ch)
    return 2 if cat in ("F", "W") else 1


def _has_data(fd: int, timeout: float = 0.05) -> bool:
    """Return True if *fd* has data ready within *timeout* seconds."""
    r, _, _ = select.select([fd], [], [], timeout)
    return bool(r)


def _drain(fd: int) -> None:
    """Drain any pending bytes from *fd*."""
    while _has_data(fd, 0.01):
        os.read(fd, 256)


class InputReader:
    """Read multi-line input from a raw terminal.

    On terminals that support the kitty keyboard protocol, Shift+Enter
    sends the input.  On other terminals (Terminal.app, iTerm2, etc.),
    pressing Enter on an **empty line** (after content) sends instead.

    Parameters
    ----------
    prompt_first : str
        Prompt string for the first line (e.g. ``"You> "``).
    prompt_cont : str
        Prompt string for continuation lines.
    box_top : str
        Decorative line printed above the input area.
    box_bottom : str
        Decorative line printed below the input area.
    """

    def __init__(
        self,
        prompt_first: str = "You> ",
        prompt_cont: str = "  .. ",
        box_top: str = "",
        box_bottom: str = "",
    ) -> None:
        self.prompt_first = prompt_first
        self.prompt_cont = prompt_cont
        self.box_top = box_top
        self.box_bottom = box_bottom
        self._fd = sys.stdin.fileno()
        self._old_attr: list | None = None
        self._kitty_supported = False

    @property
    def send_hint(self) -> str:
        """Return keybind hint text appropriate for the current terminal."""
        if self._kitty_supported:
            return "Enter: 改行  Shift+Enter: 送信  ESC: 終了"
        return "Enter: 改行  空Enter: 送信  ESC: 終了"

    # ── Context manager ──────────────────────────────────────────

    def __enter__(self) -> "InputReader":
        self._enable_raw()
        return self

    def __exit__(self, *exc: object) -> None:
        self._disable_raw()

    # ── Public API ───────────────────────────────────────────────

    def read_input(self) -> InputResult:
        """Block until the user sends input or requests exit.

        Returns an :class:`InputResult` with the collected text and
        the action that terminated the input (``"send"`` or ``"exit"``).
        """
        lines: list[str] = [""]
        cur_line = 0

        if self.box_top:
            _write(self.box_top + "\n")
        _write(self.prompt_first)

        while True:
            ch = self._read_char()

            if ch is None:
                continue

            # ── Ctrl+C ───────────────────────────────────────────
            if ch == "\x03":
                _write("\n")
                if self.box_bottom:
                    _write(self.box_bottom + "\n")
                raise KeyboardInterrupt

            # ── Ctrl+D (send) ────────────────────────────────────
            if ch == "\x04":
                _write("\n")
                if self.box_bottom:
                    _write(self.box_bottom + "\n")
                text = "\n".join(lines).strip()
                if not text:
                    return InputResult(text="", action="exit")
                return InputResult(text=text, action="send")

            # ── ESC handling ─────────────────────────────────────
            if ch == "\x1b":
                result = self._handle_escape(lines, cur_line)
                if result is not None:
                    return result
                continue

            # ── Enter ────────────────────────────────────────────
            if ch in ("\r", "\n"):
                # Fallback send: Enter on empty line when content exists
                has_content = any(l.strip() for l in lines)
                if not self._kitty_supported and has_content and lines[cur_line] == "":
                    _write("\n")
                    if self.box_bottom:
                        _write(self.box_bottom + "\n")
                    return InputResult(
                        text="\n".join(lines[:-1]).strip(),
                        action="send",
                    )
                # Normal newline
                _write("\n")
                lines.append("")
                cur_line += 1
                _write(self.prompt_cont)
                continue

            # ── Backspace ────────────────────────────────────────
            if ch in ("\x7f", "\x08"):
                if lines[cur_line]:
                    removed = lines[cur_line][-1]
                    lines[cur_line] = lines[cur_line][:-1]
                    w = _char_width(removed)
                    _write("\b" * w + " " * w + "\b" * w)
                elif cur_line > 0:
                    # Join with previous line
                    lines.pop(cur_line)
                    cur_line -= 1
                    self._redraw(lines, cur_line)
                continue

            # ── Printable character ──────────────────────────────
            if ch >= " " or ch == "\t":
                lines[cur_line] += ch
                _write(ch)

    # ── Escape sequence handling ─────────────────────────────────

    def _handle_escape(self, lines: list[str], cur_line: int) -> InputResult | None:
        """Process an ESC byte. Returns InputResult if action triggered."""
        if not _has_data(self._fd, 0.05):
            # Bare ESC → exit
            _write("\n")
            if self.box_bottom:
                _write(self.box_bottom + "\n")
            return InputResult(text="", action="exit")

        # Read the escape sequence
        seq = self._read_escape_seq()

        # Kitty protocol: Shift+Enter = \x1b[13;2u
        if seq == "[13;2u":
            _write("\n")
            if self.box_bottom:
                _write(self.box_bottom + "\n")
            text = "\n".join(lines).strip()
            if not text:
                return None  # empty → ignore
            return InputResult(text=text, action="send")

        # Other escape sequences (arrow keys, etc.) — ignore for now
        return None

    def _read_escape_seq(self) -> str:
        """Read bytes following ESC until sequence is complete."""
        seq = ""
        while _has_data(self._fd, 0.01):
            b = os.read(self._fd, 1).decode("utf-8", errors="replace")
            seq += b
            # CSI sequences end with a letter or ~ or u
            if b.isalpha() or b in ("~", "u"):
                break
        return seq

    # ── Terminal mode ────────────────────────────────────────────

    def _enable_raw(self) -> None:
        try:
            self._old_attr = termios.tcgetattr(self._fd)
            tty.setraw(self._fd)
            # Re-enable output processing so \n works as expected
            attr = termios.tcgetattr(self._fd)
            attr[1] |= termios.OPOST
            termios.tcsetattr(self._fd, termios.TCSANOW, attr)
            # Try to enable kitty keyboard protocol and probe support
            self._kitty_supported = self._probe_kitty()
        except (OSError, termios.error):
            pass

    def _probe_kitty(self) -> bool:
        """Enable kitty keyboard protocol and check if the terminal supports it.

        Sends the enable sequence followed by a query.  If the terminal
        responds within 150 ms, kitty protocol is available.
        """
        _write(_KITTY_ENABLE)
        _write(_KITTY_QUERY)
        # A supporting terminal replies with \x1b[?{flags}u
        if _has_data(self._fd, 0.15):
            _drain(self._fd)
            return True
        return False

    def _disable_raw(self) -> None:
        if self._kitty_supported:
            try:
                _write(_KITTY_DISABLE)
            except (OSError, ValueError):
                pass
            self._kitty_supported = False
        if self._old_attr is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSANOW, self._old_attr)
            except (OSError, termios.error):
                pass
            self._old_attr = None

    def _read_char(self) -> str | None:
        """Read a single UTF-8 character from stdin.

        Inspects the leading byte to determine how many continuation
        bytes are needed, then reads and decodes the full sequence.
        """
        try:
            b = os.read(self._fd, 1)
            if not b:
                return None
            first = b[0]
            # Determine expected UTF-8 byte length from leading byte
            if first < 0x80:
                # ASCII (1 byte)
                return b.decode("utf-8")
            elif first < 0xC0:
                # Unexpected continuation byte — return replacement
                return b.decode("utf-8", errors="replace")
            elif first < 0xE0:
                remaining = 1  # 2-byte sequence (e.g. Latin accents)
            elif first < 0xF0:
                remaining = 2  # 3-byte sequence (e.g. CJK, Japanese)
            else:
                remaining = 3  # 4-byte sequence (e.g. emoji)
            # Read continuation bytes
            for _ in range(remaining):
                extra = os.read(self._fd, 1)
                if not extra:
                    break
                b += extra
            return b.decode("utf-8", errors="replace")
        except OSError:
            return None

    def _redraw(self, lines: list[str], cur_line: int) -> None:
        """Redraw the current input from scratch (after backspace across lines)."""
        # Move cursor up to first line and clear everything
        for _ in range(cur_line):
            _write("\033[A")  # cursor up
        _write("\r\033[J")  # clear from cursor to end of screen

        for i, line in enumerate(lines):
            prompt = self.prompt_first if i == 0 else self.prompt_cont
            _write(prompt + line)
            if i < len(lines) - 1:
                _write("\n")
