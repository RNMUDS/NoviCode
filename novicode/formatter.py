"""Streaming-aware code block formatter with syntax highlighting."""

from __future__ import annotations

from pygments import highlight
from pygments.formatters import TerminalTrueColorFormatter
from pygments.lexers import PythonLexer, get_lexer_by_name
from pygments.util import ClassNotFound


_FORMATTER = TerminalTrueColorFormatter(style="monokai")
_FENCE = "```"


def _highlight_code(code: str, lang: str) -> str:
    """Apply pygments syntax highlighting to *code* for terminal display."""
    try:
        lexer = get_lexer_by_name(lang)
    except ClassNotFound:
        lexer = PythonLexer()
    return highlight(code, lexer, _FORMATTER)


class StreamFormatter:
    """Detects fenced code blocks in a streaming text and highlights them.

    Usage::

        fmt = StreamFormatter()
        for chunk in stream:
            out = fmt.feed(chunk)
            if out:
                sys.stdout.write(out)
        remaining = fmt.flush()
        if remaining:
            sys.stdout.write(remaining)
    """

    def __init__(self) -> None:
        self._state: str = "text"   # "text" | "fence" | "code"
        self._buffer: str = ""      # accumulated code inside a fenced block
        self._lang: str = ""        # language tag (e.g. "python")
        self._pending: str = ""     # partial fence characters

    # ── public API ──────────────────────────────────────────────

    def feed(self, chunk: str) -> str:
        """Process *chunk* and return text ready for display.

        Text outside code fences is returned immediately.
        Code inside fences is buffered until the closing fence, then
        returned with syntax highlighting applied.
        """
        output: list[str] = []
        for ch in chunk:
            result = self._process_char(ch)
            if result:
                output.append(result)
        return "".join(output)

    def flush(self) -> str:
        """Flush any remaining buffered content (e.g. unclosed code block)."""
        if self._state == "code":
            result = _highlight_code(self._buffer, self._lang)
            self._buffer = ""
            self._lang = ""
            self._state = "text"
            self._pending = ""
            return result
        if self._pending:
            out = self._pending
            self._pending = ""
            return out
        return ""

    # ── internal state machine ──────────────────────────────────

    def _process_char(self, ch: str) -> str:
        if self._state == "text":
            return self._in_text(ch)
        if self._state == "fence":
            return self._in_fence(ch)
        return self._in_code(ch)

    def _in_text(self, ch: str) -> str:
        if ch == "`":
            self._pending += ch
            if self._pending == _FENCE:
                self._pending = ""
                self._state = "fence"
                self._lang = ""
                return ""
            return ""
        if self._pending:
            out = self._pending + ch
            self._pending = ""
            return out
        return ch

    def _in_fence(self, ch: str) -> str:
        """Collecting the language tag after opening ``` until newline."""
        if ch == "\n":
            self._state = "code"
            self._buffer = ""
            return ""
        self._lang += ch
        return ""

    def _in_code(self, ch: str) -> str:
        if ch == "`":
            self._pending += ch
            if self._pending == _FENCE:
                self._pending = ""
                result = _highlight_code(self._buffer, self._lang.strip())
                self._buffer = ""
                self._lang = ""
                self._state = "text"
                return result
            return ""
        if self._pending:
            self._buffer += self._pending + ch
            self._pending = ""
            return ""
        self._buffer += ch
        return ""
