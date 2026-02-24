"""Tests for StreamFormatter — streaming syntax highlight of fenced code blocks."""

import pytest
from novicode.formatter import StreamFormatter, _highlight_code


class TestHighlightCode:
    def test_returns_string(self):
        result = _highlight_code("x = 1", "python")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_ansi_codes(self):
        result = _highlight_code("x = 1", "python")
        assert "\x1b[" in result

    def test_unknown_language_falls_back(self):
        result = _highlight_code("x = 1", "nosuchlang")
        assert isinstance(result, str)
        assert len(result) > 0


class TestStreamFormatterBasicText:
    def test_plain_text_passthrough(self):
        fmt = StreamFormatter()
        assert fmt.feed("hello world") == "hello world"

    def test_empty_chunk(self):
        fmt = StreamFormatter()
        assert fmt.feed("") == ""

    def test_multiple_chunks(self):
        fmt = StreamFormatter()
        out = fmt.feed("hello ") + fmt.feed("world")
        assert out == "hello world"

    def test_flush_on_plain_text(self):
        fmt = StreamFormatter()
        fmt.feed("hello")
        assert fmt.flush() == ""


class TestStreamFormatterCodeBlock:
    def test_single_chunk_code_block(self):
        fmt = StreamFormatter()
        text = "```python\nx = 1\n```"
        out = fmt.feed(text)
        remaining = fmt.flush()
        full = out + remaining
        assert "\x1b[" in full

    def test_streamed_char_by_char(self):
        fmt = StreamFormatter()
        text = "before```python\nprint('hi')\n```after"
        out_parts = [fmt.feed(ch) for ch in text]
        out_parts.append(fmt.flush())
        full = "".join(out_parts)
        assert "before" in full
        assert "after" in full
        assert "\x1b[" in full

    def test_code_block_in_chunks(self):
        fmt = StreamFormatter()
        chunks = ["He", "llo\n", "```py", "thon\n", "x =", " 1\n", "```", "\ndone"]
        out = ""
        for c in chunks:
            out += fmt.feed(c)
        out += fmt.flush()
        assert "Hello" in out
        assert "done" in out
        assert "\x1b[" in out

    def test_unclosed_code_block_flushed(self):
        fmt = StreamFormatter()
        fmt.feed("```python\nx = 1\n")
        result = fmt.flush()
        assert "\x1b[" in result

    def test_multiple_code_blocks(self):
        fmt = StreamFormatter()
        text = "text1```python\na=1\n```middle```javascript\nvar b=2;\n```text2"
        out = fmt.feed(text) + fmt.flush()
        assert "text1" in out
        assert "middle" in out
        assert "text2" in out

    def test_single_backtick_in_text(self):
        fmt = StreamFormatter()
        out = fmt.feed("use `x` here") + fmt.flush()
        assert "use" in out
        assert "here" in out

    def test_double_backtick_in_text(self):
        fmt = StreamFormatter()
        out = fmt.feed("use ``x`` here") + fmt.flush()
        assert "use" in out
        assert "here" in out
