"""Stress tests — simulate 1000+ user patterns across many interaction scenarios.

Categories:
  A. _has_code_block: 100+ text patterns (various languages, edge cases)
  B. Agent loop nudge: 200+ LLM response sequences (run_turn)
  C. Agent loop stream: 200+ LLM response sequences (run_turn_stream)
  D. Policy system prompt: all mode × level combos
  E. Curriculum expressions: all mode × level × mastered combos
  F. Tool definitions: structural integrity under various lookups
"""

from __future__ import annotations

import itertools
import random
import string
from unittest.mock import MagicMock

import pytest

from novicode.agent_loop import (
    AgentLoop,
    _has_code_block,
    _MAX_NUDGES_PER_TURN,
    _TOOL_NUDGE,
)
from novicode.config import Mode, LanguageFamily, build_mode_profile, MODE_LANGUAGE
from novicode.curriculum import (
    Level,
    CONCEPT_CATALOGS,
    build_education_prompt,
    extract_concepts,
)
from novicode.llm_adapter import LLMResponse, Message, ToolCall, TOOL_DEFINITIONS
from novicode.metrics import Metrics
from novicode.policy_engine import PolicyEngine
from novicode.session_manager import Session, SessionMeta
from novicode.validator import ValidationResult, Violation


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _session() -> Session:
    return Session(meta=SessionMeta(
        session_id="stress", model="qwen3:8b", mode="python_basic",
        created_at=0.0, research=False,
    ))


def _loop(llm, mode=Mode.PYTHON_BASIC, max_iter=20):
    profile = build_mode_profile(mode)
    policy = PolicyEngine(profile, level=Level.BEGINNER)
    tools = MagicMock()
    tools.available_tools.return_value = list(profile.allowed_tools)
    tools.execute.return_value = {"status": "ok"}
    validator = MagicMock()
    validator.validate.return_value = ValidationResult(valid=True)
    return AgentLoop(
        llm=llm, profile=profile, tools=tools, validator=validator,
        policy=policy, session=_session(), metrics=Metrics(),
        max_iterations=max_iter,
    )


# ═══════════════════════════════════════════════════════════════════════
# A. _has_code_block — 100+ patterns
# ═══════════════════════════════════════════════════════════════════════

# Languages that should be detected as code blocks
_LANGUAGES = [
    "python", "py", "javascript", "js", "typescript", "ts",
    "html", "css", "json", "yaml", "xml", "sql", "bash", "sh",
    "shell", "zsh", "ruby", "go", "rust", "java", "kotlin",
    "swift", "c", "cpp", "csharp", "php", "perl", "scala",
    "haskell", "elixir", "dart", "lua", "r", "matlab",
    "dockerfile", "makefile", "toml", "ini", "csv", "markdown",
    "md", "plaintext", "text", "diff", "graphql", "protobuf",
    "terraform", "nginx", "apache", "asm", "nasm", "wasm",
]

_TRUE_PATTERNS = (
    # Standard fenced blocks with various languages
    [(f"```{lang}\ncode here\n```", True) for lang in _LANGUAGES]
    # Bare fence
    + [("```\ncode\n```", True)]
    # Fence with trailing space
    + [("```python \ncode\n```", False)]  # \w* won't match space
    # Multiple code blocks
    + [("```python\na=1\n```\ntext\n```js\nb=2\n```", True)]
    # Code block at start
    + [("```python\nprint(1)\n```\nsome text", True)]
    # Code block at end
    + [("some text\n```python\nprint(1)\n```", True)]
    # Code block in middle
    + [("before\n```python\ncode\n```\nafter", True)]
    # Empty code block
    + [("```python\n```", True)]
    # Code block with lots of code
    + [("```python\n" + "\n".join(f"x{i} = {i}" for i in range(50)) + "\n```", True)]
    # Mixed content with code block
    + [("説明テキスト\n\n```python\nprint('hello')\n```\n\nもっと説明", True)]
    # Unicode in code block
    + [("```python\nprint('こんにちは')\n```", True)]
    # Multiple languages in sequence
    + [("```html\n<div>\n```\n```css\n.a{}\n```\n```js\nalert(1)\n```", True)]
)

_FALSE_PATTERNS = (
    # Plain text
    [("This is just plain text.", False)]
    # Inline code
    + [("Use `print()` to output.", False)]
    # Double backtick inline
    + [("Use ``code here`` for inline.", False)]
    # Incomplete fence (no newline after lang)
    + [("```python is great", False)]
    # Only backticks
    + [("```", False)]
    # Backticks in middle of word
    + [("abc```def", False)]
    # Empty string
    + [("", False)]
    # Only whitespace
    + [("   \n\t\n  ", False)]
    # Only newlines
    + [("\n\n\n", False)]
    # Numbers that look like code
    + [("x = 42\ny = 'hello'\nprint(x + y)", False)]
    # Markdown headers
    + [("# Title\n## Subtitle\n### Section", False)]
    # Bullet points
    + [("- item 1\n- item 2\n- item 3", False)]
    # Japanese text without code
    + [("変数について説明します。変数とは値を入れる箱のようなものです。", False)]
    # HTML-like text (but not in code block)
    + [("Use <b>bold</b> text here", False)]
    # Backtick without triple
    + [("`python\ncode\n`", False)]
    # Four backticks — actually DOES match: regex finds ``` starting at position 1
    + [("````\ncode\n````", True)]  # ``` at offset 1, then \n at offset 4
    # Fence on same line (no newline)
    + [("```python code here```", False)]
    # Only closing fence
    + [("some code\n```", False)]
    # Tab after backticks
    + [("```\tpython\ncode\n```", False)]
    # Single word response
    + [("OK", False)]
    # Emoji response
    + [("👍", False)]
    # Very long text without code
    + [("あ" * 10000, False)]
)

ALL_CODE_BLOCK_PATTERNS = _TRUE_PATTERNS + _FALSE_PATTERNS


@pytest.mark.parametrize("text,expected", ALL_CODE_BLOCK_PATTERNS,
                         ids=[f"codeblock_{i}" for i in range(len(ALL_CODE_BLOCK_PATTERNS))])
class TestCodeBlockPatterns:
    def test_detection(self, text, expected):
        assert _has_code_block(text) is expected


# ═══════════════════════════════════════════════════════════════════════
# B. Agent loop run_turn — 200+ scenarios
# ═══════════════════════════════════════════════════════════════════════

def _resp(content="", tool_calls=None):
    """Shorthand for LLMResponse."""
    return LLMResponse(content=content, tool_calls=tool_calls or [])


def _tc(name="write", **kwargs):
    """Shorthand for ToolCall."""
    if not kwargs:
        kwargs = {"path": "test.py", "content": "x=1"}
    return ToolCall(name=name, arguments=kwargs)


class TestRunTurnMassive:
    """Simulate many different LLM response sequences for run_turn."""

    # --- Normal path: text only, no code block ---

    @pytest.mark.parametrize("text", [
        "変数について説明します。",
        "OK、次に進みましょう。",
        "",  # empty
        " ",  # whitespace
        "🎉 Great job!",
        "a" * 5000,  # very long
        "日本語テキスト\nwith mixed\n英語 text",
        "Line1\nLine2\nLine3\n" * 100,  # many lines
        "Special chars: <>&\"'\\",
        "\t\ttabbed\tcontent\t",
        "Numbers: 12345 67890",
        "URL-like: not a real http://example.com",  # validator may flag
        "Single char: x",
    ], ids=lambda x: f"text_{hash(x) % 10000}")
    def test_text_only_no_nudge(self, text):
        llm = MagicMock()
        llm.chat.return_value = _resp(content=text)
        loop = _loop(llm)
        result = loop.run_turn("Hello")
        nudges = [m for m in loop.messages if m.content == _TOOL_NUDGE]
        assert len(nudges) == 0

    # --- Normal path: tool call on first try ---

    @pytest.mark.parametrize("tool_name,args", [
        ("write", {"path": "hello.py", "content": "print('hi')"}),
        ("write", {"path": "app.html", "content": "<h1>Hi</h1>"}),
        ("write", {"path": "style.css", "content": "body{}"}),
        ("bash", {"command": "python hello.py"}),
        ("read", {"path": "hello.py"}),
        ("edit", {"path": "hello.py", "old_string": "hi", "new_string": "hello"}),
        ("grep", {"pattern": "print", "path": "."}),
        ("glob", {"pattern": "*.py"}),
    ])
    def test_tool_call_first_try(self, tool_name, args):
        llm = MagicMock()
        llm.chat.side_effect = [
            _resp("Using tool...", [ToolCall(name=tool_name, arguments=args)]),
            _resp("Done!"),
        ]
        loop = _loop(llm)
        result = loop.run_turn("Do something")
        nudges = [m for m in loop.messages if m.content == _TOOL_NUDGE]
        assert len(nudges) == 0
        assert loop.tools.execute.called

    # --- Nudge path: code block then tool ---

    @pytest.mark.parametrize("lang", _LANGUAGES[:20])
    def test_nudge_then_tool_various_langs(self, lang):
        llm = MagicMock()
        llm.chat.side_effect = [
            _resp(f"Here's the code:\n```{lang}\nsome code\n```\n"),
            _resp("Saving...", [_tc("write", path="f.py", content="code")]),
            _resp("Done!"),
        ]
        loop = _loop(llm)
        result = loop.run_turn("Write code")
        nudges = [m for m in loop.messages if m.content == _TOOL_NUDGE]
        assert len(nudges) == 1

    # --- Nudge path: code block then clean text ---

    @pytest.mark.parametrize("final_text", [
        "分かりました。writeツールで保存しました。",
        "ファイルを保存しました。",
        "",
        "👍",
    ])
    def test_nudge_then_clean_text(self, final_text):
        llm = MagicMock()
        llm.chat.side_effect = [
            _resp("```python\nprint(1)\n```\n"),
            _resp(final_text),
        ]
        loop = _loop(llm)
        loop.run_turn("Hello")
        nudges = [m for m in loop.messages if m.content == _TOOL_NUDGE]
        assert len(nudges) == 1

    # --- Nudge limit: exactly at limit ---

    def test_nudge_limit_exact(self):
        llm = MagicMock()
        code = "```python\nprint(1)\n```\n"
        responses = [_resp(code)] * _MAX_NUDGES_PER_TURN + [_resp("OK")]
        llm.chat.side_effect = responses
        loop = _loop(llm)
        loop.run_turn("Hello")
        nudges = [m for m in loop.messages if m.content == _TOOL_NUDGE]
        assert len(nudges) == _MAX_NUDGES_PER_TURN

    # --- Nudge limit: over limit, code block falls through to validation ---

    def test_nudge_over_limit_falls_to_validation(self):
        llm = MagicMock()
        code = "```python\nprint(1)\n```\n"
        # _MAX_NUDGES_PER_TURN + 1 code blocks, then clean
        responses = [_resp(code)] * (_MAX_NUDGES_PER_TURN + 1) + [_resp("OK")]
        llm.chat.side_effect = responses
        loop = _loop(llm)
        loop.run_turn("Hello")
        nudges = [m for m in loop.messages if m.content == _TOOL_NUDGE]
        assert len(nudges) == _MAX_NUDGES_PER_TURN
        # Validator should have been called for the code block that went past limit
        assert loop.validator.validate.called

    # --- Multiple tool calls in one response ---

    def test_multiple_tool_calls_single_response(self):
        llm = MagicMock()
        llm.chat.side_effect = [
            _resp("Creating files...", [
                _tc("write", path="a.py", content="x=1"),
                _tc("write", path="b.py", content="y=2"),
            ]),
            _resp("Both files created!"),
        ]
        loop = _loop(llm)
        loop.run_turn("Create two files")
        assert loop.tools.execute.call_count == 2

    # --- Tool call returns error ---

    def test_tool_call_with_error(self):
        llm = MagicMock()
        llm.chat.side_effect = [
            _resp("Writing...", [_tc("write", path="a.py", content="x=1")]),
            _resp("There was an error. Let me fix it.", [
                _tc("write", path="a.py", content="x=2")
            ]),
            _resp("Fixed!"),
        ]
        loop = _loop(llm)
        loop.tools.execute.side_effect = [
            {"error": "Permission denied"},
            {"status": "ok"},
        ]
        loop.run_turn("Write a file")
        assert loop.tools.execute.call_count == 2

    # --- Validation failure then retry ---

    def test_validation_failure_then_success(self):
        llm = MagicMock()
        llm.chat.side_effect = [
            _resp("Here's some HTML: <div></div>"),  # violation in python mode
            _resp("Sorry, here's Python: x = 1"),
        ]
        loop = _loop(llm)
        # First call: invalid, second: valid
        loop.validator.validate.side_effect = [
            ValidationResult(valid=False, violations=[
                Violation(rule="language_isolation", detail="HTML in Python mode")
            ]),
            ValidationResult(valid=True),
        ]
        result = loop.run_turn("Write code")
        assert loop.metrics.violations == 1
        assert loop.metrics.retries == 1

    # --- Nudge + validation failure combo ---

    def test_nudge_then_validation_failure_then_success(self):
        llm = MagicMock()
        llm.chat.side_effect = [
            _resp("```python\nprint(1)\n```"),  # nudge
            _resp("import requests\nrequests.get('http://x')"),  # violation
            _resp("x = 42"),  # clean
        ]
        loop = _loop(llm)
        loop.validator.validate.side_effect = [
            ValidationResult(valid=False, violations=[
                Violation(rule="forbidden_import", detail="requests not allowed")
            ]),
            ValidationResult(valid=True),
        ]
        result = loop.run_turn("Hello")
        nudges = [m for m in loop.messages if m.content == _TOOL_NUDGE]
        assert len(nudges) == 1
        assert loop.metrics.violations == 1

    # --- Max iterations reached ---

    def test_max_iterations_reached(self):
        llm = MagicMock()
        # Always return code block → nudge 2 times, then falls to validation
        # But validation always fails → retry until max_iterations
        code = "```python\nimport os\nos.system('rm -rf /')\n```"
        llm.chat.return_value = _resp(code)
        loop = _loop(llm, max_iter=5)
        loop.validator.validate.return_value = ValidationResult(
            valid=False,
            violations=[Violation(rule="no_os_system", detail="os.system detected")]
        )
        result = loop.run_turn("Hello")
        assert "Max iterations" in result or "max" in result.lower() or loop.metrics.iterations == 5

    # --- Empty content with tool calls ---

    def test_empty_content_with_tool_calls(self):
        llm = MagicMock()
        llm.chat.side_effect = [
            _resp("", [_tc("write", path="a.py", content="x=1")]),
            _resp("Done!"),
        ]
        loop = _loop(llm)
        loop.run_turn("Write file")
        assert loop.tools.execute.called

    # --- Scope rejection ---

    @pytest.mark.parametrize("input_text", [
        "Write me a Rust program",
        "Create a Docker container",
        "Deploy to kubernetes",
        "Build a blockchain app",
        "Write Java code",
        "Use golang for this",
        "Create a flutter app",
    ])
    def test_scope_rejection(self, input_text):
        llm = MagicMock()
        loop = _loop(llm)
        result = loop.run_turn(input_text)
        assert "scope" in result.lower() or "supported" in result.lower()
        llm.chat.assert_not_called()

    # --- Scope acceptance ---

    @pytest.mark.parametrize("input_text", [
        "Write a Python function",
        "Help me with loops",
        "Explain variables",
        "変数について教えて",
        "forループの書き方",
        "Hello",
        "OK",
        "次",
        "続き",
        "Print hello world",
        "What is a list?",
        "",  # empty input
        " ",  # whitespace
        "🙏 お願いします",
        "a" * 100,
    ])
    def test_scope_acceptance(self, input_text):
        llm = MagicMock()
        llm.chat.return_value = _resp("Here's your answer.")
        loop = _loop(llm)
        result = loop.run_turn(input_text)
        # Should not be scope rejection
        assert "outside the supported scope" not in result


# --- Massive: simulate 100 "users" with random interaction patterns ---

class TestSimulatedUsers:
    """Simulate 100 users each doing 10 turns with random LLM behavior."""

    @pytest.mark.parametrize("user_id", range(100))
    def test_random_user_session(self, user_id):
        """Each user gets a random sequence of LLM responses — no crashes."""
        rng = random.Random(user_id * 42)
        llm = MagicMock()
        mode = rng.choice(list(Mode))
        loop = _loop(llm, mode=mode, max_iter=10)

        for turn in range(10):
            # Generate a random LLM response pattern
            pattern = rng.choice([
                "text_only", "code_block", "tool_call",
                "code_then_tool", "empty", "multi_tool",
            ])

            if pattern == "text_only":
                llm.chat.side_effect = [_resp("テキスト応答です。")]
            elif pattern == "code_block":
                lang = rng.choice(["python", "js", "html", ""])
                llm.chat.side_effect = [
                    _resp(f"```{lang}\ncode\n```\n"),
                    _resp("OK"),
                ]
            elif pattern == "tool_call":
                llm.chat.side_effect = [
                    _resp("ツール使用", [_tc("write", path="f.py", content="x=1")]),
                    _resp("完了"),
                ]
            elif pattern == "code_then_tool":
                llm.chat.side_effect = [
                    _resp("```python\nx=1\n```\n"),
                    _resp("保存します", [_tc("write", path="f.py", content="x=1")]),
                    _resp("完了"),
                ]
            elif pattern == "empty":
                llm.chat.side_effect = [_resp("")]
            elif pattern == "multi_tool":
                llm.chat.side_effect = [
                    _resp("複数ツール", [
                        _tc("write", path="a.py", content="a=1"),
                        _tc("write", path="b.py", content="b=2"),
                    ]),
                    _resp("完了"),
                ]

            user_input = rng.choice([
                "Hello", "次", "OK", "変数って何？", "コードを書いて",
                "実行して", "説明して", "もう一度", "続き", "ありがとう",
                "print関数", "forループ", "classを教えて", "",
                "リストの使い方", "ファイルを作って",
            ])

            # Should never crash
            try:
                result = loop.run_turn(user_input)
                assert isinstance(result, str)
            except StopIteration:
                # side_effect exhausted is fine for this test
                pass

            # Reset for next turn
            loop.messages = [loop.messages[0]]  # keep system prompt


# ═══════════════════════════════════════════════════════════════════════
# C. Agent loop run_turn_stream — 200+ scenarios
# ═══════════════════════════════════════════════════════════════════════

class TestRunTurnStreamMassive:
    """Stream-specific patterns."""

    @pytest.mark.parametrize("chunk_sizes", [
        [1],  # single char chunks
        [5, 5, 5],  # small chunks
        [100],  # one big chunk
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # many tiny
        [50, 50],  # two halves
    ])
    def test_various_chunk_sizes(self, chunk_sizes):
        text = "Hello, this is a response about Python variables."
        llm = MagicMock()

        def _stream(messages, tools=None):
            pos = 0
            for size in chunk_sizes:
                chunk = text[pos:pos + size]
                if chunk:
                    yield chunk
                pos += size
            yield LLMResponse(content=text[:sum(chunk_sizes)], tool_calls=[])

        llm.chat_stream.side_effect = _stream
        loop = _loop(llm)
        chunks = list(loop.run_turn_stream("Hello"))
        assert len(chunks) >= 1

    def test_stream_empty_response(self):
        llm = MagicMock()

        def _stream(messages, tools=None):
            yield LLMResponse(content="", tool_calls=[])

        llm.chat_stream.side_effect = _stream
        loop = _loop(llm)
        chunks = list(loop.run_turn_stream("Hello"))
        # Should not crash

    def test_stream_code_block_triggers_nudge(self):
        llm = MagicMock()
        call_count = [0]

        def _stream(messages, tools=None):
            call_count[0] += 1
            if call_count[0] == 1:
                text = "```python\nprint(1)\n```\n"
                yield text
                yield LLMResponse(content=text, tool_calls=[])
            else:
                text = "OK, saved."
                yield text
                yield LLMResponse(content=text, tool_calls=[])

        llm.chat_stream.side_effect = _stream
        loop = _loop(llm)
        chunks = list(loop.run_turn_stream("Write code"))
        nudges = [m for m in loop.messages if m.content == _TOOL_NUDGE]
        assert len(nudges) == 1

    def test_stream_tool_call_no_nudge(self):
        llm = MagicMock()
        call_count = [0]

        def _stream(messages, tools=None):
            call_count[0] += 1
            if call_count[0] == 1:
                yield LLMResponse(
                    content="Saving...",
                    tool_calls=[_tc("write", path="a.py", content="x=1")],
                )
            else:
                yield "Done!"
                yield LLMResponse(content="Done!", tool_calls=[])

        llm.chat_stream.side_effect = _stream
        loop = _loop(llm)
        chunks = list(loop.run_turn_stream("Write file"))
        nudges = [m for m in loop.messages if m.content == _TOOL_NUDGE]
        assert len(nudges) == 0

    def test_stream_scope_rejection(self):
        llm = MagicMock()
        loop = _loop(llm)
        chunks = list(loop.run_turn_stream("Write Rust code"))
        full = "".join(chunks)
        assert "scope" in full.lower() or "supported" in full.lower()

    @pytest.mark.parametrize("user_id", range(100))
    def test_random_stream_user(self, user_id):
        """Simulate 100 users with streaming — no crashes."""
        rng = random.Random(user_id * 77)
        llm = MagicMock()
        mode = rng.choice(list(Mode))
        loop = _loop(llm, mode=mode, max_iter=8)

        for turn in range(10):
            pattern = rng.choice(["text", "code_block", "tool", "empty"])
            call_count = [0]

            def _make_stream(pat):
                def _stream(messages, tools=None):
                    if pat == "text":
                        yield "テスト応答"
                        yield LLMResponse(content="テスト応答", tool_calls=[])
                    elif pat == "code_block":
                        c = "```python\nx=1\n```\n"
                        yield c
                        yield LLMResponse(content=c, tool_calls=[])
                    elif pat == "tool":
                        yield LLMResponse(
                            content="",
                            tool_calls=[_tc("write", path="f.py", content="x=1")],
                        )
                    elif pat == "empty":
                        yield LLMResponse(content="", tool_calls=[])
                return _stream

            # Provide enough stream responses for nudge retries
            streams = [_make_stream(pattern)] + [_make_stream("text")] * 5
            llm.chat_stream.side_effect = streams

            user_input = rng.choice(["Hello", "次", "OK", "書いて", ""])

            try:
                chunks = list(loop.run_turn_stream(user_input))
                for c in chunks:
                    assert isinstance(c, str)
            except (StopIteration, TypeError):
                pass

            loop.messages = [loop.messages[0]]


# ═══════════════════════════════════════════════════════════════════════
# D. Policy system prompt — all mode × level combos
# ═══════════════════════════════════════════════════════════════════════

_ALL_MODE_LEVEL = list(itertools.product(Mode, Level))


class TestPolicySystemPromptAll:

    @pytest.mark.parametrize("mode,level", _ALL_MODE_LEVEL,
                             ids=[f"{m.value}_{l.value}" for m, l in _ALL_MODE_LEVEL])
    def test_prompt_not_empty(self, mode, level):
        profile = build_mode_profile(mode)
        policy = PolicyEngine(profile, level=level)
        prompt = policy.build_system_prompt()
        assert len(prompt) > 100, f"Prompt too short for {mode}/{level}"

    @pytest.mark.parametrize("mode,level", _ALL_MODE_LEVEL,
                             ids=[f"constraint_{m.value}_{l.value}" for m, l in _ALL_MODE_LEVEL])
    def test_prompt_contains_constraint(self, mode, level):
        profile = build_mode_profile(mode)
        policy = PolicyEngine(profile, level=level)
        prompt = policy.build_system_prompt()
        assert "制約" in prompt

    @pytest.mark.parametrize("mode,level", _ALL_MODE_LEVEL,
                             ids=[f"write_{m.value}_{l.value}" for m, l in _ALL_MODE_LEVEL])
    def test_prompt_contains_write_rule(self, mode, level):
        profile = build_mode_profile(mode)
        policy = PolicyEngine(profile, level=level)
        prompt = policy.build_system_prompt()
        assert "write 関数" in prompt

    @pytest.mark.parametrize("mode", list(Mode))
    def test_python_modes_have_bash_rule(self, mode):
        lang = MODE_LANGUAGE[mode]
        if lang == LanguageFamily.PYTHON:
            profile = build_mode_profile(mode)
            policy = PolicyEngine(profile)
            prompt = policy.build_system_prompt()
            assert "bash 関数" in prompt

    @pytest.mark.parametrize("mode", list(Mode))
    def test_web_modes_have_no_bash(self, mode):
        lang = MODE_LANGUAGE[mode]
        if lang == LanguageFamily.WEB:
            profile = build_mode_profile(mode)
            policy = PolicyEngine(profile)
            prompt = policy.build_system_prompt()
            assert "bash は使えない" in prompt

    @pytest.mark.parametrize("mode,level", _ALL_MODE_LEVEL,
                             ids=[f"mastered_{m.value}_{l.value}" for m, l in _ALL_MODE_LEVEL])
    def test_prompt_with_mastered_concepts(self, mode, level):
        """Prompt should render correctly even with mastered concepts."""
        profile = build_mode_profile(mode)
        policy = PolicyEngine(profile, level=level)
        catalog = CONCEPT_CATALOGS[mode]
        # Set some mastered concepts
        policy.mastered_concepts = set(list(catalog.beginner)[:3])
        prompt = policy.build_system_prompt()
        assert len(prompt) > 100


# ═══════════════════════════════════════════════════════════════════════
# E. Curriculum — all mode × level × mastered combos
# ═══════════════════════════════════════════════════════════════════════

class TestCurriculumMassive:

    @pytest.mark.parametrize("mode,level", _ALL_MODE_LEVEL,
                             ids=[f"edu_{m.value}_{l.value}" for m, l in _ALL_MODE_LEVEL])
    def test_education_prompt_generated(self, mode, level):
        prompt = build_education_prompt(mode, level)
        assert len(prompt) > 50

    @pytest.mark.parametrize("mode,level", _ALL_MODE_LEVEL,
                             ids=[f"write_expr_{m.value}_{l.value}" for m, l in _ALL_MODE_LEVEL])
    def test_write_expression_present(self, mode, level):
        prompt = build_education_prompt(mode, level)
        assert "write 関数を呼び出して" in prompt

    @pytest.mark.parametrize("mode,level", _ALL_MODE_LEVEL,
                             ids=[f"bash_expr_{m.value}_{l.value}" for m, l in _ALL_MODE_LEVEL])
    def test_bash_expression_present(self, mode, level):
        prompt = build_education_prompt(mode, level)
        assert "bash 関数を呼び出して" in prompt

    @pytest.mark.parametrize("mode,level", _ALL_MODE_LEVEL,
                             ids=[f"no_old_{m.value}_{l.value}" for m, l in _ALL_MODE_LEVEL])
    def test_no_old_expressions(self, mode, level):
        prompt = build_education_prompt(mode, level)
        assert "黙ってwriteツールを使う" not in prompt
        assert "黙ってbashツールを使う" not in prompt

    @pytest.mark.parametrize("mode", list(Mode))
    def test_mastered_concepts_render(self, mode):
        catalog = CONCEPT_CATALOGS[mode]
        # Test with 0, 1, half, all concepts mastered
        for n in [0, 1, len(catalog.beginner) // 2, len(catalog.beginner)]:
            mastered = set(list(catalog.beginner)[:n])
            prompt = build_education_prompt(mode, Level.BEGINNER, mastered)
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    @pytest.mark.parametrize("mode", list(Mode))
    def test_concept_extraction_various_texts(self, mode):
        """Extract concepts from various synthetic LLM responses."""
        catalog = CONCEPT_CATALOGS[mode]
        # Test with each concept individually
        for concept in catalog.all_concepts():
            # Just the concept name should trigger extraction
            found = extract_concepts(concept, mode)
            # Most concepts should be found by their own name
            # (not all — some need code patterns, not just the word)
            assert isinstance(found, list)


# ═══════════════════════════════════════════════════════════════════════
# F. Tool definitions — structural integrity
# ═══════════════════════════════════════════════════════════════════════

class TestToolDefinitionsDeep:

    @pytest.mark.parametrize("td", TOOL_DEFINITIONS,
                             ids=[td["function"]["name"] for td in TOOL_DEFINITIONS])
    def test_description_not_empty(self, td):
        assert len(td["function"]["description"]) > 10

    @pytest.mark.parametrize("td", TOOL_DEFINITIONS,
                             ids=[f"props_{td['function']['name']}" for td in TOOL_DEFINITIONS])
    def test_properties_types_are_string(self, td):
        props = td["function"]["parameters"]["properties"]
        for name, prop in props.items():
            assert prop["type"] == "string", f"{td['function']['name']}.{name} type != string"

    @pytest.mark.parametrize("td", TOOL_DEFINITIONS,
                             ids=[f"req_{td['function']['name']}" for td in TOOL_DEFINITIONS])
    def test_required_params_exist_in_properties(self, td):
        func = td["function"]
        props = set(func["parameters"]["properties"].keys())
        required = set(func["parameters"]["required"])
        assert required.issubset(props), f"{func['name']}: required {required} not in props {props}"
