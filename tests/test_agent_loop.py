"""Tests for agent loop — nudge feature and core orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from novicode.agent_loop import AgentLoop, _has_code_block, _TOOL_NUDGE, _MAX_NUDGES_PER_TURN
from novicode.config import Mode, build_mode_profile
from novicode.curriculum import Level
from novicode.llm_adapter import LLMResponse, Message, ToolCall, TOOL_DEFINITIONS
from novicode.metrics import Metrics
from novicode.policy_engine import PolicyEngine
from novicode.session_manager import Session, SessionMeta
from novicode.validator import ValidationResult


# ── Helpers ──────────────────────────────────────────────────────────

def _make_session() -> Session:
    meta = SessionMeta(
        session_id="test123", model="qwen3:8b", mode="python_basic",
        created_at=0.0, research=False,
    )
    return Session(meta=meta)


def _make_loop(
    llm: MagicMock,
    mode: Mode = Mode.PYTHON_BASIC,
    max_iterations: int = 10,
) -> AgentLoop:
    profile = build_mode_profile(mode)
    policy = PolicyEngine(profile, level=Level.BEGINNER)

    tools = MagicMock()
    tools.available_tools.return_value = list(profile.allowed_tools)

    validator = MagicMock()
    validator.validate.return_value = ValidationResult(valid=True)

    return AgentLoop(
        llm=llm,
        profile=profile,
        tools=tools,
        validator=validator,
        policy=policy,
        session=_make_session(),
        metrics=Metrics(),
        max_iterations=max_iterations,
    )


# ── _has_code_block tests ────────────────────────────────────────────

class TestHasCodeBlock:
    def test_python_fenced_block(self):
        text = "Here is code:\n```python\nprint('hello')\n```"
        assert _has_code_block(text) is True

    def test_js_fenced_block(self):
        text = "```js\nconsole.log('hi')\n```"
        assert _has_code_block(text) is True

    def test_bare_fenced_block(self):
        text = "```\nsome code\n```"
        assert _has_code_block(text) is True

    def test_plain_text_no_block(self):
        text = "This is just a text explanation without any code."
        assert _has_code_block(text) is False

    def test_inline_code_not_detected(self):
        text = "Use `print()` to output text."
        assert _has_code_block(text) is False

    def test_incomplete_fence_not_detected(self):
        text = "```python is a language"
        assert _has_code_block(text) is False


# ── Nudge injection tests (run_turn) ────────────────────────────────

class TestNudgeRunTurn:
    def test_nudge_injected_when_code_block_without_tool(self):
        """Code block + no tool calls → nudge message added to messages."""
        llm = MagicMock()
        # First response: code block but no tools → triggers nudge
        # Second response: plain text, no code block → accepted
        llm.chat.side_effect = [
            LLMResponse(content="Here:\n```python\nprint('hi')\n```\n", tool_calls=[]),
            LLMResponse(content="OK, using write tool now.", tool_calls=[
                ToolCall(name="write", arguments={"path": "test.py", "content": "print('hi')"})
            ]),
            LLMResponse(content="Done!", tool_calls=[]),
        ]

        loop = _make_loop(llm)
        loop.tools.execute.return_value = {"status": "ok"}
        result = loop.run_turn("Write hello world")

        # The nudge message should have been injected into messages
        nudge_msgs = [m for m in loop.messages if m.content == _TOOL_NUDGE]
        assert len(nudge_msgs) >= 1, "Nudge message should be in conversation"

    def test_nudge_limit_respected(self):
        """After _MAX_NUDGES_PER_TURN nudges, code block text is passed to validation."""
        llm = MagicMock()
        code_response = LLMResponse(
            content="```python\nprint('hi')\n```\n", tool_calls=[]
        )
        final_response = LLMResponse(content="No code here.", tool_calls=[])

        # Return code blocks more times than the nudge limit, then a clean response
        responses = [code_response] * (_MAX_NUDGES_PER_TURN + 1) + [final_response]
        llm.chat.side_effect = responses

        loop = _make_loop(llm)
        result = loop.run_turn("Hello")

        # Count nudge messages
        nudge_msgs = [m for m in loop.messages if m.content == _TOOL_NUDGE]
        assert len(nudge_msgs) == _MAX_NUDGES_PER_TURN, (
            f"Expected exactly {_MAX_NUDGES_PER_TURN} nudges, got {len(nudge_msgs)}"
        )

    def test_no_nudge_when_tool_calls_present(self):
        """Tool calls in response → no nudge, tools executed normally."""
        llm = MagicMock()
        llm.chat.side_effect = [
            LLMResponse(
                content="Writing file...",
                tool_calls=[ToolCall(name="write", arguments={"path": "a.py", "content": "x=1"})],
            ),
            LLMResponse(content="Done!", tool_calls=[]),
        ]

        loop = _make_loop(llm)
        loop.tools.execute.return_value = {"status": "ok"}
        result = loop.run_turn("Create a variable")

        nudge_msgs = [m for m in loop.messages if m.content == _TOOL_NUDGE]
        assert len(nudge_msgs) == 0, "No nudge when tools are used"
        loop.tools.execute.assert_called_once()

    def test_no_nudge_for_text_only_response(self):
        """Plain text without code block → no nudge, goes to validation."""
        llm = MagicMock()
        llm.chat.return_value = LLMResponse(
            content="Let me explain variables. A variable stores a value.",
            tool_calls=[],
        )

        loop = _make_loop(llm)
        result = loop.run_turn("What is a variable?")

        nudge_msgs = [m for m in loop.messages if m.content == _TOOL_NUDGE]
        assert len(nudge_msgs) == 0
        assert "variable" in result.lower() or "変数" in result or len(result) > 0


# ── Nudge injection tests (run_turn_stream) ─────────────────────────

class TestNudgeRunTurnStream:
    def test_nudge_injected_in_stream(self):
        """Streaming: code block + no tools → nudge injected."""
        llm = MagicMock()

        code_content = "```python\nprint('hi')\n```\n"
        final_content = "Done, file saved."

        def _stream_side_effect(messages, tools=None):
            # Determine which response to return based on conversation length
            nudge_count = sum(1 for m in messages if m.content == _TOOL_NUDGE)
            if nudge_count == 0:
                yield code_content
                yield LLMResponse(content=code_content, tool_calls=[])
            else:
                yield final_content
                yield LLMResponse(content=final_content, tool_calls=[])

        llm.chat_stream.side_effect = _stream_side_effect

        loop = _make_loop(llm)
        chunks = list(loop.run_turn_stream("Hello"))

        nudge_msgs = [m for m in loop.messages if m.content == _TOOL_NUDGE]
        assert len(nudge_msgs) >= 1

    def test_no_nudge_in_stream_with_tools(self):
        """Streaming: tool calls present → no nudge."""
        llm = MagicMock()

        def _stream(messages, tools=None):
            call_count = len([m for m in messages if m.role == "user" and "Tool results" in m.content])
            if call_count == 0:
                yield LLMResponse(
                    content="Saving...",
                    tool_calls=[ToolCall(name="write", arguments={"path": "a.py", "content": "x=1"})],
                )
            else:
                yield "All done."
                yield LLMResponse(content="All done.", tool_calls=[])

        llm.chat_stream.side_effect = _stream

        loop = _make_loop(llm)
        loop.tools.execute.return_value = {"status": "ok"}
        chunks = list(loop.run_turn_stream("Create a file"))

        nudge_msgs = [m for m in loop.messages if m.content == _TOOL_NUDGE]
        assert len(nudge_msgs) == 0


# ── Constants sanity ─────────────────────────────────────────────────

class TestNudgeConstants:
    def test_max_nudges_is_positive(self):
        assert _MAX_NUDGES_PER_TURN > 0

    def test_nudge_message_mentions_write(self):
        assert "write" in _TOOL_NUDGE

    def test_nudge_message_is_japanese(self):
        assert any(ord(c) > 0x3000 for c in _TOOL_NUDGE), "Nudge should contain Japanese text"
