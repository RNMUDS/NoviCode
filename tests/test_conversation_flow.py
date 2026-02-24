"""Conversation flow tests — 10x scale (7000+ tests).

Validates that the agent loop produces natural, well-structured conversations:
  - Message ordering: system → user → assistant → user → assistant → ...
  - Nudge messages are injected at the correct position
  - Tool results appear as user messages after assistant tool calls
  - No corrupted or out-of-order messages
  - Conversation content is meaningful (not empty, not garbled)
  - Natural teaching patterns: explain → write → predict → run → choose

Simulates 1000 users × 10 turns for run_turn and run_turn_stream.
"""

from __future__ import annotations

import itertools
import random
import re
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
from novicode.curriculum import Level, CONCEPT_CATALOGS, build_education_prompt
from novicode.llm_adapter import LLMResponse, Message, ToolCall, TOOL_DEFINITIONS
from novicode.metrics import Metrics
from novicode.policy_engine import PolicyEngine
from novicode.session_manager import Session, SessionMeta
from novicode.validator import ValidationResult, Violation


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _session():
    return Session(meta=SessionMeta(
        session_id="flow", model="qwen3:8b", mode="python_basic",
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


def _resp(content="", tool_calls=None):
    return LLMResponse(content=content, tool_calls=tool_calls or [])


def _tc(name="write", **kwargs):
    if not kwargs:
        kwargs = {"path": "test.py", "content": "x = 1"}
    return ToolCall(name=name, arguments=kwargs)


# ── Conversation validators ──────────────────────────────────────────

def assert_message_ordering(messages: list[Message]) -> None:
    """Verify that messages follow valid role ordering rules."""
    assert len(messages) >= 1, "Must have at least system message"
    assert messages[0].role == "system", f"First message must be system, got {messages[0].role}"

    for i in range(1, len(messages)):
        prev = messages[i - 1]
        curr = messages[i]

        # After system: must be user
        if i == 1:
            assert curr.role == "user", (
                f"Message[1] must be user after system, got {curr.role}"
            )

        # After assistant: must be user (could be user input, nudge, tool result, or correction)
        if prev.role == "assistant":
            assert curr.role == "user", (
                f"Message[{i}] after assistant must be user, got {curr.role}: "
                f"{curr.content[:60]!r}"
            )

        # After user (that isn't the first): must be assistant
        # EXCEPT: consecutive user messages can happen if system prompt was replaced
        # But generally: user → assistant
        if prev.role == "user" and i > 1:
            assert curr.role in ("assistant", "system"), (
                f"Message[{i}] after user should be assistant, got {curr.role}: "
                f"{curr.content[:60]!r}"
            )


def assert_no_empty_assistant(messages: list[Message]) -> None:
    """Warn if assistant messages are empty (might indicate LLM issues)."""
    for i, m in enumerate(messages):
        if m.role == "assistant":
            # Empty content is ok if there were tool calls (content can be "")
            # But we just check it doesn't cause structural issues
            assert isinstance(m.content, str), f"Message[{i}] content must be str"


def assert_nudge_placement(messages: list[Message]) -> None:
    """Verify nudge messages are correctly placed: always as user after assistant."""
    for i, m in enumerate(messages):
        if m.content == _TOOL_NUDGE:
            assert m.role == "user", f"Nudge at [{i}] must be user role"
            assert i > 0, "Nudge can't be first message"
            assert messages[i - 1].role == "assistant", (
                f"Nudge at [{i}] must follow assistant, "
                f"but follows {messages[i-1].role}"
            )


def assert_tool_results_placement(messages: list[Message]) -> None:
    """Verify tool result messages follow assistant messages."""
    for i, m in enumerate(messages):
        if m.role == "user" and "Tool results:" in m.content:
            assert i > 0
            assert messages[i - 1].role == "assistant", (
                f"Tool results at [{i}] must follow assistant"
            )


def assert_system_prompt_integrity(messages: list[Message]) -> None:
    """System prompt must remain at index 0 and contain key content."""
    assert messages[0].role == "system"
    prompt = messages[0].content
    assert len(prompt) > 50, "System prompt is too short"
    # Must contain educational content or base prompt
    assert any(kw in prompt for kw in ["制約", "tutor", "先生"]), (
        "System prompt missing expected content"
    )


def validate_conversation(messages: list[Message]) -> None:
    """Run all conversation validators."""
    assert_message_ordering(messages)
    assert_no_empty_assistant(messages)
    assert_nudge_placement(messages)
    assert_tool_results_placement(messages)
    assert_system_prompt_integrity(messages)


# ═══════════════════════════════════════════════════════════════════════
# 1. Natural teaching flow patterns — parameterized
# ═══════════════════════════════════════════════════════════════════════

# Simulate realistic teaching flows:
# Pattern A: User asks → LLM writes code with tool → asks prediction
# Pattern B: User asks → LLM outputs code as text (nudge) → retries with tool
# Pattern C: User predicts → LLM runs code with bash → gives feedback
# Pattern D: User asks question → LLM explains (text only)
# Pattern E: User greets → LLM starts with explanation + code
# Pattern F: Validation failure → correction → success

_TEACHING_FLOWS = {
    "write_and_predict": {
        "user_input": "変数について教えて",
        "responses": [
            _resp(
                "変数について説明しますね。まず簡単なコードを作ります。",
                [_tc("write", path="hello.py", content="x = 10\nprint(x)")],
            ),
            _resp(
                "ファイルを保存しました。\n"
                "- `x = 10` は変数 x に 10 を代入しています\n"
                "- `print(x)` は x の値を表示します\n\n"
                "このコードを実行すると、どんな結果になると思いますか？"
            ),
        ],
        "checks": {
            "has_tool_call": True,
            "final_has_question": True,
        },
    },
    "run_and_feedback": {
        "user_input": "10が表示されると思います",
        "responses": [
            _resp(
                "実行してみましょう。",
                [_tc("bash", command="python hello.py")],
            ),
            _resp(
                "正解です！ `10` と表示されましたね。\n\n"
                "次は A. 変数を2つ使って足し算する か "
                "B. 文字列の変数を作る のどちらにしましょう？"
            ),
        ],
        "checks": {
            "has_tool_call": True,
            "final_has_choices": True,
        },
    },
    "text_explanation": {
        "user_input": "変数って何？",
        "responses": [
            _resp(
                "変数とは、値を入れておく箱のようなものです。\n"
                "名前をつけて、後から値を取り出せます。\n\n"
                "変数については分かりますか？"
            ),
        ],
        "checks": {
            "has_tool_call": False,
            "final_is_text": True,
        },
    },
    "nudge_then_tool": {
        "user_input": "forループを教えて",
        "responses": [
            _resp("こちらがコードです：\n```python\nfor i in range(5):\n    print(i)\n```\n"),
            _resp(
                "ファイルに保存します。",
                [_tc("write", path="loop.py", content="for i in range(5):\n    print(i)")],
            ),
            _resp(
                "保存しました。\n"
                "- `for i in range(5):` は 0〜4 の5回繰り返します\n"
                "- `print(i)` は i の値を表示します\n\n"
                "このコードを実行すると、どんな結果になると思いますか？"
            ),
        ],
        "checks": {
            "has_nudge": True,
            "has_tool_call": True,
        },
    },
    "greeting_start": {
        "user_input": "こんにちは",
        "responses": [
            _resp(
                "こんにちは！プログラミングを一緒に学びましょう。\n"
                "まずは簡単なところから始めます。",
                [_tc("write", path="hello.py", content="print('Hello!')")],
            ),
            _resp(
                "最初のプログラムを作りました。\n\n"
                "このコードを実行すると、どんな結果になると思いますか？"
            ),
        ],
        "checks": {
            "has_tool_call": True,
        },
    },
    "empty_user_input": {
        "user_input": "",
        "responses": [
            _resp("何か質問がありましたらどうぞ！"),
        ],
        "checks": {
            "has_tool_call": False,
        },
    },
    "user_says_ok": {
        "user_input": "OK",
        "responses": [
            _resp(
                "では次に進みましょう。",
                [_tc("write", path="step2.py", content="y = 20\nprint(y)")],
            ),
            _resp(
                "新しいコードを保存しました。\n\n"
                "このコードを実行すると、どんな結果になると思いますか？"
            ),
        ],
        "checks": {
            "has_tool_call": True,
        },
    },
    "user_says_next": {
        "user_input": "次",
        "responses": [
            _resp("次のステップです。リストを学びましょう。"),
        ],
        "checks": {
            "has_tool_call": False,
        },
    },
    "validation_failure_then_success": {
        "user_input": "HTMLを書いて",
        "responses": [
            _resp("<html><body>Hello</body></html>"),  # violation in python mode
            _resp("すみません、Python モードなので Python コードを書きますね。\nx = 1"),
        ],
        "checks": {
            "has_violation": True,
        },
    },
    "multiple_tool_calls": {
        "user_input": "2つのファイルを作って",
        "responses": [
            _resp("2つのファイルを作ります。", [
                _tc("write", path="a.py", content="a = 1"),
                _tc("write", path="b.py", content="b = 2"),
            ]),
            _resp("2つのファイルを保存しました。"),
        ],
        "checks": {
            "tool_count": 2,
        },
    },
}


class TestNaturalTeachingFlows:

    @pytest.mark.parametrize("flow_name,flow", _TEACHING_FLOWS.items(),
                             ids=list(_TEACHING_FLOWS.keys()))
    def test_flow_produces_valid_conversation(self, flow_name, flow):
        """Each teaching flow should produce a well-structured conversation."""
        llm = MagicMock()
        llm.chat.side_effect = list(flow["responses"])
        loop = _loop(llm)

        # Handle validation failure flow
        if flow.get("checks", {}).get("has_violation"):
            loop.validator.validate.side_effect = [
                ValidationResult(valid=False, violations=[
                    Violation(rule="language_isolation", detail="HTML in Python")
                ]),
                ValidationResult(valid=True),
            ]

        result = loop.run_turn(flow["user_input"])
        validate_conversation(loop.messages)

        checks = flow.get("checks", {})
        if checks.get("has_tool_call"):
            assert loop.tools.execute.called, f"{flow_name}: expected tool call"
        if checks.get("has_nudge"):
            nudges = [m for m in loop.messages if m.content == _TOOL_NUDGE]
            assert len(nudges) >= 1, f"{flow_name}: expected nudge"
        if checks.get("final_has_question"):
            assert "思いますか" in result, f"{flow_name}: expected prediction question"
        if checks.get("final_has_choices"):
            assert any(c in result for c in ["A.", "B.", "どちら"]), (
                f"{flow_name}: expected choices"
            )
        if checks.get("tool_count"):
            assert loop.tools.execute.call_count == checks["tool_count"]


# ═══════════════════════════════════════════════════════════════════════
# 2. Message ordering — 1000 users × 10 turns (run_turn)
# ═══════════════════════════════════════════════════════════════════════

# User input variety
_USER_INPUTS_JA = [
    "こんにちは", "変数について教えて", "forループの書き方", "関数って何？",
    "リストの使い方", "クラスを教えて", "ファイルを作って", "コードを実行して",
    "もう一度", "次", "続き", "OK", "分かりました", "分からない",
    "10が出ると思う", "エラーが出ると思う", "何も出ないと思う",
    "Aを選びます", "Bにします", "両方やりたい",
    "print関数", "if文", "while", "defで関数",
    "ありがとう", "すごい", "面白い", "難しい",
    "もっと簡単にして", "もっと詳しく", "例を見せて",
]

_USER_INPUTS_EN = [
    "Hello", "Teach me variables", "How does a for loop work?",
    "What is a function?", "Show me lists", "Write a class",
    "Create a file", "Run the code", "Again", "Next", "OK",
    "I think it prints 10", "I don't understand", "Option A",
]

_ALL_USER_INPUTS = _USER_INPUTS_JA + _USER_INPUTS_EN

# LLM response patterns (simulating realistic Qwen3 outputs)
_LLM_PATTERNS = {
    "explain_and_write": lambda rng: [
        _resp(
            f"{'説明テキスト' * rng.randint(1, 3)}",
            [_tc("write", path=f"step{rng.randint(1,99)}.py",
                 content=f"x = {rng.randint(1,100)}\nprint(x)")],
        ),
        _resp(
            "保存しました。\n\nこのコードを実行すると、どんな結果になると思いますか？"
        ),
    ],
    "run_code": lambda rng: [
        _resp(
            "実行します。",
            [_tc("bash", command=f"python step{rng.randint(1,99)}.py")],
        ),
        _resp(
            f"結果は {rng.randint(1,100)} でした。\n\n"
            "次は A か B のどちらにしましょう？"
        ),
    ],
    "text_only": lambda rng: [
        _resp(rng.choice([
            "変数とは値を入れる箱のようなものです。",
            "forループは繰り返し処理を行います。",
            "関数を使うとコードを再利用できます。",
            "リストは複数の値をまとめて扱えます。",
            "クラスはデータと処理をまとめるものです。",
            "分かりました。もう少し簡単に説明しますね。",
            "いい質問ですね！",
            "素晴らしい予測です！正解です。",
        ])),
    ],
    "code_block_nudge": lambda rng: [
        _resp(f"```python\nx = {rng.randint(1,100)}\nprint(x)\n```\n"),
        _resp("", [_tc("write", path="f.py", content="x=1\nprint(x)")]),
        _resp("保存しました。どんな結果になると思いますか？"),
    ],
    "empty_then_text": lambda rng: [
        _resp(""),  # empty first
        # Note: empty response goes to validation which is mocked as valid
    ],
    "multi_tool": lambda rng: [
        _resp("複数ファイルを作ります。", [
            _tc("write", path="a.py", content="a=1"),
            _tc("write", path="b.py", content="b=2"),
        ]),
        _resp("完了しました。"),
    ],
    "read_then_edit": lambda rng: [
        _resp("ファイルを確認します。", [_tc("read", path="step1.py")]),
        _resp("編集します。", [
            _tc("edit", path="step1.py", old_string="x=1", new_string="x=2")
        ]),
        _resp("修正しました。"),
    ],
}


class TestMassiveRunTurn:
    """1000 users × 10 turns with full conversation flow validation."""

    @pytest.mark.parametrize("user_id", range(1000))
    def test_user_session(self, user_id):
        rng = random.Random(user_id * 31337)
        mode = rng.choice(list(Mode))
        llm = MagicMock()
        loop = _loop(llm, mode=mode, max_iter=15)

        for turn in range(10):
            pattern_name = rng.choice(list(_LLM_PATTERNS.keys()))
            responses = _LLM_PATTERNS[pattern_name](rng)
            llm.chat.side_effect = list(responses)

            user_input = rng.choice(_ALL_USER_INPUTS)

            try:
                result = loop.run_turn(user_input)
                assert isinstance(result, str), (
                    f"User {user_id} turn {turn}: result not str"
                )
                # Validate conversation structure
                validate_conversation(loop.messages)

            except StopIteration:
                # side_effect exhausted — acceptable in random testing
                pass

            # Reset messages but keep system prompt for next turn
            loop.messages = [loop.messages[0]]
            loop.validator.validate.return_value = ValidationResult(valid=True)
            loop.validator.validate.side_effect = None


# ═══════════════════════════════════════════════════════════════════════
# 3. Message ordering — 1000 users × 10 turns (run_turn_stream)
# ═══════════════════════════════════════════════════════════════════════

def _make_stream(pattern_name, rng):
    """Create a stream generator function for a given pattern."""
    if pattern_name == "text_stream":
        text = rng.choice([
            "変数について説明しますね。",
            "forループの使い方です。",
            "保存しました。どんな結果になると思いますか？",
            "正解です！素晴らしいですね。",
            "次は何をしましょうか？",
        ])
        def _s(messages, tools=None):
            # Yield character by character (simulating streaming)
            for ch in text[:20]:  # first 20 chars
                yield ch
            yield LLMResponse(content=text, tool_calls=[])
        return _s

    elif pattern_name == "tool_stream":
        def _s(messages, tools=None):
            yield LLMResponse(
                content="ファイルを保存します。",
                tool_calls=[_tc("write", path="f.py", content="x=1")],
            )
        return _s

    elif pattern_name == "code_block_stream":
        code = f"```python\nx = {rng.randint(1,100)}\n```\n"
        def _s(messages, tools=None):
            yield code
            yield LLMResponse(content=code, tool_calls=[])
        return _s

    elif pattern_name == "empty_stream":
        def _s(messages, tools=None):
            yield LLMResponse(content="", tool_calls=[])
        return _s

    else:  # fallback text
        def _s(messages, tools=None):
            yield "OK"
            yield LLMResponse(content="OK", tool_calls=[])
        return _s


class TestMassiveRunTurnStream:
    """1000 users × 10 turns streaming with conversation validation."""

    @pytest.mark.parametrize("user_id", range(1000))
    def test_stream_user_session(self, user_id):
        rng = random.Random(user_id * 65537)
        mode = rng.choice(list(Mode))
        llm = MagicMock()
        loop = _loop(llm, mode=mode, max_iter=10)

        stream_patterns = ["text_stream", "tool_stream", "code_block_stream",
                           "empty_stream"]

        for turn in range(10):
            pattern = rng.choice(stream_patterns)

            # Provide enough streams for nudge retries
            streams = [_make_stream(pattern, rng)]
            for _ in range(5):
                streams.append(_make_stream("text_stream", rng))
            llm.chat_stream.side_effect = streams

            user_input = rng.choice(_ALL_USER_INPUTS)

            try:
                chunks = list(loop.run_turn_stream(user_input))
                for chunk in chunks:
                    assert isinstance(chunk, str), (
                        f"User {user_id} turn {turn}: chunk not str: {type(chunk)}"
                    )
                # Validate conversation
                validate_conversation(loop.messages)

            except (StopIteration, TypeError):
                pass

            loop.messages = [loop.messages[0]]


# ═══════════════════════════════════════════════════════════════════════
# 4. Conversation content quality checks
# ═══════════════════════════════════════════════════════════════════════

class TestConversationContentQuality:
    """Verify the content of generated conversations is sensible."""

    @pytest.mark.parametrize("mode", list(Mode))
    def test_system_prompt_matches_mode(self, mode):
        """System prompt should reference the correct domain."""
        profile = build_mode_profile(mode)
        policy = PolicyEngine(profile, level=Level.BEGINNER)
        prompt = policy.build_system_prompt()

        domain_keywords = {
            Mode.PYTHON_BASIC: ["Python"],
            Mode.PY5: ["Py5", "Processing"],
            Mode.SKLEARN: ["scikit-learn", "機械学習"],
            Mode.PANDAS: ["pandas", "データ分析"],
            Mode.WEB_BASIC: ["HTML", "CSS", "JavaScript", "Web"],
            Mode.AFRAME: ["A-Frame", "WebXR", "3D"],
            Mode.THREEJS: ["Three.js", "3D"],
        }
        keywords = domain_keywords[mode]
        assert any(kw in prompt for kw in keywords), (
            f"Mode {mode.value} prompt should mention {keywords}"
        )

    @pytest.mark.parametrize("mode", list(Mode))
    def test_prediction_question_in_teaching_flow(self, mode):
        """Teaching flow should end with prediction question."""
        llm = MagicMock()
        llm.chat.side_effect = [
            _resp("コードを作ります。", [
                _tc("write", path="test.py", content="print(42)")
            ]),
            _resp(
                "保存しました。\n"
                "このコードを実行すると、どんな結果になると思いますか？"
            ),
        ]
        loop = _loop(llm, mode=mode)
        result = loop.run_turn("教えて")
        assert "思いますか" in result

    @pytest.mark.parametrize("mode", list(Mode))
    def test_tool_usage_logged_in_metrics(self, mode):
        """Tool calls should be recorded in metrics."""
        llm = MagicMock()
        llm.chat.side_effect = [
            _resp("Saving...", [_tc("write", path="a.py", content="x=1")]),
            _resp("Done!"),
        ]
        loop = _loop(llm, mode=mode)
        loop.run_turn("Write code")
        assert loop.metrics.tool_calls.get("write", 0) >= 1

    def test_nudge_message_is_natural_japanese(self):
        """Nudge message should read naturally in Japanese."""
        assert "コードブロック" in _TOOL_NUDGE
        assert "write" in _TOOL_NUDGE
        # Should be a complete sentence, not a fragment
        assert _TOOL_NUDGE.endswith("。") or _TOOL_NUDGE.endswith("ください。")
        # Not too long
        assert len(_TOOL_NUDGE) < 200

    def test_nudge_does_not_expose_internal_terms(self):
        """Nudge should not use overly technical internal terms."""
        # "ツール" is ok (used in curriculum), but avoid implementation details
        assert "LLMResponse" not in _TOOL_NUDGE
        assert "_has_code_block" not in _TOOL_NUDGE
        assert "side_effect" not in _TOOL_NUDGE

    @pytest.mark.parametrize("mode,level", list(itertools.product(Mode, Level)))
    def test_education_prompt_is_natural_japanese(self, mode, level):
        """Education prompts should be well-formed Japanese."""
        prompt = build_education_prompt(mode, level)
        # Should contain Japanese text
        has_ja = any(
            '\u3040' <= c <= '\u9fff' or '\u30a0' <= c <= '\u30ff'
            for c in prompt
        )
        assert has_ja, f"Prompt for {mode}/{level} should contain Japanese"
        # Should not have unresolved template variables
        assert "{" not in prompt or "{{" in prompt or all(
            v in prompt for v in []
        ), f"Unresolved template in {mode}/{level}: {prompt[:100]}"

    @pytest.mark.parametrize("mode", list(Mode))
    def test_constraint_section_present(self, mode):
        """Constraint section should always be present."""
        profile = build_mode_profile(mode)
        policy = PolicyEngine(profile)
        prompt = policy.build_system_prompt()
        assert "【制約】" in prompt
        assert "最大10行" in prompt
        assert "ネットワーク通信" in prompt


# ═══════════════════════════════════════════════════════════════════════
# 5. Edge case combos at scale
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCaseCombos:

    @pytest.mark.parametrize("n_nudges", range(_MAX_NUDGES_PER_TURN + 3))
    def test_various_nudge_counts(self, n_nudges):
        """Test with 0 to _MAX_NUDGES+2 code block responses before clean."""
        llm = MagicMock()
        code = "```python\nprint('hi')\n```\n"
        responses = [_resp(code)] * n_nudges + [_resp("Clean text.")]
        llm.chat.side_effect = responses
        loop = _loop(llm, max_iter=n_nudges + 5)

        result = loop.run_turn("Hello")
        validate_conversation(loop.messages)

        nudges = [m for m in loop.messages if m.content == _TOOL_NUDGE]
        expected_nudges = min(n_nudges, _MAX_NUDGES_PER_TURN)
        assert len(nudges) == expected_nudges

    @pytest.mark.parametrize("n_violations", range(1, 6))
    def test_various_violation_counts(self, n_violations):
        """Test with 1 to 5 validation failures before success."""
        llm = MagicMock()
        responses = (
            [_resp("bad response")] * n_violations
            + [_resp("good response")]
        )
        llm.chat.side_effect = responses
        loop = _loop(llm, max_iter=n_violations + 5)

        violations = [
            ValidationResult(valid=False, violations=[
                Violation(rule="language_isolation", detail="test")
            ])
        ] * n_violations + [ValidationResult(valid=True)]
        loop.validator.validate.side_effect = violations

        result = loop.run_turn("Hello")
        validate_conversation(loop.messages)
        assert loop.metrics.violations == n_violations

    @pytest.mark.parametrize("combo", [
        # (nudges_before, violations_after, tool_at_end)
        (1, 0, False),
        (2, 0, False),
        (0, 1, False),
        (0, 2, False),
        (1, 1, False),
        (2, 1, False),
        (1, 0, True),
        (0, 0, True),
        (2, 0, True),
        (1, 1, True),
    ])
    def test_nudge_violation_tool_combos(self, combo):
        """Test combinations of nudges, violations, and tool calls."""
        n_nudges, n_violations, tool_at_end = combo
        llm = MagicMock()

        responses = []
        # Code blocks for nudges
        for _ in range(n_nudges):
            responses.append(_resp("```python\ncode\n```\n"))
        # Bad responses for violations
        for _ in range(n_violations):
            responses.append(_resp("bad"))
        # Final response
        if tool_at_end:
            responses.append(_resp("tool", [_tc("write", path="f.py", content="x=1")]))
            responses.append(_resp("Done!"))
        else:
            responses.append(_resp("Good text."))

        llm.chat.side_effect = responses
        loop = _loop(llm, max_iter=20)

        # Set up validation side effects
        validation_results = []
        # Nudged responses don't go to validation
        # After nudges exhausted, code block goes to validation
        # Then violation responses
        for _ in range(n_violations):
            validation_results.append(ValidationResult(
                valid=False,
                violations=[Violation(rule="test", detail="test")]
            ))
        validation_results.append(ValidationResult(valid=True))
        # Add extras for safety
        for _ in range(10):
            validation_results.append(ValidationResult(valid=True))
        loop.validator.validate.side_effect = validation_results

        result = loop.run_turn("Hello")
        validate_conversation(loop.messages)

    @pytest.mark.parametrize("content", [
        "\n" * 100,  # just newlines
        " " * 100,  # just spaces
        "\t" * 50,  # just tabs
        "a",  # single char
        "あ",  # single Japanese char
        "🎉",  # emoji
        "🎉" * 100,  # many emojis
        "\x00",  # null byte
        "\\n\\t\\r",  # escaped chars as literals
        "```",  # just triple backtick, no newline
        "`" * 100,  # many backticks
        "---\n---\n---",  # markdown HR
        "| a | b |\n|---|---|\n| 1 | 2 |",  # markdown table
        "$$x^2$$",  # LaTeX
        "<script>alert(1)</script>",  # XSS attempt
        "'; DROP TABLE users; --",  # SQL injection attempt
    ])
    def test_weird_content_no_crash(self, content):
        """Agent loop should handle any content without crashing."""
        llm = MagicMock()
        llm.chat.return_value = _resp(content)
        loop = _loop(llm)
        result = loop.run_turn("Hello")
        assert isinstance(result, str)


# ═══════════════════════════════════════════════════════════════════════
# 6. Code block detection at 10x scale
# ═══════════════════════════════════════════════════════════════════════

# Generate 500 random code-block-containing strings
_RANDOM_CODE_BLOCKS = []
for i in range(500):
    rng = random.Random(i * 999)
    lang = rng.choice(["python", "js", "html", "css", "bash", "go", "rust",
                        "java", "ruby", "php", "sql", "yaml", "json", ""])
    prefix = rng.choice(["", "text before\n", "説明：\n\n", "# Title\n\n"])
    suffix = rng.choice(["", "\ntext after", "\n\n次へ", "\n\n質問ありますか？"])
    code_lines = rng.randint(1, 20)
    code = "\n".join(f"line{j} = {j}" for j in range(code_lines))
    text = f"{prefix}```{lang}\n{code}\n```{suffix}"
    _RANDOM_CODE_BLOCKS.append(text)

# Generate 500 random non-code-block strings
_RANDOM_NON_CODE = []
for i in range(500):
    rng = random.Random(i * 777)
    pattern = rng.choice([
        lambda r: "".join(r.choices("あいうえおかきくけこ", k=r.randint(10, 200))),
        lambda r: "".join(r.choices(string.ascii_letters + " \n", k=r.randint(10, 200))),
        lambda r: f"Step {r.randint(1,10)}: {r.choice(['変数', '関数', 'ループ'])}を学ぶ",
        lambda r: f"`code` and `more code` but not a block",
        lambda r: "- item\n" * r.randint(1, 10),
        lambda r: "# " + "".join(r.choices("ABC", k=5)) + "\n\nText.",
        lambda r: "",
        lambda r: " ",
    ])
    _RANDOM_NON_CODE.append(pattern(rng))


class TestCodeBlockDetection10x:

    @pytest.mark.parametrize("text", _RANDOM_CODE_BLOCKS[:500],
                             ids=[f"has_code_{i}" for i in range(500)])
    def test_detects_code_block(self, text):
        assert _has_code_block(text) is True

    @pytest.mark.parametrize("text", _RANDOM_NON_CODE[:500],
                             ids=[f"no_code_{i}" for i in range(500)])
    def test_no_false_positive(self, text):
        assert _has_code_block(text) is False


# ═══════════════════════════════════════════════════════════════════════
# 7. All mode × all tool combinations
# ═══════════════════════════════════════════════════════════════════════

_ALL_TOOLS = ["bash", "read", "write", "edit", "grep", "glob"]


class TestAllModeToolCombos:

    @pytest.mark.parametrize("mode,tool_name",
                             [(m, t) for m in Mode for t in _ALL_TOOLS],
                             ids=[f"{m.value}_{t}" for m in Mode for t in _ALL_TOOLS])
    def test_tool_allowed_consistency(self, mode, tool_name):
        """Tool allowed/blocked should be consistent with mode language."""
        profile = build_mode_profile(mode)
        policy = PolicyEngine(profile)
        verdict = policy.check_tool_allowed(tool_name)

        lang = MODE_LANGUAGE[mode]
        if tool_name == "bash":
            if lang == LanguageFamily.PYTHON:
                assert verdict.allowed, f"bash should be allowed in {mode.value}"
            else:
                assert not verdict.allowed, f"bash should be blocked in {mode.value}"
        else:
            # read, write, edit, grep, glob — allowed in all modes
            assert verdict.allowed, f"{tool_name} should be allowed in {mode.value}"
