"""Agent loop — orchestrates user input, LLM calls, tool execution, and validation."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field

from novicode.config import ModeProfile, DEFAULT_MAX_ITERATIONS, build_mode_profile
from novicode.llm_adapter import LLMAdapter, Message, LLMResponse, ToolCall, TOOL_DEFINITIONS
from novicode.tool_registry import ToolRegistry
from novicode.security_manager import SecurityManager
from novicode.policy_engine import PolicyEngine
from novicode.validator import Validator, correction_prompt, educational_feedback
from novicode.session_manager import Session
from novicode.metrics import Metrics
from novicode.curriculum import extract_concepts
from novicode.progress import ProgressTracker


@dataclass
class StatusEvent:
    """Emitted by *run_turn_stream* to signal progress to the UI layer."""
    kind: str      # "thinking" | "tool_start" | "tool_done"
    detail: str = ""


@dataclass
class CodeWriteEvent:
    """Emitted after a successful write tool execution for UI rendering."""
    path: str
    content: str
    lang: str


# ── Text-based tool call parsing ─────────────────────────────────────

_TEXT_TOOL_RE = re.compile(
    r"<function=(\w+)>(.*?)</function>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<parameter=(\w+)>(.*?)</parameter>",
    re.DOTALL,
)

# Match: write({ path: "...", content: "..." }) or write({path:"...",content:"..."})
_FUNC_CALL_RE = re.compile(
    r"(\w+)\(\s*\{(.*?)\}\s*\)",
    re.DOTALL,
)
# Match key-value pairs inside { }: key: "value" or key: 'value'
_KV_RE = re.compile(
    r"""(\w+)\s*:\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')""",
    re.DOTALL,
)

# Match positional call: write("path", "content") or write('path', 'content')
# Also matches prefixed forms like: py5.write(...), tool.write(...)
_POSITIONAL_CALL_RE = re.compile(
    r"""(?:\w+\.)?write\(\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')\s*,\s*(?:"((?:[^"\\]|\\.)*)"|'((?:[^'\\]|\\.)*)')\s*\)""",
    re.DOTALL,
)


def _unescape(s: str) -> str:
    """Unescape common escape sequences in a string value."""
    return (
        s.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace('\\"', '"')
        .replace("\\'", "'")
    )


def _parse_text_tool_calls(text: str) -> tuple[list[ToolCall], str]:
    """Parse tool calls embedded as plain text in LLM output.

    Supports three formats:
    1. XML: ``<function=write><parameter=path>...</parameter></function>``
    2. JS-like: ``write({ path: "...", content: "..." })``
    3. Positional: ``write("path", "content")`` (also ``py5.write(...)``)

    Returns a list of parsed :class:`ToolCall` objects and the text with
    those fragments removed.
    """
    calls: list[ToolCall] = []
    patterns_matched: list[re.Pattern] = []

    # Format 1: XML-style <function=NAME>...</function>
    for m in _TEXT_TOOL_RE.finditer(text):
        name = m.group(1)
        body = m.group(2)
        args: dict[str, str] = {}
        for pm in _PARAM_RE.finditer(body):
            args[pm.group(1)] = pm.group(2)
        calls.append(ToolCall(name=name, arguments=args))
    if calls:
        patterns_matched.append(_TEXT_TOOL_RE)

    # Format 2: JS-like write({ path: "...", content: "..." })
    known_tools = {"write", "read", "edit", "bash", "grep", "glob"}
    for m in _FUNC_CALL_RE.finditer(text):
        name = m.group(1)
        if name not in known_tools:
            continue
        body = m.group(2)
        args = {}
        for kv in _KV_RE.finditer(body):
            key = kv.group(1)
            val = kv.group(2) if kv.group(2) is not None else kv.group(3)
            val = _unescape(val)
            args[key] = val
        if args:
            calls.append(ToolCall(name=name, arguments=args))
            if _FUNC_CALL_RE not in patterns_matched:
                patterns_matched.append(_FUNC_CALL_RE)

    # Format 3: Positional write("path", "content") / py5.write("path", "content")
    if not calls:
        for m in _POSITIONAL_CALL_RE.finditer(text):
            path = m.group(1) if m.group(1) is not None else m.group(2)
            content = m.group(3) if m.group(3) is not None else m.group(4)
            path = _unescape(path)
            content = _unescape(content)
            calls.append(ToolCall(name="write", arguments={"path": path, "content": content}))
        if calls:
            patterns_matched.append(_POSITIONAL_CALL_RE)

    cleaned = text
    for pat in patterns_matched:
        cleaned = pat.sub("", cleaned)
    cleaned = cleaned.strip()
    return calls, cleaned


def _lang_from_path(path: str) -> str:
    """Guess language identifier from file extension."""
    ext = os.path.splitext(path)[1].lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".sh": "bash",
        ".yml": "yaml",
        ".yaml": "yaml",
    }.get(ext, "python")


_CODE_BLOCK_RE = re.compile(r"```\w*\n")

_TOOL_NUDGE = (
    "あなたの応答にコードブロックが含まれていますが、ツールが使われていません。\n"
    "コードは必ず write ツールでファイルに保存してください。\n"
    "コードをテキストとして表示するのではなく、write 関数を呼び出してください。"
)

_TOOL_NUDGE_AFTER_WRITE = (
    "コードは既に write ツールでファイルに保存されています。\n"
    "返答にコードを書く必要はありません。マークダウンのコードブロック（```）は使わないでください。\n"
    "コードの新しい部分の説明（箇条書き2〜3個）と"
    "「実行してみましょうか？」の質問だけを書いてください。"
)

def _build_write_reminder(tool_calls: list, tool_results: list[dict]) -> str:
    """Build a reminder message with actual file paths from write/edit results."""
    paths = [
        r.get("path", "")
        for tc, r in zip(tool_calls, tool_results)
        if tc.name in ("write", "edit") and r.get("path")
    ]
    path_str = "、".join(f"`{p}`" for p in paths) if paths else "ファイル"
    return (
        f"\n\n【重要】コードは {path_str} に保存済みです。"
        "返答にコードを書かないでください（``` は禁止）。"
        "コードの説明（箇条書き2〜3個）と「実行してみましょうか？」の質問だけを書いてください。"
        "ツール名（write, read, bash 等）を返答に含めないでください。"
    )

_MAX_NUDGES_PER_TURN = 2


def _has_code_block(text: str) -> bool:
    """Return True if text contains a fenced code block."""
    return bool(_CODE_BLOCK_RE.search(text))


class AgentLoop:
    """Main agentic loop: prompt → LLM → tools → validate → iterate."""

    def __init__(
        self,
        llm: LLMAdapter,
        profile: ModeProfile,
        tools: ToolRegistry,
        validator: Validator,
        policy: PolicyEngine,
        session: Session,
        metrics: Metrics,
        progress: ProgressTracker | None = None,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        research: bool = False,
        debug: bool = False,
    ) -> None:
        self.llm = llm
        self.profile = profile
        self.tools = tools
        self.validator = validator
        self.policy = policy
        self.session = session
        self.metrics = metrics
        self.progress = progress
        self.max_iterations = max_iterations
        self.research = research
        self.debug = debug
        self.messages: list[Message] = [
            Message(role="system", content=policy.build_system_prompt())
        ]

    def run_turn(self, user_input: str) -> str:
        """Process one user turn (may involve multiple LLM iterations)."""
        self._educational_messages: list[str] = []
        nudge_count = 0
        write_used = False

        # Scope check
        scope = self.policy.check_scope(user_input)
        if not scope.allowed:
            self._log("scope_rejection", {"input": user_input, "reason": scope.reason})
            return f"Sorry, this request is outside the supported scope.\n\n{scope.reason}"

        self.messages.append(Message(role="user", content=user_input))
        self._log("user", {"content": user_input})

        final_response = ""
        for i in range(self.max_iterations):
            self.metrics.increment_iteration()

            # Call LLM
            tool_defs = self._filtered_tool_defs()
            response = self.llm.chat(self.messages, tools=tool_defs)

            # Parse text-based tool calls (e.g. <function=write>...</function>)
            if not response.tool_calls and response.content:
                text_calls, cleaned = _parse_text_tool_calls(response.content)
                if text_calls:
                    response.tool_calls.extend(text_calls)
                    response.content = cleaned

            self._log("llm_response", {"content": response.content, "tools": len(response.tool_calls)})

            if self.debug:
                print(f"  [iter {i+1}] content={response.content[:80]}... tools={len(response.tool_calls)}")

            # No tool calls → check for code blocks in text, then validate
            if not response.tool_calls:
                # Nudge: LLM output code as text instead of using tools
                if _has_code_block(response.content) and nudge_count < _MAX_NUDGES_PER_TURN:
                    nudge_count += 1
                    self._log("nudge", {"reason": "code_block_without_tool", "count": nudge_count})
                    self.messages.append(Message(role="assistant", content=response.content))
                    nudge_msg = _TOOL_NUDGE_AFTER_WRITE if write_used else _TOOL_NUDGE
                    self.messages.append(Message(role="user", content=nudge_msg))
                    if self.debug:
                        print(f"  [nudge {nudge_count}] code block detected without tool call")
                    continue

                validation = self.validator.validate(response.content, "response.py")
                if not validation.valid:
                    self._log("violation", {"violations": [v.__dict__ for v in validation.violations]})
                    self.metrics.record_violation()
                    self.metrics.record_retry()
                    # Show educational feedback to user
                    feedback = educational_feedback(validation.violations)
                    if feedback:
                        self._educational_messages.append(feedback)
                    correction = correction_prompt(validation.violations, self.profile.mode.value)
                    self.messages.append(Message(role="assistant", content=response.content))
                    self.messages.append(Message(role="user", content=correction))
                    continue

                final_response = response.content
                self.messages.append(Message(role="assistant", content=final_response))

                # Track concepts from the response
                self._track_concepts(response.content)
                break

            # Execute tool calls
            self.messages.append(Message(role="assistant", content=response.content))
            tool_results = self._execute_tools(response)
            tool_summary = json.dumps(tool_results, ensure_ascii=False)
            has_write = any(tc.name in ("write", "edit") for tc in response.tool_calls)
            if has_write:
                write_used = True
            result_msg = f"Tool results:\n{tool_summary}"
            if has_write:
                result_msg += _build_write_reminder(response.tool_calls, tool_results)
            self.messages.append(Message(role="user", content=result_msg))

            # Validate any write/edit output
            for tc, result in zip(response.tool_calls, tool_results):
                if tc.name in ("write", "edit") and "error" not in result:
                    content = tc.arguments.get("content", "")
                    path = tc.arguments.get("path", "")
                    if content:
                        vr = self.validator.validate(content, path)
                        if not vr.valid:
                            self._log("violation", {"violations": [v.__dict__ for v in vr.violations]})
                            self.metrics.record_violation()
        else:
            final_response = "(Max iterations reached. Please simplify your request.)"

        self.session.add("turn_complete", {"response_length": len(final_response)})

        # Prepend any educational messages
        if self._educational_messages:
            edu_block = "\n\n".join(self._educational_messages)
            final_response = edu_block + "\n\n---\n\n" + final_response

        return final_response

    def run_turn_stream(self, user_input: str) -> Iterator[str | StatusEvent | CodeWriteEvent]:
        """Process one user turn, yielding text chunks and status events.

        Yields ``str`` chunks as they arrive from the LLM,
        :class:`StatusEvent` instances so the UI can drive a spinner,
        and :class:`CodeWriteEvent` after successful write tool execution.
        Educational messages and level-up notifications are yielded after
        the LLM response.
        """
        self._educational_messages: list[str] = []
        nudge_count = 0
        write_used = False

        # Scope check
        scope = self.policy.check_scope(user_input)
        if not scope.allowed:
            self._log("scope_rejection", {"input": user_input, "reason": scope.reason})
            yield f"Sorry, this request is outside the supported scope.\n\n{scope.reason}"
            return

        self.messages.append(Message(role="user", content=user_input))
        self._log("user", {"content": user_input})

        final_response = ""
        for i in range(self.max_iterations):
            self.metrics.increment_iteration()

            tool_defs = self._filtered_tool_defs()
            yield StatusEvent("thinking")

            if self.debug:
                print(f"  [iter {i+1}] calling LLM (streaming)...")

            # Stream from LLM — buffer all chunks, yield only after validation
            response: LLMResponse | None = None
            streamed_chunks: list[str] = []

            for item in self.llm.chat_stream(self.messages, tools=tool_defs):
                if isinstance(item, str):
                    streamed_chunks.append(item)
                elif isinstance(item, LLMResponse):
                    response = item

            if response is None:
                continue

            # Parse text-based tool calls (e.g. <function=write>...</function>)
            if not response.tool_calls and response.content:
                text_calls, cleaned = _parse_text_tool_calls(response.content)
                if text_calls:
                    response.tool_calls.extend(text_calls)
                    response.content = cleaned
                    # Rebuild streamed_chunks to match cleaned content
                    streamed_chunks = [cleaned] if cleaned else []

            self._log("llm_response", {
                "content": response.content, "tools": len(response.tool_calls)
            })

            # No tool calls → validate before yielding to user
            if not response.tool_calls:
                # Nudge: LLM output code as text instead of using tools
                if _has_code_block(response.content) and nudge_count < _MAX_NUDGES_PER_TURN:
                    nudge_count += 1
                    self._log("nudge", {"reason": "code_block_without_tool", "count": nudge_count})
                    self.messages.append(Message(role="assistant", content=response.content))
                    nudge_msg = _TOOL_NUDGE_AFTER_WRITE if write_used else _TOOL_NUDGE
                    self.messages.append(Message(role="user", content=nudge_msg))
                    if self.debug:
                        print(f"  [nudge {nudge_count}] code block detected without tool call")
                    continue

                validation = self.validator.validate(response.content, "response.py")
                if not validation.valid:
                    self._log("violation", {
                        "violations": [v.__dict__ for v in validation.violations]
                    })
                    self.metrics.record_violation()
                    self.metrics.record_retry()
                    feedback = educational_feedback(validation.violations)
                    if feedback:
                        self._educational_messages.append(feedback)
                    correction = correction_prompt(
                        validation.violations, self.profile.mode.value
                    )
                    self.messages.append(
                        Message(role="assistant", content=response.content)
                    )
                    self.messages.append(Message(role="user", content=correction))
                    continue

                # Validation passed — yield all buffered chunks to user
                for chunk in streamed_chunks:
                    yield chunk

                final_response = response.content
                self.messages.append(
                    Message(role="assistant", content=final_response)
                )
                self._track_concepts(response.content)
                break

            # Tool calls path — execute tools, suppress text if write/edit
            # (write responses often echo code in text; the follow-up iter
            #  provides the clean explanation, so we skip this text)
            has_write = any(tc.name in ("write", "edit") for tc in response.tool_calls)
            if not has_write:
                for chunk in streamed_chunks:
                    yield chunk
            self.messages.append(
                Message(role="assistant", content=response.content)
            )
            tool_names = ", ".join(tc.name for tc in response.tool_calls)
            yield StatusEvent("tool_start", tool_names)
            tool_results = self._execute_tools(response)
            yield StatusEvent("tool_done")
            tool_summary = json.dumps(tool_results, ensure_ascii=False)
            if has_write:
                write_used = True
            result_msg = f"Tool results:\n{tool_summary}"
            if has_write:
                result_msg += _build_write_reminder(response.tool_calls, tool_results)
            self.messages.append(Message(role="user", content=result_msg))

            for tc, result in zip(response.tool_calls, tool_results):
                if tc.name in ("write", "edit") and "error" not in result:
                    content = tc.arguments.get("content", "")
                    path = tc.arguments.get("path", "")
                    if content:
                        vr = self.validator.validate(content, path)
                        if not vr.valid:
                            self._log("violation", {
                                "violations": [v.__dict__ for v in vr.violations]
                            })
                            self.metrics.record_violation()

            # Yield CodeWriteEvent for each successful write/edit
            for tc, result in zip(response.tool_calls, tool_results):
                if tc.name in ("write", "edit") and "error" not in result:
                    wpath = tc.arguments.get("path", "")
                    wcontent = tc.arguments.get("content", "")
                    if wpath and wcontent:
                        yield CodeWriteEvent(
                            path=wpath,
                            content=wcontent,
                            lang=_lang_from_path(wpath),
                        )
        else:
            yield "(Max iterations reached. Please simplify your request.)"
            final_response = "(Max iterations reached.)"

        self.session.add("turn_complete", {"response_length": len(final_response)})

        # Yield educational messages (level-up notifications etc.)
        if self._educational_messages:
            edu_block = "\n\n".join(self._educational_messages)
            yield "\n\n" + edu_block

    def _execute_tools(self, response: LLMResponse) -> list[dict]:
        results = []
        for tc in response.tool_calls:
            self.metrics.record_tool_call(tc.name)
            self._log("tool_call", {"name": tc.name, "args": tc.arguments})
            result = self.tools.execute(tc.name, tc.arguments)
            self._log("tool_result", {"name": tc.name, "result": _truncate(result)})
            results.append(result)
        return results

    def _filtered_tool_defs(self) -> list[dict]:
        allowed = self.tools.available_tools()
        return [td for td in TOOL_DEFINITIONS if td["function"]["name"] in allowed]

    def _log(self, entry_type: str, data: dict) -> None:
        if self.research:
            self.session.add(entry_type, data)

    def _track_concepts(self, text: str) -> None:
        """Extract concepts from LLM response and update progress."""
        if self.progress is None:
            return
        concepts = extract_concepts(text, self.profile.mode)
        if concepts:
            prev_mastered = self.progress.mastered_concepts().copy()
            self.progress.record_concepts(concepts)
            self.metrics.concepts_taught.extend(concepts)
            self._log("concepts", {"found": concepts})

            cur_mastered = self.progress.mastered_concepts()

            # Check for level-up
            new_level = self.progress.update_level()
            if new_level is not None:
                level_ja = {
                    "beginner": "初級", "intermediate": "中級", "advanced": "上級"
                }
                msg = (
                    f"🎉 レベルアップ！ {level_ja.get(new_level.value, new_level.value)} "
                    f"に到達しました！"
                )
                self._educational_messages.append(msg)
                self._log("level_up", {"new_level": new_level.value})
                self.policy.level = new_level

            # Rebuild system prompt when mastered concepts change
            cur_mastered = self.progress.mastered_concepts()
            if cur_mastered != prev_mastered or new_level is not None:
                self.policy.mastered_concepts = cur_mastered
                self.messages[0] = Message(
                    role="system", content=self.policy.build_system_prompt()
                )
                self.progress.save()

    def restore_messages(self, messages: list[Message]) -> None:
        """Restore conversation history (for session resume)."""
        self.messages = messages


def _truncate(d: dict, limit: int = 500) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, str) and len(v) > limit:
            out[k] = v[:limit] + "..."
        else:
            out[k] = v
    return out
