"""Agent loop — orchestrates user input, LLM calls, tool execution, and validation."""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator

from novicode.config import ModeProfile, DEFAULT_MAX_ITERATIONS, build_mode_profile
from novicode.llm_adapter import LLMAdapter, Message, LLMResponse, TOOL_DEFINITIONS
from novicode.tool_registry import ToolRegistry
from novicode.security_manager import SecurityManager
from novicode.policy_engine import PolicyEngine
from novicode.validator import Validator, correction_prompt, educational_feedback
from novicode.session_manager import Session
from novicode.metrics import Metrics
from novicode.curriculum import extract_concepts
from novicode.progress import ProgressTracker


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
            self._log("llm_response", {"content": response.content, "tools": len(response.tool_calls)})

            if self.debug:
                print(f"  [iter {i+1}] content={response.content[:80]}... tools={len(response.tool_calls)}")

            # No tool calls → validate and return text
            if not response.tool_calls:
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
            self.messages.append(Message(role="user", content=f"Tool results:\n{tool_summary}"))

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

    def run_turn_stream(self, user_input: str) -> Iterator[str]:
        """Process one user turn, yielding text chunks for streaming display.

        Yields str chunks as they arrive from the LLM. Educational messages
        and level-up notifications are yielded before the LLM response.
        """
        self._educational_messages: list[str] = []

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

            if self.debug:
                print(f"  [iter {i+1}] calling LLM (streaming)...")

            # Stream from LLM
            response: LLMResponse | None = None
            streamed_chunks: list[str] = []
            is_first_text_iter = (i == 0) or not streamed_chunks

            for item in self.llm.chat_stream(self.messages, tools=tool_defs):
                if isinstance(item, str):
                    streamed_chunks.append(item)
                    # Only stream to user on the first successful text iteration
                    # (retries due to validation are not streamed)
                    if is_first_text_iter:
                        yield item
                elif isinstance(item, LLMResponse):
                    response = item

            if response is None:
                continue

            self._log("llm_response", {
                "content": response.content, "tools": len(response.tool_calls)
            })

            # No tool calls → validate
            if not response.tool_calls:
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
                    # Next iteration won't stream to user (is_first_text_iter = False)
                    is_first_text_iter = False
                    continue

                final_response = response.content
                self.messages.append(
                    Message(role="assistant", content=final_response)
                )
                self._track_concepts(response.content)
                break

            # Tool calls path (non-streaming)
            self.messages.append(
                Message(role="assistant", content=response.content)
            )
            tool_results = self._execute_tools(response)
            tool_summary = json.dumps(tool_results, ensure_ascii=False)
            self.messages.append(
                Message(role="user", content=f"Tool results:\n{tool_summary}")
            )

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
                self.progress.save()

            # Rebuild system prompt if mastered concepts changed
            current_mastered = self.progress.mastered_concepts()
            if current_mastered != prev_mastered:
                self.policy.mastered_concepts = current_mastered
                self.messages[0] = Message(
                    role="system", content=self.policy.build_system_prompt()
                )

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
