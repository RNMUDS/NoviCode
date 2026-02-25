"""LLM adapter — communicates with Ollama to run local LLM models."""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.request
import urllib.error
from collections.abc import Iterator
from dataclasses import dataclass, field

from novicode.config import OLLAMA_BASE_URL

_MAX_CONNECT_RETRIES = 3
_RETRY_DELAY = 2.0  # seconds between retries
_QUEUE_POLL_INTERVAL = 0.5  # seconds — how often the main thread wakes to check signals


@dataclass
class Message:
    role: str        # "system" | "user" | "assistant"
    content: str


@dataclass
class ToolCall:
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a shell command. Use this to run code: `python file.py`. "
                "You MUST use this tool to execute code — never guess or fabricate output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to run"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                "Read file contents. Use this to check existing files before editing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": (
                "Create or overwrite a file. You MUST use this tool to save code — "
                "never output code as plain text in your response."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit",
            "description": (
                "Replace text in an existing file. "
                "Use this for small modifications to code you already wrote."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path"},
                    "old_string": {"type": "string", "description": "Text to find"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents with regex",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Directory to search"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                },
                "required": ["pattern"],
            },
        },
    },
]


def _stream_reader(resp: object, q: queue.Queue) -> None:
    """Read lines from an HTTP response in a background thread.

    Puts ``("line", bytes)`` for each line, ``("done", None)`` on
    completion, or ``("error", exc)`` on failure.
    """
    try:
        for raw_line in resp:
            q.put(("line", raw_line))
        q.put(("done", None))
    except Exception as exc:
        q.put(("error", exc))
    finally:
        try:
            resp.close()
        except Exception:
            pass


class LLMAdapter:
    """Sends chat requests to a local Ollama instance."""

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.model = model
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")

    def _open_with_retry(
        self,
        req: urllib.request.Request,
        timeout: float = 300,
    ) -> "http.client.HTTPResponse":
        """Open a URL request with retries on transient failures.

        Raises :class:`urllib.error.HTTPError` immediately on 4xx responses
        (client errors are not transient).  Retries only on connection-level
        failures and 5xx server errors.
        """
        last_exc: Exception | None = None
        for attempt in range(_MAX_CONNECT_RETRIES):
            try:
                return urllib.request.urlopen(req, timeout=timeout)
            except urllib.error.HTTPError as exc:
                if 400 <= exc.code < 500:
                    raise  # client errors are not transient
                last_exc = exc
            except urllib.error.URLError as exc:
                last_exc = exc
            if attempt < _MAX_CONNECT_RETRIES - 1:
                time.sleep(_RETRY_DELAY)
        raise ConnectionError(
            f"Ollama ({self.base_url}) に接続できません: {last_exc}\n"
            f"  確認: ollama serve が起動しているか / ollama pull {self.model}"
        ) from last_exc

    def _open_chat(
        self,
        payload: dict,
        timeout: float = 300,
    ) -> "http.client.HTTPResponse":
        """Open /api/chat with automatic tool-fallback on HTTP 400."""
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            return self._open_with_retry(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 400 and "tools" in payload:
                # Model likely doesn't support tool calling — retry without
                payload.pop("tools", None)
                body2 = json.dumps(payload).encode()
                req2 = urllib.request.Request(
                    f"{self.base_url}/api/chat",
                    data=body2,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                return self._open_with_retry(req2, timeout=timeout)
            raise ConnectionError(
                f"Ollama エラー (HTTP {exc.code}): {exc.reason}"
            ) from exc

    def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to Ollama (non-streaming)."""
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        with self._open_chat(payload) as resp:
            data = json.loads(resp.read().decode())

        return self._parse_response(data)

    def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Iterator[str | LLMResponse]:
        """Stream a chat completion from Ollama.

        Yields ``str`` chunks as they arrive, followed by a final
        :class:`LLMResponse`.  The socket reading runs in a daemon thread
        so the main thread stays responsive to signals (Ctrl+C).
        """
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        resp = self._open_chat(payload)

        # Read the HTTP response in a daemon thread so the main thread
        # can always respond to signals (KeyboardInterrupt / SIGALRM).
        q: queue.Queue = queue.Queue()
        reader_thread = threading.Thread(
            target=_stream_reader, args=(resp, q), daemon=True,
        )
        reader_thread.start()

        accumulated_content = ""
        tool_calls: list[ToolCall] = []
        last_data: dict = {}

        try:
            while True:
                try:
                    kind, value = q.get(timeout=_QUEUE_POLL_INTERVAL)
                except queue.Empty:
                    # Main thread wakes up here — Python checks pending
                    # signals (KeyboardInterrupt, SIGALRM) between bytecodes.
                    continue

                if kind == "done":
                    break

                if kind == "error":
                    raise value  # type: ignore[misc]

                # kind == "line"
                line = value.decode().strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                last_data = data
                msg = data.get("message", {})
                chunk = msg.get("content", "")

                if chunk:
                    accumulated_content += chunk
                    yield chunk

                # Tool calls come in the final message
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args = func.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {"raw": args}
                    tool_calls.append(ToolCall(name=name, arguments=args))
        finally:
            try:
                resp.close()
            except Exception:
                pass

        # Yield the final complete response
        yield LLMResponse(
            content=accumulated_content,
            tool_calls=tool_calls,
            raw=last_data,
        )

    def _parse_response(self, data: dict) -> LLMResponse:
        msg = data.get("message", {})
        content = msg.get("content", "")
        tool_calls: list[ToolCall] = []

        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            name = func.get("name", "")
            args = func.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"raw": args}
            tool_calls.append(ToolCall(name=name, arguments=args))

        return LLMResponse(content=content, tool_calls=tool_calls, raw=data)

    def ping(self) -> bool:
        """Check connectivity to Ollama."""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())
            models = [m["name"] for m in data.get("models", [])]
            return self.model in models or any(
                m.startswith(self.model.split(":")[0]) for m in models
            )
        except Exception:
            return False
