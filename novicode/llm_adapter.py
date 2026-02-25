"""LLM adapter — communicates with Ollama to run Qwen3 models."""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from collections.abc import Iterator
from dataclasses import dataclass, field

from novicode.config import OLLAMA_BASE_URL


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


class LLMAdapter:
    """Sends chat requests to a local Ollama instance."""

    def __init__(self, model: str, base_url: str | None = None) -> None:
        self.model = model
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")

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

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                f"Ensure Ollama is running with '{self.model}' loaded."
            ) from exc

        return self._parse_response(data)

    def chat_stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        chunk_timeout: float = 120.0,
    ) -> Iterator[str | LLMResponse]:
        """Stream a chat completion from Ollama.

        Yields:
            str: content chunks as they arrive
            LLMResponse: final complete response (always the last item)

        Parameters
        ----------
        chunk_timeout:
            Maximum seconds to wait for the next chunk of data.
            If no data arrives within this window, a ``TimeoutError``
            is raised so the caller can abort gracefully.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
        }
        if tools:
            payload["tools"] = tools

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            resp = urllib.request.urlopen(req, timeout=300)
        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Cannot reach Ollama at {self.base_url}. "
                f"Ensure Ollama is running with '{self.model}' loaded."
            ) from exc

        # Set a per-read timeout on the underlying socket so we don't
        # block forever if Ollama stops sending data mid-stream.
        sock = resp.fp.raw._sock if hasattr(resp.fp, "raw") else None
        if sock is not None:
            try:
                sock.settimeout(chunk_timeout)
            except Exception:
                pass

        accumulated_content = ""
        tool_calls: list[ToolCall] = []
        last_data: dict = {}

        try:
            for raw_line in resp:
                line = raw_line.decode().strip()
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
        except (TimeoutError, OSError) as exc:
            resp.close()
            raise TimeoutError(
                f"Ollama did not respond for {chunk_timeout:.0f}s. "
                "The model may be overloaded or unresponsive."
            ) from exc
        finally:
            resp.close()

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
