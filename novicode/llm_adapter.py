"""LLM adapter — communicates with Ollama to run Qwen3 models."""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from dataclasses import dataclass, field

from novicode.config import OLLAMA_BASE_URL, SUPPORTED_MODELS


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
            "description": "Execute a bash command",
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
            "description": "Read a file",
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
            "description": "Write content to a file",
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
            "description": "Edit a file by replacing old text with new text",
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
        if model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model}")
        self.model = model
        self.base_url = (base_url or OLLAMA_BASE_URL).rstrip("/")

    def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> LLMResponse:
        """Send a chat completion request to Ollama."""
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
