"""Tests for LLM adapter — tool definitions and structure."""

import pytest

from novicode.llm_adapter import TOOL_DEFINITIONS, Message, ToolCall, LLMResponse


class TestToolDefinitions:
    """Verify TOOL_DEFINITIONS have correct structure and enhanced descriptions."""

    def test_all_tools_have_type_function(self):
        for td in TOOL_DEFINITIONS:
            assert td["type"] == "function", f"Tool missing type=function: {td}"

    def test_all_tools_have_function_key(self):
        for td in TOOL_DEFINITIONS:
            assert "function" in td
            func = td["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func

    def test_all_tools_have_required_fields_in_params(self):
        for td in TOOL_DEFINITIONS:
            func = td["function"]
            params = func["parameters"]
            assert params["type"] == "object"
            assert "properties" in params
            assert "required" in params
            assert isinstance(params["required"], list)
            assert len(params["required"]) > 0

    def test_expected_tool_names(self):
        names = {td["function"]["name"] for td in TOOL_DEFINITIONS}
        expected = {"bash", "read", "write", "edit", "grep", "glob"}
        assert names == expected

    def test_bash_description_contains_must(self):
        bash = _get_tool("bash")
        desc = bash["function"]["description"]
        assert "MUST" in desc, f"bash description should contain MUST: {desc}"

    def test_write_description_contains_must(self):
        write = _get_tool("write")
        desc = write["function"]["description"]
        assert "MUST" in desc, f"write description should contain MUST: {desc}"

    def test_bash_description_mentions_execute(self):
        bash = _get_tool("bash")
        desc = bash["function"]["description"].lower()
        assert "execute" in desc or "run" in desc

    def test_write_description_mentions_save(self):
        write = _get_tool("write")
        desc = write["function"]["description"].lower()
        assert "save" in desc or "write" in desc or "create" in desc

    def test_bash_has_command_param(self):
        bash = _get_tool("bash")
        props = bash["function"]["parameters"]["properties"]
        assert "command" in props
        assert "command" in bash["function"]["parameters"]["required"]

    def test_write_has_path_and_content_params(self):
        write = _get_tool("write")
        props = write["function"]["parameters"]["properties"]
        assert "path" in props
        assert "content" in props
        required = write["function"]["parameters"]["required"]
        assert "path" in required
        assert "content" in required

    def test_edit_has_all_params(self):
        edit = _get_tool("edit")
        props = edit["function"]["parameters"]["properties"]
        assert "path" in props
        assert "old_string" in props
        assert "new_string" in props


class TestDataClasses:
    """Basic sanity checks for Message, ToolCall, LLMResponse."""

    def test_message_fields(self):
        m = Message(role="user", content="hello")
        assert m.role == "user"
        assert m.content == "hello"

    def test_tool_call_fields(self):
        tc = ToolCall(name="write", arguments={"path": "a.py", "content": "x=1"})
        assert tc.name == "write"
        assert tc.arguments["path"] == "a.py"

    def test_llm_response_defaults(self):
        r = LLMResponse()
        assert r.content == ""
        assert r.tool_calls == []
        assert r.raw == {}

    def test_llm_response_with_tool_calls(self):
        tc = ToolCall(name="bash", arguments={"command": "ls"})
        r = LLMResponse(content="Running...", tool_calls=[tc])
        assert len(r.tool_calls) == 1
        assert r.tool_calls[0].name == "bash"


def _get_tool(name: str) -> dict:
    """Helper to find a tool definition by name."""
    for td in TOOL_DEFINITIONS:
        if td["function"]["name"] == name:
            return td
    raise KeyError(f"Tool '{name}' not found in TOOL_DEFINITIONS")
