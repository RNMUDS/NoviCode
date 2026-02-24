"""Tests for policy engine."""

import pytest
from novicode.config import Mode, build_mode_profile
from novicode.policy_engine import PolicyEngine


@pytest.fixture
def python_policy():
    return PolicyEngine(build_mode_profile(Mode.PYTHON_BASIC))


@pytest.fixture
def web_policy():
    return PolicyEngine(build_mode_profile(Mode.THREEJS))


class TestToolPolicy:
    def test_python_allows_bash(self, python_policy):
        assert python_policy.check_tool_allowed("bash").allowed

    def test_python_allows_write(self, python_policy):
        assert python_policy.check_tool_allowed("write").allowed

    def test_web_blocks_bash(self, web_policy):
        assert not web_policy.check_tool_allowed("bash").allowed

    def test_web_allows_write(self, web_policy):
        assert web_policy.check_tool_allowed("write").allowed


class TestFileExtension:
    def test_python_allows_py(self, python_policy):
        assert python_policy.check_file_extension("test.py").allowed

    def test_python_blocks_html(self, python_policy):
        assert not python_policy.check_file_extension("test.html").allowed

    def test_web_allows_html(self, web_policy):
        assert web_policy.check_file_extension("test.html").allowed

    def test_web_allows_js(self, web_policy):
        assert web_policy.check_file_extension("test.js").allowed

    def test_web_blocks_py(self, web_policy):
        assert not web_policy.check_file_extension("test.py").allowed


class TestSystemPromptToolRules:
    """Verify build_system_prompt includes tool usage rules."""

    def test_python_prompt_contains_write_function(self, python_policy):
        prompt = python_policy.build_system_prompt()
        assert "write 関数" in prompt, "Python prompt should mention write 関数"

    def test_python_prompt_contains_bash_function(self, python_policy):
        prompt = python_policy.build_system_prompt()
        assert "bash 関数" in prompt, "Python prompt should mention bash 関数"

    def test_web_prompt_contains_write_function(self, web_policy):
        prompt = web_policy.build_system_prompt()
        assert "write 関数" in prompt, "Web prompt should mention write 関数"

    def test_web_prompt_contains_bash_unavailable(self, web_policy):
        prompt = web_policy.build_system_prompt()
        assert "bash は使えない" in prompt, "Web prompt should say bash is unavailable"

    def test_web_prompt_bash_unavailable_overrides_generic(self, web_policy):
        """Web prompt should contain 'bash は使えない' constraint even though
        the shared education template mentions bash 関数 generically."""
        prompt = web_policy.build_system_prompt()
        # The constraint section explicitly says bash is unavailable
        assert "bash は使えない" in prompt
        # And the constraint mentions using the browser instead
        assert "ブラウザ" in prompt


class TestScopeCheck:
    def test_rejects_rust(self, python_policy):
        assert not python_policy.check_scope("Write me a Rust program").allowed

    def test_rejects_docker(self, python_policy):
        assert not python_policy.check_scope("Create a docker container").allowed

    def test_accepts_python(self, python_policy):
        assert python_policy.check_scope("Write a sorting algorithm").allowed

    def test_accepts_generic(self, python_policy):
        assert python_policy.check_scope("Help me with loops").allowed
