"""Tests for policy engine."""

import pytest
from rnnr.config import Mode, build_mode_profile
from rnnr.policy_engine import PolicyEngine


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


class TestScopeCheck:
    def test_rejects_rust(self, python_policy):
        assert not python_policy.check_scope("Write me a Rust program").allowed

    def test_rejects_docker(self, python_policy):
        assert not python_policy.check_scope("Create a docker container").allowed

    def test_accepts_python(self, python_policy):
        assert python_policy.check_scope("Write a sorting algorithm").allowed

    def test_accepts_generic(self, python_policy):
        assert python_policy.check_scope("Help me with loops").allowed
