"""Tests for the validation layer."""

import pytest
from novicode.config import Mode, build_mode_profile
from novicode.validator import Validator, correction_prompt


@pytest.fixture
def python_validator():
    return Validator(build_mode_profile(Mode.PYTHON_BASIC))


@pytest.fixture
def web_validator():
    return Validator(build_mode_profile(Mode.AFRAME))


@pytest.fixture
def pandas_validator():
    return Validator(build_mode_profile(Mode.PANDAS))


class TestLanguageIsolation:
    def test_python_mode_rejects_html(self, python_validator):
        code = '<html><body><div>hello</div></body></html>'
        result = python_validator.validate(code, "test.py")
        assert not result.valid
        assert any(v.rule == "language_isolation" for v in result.violations)

    def test_python_mode_accepts_python(self, python_validator):
        code = "def hello():\n    print('hello')\n"
        result = python_validator.validate(code, "test.py")
        assert result.valid

    def test_python_mode_rejects_js_patterns(self, python_validator):
        code = "document.getElementById('x')\nconsole.log('hi')\n"
        result = python_validator.validate(code, "test.py")
        assert not result.valid

    def test_web_mode_accepts_html(self, web_validator):
        code = '<html><body><a-scene></a-scene></body></html>'
        result = web_validator.validate(code, "test.html")
        assert result.valid


class TestImportValidation:
    def test_allowed_import(self, python_validator):
        code = "import math\nimport random\n"
        result = python_validator.validate(code, "test.py")
        assert result.valid

    def test_forbidden_import(self, python_validator):
        code = "import requests\n"
        result = python_validator.validate(code, "test.py")
        assert not result.valid
        assert any(v.rule == "forbidden_import" for v in result.violations)

    def test_pandas_allows_numpy(self, pandas_validator):
        code = "import numpy as np\nimport pandas as pd\n"
        result = pandas_validator.validate(code, "test.py")
        assert result.valid


class TestLineLimits:
    def test_exceeds_max_lines(self, python_validator):
        code = "\n".join(f"x = {i}" for i in range(400))
        result = python_validator.validate(code, "test.py")
        assert not result.valid
        assert any(v.rule == "max_lines" for v in result.violations)

    def test_within_max_lines(self, python_validator):
        code = "\n".join(f"x = {i}" for i in range(40))
        result = python_validator.validate(code, "test.py")
        assert result.valid


class TestForbiddenPatterns:
    def test_rejects_url(self, python_validator):
        code = 'url = "https://api.example.com/data"\n'
        result = python_validator.validate(code, "test.py")
        assert not result.valid

    def test_rejects_pip_install(self, python_validator):
        code = '# pip install requests\n'
        result = python_validator.validate(code, "test.py")
        assert not result.valid

    def test_rejects_subprocess(self, python_validator):
        code = 'import subprocess\nsubprocess.run(["ls"])\n'
        result = python_validator.validate(code, "test.py")
        assert not result.valid


class TestBatchValidation:
    def test_max_files_exceeded(self, python_validator):
        files = {f"file{i}.py": "x = 1\n" for i in range(5)}
        result = python_validator.validate_batch(files)
        assert not result.valid
        assert any(v.rule == "max_files" for v in result.violations)


class TestCorrectionPrompt:
    def test_correction_prompt_format(self):
        from novicode.validator import Violation
        violations = [Violation(rule="language_isolation", detail="HTML in Python mode")]
        prompt = correction_prompt(violations, "python_basic")
        assert "language_isolation" in prompt
        assert "python_basic" in prompt
