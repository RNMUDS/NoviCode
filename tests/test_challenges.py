"""Tests for challenges module."""

import pytest
from novicode.config import Mode
from novicode.curriculum import Level
from novicode.challenges import (
    CHALLENGES,
    Challenge,
    get_challenges,
    get_random_challenge,
    get_challenge_by_id,
    format_challenge,
    format_hint,
)


class TestChallengeDefinitions:
    def test_total_challenge_count(self):
        assert len(CHALLENGES) == 42

    def test_six_per_mode(self):
        for mode in Mode:
            mode_challenges = [c for c in CHALLENGES if c.mode == mode]
            assert len(mode_challenges) == 6, f"{mode.value} has {len(mode_challenges)} challenges, expected 6"

    def test_two_per_level_per_mode(self):
        for mode in Mode:
            for level in Level:
                level_challenges = get_challenges(mode, level)
                assert len(level_challenges) == 2, (
                    f"{mode.value}/{level.value} has {len(level_challenges)} challenges, expected 2"
                )

    def test_unique_ids(self):
        ids = [c.id for c in CHALLENGES]
        assert len(ids) == len(set(ids)), "Duplicate challenge IDs found"

    def test_all_challenges_have_content(self):
        for c in CHALLENGES:
            assert c.title, f"Challenge {c.id} has empty title"
            assert c.description, f"Challenge {c.id} has empty description"
            assert c.hint, f"Challenge {c.id} has empty hint"


class TestChallengeRetrieval:
    def test_get_challenges_by_mode_level(self):
        challenges = get_challenges(Mode.PYTHON_BASIC, Level.BEGINNER)
        assert len(challenges) == 2
        assert all(c.mode == Mode.PYTHON_BASIC for c in challenges)
        assert all(c.level == Level.BEGINNER for c in challenges)

    def test_get_random_challenge(self):
        ch = get_random_challenge(Mode.PYTHON_BASIC, Level.BEGINNER)
        assert ch is not None
        assert ch.mode == Mode.PYTHON_BASIC
        assert ch.level == Level.BEGINNER

    def test_get_challenge_by_id(self):
        ch = get_challenge_by_id("py_b1")
        assert ch is not None
        assert ch.id == "py_b1"
        assert ch.mode == Mode.PYTHON_BASIC

    def test_get_challenge_by_invalid_id(self):
        ch = get_challenge_by_id("nonexistent")
        assert ch is None


class TestChallengeFormatting:
    def test_format_challenge(self):
        ch = get_challenge_by_id("py_b1")
        assert ch is not None
        output = format_challenge(ch)
        assert "チャレンジ" in output
        assert ch.title in output
        assert "/hint" in output

    def test_format_hint(self):
        ch = get_challenge_by_id("py_b1")
        assert ch is not None
        output = format_hint(ch)
        assert "ヒント" in output
        assert ch.hint in output
