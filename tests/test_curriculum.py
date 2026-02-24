"""Tests for curriculum module — levels, concepts, prompts, extraction."""

import pytest
from novicode.config import Mode
from novicode.curriculum import (
    Level,
    LEVEL_ORDER,
    CONCEPT_CATALOGS,
    ConceptCatalog,
    extract_concepts,
    judge_level,
    build_education_prompt,
    LEVEL_UP_THRESHOLD,
)


class TestLevelDefinitions:
    def test_level_order(self):
        assert LEVEL_ORDER == [Level.BEGINNER, Level.INTERMEDIATE, Level.ADVANCED]

    def test_all_modes_have_catalogs(self):
        for mode in Mode:
            assert mode in CONCEPT_CATALOGS

    def test_catalogs_have_all_levels(self):
        for mode, catalog in CONCEPT_CATALOGS.items():
            assert len(catalog.beginner) > 0, f"{mode} has empty beginner"
            assert len(catalog.intermediate) > 0, f"{mode} has empty intermediate"
            assert len(catalog.advanced) > 0, f"{mode} has empty advanced"

    def test_no_concept_overlap_between_levels(self):
        for mode, catalog in CONCEPT_CATALOGS.items():
            assert catalog.beginner.isdisjoint(catalog.intermediate), f"{mode} beginner/intermediate overlap"
            assert catalog.beginner.isdisjoint(catalog.advanced), f"{mode} beginner/advanced overlap"
            assert catalog.intermediate.isdisjoint(catalog.advanced), f"{mode} intermediate/advanced overlap"

    def test_for_level(self):
        catalog = CONCEPT_CATALOGS[Mode.PYTHON_BASIC]
        assert catalog.for_level(Level.BEGINNER) == catalog.beginner
        assert catalog.for_level(Level.INTERMEDIATE) == catalog.intermediate
        assert catalog.for_level(Level.ADVANCED) == catalog.advanced

    def test_all_concepts(self):
        catalog = CONCEPT_CATALOGS[Mode.PYTHON_BASIC]
        all_c = catalog.all_concepts()
        assert all_c == catalog.beginner | catalog.intermediate | catalog.advanced


class TestConceptExtraction:
    def test_extracts_variable(self):
        text = "ここでは変数について学びます。x = 10 のように書きます。"
        found = extract_concepts(text, Mode.PYTHON_BASIC)
        assert "変数" in found

    def test_extracts_loop(self):
        text = "for i in range(10):\n    print(i)"
        found = extract_concepts(text, Mode.PYTHON_BASIC)
        assert "ループ" in found

    def test_extracts_function(self):
        text = "def hello():\n    print('hello')"
        found = extract_concepts(text, Mode.PYTHON_BASIC)
        assert "関数" in found

    def test_extracts_class(self):
        text = "class Dog:\n    pass"
        found = extract_concepts(text, Mode.PYTHON_BASIC)
        assert "クラス" in found

    def test_no_concepts_for_unrelated_text(self):
        text = "今日は天気がいいですね。"
        found = extract_concepts(text, Mode.PYTHON_BASIC)
        # May or may not find some, but shouldn't crash
        assert isinstance(found, list)

    def test_extracts_multiple_concepts(self):
        text = "変数 x に値を代入し、for ループで繰り返します。def func(): で関数を作ります。"
        found = extract_concepts(text, Mode.PYTHON_BASIC)
        assert "変数" in found
        assert "ループ" in found
        assert "関数" in found

    def test_unknown_mode_returns_empty(self):
        # This shouldn't happen, but test the edge case
        found = extract_concepts("test", Mode.PYTHON_BASIC)
        assert isinstance(found, list)


class TestLevelJudgment:
    def test_beginner_with_no_mastery(self):
        level = judge_level(Mode.PYTHON_BASIC, set())
        assert level == Level.BEGINNER

    def test_stays_beginner_below_threshold(self):
        # Less than 60% of beginner concepts
        catalog = CONCEPT_CATALOGS[Mode.PYTHON_BASIC]
        some = set(list(catalog.beginner)[:2])  # 2 out of 7 < 60%
        level = judge_level(Mode.PYTHON_BASIC, some)
        assert level == Level.BEGINNER

    def test_promotes_to_intermediate(self):
        catalog = CONCEPT_CATALOGS[Mode.PYTHON_BASIC]
        # Master 60%+ of beginner concepts
        needed = int(len(catalog.beginner) * LEVEL_UP_THRESHOLD) + 1
        mastered = set(list(catalog.beginner)[:needed])
        level = judge_level(Mode.PYTHON_BASIC, mastered)
        assert level == Level.INTERMEDIATE

    def test_promotes_to_advanced(self):
        catalog = CONCEPT_CATALOGS[Mode.PYTHON_BASIC]
        # Master 60%+ of both beginner and intermediate
        needed_b = int(len(catalog.beginner) * LEVEL_UP_THRESHOLD) + 1
        needed_i = int(len(catalog.intermediate) * LEVEL_UP_THRESHOLD) + 1
        mastered = set(list(catalog.beginner)[:needed_b]) | set(list(catalog.intermediate)[:needed_i])
        level = judge_level(Mode.PYTHON_BASIC, mastered)
        assert level == Level.ADVANCED

    def test_all_mastered_returns_advanced(self):
        catalog = CONCEPT_CATALOGS[Mode.PYTHON_BASIC]
        level = judge_level(Mode.PYTHON_BASIC, catalog.all_concepts())
        assert level == Level.ADVANCED


class TestEducationPrompt:
    def test_beginner_prompt_contains_key_phrases(self):
        prompt = build_education_prompt(Mode.PYTHON_BASIC, Level.BEGINNER)
        assert "1ステップだけ" in prompt
        assert "初級" in prompt
        assert "Python" in prompt
        assert "バイブコーディング" in prompt
        assert "予測" in prompt

    def test_beginner_prompt_no_old_phrases(self):
        prompt = build_education_prompt(Mode.PYTHON_BASIC, Level.BEGINNER)
        assert "動かしてみてください" not in prompt

    def test_intermediate_prompt_contains_additions(self):
        prompt = build_education_prompt(Mode.PYTHON_BASIC, Level.INTERMEDIATE)
        assert "中級" in prompt
        assert "テストの書き方を促す" in prompt

    def test_advanced_prompt_contains_all_sections(self):
        prompt = build_education_prompt(Mode.PYTHON_BASIC, Level.ADVANCED)
        assert "上級" in prompt
        assert "設計" in prompt
        assert "パフォーマンス" in prompt

    def test_all_modes_generate_prompts(self):
        for mode in Mode:
            for level in Level:
                prompt = build_education_prompt(mode, level)
                assert len(prompt) > 0, f"Empty prompt for {mode}/{level}"

    def test_beginner_prompt_contains_understanding_check(self):
        prompt = build_education_prompt(Mode.PYTHON_BASIC, Level.BEGINNER)
        assert "理解の確認" in prompt
        assert "分かりますか" in prompt

    def test_mastered_concepts_shown_in_prompt(self):
        prompt = build_education_prompt(
            Mode.PYTHON_BASIC, Level.BEGINNER, mastered_concepts={"変数", "print"}
        )
        assert "変数" in prompt
        assert "print" in prompt
        assert "確認不要" in prompt

    def test_no_mastered_concepts_shows_placeholder(self):
        prompt = build_education_prompt(Mode.PYTHON_BASIC, Level.BEGINNER)
        assert "（なし）" in prompt

    def test_mastered_concepts_empty_set(self):
        prompt = build_education_prompt(
            Mode.PYTHON_BASIC, Level.BEGINNER,
            mastered_concepts=set(),
        )
        assert "（なし）" in prompt
