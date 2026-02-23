"""Tests for progress tracking module."""

import json
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from novicode.config import Mode
from novicode.curriculum import Level
from novicode.progress import ProgressTracker, MASTERY_THRESHOLD, PROGRESS_DIR


class TestConceptRecording:
    def test_record_single_concept(self):
        tracker = ProgressTracker(mode=Mode.PYTHON_BASIC)
        tracker.record_concepts(["変数"])
        assert tracker.concept_counts["変数"] == 1

    def test_record_multiple_concepts(self):
        tracker = ProgressTracker(mode=Mode.PYTHON_BASIC)
        tracker.record_concepts(["変数", "ループ", "関数"])
        assert tracker.concept_counts["変数"] == 1
        assert tracker.concept_counts["ループ"] == 1
        assert tracker.concept_counts["関数"] == 1

    def test_record_accumulates(self):
        tracker = ProgressTracker(mode=Mode.PYTHON_BASIC)
        tracker.record_concepts(["変数"])
        tracker.record_concepts(["変数"])
        tracker.record_concepts(["変数"])
        assert tracker.concept_counts["変数"] == 3

    def test_empty_concepts(self):
        tracker = ProgressTracker(mode=Mode.PYTHON_BASIC)
        tracker.record_concepts([])
        assert len(tracker.concept_counts) == 0


class TestMastery:
    def test_not_mastered_below_threshold(self):
        tracker = ProgressTracker(mode=Mode.PYTHON_BASIC)
        tracker.record_concepts(["変数", "変数"])  # 2 times, threshold is 3
        assert "変数" not in tracker.mastered_concepts()

    def test_mastered_at_threshold(self):
        tracker = ProgressTracker(mode=Mode.PYTHON_BASIC)
        for _ in range(MASTERY_THRESHOLD):
            tracker.record_concepts(["変数"])
        assert "変数" in tracker.mastered_concepts()

    def test_mastered_above_threshold(self):
        tracker = ProgressTracker(mode=Mode.PYTHON_BASIC)
        for _ in range(MASTERY_THRESHOLD + 2):
            tracker.record_concepts(["変数"])
        assert "変数" in tracker.mastered_concepts()


class TestLevelUpdate:
    def test_starts_at_beginner(self):
        tracker = ProgressTracker(mode=Mode.PYTHON_BASIC)
        assert tracker.level == Level.BEGINNER

    def test_no_level_change_initially(self):
        tracker = ProgressTracker(mode=Mode.PYTHON_BASIC)
        result = tracker.update_level()
        assert result is None
        assert tracker.level == Level.BEGINNER

    def test_level_up_returns_new_level(self):
        tracker = ProgressTracker(mode=Mode.PYTHON_BASIC)
        # Master 60%+ of beginner concepts (5 of 7 = 71%)
        from novicode.curriculum import CONCEPT_CATALOGS
        catalog = CONCEPT_CATALOGS[Mode.PYTHON_BASIC]
        beginner_list = list(catalog.beginner)[:5]
        for concept in beginner_list:
            for _ in range(MASTERY_THRESHOLD):
                tracker.record_concepts([concept])
        result = tracker.update_level()
        assert result == Level.INTERMEDIATE
        assert tracker.level == Level.INTERMEDIATE

    def test_no_change_returns_none(self):
        tracker = ProgressTracker(mode=Mode.PYTHON_BASIC)
        tracker.record_concepts(["変数"])  # Only 1 occurrence
        result = tracker.update_level()
        assert result is None


class TestDisplay:
    def test_display_output_format(self):
        tracker = ProgressTracker(mode=Mode.PYTHON_BASIC)
        tracker.record_concepts(["変数", "変数", "変数"])
        output = tracker.display()
        assert "学習進捗" in output
        assert "python_basic" in output
        assert "✅" in output  # 変数 should be mastered
        assert "初級" in output

    def test_display_shows_partial_progress(self):
        tracker = ProgressTracker(mode=Mode.PYTHON_BASIC)
        tracker.record_concepts(["ループ"])
        output = tracker.display()
        assert "🔄" in output  # partial progress


class TestPersistence:
    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "progress"
            with patch("novicode.progress.PROGRESS_DIR", test_dir):
                # Save
                tracker = ProgressTracker(mode=Mode.PYTHON_BASIC)
                tracker.record_concepts(["変数", "変数", "変数"])
                tracker.save()

                # Load
                loaded = ProgressTracker.load(Mode.PYTHON_BASIC)
                # Since PROGRESS_DIR is patched only for save, we need to
                # verify the file was written
                path = test_dir / "python_basic.json"
                assert path.exists()
                data = json.loads(path.read_text())
                assert data["concept_counts"]["変数"] == 3

    def test_load_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "progress"
            with patch("novicode.progress.PROGRESS_DIR", test_dir):
                tracker = ProgressTracker.load(Mode.PYTHON_BASIC)
                assert tracker.mode == Mode.PYTHON_BASIC
                assert len(tracker.concept_counts) == 0
                assert tracker.level == Level.BEGINNER

    def test_load_corrupt_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_dir = Path(tmpdir) / "progress"
            test_dir.mkdir(parents=True)
            path = test_dir / "python_basic.json"
            path.write_text("not valid json {{{")
            with patch("novicode.progress.PROGRESS_DIR", test_dir):
                tracker = ProgressTracker.load(Mode.PYTHON_BASIC)
                assert tracker.mode == Mode.PYTHON_BASIC
                assert len(tracker.concept_counts) == 0
