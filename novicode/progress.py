"""Progress tracking — concept occurrence counting and persistence."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from novicode.config import Mode
from novicode.curriculum import (
    CONCEPT_CATALOGS,
    Level,
    LEVEL_ORDER,
    judge_level,
)

MASTERY_THRESHOLD = 3  # concept seen 3+ times → mastered
PROGRESS_DIR = Path.home() / ".novicode" / "progress"


@dataclass
class ProgressTracker:
    """Tracks concept occurrences and determines mastery."""

    mode: Mode
    concept_counts: dict[str, int] = field(default_factory=dict)
    _level: Level = Level.BEGINNER

    @property
    def level(self) -> Level:
        return self._level

    def record_concepts(self, concepts: list[str]) -> None:
        """Record concept occurrences from a single LLM response."""
        for concept in concepts:
            self.concept_counts[concept] = self.concept_counts.get(concept, 0) + 1

    def mastered_concepts(self) -> set[str]:
        """Return set of concepts seen >= MASTERY_THRESHOLD times."""
        return {c for c, n in self.concept_counts.items() if n >= MASTERY_THRESHOLD}

    def update_level(self) -> Level | None:
        """Re-evaluate level. Returns new level if changed, else None."""
        old = self._level
        new = judge_level(self.mode, self.mastered_concepts())
        self._level = new
        if new != old:
            return new
        return None

    # ── Display ──────────────────────────────────────────────────

    def display(self) -> str:
        """Generate display string for /progress command."""
        catalog = CONCEPT_CATALOGS.get(self.mode)
        if catalog is None:
            return "No curriculum defined for this mode."

        lines: list[str] = []
        lines.append(f"📊 学習進捗 — {self.mode.value} (レベル: {self._level.value})")
        lines.append("")

        mastered = self.mastered_concepts()

        for level in LEVEL_ORDER:
            level_concepts = sorted(catalog.for_level(level))
            label = {"beginner": "初級", "intermediate": "中級", "advanced": "上級"}[level.value]
            lines.append(f"【{label}】")
            for concept in level_concepts:
                count = self.concept_counts.get(concept, 0)
                if concept in mastered:
                    mark = "✅"
                elif count > 0:
                    mark = f"🔄 ({count}/{MASTERY_THRESHOLD})"
                else:
                    mark = "⬜"
                lines.append(f"  {mark} {concept}")
            lines.append("")

        total = len(catalog.all_concepts())
        mastered_count = len(mastered & catalog.all_concepts())
        lines.append(f"習得率: {mastered_count}/{total} ({mastered_count * 100 // total if total else 0}%)")
        return "\n".join(lines)

    # ── Persistence ──────────────────────────────────────────────

    def save(self) -> Path:
        """Persist progress to disk."""
        PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
        path = PROGRESS_DIR / f"{self.mode.value}.json"
        data = {
            "mode": self.mode.value,
            "level": self._level.value,
            "concept_counts": self.concept_counts,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        return path

    @classmethod
    def load(cls, mode: Mode) -> ProgressTracker:
        """Load progress from disk, or return fresh tracker."""
        path = PROGRESS_DIR / f"{mode.value}.json"
        if not path.exists():
            return cls(mode=mode)
        try:
            data = json.loads(path.read_text())
            tracker = cls(
                mode=mode,
                concept_counts=data.get("concept_counts", {}),
                _level=Level(data.get("level", "beginner")),
            )
            return tracker
        except (json.JSONDecodeError, ValueError, KeyError):
            return cls(mode=mode)
