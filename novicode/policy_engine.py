"""Policy engine — enforces mode-specific rules on tool calls and content."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from novicode.config import (
    SCOPE_DESCRIPTION,
    ModeProfile,
    LanguageFamily,
)
from novicode.curriculum import Level, build_education_prompt


@dataclass(frozen=True)
class PolicyVerdict:
    allowed: bool
    reason: str = ""


class PolicyEngine:
    """Evaluates requests and tool calls against the active mode profile."""

    def __init__(self, profile: ModeProfile, level: Level = Level.BEGINNER) -> None:
        self.profile = profile
        self.level = level
        self.mastered_concepts: set[str] = set()

    def check_tool_allowed(self, tool_name: str) -> PolicyVerdict:
        """Is this tool permitted in the current mode?"""
        if tool_name not in self.profile.allowed_tools:
            return PolicyVerdict(
                allowed=False,
                reason=(
                    f"Tool '{tool_name}' is not allowed in "
                    f"mode '{self.profile.mode.value}'. "
                    f"Allowed: {', '.join(sorted(self.profile.allowed_tools))}"
                ),
            )
        return PolicyVerdict(allowed=True)

    def check_file_extension(self, filename: str) -> PolicyVerdict:
        """Is this file extension permitted for the current language family?"""
        ext = _get_extension(filename)
        if ext and ext not in self.profile.allowed_extensions:
            return PolicyVerdict(
                allowed=False,
                reason=(
                    f"Extension '{ext}' is forbidden in mode "
                    f"'{self.profile.mode.value}'. "
                    f"Allowed: {', '.join(sorted(self.profile.allowed_extensions))}"
                ),
            )
        return PolicyVerdict(allowed=True)

    def check_scope(self, user_message: str) -> PolicyVerdict:
        """Basic keyword heuristic to reject clearly out-of-scope requests."""
        lowered = user_message.lower()
        out_of_scope_keywords = [
            "rust", "golang", "kotlin", "swift", "c++",
            "c#", "ruby", "php", "perl", "scala", "haskell",
            "elixir", "dart", "flutter", "react native",
            "terraform", "kubernetes", "docker", "ansible",
            "blockchain", "solidity", "web3",
        ]
        # "java" needs special handling: must not match "javascript"
        import re
        if re.search(r"java(?!script)", lowered):
            return PolicyVerdict(
                allowed=False,
                reason=SCOPE_DESCRIPTION,
            )
        for kw in out_of_scope_keywords:
            if kw in lowered:
                return PolicyVerdict(
                    allowed=False,
                    reason=SCOPE_DESCRIPTION,
                )
        return PolicyVerdict(allowed=True)

    def build_system_prompt(self) -> str:
        """Return the full system prompt for the active mode, including education."""
        base = self.profile.system_prompt
        constraint = (
            "\n\n【制約】\n"
            "- このモードで許可された言語・ライブラリだけを使う。\n"
            "- 1回の返答のコードは最大10行。短く保つ。\n"
            "- ネットワーク通信・パッケージ追加は禁止。\n"
        )

        education = build_education_prompt(
            self.profile.mode, self.level, self.mastered_concepts,
        )
        if education:
            return education + "\n\n" + base + constraint
        return base + constraint


def _get_extension(filename: str) -> str:
    """Extract file extension including the dot."""
    dot = filename.rfind(".")
    if dot == -1:
        return ""
    return filename[dot:]
