"""Metrics tracker — iteration counts, tool usage, violations."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class Metrics:
    iterations: int = 0
    tool_calls: dict[str, int] = field(default_factory=dict)
    violations: int = 0
    retries: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    concepts_taught: list[str] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    def record_tool_call(self, tool_name: str) -> None:
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1

    def record_violation(self) -> None:
        self.violations += 1

    def record_retry(self) -> None:
        self.retries += 1

    def increment_iteration(self) -> None:
        self.iterations += 1

    def elapsed_seconds(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> dict:
        return {
            "iterations": self.iterations,
            "tool_calls": dict(self.tool_calls),
            "violations": self.violations,
            "retries": self.retries,
            "concepts_taught": list(set(self.concepts_taught)),
            "elapsed_s": round(self.elapsed_seconds(), 2),
        }

    def display(self) -> str:
        lines = [
            f"Iterations : {self.iterations}",
            f"Violations : {self.violations}",
            f"Retries    : {self.retries}",
            f"Concepts   : {len(set(self.concepts_taught))} unique",
            f"Elapsed    : {self.elapsed_seconds():.1f}s",
            f"Tool calls :",
        ]
        for name, count in sorted(self.tool_calls.items()):
            lines.append(f"  {name}: {count}")
        return "\n".join(lines)
