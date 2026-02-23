"""Session manager — persistence, resume, and research export."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path


SESSIONS_DIR = Path.home() / ".rnnr_coding" / "sessions"


@dataclass
class SessionMeta:
    session_id: str
    model: str
    mode: str
    created_at: float
    research: bool = False


@dataclass
class SessionEntry:
    timestamp: float
    entry_type: str  # "user", "assistant", "tool_call", "tool_result", "violation", "metric"
    data: dict = field(default_factory=dict)


@dataclass
class Session:
    meta: SessionMeta
    entries: list[SessionEntry] = field(default_factory=list)

    def add(self, entry_type: str, data: dict | None = None) -> None:
        self.entries.append(
            SessionEntry(timestamp=time.time(), entry_type=entry_type, data=data or {})
        )

    def save(self) -> Path:
        path = SESSIONS_DIR / f"{self.meta.session_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            # First line: metadata
            f.write(json.dumps({"_meta": asdict(self.meta)}) + "\n")
            for entry in self.entries:
                f.write(json.dumps(asdict(entry)) + "\n")
        return path

    def export_jsonl(self, output_path: str | None = None) -> Path:
        path = Path(output_path) if output_path else (
            SESSIONS_DIR / f"{self.meta.session_id}_export.jsonl"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(json.dumps({"_meta": asdict(self.meta)}) + "\n")
            for entry in self.entries:
                f.write(json.dumps(asdict(entry)) + "\n")
        return path


class SessionManager:
    """Creates, loads, lists, and exports sessions."""

    def __init__(self) -> None:
        SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    def create(self, model: str, mode: str, research: bool = False) -> Session:
        meta = SessionMeta(
            session_id=uuid.uuid4().hex[:12],
            model=model,
            mode=mode,
            created_at=time.time(),
            research=research,
        )
        return Session(meta=meta)

    def load(self, session_id: str) -> Session:
        path = SESSIONS_DIR / f"{session_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")

        with open(path) as f:
            lines = f.readlines()

        first = json.loads(lines[0])
        meta = SessionMeta(**first["_meta"])
        entries: list[SessionEntry] = []
        for line in lines[1:]:
            raw = json.loads(line)
            entries.append(SessionEntry(**raw))
        return Session(meta=meta, entries=entries)

    def list_sessions(self) -> list[dict]:
        result = []
        for p in sorted(SESSIONS_DIR.glob("*.jsonl")):
            if p.name.endswith("_export.jsonl"):
                continue
            try:
                with open(p) as f:
                    first = json.loads(f.readline())
                meta = first.get("_meta", {})
                result.append(meta)
            except Exception:
                continue
        return result
