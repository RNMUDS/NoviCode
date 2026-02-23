"""Security manager — blocks dangerous commands and path traversal."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from novicode.config import WORKING_DIR


@dataclass(frozen=True)
class SecurityVerdict:
    allowed: bool
    reason: str = ""
    lesson: str = ""


# Shell patterns that are always blocked
_BLOCKED_COMMANDS: list[re.Pattern] = [
    re.compile(r"\bsudo\b"),
    re.compile(r"\bchmod\b"),
    re.compile(r"\bchown\b"),
    re.compile(r"\bdd\b\s"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"/dev/"),
    re.compile(r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|--recursive)\b.*(/|\s)"),
    re.compile(r"\brm\s+-rf\s+/"),
    re.compile(r"\bcurl\b.*\|\s*\bbash\b"),
    re.compile(r"\bwget\b.*\|\s*\bbash\b"),
    re.compile(r"\bpip\s+install\b"),
    re.compile(r"\bpip3\s+install\b"),
    re.compile(r"\bnpm\s+install\b"),
    re.compile(r"\byarn\s+add\b"),
    re.compile(r"\bcurl\b"),
    re.compile(r"\bwget\b"),
    re.compile(r"\bnc\b\s"),
    re.compile(r"\bnetcat\b"),
    re.compile(r"\bssh\b"),
    re.compile(r"\bscp\b"),
    re.compile(r"\brsync\b"),
    re.compile(r"\btelnet\b"),
    re.compile(r"\bnmap\b"),
    re.compile(r"\biptables\b"),
    re.compile(r"\bsystemctl\b"),
    re.compile(r"\bservice\b"),
    re.compile(r"\bkill\b"),
    re.compile(r"\bkillall\b"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bmount\b"),
    re.compile(r"\bumount\b"),
    re.compile(r"\bfdisk\b"),
    re.compile(r"\bparted\b"),
    re.compile(r"\bdocker\b"),
    re.compile(r"\bpodman\b"),
]

# ── Security lessons — educational messages when commands are blocked ──

SECURITY_LESSONS: dict[str, str] = {
    "sudo": (
        "🔒 【セキュリティ学習】\n"
        "sudo は管理者権限でコマンドを実行します。\n"
        "実際の開発では「最小権限の原則」が重要です。\n"
        "→ プログラムには必要最小限の権限だけを与えましょう。"
    ),
    "curl": (
        "🔒 【セキュリティ学習】\n"
        "curl は外部サーバーと通信するコマンドです。\n"
        "知らないURLからデータを取得すると、マルウェアや情報漏洩のリスクがあります。\n"
        "→ 通信先は信頼できるソースだけに限定しましょう。"
    ),
    "wget": (
        "🔒 【セキュリティ学習】\n"
        "wget は外部からファイルをダウンロードするコマンドです。\n"
        "信頼できないソースからのダウンロードは危険です。\n"
        "→ ダウンロード元の信頼性を必ず確認しましょう。"
    ),
    "pip_install": (
        "🔒 【セキュリティ学習】\n"
        "pip install は外部パッケージをインストールします。\n"
        "悪意あるパッケージが混入する「サプライチェーン攻撃」のリスクがあります。\n"
        "→ 本番では requirements.txt でバージョンを固定し、信頼性を検証しましょう。"
    ),
    "npm_install": (
        "🔒 【セキュリティ学習】\n"
        "npm install は外部パッケージをインストールします。\n"
        "依存関係の脆弱性がプロジェクト全体に影響します。\n"
        "→ npm audit で脆弱性を確認し、lock ファイルでバージョンを固定しましょう。"
    ),
    "rm_rf": (
        "🔒 【セキュリティ学習】\n"
        "rm -rf は確認なしにファイルを一括削除します。取り消しできません。\n"
        "→ 削除操作は対象を限定し、バックアップを取ってから行いましょう。"
    ),
    "subprocess": (
        "🔒 【セキュリティ学習】\n"
        "subprocess はシェルコマンドをプログラムから実行します。\n"
        "ユーザー入力をそのまま渡すと「コマンドインジェクション」攻撃が可能になります。\n"
        "→ 外部コマンド実行は避け、Python標準ライブラリで代替しましょう。"
    ),
    "chmod": (
        "🔒 【セキュリティ学習】\n"
        "chmod はファイルの権限を変更します。\n"
        "777 のような広い権限設定はセキュリティリスクです。\n"
        "→ 必要最小限の権限だけを設定しましょう。"
    ),
    "ssh": (
        "🔒 【セキュリティ学習】\n"
        "ssh はリモートサーバーに接続するコマンドです。\n"
        "認証情報の管理と接続先の信頼性が重要です。\n"
        "→ 鍵認証を使い、パスワード認証は避けましょう。"
    ),
    "docker": (
        "🔒 【セキュリティ学習】\n"
        "docker はコンテナを管理するツールです。\n"
        "root 権限で動作するため、セキュリティ設定が重要です。\n"
        "→ 信頼できるイメージのみを使い、権限を制限しましょう。"
    ),
}

# Mapping: blocked pattern keyword → lesson key
_PATTERN_LESSON_MAP: dict[str, str] = {
    r"\bsudo\b": "sudo",
    r"\bcurl\b": "curl",
    r"\bwget\b": "wget",
    r"\bpip\s+install\b": "pip_install",
    r"\bpip3\s+install\b": "pip_install",
    r"\bnpm\s+install\b": "npm_install",
    r"\byarn\s+add\b": "npm_install",
    r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|--recursive)\b": "rm_rf",
    r"\brm\s+-rf\s+/": "rm_rf",
    r"\bchmod\b": "chmod",
    r"\bssh\b": "ssh",
    r"\bdocker\b": "docker",
}


# Python imports that are never allowed (network/system)
_BLOCKED_PYTHON_IMPORTS: set[str] = {
    "subprocess", "os.system", "shutil", "socket", "http", "urllib",
    "requests", "httpx", "aiohttp", "flask", "django", "fastapi",
    "paramiko", "fabric", "boto3", "botocore", "google.cloud",
    "azure", "ftplib", "smtplib", "imaplib", "poplib",
    "ctypes", "cffi", "multiprocessing",
    "webbrowser", "antigravity",
}


class SecurityManager:
    """Validates commands and file paths against security policy."""

    def __init__(self, working_dir: str | None = None) -> None:
        self.working_dir = os.path.realpath(working_dir or WORKING_DIR)

    def check_command(self, command: str) -> SecurityVerdict:
        """Check a shell command against the blocklist."""
        for pattern in _BLOCKED_COMMANDS:
            if pattern.search(command):
                lesson = _find_lesson(pattern.pattern)
                return SecurityVerdict(
                    allowed=False,
                    reason=f"Blocked command pattern: {pattern.pattern}",
                    lesson=lesson,
                )
        return SecurityVerdict(allowed=True)

    def check_path(self, path: str) -> SecurityVerdict:
        """Ensure path is within the working directory (no traversal)."""
        real = os.path.realpath(path)
        if not real.startswith(self.working_dir):
            return SecurityVerdict(
                allowed=False,
                reason=f"Path escapes working directory: {real}",
            )
        # block symlink traversal
        if os.path.islink(path):
            target = os.path.realpath(path)
            if not target.startswith(self.working_dir):
                return SecurityVerdict(
                    allowed=False,
                    reason=f"Symlink points outside working directory: {target}",
                )
        return SecurityVerdict(allowed=True)

    def check_python_imports(self, imports: set[str]) -> SecurityVerdict:
        """Check if any import is in the global blocklist."""
        blocked = imports & _BLOCKED_PYTHON_IMPORTS
        if blocked:
            lesson = ""
            if "subprocess" in blocked:
                lesson = SECURITY_LESSONS.get("subprocess", "")
            return SecurityVerdict(
                allowed=False,
                reason=f"Blocked imports: {', '.join(sorted(blocked))}",
                lesson=lesson,
            )
        return SecurityVerdict(allowed=True)


def _find_lesson(pattern_str: str) -> str:
    """Find the best matching security lesson for a blocked pattern."""
    for pat_key, lesson_key in _PATTERN_LESSON_MAP.items():
        if pat_key in pattern_str or pattern_str in pat_key:
            return SECURITY_LESSONS.get(lesson_key, "")
    return ""
