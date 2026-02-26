"""Policy engine — enforces mode-specific rules on tool calls and content."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from novicode.config import (
    SCOPE_DESCRIPTION,
    Mode,
    ModeProfile,
    LanguageFamily,
    MODE_LANGUAGE,
)
from novicode.curriculum import Level, build_education_prompt


# ── モード別の会話例 ─────────────────────────────────────────────────
# 挨拶・曖昧な入力に対して、LLM が具体的な例を提示できるようにする

_MODE_EXAMPLES: dict[Mode, list[str]] = {
    Mode.PYTHON_BASIC: [
        "「じゃんけんゲームを作って」",
        "「1から100までの合計を計算して」",
        "「九九の表を作って」",
    ],
    Mode.PY5: [
        "「カラフルな円を描いて」",
        "「虹色のアニメーションを作って」",
        "「ランダムに星を描いて」",
    ],
    Mode.SKLEARN: [
        "「アヤメのデータを分類して」",
        "「回帰分析を試して」",
        "「クラスタリングの例を見せて」",
    ],
    Mode.PANDAS: [
        "「サンプルデータで棒グラフを作って」",
        "「CSVを読み込んで分析して」",
        "「データの平均と合計を計算して」",
    ],
    Mode.WEB_BASIC: [
        "「ボタンをクリックしたら色が変わるページを作って」",
        "「自己紹介ページを作って」",
        "「カウンターアプリを作って」",
    ],
    Mode.AFRAME: [
        "「3Dの箱を表示して」",
        "「VR空間に球を並べて」",
        "「回転するオブジェクトを作って」",
    ],
    Mode.THREEJS: [
        "「回転する立方体を作って」",
        "「3Dシーンにライトを追加して」",
        "「パーティクルアニメーションを作って」",
    ],
}


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
        lang = MODE_LANGUAGE.get(self.profile.mode, LanguageFamily.PYTHON)

        # 会話ルールを最優先（プロンプト冒頭）に配置
        examples = _MODE_EXAMPLES.get(self.profile.mode, [])
        example_lines = "、".join(examples) if examples else ""
        example_hint = (
            f"\nたとえば {example_lines} のように話しかけてみてください、"
            "と具体例を挙げて案内してください。"
            if example_lines else ""
        )

        conversation_rule = (
            "【最重要ルール】ユーザーが「こんにちは」「何ができる？」「ありがとう」"
            "のように会話しているときは、普通に日本語で会話してください。"
            "コードやツールは絶対に使わないでください。"
            "コードを書くのは「○○を作って」「○○を書いて」「プログラムして」"
            "のようにユーザーがコード作成を頼んだときだけです。\n"
            "ユーザーの入力が挨拶や曖昧な質問のときは、フレンドリーに返事をしたあと、"
            "何ができるかを具体例つきで案内してください。"
            f"{example_hint}\n\n"
            "【返答スタイル】\n"
            "- 不要な謝罪（「申し訳ございません」「すみません」等）で返答を始めないこと。\n"
            "- bash 実行後は、実際の出力をそのまま伝えること。"
            "「できました」「描けました」と成功を推測しない。\n\n"
        )

        if lang == LanguageFamily.PYTHON:
            tool_rules = (
                "- コードは必ず write 関数を呼び出してファイルに保存すること。"
                "コードをテキストとして返答に含めてはいけない。\n"
                "- マークダウンのコードブロック（``` ... ```）でコードを書いてはいけない。\n"
                "- コードの実行は必ず bash 関数を呼び出して行うこと（例: bash で `python ファイル名.py`）。"
                "実行結果を推測・捏造してはいけない。\n"
                "- コードを保存したら、簡単に説明して「実行してみましょうか？」と聞く。\n"
                "- ユーザーが「はい」「うん」「お願い」「実行して」など肯定的に返答したら、"
                "即座に bash でコードを実行し、結果を表示すること。再度コードを書き直さない。\n"
            )
        else:
            tool_rules = (
                "- コードは必ず write 関数を呼び出してファイルに保存すること。"
                "コードをテキストとして返答に含めてはいけない。\n"
                "- マークダウンのコードブロック（``` ... ```）でコードを書いてはいけない。\n"
                "- Web モードでは bash は使えない。ファイル保存後「ブラウザで開いてください」と案内する。\n"
            )

        tool_section = (
            "\n\n【ツール使用ルール】\n"
            f"{tool_rules}"
            "- ツール名（write, read, bash, edit, grep, glob）を"
            "ユーザーへの返答に含めてはいけない。\n"
            "  ツールは黙って使い、ユーザーには結果だけ伝える。\n"
            "\n【ツール呼び出しフォーマット（重要）】\n"
            "コードをファイルに保存するときは、必ずこの形式で書く:\n"
            '<function=write>\n'
            '<parameter=path>ファイル名.py</parameter>\n'
            '<parameter=content>\n'
            'ここにコードを書く（普通に改行する）\n'
            '</parameter>\n'
            '</function>\n'
            "コードは \\n で1行にせず、普通に改行して複数行で書くこと。\n"
        )

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
            return conversation_rule + education + tool_section + "\n\n" + base + constraint
        return conversation_rule + base + tool_section + constraint


def _get_extension(filename: str) -> str:
    """Extract file extension including the dot."""
    dot = filename.rfind(".")
    if dot == -1:
        return ""
    return filename[dot:]
