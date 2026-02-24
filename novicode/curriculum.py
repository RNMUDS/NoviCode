"""Curriculum definitions — levels, concept catalogs, and educational prompts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import FrozenSet

from novicode.config import Mode


# ── Levels ─────────────────────────────────────────────────────────

class Level(str, Enum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


LEVEL_ORDER: list[Level] = [Level.BEGINNER, Level.INTERMEDIATE, Level.ADVANCED]

LEVEL_UP_THRESHOLD = 0.6  # 60% of concepts mastered → next level


# ── Concept catalogs per mode × level ──────────────────────────────

@dataclass(frozen=True)
class ConceptCatalog:
    beginner: FrozenSet[str]
    intermediate: FrozenSet[str]
    advanced: FrozenSet[str]

    def for_level(self, level: Level) -> FrozenSet[str]:
        return getattr(self, level.value)

    def all_concepts(self) -> FrozenSet[str]:
        return self.beginner | self.intermediate | self.advanced


CONCEPT_CATALOGS: dict[Mode, ConceptCatalog] = {
    Mode.PYTHON_BASIC: ConceptCatalog(
        beginner=frozenset({
            "変数", "型", "print", "条件分岐", "ループ", "関数", "リスト",
        }),
        intermediate=frozenset({
            "クラス", "例外処理", "ファイル操作", "リスト内包表記",
            "テストの書き方", "モジュール",
        }),
        advanced=frozenset({
            "デコレータ", "ジェネレータ", "設計パターン",
            "セキュリティ", "パフォーマンス",
        }),
    ),
    Mode.PY5: ConceptCatalog(
        beginner=frozenset({
            "setup関数", "draw関数", "座標系", "図形描画", "色指定", "変数", "ループ",
        }),
        intermediate=frozenset({
            "アニメーション", "マウス操作", "キーボード操作",
            "関数分割", "配列で管理", "条件分岐",
        }),
        advanced=frozenset({
            "クラスで管理", "パーティクル", "ベクトル演算",
            "3D描画", "シェーダー",
        }),
    ),
    Mode.SKLEARN: ConceptCatalog(
        beginner=frozenset({
            "numpy配列", "データ読み込み", "散布図", "平均と分散",
            "train_test_split", "線形回帰", "精度評価",
        }),
        intermediate=frozenset({
            "前処理", "標準化", "分類", "決定木",
            "交差検証", "混同行列",
        }),
        advanced=frozenset({
            "パイプライン", "ハイパーパラメータ調整",
            "アンサンブル", "次元削減", "クラスタリング",
        }),
    ),
    Mode.PANDAS: ConceptCatalog(
        beginner=frozenset({
            "DataFrame作成", "CSV読み込み", "列選択", "行フィルタ",
            "基本統計量", "棒グラフ", "折れ線グラフ",
        }),
        intermediate=frozenset({
            "groupby", "結合", "欠損値処理",
            "ピボットテーブル", "複数グラフ", "日時処理",
        }),
        advanced=frozenset({
            "メソッドチェーン", "apply関数", "大規模データ",
            "可視化カスタマイズ", "データパイプライン",
        }),
    ),
    Mode.AFRAME: ConceptCatalog(
        beginner=frozenset({
            "シーン構成", "エンティティ", "ジオメトリ",
            "マテリアル", "ポジション", "回転", "スケール",
        }),
        intermediate=frozenset({
            "アニメーション", "イベント", "カスタムコンポーネント",
            "テクスチャ", "ライティング", "カメラ設定",
        }),
        advanced=frozenset({
            "物理エンジン", "VR操作", "パフォーマンス最適化",
            "外部モデル読み込み", "シェーダー",
        }),
    ),
    Mode.THREEJS: ConceptCatalog(
        beginner=frozenset({
            "シーン", "カメラ", "レンダラー", "メッシュ",
            "ジオメトリ", "マテリアル", "アニメーションループ",
        }),
        intermediate=frozenset({
            "ライト", "テクスチャ", "OrbitControls",
            "グループ", "レイキャスト", "イベント処理",
        }),
        advanced=frozenset({
            "シェーダー", "ポストプロセッシング", "物理演算",
            "LOD", "パフォーマンス最適化",
        }),
    ),
}


# ── Concept extraction from LLM response ───────────────────────────

# Mapping: Japanese concept name → regex patterns to detect it in text
_CONCEPT_PATTERNS: dict[str, list[str]] = {
    # Python basic - beginner
    "変数": [r"変数", r"\bvariable\b"],
    "型": [r"型\b", r"\btype\b", r"\bint\b", r"\bstr\b", r"\bfloat\b", r"\bbool\b"],
    "print": [r"\bprint\s*\(", r"print関数"],
    "条件分岐": [r"条件分岐", r"\bif\b.*:", r"\belse\b", r"\belif\b"],
    "ループ": [r"ループ", r"\bfor\b.*:", r"\bwhile\b.*:"],
    "関数": [r"関数", r"\bdef\s+\w+"],
    "リスト": [r"リスト", r"\blist\b"],
    # Python basic - intermediate
    "クラス": [r"クラス", r"\bclass\s+\w+"],
    "例外処理": [r"例外処理", r"\btry\b.*:", r"\bexcept\b"],
    "ファイル操作": [r"ファイル操作", r"\bopen\s*\(", r"ファイル"],
    "リスト内包表記": [r"リスト内包", r"内包表記", r"\[.*\bfor\b.*\bin\b.*\]"],
    "テストの書き方": [r"テスト", r"\btest_\w+", r"\bassert\b"],
    "モジュール": [r"モジュール", r"\bimport\s+\w+"],
    # Python basic - advanced
    "デコレータ": [r"デコレータ", r"@\w+"],
    "ジェネレータ": [r"ジェネレータ", r"\byield\b"],
    "設計パターン": [r"設計パターン", r"デザインパターン"],
    "セキュリティ": [r"セキュリティ", r"安全"],
    "パフォーマンス": [r"パフォーマンス", r"高速化", r"最適化"],
    # py5
    "setup関数": [r"\bsetup\s*\(\)", r"setup関数"],
    "draw関数": [r"\bdraw\s*\(\)", r"draw関数"],
    "座標系": [r"座標"],
    "図形描画": [r"図形", r"\bcircle\b", r"\brect\b", r"\bellipse\b"],
    "色指定": [r"色", r"\bfill\b", r"\bstroke\b", r"\bbackground\b"],
    "アニメーション": [r"アニメーション", r"animate"],
    "マウス操作": [r"マウス", r"\bmouse_"],
    "キーボード操作": [r"キーボード", r"\bkey_"],
    "関数分割": [r"関数分割", r"関数に分け"],
    "配列で管理": [r"配列", r"リストで管理"],
    "パーティクル": [r"パーティクル", r"particle"],
    "ベクトル演算": [r"ベクトル", r"vector"],
    "3D描画": [r"3D", r"三次元"],
    "シェーダー": [r"シェーダー", r"shader"],
    # sklearn
    "numpy配列": [r"numpy", r"\bnp\.\b"],
    "データ読み込み": [r"データ読み込み", r"load_"],
    "散布図": [r"散布図", r"scatter"],
    "平均と分散": [r"平均", r"分散", r"\bmean\b", r"\bvar\b"],
    "train_test_split": [r"train_test_split"],
    "線形回帰": [r"線形回帰", r"LinearRegression"],
    "精度評価": [r"精度", r"\bscore\b", r"\baccuracy\b"],
    "前処理": [r"前処理", r"preprocessing"],
    "標準化": [r"標準化", r"StandardScaler"],
    "分類": [r"分類", r"classification"],
    "決定木": [r"決定木", r"DecisionTree"],
    "交差検証": [r"交差検証", r"cross_val"],
    "混同行列": [r"混同行列", r"confusion_matrix"],
    "パイプライン": [r"パイプライン", r"Pipeline"],
    "ハイパーパラメータ調整": [r"ハイパーパラメータ", r"GridSearch"],
    "アンサンブル": [r"アンサンブル", r"ensemble"],
    "次元削減": [r"次元削減", r"PCA"],
    "クラスタリング": [r"クラスタリング", r"KMeans"],
    # pandas
    "DataFrame作成": [r"DataFrame", r"データフレーム"],
    "CSV読み込み": [r"CSV", r"read_csv"],
    "列選択": [r"列選択", r"列を選"],
    "行フィルタ": [r"フィルタ", r"行を絞"],
    "基本統計量": [r"統計量", r"\bdescribe\b"],
    "棒グラフ": [r"棒グラフ", r"\bbar\b"],
    "折れ線グラフ": [r"折れ線", r"\bplot\b"],
    "groupby": [r"\bgroupby\b"],
    "結合": [r"結合", r"\bmerge\b", r"\bjoin\b"],
    "欠損値処理": [r"欠損値", r"\bdropna\b", r"\bfillna\b"],
    "ピボットテーブル": [r"ピボット", r"pivot_table"],
    "複数グラフ": [r"複数グラフ", r"subplot"],
    "日時処理": [r"日時", r"datetime", r"to_datetime"],
    "メソッドチェーン": [r"メソッドチェーン", r"チェーン"],
    "apply関数": [r"\bapply\b"],
    "大規模データ": [r"大規模", r"chunk"],
    "可視化カスタマイズ": [r"カスタマイズ", r"凡例", r"軸ラベル"],
    "データパイプライン": [r"データパイプライン"],
    # aframe
    "シーン構成": [r"シーン", r"a-scene"],
    "エンティティ": [r"エンティティ", r"a-entity"],
    "ジオメトリ": [r"ジオメトリ", r"geometry"],
    "マテリアル": [r"マテリアル", r"material"],
    "ポジション": [r"ポジション", r"position"],
    "回転": [r"回転", r"rotation"],
    "スケール": [r"スケール", r"scale"],
    "イベント": [r"イベント", r"event"],
    "カスタムコンポーネント": [r"カスタムコンポーネント", r"registerComponent"],
    "テクスチャ": [r"テクスチャ", r"texture"],
    "ライティング": [r"ライティング", r"ライト", r"light"],
    "カメラ設定": [r"カメラ", r"camera"],
    "物理エンジン": [r"物理", r"physics"],
    "VR操作": [r"VR", r"コントローラ"],
    "パフォーマンス最適化": [r"パフォーマンス最適化", r"最適化"],
    "外部モデル読み込み": [r"外部モデル", r"gltf", r"glb"],
    # threejs
    "シーン": [r"シーン", r"\bScene\b"],
    "カメラ": [r"カメラ", r"\bCamera\b"],
    "レンダラー": [r"レンダラー", r"\bRenderer\b"],
    "メッシュ": [r"メッシュ", r"\bMesh\b"],
    "アニメーションループ": [r"アニメーションループ", r"requestAnimationFrame"],
    "ライト": [r"ライト", r"\bLight\b"],
    "OrbitControls": [r"OrbitControls"],
    "グループ": [r"グループ", r"\bGroup\b"],
    "レイキャスト": [r"レイキャスト", r"Raycaster"],
    "イベント処理": [r"イベント処理", r"addEventListener"],
    "ポストプロセッシング": [r"ポストプロセッシング", r"postprocessing"],
    "物理演算": [r"物理演算", r"physics"],
    "LOD": [r"\bLOD\b"],
}


def extract_concepts(text: str, mode: Mode) -> list[str]:
    """Extract concepts mentioned in LLM response text for the given mode."""
    catalog = CONCEPT_CATALOGS.get(mode)
    if catalog is None:
        return []

    all_concepts = catalog.all_concepts()
    found: list[str] = []

    for concept in all_concepts:
        patterns = _CONCEPT_PATTERNS.get(concept, [])
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                found.append(concept)
                break

    return found


# ── Level auto-judgment ─────────────────────────────────────────────

def judge_level(mode: Mode, mastered: set[str]) -> Level:
    """Determine level based on mastered concepts.

    If 60%+ of current level's concepts are mastered, promote to next level.
    """
    catalog = CONCEPT_CATALOGS.get(mode)
    if catalog is None:
        return Level.BEGINNER

    for i, level in enumerate(LEVEL_ORDER):
        level_concepts = catalog.for_level(level)
        if not level_concepts:
            continue
        mastered_count = len(mastered & level_concepts)
        ratio = mastered_count / len(level_concepts)
        if ratio < LEVEL_UP_THRESHOLD:
            return level

    return Level.ADVANCED


# ── Educational system prompts ──────────────────────────────────────

_BEGINNER_PROMPT_TEMPLATE = """\
あなたはプログラミングの先生です。初心者に{domain}を教えています。

【絶対ルール：1回の返答では「1ステップだけ」】
- 1回の返答に書くコードは最大10行。それ以上は絶対に書かない。
- 1回の返答で教える概念は1つだけ。複数の概念を同時に説明しない。
- コードを出したら「動かしてみてください」で終わる。その先は書かない。
- ユーザーが「次」「続き」「OK」などと言うまで、次のステップに進まない。

【返答の形式】
1. 今やることを1〜2文で説明する
2. コードを書く（最大10行）。コードは必ず ```言語名 と ``` で囲む（例: ```python ... ```）
3. コードの中の新しい部分だけを短く説明する（箇条書き2〜3個）
4. 「動かしてみてください」で終わる

【やってはいけないこと】
- 完成品を一度に出すこと
- 先のステップを予告すること（「次は〇〇をします」は不要）
- 長い解説を書くこと

【説明のルール】
- 新しい概念が出たら名前を教える（「これは〇〇と言います」）
- 説明は短く。1つの概念につき1〜2文。

【理解の確認】
- 「関数」「変数」「ループ」など、プログラミング用語を初めて使うときは
  「〇〇については分かりますか？」と聞いてから説明する。
- ユーザーが「分かる」と答えたら説明をスキップし、「分からない」と答えたら簡潔に説明する。
- 以下の概念はユーザーが既に理解済みなので確認不要：{mastered_concepts}

【今のレベル】初級 — {beginner_topics}を学んでいます
"""

_INTERMEDIATE_ADDITION = """\

【中級の追加ルール】
- エラーハンドリングの必要性を1〜2文で伝える
- テストの書き方を促す（「テストも書いてみましょう」程度）

【今のレベル】中級 — {intermediate_topics}を学んでいます
"""

_ADVANCED_ADDITION = """\

【上級の追加ルール】
- 設計やパフォーマンスのポイントを1〜2文で添える
- セキュリティの注意点があれば短く伝える

【今のレベル】上級 — {advanced_topics}を学んでいます
"""


_MODE_DOMAINS: dict[Mode, str] = {
    Mode.PYTHON_BASIC: "Python",
    Mode.PY5: "Py5（Processing for Python）を使ったクリエイティブコーディング",
    Mode.SKLEARN: "scikit-learn を使った機械学習",
    Mode.PANDAS: "pandas / matplotlib / seaborn を使ったデータ分析",
    Mode.AFRAME: "A-Frame を使った WebXR / 3D",
    Mode.THREEJS: "Three.js を使った 3D グラフィックス",
}


def build_education_prompt(
    mode: Mode,
    level: Level,
    mastered_concepts: set[str] | None = None,
) -> str:
    """Build the educational system prompt for a given mode and level."""
    catalog = CONCEPT_CATALOGS.get(mode)
    if catalog is None:
        return ""

    domain = _MODE_DOMAINS.get(mode, str(mode.value))
    beginner_topics = "、".join(sorted(catalog.beginner))

    mastered_str = "（なし）"
    if mastered_concepts:
        mastered_str = "、".join(sorted(mastered_concepts))

    prompt = _BEGINNER_PROMPT_TEMPLATE.format(
        domain=domain,
        beginner_topics=beginner_topics,
        mastered_concepts=mastered_str,
    )

    if level in (Level.INTERMEDIATE, Level.ADVANCED):
        intermediate_topics = "、".join(sorted(catalog.intermediate))
        prompt += _INTERMEDIATE_ADDITION.format(
            intermediate_topics=intermediate_topics,
        )

    if level == Level.ADVANCED:
        advanced_topics = "、".join(sorted(catalog.advanced))
        prompt += _ADVANCED_ADDITION.format(
            advanced_topics=advanced_topics,
        )

    return prompt
