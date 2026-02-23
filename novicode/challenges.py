"""Challenge problems — practice exercises per mode and level."""

from __future__ import annotations

import random
from dataclasses import dataclass

from novicode.config import Mode
from novicode.curriculum import Level


@dataclass(frozen=True)
class Challenge:
    id: str
    mode: Mode
    level: Level
    title: str
    description: str
    hint: str


# ── Challenge definitions (6 per mode × 6 modes = 36) ─────────────

CHALLENGES: list[Challenge] = [
    # ── python_basic ──────────────────────────────────────────
    Challenge(
        id="py_b1", mode=Mode.PYTHON_BASIC, level=Level.BEGINNER,
        title="数当てゲーム",
        description=(
            "1〜100のランダムな数を生成し、ユーザーに当てさせるゲームを作ってください。\n"
            "大きい・小さいのヒントを出し、正解したら回数を表示します。"
        ),
        hint="random.randint() でランダムな数を作り、while ループで繰り返し入力を受け付けます。",
    ),
    Challenge(
        id="py_b2", mode=Mode.PYTHON_BASIC, level=Level.BEGINNER,
        title="FizzBuzz",
        description=(
            "1〜100の数字を表示し、3の倍数は Fizz、5の倍数は Buzz、\n"
            "両方の倍数は FizzBuzz と表示するプログラムを作ってください。"
        ),
        hint="for ループと if/elif/else を使います。% 演算子で余りを求められます。",
    ),
    Challenge(
        id="py_i1", mode=Mode.PYTHON_BASIC, level=Level.INTERMEDIATE,
        title="TODOリスト（クラス版）",
        description=(
            "TODOリストをクラスで実装してください。\n"
            "追加・完了・一覧表示・ファイル保存/読み込み機能を含めます。"
        ),
        hint="Todo クラスと TodoList クラスを作ります。json モジュールでファイル保存できます。",
    ),
    Challenge(
        id="py_i2", mode=Mode.PYTHON_BASIC, level=Level.INTERMEDIATE,
        title="テスト付き電卓",
        description=(
            "四則演算の関数を作り、それぞれにテストを書いてください。\n"
            "ゼロ除算のエラーハンドリングも含めます。"
        ),
        hint="各演算を関数にし、assert 文でテストします。ZeroDivisionError を try/except で処理します。",
    ),
    Challenge(
        id="py_a1", mode=Mode.PYTHON_BASIC, level=Level.ADVANCED,
        title="デコレータでログ出力",
        description=(
            "関数の呼び出しと実行時間を自動的にログ出力するデコレータを作ってください。\n"
            "デコレータを3つ以上の関数に適用して動作確認します。"
        ),
        hint="functools.wraps を使い、time.time() で実行時間を計測します。",
    ),
    Challenge(
        id="py_a2", mode=Mode.PYTHON_BASIC, level=Level.ADVANCED,
        title="ジェネレータでファイル処理",
        description=(
            "大きなCSVファイルを1行ずつ読み込むジェネレータを作ってください。\n"
            "メモリを大量消費せずにデータを処理するパイプラインを構築します。"
        ),
        hint="yield でデータを返すジェネレータ関数を作り、パイプライン状に繋げます。",
    ),

    # ── py5 ───────────────────────────────────────────────────
    Challenge(
        id="p5_b1", mode=Mode.PY5, level=Level.BEGINNER,
        title="カラフルな円の描画",
        description=(
            "画面にランダムな位置・色・サイズの円を10個描いてください。\n"
            "setup() と size() を使います。"
        ),
        hint="random.randint() で位置と色を決め、circle() で描画します。",
    ),
    Challenge(
        id="p5_b2", mode=Mode.PY5, level=Level.BEGINNER,
        title="マウス追従する図形",
        description=(
            "マウスの位置に図形が追従するスケッチを作ってください。\n"
            "draw() ループの基本を学びます。"
        ),
        hint="draw() 内で mouse_x, mouse_y を使って図形を描画します。",
    ),
    Challenge(
        id="p5_i1", mode=Mode.PY5, level=Level.INTERMEDIATE,
        title="バウンドするボール",
        description=(
            "画面の端で跳ね返るボールをアニメーションしてください。\n"
            "速度の変数と条件分岐を使います。"
        ),
        hint="x, y 座標と vx, vy 速度の変数を使い、壁に当たったら速度を反転させます。",
    ),
    Challenge(
        id="p5_i2", mode=Mode.PY5, level=Level.INTERMEDIATE,
        title="キーボードで動くキャラクター",
        description=(
            "矢印キーで画面上のキャラクター（四角形）を動かせるようにしてください。\n"
            "キーボードイベントの処理を学びます。"
        ),
        hint="key_pressed() をチェックし、キーに応じて座標を更新します。",
    ),
    Challenge(
        id="p5_a1", mode=Mode.PY5, level=Level.ADVANCED,
        title="パーティクルシステム",
        description=(
            "Particle クラスを作り、マウスクリックで粒子が飛び散るエフェクトを作ってください。\n"
            "寿命・重力・フェードアウトを含めます。"
        ),
        hint="Particle クラスに位置・速度・寿命を持たせ、リストで管理します。",
    ),
    Challenge(
        id="p5_a2", mode=Mode.PY5, level=Level.ADVANCED,
        title="フラクタル木",
        description=(
            "再帰関数を使ってフラクタルの木を描画してください。\n"
            "枝の角度と長さを再帰ごとに変化させます。"
        ),
        hint="再帰関数で translate/rotate を使い、枝の深さごとに短くしていきます。",
    ),

    # ── sklearn ───────────────────────────────────────────────
    Challenge(
        id="sk_b1", mode=Mode.SKLEARN, level=Level.BEGINNER,
        title="アイリス分類入門",
        description=(
            "sklearn のアイリスデータセットを読み込み、\n"
            "データの基本統計量と散布図を表示してください。"
        ),
        hint="load_iris() でデータを読み込み、numpy で統計量を計算します。",
    ),
    Challenge(
        id="sk_b2", mode=Mode.SKLEARN, level=Level.BEGINNER,
        title="線形回帰で予測",
        description=(
            "簡単なデータセットに線形回帰モデルを適用し、\n"
            "予測結果と実測値を比較してください。"
        ),
        hint="LinearRegression の fit() と predict() を使います。train_test_split でデータを分割します。",
    ),
    Challenge(
        id="sk_i1", mode=Mode.SKLEARN, level=Level.INTERMEDIATE,
        title="決定木で分類＋評価",
        description=(
            "決定木分類器でアイリスを分類し、\n"
            "交差検証で精度を評価してください。混同行列も表示します。"
        ),
        hint="DecisionTreeClassifier と cross_val_score を使い、confusion_matrix で評価します。",
    ),
    Challenge(
        id="sk_i2", mode=Mode.SKLEARN, level=Level.INTERMEDIATE,
        title="前処理パイプライン",
        description=(
            "欠損値の補完→標準化→モデル学習を Pipeline で構築してください。"
        ),
        hint="Pipeline([('scaler', StandardScaler()), ('clf', ...)]) のように繋げます。",
    ),
    Challenge(
        id="sk_a1", mode=Mode.SKLEARN, level=Level.ADVANCED,
        title="GridSearch でチューニング",
        description=(
            "GridSearchCV を使ってハイパーパラメータを最適化し、\n"
            "最良のモデルの性能を報告してください。"
        ),
        hint="param_grid 辞書を定義し、GridSearchCV の best_params_ を確認します。",
    ),
    Challenge(
        id="sk_a2", mode=Mode.SKLEARN, level=Level.ADVANCED,
        title="PCA + クラスタリング",
        description=(
            "PCA で次元削減した後、KMeans でクラスタリングし、\n"
            "結果を2D散布図で可視化してください。"
        ),
        hint="PCA(n_components=2) で次元削減し、KMeans で分類、scatter で色分け表示します。",
    ),

    # ── pandas ────────────────────────────────────────────────
    Challenge(
        id="pd_b1", mode=Mode.PANDAS, level=Level.BEGINNER,
        title="売上データ分析入門",
        description=(
            "商品名・個数・単価のデータを DataFrame で作成し、\n"
            "売上合計を計算して棒グラフで表示してください。"
        ),
        hint="pd.DataFrame({'商品': [...], '個数': [...], '単価': [...]}) でデータを作ります。",
    ),
    Challenge(
        id="pd_b2", mode=Mode.PANDAS, level=Level.BEGINNER,
        title="CSVを読み込んで集計",
        description=(
            "CSVファイルを読み込み、基本統計量を表示し、\n"
            "特定の条件でフィルタリングしてください。"
        ),
        hint="pd.read_csv() で読み込み、describe() で統計量、df[df['列'] > 値] でフィルタします。",
    ),
    Challenge(
        id="pd_i1", mode=Mode.PANDAS, level=Level.INTERMEDIATE,
        title="GroupBy で集計レポート",
        description=(
            "カテゴリ別に groupby で集計し、結果を複数のグラフで\n"
            "並べて表示するレポートを作ってください。"
        ),
        hint="df.groupby('カテゴリ').sum() で集計し、subplot で複数グラフを並べます。",
    ),
    Challenge(
        id="pd_i2", mode=Mode.PANDAS, level=Level.INTERMEDIATE,
        title="欠損値処理と結合",
        description=(
            "欠損値を含むデータの処理方法を3パターン試し、\n"
            "2つのデータフレームを結合してください。"
        ),
        hint="dropna(), fillna(0), fillna(mean()) の3パターンを比較します。merge() で結合します。",
    ),
    Challenge(
        id="pd_a1", mode=Mode.PANDAS, level=Level.ADVANCED,
        title="メソッドチェーンでデータ加工",
        description=(
            "メソッドチェーンを使って読み込み→フィルタ→変換→集計を\n"
            "1つの式で書いてください。"
        ),
        hint="df.query().assign().groupby().agg() のようにチェーンで繋げます。",
    ),
    Challenge(
        id="pd_a2", mode=Mode.PANDAS, level=Level.ADVANCED,
        title="大規模CSVのチャンク処理",
        description=(
            "chunksize を指定して大きなCSVを分割読み込みし、\n"
            "メモリ効率の良い集計を行ってください。"
        ),
        hint="pd.read_csv('file.csv', chunksize=1000) でイテレータとして読み込みます。",
    ),

    # ── aframe ────────────────────────────────────────────────
    Challenge(
        id="af_b1", mode=Mode.AFRAME, level=Level.BEGINNER,
        title="基本3Dシーン",
        description=(
            "A-Frame で箱・球・平面を配置した基本シーンを作ってください。\n"
            "色と位置を設定します。"
        ),
        hint="a-box, a-sphere, a-plane エンティティを使い、position と color を設定します。",
    ),
    Challenge(
        id="af_b2", mode=Mode.AFRAME, level=Level.BEGINNER,
        title="スカイボックスとテキスト",
        description=(
            "a-sky で背景色を設定し、a-text で3D空間にテキストを表示してください。"
        ),
        hint="<a-sky color='#87CEEB'> で空色の背景、<a-text value='Hello'> でテキストを表示します。",
    ),
    Challenge(
        id="af_i1", mode=Mode.AFRAME, level=Level.INTERMEDIATE,
        title="アニメーション付きオブジェクト",
        description=(
            "a-animation で回転し続けるオブジェクトと、\n"
            "クリックで色が変わるオブジェクトを作ってください。"
        ),
        hint="animation 属性で rotation をループさせ、cursor と click イベントで色を変更します。",
    ),
    Challenge(
        id="af_i2", mode=Mode.AFRAME, level=Level.INTERMEDIATE,
        title="カスタムコンポーネント",
        description=(
            "AFRAME.registerComponent で独自コンポーネントを作り、\n"
            "tick 関数で毎フレーム動くオブジェクトを作ってください。"
        ),
        hint="registerComponent('my-comp', { tick: function() { ... } }) で定義します。",
    ),
    Challenge(
        id="af_a1", mode=Mode.AFRAME, level=Level.ADVANCED,
        title="VRインタラクション",
        description=(
            "VRコントローラーで3Dオブジェクトを掴んで動かせるシーンを作ってください。"
        ),
        hint="hand-controls コンポーネントと gripdown/gripup イベントを使います。",
    ),
    Challenge(
        id="af_a2", mode=Mode.AFRAME, level=Level.ADVANCED,
        title="外部3Dモデル表示",
        description=(
            "glTF 形式の3Dモデルを読み込んで表示し、\n"
            "ライティングとカメラを調整してください。"
        ),
        hint="a-gltf-model と a-asset-item を使い、a-light でライティングを設定します。",
    ),

    # ── threejs ───────────────────────────────────────────────
    Challenge(
        id="tj_b1", mode=Mode.THREEJS, level=Level.BEGINNER,
        title="回転する立方体",
        description=(
            "Three.js で立方体を表示し、アニメーションループで回転させてください。"
        ),
        hint="BoxGeometry + MeshBasicMaterial + Mesh を作り、animate 関数で rotation を更新します。",
    ),
    Challenge(
        id="tj_b2", mode=Mode.THREEJS, level=Level.BEGINNER,
        title="複数のジオメトリ",
        description=(
            "立方体・球・円柱を並べて表示し、それぞれ違う色をつけてください。"
        ),
        hint="BoxGeometry, SphereGeometry, CylinderGeometry を使い、position で配置します。",
    ),
    Challenge(
        id="tj_i1", mode=Mode.THREEJS, level=Level.INTERMEDIATE,
        title="ライティングとテクスチャ",
        description=(
            "DirectionalLight と AmbientLight を配置し、\n"
            "MeshStandardMaterial でリアルな質感を表現してください。"
        ),
        hint="MeshStandardMaterial を使い、DirectionalLight で影を付けます。",
    ),
    Challenge(
        id="tj_i2", mode=Mode.THREEJS, level=Level.INTERMEDIATE,
        title="OrbitControls でカメラ操作",
        description=(
            "OrbitControls を導入し、マウスでカメラを自由に回転・ズームできるようにしてください。"
        ),
        hint="OrbitControls(camera, renderer.domElement) を作り、animate 内で update() を呼びます。",
    ),
    Challenge(
        id="tj_a1", mode=Mode.THREEJS, level=Level.ADVANCED,
        title="レイキャストでクリック検出",
        description=(
            "Raycaster を使ってマウスクリックしたオブジェクトを検出し、\n"
            "色を変更するインタラクションを作ってください。"
        ),
        hint="Raycaster + mouse イベントで intersectObjects() を呼び、当たったメッシュの色を変更します。",
    ),
    Challenge(
        id="tj_a2", mode=Mode.THREEJS, level=Level.ADVANCED,
        title="パフォーマンス最適化シーン",
        description=(
            "1000個のオブジェクトを表示するシーンを作り、\n"
            "InstancedMesh を使ってパフォーマンスを最適化してください。"
        ),
        hint="InstancedMesh で同じジオメトリの大量描画を効率化し、setMatrixAt で配置します。",
    ),
]


def get_challenges(mode: Mode, level: Level) -> list[Challenge]:
    """Get challenges for a specific mode and level."""
    return [c for c in CHALLENGES if c.mode == mode and c.level == level]


def get_random_challenge(mode: Mode, level: Level) -> Challenge | None:
    """Get a random challenge for the current mode and level."""
    candidates = get_challenges(mode, level)
    if not candidates:
        return None
    return random.choice(candidates)


def get_challenge_by_id(challenge_id: str) -> Challenge | None:
    """Look up a challenge by its ID."""
    for c in CHALLENGES:
        if c.id == challenge_id:
            return c
    return None


def format_challenge(challenge: Challenge) -> str:
    """Format a challenge for display."""
    level_ja = {"beginner": "初級", "intermediate": "中級", "advanced": "上級"}
    return (
        f"🎯 【チャレンジ: {challenge.title}】\n"
        f"レベル: {level_ja.get(challenge.level.value, challenge.level.value)}\n"
        f"\n{challenge.description}\n"
        f"\n💡 ヒントが必要なら /hint と入力してください。"
    )


def format_hint(challenge: Challenge) -> str:
    """Format a challenge hint for display."""
    return f"💡 【ヒント: {challenge.title}】\n{challenge.hint}"
