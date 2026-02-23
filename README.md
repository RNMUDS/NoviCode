# RNNR_Coding

**プログラミング学習のための、やさしいAIコーディングアシスタント**

RNNR_Coding は、あなたのパソコンだけで動くプログラミング学習用のAIです。
インターネットに接続しなくても使えます。
[Ollama](https://ollama.ai) というソフトの上で動きます。

ふつうのAIコーディングツールと違い、**学べる範囲をあえて絞っている**のが特徴です。
「なんでもできる」のではなく、「決められた範囲を、安全に、しっかり学べる」ことを大切にしています。

---

## なぜ範囲を絞っているの？

ふつうのAIコーディングツールには、学習で使うと困る点があります。

- **言語がごちゃ混ぜになる** — 「グラフを作って」と頼むと、Python で返ってきたり JavaScript で返ってきたり。どちらで動かせばいいかわからなくなります
- **習っていないものが出てくる** — まだ知らないライブラリや難しい書き方が突然出てきて混乱します
- **記録が残らない** — 先生がどんなやりとりがあったか確認できません
- **危ないことができてしまう** — ファイルの削除やソフトのインストールなど、意図しない操作が起きることがあります

RNNR_Coding は、これらの問題をすべて防ぐように作られています。

---

## 6つの学習モード

使いたい分野を選んでからスタートします。モードごとに使える言語とライブラリが決まっています。

| モード | 何を学べる？ | 使う言語 | 使えるライブラリ |
|--------|-------------|---------|----------------|
| `python_basic` | Pythonの基礎 | Python | 標準ライブラリのみ |
| `py5` | お絵かき・アニメーション | Python | py5（Processing） |
| `sklearn` | AIと機械学習の入門 | Python | scikit-learn, numpy |
| `pandas` | データ分析とグラフ | Python | pandas, matplotlib, seaborn |
| `aframe` | ブラウザで3D・VR | HTML + JS | A-Frame |
| `threejs` | ブラウザで3Dグラフィックス | HTML + JS | Three.js |

この6つ以外のお願い（たとえば「Javaで書いて」「Dockerを使って」など）には、丁寧にお断りします。

### 言語の混在は起きません

- Python系のモード（python_basic, py5, sklearn, pandas）では、HTMLやJavaScriptは一切生成されません
- Web系のモード（aframe, threejs）では、Pythonコードは一切生成されません

AIが間違えて別の言語を出力しようとしても、チェック機能が検出して自動的にやり直します。

---

## 使えるAIモデル

RNNR_Coding が使えるモデルは2つだけです。

| モデル | 必要なメモリ | 特徴 |
|--------|------------|------|
| `qwen3:8b` | 8 GB 以上 | 軽量。メモリが少ないパソコンでもOK |
| `qwen3-coder:30b` | 32 GB 以上 | 高性能。より正確なコード生成 |

起動時にパソコンのメモリを調べて、自動的に最適なモデルを選んでくれます。

---

## はじめる前に必要なもの

RNNR_Coding を動かすには、以下の3つが必要です。
はじめての方は、上から順番にセットアップしてください。

| 必要なもの | 何をするもの？ | 確認コマンド |
|-----------|--------------|-------------|
| Python 3.10 以上 | RNNR_Coding 本体を動かす | `python3 --version` |
| Git | ソースコードをダウンロードする | `git --version` |
| Ollama | AIモデルを動かすためのソフト | `ollama --version` |

---

## ステップ 1: Python のインストール

ターミナルで以下を入力して、バージョンが表示されればOKです。

```bash
python3 --version
```

表示されない場合は、お使いのパソコンに合わせてインストールしてください。

**Mac の場合：**
```bash
# Homebrew がまだない場合は、先に Homebrew をインストール
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python をインストール
brew install python
```

**Linux (Ubuntu) の場合：**
```bash
sudo apt update && sudo apt install python3 python3-pip
```

---

## ステップ 2: Git のインストール

```bash
git --version
```

表示されない場合：

**Mac の場合：**
```bash
# 初回実行時に自動でインストールされます
xcode-select --install
```

**Linux (Ubuntu) の場合：**
```bash
sudo apt install git
```

---

## ステップ 3: Ollama のインストール

Ollama は、AIモデルをパソコンの中で動かすためのソフトです。
これがないと RNNR_Coding は動きません。

**Mac の場合：**

https://ollama.com/download/mac からダウンロードして、アプリケーションフォルダに入れてください。

または Homebrew でもインストールできます。
```bash
brew install ollama
```

**Linux の場合：**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Ollama が動いているか確認する

インストール後、ターミナルで以下を入力してください。

```bash
ollama --version
```

バージョンが表示されればインストール成功です。

次に、Ollama を起動します。

```bash
ollama serve
```

`ollama serve` を実行すると、そのターミナルは Ollama のサーバーとして使われるため、他のコマンドを入力できなくなります。
**ターミナルをもう1つ開いて**、そちらで次のステップに進んでください。

> **Mac でアプリ版を使っている場合：** Ollama.app を起動すれば自動で `ollama serve` が動きます。メニューバーにラマのアイコンが出ていればOKです。この場合、ターミナルを別に開く必要はありません。

> **「Error: ollama server not responding」と出たら？**
> Ollama がまだ起動していません。別のターミナルで `ollama serve` を実行するか、Mac の場合は Ollama.app を起動してください。

---

## ステップ 4: AIモデルのダウンロード

Ollama が起動している状態で、AIモデルをダウンロードします。
（初回は数GBのダウンロードがあるので、少し時間がかかります）

```bash
ollama pull qwen3:8b
```

> メモリが 32GB 以上あるパソコンでは、より高性能なモデルも使えます。
> ```bash
> ollama pull qwen3-coder:30b
> ```

ダウンロードが終わったら、正しく入ったか確認しましょう。

```bash
ollama list
```

`qwen3:8b` が一覧に表示されていればOKです。

---

## ステップ 5: RNNR_Coding のインストール

```bash
git clone https://github.com/RNMUDS/RNNR_Coding.git
cd RNNR_Coding
pip install -e .
```

これで準備完了です。

---

## うまくいかないときは

| 症状 | 原因と対処 |
|------|-----------|
| `ollama: command not found` | Ollama がインストールされていません。ステップ3を確認してください |
| `Error: ollama server not responding` | Ollama が起動していません。`ollama serve` を実行するか、Ollama.app を起動してください |
| `python3: command not found` | Python がインストールされていません。ステップ1を確認してください |
| `git: command not found` | Git がインストールされていません。ステップ2を確認してください |
| `pip: command not found` | `pip3 install -e .` を試してください |
| モデルのダウンロードが遅い | 初回は数GBあるので時間がかかります。Wi-Fi環境での実行をおすすめします |

---

## 使い方

### 起動する

```bash
# Pythonの基礎を学ぶ
rnnr --mode python_basic

# データ分析を学ぶ（学習ログあり）
rnnr --mode pandas --research

# 3D（A-Frame）を学ぶ
rnnr --mode aframe

# モデルを指定して起動
rnnr --mode sklearn --model qwen3-coder:30b
```

### 会話中に使えるコマンド

| コマンド | 何ができる？ |
|---------|------------|
| `/help` | コマンド一覧を表示 |
| `/exit` | 終了する |
| `/clear` | 会話履歴をクリア |
| `/metrics` | 学習の統計を表示 |
| `/trace` | 直前のAIとのやりとりを表示 |
| `/status` | 今のセッション情報を表示 |
| `/save` | セッションを保存する |

### 起動オプション一覧

| オプション | 説明 |
|-----------|------|
| `--mode` | **必須。** 学習モードを選ぶ（上の6つから1つ） |
| `--model` | 使うモデルを指定（省略すると自動選択） |
| `--safe-mode` | より安全な制限を有効にする |
| `--debug` | デバッグ情報を表示する |
| `--max-iterations` | AIの最大思考回数（初期値: 50） |
| `--research` | 研究モード（すべてのやりとりを記録） |
| `--resume セッションID` | 前回の続きから再開 |
| `--list-sessions` | 保存済みセッション一覧 |
| `--export-session セッションID` | セッションをファイルに書き出す |

---

## 安全のしくみ

RNNR_Coding は、学習者が安心して使えるようにたくさんの安全対策をしています。

### やってはいけないことを自動でブロック

- `sudo`, `rm -rf`, `chmod` などの危険なコマンド
- `curl`, `wget`, `ssh` などのネットワーク通信
- `pip install`, `npm install` などのパッケージ追加
- 作業フォルダの外へのファイル書き込み

### コードの中身もチェック

- `requests`, `subprocess` などの危険なライブラリの使用を検出
- モードで許可されていないライブラリの使用を検出
- URLやAPIへのアクセスを検出

すべてブロックされ、安全な範囲でのみ動作します。

---

## 研究モード（先生・研究者向け）

`--research` をつけて起動すると、すべてのやりとりが記録されます。

```bash
rnnr --mode python_basic --research
```

記録される内容：

- 学習者の質問（プロンプト）
- AIの回答
- バリデーション違反（言語混在など）
- ツールの使用状況
- 思考回数と時間

### データの書き出し

```bash
# セッション一覧を見る
rnnr --mode python_basic --list-sessions

# 特定のセッションを書き出す
rnnr --mode python_basic --export-session abc123def456
```

JSONL形式で出力されるので、pandas や jq で分析できます。

---

## プロジェクト構成

```
rnnr/
├── main.py              # メインの起動ファイル
├── cli.py               # コマンドライン引数の処理
├── agent_loop.py        # AIとの対話ループ
├── llm_adapter.py       # Ollamaとの通信
├── tool_registry.py     # ツールの管理と実行
├── security_manager.py  # 安全チェック（コマンド・パス）
├── policy_engine.py     # モード別のルール管理
├── validator.py         # 出力の検証（言語分離など）
├── session_manager.py   # セッションの保存・再開
├── metrics.py           # 統計の記録
├── config.py            # 設定値・定数
└── tools/               # 各ツールの実装
    ├── bash_tool.py     # シェルコマンド実行
    ├── read_tool.py     # ファイル読み取り
    ├── write_tool.py    # ファイル書き込み
    ├── edit_tool.py     # ファイル編集
    ├── grep_tool.py     # テキスト検索
    └── glob_tool.py     # ファイル名検索
```

---

## ふつうのAIツールとのちがい

| 比較ポイント | ふつうのAIツール | RNNR_Coding |
|------------|----------------|-------------|
| 対応範囲 | なんでもOK | 6つの学習分野だけ |
| 動く場所 | クラウド（ネット必要） | 自分のパソコンだけ |
| 言語の混在 | よくある | 起きない |
| ライブラリ制限 | なし | モードごとに許可制 |
| ネット通信 | できる | できない |
| パッケージ追加 | できる | できない |
| 学習ログ | なし or バラバラ | JSONL形式で統一 |
| 再現性 | なし | あり |

---

## ライセンス

MIT ライセンスです。自由に使えます。詳しくは [LICENSE](LICENSE) をご覧ください。

---

## 開発に参加するには

プルリクエスト歓迎です。以下のルールを守ってください。

1. モードごとの言語分離を壊さないこと
2. 対応モデルを勝手に増やさないこと（事前に相談）
3. 新しいバリデーションルールにはテストをつけること
4. 既存のコードスタイル（型ヒント、docstring）に合わせること
