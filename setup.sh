#!/usr/bin/env bash
# NoviCode セットアップスクリプト
set -e

GREEN='\033[38;2;118;185;0m'
BOLD='\033[1m'
DIM='\033[90m'
WHITE='\033[97m'
RESET='\033[0m'

# uv が入っているか確認
if ! command -v uv &> /dev/null; then
    echo ""
    echo -e "  ${GREEN}❌${RESET} uv が見つかりません"
    echo -e "  ${DIM}インストール: curl -LsSf https://astral.sh/uv/install.sh | sh${RESET}"
    echo ""
    exit 1
fi

# uv sync 実行（出力はそのまま表示）
uv sync

# macOS: .venv の UF_HIDDEN フラグを除去
# (.で始まるディレクトリに macOS が自動付与 → site.py が .pth を無視する問題の回避)
if [[ "$(uname)" == "Darwin" ]] && [[ -d .venv ]]; then
    chflags -R nohidden .venv 2>/dev/null || true
fi

# ガイド表示
echo ""
echo -e "  ${GREEN}✓${RESET} セットアップ完了！"
echo ""
echo -e "  ${DIM}──────────────────────────────────────────────${RESET}"
echo -e "  ${WHITE}次のコマンドで起動できます:${RESET}"
echo ""
echo -e "    ${BOLD}${GREEN}uv run novicode${RESET}                       ${DIM}# モード選択画面から開始${RESET}"
echo ""
echo -e "  ${DIM}モードを指定して起動:${RESET}"
echo -e "    ${WHITE}uv run novicode --mode py5${RESET}            ${DIM}# Py5 クリエイティブコーディング${RESET}"
echo -e "    ${WHITE}uv run novicode --mode python_basic${RESET}   ${DIM}# Python 基礎${RESET}"
echo -e "    ${WHITE}uv run novicode --mode pandas${RESET}         ${DIM}# pandas データ分析${RESET}"
echo -e "    ${WHITE}uv run novicode --mode sklearn${RESET}        ${DIM}# scikit-learn 機械学習${RESET}"
echo -e "    ${WHITE}uv run novicode --mode web_basic${RESET}      ${DIM}# Web 基礎 (HTML/CSS/JS)${RESET}"
echo -e "    ${WHITE}uv run novicode --mode aframe${RESET}         ${DIM}# A-Frame WebXR${RESET}"
echo -e "    ${WHITE}uv run novicode --mode threejs${RESET}        ${DIM}# Three.js 3D${RESET}"
echo -e "  ${DIM}──────────────────────────────────────────────${RESET}"
echo ""
