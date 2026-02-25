"""Entry point for NoviCode."""

from __future__ import annotations

import json
import os
import sys
import time

from novicode.cli import build_parser
from novicode.config import (
    Mode,
    WORKING_DIR,
    build_mode_profile,
    validate_model,
    get_system_ram_gb,
    list_ollama_models,
)
from novicode.llm_adapter import LLMAdapter
from novicode.security_manager import SecurityManager
from novicode.policy_engine import PolicyEngine
from novicode.validator import Validator
from novicode.tool_registry import ToolRegistry
from novicode.session_manager import SessionManager
from novicode.metrics import Metrics
from novicode.agent_loop import AgentLoop, StatusEvent, CodeWriteEvent
from novicode.curriculum import Level
from novicode.progress import ProgressTracker
from novicode.challenges import (
    get_random_challenge,
    format_challenge,
    format_hint,
    Challenge,
)
from novicode.formatter import StreamFormatter, _highlight_code
from novicode.spinner import Spinner
from novicode.input_reader import InputReader


# ── ANSI color constants ─────────────────────────────────────
_GREEN = "\033[38;2;118;185;0m"   # NVIDIA Green #76B900
_BOLD  = "\033[1m"
_DIM   = "\033[90m"
_WHITE = "\033[97m"
_RESET = "\033[0m"

# Gradient: bright lime → NVIDIA green → deep emerald (top → bottom)
_GRADIENT = [
    "\033[38;2;166;227;34m",   # #A6E322  bright lime
    "\033[38;2;142;206;17m",   # #8ECE11
    "\033[38;2;118;185;0m",    # #76B900  NVIDIA Green
    "\033[38;2;94;164;0m",     # #5EA400
    "\033[38;2;70;143;0m",     # #468F00
    "\033[38;2;50;122;0m",     # #327A00  deep emerald
]

_BANNER_LINES = [
    " ███╗   ██╗ ██████╗ ██╗   ██╗██╗ ██████╗ ██████╗ ██████╗ ███████╗",
    " ████╗  ██║██╔═══██╗██║   ██║██║██╔════╝██╔═══██╗██╔══██╗██╔════╝",
    " ██╔██╗ ██║██║   ██║██║   ██║██║██║     ██║   ██║██║  ██║█████╗  ",
    " ██║╚██╗██║██║   ██║╚██╗ ██╔╝██║██║     ██║   ██║██║  ██║██╔══╝  ",
    " ██║ ╚████║╚██████╔╝ ╚████╔╝ ██║╚██████╗╚██████╔╝██████╔╝███████╗",
    " ╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝",
]

BANNER = (
    "\n"
    + "\n".join(
        f"{_BOLD}{c}{line}{_RESET}"
        for c, line in zip(_GRADIENT, _BANNER_LINES)
    )
    + "\n"
    + f"{_DIM}  🎓 P R O G R A M M I N G   L E A R N I N G   A G E N T 🎓{_RESET}\n"
    + f"{_DIM}  v0.2.0 // Offline • Local LLM • Powered by Ollama{_RESET}\n"
)

INTERACTIVE_HELP = """
Commands:
  /help      — Show this help
  /exit      — Exit the session
  /clear     — Clear conversation history
  /metrics   — Show session metrics
  /trace     — Show last LLM interaction trace
  /status    — Show session status
  /save      — Save session to disk
  /progress  — Show learning progress
  /level     — Show current level
  /challenge — Get a practice challenge
  /hint      — Show hint for current challenge

Keybinds:
  Enter        — 改行（複数行入力）
  Shift+Enter  — 送信
  Ctrl+D       — 送信（フォールバック）
  ESC          — 終了
"""


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable string."""
    gb = size_bytes / (1024**3)
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    mb = size_bytes / (1024**2)
    return f"{mb:.0f} MB"


def _select_model_interactive() -> str:
    """Show interactive model selection and return the chosen model name.

    Exits the process if no models are available.
    """
    models = list_ollama_models()

    if not models:
        print(f"\n  {_GREEN}❌ Ollama に接続できないか、モデルがインストールされていません。{_RESET}")
        print(f"  {_DIM}Ollama が起動しているか確認してください: ollama serve{_RESET}")
        sys.exit(1)

    if len(models) == 1:
        name = models[0]["name"]
        print(f"\n  {_GREEN}🧠 モデル自動選択:{_RESET} {_WHITE}{name}{_RESET}")
        return name

    # Display selection menu
    sep = "─" * 46
    print(f"\n  {_GREEN}利用可能なモデル:{_RESET}")
    print(f"  {_DIM}{sep}{_RESET}")
    for i, m in enumerate(models, 1):
        size_str = _format_size(m["size"])
        name = m["name"]
        print(f"    {_WHITE}{i:>2}. {name:<36}{_RESET}{_DIM}({size_str}){_RESET}")
    print(f"  {_DIM}{sep}{_RESET}")

    while True:
        try:
            raw = input(f"  {_GREEN}番号を入力 (default: 1):{_RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)

        if not raw:
            return models[0]["name"]

        try:
            idx = int(raw)
        except ValueError:
            print(f"  {_DIM}数字を入力してください{_RESET}")
            continue

        if 1 <= idx <= len(models):
            return models[idx - 1]["name"]
        print(f"  {_DIM}1〜{len(models)} の番号を入力してください{_RESET}")


def _render_code_card(event: CodeWriteEvent) -> str:
    """Render a CodeWriteEvent as a syntax-highlighted code card."""
    filename = os.path.basename(event.path)
    bar = "─" * 40
    highlighted = _highlight_code(event.content, event.lang).rstrip("\n")
    return (
        f"\n  {_GREEN}📝 {filename}{_RESET}\n"
        f"  {_DIM}{bar}{_RESET}\n"
        f"{highlighted}\n"
        f"  {_DIM}{bar}{_RESET}\n"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    sm = SessionManager()

    # ── List sessions ───────────────────────────────────────────
    if args.list_sessions:
        sessions = sm.list_sessions()
        if not sessions:
            print("No saved sessions.")
        else:
            for s in sessions:
                ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(s.get("created_at", 0)))
                print(f"  {s['session_id']}  mode={s['mode']}  model={s['model']}  {ts}")
        return

    # ── Export session ──────────────────────────────────────────
    if args.export_session:
        try:
            session = sm.load(args.export_session)
            path = session.export_jsonl()
            print(f"Exported to: {path}")
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            sys.exit(1)
        return

    # ── Select / validate model ────────────────────────────────
    model_name = validate_model(args.model)
    if model_name == "auto":
        model_name = _select_model_interactive()

    mode = Mode(args.mode)

    # ── PY5 モード: py5 自動インストール ────────────────────────
    if mode == Mode.PY5:
        try:
            import py5  # noqa: F401
        except ImportError:
            import importlib
            import shutil
            import subprocess

            print("py5 が見つかりません。インストールしています...")
            uv = shutil.which("uv")
            if not uv:
                print("Error: uv が見つかりません。")
                print("  手動で実行してください: uv sync --extra py5")
                sys.exit(1)
            result = subprocess.run(
                [uv, "sync", "--extra", "py5"],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                print("Error: py5 のインストールに失敗しました。")
                print("  手動で実行してください: uv sync --extra py5")
                if result.stderr:
                    print(result.stderr)
                sys.exit(1)
            print("py5 をインストールしました。")
            importlib.invalidate_caches()
            try:
                import py5  # noqa: F401
            except ImportError:
                print("py5 のインポートに失敗しました。再起動します...")
                os.execv(sys.executable, [sys.executable] + sys.argv)

    profile = build_mode_profile(mode)
    ram_gb = get_system_ram_gb()

    # ── Load progress & determine level ─────────────────────────
    progress = ProgressTracker.load(mode)
    if args.level:
        progress._level = Level(args.level)
    level = progress.level

    level_ja = {"beginner": "初級", "intermediate": "中級", "advanced": "上級"}

    sep = f"{_DIM}  {'─' * 48}{_RESET}"
    print(BANNER)
    print(sep)
    print(f"  {_GREEN}🧠 Model{_RESET}   {_WHITE}{model_name}{_RESET}")
    print(f"  {_GREEN}📚 Mode{_RESET}    {_WHITE}{mode.value}{_RESET}")
    print(f"  {_GREEN}🎯 Level{_RESET}   {_WHITE}{level_ja.get(level.value, level.value)} ({level.value}){_RESET}")
    print(f"  {_GREEN}💾 RAM{_RESET}     {_WHITE}{ram_gb:.1f} GB{_RESET}")
    print(f"  {_GREEN}📁 WorkDir{_RESET} {_WHITE}{WORKING_DIR}{_RESET}")
    print(f"  {_GREEN}🔬 Research{_RESET} {_WHITE}{'ON' if args.research else 'OFF'}{_RESET}")
    print(sep)
    print()
    print(f"  {_GREEN}💡 使い方{_RESET}  {_WHITE}Enter で改行、{_BOLD}空行+Enter{_RESET}{_WHITE} で送信（複数行OK）  {_DIM}ESC: 終了  Ctrl+D: 送信{_RESET}")
    print(f"  {_GREEN}📝 コマンド{_RESET} {_DIM}/help{_RESET} 一覧  {_DIM}/exit{_RESET} 終了  {_DIM}/challenge{_RESET} 練習問題  {_DIM}/progress{_RESET} 進捗")

    # ── Initialize components ───────────────────────────────────
    security = SecurityManager(WORKING_DIR)
    policy = PolicyEngine(profile, level=level)
    validator = Validator(profile)
    llm = LLMAdapter(model_name)
    metrics = Metrics()

    # Resume or create session
    if args.resume:
        try:
            session = sm.load(args.resume)
            print(f"  {_GREEN}🆔 Session{_RESET} {_WHITE}{session.meta.session_id} (resumed){_RESET}")
        except FileNotFoundError:
            print(f"Session not found: {args.resume}")
            sys.exit(1)
    else:
        session = sm.create(model_name, mode.value, research=args.research)
        print(f"  {_GREEN}🆔 Session{_RESET} {_WHITE}{session.meta.session_id}{_RESET}")

    tools = ToolRegistry(security, policy, profile, WORKING_DIR)
    loop = AgentLoop(
        llm=llm,
        profile=profile,
        tools=tools,
        validator=validator,
        policy=policy,
        session=session,
        metrics=metrics,
        progress=progress,
        max_iterations=args.max_iterations,
        research=args.research,
        debug=args.debug,
    )

    # ── Check Ollama connectivity ───────────────────────────────
    if not llm.ping():
        print(f"\n  WARNING: Cannot reach Ollama or model '{model_name}' not loaded.")
        print(f"  Run: ollama pull {model_name}")
        print()

    print()  # blank line before prompt

    # ── Track current challenge for /hint ───────────────────────
    current_challenge: Challenge | None = None

    # ── Interactive loop ────────────────────────────────────────
    _BOX_W = 48
    _BOX_BOT = f"{_DIM}╰{'─' * _BOX_W}{_RESET}"
    _BOX_L   = f"{_DIM}│{_RESET}"
    _PROMPT_FIRST = f"{_BOX_L} {_GREEN}{_BOLD}You>{_RESET} "
    _PROMPT_CONT  = f"{_BOX_L} {_DIM}  ..{_RESET}  "

    _BOX_HINT = f" {_DIM}Enter: 改行  空Enter: 送信  ESC: 終了{_RESET}"
    _BOX_TOP = f"{_DIM}╭{'─' * _BOX_W}{_RESET}{_BOX_HINT}"

    reader = InputReader(
        prompt_first=_PROMPT_FIRST,
        prompt_cont=_PROMPT_CONT,
        box_top=_BOX_TOP,
        box_bottom=_BOX_BOT,
    )

    try:
        with reader:
            while True:
                result = reader.read_input()

                if result.action == "exit":
                    break

                user_input = result.text
                if not user_input:
                    continue

                # Interactive commands
                if user_input == "/exit":
                    break
                elif user_input == "/help":
                    print(INTERACTIVE_HELP)
                    continue
                elif user_input == "/clear":
                    loop.messages = loop.messages[:1]  # keep system prompt
                    print("Conversation cleared.")
                    continue
                elif user_input == "/metrics":
                    print(metrics.display())
                    continue
                elif user_input == "/trace":
                    if len(loop.messages) >= 2:
                        last = loop.messages[-1]
                        print(f"[{last.role}] {last.content[:500]}")
                    else:
                        print("No trace available.")
                    continue
                elif user_input == "/status":
                    print(f"Session : {session.meta.session_id}")
                    print(f"Model   : {model_name}")
                    print(f"Mode    : {mode.value}")
                    print(f"Level   : {level_ja.get(progress.level.value, progress.level.value)}")
                    print(f"Iters   : {metrics.iterations}")
                    print(f"Elapsed : {metrics.elapsed_seconds():.1f}s")
                    continue
                elif user_input == "/save":
                    path = session.save()
                    progress.save()
                    print(f"Session saved: {path}")
                    continue
                elif user_input == "/progress":
                    print(progress.display())
                    continue
                elif user_input == "/level":
                    lv = progress.level
                    print(f"現在のレベル: {level_ja.get(lv.value, lv.value)} ({lv.value})")
                    mastered = progress.mastered_concepts()
                    print(f"習得済み概念: {len(mastered)} 個")
                    if mastered:
                        print(f"  {', '.join(sorted(mastered))}")
                    continue
                elif user_input == "/challenge":
                    ch = get_random_challenge(mode, progress.level)
                    if ch:
                        current_challenge = ch
                        print(format_challenge(ch))
                    else:
                        print("このモード・レベルにはチャレンジがありません。")
                    continue
                elif user_input == "/hint":
                    if current_challenge:
                        print(format_hint(current_challenge))
                    else:
                        print("先に /challenge でチャレンジを取得してください。")
                    continue

                # Run agent turn (streaming with syntax highlighting)
                # Suspend raw mode so Ctrl+C generates SIGINT during LLM calls
                reader.suspend()
                fmt = StreamFormatter()
                header_shown = False
                spinner = Spinner()
                try:
                    for chunk in loop.run_turn_stream(user_input):
                        if isinstance(chunk, StatusEvent):
                            if chunk.kind == "thinking":
                                spinner.start("考え中...")
                            elif chunk.kind == "tool_start":
                                spinner.start(f"実行中: {chunk.detail}...")
                            elif chunk.kind == "tool_done":
                                spinner.start("考え中...")
                            continue
                        if isinstance(chunk, CodeWriteEvent):
                            spinner.stop()
                            sys.stdout.write(_render_code_card(chunk))
                            sys.stdout.flush()
                            continue
                        spinner.stop()
                        output = fmt.feed(chunk)
                        if output:
                            if not header_shown:
                                sys.stdout.write(f"\n{_WHITE}{_BOLD}Assistant>{_RESET}\n")
                                header_shown = True
                            sys.stdout.write(output)
                            sys.stdout.flush()
                    remaining = fmt.flush()
                    if remaining:
                        if not header_shown:
                            sys.stdout.write(f"\n{_WHITE}{_BOLD}Assistant>{_RESET}\n")
                            header_shown = True
                        sys.stdout.write(remaining)
                        sys.stdout.flush()
                    if header_shown:
                        sys.stdout.write("\n\n")
                        sys.stdout.flush()
                except KeyboardInterrupt:
                    spinner.stop()
                    sys.stdout.write(f"\n  {_DIM}(Ctrl+C: 中断しました){_RESET}\n\n")
                    sys.stdout.flush()
                except (TimeoutError, ConnectionError) as exc:
                    spinner.stop()
                    sys.stdout.write(f"\n  {_DIM}⚠ {exc}{_RESET}\n\n")
                    sys.stdout.flush()
                finally:
                    spinner.stop()
                    reader.resume()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        progress.save()
        if args.research:
            session.add("metrics_final", metrics.summary())
            path = session.save()
            print(f"\nResearch session saved: {path}")


if __name__ == "__main__":
    main()
