"""Entry point for RNNR_Coding."""

from __future__ import annotations

import json
import os
import sys
import time

from rnnr.cli import build_parser
from rnnr.config import (
    Mode,
    WORKING_DIR,
    build_mode_profile,
    validate_model,
    get_system_ram_gb,
)
from rnnr.llm_adapter import LLMAdapter
from rnnr.security_manager import SecurityManager
from rnnr.policy_engine import PolicyEngine
from rnnr.validator import Validator
from rnnr.tool_registry import ToolRegistry
from rnnr.session_manager import SessionManager
from rnnr.metrics import Metrics
from rnnr.agent_loop import AgentLoop


BANNER = r"""
 ____  _   _ _   _ ____     ____          _ _
|  _ \| \ | | \ | |  _ \   / ___|___   __| (_)_ __   __ _
| |_) |  \| |  \| | |_) | | |   / _ \ / _` | | '_ \ / _` |
|  _ <| |\  | |\  |  _ <  | |__| (_) | (_| | | | | | (_| |
|_| \_\_| \_|_| \_|_| \_\  \____\___/ \__,_|_|_| |_|\__, |
                                                      |___/
  Curriculum-Constrained Local AI Coding Agent  v0.1.0
"""

INTERACTIVE_HELP = """
Commands:
  /help     — Show this help
  /exit     — Exit the session
  /clear    — Clear conversation history
  /metrics  — Show session metrics
  /trace    — Show last LLM interaction trace
  /status   — Show session status
  /save     — Save session to disk
"""


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

    # ── Validate model ──────────────────────────────────────────
    try:
        model_name = validate_model(args.model)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    mode = Mode(args.mode)
    profile = build_mode_profile(mode)
    ram_gb = get_system_ram_gb()

    print(BANNER)
    print(f"  Model   : {model_name}")
    print(f"  Mode    : {mode.value}")
    print(f"  RAM     : {ram_gb:.1f} GB")
    print(f"  Research: {'ON' if args.research else 'OFF'}")
    print(f"  WorkDir : {WORKING_DIR}")
    print()

    # ── Initialize components ───────────────────────────────────
    security = SecurityManager(WORKING_DIR)
    policy = PolicyEngine(profile)
    validator = Validator(profile)
    llm = LLMAdapter(model_name)
    metrics = Metrics()

    # Resume or create session
    if args.resume:
        try:
            session = sm.load(args.resume)
            print(f"  Resumed session: {session.meta.session_id}")
        except FileNotFoundError:
            print(f"Session not found: {args.resume}")
            sys.exit(1)
    else:
        session = sm.create(model_name, mode.value, research=args.research)
        print(f"  Session : {session.meta.session_id}")

    tools = ToolRegistry(security, policy, profile, WORKING_DIR)
    loop = AgentLoop(
        llm=llm,
        profile=profile,
        tools=tools,
        validator=validator,
        policy=policy,
        session=session,
        metrics=metrics,
        max_iterations=args.max_iterations,
        research=args.research,
        debug=args.debug,
    )

    # ── Check Ollama connectivity ───────────────────────────────
    if not llm.ping():
        print(f"\n  WARNING: Cannot reach Ollama or model '{model_name}' not loaded.")
        print(f"  Run: ollama pull {model_name}")
        print()

    print(f"\n  Type /help for commands. Start coding!\n")

    # ── Interactive loop ────────────────────────────────────────
    try:
        while True:
            try:
                user_input = input("You> ").strip()
            except EOFError:
                break

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
                print(f"Iters   : {metrics.iterations}")
                print(f"Elapsed : {metrics.elapsed_seconds():.1f}s")
                continue
            elif user_input == "/save":
                path = session.save()
                print(f"Session saved: {path}")
                continue

            # Run agent turn
            response = loop.run_turn(user_input)
            print(f"\nAssistant> {response}\n")

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if args.research:
            session.add("metrics_final", metrics.summary())
            path = session.save()
            print(f"\nResearch session saved: {path}")


if __name__ == "__main__":
    main()
