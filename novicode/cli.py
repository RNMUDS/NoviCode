"""CLI argument parser for NoviCode."""

from __future__ import annotations

import argparse

from novicode.config import SUPPORTED_MODELS, Mode, DEFAULT_MAX_ITERATIONS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="novicode",
        description="NoviCode — プログラミング学習に特化したバイブコーディングツール",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="auto",
        help=f"Model to use: {', '.join(sorted(SUPPORTED_MODELS))} or 'auto' (default: auto)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=[m.value for m in Mode],
        help="Required. One of: " + ", ".join(m.value for m in Mode),
    )
    parser.add_argument(
        "--safe-mode",
        action="store_true",
        default=False,
        help="Enable extra safety restrictions",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Print debug information",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Max agent loop iterations (default: {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--research",
        action="store_true",
        default=False,
        help="Enable research mode (logs all prompts, outputs, and violations)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="SESSION_ID",
        help="Resume a previous session by ID",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        default=False,
        help="List all saved sessions and exit",
    )
    parser.add_argument(
        "--export-session",
        type=str,
        default=None,
        metavar="SESSION_ID",
        help="Export a session to JSONL and exit",
    )
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        choices=["beginner", "intermediate", "advanced"],
        help="Override starting level (default: auto-detected from progress)",
    )
    return parser
