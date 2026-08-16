"""Command-line launcher for the optional Streamlit run dashboard."""

import argparse
from collections.abc import Sequence
import importlib.util
import math
from pathlib import Path
import subprocess
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deeptool-dashboard")
    parser.add_argument("runs_dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--refresh", type=float, default=2.0)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate options and launch the packaged Streamlit application."""
    parser = _parser()
    args = parser.parse_args(argv)
    root = args.runs_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"runs directory does not exist: {root}")
    if not args.host:
        parser.error("host must be non-empty")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if not math.isfinite(args.refresh) or args.refresh < 0:
        parser.error("refresh must be a finite non-negative number")
    if importlib.util.find_spec("streamlit") is None:
        parser.error(
            'dashboard dependencies are missing; run: '
            'uv add "deeptool[dashboard]"'
        )
    app = Path(__file__).with_name("_dashboard_app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        f"--server.address={args.host}",
        f"--server.port={args.port}",
        f"--server.headless={str(args.no_browser).lower()}",
        "--",
        str(root),
        "--refresh",
        str(args.refresh),
    ]
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
