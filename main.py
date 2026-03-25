"""
Run the FastAPI RAG API (uvicorn) and the Streamlit UI together.

From the project root:

  python main.py
  python main.py --reload

Streamlit uses API_BASE pointing at this uvicorn instance (overridden for the
Streamlit child process so it matches --host / --api-port).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND_SCRIPT = ROOT / "frontend" / "main.py"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def main() -> int:
    _load_dotenv()

    parser = argparse.ArgumentParser(description="Run FastAPI + Streamlit for RAG")
    parser.add_argument(
        "--host",
        default=os.environ.get("UVICORN_HOST", "127.0.0.1"),
        help="Bind address for API and Streamlit",
    )
    parser.add_argument(
        "--api-port",
        type=int,
        default=int(os.environ.get("UVICORN_PORT", "8000")),
        help="Uvicorn port",
    )
    parser.add_argument(
        "--streamlit-port",
        type=int,
        default=int(os.environ.get("STREAMLIT_SERVER_PORT", "8501")),
        help="Streamlit server port",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Uvicorn auto-reload on code changes (dev)",
    )
    args = parser.parse_args()

    if not FRONTEND_SCRIPT.is_file():
        print(f"Missing Streamlit app: {FRONTEND_SCRIPT}", file=sys.stderr)
        return 1

    uvicorn_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        args.host,
        "--port",
        str(args.api_port),
    ]
    if args.reload:
        uvicorn_cmd.append("--reload")

    streamlit_env = os.environ.copy()
    streamlit_env["API_BASE"] = f"http://{args.host}:{args.api_port}"

    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(FRONTEND_SCRIPT),
        "--server.address",
        args.host,
        "--server.port",
        str(args.streamlit_port),
        "--browser.gatherUsageStats",
        "false",
    ]

    procs: list[subprocess.Popen] = []

    def shutdown() -> None:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        for p in procs:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()

    api_url = f"http://{args.host}:{args.api_port}"
    ui_url = f"http://{args.host}:{args.streamlit_port}"
    print(f"[runner] API (uvicorn): {api_url}")
    print(f"[runner] UI  (streamlit): {ui_url}")
    print("[runner] Press Ctrl+C to stop both\n")

    procs.append(subprocess.Popen(uvicorn_cmd, cwd=ROOT))
    time.sleep(1.5)
    procs.append(
        subprocess.Popen(streamlit_cmd, cwd=ROOT, env=streamlit_env),
    )

    try:
        while True:
            for p in procs:
                code = p.poll()
                if code is not None:
                    print(f"[runner] A process exited with code {code}; stopping others.")
                    shutdown()
                    return int(code) if code is not None else 0
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[runner] Interrupted; shutting down...")
        shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
