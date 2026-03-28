"""
run_server.py  —  CryptoBot Pro server launcher
================================================
Alternative to: uvicorn dashboard.app:app --reload --port 8000

Works with ANY uvicorn version — uses subprocess, not uvicorn Python API.

Usage:
    python run_server.py              # port 8000, with reload
    python run_server.py --port 8080
    python run_server.py --no-reload  # disable auto-reload
"""
import sys
import os
import subprocess
import argparse


def main():
    p = argparse.ArgumentParser(description="CryptoBot Pro server")
    p.add_argument("--host",      default="0.0.0.0")
    p.add_argument("--port",      default=8000, type=int)
    p.add_argument("--no-reload", action="store_true", dest="no_reload")
    p.add_argument("--log-level", default="info")
    args = p.parse_args()

    # Always run from project root
    root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "dashboard.app:app",
        "--host",      args.host,
        "--port",      str(args.port),
        "--log-level", args.log_level,
    ]
    if not args.no_reload:
        cmd += ["--reload",
                "--reload-dir", "dashboard",
                "--reload-dir", "core",
                "--reload-dir", "ai_engine",
                "--reload-dir", "indicators",
                "--reload-dir", "execution"]

    print(f"  CryptoBot Pro  →  http://{args.host}:{args.port}")
    print(f"  Command: {' '.join(cmd[2:])}")
    print(f"  Ctrl+C to stop\n")

    try:
        subprocess.run(cmd, cwd=root)
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()