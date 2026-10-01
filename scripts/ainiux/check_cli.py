"""Show what the real command line does for the banner/help cases.

Runs pieni.py as a subprocess with stdin closed, so an interactive start exits at
once instead of waiting for input, and prints the captured output for each case.

    python3 scripts/ainiux/check_cli.py
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIENI = ROOT / "pieni.py"


def interpreter():
    """The project virtualenv when it exists, so the SDKs are importable."""
    candidate = ROOT / ".venv" / "bin" / "python"
    return str(candidate) if os.access(candidate, os.X_OK) else sys.executable


def run(label, arguments, environment=None, cwd=None):
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    if environment:
        env.update(environment)
    completed = subprocess.run(
        [interpreter(), str(PIENI), *arguments],
        cwd=cwd or ROOT, env=env, stdin=subprocess.DEVNULL,
        capture_output=True, text=True, timeout=60)
    print(f"=== {label}: {' '.join(arguments) or '(no arguments)'}")
    print(f"exit code: {completed.returncode}")
    for line in completed.stdout.splitlines():
        print(f"  out| {line}")
    for line in completed.stderr.splitlines():
        print(f"  err| {line}")
    print()
    return completed


def main():
    # A scratch directory, so the runs do not leave a .pieni/pieni.db in the project.
    with tempfile.TemporaryDirectory() as scratch:
        run("no arguments", [])
        run("--help", ["--help"])
        run("interactive start", ["openai", "-m", "gpt-x"],
            {"OPENAI_API_KEY": "not-a-real-key"}, cwd=scratch)
        # A closed local port: the headless path fails immediately without leaving
        # this machine, which is enough to show that no banner is printed.
        run("headless", ["http://127.0.0.1:9", "-m", "test", "-r", "hi"], cwd=scratch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
