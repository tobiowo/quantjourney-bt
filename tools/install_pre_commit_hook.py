# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Install the private-artifact guard as a Git pre-commit hook.

The hook runs ``tools/check_private_artifacts.py --staged`` and refuses the
commit when private scratch directories, data dumps, oversized files, or
token-shaped strings are staged.

Usage:
    python tools/install_pre_commit_hook.py
    python tools/install_pre_commit_hook.py --dry-run
    python tools/install_pre_commit_hook.py --force

Note: Git runs hooks through ``sh``. In restricted sandboxes where ``sh`` cannot
start, install this on a normal machine (or run the checker manually before
committing).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK_MARKER = "# Installed by tools/install_pre_commit_hook.py"

HOOK_TEMPLATE = """#!/bin/sh
{marker}
# Blocks private scratch directories, data dumps, oversized files, and secrets.
set -e
root=$(git rev-parse --show-toplevel)
exec "{python}" "$root/tools/check_private_artifacts.py" --staged
"""


def _repo_root() -> Path:
    completed = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(completed.stdout.strip())


def _interpreter() -> Path:
    """Prefer the repository virtual environment when it exists."""
    for candidate in (ROOT / ".venv" / "Scripts" / "python.exe", ROOT / ".venv" / "bin" / "python"):
        if candidate.is_file():
            return candidate
    return Path(sys.executable)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="print the hook instead of writing it"
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing hook")
    args = parser.parse_args(argv)

    root = _repo_root()
    hooks_dir = root / ".git" / "hooks"
    hook_path = hooks_dir / "pre-commit"
    content = HOOK_TEMPLATE.format(marker=HOOK_MARKER, python=_interpreter().as_posix())

    if args.dry_run:
        print(f"would write {hook_path}:\n")
        print(content)
        return 0

    if hook_path.exists() and not args.force:
        existing = hook_path.read_text(encoding="utf-8", errors="ignore")
        if HOOK_MARKER not in existing:
            print(
                f"refusing to overwrite an existing hook at {hook_path} (use --force)",
                file=sys.stderr,
            )
            return 1

    hooks_dir.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(content, encoding="utf-8")
    hook_path.chmod(0o755)
    print(f"installed {hook_path}")

    hooks_path = subprocess.run(
        ["git", "config", "--get", "core.hooksPath"],
        cwd=root,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if hooks_path:
        print(
            f"warning: core.hooksPath is set to {hooks_path!r}; Git will not run this hook "
            "until that setting is removed",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
