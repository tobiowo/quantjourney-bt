# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Block private, generated, or secret-bearing files from being committed.

``.gitignore`` stops untracked files from being added with ``git add .``, but it
does not stop ``git add -f``, and it cannot help once a file is already staged.
This guard is the second line of defence: it inspects the exact paths that are
about to be committed and refuses the ones that should never leave the machine
(private scratch directories, market-data dumps, report artifacts, oversized
files, and text containing token-shaped strings).

It is deliberately shell-free so it can run anywhere Python runs, including
restricted environments where Git's ``sh``-based hooks cannot start.

Usage:
    python tools/check_private_artifacts.py --staged   # what a commit will add
    python tools/check_private_artifacts.py --all      # the tracked tree
    python tools/check_private_artifacts.py --all --max-mb 2

Exit code 0 means nothing matched; 1 lists the offending paths and reasons.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Root-level directories that hold local work, fetched data, or run artifacts.
BLOCKED_DIRS = {
    "private",
    "scratch",
    "notes",
    "_requests",
    "data",
    "tmp",
    "reports",
    "plots",
}

# File types that are almost always data dumps, caches, or credentials.
BLOCKED_SUFFIXES = {
    ".parquet",
    ".feather",
    ".pkl",
    ".pickle",
    ".h5",
    ".hdf5",
    ".sqlite",
    ".sqlite3",
    ".db",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".crt",
    ".csr",
    ".log",
}

# Environment files are blocked except the documented example template.
ALLOWED_NAMES = {".env.example"}

# Token-shaped strings that should never be committed, even in a test fixture.
SECRET_PATTERNS = (
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def _git_paths(*args: str) -> list[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [path for path in completed.stdout.split("\0") if path]


def _staged_paths() -> list[str]:
    # Added, copied, modified and renamed files: the ones a commit would include.
    return _git_paths("diff", "--cached", "--name-only", "-z", "--diff-filter=ACMR")


def _tracked_paths() -> list[str]:
    return _git_paths("ls-files", "-z")


def _is_text(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return b"\0" not in handle.read(2048)
    except OSError:
        return False


def _text_issues(path: Path) -> list[str]:
    if not _is_text(path):
        return []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return []
    return [
        f"matches secret pattern {pattern.pattern!r}"
        for pattern in SECRET_PATTERNS
        if pattern.search(text)
    ]


def check_path(relative: str, max_bytes: int) -> list[str]:
    """Return the reasons this path must not be committed (empty means clean)."""
    path = ROOT / relative
    issues: list[str] = []

    parts = Path(relative).parts
    if parts and parts[0] in BLOCKED_DIRS:
        issues.append(f"lives under blocked directory {parts[0]!r}")

    name = path.name
    lower_name = name.lower()
    if lower_name == ".env" or (lower_name.startswith(".env.") and name not in ALLOWED_NAMES):
        issues.append("environment file")
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        issues.append(f"blocked file type {path.suffix.lower()!r}")

    if path.is_file():
        size = path.stat().st_size
        if size > max_bytes:
            issues.append(
                f"{size / 1024 / 1024:.1f} MB exceeds the {max_bytes / 1024 / 1024:.1f} MB limit"
            )
        issues.extend(_text_issues(path))
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--staged", action="store_true", help="check paths staged for the next commit"
    )
    group.add_argument("--all", action="store_true", help="check every tracked path")
    parser.add_argument(
        "--max-mb", type=float, default=5.0, help="per-file size limit in MB (default 5)"
    )
    args = parser.parse_args(argv)

    max_bytes = max(1, int(args.max_mb * 1024 * 1024))
    paths = _staged_paths() if args.staged else _tracked_paths()
    violations = [(path, check_path(path, max_bytes)) for path in paths]
    violations = [(path, reasons) for path, reasons in violations if reasons]

    if not violations:
        scope = "staged paths" if args.staged else "tracked paths"
        print(f"check_private_artifacts: {len(paths)} {scope} clean")
        return 0

    print(
        "check_private_artifacts: refusing to commit private or sensitive content", file=sys.stderr
    )
    for path, reasons in violations:
        print(f"  {path}", file=sys.stderr)
        for reason in reasons:
            print(f"    - {reason}", file=sys.stderr)
    print(
        "\nUnstage the files (git restore --staged <path>) or remove them from the tree.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
