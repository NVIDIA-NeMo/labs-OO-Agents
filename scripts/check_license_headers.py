# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fail if any source Python file is missing an SPDX license header.

Run locally with ``uv run python scripts/check_license_headers.py``; the CI
``lint`` job runs it on every merge request. Empty ``__init__.py`` package
markers (0 bytes) are exempt — they hold no copyrightable content.

Uses Git's file and exclude handling as the source of truth: tracked Python
files and untracked, nonignored Python files are checked. Files excluded by
``.gitignore``, ``.git/info/exclude``, or the user's global Git excludes are
not repository source and are not scanned.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REQUIRED = "SPDX-License-Identifier"
# Only the first few lines are checked (header sits above any docstring, after
# an optional shebang/coding line).
HEAD_LINES = 5


def source_python_files(root: Path) -> list[Path]:
    """Return tracked and untracked, nonignored Python files under *root*."""
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "*.py",
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )
    relative_paths = (os.fsdecode(raw) for raw in result.stdout.split(b"\0") if raw)
    return sorted(root / relative for relative in relative_paths)


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    files = source_python_files(root)
    missing: list[str] = []
    for path in files:
        try:
            with path.open(encoding="utf-8") as fh:
                head = "".join(fh.readline() for _ in range(HEAD_LINES))
        except OSError:
            continue
        if head.strip() == "":
            continue  # empty package marker — exempt
        if REQUIRED not in head:
            missing.append(str(path.relative_to(root)))

    if missing:
        print(f"ERROR: {len(missing)} file(s) missing an SPDX license header:")
        for rel in missing:
            print(f"  {rel}")
        print(
            "\nAdd the header to the top of each file (after any shebang):\n"
            "  # SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION "
            "& AFFILIATES. All rights reserved.\n"
            "  # SPDX-License-Identifier: Apache-2.0"
        )
        return 1

    print(f"OK: all {len(files)} source Python files carry an SPDX header.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
