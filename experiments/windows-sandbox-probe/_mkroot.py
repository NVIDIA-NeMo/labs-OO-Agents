# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the private runtime root used by the probe and the sandbox backend.

This is a thin wrapper: the implementation lives in
``nooa.runtime.sandbox.win_runtime`` so the experiment and the shipped backend
cannot drift apart. It exists here because the probe's README documents it, and
because experiments should stay runnable without installing the package.

Defaults to ``<this directory>/.rt`` (the probe's isolated copy) rather than the
per-user cache the backend uses, so a probe run never disturbs a root that a
real session depends on.

    python _mkroot.py              # build the whole root (~4.5 min, ~420 MB)
    python _mkroot.py --sync-nooa  # refresh only the nooa package (<2 s)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# The probe may be run without an installed package.
sys.path.insert(0, str(REPO / "src"))

from nooa.runtime.sandbox.win_runtime import (  # noqa: E402
    RuntimeRoot,
    build_root,
    sync_package,
)

DST = HERE / ".rt"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sync-nooa",
        action="store_true",
        help="refresh the nooa package in the existing root (sub-second) and exit",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="with --sync-nooa, copy even when the source signature is unchanged",
    )
    args = parser.parse_args()
    root = RuntimeRoot(DST)

    if args.sync_nooa:
        root.check()
        print(f"nooa package: {sync_package(root, force=args.force)}")
        print("package present:", (root.package / "__init__.py").exists())
        return 0

    print(build_root(root))
    print(f"nooa package: {sync_package(root, force=True)}")
    print("python.exe:", root.python.exists())
    return 0


if __name__ == "__main__":
    sys.exit(main())
