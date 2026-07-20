# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check for added files exceeding size limit.

Only checks files that are staged for commit (new or modified),
matching the behavior of pre-commit/pre-commit-hooks check-added-large-files.
"""

import os
import subprocess
import sys

MAX_KB = 5000

# Get list of staged files (added or modified)
result = subprocess.run(
    ["git", "diff", "--cached", "--name-only", "--diff-filter=ACM"],
    capture_output=True,
    text=True,
)
staged = set(result.stdout.strip().splitlines()) if result.stdout.strip() else set()

retval = 0
for filename in sys.argv[1:]:
    if filename not in staged:
        continue
    try:
        size = os.path.getsize(filename)
    except OSError:
        continue
    if size > MAX_KB * 1024:
        print(f"{filename} ({size} bytes) exceeds {MAX_KB} kB limit")
        retval = 1
sys.exit(retval)
