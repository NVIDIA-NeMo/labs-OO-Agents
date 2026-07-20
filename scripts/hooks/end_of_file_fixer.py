# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fix files to end with exactly one newline."""

import sys

retval = 0
for filename in sys.argv[1:]:
    try:
        with open(filename, "rb") as f:
            contents = f.read()
    except (PermissionError, IsADirectoryError):
        continue
    if not contents:
        continue
    if not contents.endswith(b"\n") or contents.endswith(b"\n\n"):
        if not contents.endswith(b"\n"):
            fixed = contents + b"\n"
        else:
            fixed = contents.rstrip(b"\n") + b"\n"
        with open(filename, "wb") as f:
            f.write(fixed)
        print(f"Fixing {filename}")
        retval = 1
sys.exit(retval)
