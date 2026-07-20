# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check for merge conflict markers."""

import re
import sys

PATTERN = re.compile(r"^(<{7}|>{7}|={7}|\|{7})( |$)", re.MULTILINE)

retval = 0
for filename in sys.argv[1:]:
    try:
        with open(filename) as f:
            content = f.read()
    except (UnicodeDecodeError, PermissionError):
        continue
    if PATTERN.search(content):
        print(f"Merge conflict marker found in {filename}")
        retval = 1
sys.exit(retval)
