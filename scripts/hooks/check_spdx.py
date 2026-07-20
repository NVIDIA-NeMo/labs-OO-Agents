# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ensure Python source files have an SPDX-License-Identifier header.

Auto-adds the header if missing (pre-commit will then re-stage the file).
"""

import sys

MARKER = "SPDX-License-Identifier"
SEARCH_LINES = 5
HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES."
    " All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n"
)

retval = 0
for filename in sys.argv[1:]:
    try:
        with open(filename) as f:
            content = f.read()
    except (UnicodeDecodeError, PermissionError):
        continue

    head = content.split("\n", SEARCH_LINES)[:SEARCH_LINES]
    if any(MARKER in line for line in head):
        continue

    # Auto-add header, preserving any shebang on line 1
    if content.startswith("#!"):
        shebang, rest = content.split("\n", 1)
        content = shebang + "\n" + HEADER + rest
    else:
        content = HEADER + content

    with open(filename, "w") as f:
        f.write(content)

    print(f"Added SPDX header: {filename}")
    retval = 1  # Signal pre-commit to re-stage the file

sys.exit(retval)
