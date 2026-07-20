# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check and fix trailing whitespace (mimics pre-commit-hooks trailing-whitespace)."""

import sys


def _process_line(line: bytes, chars: bytes | None) -> bytes:
    """Strip trailing whitespace from a line. Match official hook default: strip all (no markdown preserve)."""
    if line.endswith(b"\r\n"):
        eol = b"\r\n"
        line = line[:-2]
    elif line.endswith(b"\n"):
        eol = b"\n"
        line = line[:-1]
    else:
        eol = b""
    strip_chars = chars if chars is not None else None  # None = all whitespace
    return line.rstrip(strip_chars) + eol


def _fix_file(filename: str, chars: bytes | None) -> bool:
    """Fix trailing whitespace in file. Return True if file was modified."""
    try:
        with open(filename, "rb") as f:
            lines = f.readlines()
    except (PermissionError, IsADirectoryError):
        return False
    newlines = [_process_line(line, chars) for line in lines]
    if newlines != lines:
        with open(filename, "wb") as f:
            for line in newlines:
                f.write(line)
        return True
    return False


def main() -> int:
    retval = 0
    for filename in sys.argv[1:]:
        if _fix_file(filename, chars=None):
            print(f"Fixing {filename}")
            retval = 1
    return retval


if __name__ == "__main__":
    sys.exit(main())
