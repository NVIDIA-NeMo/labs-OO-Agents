# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workspace file references shared by coding host input adapters."""

import os
import re
from pathlib import Path

# Submit-time form of the mention: greedy so it grabs the whole path token.
# Trailing sentence punctuation is peeled back off the captured token below so
# "see @docs/file.md." still resolves to docs/file.md.
_MENTION_TOKEN = re.compile(r"(?:^|(?<=\s))@(\S+)")

# Punctuation that is almost always sentence-trailing rather than part of a
# real filename, stripped from the right of a captured mention before resolving.
_TRAILING_PUNCT = ".,;:!?)]}\"'"


def expand_mentions(text: str, *, base_dir: str | Path | None = None) -> str:
    """Expand inline ``@path`` mentions into Markdown links before sending.

    ``@docs/blah.md`` becomes ``[docs/blah.md](</abs/docs/blah.md>)`` so the
    agent receives an unambiguous absolute path while the user keeps the short
    form they typed. Only mentions resolving to an existing file/dir expand;
    emails (``a@b``), nonexistent paths, etc. are left untouched. Trailing
    sentence punctuation is peeled off before resolving. The link target is
    angle-bracketed so a path containing ``)`` can't break out of the link.
    """

    resolved_base = (
        Path(base_dir).expanduser().resolve() if isinstance(base_dir, (str, os.PathLike)) else None
    )

    def _sub(m: "re.Match") -> str:
        raw = m.group(1)
        # Peel trailing punctuation, but try the full token first so a real
        # filename that legitimately ends in such a char still resolves.
        stripped = raw.rstrip(_TRAILING_PUNCT)
        for candidate in (raw, stripped):
            if not candidate.strip("./"):
                continue
            p = Path(os.path.expanduser(candidate))
            resolved = p if p.is_absolute() or resolved_base is None else resolved_base / p
            if resolved.exists():
                trailer = raw[len(candidate) :]
                label = candidate.rstrip("/")
                return f"[{label}](<{resolved.resolve()}>){trailer}"
        return m.group(0)

    return _MENTION_TOKEN.sub(_sub, text)
