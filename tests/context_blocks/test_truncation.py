# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for context-block-level pass-through (block-level truncation removed).

Block-level head/tail squashing has been removed entirely — ``truncate_content``
and ``_truncate_blocks`` are gone. Per-value bounds come from ``cfg.context_blocks``
/ ``cfg.events`` at render time; whole-block eviction (L4) handles overflow at
the assembly step.
"""

from nooa.context_blocks.models import BlockMetadata, ResolvedBlock, Role


class TestResolvedBlockHoldsContentVerbatim:
    """A ResolvedBlock just stores its content; no automatic truncation."""

    def test_long_content_stored_unchanged(self):
        content = "z" * 100_000
        block = ResolvedBlock(
            key="big", content=content, role=Role.SYSTEM, metadata=BlockMetadata()
        )
        assert block.content == content
        assert len(block.content) == 100_000
