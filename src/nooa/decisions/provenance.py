# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compact, deterministic provenance for logical decision questions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nooa.decisions.client import DecisionQuestion


LLM_FALLBACK_SCHEMA_VERSION = "decide-predict-v1"


def question_digest(questions: Mapping[str, DecisionQuestion]) -> str:
    """Hash normalized question names, instructions, criteria, and candidate IDs."""
    canonical = json.dumps(
        {name: asdict(question) for name, question in questions.items()},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = ["LLM_FALLBACK_SCHEMA_VERSION", "question_digest"]
