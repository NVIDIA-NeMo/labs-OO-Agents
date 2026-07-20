# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capability agent: constructing complex / non-serializable return types in CodeAct.

Two CodeAct methods exercise the construction-guidance the framework gives the model
for return types that have no JSON schema (so they fall back to an ``Any`` tool schema):

- ``build_table -> pd.DataFrame`` — pandas has a curated ``spec.define_doc`` adapter, so
  the model gets a concise construction hint.
- ``build_widget -> Widget`` — a **novel, non-Pydantic** class with **no** ``define_doc``
  adapter and a deliberately non-guessable constructor (private ``__init__``; build via
  ``Widget.of(...)``). The model can only get this right from the **auto-doc fallback**
  that folds ``doc(Widget)`` into the ``return_result`` tool — so a pass here proves that
  fallback actually conveys construction info (the model has no training prior for Widget).

Scored by ``ConstructionScorer`` below, which inspects the *live* returned object.
"""

from __future__ import annotations

import pandas as pd

from nooa import Agent


class Widget:
    """An immutable widget. Do NOT call ``Widget(...)`` directly — that raises.

    Construct one with the factory:

        Widget.of(label="alpha", capacity=5)

    ``label`` is the widget's name and ``capacity`` is its integer size.
    """

    def __init__(self, label: str, capacity: int, _via_factory: bool = False) -> None:
        if not _via_factory:
            raise RuntimeError(
                "Widget() is private — construct with Widget.of(label=..., capacity=...)."
            )
        self.label = label
        self.capacity = capacity

    @classmethod
    def of(cls, label: str, capacity: int) -> Widget:
        """Build a Widget. ``label`` = name, ``capacity`` = integer size."""
        return cls(label, capacity, _via_factory=True)

    def __repr__(self) -> str:
        return f"Widget(label={self.label!r}, capacity={self.capacity})"


class ConstructionAgent(Agent):
    """You build and return data objects exactly as instructed."""

    async def build_table(self, rows: list[dict]) -> pd.DataFrame:
        """Build a pandas DataFrame from {rows} (a list of row dicts) and return it."""
        ...

    async def build_widget(self, name: str, size: int) -> Widget:
        """Build a Widget whose name is {name} and whose size is {size}, and return it."""
        ...


class ConstructionScorer:
    """Score a constructed object: right type AND right contents.

    Inspects the live returned object (``ctx.actual``):
    - DataFrame → compare ``to_dict(orient='records')`` to the expected rows.
    - Widget    → compare (label, capacity) to the expected mapping.
    Uses duck-typing on the class name to be robust across import paths.
    """

    def score(self, ctx):
        from eval_pipeline.models import ScoreResult

        actual = ctx.actual
        expected = ctx.expected
        tname = type(actual).__name__

        if isinstance(actual, pd.DataFrame):
            got = actual.to_dict(orient="records")
            ok = got == expected
            return ScoreResult(
                score=1.0 if ok else 0.0,
                reasoning=f"DataFrame records {'==' if ok else '!='} expected: got={got}",
            )

        if tname == "Widget":
            ok = getattr(actual, "label", None) == expected.get("name") and getattr(
                actual, "capacity", None
            ) == expected.get("size")
            return ScoreResult(
                score=1.0 if ok else 0.0,
                reasoning=(
                    f"Widget(label={getattr(actual, 'label', None)!r}, "
                    f"capacity={getattr(actual, 'capacity', None)!r}) "
                    f"{'matches' if ok else 'does not match'} expected {expected}"
                ),
            )

        return ScoreResult(
            score=0.0,
            reasoning=f"Unexpected result type {tname!r} (value={actual!r}); expected a constructed object.",
        )


__all__ = ["ConstructionAgent", "ConstructionScorer", "Widget"]
