# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``CodeActConfig.prefill`` — control or disable the auto turn-1 prefill.

The config field accepts:

  * default — ``InspectInputsPrefill()`` auto-renders every parameter via
    pformat (the slice-keys / items markers from truncation 3.0).
  * ``None`` — disable prefill entirely. Pre-ellipsis code still runs.
  * a custom ``Prefill`` instance — implement the single-method protocol
    ``get_code(call, config=None) -> str | None`` and pass it to render
    whatever turn-1 setup the task needs.
"""

from __future__ import annotations

from nooa.config.strategy_config import CodeActConfig
from nooa.strategies.prefill import InspectInputsPrefill


class TestPrefillConfigDefault:
    def test_default_is_inspect_inputs_prefill(self):
        """Default is ``InspectInputsPrefill()`` — existing agents see no behavior change."""
        cfg = CodeActConfig()
        assert isinstance(cfg.prefill, InspectInputsPrefill)


class TestPrefillConfigDisable:
    def test_none_disables_prefill(self):
        """Authors set ``prefill=None`` to skip the auto step (e.g. when they
        write their own setup in pre-ellipsis code)."""
        cfg = CodeActConfig(prefill=None)
        assert cfg.prefill is None

    def test_merge_preserves_none(self):
        """``merge_with`` carries ``prefill=None`` forward when the override
        explicitly disables prefill."""
        merged = CodeActConfig().merge_with(CodeActConfig(prefill=None))
        assert merged.prefill is None


class TestPrefillConfigCustom:
    """Custom Prefill plugins are accepted as long as they satisfy the
    ``Prefill`` protocol (a single ``get_code`` method)."""

    def test_accepts_custom_prefill_instance(self):
        """Any object with the protocol shape is accepted — Pydantic uses
        ``arbitrary_types_allowed`` since ``Prefill`` is a structural Protocol."""

        class DataFramePrefill:
            """Custom prefill that prints describe()/head() instead of dumping the frame."""

            def get_code(self, call, config=None) -> str | None:
                return "print('custom prefill output')"

        prefill = DataFramePrefill()
        cfg = CodeActConfig(prefill=prefill)
        assert cfg.prefill is prefill

    def test_merge_can_swap_prefill(self):
        """``merge_with`` propagates a custom prefill from the override."""

        class CustomPrefill:
            def get_code(self, call, config=None) -> str | None:
                return None

        custom = CustomPrefill()
        merged = CodeActConfig().merge_with(CodeActConfig(prefill=custom))
        assert merged.prefill is custom


class TestStrategyPromptOverrideUsesExistingAPI:
    """Strategy-prompt overrides go through the existing decorator API.

    There's no new config field for this — agents override the strategy
    prompt via ``@strategy(..., ScopedContext(context={"strategy_prompt": ...}))``.
    The context-builder pipeline applies the decorator-context phase
    *after* strategy overrides, so the agent's value wins. ``self.context``
    (the persistent-blocks phase) deliberately runs *before* strategy
    overrides — that's how strategies declare their default instructions
    without each agent having to clear them first.
    """

    def test_decorator_context_attaches_strategy_prompt_override(self):
        """Regression: ``@strategy(..., ScopedContext(context={...}))`` stashes the
        override dict on the wrapped function so the context-builder pipeline can
        apply it during phase 3 (decorator context). This is the documented path
        for replacing the default ``strategy_prompt`` block — guard it against
        accidental removal of the metadata-attachment behaviour."""
        from nooa import Agent
        from nooa.context_blocks import ScopedContext
        from nooa.decorators import strategy
        from nooa.strategies import CodeActStrategy

        class MyAgent(Agent):
            """Test agent."""

            @strategy(
                CodeActStrategy(),
                ScopedContext(context={"strategy_prompt": "domain-specific"}),
            )
            async def task(self, x: int) -> int:
                """Do task."""
                ...

        # The decorator stashes the context override on the function.
        ctx = MyAgent.task._strategy_context
        assert ctx == {"strategy_prompt": "domain-specific"}
