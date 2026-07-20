# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-parameter ``Annotated[T, spec(max_length=...)]`` truncation overrides.

Lets agent authors override the framework's default truncation knobs on a
per-parameter basis without changing the agent-level TruncationConfig:

    async def analyze(
        self,
        short: Annotated[list, spec(max_length=3)],
        full:  Annotated[list, spec(max_length=20)],
        plain: list,                                     # uses default
    ) -> None: ...

The override is plumbed through ``CurrentCall.param_specs`` and applied by
``format_parameters_as_code`` when rendering each parameter's value.
"""

from __future__ import annotations

from typing import Annotated

from nooa.agentdoc import spec
from nooa.config.truncation_config import FormatConfig, TruncationConfig
from nooa.strategies.current_call import CurrentCall


class _Demo:
    async def analyze(
        self,
        short: Annotated[list, spec(max_length=3)],
        full: Annotated[list, spec(max_length=20)],
        plain: list,
    ) -> None: ...


class _StringDemo:
    async def render(
        self,
        short_str: Annotated[str, spec(max_string=20)],
        plain: str,
    ) -> None: ...


def _call(method, kwargs):
    return CurrentCall.from_method(method, args=(), kwargs=kwargs)


def _tc(max_length=10, max_string=200, max_depth=4):
    return TruncationConfig(
        prefill_format=FormatConfig(
            max_length=max_length, max_string=max_string, max_depth=max_depth
        )
    )


class TestParamSpecExtraction:
    def test_extracts_max_length_per_param(self):
        call = _call(_Demo.analyze, {"short": [], "full": [], "plain": []})
        assert call.param_specs == {
            "short": {"max_length": 3},
            "full": {"max_length": 20},
        }

    def test_param_without_annotated_spec_has_no_entry(self):
        call = _call(_Demo.analyze, {"short": [], "full": [], "plain": []})
        assert "plain" not in call.param_specs


class TestFormatParametersAsCodeUsesOverride:
    def test_short_param_truncates_tighter_than_default(self):
        data = list(range(100))
        call = _call(_Demo.analyze, {"short": data, "full": data, "plain": data})
        out = call.format_parameters_as_code(tc=_tc(max_length=10))

        # max_length=3 → ceiling(3/2)=2 head + 1 tail
        assert "short = list(len=100, [:2]=[0, 1], [-1:]=[99])" in out
        # max_length=20 → 10 head + 10 tail
        assert (
            "full = list(len=100, [:10]=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [-10:]=[90, 91, 92, 93, 94, 95, 96, 97, 98, 99])"
            in out
        )
        # no override → tc default (10 → 5+5)
        assert "plain = list(len=100, [:5]=[0, 1, 2, 3, 4], [-5:]=[95, 96, 97, 98, 99])" in out

    def test_max_string_override(self):
        """Per-param max_string override produces a tighter str(len=N, ...) marker
        than the agent-level default. Both params truncate but at different
        widths — the marker's `[:H]=...` slice key reflects each param's H."""
        long = "x" * 1000
        call = _call(_StringDemo.render, {"short_str": long, "plain": long})
        out = call.format_parameters_as_code(tc=_tc(max_string=200))

        # short_str: max_string=20 → [:10]/[-10]
        assert "short_str = str(len=1000, [:10]=" in out
        # plain: tc default max_string=200 → [:100]/[-100]
        assert "plain = str(len=1000, [:100]=" in out

    def test_explicit_value_formatter_still_wins(self):
        # When the caller passes an explicit value_formatter, per-param specs
        # are ignored — the caller knows what they're doing.
        data = list(range(100))
        call = _call(_Demo.analyze, {"short": data, "full": data, "plain": data})
        out = call.format_parameters_as_code(value_formatter=lambda v: f"<len={len(v)}>")

        assert out.count("<len=100>") == 3
        assert "list(len=" not in out


class TestSpecAcceptsTruncationKnobs:
    def test_spec_accepts_max_length(self):
        ann = spec(max_length=20)
        assert ann is not None
        assert ann.kwargs == {"max_length": 20}

    def test_spec_accepts_max_string(self):
        ann = spec(max_string=500)
        assert ann is not None
        assert ann.kwargs == {"max_string": 500}

    def test_spec_accepts_max_depth(self):
        ann = spec(max_depth=6)
        assert ann is not None
        assert ann.kwargs == {"max_depth": 6}

    def test_spec_combines_with_existing_knobs(self):
        ann = spec(hidden=True, max_length=20, description="foo")
        assert ann is not None
        assert ann.kwargs == {"hidden": True, "max_length": 20, "description": "foo"}


class _NullStringDemo:
    async def process(
        self,
        content: Annotated[str, spec(max_string=None)],
        summary: str,
    ) -> None: ...


class TestInspectInputsPrefillHonorsParamSpecs:
    """InspectInputsPrefill.get_code() should emit per-parameter truncation
    overrides from Annotated[T, spec(...)] metadata rather than using the
    same config-level defaults for all parameters."""

    def test_max_string_none_override_emits_none_literal(self):
        """spec(max_string=None) should produce ``pprint(content, ..., max_string=None, ...)``
        so the parameter is NOT truncated — this is the reported bug."""
        from nooa.strategies.prefill import InspectInputsPrefill

        call = _call(_NullStringDemo.process, {"content": "x" * 1000, "summary": "short"})
        prefill = InspectInputsPrefill()
        code = prefill.get_code(call, config=_tc(max_string=500))

        # content should use max_string=None (from spec override)
        assert "max_string=None" in code
        # summary should use the config default (500)
        assert "max_string=500" in code

    def test_max_length_override_in_prefill(self):
        """spec(max_length=3) should override the config default in the prefill code."""
        from nooa.strategies.prefill import InspectInputsPrefill

        call = _call(_Demo.analyze, {"short": [1, 2, 3], "full": [4, 5, 6], "plain": [7, 8, 9]})
        prefill = InspectInputsPrefill()
        code = prefill.get_code(call, config=_tc(max_length=10))

        # short has spec(max_length=3) override
        assert "pprint(short, max_length=3," in code
        # full has spec(max_length=20) override
        assert "pprint(full, max_length=20," in code
        # plain has no override → config default (10)
        assert "pprint(plain, max_length=10," in code
