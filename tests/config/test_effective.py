# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``nooa.config.effective(Model)``: method > agent instance > class > the model's defaults."""

import asyncio
import contextvars
import re
import threading
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from nooa import Agent, CodeActStrategy, strategy
from nooa.config import CaptureConfig, TruncationConfig, effective, merge_config
from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG
from nooa.unifiedllm import FakeLLMClient


def _seen() -> int:
    return effective(TruncationConfig).capture.max_stdout


def _agent(*, cls_stdout=None, inst_stdout=None, method_stdout=None):
    """An agent whose ``implemented()`` and generated ``generated(probe)`` report ``max_stdout``."""
    llm = FakeLLMClient.with_tool_call(
        "execute_python", {"code": "return_result(probe())"}, strict_exhaustion=True
    )
    cls_kwargs = {}
    if cls_stdout is not None:
        cls_kwargs["truncation"] = TruncationConfig(capture=CaptureConfig(max_stdout=cls_stdout))
    method_kwargs = {}
    if method_stdout is not None:
        method_kwargs["truncation"] = TruncationConfig(
            capture=CaptureConfig(max_stdout=method_stdout)
        )

    class Probe(Agent, llm=llm, **cls_kwargs):
        async def implemented(self) -> int:
            return _seen()

        @strategy(CodeActStrategy(), **method_kwargs)
        async def generated(self, probe: object) -> int:
            """Call probe() and return its result."""
            ...

    inst_kwargs = {}
    if inst_stdout is not None:
        inst_kwargs["truncation"] = TruncationConfig(capture=CaptureConfig(max_stdout=inst_stdout))
    return Probe(**inst_kwargs)


def test_outside_any_agent_it_is_the_defaults():
    assert effective(TruncationConfig) is DEFAULT_TRUNCATION_CONFIG


async def test_no_override_is_the_defaults_in_agent_code():
    agent = _agent()
    try:
        assert await agent.implemented() == 50_000
        assert await agent.generated(_seen) == 50_000
    finally:
        await agent.aclose()


async def test_class_level_beats_the_defaults():
    agent = _agent(cls_stdout=111)
    try:
        assert await agent.implemented() == 111
        assert await agent.generated(_seen) == 111
    finally:
        await agent.aclose()


async def test_instance_level_beats_the_class():
    agent = _agent(cls_stdout=111, inst_stdout=222)
    try:
        assert await agent.implemented() == 222
        assert await agent.generated(_seen) == 222
    finally:
        await agent.aclose()


async def test_method_level_beats_the_instance_only_inside_that_method():
    agent = _agent(cls_stdout=111, inst_stdout=222, method_stdout=333)
    try:
        assert await agent.generated(_seen) == 333
        assert await agent.implemented() == 222
        assert effective(TruncationConfig) is DEFAULT_TRUNCATION_CONFIG
    finally:
        await agent.aclose()


async def test_an_explicit_agent_is_its_config_and_the_runtime_agrees():
    agent = _agent(inst_stdout=222)
    try:
        assert effective(TruncationConfig, agent).capture.max_stdout == 222
        assert agent.runtime.truncation_config is effective(TruncationConfig, agent)
    finally:
        await agent.aclose()


class _Threads(Agent, llm=FakeLLMClient([])):
    async def threads(self) -> dict[str, int]:
        seen: dict[str, int] = {}

        def record(key: str) -> None:
            seen[key] = _seen()

        plain = threading.Thread(target=record, args=("plain",))
        copied = threading.Thread(target=contextvars.copy_context().run, args=(record, "copied"))
        for thread in (plain, copied):
            thread.start()
            await asyncio.to_thread(thread.join)
        await asyncio.to_thread(record, "to_thread")
        return seen


async def test_plain_threads_need_copy_context():
    agent = _Threads(truncation=TruncationConfig(capture=CaptureConfig(max_stdout=222)))
    try:
        seen = await agent.threads()
    finally:
        await agent.aclose()
    assert seen == {"plain": 50_000, "copied": 222, "to_thread": 222}


# Any other config model: the same rule, through configs=[...]


class _Sub(BaseModel):
    model_config = ConfigDict(frozen=True)

    x: int = 1
    y: int = 2


class _Limits(BaseModel):
    """A stand-in for a package's own config model (core never names one)."""

    model_config = ConfigDict(frozen=True)

    a: int = 10
    b: int = 20
    sub: _Sub = _Sub()


def _limits() -> tuple[int, int]:
    limits = effective(_Limits)
    return limits.a, limits.b


def _limits_agent(*, cls=None, inst=None, method=None):
    llm = FakeLLMClient.with_tool_call(
        "execute_python", {"code": "return_result(probe())"}, strict_exhaustion=True
    )
    cls_kwargs: dict[str, Any] = {} if cls is None else {"configs": [cls]}
    method_kwargs: dict[str, Any] = {} if method is None else {"configs": [method]}
    inst_kwargs: dict[str, Any] = {} if inst is None else {"configs": [inst]}

    class Probe(Agent, llm=llm, **cls_kwargs):
        async def implemented(self) -> tuple[int, int]:
            return _limits()

        @strategy(CodeActStrategy(), **method_kwargs)
        async def generated(self, probe: object) -> tuple[int, int]:
            """Call probe() and return its result."""
            ...

    return Probe(**inst_kwargs)


def test_a_package_model_outside_any_agent_is_its_defaults():
    assert effective(_Limits) == _Limits()
    assert effective(_Limits) is effective(_Limits)  # one defaults instance


async def test_a_package_model_follows_class_then_instance_then_method():
    agent = _limits_agent(cls=_Limits(a=11), inst=_Limits(b=22), method=_Limits(a=33))
    try:
        assert await agent.implemented() == (11, 22)
        assert effective(_Limits, agent) == _Limits(a=11, b=22)
        assert await agent.generated(_limits) == (33, 22)
        assert await agent.implemented() == (11, 22)
        assert effective(_Limits) == _Limits()
    finally:
        await agent.aclose()


async def test_a_package_model_unset_on_an_agent_is_its_defaults_inside_it():
    agent = _limits_agent(method=_Limits(a=33))
    try:
        assert await agent.implemented() == (10, 20)
        assert await agent.generated(_limits) == (33, 20)
    finally:
        await agent.aclose()


async def test_a_package_model_does_not_change_the_truncation_config():
    agent = _limits_agent(cls=_Limits(a=11))
    try:
        assert effective(TruncationConfig, agent) is DEFAULT_TRUNCATION_CONFIG
    finally:
        await agent.aclose()


def test_sub_models_merge_field_by_field():
    base = _Limits(a=11, sub=_Sub(x=5))
    merged = merge_config(base, _Limits(sub=_Sub(y=7)))
    assert merged == _Limits(a=11, sub=_Sub(x=5, y=7))
    assert merge_config(base, None) is base
    with pytest.raises(ValueError, match="model_fields_set"):
        merge_config(base, _Limits.model_validate({}))


def test_truncation_merge_with_is_the_generic_merge():
    base = TruncationConfig(capture=CaptureConfig(max_stdout=1_000, max_stderr=500))
    other = TruncationConfig(capture=CaptureConfig(max_stderr=7), placeholder_chars=9)
    assert base.merge_with(other) == merge_config(base, other)
    assert base.merge_with(other).capture == CaptureConfig(max_stdout=1_000, max_stderr=7)


class _OtherSub(BaseModel):
    model_config = ConfigDict(frozen=True)

    z: int = 3


class _Union(BaseModel):
    model_config = ConfigDict(frozen=True)

    sub: _Sub | _OtherSub = _Sub()


def test_a_sub_model_of_another_type_replaces_the_base_value():
    # #580 review: _merge recursed whenever both values were models, so a
    # union field holding another model type raised AttributeError.
    merged = merge_config(_Union(sub=_Sub(x=5)), _Union(sub=_OtherSub(z=9)))
    assert merged == _Union(sub=_OtherSub(z=9))
    # The same type still merges field by field.
    assert merge_config(_Union(sub=_Sub(x=5)), _Union(sub=_Sub(y=7))) == _Union(sub=_Sub(x=5, y=7))


def test_an_invalid_merge_raises_and_a_valid_one_keeps_its_explicit_fields():
    # #580 review: model_copy() skipped validation, so a merge could violate a
    # cross-field rule (a tail as large as a stream's whole budget).
    base = TruncationConfig(capture=CaptureConfig(tail=1_500))
    with pytest.raises(ValueError, match="capture.tail"):
        base.merge_with(TruncationConfig(capture=CaptureConfig(max_stdout=1_000)))
    merged = base.merge_with(TruncationConfig(capture=CaptureConfig(max_stdout=2_000)))
    assert (merged.capture.max_stdout, merged.capture.tail) == (2_000, 1_500)
    # The result is the model_copy(): it can still be merged as an override.
    assert "capture" in merged.model_fields_set
    assert merge_config(TruncationConfig(), merged).capture.max_stdout == 2_000


async def test_the_state_block_expression_ignores_a_method_level_override():
    # #580 review: the expression read effective(TruncationConfig, self), which
    # returns a running method's override first, so an agent built inside a
    # parent's generated method embedded the parent method's state_format.
    from nooa.config import FormatConfig
    from nooa.config.effective import _current_truncation_config_var

    expected = "pformat(self, max_length=50, max_string=500, max_depth=4)"
    method = DEFAULT_TRUNCATION_CONFIG.merge_with(
        TruncationConfig(state_format=FormatConfig(max_length=7, max_string=8, max_depth=1))
    )
    token = _current_truncation_config_var.set(method)
    try:
        agent = Agent(llm=FakeLLMClient([]))  # built while the method runs
        assert agent._state_block_expr() == expected
    finally:
        _current_truncation_config_var.reset(token)
    try:
        assert agent._state_block_expr() == expected
    finally:
        await agent.aclose()


async def test_a_subclass_overrides_only_the_fields_it_sets():
    class Other(BaseModel):
        model_config = ConfigDict(frozen=True)
        c: int = 0

    class Base(Agent, llm=FakeLLMClient([]), configs=[_Limits(a=11), Other(c=1)]):
        pass

    class Child(Base, configs=[_Limits(b=22)]):
        pass

    agent = Child()
    try:
        limits = effective(_Limits, agent)
        assert (limits.a, limits.b) == (11, 22)  # merged over the base class
        assert effective(Other, agent) == Other(c=1)  # inherited
    finally:
        await agent.aclose()


def test_configs_are_checked():
    with pytest.raises(TypeError, match="truncation="):
        _limits_agent(inst=TruncationConfig(placeholder_chars=5))
    with pytest.raises(ValueError, match="twice"):

        class Twice(Agent, llm=FakeLLMClient([]), configs=[_Limits(a=1), _Limits(b=2)]):
            pass

    with pytest.raises(TypeError, match="config model instances"):

        @strategy(CodeActStrategy(), configs=[object()])  # type: ignore[list-item]
        async def method(self) -> int:
            """Return 1."""
            ...


def test_core_does_not_import_package_configs():
    """Core (src/nooa) never names a package: nooa_atom / nooa_cli stay out of it."""
    import pathlib

    import nooa

    root = pathlib.Path(nooa.__file__).parent
    offenders = [
        f"{path.relative_to(root)}:{number}"
        for path in root.rglob("*.py")
        if "tests" not in path.relative_to(root).parts
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.match(r"\s*(from|import)\s+(nooa_atom|nooa_cli)\b", line)
        or "AtomLimits" in line
        or "CodingLimits" in line
    ]
    assert not offenders, offenders


# A subclass's truncation= / configs= are partial overrides over its base classes'.


class _BaseAgent(
    Agent,
    llm=FakeLLMClient([]),
    truncation=TruncationConfig(placeholder_chars=10_000),
    configs=[_Limits(a=1)],
):
    pass


class _ChildAgent(
    _BaseAgent,
    truncation=TruncationConfig(capture=CaptureConfig(max_shell_output=123)),
    configs=[_Limits(b=2)],
):
    pass


class _GrandchildAgent(_ChildAgent, configs=[_Limits(a=3)]):
    pass


def test_a_subclass_truncation_merges_over_its_base_classes():
    config = effective(TruncationConfig, _ChildAgent())
    assert config.placeholder_chars == 10_000  # from the base class
    assert config.capture.max_shell_output == 123  # from the subclass
    assert config.capture.max_stdout == DEFAULT_TRUNCATION_CONFIG.capture.max_stdout


def test_a_subclass_configs_merge_over_its_base_classes():
    child = effective(_Limits, _ChildAgent())
    assert (child.a, child.b) == (1, 2)
    grandchild = effective(_Limits, _GrandchildAgent())
    assert (grandchild.a, grandchild.b) == (3, 2)  # nearest class wins per field
    base = effective(_Limits, _BaseAgent())
    assert (base.a, base.b) == (1, 20)  # a subclass never changes its base


def test_an_instance_merges_over_the_merged_classes():
    agent = _ChildAgent(configs=[_Limits(a=7)], truncation=TruncationConfig(placeholder_chars=5))
    limits = effective(_Limits, agent)
    assert (limits.a, limits.b) == (7, 2)
    config = effective(TruncationConfig, agent)
    assert (config.placeholder_chars, config.capture.max_shell_output) == (5, 123)
