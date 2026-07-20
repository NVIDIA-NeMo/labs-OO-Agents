# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that pformat/pprint exclude fields marked with hidden."""

from typing import Annotated

from pydantic import BaseModel

from nooa.agentdoc import hidden, pformat, spec


def test_pprint_excludes_annotated_hidden_field():
    """Instance repr from pformat() must not include Annotated[T, hidden] fields."""

    class Model(BaseModel):
        public: str = "ok"
        secret: Annotated[str, hidden] = "hidden_value"

    instance = Model(public="a", secret="b")
    result = pformat(instance)

    assert "public" in result
    assert "secret" not in result
    assert "hidden_value" not in result


def test_pformat_plain_class_excludes_annotated_hidden():
    """pformat() on a plain class instance must not show Annotated[T, hidden] fields."""

    class Agent:
        name: str
        api_key: Annotated[str, hidden]

        def __init__(self, name: str, api_key: str) -> None:
            self.name = name
            self.api_key = api_key

    agent = Agent(name="myagent", api_key="secret123")
    result = pformat(agent)

    assert "myagent" in result
    assert "name" in result
    assert "api_key" not in result
    assert "secret123" not in result


def test_pformat_plain_class_excludes_spec_hidden():
    """pformat() on a plain class instance must not show spec(hidden=True) fields."""

    class Config:
        host: str = "localhost"
        password: str = ""

        def __init__(self, host: str, password: str) -> None:
            self.host = host
            self.password = password

    spec(Config, "password", hidden=True)

    cfg = Config(host="prod.example.com", password="hunter2")
    result = pformat(cfg)

    assert "host" in result
    assert "prod.example.com" in result
    assert "password" not in result
    assert "hunter2" not in result


def test_pformat_plain_class_excludes_underscore_prefix():
    """pformat() on a plain class instance must not show _-prefixed attributes."""

    class Worker:
        label: str = ""

        def __init__(self, label: str) -> None:
            self.label = label
            self._cache: dict = {}
            self._internal = 42

    worker = Worker(label="worker-1")
    result = pformat(worker)

    assert "label" in result
    assert "worker-1" in result
    assert "_cache" not in result
    assert "_internal" not in result


def test_pformat_plain_class_excludes_dynamic_underscore_attr():
    """pformat() must not show dynamically-set _-prefixed instance attrs."""

    class Processor:
        value: int = 0

        def __init__(self, value: int) -> None:
            self.value = value

    proc = Processor(value=7)
    proc._runtime_state = "active"  # type: ignore[attr-defined]  # dynamically set after construction

    result = pformat(proc)

    assert "value" in result
    assert "_runtime_state" not in result
    assert "active" not in result


def test_pformat_instance_mode_repr():
    """instance_mode='repr' (default) formats as ClassName(field=value, ...)."""

    class Agent:
        name: str = "agent"
        count: int = 0

        def __init__(self, name: str, count: int) -> None:
            self.name = name
            self.count = count

    result = pformat(Agent(name="prod", count=5))
    assert "Agent(" in result
    assert "prod" in result
    assert "5" in result


def test_pformat_instance_mode_type():
    """instance_mode='type' formats as full class structure with current values."""

    class Agent:
        name: str = "agent"
        count: int = 0

        def __init__(self, name: str, count: int) -> None:
            self.name = name
            self.count = count

    result = pformat(Agent(name="prod", count=5), instance_mode="type")
    # Should show class syntax, not repr
    assert "class Agent" in result or "name: str" in result
    assert "prod" in result


def test_pformat_excludes_callable_instance_attribute():
    """pformat() excludes instance attributes that are callables (e.g. assigned lambdas)."""

    class Worker:
        value: int

        def __init__(self, value: int) -> None:
            self.value = value
            self.callback = lambda x: x * 2  # type: ignore[attr-defined]

    result = pformat(Worker(value=42))
    assert "value" in result
    assert "42" in result
    assert "callback" not in result
    assert "lambda" not in result


def test_pformat_excludes_spec_hidden_true_annotation():
    """Annotated[T, spec(hidden=True)] must hide the field — not just Annotated[T, hidden]."""
    from nooa.agentdoc import doc, spec

    class Agent:
        label: str = "ok"
        api_key: Annotated[str, spec(hidden=True)] = ""

    agent = Agent()
    agent.api_key = "sk-secret"

    assert "api_key" not in pformat(agent)
    assert "sk-secret" not in pformat(agent)
    assert "api_key" not in doc(agent)
    assert "sk-secret" not in doc(agent)
    assert "label" in pformat(agent)


def test_pformat_type_mode_excludes_hidden_dict_attrs():
    """doc(instance) must not leak Annotated[T, hidden] fields via __dict__ extra-attr path."""
    from nooa.agentdoc import doc

    class Agent:
        label: str = "default"
        api_key: Annotated[str, hidden] = ""

    agent = Agent()
    agent.label = "prod"
    agent.api_key = "sk-secret"

    result = doc(agent)
    assert "label" in result
    assert "api_key" not in result
    assert "sk-secret" not in result


def test_pformat_slots_only_class_no_annotations():
    """pformat() on a __slots__ class without type annotations shows slot values."""

    class Point:
        __slots__ = ("x", "y")

        def __init__(self, x: int, y: int) -> None:
            self.x = x
            self.y = y

    result = pformat(Point(3, 4))
    assert "Point" in result
    assert "3" in result
    assert "4" in result


def test_pformat_does_not_trigger_class_descriptor_for_annotated_field():
    """Class-level descriptor lookup must be static and skipped, not executed."""

    class ExplodingDescriptor:
        def __init__(self) -> None:
            self.accessed = False

        def __get__(self, obj, objtype=None):
            self.accessed = True
            raise RuntimeError("descriptor should not be invoked by pformat")

    class Config:
        dangerous: str = ExplodingDescriptor()  # type: ignore[assignment]

    result = pformat(Config())

    descriptor = Config.__dict__["dangerous"]
    assert descriptor.accessed is False
    assert "dangerous" not in result


def test_pformat_single_string_slots_treated_as_one_slot_name():
    """A single-string __slots__ value is one slot name, not characters."""

    class SingleSlot:
        __slots__ = "value"
        value: int

        def __init__(self, value: int) -> None:
            self.value = value

    result = pformat(SingleSlot(42))

    assert "SingleSlot" in result
    assert "value" in result
    assert "42" in result
