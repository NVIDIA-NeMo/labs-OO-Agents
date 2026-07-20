# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard against dynamic method addition to Agent classes and instances.

The guard is **unconditional**: attaching a method-shaped callable to any
Agent instance or class raises ``DynamicMethodAdditionError`` regardless of
whether we're inside ``execute_code()``, a test, or anywhere else. Class
construction goes through ``type.__setattr__`` in ``AgentMeta.__new__`` and
therefore bypasses the guard.

LLM-defined helpers live as REPL-local callables (captured via
``__repl_captured_locals__``) and are invoked as ``helper(self, x)`` — never
``self.helper(x)`` — since nothing attaches them to the agent.
"""

import functools

import pytest

from nooa import Agent
from nooa.errors import DynamicMethodAdditionError
from nooa.unifiedllm import FakeLLMClient

_TEST_LLM = FakeLLMClient()


class _A(Agent, llm=_TEST_LLM):
    pass


# ---------------------------------------------------------------------------
# Unit-level: Agent.__setattr__ unconditionally rejects method-like values
# ---------------------------------------------------------------------------


def test_setattr_lambda_rejected():
    """`self.foo = lambda ...` raises DynamicMethodAdditionError anywhere, anytime."""
    a = _A()
    with pytest.raises(DynamicMethodAdditionError, match="Cannot attach callable"):
        a.new_tool = lambda self, x: x


def test_setattr_function_rejected():
    """Assigning a plain function object to an agent attribute is rejected."""
    a = _A()

    def helper(self, x):
        return x + 1

    with pytest.raises(DynamicMethodAdditionError):
        a.helper = helper


def test_setattr_via_builtin_setattr_rejected():
    """Going through the `setattr()` builtin is rejected the same way."""
    a = _A()
    with pytest.raises(DynamicMethodAdditionError):
        setattr(a, "bad", lambda self: None)  # noqa: B010


def test_setattr_partial_of_function_rejected():
    """`functools.partial(fn, ...)` wrapping a function is also method-like."""
    a = _A()

    def base(self, x):
        return x

    with pytest.raises(DynamicMethodAdditionError):
        a.partial_tool = functools.partial(base, a)


def test_setattr_staticmethod_rejected():
    """A staticmethod wrapper counts as a method-like value."""
    a = _A()
    with pytest.raises(DynamicMethodAdditionError):
        a.s = staticmethod(lambda: 1)


def test_setattr_classmethod_rejected():
    """A classmethod wrapper counts as a method-like value."""
    a = _A()
    with pytest.raises(DynamicMethodAdditionError):
        a.c = classmethod(lambda cls: 1)


def test_setattr_bound_method_rejected():
    """A bound method (from another instance) is method-shaped and rejected."""

    class Source:
        def fn(self, x):
            return x + 1

    src = Source()
    a = _A()
    with pytest.raises(DynamicMethodAdditionError):
        a.borrowed = src.fn  # bound method — inspect.ismethod(src.fn) is True


def test_setattr_private_callable_allowed():
    """Underscore-prefixed names are exempt — they hold framework internals."""
    a = _A()
    # Framework patterns: unsubscribe handlers, cancellation tokens, cached
    # bound methods, etc. all live on `_private` attrs.
    a._handler = lambda evt: None
    a._cleanup = lambda: None
    assert callable(a._handler)
    assert callable(a._cleanup)


def test_setattr_data_attributes_are_allowed():
    """Plain data attributes still flow through — ints, dicts, lists, etc."""
    a = _A()
    a.number = 42
    a.payload = {"a": 1}
    a.items = [1, 2, 3]
    assert a.number == 42
    assert a.payload == {"a": 1}
    assert a.items == [1, 2, 3]


def test_setattr_tool_instance_allowed():
    """Tool-like objects with ``__call__`` are NOT method-shaped — they pass."""
    a = _A()

    class Tool:
        def __call__(self, x):
            return x

    tool = Tool()
    a.my_tool = tool
    assert a.my_tool is tool


def test_setattr_class_attribute_allowed():
    """Assigning a class (e.g. a child-agent class) is allowed — classes aren't methods."""
    a = _A()

    class Child:
        pass

    a.child_cls = Child
    assert a.child_cls is Child


# ---------------------------------------------------------------------------
# AgentMeta.__setattr__ guard (class mutation from anywhere)
# ---------------------------------------------------------------------------


def test_class_mutation_callable_rejected():
    """`type(self).foo = fn` / `AgentClass.foo = fn` is rejected unconditionally."""

    class B(Agent, llm=_TEST_LLM):
        pass

    with pytest.raises(DynamicMethodAdditionError):
        B.injected = lambda self: 1


def test_class_mutation_data_still_allowed():
    """Class-level data attributes still work (common pattern for CATEGORIES etc.)."""

    class B(Agent, llm=_TEST_LLM):
        pass

    B.flag = True
    B.labels = ["a", "b"]
    assert B.flag is True
    assert B.labels == ["a", "b"]


def test_class_construction_metaclass_bypass_works():
    """Metaclass __new__ wraps async generation/traced methods via type.__setattr__.

    Regression: if AgentMeta.__setattr__ weren't bypassed during class creation,
    the wrapping itself (setattr of an async function onto the class) would
    trip the guard and break every Agent subclass.
    """

    class B(Agent, llm=_TEST_LLM):
        async def do(self) -> str:
            """A traced public method."""
            return "ok"

    # If class construction works, `do` is on the class, is async, and callable.
    assert callable(B.do)


# ---------------------------------------------------------------------------
# Integration: LLM-generated helpers live as REPL locals, not on the agent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_code_blocks_self_method_assignment():
    """LLM-generated code doing `self.foo = lambda ...` surfaces the guard error."""

    class A(Agent, llm=_TEST_LLM):
        pass

    agent = A()
    result = await agent.runtime.execute_code(
        "self.bad = lambda x: x + 1",
        wrap_in_function=True,
        validate=False,
    )
    assert result.error is not None, "expected guard to raise"
    assert isinstance(result.error, DynamicMethodAdditionError)
    assert not hasattr(agent, "bad")


@pytest.mark.asyncio
async def test_execute_code_blocks_class_mutation():
    """LLM-generated code doing `type(self).foo = fn` is blocked."""

    class A(Agent, llm=_TEST_LLM):
        pass

    agent = A()
    result = await agent.runtime.execute_code(
        "type(self).injected = lambda self: 1",
        wrap_in_function=True,
        validate=False,
    )
    assert isinstance(result.error, DynamicMethodAdditionError)
    assert not hasattr(A, "injected")


@pytest.mark.asyncio
async def test_execute_code_allows_data_attribute_assignment():
    """Regression: data attribute writes from exec'd code still succeed."""

    class A(Agent, llm=_TEST_LLM):
        pass

    agent = A()
    result = await agent.runtime.execute_code(
        "self.value = 123\nself.tags = ['a', 'b']",
        wrap_in_function=True,
        validate=False,
    )
    assert result.error is None, f"unexpected error: {result.error!r}"
    assert agent.value == 123
    assert agent.tags == ["a", "b"]


@pytest.mark.asyncio
async def test_repl_helper_is_callable_same_turn_via_standalone_name():
    """`def helper(self, x):` defined in a cell is callable as `helper(self, x)` in the same cell."""

    class A(Agent, llm=_TEST_LLM):
        pass

    agent = A()
    code = "def double(self, x):\n    return x * 2\nself.result = double(self, 21)\n"
    result = await agent.runtime.execute_code(code, wrap_in_function=True, validate=False)
    assert result.error is None, f"unexpected error: {result.error!r}"
    assert agent.result == 42


@pytest.mark.asyncio
async def test_execute_code_blocks_named_class_mutation():
    """`MyAgent.foo = fn` inside generated code is rejected end-to-end.

    The pre-existing ClassAssignmentValidator (code_validator.py) catches this
    at AST validation time; this regression test pins the full pipeline (parse →
    validate → reject) for the canonical "LLM references the agent class by
    name and mutates it" pattern. AgentMeta.__setattr__ is the runtime
    backstop if validation is ever disabled.
    """
    from nooa.errors import RestrictedCodeError

    class MyAgent(Agent, llm=_TEST_LLM):
        async def task(self) -> str:
            """Existing method."""
            return "ok"

    agent = MyAgent()
    result = await agent.runtime.execute_code(
        "MyAgent.injected = lambda self: 'patched'",
        builtins={"MyAgent": MyAgent},
        wrap_in_function=True,
        validate=True,
    )
    assert result.error is not None
    assert isinstance(result.error, RestrictedCodeError)
    assert not hasattr(MyAgent, "injected")


@pytest.mark.asyncio
async def test_execute_code_blocks_named_class_setattr_call():
    """`setattr(MyAgent, "foo", fn)` is rejected end-to-end."""
    from nooa.errors import RestrictedCodeError

    class MyAgent(Agent, llm=_TEST_LLM):
        pass

    agent = MyAgent()
    result = await agent.runtime.execute_code(
        "setattr(MyAgent, 'injected', lambda self: 1)",
        builtins={"MyAgent": MyAgent},
        wrap_in_function=True,
        validate=True,
    )
    assert result.error is not None
    assert isinstance(result.error, RestrictedCodeError)
    assert not hasattr(MyAgent, "injected")


@pytest.mark.asyncio
async def test_execute_code_blocks_aliased_class_mutation():
    """Aliasing the class via a local var doesn't escape the validator.

    The ClassAssignmentValidator tracks variables assigned from `type(self)`
    in `class_ref_vars`. This test pins that the alias is followed.
    """
    from nooa.errors import RestrictedCodeError

    class MyAgent(Agent, llm=_TEST_LLM):
        pass

    agent = MyAgent()
    result = await agent.runtime.execute_code(
        "cls = type(self)\ncls.injected = lambda self: 1",
        wrap_in_function=True,
        validate=True,
    )
    assert result.error is not None
    assert isinstance(result.error, RestrictedCodeError)
    assert not hasattr(MyAgent, "injected")


@pytest.mark.asyncio
async def test_execute_code_blocks_object_dunder_setattr_bypass():
    """`object.__setattr__(self, ...)` would bypass Agent.__setattr__; validator must catch it.

    Regression for the Codex adversarial review finding: runtime hooks don't
    fire on `object.__setattr__` (it goes through the C-level slot directly),
    so the validator is the only safety net.
    """
    from nooa.errors import RestrictedCodeError

    class A(Agent, llm=_TEST_LLM):
        pass

    agent = A()
    result = await agent.runtime.execute_code(
        "object.__setattr__(self, 'bad', lambda x: x + 1)",
        wrap_in_function=True,
        validate=True,
    )
    assert result.error is not None, "expected validator to reject"
    assert isinstance(result.error, RestrictedCodeError)
    assert "object.__setattr__" in str(result.error)
    assert not hasattr(agent, "bad")


@pytest.mark.asyncio
async def test_execute_code_blocks_type_dunder_setattr_bypass():
    """`type.__setattr__(type(self), ...)` would bypass AgentMeta.__setattr__."""
    from nooa.errors import RestrictedCodeError

    class A(Agent, llm=_TEST_LLM):
        pass

    agent = A()
    result = await agent.runtime.execute_code(
        "type.__setattr__(type(self), 'injected', lambda self: 1)",
        wrap_in_function=True,
        validate=True,
    )
    assert result.error is not None
    assert isinstance(result.error, RestrictedCodeError)
    assert "type.__setattr__" in str(result.error)
    assert not hasattr(A, "injected")


@pytest.mark.asyncio
async def test_execute_code_blocks_dict_subscript_bypass():
    """`self.__dict__['foo'] = lambda: ...` is rejected by the existing __dict__ guard."""
    from nooa.errors import RestrictedCodeError

    class A(Agent, llm=_TEST_LLM):
        pass

    agent = A()
    result = await agent.runtime.execute_code(
        "self.__dict__['bad'] = lambda: 1",
        wrap_in_function=True,
        validate=True,
    )
    assert result.error is not None
    assert isinstance(result.error, RestrictedCodeError)
    assert not hasattr(agent, "bad")


def test_validator_flags_object_dunder_setattr():
    """Unit-level: SecurityValidator catches object.__setattr__ as a forbidden pattern."""
    from nooa.runtime.code_validator import (
        UnifiedCodeValidator,
        ValidationContext,
    )

    code = "object.__setattr__(self, 'foo', lambda: 1)"

    class _A(Agent, llm=_TEST_LLM):
        pass

    a = _A()
    ctx = ValidationContext(agent=a)

    with pytest.raises(Exception, match=r"object\.__setattr__"):
        UnifiedCodeValidator().validate(code, ctx)


def test_validator_flags_super_dunder_setattr():
    """Unit-level: super(...).__setattr__ is rejected by E102.

    `super(SomeAgent, self).__setattr__("foo", fn)` would route to the parent
    class's __setattr__, bypassing Agent.__setattr__. Catch it at validation.
    """
    from nooa.runtime.code_validator import (
        UnifiedCodeValidator,
        ValidationContext,
    )

    code = "super(type(self), self).__setattr__('foo', lambda: 1)"

    class _A(Agent, llm=_TEST_LLM):
        pass

    a = _A()
    ctx = ValidationContext(agent=a)

    with pytest.raises(Exception, match=r"super\(\)\.__setattr__"):
        UnifiedCodeValidator().validate(code, ctx)


def test_validator_flags_type_dunder_setattr():
    """Unit-level: same for type.__setattr__."""
    from nooa.runtime.code_validator import (
        UnifiedCodeValidator,
        ValidationContext,
    )

    code = "type.__setattr__(type(self), 'foo', lambda self: 1)"

    class _A(Agent, llm=_TEST_LLM):
        pass

    a = _A()
    ctx = ValidationContext(agent=a)

    with pytest.raises(Exception, match=r"type\.__setattr__"):
        UnifiedCodeValidator().validate(code, ctx)


def test_validator_allows_type_as_a_call():
    """Regression: `type(x)` (calling type as a builtin) is fine, only attribute access on bare `type`/`object` is flagged."""
    from nooa.runtime.code_validator import (
        UnifiedCodeValidator,
        ValidationContext,
    )

    code = "cls = type(self)\nname = cls.__name__"

    class _A(Agent, llm=_TEST_LLM):
        pass

    a = _A()
    ctx = ValidationContext(agent=a)

    # __name__ on a derived class is not in the dangerous-dunder list and
    # `type(self)` is a regular function call, not attribute access.
    UnifiedCodeValidator().validate(code, ctx)


@pytest.mark.asyncio
async def test_repl_helper_not_attached_to_agent():
    """Helpers defined in a REPL cell are NOT attached to the agent class/instance."""

    class A(Agent, llm=_TEST_LLM):
        pass

    agent = A()
    code = "def helper(self, x):\n    return x\n"
    result = await agent.runtime.execute_code(code, wrap_in_function=True, validate=False)
    assert result.error is None
    assert not hasattr(agent, "helper")
    assert not hasattr(A, "helper")
