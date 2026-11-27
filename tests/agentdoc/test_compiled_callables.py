# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""doc() for compiled callables and classes (C extensions, Boost.Python)."""

import math

from nooa.agentdoc import doc
from nooa.agentdoc._structured import is_compiled_callable
from nooa.agentdoc.markers import hidden

NOTE = "No docstring or signature is available for this compiled function."


def _make_boost_function_type() -> type:
    """Stand-in for ``Boost.Python.function``: no __doc__, no inspectable signature."""

    def _no_signature(self):
        raise ValueError("no signature found")

    def _call(self, *args, **kwargs):
        raise AssertionError("doc() must not call a compiled function")

    return type(
        "function",
        (),
        {
            "__module__": "Boost.Python",
            "__doc__": None,
            "__call__": _call,
            "__signature__": property(_no_signature),
        },
    )


BoostFunction = _make_boost_function_type()


def _boost_fn(name: str, module: str = "Mesh") -> object:
    fn = BoostFunction()
    fn.__name__ = name
    fn.__module__ = module
    return fn


class _BoostClassMeta(type):
    """Stands in for Boost.Python's metaclass (``Boost.Python.class``)."""


_BoostClassMeta.__module__ = "Boost.Python"

FakeCompiled = _BoostClassMeta(
    "FakeCompiled",
    (),
    {
        "__module__": "fake_ext",
        "GetPoints": _boost_fn("GetPoints"),
        "Define": staticmethod(_boost_fn("Define")),
        "_GetStaticType": staticmethod(_boost_fn("_GetStaticType")),
        "Hypot": staticmethod(math.hypot),
        "Join": str.join,
    },
)


def test_stand_ins_are_compiled_callables():
    assert is_compiled_callable(math.hypot)
    assert is_compiled_callable(str.join)
    assert is_compiled_callable(_boost_fn("GetPoints"))
    assert not is_compiled_callable(test_stand_ins_are_compiled_callables)
    assert not is_compiled_callable(BoostFunction)


def test_builtin_with_text_signature_renders_as_function():
    assert doc(len).splitlines()[0] == "def len(obj):"


def test_builtin_without_signature_uses_ellipsis_and_docstring():
    lines = doc(math.hypot).splitlines()
    assert lines[0] == "def hypot(...):"
    assert "Multidimensional Euclidean distance" in doc(math.hypot)
    assert NOTE not in doc(math.hypot)


def test_method_descriptor_renders_signature():
    result = doc(str.join)
    assert result.splitlines()[0] == "def str.join(self, iterable):"
    assert "class " not in result


def test_boost_function_without_doc_or_signature_shows_note():
    result = doc(_boost_fn("CreateGeomSubset", module="Subset"))
    lines = result.splitlines()
    assert lines[0] == "def CreateGeomSubset(...):"
    assert "class function" not in result
    assert lines[1] == f"    # {NOTE}"
    assert lines[2].startswith("    # doc(Subset) lists related names; dir() shows all;")
    assert len(lines) == 3


def test_compiled_class_lists_compiled_methods_in_definition_order():
    result = doc(FakeCompiled)
    lines = result.splitlines()
    assert lines[0] == "class FakeCompiled:"
    defs = [line.strip() for line in lines if line.strip().startswith("def ")]
    assert defs == [
        "def GetPoints(...):",
        "def Define(...):",
        "def Hypot(...):",
        "def Join(self, iterable):",
    ]
    # The long note belongs to doc() of a single function only.
    assert NOTE not in result


def test_compiled_methods_respect_hidden_marker():
    secret = hidden(_boost_fn("Secret"))
    cls = type("WithHidden", (FakeCompiled,), {"__module__": "fake_ext", "Secret": secret})
    result = doc(cls)
    assert "def Secret" not in result
    assert "def GetPoints(...):" in result  # inherited from a non-builtins base


def test_python_subclass_of_builtin_does_not_list_builtin_methods():
    class MyDict(dict):
        def extra(self) -> int:
            """Extra method."""
            return 1

    result = doc(MyDict)
    assert "def extra(self) -> int:" in result
    for name in ("keys", "items", "get", "update", "fromkeys"):
        assert f"def {name}" not in result


def test_c_heap_type_is_compiled():
    """C types created with PyType_FromSpec are heap types but still compiled."""
    import queue

    from nooa.agentdoc._structured import is_compiled_type

    assert is_compiled_type(queue.SimpleQueue)
    result = doc(queue.SimpleQueue)
    assert "def put(" in result
    assert "def get(" in result


def test_python_subclass_of_c_heap_type_is_not_compiled():
    import queue

    from nooa.agentdoc._structured import is_compiled_type

    class MyQueue(queue.SimpleQueue):
        def extra(self) -> int:
            """Extra method."""
            return 1

    assert not is_compiled_type(MyQueue)
    result = doc(MyQueue)
    assert "def extra(self) -> int:" in result
    assert "def put(" not in result


def test_derived_non_method_attribute_shadows_compiled_base_method():
    child = type("Child", (FakeCompiled,), {"__module__": "fake_ext", "GetPoints": None})
    result = doc(child)
    assert "def GetPoints" not in result
    assert "def Define(...):" in result  # not overridden, still inherited


def test_doc_of_several_compiled_callables():
    result = doc([len, str.join])
    assert "def len(obj):" in result
    assert "def str.join(self, iterable):" in result


def test_python_subclass_of_a_c_type_lists_only_its_own_methods():
    import collections

    from nooa.agentdoc.core import doc

    class Ordered(collections.OrderedDict):
        def mine(self):
            """Mine."""

    text = doc(Ordered)
    assert "def mine(self)" in text
    assert "setdefault" not in text and "fromkeys" not in text
