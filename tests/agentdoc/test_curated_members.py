# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""doc() on a method that a registered (curated) class view documents."""

from __future__ import annotations

import sys

import pytest

from nooa.agentdoc import doc, spec
from nooa.agentdoc.ext import CallableInfo, ModuleInfo, TypeInfo
from nooa.agentdoc.registry import (
    find_curated_member,
    get_attribute_hint,
    register_attribute_hint,
    unregister_attribute_hint,
    unregister_module_info_extractor,
    unregister_type_info_extractor,
)


class Widget:
    def resize(self, width, height):
        """Real docstring that the curated view replaces."""

    @classmethod
    def load(cls, path):
        pass

    def plain(self):
        """Not curated."""


class FancyWidget(Widget):
    pass


@pytest.fixture(autouse=True)
def curated_widget():
    @spec.define_doc(Widget)
    def _extract(_obj) -> TypeInfo:
        return TypeInfo(
            name="pkg.Widget",
            base=None,
            fields=[],
            methods=[
                CallableInfo("resize", "(self, width: int, height: int)", "None", "Resize in px."),
                CallableInfo("load", "(path: str)", "Widget", "Load from disk."),
            ],
            docstring="A widget.\nPitfall: resize() takes (width, height), not a tuple.",
        )

    yield
    unregister_type_info_extractor(Widget)


def test_function_on_the_class_shows_curated_entry_owner_notes_and_pointer():
    out = doc(Widget.resize)
    assert out == (
        "def pkg.Widget.resize(self, width: int, height: int) -> None:\n"
        '    """\n'
        "    Resize in px.\n"
        "    \n"
        "    From doc(pkg.Widget):\n"
        "    A widget.\n"
        "    Pitfall: resize() takes (width, height), not a tuple.\n"
        "    \n"
        "    Other members: doc(pkg.Widget)\n"
        '    """'
    )


def test_bound_and_class_methods_and_subclasses_resolve_the_owner():
    assert "def pkg.Widget.resize(self, width: int" in doc(Widget().resize)
    assert "def pkg.Widget.resize(self, width: int" in doc(FancyWidget().resize)
    assert "def pkg.Widget.load(path: str) -> Widget:" in doc(Widget.load)


def test_uncurated_method_of_a_curated_class_keeps_normal_introspection():
    assert find_curated_member(Widget.plain) is None
    out = doc(Widget.plain)
    assert "Not curated." in out and "pkg.Widget" not in out


def test_unrelated_callables_are_not_matched():
    def resize(self, width, height):
        pass

    assert find_curated_member(resize) is None
    assert find_curated_member(len) is None


def test_long_owner_notes_keep_only_items_naming_the_member():
    from nooa.agentdoc._pformat import _curated_member_info

    lines = [f"Idiom {i}." for i in range(25)]
    lines += [
        "- Pitfall: resize() returns a new widget;",
        "  the original is unchanged.",
        "- Other.",
    ]
    owner = TypeInfo("pkg.Widget", None, [], [], "\n".join(lines))
    member = CallableInfo("resize", "(self)", "None", "Resize.")
    assert _curated_member_info(owner, member).docstring == (
        "Resize.\n\n"
        "From doc(pkg.Widget):\n"
        "- Pitfall: resize() returns a new widget;\n"
        "  the original is unchanged.\n\n"
        "Other members: doc(pkg.Widget)"
    )


# --------------------------------------------------------------------------- module functions,
# aliases and methods shared by several curated classes


def _shared_impl(self):
    """One function object exposed on several classes (like Boost.Python's inherited methods)."""


class Base:
    name = _shared_impl


class Left(Base):
    pass


class Right(Base):
    pass


def _views(base_curated: bool):
    registered = []
    entries = [(Left, "pkg.Left", "Left name."), (Right, "pkg.Right", "Right name.")]
    if base_curated:
        entries.append((Base, "pkg.Base", "Any name."))
    for cls, view_name, text in entries:

        def _extract(_obj, view_name=view_name, text=text) -> TypeInfo:
            return TypeInfo(
                name=view_name,
                base=None,
                fields=[],
                methods=[CallableInfo("name", "(self)", "str", text)],
                docstring=f"{view_name} notes.",
            )

        spec.define_doc(cls)(_extract)
        registered.append(cls)
    return registered


@pytest.fixture
def shared_views(request):
    registered = _views(request.param)
    yield
    for cls in registered:
        unregister_type_info_extractor(cls)


@pytest.mark.parametrize("shared_views", [False], indirect=True)
def test_unbound_shared_method_lists_every_curated_class_without_their_notes(shared_views):
    out = doc(Left.name)
    # The uncurated defining class is named like its module's curated views (pkg.*).
    assert out.startswith("def pkg.Base.name(self) -> str:")
    assert "- pkg.Left.name: Left name." in out
    assert "- pkg.Right.name: Right name." in out
    assert "Inherited from pkg.Base by pkg.Left, pkg.Right" in out
    assert "notes." not in out  # neither subclass's notes are shown as the owner's


@pytest.mark.parametrize("shared_views", [True], indirect=True)
def test_unbound_shared_method_shows_the_curated_defining_class(shared_views):
    out = doc(Right.name)
    assert out.startswith("def pkg.Base.name(self) -> str:")
    assert "Any name." in out and "pkg.Base notes." in out
    assert "Left name." not in out and "Right name." not in out


@pytest.mark.parametrize("shared_views", [True], indirect=True)
def test_bound_shared_method_shows_the_instance_class(shared_views):
    assert doc(Left().name).startswith("def pkg.Left.name(self) -> str:")
    assert doc(Right().name).startswith("def pkg.Right.name(self) -> str:")


def test_alias_resolves_to_its_curated_name():
    class Arr:
        FromBuffer = staticmethod(lambda buffer: None)
        FromNumpy = FromBuffer

    @spec.define_doc(Arr)
    def _extract(_obj) -> TypeInfo:
        return TypeInfo(
            "pkg.Arr", None, [], [CallableInfo("FromNumpy", "(array)", "Arr", "From NumPy.")], ""
        )

    try:
        curated = find_curated_member(Arr.FromNumpy)
        assert curated is not None and curated.member.name == "FromNumpy"
        assert doc(Arr.FromNumpy).startswith("def pkg.Arr.FromNumpy(array) -> Arr:")
    finally:
        unregister_type_info_extractor(Arr)


def test_curated_module_function_shows_its_entry():
    import types

    module = types.ModuleType("fake_curated_mod")

    def dot(a, b):
        pass

    module.dot = dot
    sys.modules[module.__name__] = module

    @spec.define_doc(module)
    def _extract(_mod) -> ModuleInfo:
        return ModuleInfo(
            name="fake_curated_mod",
            docstring="Fake math.",
            functions=[CallableInfo("dot", "(a: Vec, b: Vec)", "float", "Dot product.")],
        )

    try:
        out = doc(dot)
        assert out.startswith("def fake_curated_mod.dot(a: Vec, b: Vec) -> float:")
        assert "Dot product." in out and "From doc(fake_curated_mod):" in out
    finally:
        unregister_module_info_extractor(module)
        del sys.modules[module.__name__]


# --------------------------------------------------------------------------- missing attributes


def test_attribute_hint_for_class_instance_subclass_and_module():
    import types

    register_attribute_hint(Widget, "resise", "use Widget.resize(width, height)")
    module = types.ModuleType("fake_hint_mod")
    register_attribute_hint(module, "Wdget", "the class is Widget")
    try:
        for target in (Widget, Widget(), FancyWidget, FancyWidget()):
            with pytest.raises(AttributeError) as info:
                target.resise  # noqa: B018 - the AttributeError is the point
            assert get_attribute_hint(info.value) == "use Widget.resize(width, height)"
        with pytest.raises(AttributeError) as info:
            module.Wdget  # noqa: B018
        assert get_attribute_hint(info.value) == "the class is Widget"
        with pytest.raises(AttributeError) as info:
            Widget.other  # noqa: B018
        assert get_attribute_hint(info.value) is None
    finally:
        unregister_attribute_hint(Widget, "resise")
        unregister_attribute_hint(module, "Wdget")


def test_error_formatting_appends_the_attribute_hint():
    from nooa.errors.formatting import format_error_for_llm

    register_attribute_hint(Widget, "resise", "use Widget.resize(width, height)")
    try:
        try:
            Widget.resise  # noqa: B018
        except AttributeError as error:
            text = format_error_for_llm(error)
        assert text.endswith("Hint: use Widget.resize(width, height)")
    finally:
        unregister_attribute_hint(Widget, "resise")
