# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""doc() on a method that a registered (curated) class view documents."""

from __future__ import annotations

import pytest

from nooa.agentdoc import doc, spec
from nooa.agentdoc.ext import CallableInfo, TypeInfo
from nooa.agentdoc.registry import find_curated_member, unregister_type_info_extractor


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
