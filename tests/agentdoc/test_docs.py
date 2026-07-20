# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for agentdoc._docs — spec() universal annotation function."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel
from pydantic import Field as PydanticField

from nooa.agentdoc._docs import SpecAnnotation, spec
from nooa.agentdoc._metadata import get_docs_metadata, get_field_metadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MyTool:
    """A tool that does things."""

    def run(self) -> str:
        return "result"


# ---------------------------------------------------------------------------
# SpecAnnotation
# ---------------------------------------------------------------------------


class TestSpecAnnotation:
    def test_repr(self):
        ann = SpecAnnotation(hidden=True, description="hi")
        r = repr(ann)
        assert "hidden=True" in r
        assert "description='hi'" in r

    def test_call_as_decorator_sets_metadata(self):
        @SpecAnnotation(hidden=True)
        class C:
            pass

        assert get_docs_metadata(C).get("hidden") is True

    def test_call_sets_agentdoc_hidden_attr(self):
        @SpecAnnotation(hidden=True)
        def f():
            pass

        assert getattr(f, "_agentdoc_hidden", False) is True

    def test_call_returns_target(self):
        class C:
            pass

        result = SpecAnnotation(description="x")(C)
        assert result is C


# ---------------------------------------------------------------------------
# spec() — annotation / decorator form (no positional args)
# ---------------------------------------------------------------------------


class TestDocsNoTarget:
    def test_returns_docs_annotation_instance(self):
        result = spec(hidden=True)
        assert isinstance(result, SpecAnnotation)

    def test_kwargs_stored_on_annotation(self):
        ann = spec(description="test desc", expand=False)
        assert isinstance(ann, SpecAnnotation)
        assert ann.kwargs["description"] == "test desc"
        assert ann.kwargs["expand"] is False

    def test_empty_call_returns_annotation(self):
        ann = spec()
        assert isinstance(ann, SpecAnnotation)
        assert ann.kwargs == {}


# ---------------------------------------------------------------------------
# spec() as decorator
# ---------------------------------------------------------------------------


class TestDocsDecorator:
    def test_hidden_decorator(self):
        ann = spec(hidden=True)
        assert isinstance(ann, SpecAnnotation)

        @ann
        class C:
            pass

        assert get_docs_metadata(C).get("hidden") is True

    def test_description_decorator(self):
        ann = spec(description="My description")
        assert isinstance(ann, SpecAnnotation)

        @ann
        def f():
            pass

        assert get_docs_metadata(f).get("description") == "My description"

    def test_expand_false_decorator(self):
        ann = spec(expand=False)
        assert isinstance(ann, SpecAnnotation)

        @ann
        class CompactThing:
            pass

        assert get_docs_metadata(CompactThing).get("expand") is False

    def test_decorator_returns_original_object(self):
        ann = spec(hidden=True)
        assert isinstance(ann, SpecAnnotation)

        @ann
        class C:
            pass

        assert isinstance(C, type)
        assert C.__name__ == "C"


# ---------------------------------------------------------------------------
# spec() in Annotated[..., spec(...)]
# ---------------------------------------------------------------------------


class TestDocsAnnotated:
    def test_annotated_marker_is_docs_annotation(self):
        marker = spec(hidden=True)
        annotated = Annotated[str, marker]
        # The marker is stored as metadata; check it's a SpecAnnotation
        import typing

        args = typing.get_args(annotated)
        assert args[1] is marker

    def test_annotated_marker_kwargs(self):
        marker = spec(description="a string field")
        assert isinstance(marker, SpecAnnotation)
        assert marker.kwargs["description"] == "a string field"


# ---------------------------------------------------------------------------
# spec() — imperative form with positional target
# ---------------------------------------------------------------------------


class TestDocsAnnotatedDescription:
    """doc() renders descriptions from all supported annotation forms."""

    def test_plain_string_annotation(self):
        """Annotated[T, 'string'] works as description."""
        from nooa.agentdoc import doc

        class Config:
            name: Annotated[str, "the user name"] = ""  # type: ignore[assignment]

        result = doc(Config)
        assert "the user name" in result
        assert "Annotated" not in result

    def test_annotated_docs_description_shown_in_doc(self):
        """Annotated[T, spec(description=...)] shown as # comment."""
        from nooa.agentdoc import doc

        class Config:
            query: Annotated[str, spec(description="The search query")] = ""  # type: ignore[assignment]

        result = doc(Config)
        assert "The search query" in result
        assert "Annotated" not in result

    def test_pydantic_field_description_on_plain_class(self):
        """Annotated[T, Field(description=...)] works on plain (non-BaseModel) classes."""
        from nooa.agentdoc import doc

        class Config:
            name: Annotated[str, PydanticField(description="the user name")] = ""  # type: ignore[assignment]

        result = doc(Config)
        assert "the user name" in result
        assert "Annotated" not in result

    def test_pydantic_field_description_on_base_model(self):
        """Field(description=...) on BaseModel still works."""
        from nooa.agentdoc import doc

        class Config(BaseModel):
            name: str = PydanticField(default="", description="the user name")

        result = doc(Config)
        assert "the user name" in result

    def test_imperative_docs_description_shown_in_doc(self):
        """spec(Cls, 'field', description=...) shown as # comment."""
        from nooa.agentdoc import doc

        class Config:
            batch_size: int = 64

        spec(Config, "batch_size", description="Records per batch")
        result = doc(Config)
        assert "Records per batch" in result

    def test_type_stripped_not_annotated_in_output(self):
        """Type shown as plain base type, not the full Annotated[...] wrapper."""
        from nooa.agentdoc import doc

        class Config:
            count: Annotated[int, PydanticField(description="item count")] = 0  # type: ignore[assignment]

        result = doc(Config)
        assert "int" in result
        assert "Annotated" not in result


class TestDocsImperative:
    def test_imperative_returns_none(self):
        class C:
            pass

        result = spec(C, hidden=True)
        assert result is None

    def test_imperative_sets_metadata_on_class(self):
        class C:
            pass

        spec(C, description="class desc")
        assert get_docs_metadata(C)["description"] == "class desc"

    def test_imperative_field_form(self):
        class C:
            x: int = 0

        spec(C, "x", hidden=True)
        assert get_field_metadata(C, "x")["hidden"] is True

    def test_imperative_on_instance(self):
        class C:
            pass

        obj = C()
        # Should not raise (even if it can't set on all objects)
        spec(obj, description="instance desc")

    def test_imperative_hidden_respected_by_doc(self):
        """spec(Class, 'field', hidden=True) must hide the field from doc() output."""
        from nooa.agentdoc import doc

        class Config:
            host: str = "localhost"
            password: str = "secret"

        spec(Config, "password", hidden=True)
        result = doc(Config)
        assert "host" in result
        assert "password" not in result

    def test_imperative_hidden_respected_by_pformat(self):
        """spec(Class, 'field', hidden=True) must hide the field from pformat() output."""
        from nooa.agentdoc import pformat

        class Config:
            host: str = "localhost"
            password: str = "secret"

            def __init__(self):
                self.host = "localhost"
                self.password = "secret"

        spec(Config, "password", hidden=True)
        result = pformat(Config())
        assert "host" in result
        assert "password" not in result

    def test_imperative_hidden_consistent_doc_and_pformat(self):
        """hidden=True via imperative form hides from both doc() and pformat()."""
        from nooa.agentdoc import doc, pformat

        class Credentials:
            username: str = ""
            token: str = ""
            secret: str = ""

            def __init__(self):
                self.username = "alice"
                self.token = "tok_abc"
                self.secret = "s3cr3t"

        spec(Credentials, "token", hidden=True)
        spec(Credentials, "secret", hidden=True)

        doc_out = doc(Credentials)
        pformat_out = pformat(Credentials())

        assert "username" in doc_out
        assert "token" not in doc_out
        assert "secret" not in doc_out

        assert "username" in pformat_out
        assert "token" not in pformat_out
        assert "secret" not in pformat_out

    def test_imperative_description_in_doc(self):
        """spec(Class, 'field', description=...) shows as inline comment in doc()."""
        from nooa.agentdoc import doc

        class Server:
            host: str = "localhost"
            port: int = 8080

        spec(Server, "host", description="Hostname or IP address")
        result = doc(Server)
        assert "Hostname or IP address" in result


# ---------------------------------------------------------------------------
# spec.define_doc()
# ---------------------------------------------------------------------------


class TestDocsDefine:
    def test_define_doc_returns_callable(self):
        # spec.define_doc(T) returns a decorator
        decorator = spec.define_doc(MyTool)
        assert callable(decorator)


# ---------------------------------------------------------------------------
# TestImperativeSpecMROIsolation
# ---------------------------------------------------------------------------


class TestImperativeSpecMROIsolation:
    def test_spec_child_does_not_mutate_parent_metadata(self):
        """spec(Child, field, hidden=True) must not touch Parent's metadata dict."""
        from nooa.agentdoc._metadata import get_field_metadata

        class Parent:
            x: int = 1

        class Child(Parent):
            pass

        spec(Child, "x", hidden=True)
        # Parent's own metadata must be untouched
        assert get_field_metadata(Parent, "x").get("hidden") is not True

    def test_spec_parent_hidden_child_spec_visible(self):
        """spec(Parent, field, hidden=True) + spec(Child, field, hidden=False) → visible in doc(Child)."""
        from nooa.agentdoc import doc

        class Parent:
            secret: str = "shh"

        class Child(Parent):
            pass

        spec(Parent, "secret", hidden=True)
        spec(Child, "secret", hidden=False)
        result = doc(Child)
        assert "secret" in result

    def test_hidden_false_on_parent_method_not_inherited_by_child(self):
        """spec(Parent, '_method', hidden=False) must not make _method visible on Child."""
        from nooa.agentdoc import doc

        class Parent:
            def _internal(self): ...

        class Child(Parent):
            pass

        spec(Parent, "_internal", hidden=False)
        result = doc(Child)
        assert "_internal" not in result


class TestPydanticReprFalse:
    """Fields with repr=False should be hidden in doc() output."""

    def test_repr_false_hidden_in_doc(self):
        from nooa.agentdoc import doc

        class Config(BaseModel):
            visible: str = PydanticField(default="shown")
            secret: str = PydanticField(default="hidden", repr=False)

        result = doc(Config)
        assert "visible" in result
        assert "secret" not in result

    def test_repr_true_shown_in_doc(self):
        from nooa.agentdoc import doc

        class Config(BaseModel):
            name: str = PydanticField(default="alice", repr=True)

        result = doc(Config)
        assert "name" in result


class TestDefineDocEndToEnd:
    def test_define_doc_type_only_path(self):
        """spec.define_doc() with a plain TypeInfo return is used by doc(MyClass)."""
        from nooa.agentdoc import doc
        from nooa.agentdoc.ext import FieldInfo, TypeInfo
        from nooa.agentdoc.registry import clear_registry

        class ThirdParty:
            pass

        clear_registry()

        @spec.define_doc(ThirdParty)
        def _(obj):
            return TypeInfo(
                name="ThirdParty",
                base=None,
                fields=[FieldInfo(name="custom_field", type="str", default="hello")],
                methods=[],
                docstring="A third-party type.",
            )

        result = doc(ThirdParty)
        assert "ThirdParty" in result
        assert "custom_field" in result
        clear_registry()

    def test_define_doc_tuple_path_for_instance(self):
        """spec.define_doc() returning (TypeInfo, values_dict) is used by doc(instance)."""
        from nooa.agentdoc import doc
        from nooa.agentdoc.ext import FieldInfo, TypeInfo
        from nooa.agentdoc.registry import clear_registry

        class ThirdParty:
            def __init__(self, value):
                self.value = value

        clear_registry()

        @spec.define_doc(ThirdParty)
        def _(obj):
            type_info = TypeInfo(
                name="ThirdParty",
                base=None,
                fields=[FieldInfo(name="value", type="str", default="default")],
                methods=[],
                docstring="A third-party type.",
            )
            if isinstance(obj, type):
                return type_info
            return type_info, {"value": obj.value}

        instance = ThirdParty("runtime_value")
        result = doc(instance)
        assert "ThirdParty" in result
        clear_registry()
