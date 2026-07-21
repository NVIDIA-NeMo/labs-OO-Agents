# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""spec() — step 1: specify how a type renders.

Use ``spec`` to attach visibility rules, descriptions, and rendering hints
to fields, methods, or whole types. ``doc()`` and ``pformat()`` pick them
up automatically.

Which form to use:

- **Annotated marker** — for fields in types you own::

      api_key: Annotated[str, spec(hidden=True)] = ""
      label:   Annotated[str, spec(description="Display name")] = "agent"

- **Decorator** — for methods or classes you own::

      @spec(hidden=True)          # hide a method
      def _internal(self): ...

      @spec(expand=False)         # collapse to one-liner in doc()
      class SubConfig: ...

      @spec(hidden=False)         # opt a dunder or _-prefixed name into doc()
      def __init__(self): ...

- **Imperative** — for types you don't own::

      spec(ThirdPartyClass, "verbose_field", hidden=True)
      spec(ThirdPartyClass, "verbose_field", description="Full config dump")

  Quick rule: use ``@hidden`` for simple method hiding; use ``@spec(hidden=True)``
  when you also need to attach an expand hint in the same call.

Extension point:
   ``spec.define_doc(Type)``  — specify a custom TypeInfo extractor for doc()
"""

from __future__ import annotations

import contextlib
from typing import Annotated, Any

from nooa.agentdoc._metadata import set_docs_metadata, set_field_metadata

_SENTINEL = object()


def _sync_hidden(target: Any, hidden_value: bool | None) -> None:
    """Sync the ``_agentdoc_hidden`` attribute on *target* based on *hidden_value*.

    Shared by ``SpecAnnotation.__call__`` (decorator form) and
    ``Spec.__call__`` (imperative form).
    """
    if hidden_value is True:
        with contextlib.suppress(AttributeError, TypeError):
            target._agentdoc_hidden = True
    elif hidden_value is False:
        try:
            target._agentdoc_hidden = False
        except (AttributeError, TypeError):
            # C-extension method: fall back to class-level metadata
            owner = getattr(target, "__objclass__", None)
            mname = getattr(target, "__name__", None)
            if owner is not None and mname is not None:
                set_field_metadata(owner, mname, hidden=False)


class SpecAnnotation:
    """A specification returned by ``spec(**kwargs)`` when no positional target is given.

    Works as:
    - Annotated metadata: ``Annotated[str, spec(hidden=True)]``
    - Decorator: ``@spec(hidden=True)``
    """

    __slots__ = ("kwargs",)

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __call__(self, target: Any) -> Any:
        """Apply as decorator: ``@spec(...)``."""
        # Sync _agentdoc_hidden attribute so is_hidden_method() and the _ prefix
        # override check in _extract_methods() both see the correct state.
        _sync_hidden(target, self.kwargs.get("hidden"))
        set_docs_metadata(target, **self.kwargs)
        return target

    def __repr__(self) -> str:
        kw = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"spec({kw})"


class Spec:
    """The ``spec`` singleton — step 1: specify how a type renders.

    Attach visibility rules, descriptions, and rendering hints to fields,
    methods, or whole types; ``doc()`` and ``pformat()`` pick them up
    automatically.

    **Default visibility rules:**

    - Public names (no leading underscore) are **visible** by default.
    - ``_private`` names (single or double leading underscore) are **hidden**
      by default.  Use ``spec(hidden=False)`` or ``@spec(hidden=False)`` to
      opt a private name back in.

    Which form to use:

    - **Annotated marker** — for fields in types you own::

          api_key: Annotated[str, spec(hidden=True)] = ""

    - **Decorator** — for methods or classes you own::

          @spec(hidden=True)
          def _internal(self): ...

          @spec(hidden=False)       # opt a _private method back in
          def _shown_helper(self): ...

    - **Imperative** — for types you don't own::

          spec(ThirdPartyClass, "verbose_field", hidden=True)
    """

    def __call__(
        self,
        target: Any = _SENTINEL,
        field: Annotated[
            str | None,
            "Field or method name — second positional arg in imperative form: spec(MyClass, 'field_name', hidden=True)",
        ] = None,
        /,
        *,
        hidden: Annotated[
            bool | None, "If True, exclude from documentation; False to force-show _-prefixed names"
        ] = None,
        description: Annotated[str | None, "Shown as inline # comment in doc() output"] = None,
        expand: Annotated[
            bool | None, "If False, collapse sub-type to a one-liner in doc()"
        ] = None,
        max_length: Annotated[
            int | None,
            "Override max items per container in pformat — parameter annotations only, e.g. Annotated[list, spec(max_length=20)] (not honored on class fields)",
        ] = _SENTINEL,
        max_string: Annotated[
            int | None,
            "Override max characters per string in pformat for this annotation (parameter annotation or class field)",
        ] = _SENTINEL,
        max_depth: Annotated[
            int | None,
            "Override max nesting depth in pformat — parameter annotations only (not honored on class fields)",
        ] = _SENTINEL,
    ) -> SpecAnnotation | None:
        """Specify documentation metadata — step 1.

        **No positional args** → returns a ``SpecAnnotation`` for use as an
        ``Annotated`` marker or decorator::

            api_key: Annotated[str, spec(hidden=True)] = ""   # marker form
            @spec(hidden=True)                                  # decorator form
            def _internal(self): ...

        Note: ``target`` has an internal sentinel default; when rendered by
        ``doc(spec)``, it appears as ``target: Any = ...``.  This ``...`` is
        **not** the NVIDIA OO Agents generation-method ellipsis — it just means "omit
        to get a ``SpecAnnotation`` back; supply a class/function to use the
        imperative form".

        **Positional target** → imperative form, mutates in-place, returns None.
        First arg is the class/function; second (optional) is the field name::

            spec(MyClass, "field_name", hidden=True)    # hide a field
            spec(MyClass, "field_name", description="…") # describe a field
            spec(MyClass, hidden=True)                  # hide the whole class

        Examples::

            # Annotated marker — inline with the type annotation
            class MyAgent:
                api_key: Annotated[str, spec(hidden=True)] = ""
                label: Annotated[str, spec(description="Display name")] = "agent"
                config: Annotated[Config, spec(expand=False)] = Config()

            # Decorator — on a method or class
            @spec(hidden=True)
            def _internal(self): ...

            @spec(expand=False)
            class SubConfig: ...

            # Imperative — for types you don't own
            spec(ThirdPartyClass, "verbose_field", hidden=True)
            spec(ThirdPartyClass, "note", description="See docs")
        """
        kwargs: dict[str, Any] = {}
        if hidden is not None:
            kwargs["hidden"] = hidden
        if description is not None:
            kwargs["description"] = description
        if expand is not None:
            kwargs["expand"] = expand
        if max_length is not _SENTINEL:
            kwargs["max_length"] = max_length
        if max_string is not _SENTINEL:
            kwargs["max_string"] = max_string
        if max_depth is not _SENTINEL:
            kwargs["max_depth"] = max_depth

        if target is _SENTINEL:
            return SpecAnnotation(**kwargs)

        # Imperative form: mutate target and return None
        if field is not None:
            set_field_metadata(target, field, **kwargs)
        else:
            set_docs_metadata(target, **kwargs)
            # Sync _agentdoc_hidden so _extract_methods sees the override
            _sync_hidden(target, hidden)

        return None

    def define_doc(self, target) -> Any:
        """Define a custom extractor for *target* in ``doc()``.

        Completely replaces the default introspection. Works for both types and
        modules.

        **For a type** — extractor receives the type or an instance, returns
        ``TypeInfo`` (or ``(TypeInfo, values_dict)`` to include current values)::

            @spec.define_doc(MyClass)
            def _(cls_or_instance) -> TypeInfo | tuple[TypeInfo, dict]:
                if isinstance(cls_or_instance, type):
                    return TypeInfo(name="MyClass", fields=[...], ...)
                return TypeInfo(...), {"field": cls_or_instance.field}

        **For a module** — extractor receives the module, returns ``ModuleInfo``::

            import numpy
            from nooa.agentdoc.ext import ModuleInfo, CallableInfo

            @spec.define_doc(numpy)
            def _(mod) -> ModuleInfo:
                return ModuleInfo(
                    name="numpy",
                    docstring="NumPy — N-dimensional array computing.",
                    functions=[CallableInfo(name="array", ...)],
                    submodules=[("numpy.linalg", "Linear algebra.")],
                )
        """
        import inspect

        if inspect.ismodule(target):
            from nooa.agentdoc.registry import register_module_info_extractor

            return register_module_info_extractor(target)

        from nooa.agentdoc.registry import register_type_info_extractor

        return register_type_info_extractor(target)


spec = Spec()

# Opt __call__ into doc(spec) — show the full invocation signature.
# Set directly to avoid the chicken-and-egg of using spec() before it exists.
Spec.__call__._agentdoc_hidden = False  # type: ignore[attr-defined]
