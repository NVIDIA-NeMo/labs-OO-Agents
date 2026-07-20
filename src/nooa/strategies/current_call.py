# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CurrentCall dataclass - represents a method invocation being generated.

Captures all context needed by strategies to generate code for a method call.
"""

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, get_type_hints
from uuid import uuid4

from nooa.ellipsis_detection import get_pre_ellipsis_code

if TYPE_CHECKING:
    from nooa.config.truncation_config import TruncationConfig


@dataclass(frozen=True)
class CurrentCall:
    """Represents a method call being generated.

    This is an immutable snapshot of a method invocation, containing
    all the context a strategy needs to generate code.

    Attributes:
        id: Unique identifier for this call (for correlation/tracing).
        method_name: Name of the method being called.
        decorator: Type of decorator ('plan', 'agent').
        signature: Method signature string (optional).
        docstring: Method docstring (optional).
        args: Positional arguments passed to the method.
        kwargs: Keyword arguments passed to the method.
        parent_id: ID of parent call for nested invocations (optional).
        is_async: Whether the method is async (for proper def/async def in prompts).
        return_type: Return type annotation (optional, for prefill/error hints).
        pre_ellipsis_code: Setup code before `...` marker (optional, for prefill).

    Example:
        call = CurrentCall(
            id="call_abc123",
            method_name="analyze",
            decorator="plan",
            signature="(self, data: str) -> dict",
            docstring="Analyze the data.",
            args=("test data",),
            kwargs={},
        )
    """

    id: str
    method_name: str
    decorator: str
    signature: str | None = None
    docstring: str | None = None
    args: tuple[Any, ...] = field(default_factory=tuple)
    kwargs: dict[str, Any] = field(default_factory=dict)
    parent_id: str | None = None
    is_async: bool = False
    return_type: type | None = None
    pre_ellipsis_code: str | None = None
    session_locals: dict[str, Any] | None = None
    # Per-parameter spec overrides extracted from Annotated metadata.
    # Each value is a dict of kwargs from spec() — e.g. {"max_length": 20,
    # "max_string": 500}. Used by format_parameters_as_code to override the
    # default truncation knobs for individual parameters.
    param_specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Ordered parameter names (excluding 'self'), captured from the live method's
    # inspect.signature in from_method(). When present, this is authoritative and
    # avoids re-parsing the stringified signature (which can't reliably split on
    # commas inside Annotated[...]/defaults).
    param_names: list[str] | None = None

    def __hash__(self) -> int:
        """Hash by id for use in sets/dicts."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Equal if ids match."""
        if not isinstance(other, CurrentCall):
            return NotImplemented
        return self.id == other.id

    def bound_parameters(self) -> dict[str, Any]:
        """Return effective parameter name → value, each input represented exactly once.

        Positional arguments are mapped to the authoritative parameter names
        (``self.param_names``, captured from the live signature by :meth:`from_method`);
        keyword arguments override and extend them. Positional args beyond the named
        parameters (e.g. ``*args``, or when no ``param_names`` is available) are
        included under synthetic ``arg_<i>`` keys.

        This de-duplicates the overlap created by :meth:`from_method`, which stores
        positional args in both ``args`` and (mapped by name) ``kwargs`` for template
        expansion. Concatenating ``args + kwargs.values()`` therefore counts a
        positional argument twice; iterating ``bound_parameters().values()`` counts
        it once.

        When a keyword argument's name collides with a synthetic ``arg_<i>`` key,
        the keyword argument wins (kwargs are authoritative).

        Returns:
            Ordered mapping of effective parameter name to value.

        Example:
            For a method ``analyze(self, image)`` invoked positionally as
            ``analyze(img)``, ``from_method`` stores ``img`` in both ``args`` and
            ``kwargs["image"]``. ``bound_parameters()`` collapses that overlap and
            returns ``{"image": img}`` — ``img`` appears exactly once.
        """
        result: dict[str, Any] = {}
        param_names = self.param_names or []
        for i, value in enumerate(self.args):
            if i < len(param_names):
                result[param_names[i]] = value
            else:
                result[f"arg_{i}"] = value
        result.update(self.kwargs)
        return result

    def format_parameters_as_code(
        self,
        value_formatter: Callable[[Any], str] | None = None,
        *,
        tc: "TruncationConfig | None" = None,
    ) -> str:
        """Format method parameters as Python variable assignments.

        Creates a string with one assignment per line, suitable for inclusion
        in LLM prompts to show the current parameter values.

        Args:
            value_formatter: Optional callable to format each parameter value.
                When provided, takes precedence over ``tc``.
                When both are None, defaults to ``repr``.
            tc: Optional TruncationConfig.  When provided (and ``value_formatter``
                is None), uses ``pformat`` with ``tc.prefill`` structural limits
                (``max_length``, ``max_string``, ``max_depth``).
                This matches how ``InspectInputsPrefill`` formats parameter values.

        Returns:
            Formatted parameter assignments (e.g., "data = 'test'\\nthreshold = 0.5")
            or empty string if no parameters.

        Example:
            >>> call = CurrentCall(method_name="analyze", args=("test",), kwargs={"threshold": 0.5})
            >>> print(call.format_parameters_as_code())
            data = "test"
            threshold = 0.5
        """
        # Build the default formatter (used when no per-param spec override).
        if value_formatter is not None:
            default_fmt = value_formatter
        elif tc is not None:
            from nooa.agentdoc import pformat

            _tc = tc

            def _default_fmt(v: Any) -> str:
                return pformat(v, **_tc.prefill_format.model_dump())

            default_fmt = _default_fmt
        else:
            default_fmt = repr

        # Build a per-parameter formatter that honors `Annotated[T, spec(max_*)]`
        # overrides when present, falling back to the default for everything else.
        # When `value_formatter` is explicitly supplied, it wins for all params
        # (callers know what they're doing) — only the tc/repr path consults
        # per-param specs.
        def fmt_for(name: str) -> Callable[[Any], str]:
            if value_formatter is not None or not self.param_specs.get(name):
                return default_fmt
            override = self.param_specs[name]
            if not any(k in override for k in ("max_length", "max_string", "max_depth")):
                return default_fmt
            from nooa.agentdoc import pformat

            tc_max_length = tc.prefill_format.max_length if tc is not None else None
            tc_max_string = tc.prefill_format.max_string if tc is not None else None
            tc_max_depth = tc.prefill_format.max_depth if tc is not None else None
            ml = override.get("max_length", tc_max_length)
            ms = override.get("max_string", tc_max_string)
            md = override.get("max_depth", tc_max_depth)

            def _fmt(v: Any) -> str:
                return pformat(v, max_length=ml, max_string=ms, max_depth=md)

            return _fmt

        # Map each effective input to its name exactly once. bound_parameters() is
        # the single source of truth for positional→name binding — it uses the
        # authoritative param_names (captured from the live signature by
        # from_method / actor), maps overflow positionals to ``arg_<i>`` keys, and
        # lets kwargs override. Delegating keeps this formatter, _assert_param_sizes,
        # and the media collection consistent. The per-parameter formatter still
        # picks up any Annotated spec() overrides (arg_<i> keys have none, so they
        # fall back to default_fmt).
        param_dict = self.bound_parameters()
        if not param_dict:
            return ""
        try:
            lines = [f"{name} = {fmt_for(name)(value)}" for name, value in param_dict.items()]
            return "\n".join(lines)
        except (ValueError, AttributeError):
            # A value's repr/formatter raised (e.g. a broken __repr__). Fall back to
            # formatting only the explicitly-passed kwargs, which are the safe values.
            if not self.kwargs:
                return ""
            lines = [f"{name} = {fmt_for(name)(value)}" for name, value in self.kwargs.items()]
            return "\n".join(lines)

    def format_signature(self) -> str:
        """Format method signature with type annotations (no values).

        Returns the full method definition line suitable for showing in prompts.

        Returns:
            Formatted signature (e.g., "async def calculate(self, a: int, b: int) -> int")

        Example:
            >>> call = CurrentCall(method_name="analyze", signature="(self, data: str) -> dict", is_async=True)
            >>> call.format_signature()
            'async def analyze(self, data: str) -> dict'
        """
        if self.signature:
            prefix = "async def" if self.is_async else "def"
            return f"{prefix} {self.method_name}{self.signature}"

        # Fallback when no signature available
        prefix = "async def" if self.is_async else "def"
        return f"{prefix} {self.method_name}(self, ...) -> ..."

    @classmethod
    def from_method(
        cls,
        method: Callable[..., Any],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
        decorator: str = "plan",
        parent_id: str | None = None,
    ) -> "CurrentCall":
        """Create CurrentCall from a method object.

        Extracts signature and docstring automatically.
        Maps positional arguments to parameter names for template expansion.

        Args:
            method: The method being called.
            args: Positional arguments for the call.
            kwargs: Keyword arguments for the call.
            decorator: Decorator type ('plan', 'agent').
            parent_id: Parent call ID for nested calls.

        Returns:
            CurrentCall instance with extracted metadata.
        """
        # Generate unique ID
        call_id = f"call_{uuid4().hex[:12]}"

        # Extract signature
        try:
            sig = inspect.signature(method)
            signature_str = str(sig)
        except (ValueError, TypeError):
            sig = None
            signature_str = None

        # Extract docstring
        docstring = inspect.getdoc(method)

        # Check if async
        is_async = inspect.iscoroutinefunction(method)

        # Map positional args to parameter names (for template expansion)
        # This ensures positional arguments are available as template variables
        merged_kwargs = dict(kwargs or {})
        if sig and args:
            # Get parameter names (excluding 'self')
            param_names = [p for p in sig.parameters.keys() if p != "self"]
            # Map positional args to their parameter names
            for i, value in enumerate(args):
                if i < len(param_names):
                    param_name = param_names[i]
                    # Don't overwrite explicitly passed kwargs
                    if param_name not in merged_kwargs:
                        merged_kwargs[param_name] = value

        # Extract return type — use get_type_hints to resolve PEP 563
        # stringified annotations back to actual type objects.
        return_type = None
        param_hints: dict[str, Any] = {}
        if sig:
            try:
                hints = get_type_hints(method, include_extras=True)
                return_annotation = hints.get("return", inspect.Signature.empty)
                # Stash per-parameter hints for spec extraction below.
                param_hints = {k: v for k, v in hints.items() if k != "return"}
            except (NameError, TypeError, AttributeError):
                return_annotation = sig.return_annotation
            if return_annotation is not inspect.Signature.empty:
                return_type = return_annotation

        # Extract per-parameter spec() overrides from Annotated metadata.
        # ``Annotated[T, spec(max_length=20)]`` lets agents override the
        # framework's default truncation knobs for a single parameter without
        # changing the agent-level TruncationConfig.
        param_specs: dict[str, dict[str, Any]] = {}
        try:
            from typing import Annotated as _Annotated
            from typing import get_origin

            from nooa.agentdoc._docs import SpecAnnotation

            for pname, hint in param_hints.items():
                if get_origin(hint) is not _Annotated.__class__ and not (
                    hasattr(hint, "__metadata__")
                ):
                    continue
                metadata = getattr(hint, "__metadata__", None) or ()
                merged: dict[str, Any] = {}
                for m in metadata:
                    if isinstance(m, SpecAnnotation):
                        merged.update(m.kwargs)
                if merged:
                    param_specs[pname] = merged
        except Exception:
            param_specs = {}

        # Extract pre-ellipsis code (setup code before ... marker)
        pre_ellipsis_code = get_pre_ellipsis_code(method)

        # Authoritative ordered names from the live signature (excludes 'self').
        live_param_names = [p for p in sig.parameters if p != "self"] if sig is not None else None

        return cls(
            id=call_id,
            method_name=method.__name__,
            decorator=decorator,
            signature=signature_str,
            docstring=docstring,
            args=args,
            kwargs=merged_kwargs,
            parent_id=parent_id,
            is_async=is_async,
            return_type=return_type,
            pre_ellipsis_code=pre_ellipsis_code,
            param_specs=param_specs,
            param_names=live_param_names,
        )
