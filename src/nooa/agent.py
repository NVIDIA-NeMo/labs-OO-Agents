# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent base class."""

import logging
from typing import TYPE_CHECKING, Annotated, Any, cast
from uuid import uuid4

from nooa.agentdoc import hidden
from nooa.context_blocks import DynamicContext
from nooa.metaclass import AgentMeta, no_trace
from nooa.runtime.context_vars import _parent_agent_var
from nooa.storage.markers import nosnapshot

if TYPE_CHECKING:
    from nooa.agentdoc.ext import TypeInfo
    from nooa.config.execution_config import ExecutionConfig
    from nooa.config.truncation_config import TruncationConfig
    from nooa.context_blocks.models import ContextWindowStats
    from nooa.context_blocks.render_config import RenderConfig
    from nooa.runtime.actor import ActorRuntime
    from nooa.runtime.context import ContextApi
    from nooa.runtime.context_manager import ContextManager
    from nooa.runtime.event_manager import EventManager
    from nooa.runtime.event_query import EventQuery
    from nooa.runtime.events import EventsApi
    from nooa.storage.manager import StorageManager
    from nooa.unifiedllm import UnifiedLLM

logger = logging.getLogger(__name__)

_auto_tracing_attempted = False


def _try_auto_enable_tracing() -> None:
    """Auto-enable OTLP tracing if the viewer is reachable. Called once per process."""
    global _auto_tracing_attempted
    if _auto_tracing_attempted:
        return
    _auto_tracing_attempted = True
    try:
        from nooa.tracing import enable_tracing

        enable_tracing()
    except ImportError:
        pass


# Internal sentinel for LLM cascading resolution
class _InheritSentinel:
    """Internal sentinel for LLM cascading resolution.

    NOT part of public API. Users should omit the llm parameter to enable cascading.
    This sentinel distinguishes "parameter omitted" from "parameter explicitly set to None".
    """

    def __repr__(self) -> str:
        return "INHERIT"


INHERIT = _InheritSentinel()


def _validate_llm_param(llm: object, class_name: str) -> None:
    """Validate the llm parameter for __init_subclass__ and __init__."""
    if llm is None:
        raise ValueError(
            f"{class_name}: llm=None is not allowed. "
            "To enable cascading, omit the llm parameter entirely. "
            "To use an explicit LLM, pass llm=my_llm."
        )


class Agent(metaclass=AgentMeta):
    """You are {type(self).__name__}, a Python agent working in an interactive session.

    ## Context blocks
    Your prompt is organized in XML context blocks: `<name>CONTENT</name>`.
    Blocks produced by `self.context[...]` carry an `expr="..."` attribute whose value is the Python expression re-evaluated each turn.
    Event history: system entries in `<sys tag="N">`; reference via `self.events["N"]`.

    ## Truncation
    - A bare Python literal (`[1, 2, 3]`, `{{1: 2}}`, `'hello'`) is always complete.
    - Truncated values use a `type(len=N, ...)` (or `type(repr_len=N, ...)`) marker:
        list(len=100, [:5]=[...], [-5:]=[...])
        tuple(len=100, [:5]=(...), [-5:]=[...])
        dict(len=100, items={{...}})
        set(len=100, items={{...}})
        str(len=100000, [:250]='...', [-250:]='...')
        ndarray(repr_len=233, [:100]='...', [-100:]='...')
    - Structured instances (dataclasses, Pydantic, custom classes) render as `ClassName(field=value, ...)`; a trailing `...` means fields were elided:
        Config(name='foo', enabled=True, ...)
    - The variable itself is **not** truncated — index/iterate it directly to operate on the full data.
    - `<truncated>...</truncated>` in captured stdout/stderr is **not recoverable**.
    """

    # Agent instances should never be serialized as nested values inside other objects.
    __nosnapshot__ = True

    # Framework attributes — hidden from LLM, excluded from snapshots
    runtime: Annotated["ActorRuntime", hidden, nosnapshot]
    _storage: Annotated["StorageManager", hidden, nosnapshot]
    event_manager: Annotated["EventManager", hidden, nosnapshot]
    context_manager: Annotated["ContextManager", hidden, nosnapshot]
    event_query: Annotated["EventQuery | None", hidden, nosnapshot]
    render_config: Annotated["RenderConfig", hidden, nosnapshot]
    _agent_id: Annotated[str, hidden, nosnapshot]
    _llm: Annotated["UnifiedLLM", hidden, nosnapshot]
    _truncation: Annotated["TruncationConfig", hidden, nosnapshot]
    context: Annotated["ContextApi", hidden, nosnapshot]
    events: Annotated["EventsApi", hidden, nosnapshot]

    # Class-level framework attrs — hidden from LLM (gl-78).
    # _abc_impl is injected by CPython's ABCMeta; the rest are set in
    # __init_subclass__.  Annotating them here makes is_hidden_field() find
    # them via MRO so _iter_agent_attrs skips them.
    _abc_impl: Annotated[Any, hidden]
    _enable_tracing: Annotated[bool, hidden]
    _execution_config: Annotated["ExecutionConfig", hidden]
    _agent_llm: Annotated["UnifiedLLM | _InheritSentinel", hidden]
    _agent_truncation: Annotated["TruncationConfig", hidden]
    _agent_context_blocks: Annotated["dict[str, str | DynamicContext | None]", hidden]
    _agent_event_query: Annotated["EventQuery | None", hidden]

    # Enable tracing for Agent classes (convention for metaclass)
    _enable_tracing = True  # type: ignore[assignment]

    def __init_subclass__(
        cls,
        llm: "UnifiedLLM | _InheritSentinel" = INHERIT,
        truncation: "TruncationConfig | None" = None,
        execution: "ExecutionConfig | None" = None,
        context: "dict[str, str | DynamicContext | None] | None" = None,
        event_query: "EventQuery | None" = None,
        **kwargs: Any,
    ):
        """Configure agent class with metaclass.

        Args:
            llm: LLM client for this agent class. Omit to enable cascading.
            truncation: Optional truncation configuration for stdout/stderr/pprint limits.
            execution: ExecutionConfig for framework-level execution guards.
            context: Class-level context block overrides.
                - str: Static content
                - DynamicContext("expr"): DynamicContext expression, re-evaluated each turn
                - None: Remove block
            event_query: Default EventQuery for filtering events in context.
            **kwargs: Additional arguments for multiple inheritance support.
        """
        _validate_llm_param(llm, cls.__name__)

        super().__init_subclass__(**kwargs)

        if llm is not INHERIT:
            cls._agent_llm = llm  # type: ignore[attr-defined]
        if truncation is not None:
            cls._agent_truncation = truncation  # type: ignore[attr-defined]
        if context is not None:
            cls._agent_context_blocks = context  # type: ignore[attr-defined]
        if event_query is not None:
            cls._agent_event_query = event_query  # type: ignore[attr-defined]

        from nooa.config.execution_config import ExecutionConfig as _EC

        cls._execution_config = execution or _EC()  # type: ignore[attr-defined]

    def __init__(
        self,
        llm: "UnifiedLLM | _InheritSentinel" = INHERIT,
        *,
        truncation: "TruncationConfig | None" = None,
        render_config: "RenderConfig | None" = None,
        context: "dict[str, str | DynamicContext | None] | None" = None,
        event_query: "EventQuery | None" = None,
        storage: "StorageManager | None" = None,
    ):
        """Initialize agent with its own runtime.

        Args:
            llm: LLM client to use. Omit to enable cascading resolution.
            truncation: Optional truncation configuration.
            render_config: RenderConfig for block/provider formatter selection.
            context: Instance-level context block overrides.
                - str: Static content
                - DynamicContext("expr"): DynamicContext expression, re-evaluated each turn
                - None: Remove block
            event_query: Instance-level EventQuery for filtering events in context.
            storage: Optional StorageManager for persistence. Defaults to
                InMemoryStorageManager (no persistence, same as current behavior).

        Core attributes (all hidden from LLM):
        - context_manager: ContextManager — raw context block state
        - event_manager: EventManager — event bus for callbacks
        - runtime: RuntimeServices provider (generate, execute_code)

        LLM-facing wrappers (always present, hidden from LLM by default):
        - context: ContextApi — dict-like prompt context API
        - events: EventsApi — query past events by type/tag/text
        """
        _try_auto_enable_tracing()

        _validate_llm_param(llm, self.__class__.__name__)

        # Deferred imports — these live here (not at module top) to break
        # circular dependencies between agent.py and the runtime package.
        from nooa.runtime.actor import ActorRuntime
        from nooa.runtime.context_manager import ContextManager
        from nooa.runtime.event_manager import EventManager
        from nooa.storage import InMemoryStorageManager

        # Generate agent ID
        self._agent_id = str(uuid4())

        # Storage owns the EventBackend; the agent owns the EventManager.
        # Storage swaps repoint the manager via event_manager.set_backend().
        self._storage = storage or InMemoryStorageManager()
        self.event_manager = EventManager(backend=self._storage.event_backend)

        # Resolve LLM client with cascading resolution
        instance_llm = None if llm is INHERIT else llm
        self._llm = self._resolve_llm(instance_llm)

        # Resolve and store truncation config (with merge semantics)
        self._truncation = self._resolve_truncation(truncation)

        # Resolve render config (instance overrides class default)
        from nooa.context_blocks.render_config import RenderConfig as _RC

        self.render_config = render_config or _RC()

        # Resolve and store event query (instance overrides class-level)
        self.event_query = self._resolve_event_query(event_query)

        # Initialize context state (always present, hidden)
        self.context_manager = ContextManager()

        # Register framework blocks as protected (re-evaluated each LLM turn).
        # ``system_prompt`` and ``self`` are stable across turns — cacheable prefix.
        # ``state`` is the instance's current field values — re-evaluated each
        # turn since skills can attach at runtime and field values change.
        cm = self.context_manager
        cm.set_static_protected("system_prompt", expr="self._resolve_system_prompt()")
        cm.set_static_protected("self", expr="doc(type(self))")
        cm.set_dynamic_protected(
            "state",
            "pformat(self, max_length=50, max_string=500, max_depth=4)",
        )

        # Apply class-level context blocks (from __init_subclass__)
        class_context = getattr(self.__class__, "_agent_context_blocks", None)
        if class_context:
            self._apply_context_dict(class_context)

        # Apply instance-level context blocks (overrides class-level)
        if context:
            self._apply_context_dict(context)

        # Create ContextApi and EventsApi wrappers (always present, hidden from LLM by default).
        # Subclasses opt in per-instance: spec(self, "context", hidden=False) in __init__.
        from nooa.runtime.context import ContextApi
        from nooa.runtime.events import EventsApi

        self.context = ContextApi(self)
        self.events = EventsApi(self)

        # Create runtime (manages execution, caching, signals)
        self.runtime = ActorRuntime(self)

    @no_trace
    @hidden
    def _apply_context_dict(self, blocks: "dict[str, str | DynamicContext | None]") -> None:
        """Apply a dict of context block overrides.

        Used by __init__ to apply class-level and instance-level context.
        Routing between protected and unprotected APIs is handled by
        ContextManager.apply_override().

        Args:
            blocks: Dict of overrides:
                - str: Static content
                - DynamicContext("expr"): DynamicContext expression
                - None: Remove block (including protected blocks)
        """
        for key, value in blocks.items():
            self.context_manager.apply_override(key, value)

    @no_trace
    @hidden
    def _resolve_llm(self, instance_llm: "UnifiedLLM | None") -> "UnifiedLLM":
        """Resolve which LLM to use via cascading resolution.

        Resolution order:
        1. Instance-level: MyAgent(llm=explicit_llm)
        2. Class hierarchy: class MyAgent(Agent, llm=class_llm) or inherited via MRO
        3. Runtime propagation: Parent agent via context variable
        4. Error: No LLM found

        Args:
            instance_llm: LLM passed to __init__, or None if INHERIT was used

        Returns:
            Resolved UnifiedLLM instance

        Raises:
            ValueError: If no LLM can be resolved through cascading
        """
        # 1. Instance-level explicit
        if instance_llm is not None:
            return instance_llm

        # 2. Class hierarchy (getattr walks the MRO automatically)
        class_llm = getattr(self.__class__, "_agent_llm", None)
        if class_llm is not None:
            return cast("UnifiedLLM", class_llm)

        # 3. Runtime parent propagation
        parent = _parent_agent_var.get()
        if parent is not None and hasattr(parent, "_llm"):
            return parent._llm

        # 4. No LLM found - provide helpful error
        raise ValueError(
            f"No LLM available for {self.__class__.__name__}. "
            f"Resolution attempted:\n"
            f"  1. Instance-level: Not provided\n"
            f"  2. Class hierarchy: Not set (checked full MRO)\n"
            f"  3. Runtime parent: No parent agent in context\n"
            f"\n"
            f"Solutions:\n"
            f"  - Pass llm=my_llm to __init__\n"
            f"  - Set llm=my_llm in class definition\n"
            f"  - Inherit from an Agent class with an LLM\n"
            f"  - Instantiate within a parent agent's generated code"
        )

    @no_trace
    @hidden
    def _resolve_truncation(
        self, instance_truncation: "TruncationConfig | None"
    ) -> "TruncationConfig":
        """Resolve truncation config with merge semantics.

        Resolution order (earlier merged with later, later takes precedence):
        1. Default config (DEFAULT_TRUNCATION_CONFIG)
        2. Class-level: class MyAgent(Agent, truncation=...)
        3. Instance-level: MyAgent(truncation=...)

        Args:
            instance_truncation: Truncation config passed to __init__

        Returns:
            Resolved TruncationConfig after merging all levels
        """
        # Import at runtime to avoid circular dependency
        from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG

        # Start with default config
        config = DEFAULT_TRUNCATION_CONFIG

        # Merge class-level config if present
        class_truncation = getattr(self.__class__, "_agent_truncation", None)
        if class_truncation is not None:
            config = config.merge_with(class_truncation)

        # Merge instance-level config if present
        if instance_truncation is not None:
            config = config.merge_with(instance_truncation)

        return config

    @no_trace
    @hidden
    def _resolve_event_query(
        self, instance_event_query: "EventQuery | None"
    ) -> "EventQuery | None":
        """Resolve event query with override semantics.

        Resolution order (later overrides earlier):
        1. Class-level: class MyAgent(Agent, event_query=...)
        2. Instance-level: MyAgent(event_query=...)

        Args:
            instance_event_query: EventQuery passed to __init__

        Returns:
            Resolved EventQuery or None if no query specified
        """
        # Instance-level overrides class-level
        if instance_event_query is not None:
            return instance_event_query

        # Check class-level
        class_event_query = getattr(self.__class__, "_agent_event_query", None)
        if class_event_query is not None:
            return cast("EventQuery", class_event_query)

        # No event query specified
        return None

    @property
    @hidden
    def agent_id(self) -> str:
        """Agent ID."""
        return self._agent_id

    @property
    @hidden
    def context_stats(self) -> "ContextWindowStats | None":
        """Most recent context window utilization stats, or None before first generation."""
        return self.runtime._last_context_stats

    @property
    @hidden
    def llm(self) -> "UnifiedLLM":
        """The agent's resolved LLM client.

        Public accessor for the LLM resolved at construction time (see
        ``_resolve_llm``). Framework/host code (e.g. the TUI's model
        switcher) reads this instead of reaching into ``_llm``.
        """
        return self._llm

    @no_trace
    @hidden
    def set_llm(self, llm: "UnifiedLLM") -> None:
        """Replace the agent's LLM client.

        Used by hosts that switch models at runtime (e.g. the TUI
        ``/switch`` command). Callers that maintain model-derived state
        (summarizer budgets, context-window limits) must refresh it after
        calling this — see ``apply_model_limits``.
        """
        self._llm = llm

    @no_trace
    @hidden
    def _resolve_system_prompt(self) -> str:
        """Resolve the system prompt from the class docstring.

        Walks the MRO to find the nearest class with a docstring — that
        docstring IS the system prompt. Placeholders like ``{type(self).__name__}``
        are resolved as Python expressions (same mechanism as method docstrings).

        Subclasses customize by writing a class docstring — no method override needed.
        """
        import string

        # Walk MRO to find nearest docstring (Python doesn't inherit __doc__)
        doc = ""
        for cls in type(self).__mro__:
            if cls.__doc__:
                doc = cls.__doc__
                break
        if not doc or "{" not in doc:
            return doc
        # Resolve {expr} placeholders using the agent's namespace
        formatter = string.Formatter()
        parts = []
        for literal, field, fmt_spec, conversion in formatter.parse(doc):
            parts.append(literal)
            if field is not None:
                try:
                    value = eval(field, {"self": self, "type": type})  # noqa: S307
                    if conversion == "r":
                        value = repr(value)
                    elif conversion == "s":
                        value = str(value)
                    if fmt_spec:
                        value = format(value, fmt_spec)
                    parts.append(str(value))
                except Exception:
                    placeholder = field
                    if conversion:
                        placeholder = f"{field}!{conversion}"
                    if fmt_spec:
                        placeholder = f"{placeholder}:{fmt_spec}"
                    parts.append("{" + placeholder + "}")
        return "".join(parts)

    def __setattr__(self, name: str, value: Any) -> None:
        from nooa.runtime.method_guard import guard_dynamic_method

        guard_dynamic_method(self, name, value)
        super().__setattr__(name, value)

    # -------------------------------------------------------------------------
    # agentdoc protocol implementations
    # -------------------------------------------------------------------------

    @classmethod
    def __type_info__(cls) -> "TypeInfo":
        """Return TypeInfo for this Agent class, filtering framework internals.

        This method implements the agentdoc protocol for custom type introspection.
        It uses @hidden to determine what to show/hide, instead of underscore convention.

        Returns:
            TypeInfo with filtered fields and methods
        """
        import inspect

        from nooa.agentdoc.ext import TypeInfo, extract_callable_info, extract_type_info
        from nooa.agentdoc.visibility import is_hidden_field, is_hidden_method

        # Get base type info via automatic extraction (skip protocol to avoid recursion)
        base_info = extract_type_info(cls, _skip_protocol=True)

        # Build method list: dunder and single-underscore methods are hidden by default.
        # Use @shown to opt a _private method back in.
        seen_names: set[str] = set()
        all_methods = []
        for name, value in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith("__") and name.endswith("__"):
                continue
            if name in seen_names:
                continue
            explicitly_shown = getattr(value, "_agentdoc_hidden", None) is False
            if name.startswith("_") and not explicitly_shown:
                continue
            raw = next(
                (vars(klass).get(name) for klass in cls.__mro__ if name in vars(klass)), None
            )
            if is_hidden_method(value) or (raw is not None and is_hidden_method(raw)):
                continue
            seen_names.add(name)
            all_methods.append(extract_callable_info(value))
        for name, value in inspect.getmembers(cls, predicate=inspect.ismethod):  # type: ignore[assignment]
            if name.startswith("__") and name.endswith("__"):
                continue
            if name in seen_names:
                continue
            explicitly_shown = getattr(value, "_agentdoc_hidden", None) is False
            if name.startswith("_") and not explicitly_shown:
                continue
            raw = next(
                (vars(klass).get(name) for klass in cls.__mro__ if name in vars(klass)), None
            )
            if is_hidden_method(value) or (raw is not None and is_hidden_method(raw)):
                continue
            seen_names.add(name)
            all_methods.append(extract_callable_info(value))
        source_order: dict[str, int] = {}
        idx = 0
        for klass in cls.__mro__:
            for name, value in klass.__dict__.items():
                try:
                    qualname = (
                        getattr(value, "__qualname__", None) or f"{klass.__qualname__}.{name}"
                    )
                except Exception:
                    qualname = f"{klass.__qualname__}.{name}"
                if qualname not in source_order:
                    source_order[qualname] = idx
                    idx += 1
        filtered_methods = sorted(all_methods, key=lambda m: source_order.get(m.name, idx))

        # Filter out @hidden fields (Annotated[T, hidden])
        filtered_fields = [f for f in base_info.fields if not is_hidden_field(cls, f.name)]

        return TypeInfo(
            name=base_info.name,
            base=base_info.base,
            fields=filtered_fields,
            methods=filtered_methods,
            docstring=base_info.docstring,
        )

    def __instance_values__(self) -> dict[str, Any]:
        """Return instance values, filtering framework attributes.

        This method implements the agentdoc protocol for custom instance value
        extraction. It returns only user-defined state, hiding framework internals.

        Includes both:
        - Instance attributes (from self.__dict__)
        - Class-level attributes like tools and child agent classes

        Returns:
            Dictionary of attribute name to value for documentation
        """
        import inspect

        from nooa.agentdoc.visibility import is_hidden_field, is_hidden_method

        values = {}

        # 1. Instance attributes from __dict__
        for name, value in self.__dict__.items():
            if is_hidden_field(self, name):
                continue
            # Keep non-callable values and classes (child agents); skip bound methods
            is_class = isinstance(value, type)
            if callable(value) and not is_class:
                continue
            values[name] = value

        # 2. Class-level attributes (tools, child agent classes, etc.)
        # getmembers_static avoids triggering descriptors during enumeration.
        for name, class_attr in inspect.getmembers_static(type(self)):
            if name in values or is_hidden_field(self, name):
                continue
            # @hidden @property / @hidden @cached_property: is_hidden_field only checks
            # Annotated markers; also check is_hidden_method (returns False for non-decorated).
            if is_hidden_method(class_attr):
                continue

            # Skip methods, classmethods, staticmethods
            if inspect.isfunction(class_attr) or inspect.ismethod(class_attr):
                continue
            if isinstance(class_attr, (classmethod, staticmethod)):
                continue

            # Resolve instance value (may differ from class_attr for descriptors)
            # Properties are included: call getattr on the live instance so their
            # computed value appears in doc(instance). Skip on any exception.
            try:
                value = getattr(self, name)
            except (AttributeError, TypeError):
                continue
            except Exception:
                logger.debug(
                    "Skipping attribute %r: getattr raised unexpectedly", name, exc_info=True
                )
                continue

            # Include classes (child agents) and non-callable values (tools, data)
            if isinstance(value, type) or not callable(value):
                values[name] = value

        return values
