# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified module restrictions for agent code execution.

Single source of truth for blocked modules, blocked calls, and restricted modules.
Consumed by CodeActConfig (defaults), exec_globals stripping, and BlockingCallValidator.
"""

import types
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

# Process-global override for restricted_imports.
# When set (not None), RestrictionsConfig() uses this instead of its field default.
_global_restricted_imports: frozenset[str] | None = None


def set_restricted_imports(modules: frozenset[str] | None) -> None:
    """Set the process-global import restriction list.

    This overrides the default ``restricted_imports`` on every
    ``RestrictionsConfig()`` constructed after this call.

    Args:
        modules: frozenset of module names to deny, or None to clear
                 the override (revert to RestrictionsConfig field default).

    Example::

        from nooa.runtime.restrictions import (
            set_restricted_imports, DEFAULT_RESTRICTED_IMPORTS,
        )

        # Deny a small set of dangerous modules
        set_restricted_imports(DEFAULT_RESTRICTED_IMPORTS)

        # Allow everything
        set_restricted_imports(frozenset())

        # Clear override (use RestrictionsConfig field default)
        set_restricted_imports(None)
    """
    global _global_restricted_imports
    _global_restricted_imports = modules


def get_restricted_imports() -> frozenset[str] | None:
    """Return the process-global restricted_imports override, or None if unset."""
    return _global_restricted_imports


# Fully blocked — stripped from exec_globals, block the event loop.
# These modules have no legitimate async use in CodeAct.
DEFAULT_BLOCKED_MODULES: frozenset[str] = frozenset(
    {
        "subprocess",
        "socket",
        "http.client",
        "urllib.request",
        "ftplib",
        "smtplib",
        "imaplib",
        "poplib",
        "telnetlib",
        "xmlrpc.client",
        "select",
        "signal",
    }
)

# Specific calls blocked on otherwise-allowed modules.
# Keys are module names, values are frozensets of blocked function/method names.
# Dotted names (e.g. "Thread.join") match class-qualified instance method calls.
# Plain names (e.g. "sleep") match module-level function calls.
DEFAULT_BLOCKED_CALLS: dict[str, frozenset[str]] = {
    "time": frozenset({"sleep"}),
    "os": frozenset({"system", "popen", "wait", "waitpid", "waitid"}),
    "threading": frozenset({"Thread.join", "Lock.acquire", "Event.wait", "Condition.wait"}),
    "multiprocessing": frozenset({"Process.join", "Queue.get", "Queue.put"}),
    "asyncio": frozenset({"run", "run_coroutine_threadsafe", "run_until_complete", "run_forever"}),
    "importlib": frozenset({"import_module"}),
}


# Tier 2: restricted — denied at AST import validation but not stripped from namespace.
# Empty by default (all imports allowed in sandboxed environments).
# Developers can set DEFAULT_RESTRICTED_IMPORTS for a small deny list,
# or RESTRICTED_MODULES for strict lockdown.
DEFAULT_RESTRICTED_IMPORTS: frozenset[str] = frozenset()


class RestrictionsConfig(BaseModel):
    """Reusable config for code execution restrictions.

    Three-tier module restriction model:

    Tier 1 — **blocked** (hard block):
        Stripped from exec_globals AND blocked at AST import validation.
        Event-loop hazards: subprocess, socket, etc.

    Tier 2 — **restricted** (soft block):
        Blocked at AST import validation only.
        Modules with side effects that shouldn't be casually imported.

    Tier 3 — **allowed** (everything else):
        Any installed module can be imported freely.

    Keeps defaults co-located with the constants they reference.
    Embed in strategy configs (e.g. CodeActConfig) for composition.
    """

    model_config = ConfigDict(frozen=True)

    blocked_modules: frozenset[str] = DEFAULT_BLOCKED_MODULES
    blocked_calls: dict[str, frozenset[str]] = DEFAULT_BLOCKED_CALLS
    restricted_imports: frozenset[str] = frozenset()

    @model_validator(mode="wrap")
    @classmethod
    def _apply_global_restricted_imports(cls, data: Any, handler: Any) -> "RestrictionsConfig":
        """Apply process-global restricted_imports if not explicitly set."""
        if isinstance(data, dict) and "restricted_imports" not in data:
            override = _global_restricted_imports
            if override is not None:
                data["restricted_imports"] = override
        result = handler(data)
        # For non-dict inputs (e.g. model instances), apply override post-validation
        # if restricted_imports still has the field default
        if not isinstance(data, dict) and result.restricted_imports == frozenset():
            override = _global_restricted_imports
            if override is not None:
                result = result.model_copy(update={"restricted_imports": override})
        return result


def match_blocked_module(
    module_name: str,
    lookup: frozenset[str] | dict[str, frozenset[str]],
) -> str | None:
    """Match a module name against a lookup, checking parent modules too.

    Matches exact names and child-of-parent (e.g. "asyncio.runners" matches
    "asyncio" in the lookup). Does NOT match parents of entries — if
    "http.client" is in the lookup, "http" alone does not match.

    Returns the matched key or None.
    """
    if module_name in lookup:
        return module_name
    parts = module_name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        parent = ".".join(parts[:i])
        if parent in lookup:
            return parent
    return None


def is_from_blocked_module(obj: Any, blocked_modules: frozenset[str]) -> bool:
    """Check if an object originates from a blocked module.

    For module objects, checks obj.__name__. For other objects (functions,
    classes), checks obj.__module__. This is safe with the curated
    DEFAULT_BLOCKED_MODULES list, but could over-strip if a broadly-used
    module like "io" were added, since builtins like open() have
    __module__="io".
    """
    module_name: str | None
    if isinstance(obj, types.ModuleType):
        module_name = obj.__name__
    else:
        module_name = getattr(obj, "__module__", None)
    if not module_name:
        return False
    return match_blocked_module(module_name, blocked_modules) is not None


# Restricted — require explicit declaration (allow_imports or normal import).
# Superset of DEFAULT_BLOCKED_MODULES.
RESTRICTED_MODULES: frozenset[str] = frozenset(
    {
        # Blocked modules (all restricted too)
        *DEFAULT_BLOCKED_MODULES,
        # External resource access
        "os",
        "shutil",
        "pathlib",
        "tempfile",
        "glob",
        "requests",
        "httpx",
        "aiohttp",
        "urllib",
        "http",
        # Database
        "sqlite3",
        "psycopg2",
        "pymongo",
        # LLM SDKs
        "openai",
        "anthropic",
        "litellm",
        # System
        "sys",
        "ctypes",
        "multiprocessing",
        "threading",
    }
)
