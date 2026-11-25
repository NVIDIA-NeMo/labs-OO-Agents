# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``import nooa`` does not load the strategies or LiteLLM; the names still resolve."""

import subprocess
import sys

import pytest

_HEAVY = ("litellm", "nooa.strategies", "nooa.unifiedllm")


@pytest.mark.parametrize("statement", ["import nooa", "import nooa.runtime"])
def test_package_import_does_not_load_the_llm_stack(statement):
    code = f"{statement}\nimport sys\nprint([m for m in {_HEAVY!r} if m in sys.modules])\n"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "[]"


def test_lazy_names_resolve_to_the_defining_objects():
    import nooa
    import nooa.llm_config
    import nooa.strategies
    import nooa.unifiedllm
    from nooa import CodeActStrategy, LLMResponse, llm_config_chain, set_default_strategy

    assert CodeActStrategy is nooa.strategies.CodeActStrategy
    assert set_default_strategy is nooa.strategies.set_default_strategy
    assert LLMResponse is nooa.unifiedllm.LLMResponse
    assert llm_config_chain is nooa.llm_config.llm_config_chain
    for name in nooa._LAZY:
        assert name in nooa.__all__
        assert getattr(nooa, name) is not None


def test_unknown_attribute_still_raises():
    import nooa

    with pytest.raises(AttributeError):
        nooa.does_not_exist  # noqa: B018


@pytest.mark.parametrize("name", ["agent.Agent", "does_not_exist.x", ".agent", "a..b"])
def test_non_identifier_attribute_returns_the_getattr_default(name):
    """Only plain names fall back to a submodule import; others raise AttributeError."""
    import nooa

    assert getattr(nooa, name, None) is None


def _run(code: str) -> str:
    return subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()


# What ``nooa/__init__`` itself needs: the version, the native-Windows check
# and the SIGUSR2 debug handler (all stdlib-only).
_INIT_MODULES = {"nooa", "nooa._version", "nooa._platform_policy", "nooa._debug_handler"}


def test_subpackage_import_loads_only_stdlib_and_nooa_init():
    """``import nooa.overflow`` (stdlib-only) must not load the framework (#590)."""
    code = (
        "import sys\n"
        "before = set(sys.modules)\n"
        "import nooa.overflow\n"
        "print('\\n'.join(sorted(set(sys.modules) - before)))\n"
    )
    loaded = _run(code).splitlines()
    nooa_modules = {m for m in loaded if m == "nooa" or m.startswith("nooa.")}
    extra = {m for m in nooa_modules - _INIT_MODULES if not m.startswith("nooa.overflow")}
    assert extra == set()
    third_party = {
        m
        for m in loaded
        if m.split(".")[0] not in sys.stdlib_module_names and m not in nooa_modules
    }
    assert third_party == set()


def test_every_public_name_resolves_in_a_fresh_interpreter():
    code = (
        "import warnings\n"
        "warnings.simplefilter('ignore')\n"
        "import nooa\n"
        "missing = [n for n in nooa.__all__ if getattr(nooa, n, None) is None]\n"
        "print(missing)\n"
    )
    assert _run(code) == "[]"
    assert _run("from nooa import *\nprint(Agent.__name__)") == "Agent"


def test_reflexion_strategy_still_routes_through_experimental():
    import nooa
    import nooa.experimental

    with pytest.warns(FutureWarning):
        nooa.ReflexionStrategy()
    assert nooa.ReflexionStrategy is nooa.experimental.ReflexionStrategy


def test_submodules_resolve_as_attributes_after_bare_import():
    """``import nooa`` used to load most submodules; attribute access still works."""
    code = "import nooa\nprint(nooa.agentdoc.__name__, nooa.runtime.__name__)\n"
    assert _run(code) == "nooa.agentdoc nooa.runtime"


def test_debug_handler_is_installed_at_import():
    code = (
        "import signal\n"
        "import nooa\n"
        "from nooa.runtime import debug_handler\n"
        "print(debug_handler._handler_installed,"
        " signal.getsignal(signal.SIGUSR2) is debug_handler._debug_signal_handler)\n"
    )
    if not hasattr(__import__("signal"), "SIGUSR2"):
        pytest.skip("SIGUSR2 is POSIX-only")
    assert _run(code) == "True True"


def test_cell_namespace_has_builtin_providers_without_importing_them_first():
    code = (
        "from nooa.runtime.namespace_providers import cell_namespace_names\n"
        "names = cell_namespace_names()\n"
        "print(all(n in names for n in ('doc', 'show', 'Image', 'strategy')))\n"
    )
    assert _run(code) == "True"


@pytest.mark.parametrize(
    "module", ["nooa.errors", "nooa.llm_types", "nooa._platform", "nooa.unifiedllm"]
)
def test_module_imports_first_in_a_fresh_interpreter(module):
    """Import cycles the eager ``nooa/__init__`` used to hide (it loaded Agent first)."""
    assert _run(f"import {module}\nprint('ok')") == "ok"
