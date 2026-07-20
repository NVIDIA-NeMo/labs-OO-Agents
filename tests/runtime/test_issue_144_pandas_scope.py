# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for issue #144.

The REPL injects module-level names from the agent's module into exec_globals
via filter_module_globals(). If the agent module has `import pandas as pd`,
then `pd` lands in scope and `import pandas as pd` (and any alias) is silently
stripped rather than raising RestrictedCodeError.

If the agent module does NOT import pandas, or imports it inside `with hidden:`,
then `pd` is correctly absent from scope and `import pandas as pd` raises
RestrictedCodeError — that is the right behaviour.

Issue #144 is caused by the KDDAgent not having `import pandas as pd` at module
level. The fix is in the agent file, not the framework. These tests pin the
framework contract so we catch regressions.
"""

import pytest

from nooa import Agent
from nooa.unifiedllm import FakeLLMClient

_TEST_LLM = FakeLLMClient()


# ---------------------------------------------------------------------------
# Helpers: two agent classes defined in modules that do / don't import pandas
# ---------------------------------------------------------------------------

# This module (test_issue_144_pandas_scope) does NOT import pandas, so agents
# defined here correctly have no pd in scope. That is used for the negative tests.


def _make_agent_with_pandas_in_scope():
    """Return an Agent instance whose module has `import pandas as pd` in globals.

    We simulate this by defining the class inside a dynamically constructed
    module that has pd in its __dict__, then patching __module__ so that
    inspect.getmodule() finds our fake module.
    """
    import sys
    import types

    # We need a real pandas module object (without actually importing inside a
    # subprocess — just find it from sys.modules if already loaded, or skip).
    pandas = sys.modules.get("pandas")
    if pandas is None:
        pytest.skip("pandas not importable in this environment")

    mod_name = "_test_agent_with_pandas"
    mod = types.ModuleType(mod_name)
    mod.pd = pandas
    mod.Agent = Agent
    mod.FakeLLMClient = FakeLLMClient
    sys.modules[mod_name] = mod

    # Define the agent class so its __module__ points at our fake module
    AgentClass = type(
        "PandasAgent",
        (Agent,),
        {"__module__": mod_name, "__qualname__": "PandasAgent"},
        llm=FakeLLMClient(),
    )
    mod.PandasAgent = AgentClass
    return AgentClass()


# ---------------------------------------------------------------------------
# Framework contract: module WITH pandas import
# ---------------------------------------------------------------------------


class TestModuleWithPandasImport:
    """When the agent module has `import pandas as pd`, the REPL must:
    - expose pd as a bare name (no import statement needed)
    - silently strip `import pandas as pd` (redundant, pd already in scope)
    - allow any alias: `import pandas as foo`
    - list pd in the 'Available in scope:' error hint
    """

    @pytest.fixture
    def agent(self):
        import sys

        inst = _make_agent_with_pandas_in_scope()
        yield inst
        sys.modules.pop("_test_agent_with_pandas", None)

    @pytest.mark.asyncio
    async def test_pd_usable_without_import(self, agent):
        """pd.DataFrame() works with no import statement in the cell."""
        result = await agent.runtime.execute_code("print(type(pd).__name__)")
        assert result.error is None, f"pd not in scope: {result.error}"
        assert "DataFrame" in result.stdout or "module" in result.stdout

    @pytest.mark.asyncio
    async def test_import_pandas_as_pd_is_silently_stripped(self, agent):
        """`import pandas as pd` is stripped (pd already present) — no error."""
        result = await agent.runtime.execute_code("import pandas as pd\nprint(pd.__name__)")
        assert result.error is None, (
            f"import pandas as pd raised an error even though pd is in scope: {result.error}"
        )
        assert "pandas" in result.stdout

    @pytest.mark.asyncio
    async def test_import_pandas_bare_is_allowed(self, agent):
        """`import pandas` is allowed because pandas is in importable_modules."""
        result = await agent.runtime.execute_code("import pandas\nprint(pandas.__name__)")
        assert result.error is None, f"import pandas raised an error: {result.error}"
        assert "pandas" in result.stdout

    @pytest.mark.asyncio
    async def test_import_pandas_with_any_alias_is_allowed(self, agent):
        """`import pandas as wtf` is allowed — the module is importable."""
        result = await agent.runtime.execute_code("import pandas as wtf\nprint(wtf.__name__)")
        assert result.error is None, f"import pandas as wtf raised an error: {result.error}"
        assert "pandas" in result.stdout

    @pytest.mark.asyncio
    async def test_blocked_import_shows_error(self, agent):
        """Blocked modules (subprocess) still produce an error."""
        result = await agent.runtime.execute_code("import subprocess")
        assert result.error is not None
        assert "subprocess" in str(result.error)
        assert "blocked" in str(result.error).lower()


# ---------------------------------------------------------------------------
# Framework contract: module WITHOUT pandas import
# ---------------------------------------------------------------------------


class TestModuleWithoutPandasImport:
    """When the agent module does NOT import pandas, the REPL must:
    - raise RestrictedCodeError for `import pandas as pd`
    - raise NameError for bare `pd.DataFrame()`
    This is correct behaviour — the agent author must add the import.
    """

    @pytest.fixture
    def agent(self):
        # This agent is defined in the current test module which has no
        # pandas import, so pd will not be in exec_globals.
        class BareAgent(Agent, llm=_TEST_LLM):
            pass

        return BareAgent()

    @pytest.mark.asyncio
    async def test_import_pandas_succeeds_with_deny_list(self, agent):
        """`import pandas as pd` succeeds under deny-list model.

        With deny-list import policy (empty restricted_imports default),
        all imports pass AST validation. If the module is installed, it works.
        If not installed, ModuleNotFoundError at runtime.
        """
        result = await agent.runtime.execute_code("import pandas as pd")
        # Either succeeds (pandas installed) or ModuleNotFoundError (not installed)
        if result.error is not None:
            assert isinstance(result.error, ModuleNotFoundError), (
                f"Expected None or ModuleNotFoundError, got: {type(result.error).__name__}: {result.error}"
            )

    @pytest.mark.asyncio
    async def test_pd_not_in_scope(self, agent):
        """Bare `pd.DataFrame()` raises NameError — correct."""
        result = await agent.runtime.execute_code("pd.DataFrame({})")
        assert isinstance(result.error, NameError), (
            f"Expected NameError, got: {type(result.error).__name__}: {result.error}"
        )

    @pytest.mark.asyncio
    async def test_pd_absent_from_scope_hint(self, agent):
        """pd must NOT appear in 'Available in scope' hint when not imported."""
        result = await agent.runtime.execute_code("import subprocess")
        assert result.error is not None
        # The hint lists available names — pd should not be in it
        error_msg = str(result.error)
        # Check that pd is not listed as a standalone word in the hint
        import re

        assert not re.search(r"\bpd\b", error_msg), (
            f"pd incorrectly appears in scope hint: {error_msg}"
        )
