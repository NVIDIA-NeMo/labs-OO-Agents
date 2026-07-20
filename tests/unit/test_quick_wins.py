# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Targeted unit tests for previously uncovered modules.

Covers:
- nooa.experimental (factory functions with FutureWarning)
- nooa.strategies.experimental (backward-compat re-export)
- nooa._visible (_Visible context manager)
- nooa_cli.commands._template (Click command)
- nooa.nemo_flow_middleware (install_nemo_flow, nemo_flow_scope, middleware)
"""

from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# nooa.experimental (canonical path)
# ---------------------------------------------------------------------------


class TestExperimentalStrategies:
    """Factory functions for experimental strategies emit FutureWarning."""

    def test_pure_python_strategy_warns(self):
        from nooa.experimental import PurePythonStrategy

        with pytest.warns(FutureWarning, match="experimental"):
            strategy = PurePythonStrategy()

        from nooa.strategies.pure_python import (
            PurePythonStrategy as _Real,
        )

        assert isinstance(strategy, _Real)

    def test_codeact_lite_strategy_warns(self):
        from nooa.experimental import CodeActLiteStrategy

        with pytest.warns(FutureWarning, match="experimental"):
            strategy = CodeActLiteStrategy()

        from nooa.strategies.codeact_lite import (
            CodeActLiteStrategy as _Real,
        )

        assert isinstance(strategy, _Real)

    def test_reflexion_strategy_warns(self):
        from nooa.experimental import ReflexionStrategy

        with pytest.warns(FutureWarning, match="experimental"):
            strategy = ReflexionStrategy()

        from nooa.strategies.reflexion import (
            ReflexionStrategy as _Real,
        )

        assert isinstance(strategy, _Real)

    def test_warning_message_contains_alternatives(self):
        from nooa.experimental import PurePythonStrategy

        with pytest.warns(FutureWarning, match="CodeActStrategy"):
            PurePythonStrategy()

    def test_pure_python_strategy_kwargs_forwarded(self):
        from nooa.experimental import PurePythonStrategy

        with pytest.warns(FutureWarning):
            strategy = PurePythonStrategy(max_iterations=5)

        assert strategy.max_iterations == 5

    def test_all_exports(self):
        import nooa.experimental as exp

        assert "PurePythonStrategy" in exp.__all__
        assert "CodeActLiteStrategy" in exp.__all__
        assert "ReflexionStrategy" in exp.__all__

    def test_strategies_experimental_still_works(self):
        """nooa.strategies.experimental is a backward-compat re-export."""
        from nooa.strategies.experimental import PurePythonStrategy

        with pytest.warns(FutureWarning, match="experimental"):
            strategy = PurePythonStrategy()

        from nooa.strategies.pure_python import PurePythonStrategy as _Real

        assert isinstance(strategy, _Real)


# ---------------------------------------------------------------------------
# nooa._visible
# ---------------------------------------------------------------------------


class TestVisible:
    """_Visible is a no-op context manager."""

    def test_visible_is_singleton(self):
        from nooa._visible import visible

        assert visible is not None

    def test_context_manager_enter_returns_self(self):
        from nooa._visible import visible

        with visible as v:
            assert v is visible

    def test_context_manager_exit_returns_false(self):
        from nooa._visible import _Visible

        instance = _Visible()
        result = instance.__exit__(None, None, None)
        assert result is False

    def test_repr(self):
        from nooa._visible import visible

        assert repr(visible) == "visible"

    def test_no_op_nested(self):
        from nooa._visible import visible

        # Nested use is fine — should not raise
        with visible:
            with visible:
                pass

    def test_exit_with_exception_does_not_suppress(self):
        from nooa._visible import _Visible

        instance = _Visible()
        # __exit__ returning False means exceptions propagate
        assert instance.__exit__(ValueError, ValueError("test"), None) is False


# ---------------------------------------------------------------------------
# nooa_cli.commands._template
# ---------------------------------------------------------------------------


class TestTemplateCommand:
    """Click _template command exercises basic hello-world logic."""

    def _get_command(self):
        from nooa_cli.commands._template import command

        return command

    def test_default_target(self):
        runner = CliRunner()
        result = runner.invoke(self._get_command(), [])
        assert result.exit_code == 0
        assert "Hello, world!" in result.output

    def test_custom_target(self):
        runner = CliRunner()
        result = runner.invoke(self._get_command(), ["Alice"])
        assert result.exit_code == 0
        assert "Hello, Alice!" in result.output

    def test_verbose_flag(self):
        runner = CliRunner()
        result = runner.invoke(self._get_command(), ["--verbose"])
        assert result.exit_code == 0
        assert "Hello, world!" in result.output
        assert "(verbose mode)" in result.output

    def test_verbose_short_flag(self):
        runner = CliRunner()
        result = runner.invoke(self._get_command(), ["-v"])
        assert result.exit_code == 0
        assert "(verbose mode)" in result.output

    def test_custom_target_verbose(self):
        runner = CliRunner()
        result = runner.invoke(self._get_command(), ["Bob", "--verbose"])
        assert result.exit_code == 0
        assert "Hello, Bob!" in result.output
        assert "(verbose mode)" in result.output

    def test_no_verbose_flag_no_verbose_output(self):
        runner = CliRunner()
        result = runner.invoke(self._get_command(), [])
        assert "(verbose mode)" not in result.output


# ---------------------------------------------------------------------------
# nooa.nemo_flow_middleware
# ---------------------------------------------------------------------------


def _make_fake_nemo_flow():
    """Build a MagicMock that looks like nemo_flow."""
    fake = MagicMock()
    # scope.scope() needs to work as a context manager
    fake_handle = MagicMock()
    fake_handle.uuid = "test-uuid-1234"
    fake.scope.scope.return_value.__enter__ = MagicMock(return_value=fake_handle)
    fake.scope.scope.return_value.__exit__ = MagicMock(return_value=False)
    return fake, fake_handle


@contextmanager
def _nemo_flow_patched():
    """Patch sys.modules with a fake nemo_flow, reload nemo_flow_middleware, yield module."""

    fake_nemo_flow, _ = _make_fake_nemo_flow()
    fake_llm_request = MagicMock()

    # Ensure the module is imported before patching (KeyError if not in sys.modules)
    import nooa.nemo_flow_middleware as _nm_ensure  # noqa: F811, F401

    with patch.dict(
        sys.modules,
        {
            "nemo_relay": fake_nemo_flow,
            "nemo_relay.LLMRequest": fake_llm_request,
        },
    ):
        nm = sys.modules["nooa.nemo_flow_middleware"]
        importlib.reload(nm)
        try:
            yield nm, fake_nemo_flow
        finally:
            pass  # Don't reload here — still inside patch.dict so nemo_flow is present

    # Reload AFTER patch.dict exits (nemo_flow removed from sys.modules)
    importlib.reload(nm)


class TestNemoFlowMiddlewareWithoutNemoFlow:
    """When nemo_flow is not installed, install_nemo_flow and nemo_flow_scope raise ImportError."""

    def test_install_nemo_flow_raises_import_error(self, monkeypatch):
        import nooa.nemo_flow_middleware as nm

        monkeypatch.setattr(nm, "_HAS_NEMO_FLOW", False)
        with pytest.raises(ImportError, match="nemo_flow"):
            nm.install_nemo_flow(MagicMock())

    async def test_nemo_flow_scope_raises_import_error(self, monkeypatch):
        import nooa.nemo_flow_middleware as nm

        monkeypatch.setattr(nm, "_HAS_NEMO_FLOW", False)
        with pytest.raises(ImportError, match="nemo_flow"):
            async with nm.nemo_flow_scope(MagicMock(), "test"):
                pass


class TestNemoFlowMiddlewareWithFakeNemoFlow:
    """Tests with nemo_flow mocked in sys.modules."""

    def test_has_nemo_flow_true_when_patched(self):
        with _nemo_flow_patched() as (nm, _fake):
            assert nm._HAS_NEMO_FLOW is True

    def test_install_nemo_flow_registers_middleware(self):
        with _nemo_flow_patched() as (nm, _fake):
            event_manager = MagicMock()
            event_manager.intercept.return_value = MagicMock()

            uninstall = nm.install_nemo_flow(event_manager)

            # intercept should have been called 3 times (agent, llm, execute_python)
            assert event_manager.intercept.call_count == 3
            assert callable(uninstall)

    def test_install_nemo_flow_uninstall_calls_all_unsubs(self):
        with _nemo_flow_patched() as (nm, _fake):
            unsub1 = MagicMock()
            unsub2 = MagicMock()
            unsub3 = MagicMock()
            event_manager = MagicMock()
            event_manager.intercept.side_effect = [unsub1, unsub2, unsub3]

            uninstall = nm.install_nemo_flow(event_manager)
            uninstall()

            unsub1.assert_called_once()
            unsub2.assert_called_once()
            unsub3.assert_called_once()

    async def test_nemo_flow_scope_installs_and_uninstalls(self):
        with _nemo_flow_patched() as (nm, fake_nemo_flow):
            agent = MagicMock()
            event_manager = MagicMock()
            agent.event_manager = event_manager
            unsub = MagicMock()
            event_manager.intercept.return_value = unsub

            async with nm.nemo_flow_scope(agent, "test-scope"):
                # We're inside the scope; intercepts should be installed
                assert event_manager.intercept.call_count == 3

            # After exit, all unsubs called
            assert unsub.call_count == 3

    def test_sensitive_keys_constant(self):
        import nooa.nemo_flow_middleware as nm

        assert "api_key" in nm._SENSITIVE_KEYS
        assert "api_base" in nm._SENSITIVE_KEYS
        assert "base_url" in nm._SENSITIVE_KEYS

    def test_non_serializable_keys_constant(self):
        import nooa.nemo_flow_middleware as nm

        assert "tools" in nm._NON_SERIALIZABLE_KEYS
        assert "output_model" in nm._NON_SERIALIZABLE_KEYS
