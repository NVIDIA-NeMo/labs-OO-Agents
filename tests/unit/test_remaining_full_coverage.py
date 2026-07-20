# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests targeting remaining uncovered lines across nooa.

Covers:
- agent.py: __type_info__ dedup guard, private filter, hidden check, __qualname__ exception
- agents/summarization.py: target_event_manager None guards, done-task check, non-AfterTurn, etc.
- runtime/debug_handler.py: traceback.print_stack exception path
- runtime/media_capture.py: PIL and matplotlib conversion paths
- runtime/method_wrapper.py: resolved_strategy falsy check
- strategies/generated_code.py: TypeError in _is_pydantic_model issubclass
- strategies/prefill.py: ImportError for Media import
- skill.py: _find_skill_md on non-directory
- storage/snapshot.py: is_nosnapshot_value continue path
- storage/sqlite.py: SessionAlreadyActiveError, close-on-connect-failure
- nemo_flow_middleware.py: model name extraction, _wrapper fallback return
- tools/web_publisher.py: RichOutput, WebPublisher methods
"""

from __future__ import annotations

import asyncio
import importlib
import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nooa.unifiedllm import FakeLLMClient

# =============================================================================
# agent.py — __type_info__ coverage
# =============================================================================


class TestTypeInfoDedupAndFilters:
    """Cover dedup guard, private filter, hidden check, and __qualname__ exception."""

    def test_dedup_guard_in_isfunction_loop(self):
        """Line 447: dedup guard in isfunction loop.

        Force inspect.getmembers(cls, isfunction) to yield a duplicate name by
        patching getmembers to return it twice.
        """
        import inspect

        from nooa.agent import Agent

        llm = FakeLLMClient()

        class _Dup(Agent, llm=llm):
            def tool(self) -> int:
                return 1

        # Wrap getmembers so the isfunction call returns 'tool' twice
        _real_getmembers = inspect.getmembers

        def _patched_getmembers(cls, predicate=None):
            result = _real_getmembers(cls, predicate)
            if predicate is inspect.isfunction:
                # Duplicate the 'tool' entry to trigger the seen_names guard
                tool_entries = [(n, v) for n, v in result if n == "tool"]
                if tool_entries:
                    result = result + tool_entries  # add duplicate
            return result

        with patch("inspect.getmembers", side_effect=_patched_getmembers):
            info = _Dup.__type_info__()
        names = [m.name for m in info.methods]
        tool_matches = [n for n in names if "tool" in n]
        # tool should appear only once despite the duplicate
        assert len(tool_matches) == 1

    def test_dedup_guard_in_ismethod_loop(self):
        """Line 462: dedup guard in ismethod loop.

        A public classmethod appears in ismethod. If its name was already in
        seen_names (from isfunction loop), the dedup guard skips it.
        We achieve this by injecting the name into the isfunction results.
        """
        import inspect

        from nooa.agent import Agent

        llm = FakeLLMClient()

        class _DedupM(Agent, llm=llm):
            @classmethod
            def my_cm(cls) -> str:
                return "hi"

        # Wrap getmembers so isfunction returns 'my_cm' too (normally only ismethod has it)
        _real_getmembers = inspect.getmembers

        def _patched_getmembers(cls, predicate=None):
            result = _real_getmembers(cls, predicate)
            if predicate is inspect.isfunction:
                # Inject a fake 'my_cm' as if isfunction found it
                cm_val = getattr(cls, "my_cm", None)
                if cm_val is not None:
                    # Use the underlying function to pass isfunction check context
                    result = result + [("my_cm", cm_val)]
            return result

        with patch("inspect.getmembers", side_effect=_patched_getmembers):
            info = _DedupM.__type_info__()
        names = [m.name for m in info.methods]
        cm_matches = [n for n in names if "my_cm" in n]
        # Should appear exactly once — second occurrence deduped
        assert len(cm_matches) == 1

    def test_private_method_filtered_in_ismethod_loop(self):
        """Line 464-465: private classmethod is filtered by the underscore check."""
        from nooa.agent import Agent

        llm = FakeLLMClient()

        class _AgentPM(Agent, llm=llm):
            @classmethod
            def _private_cm(cls) -> str:
                return "hidden"

            def visible(self) -> int:
                return 1

        info = _AgentPM.__type_info__()
        names = [m.name for m in info.methods]
        assert not any("_private_cm" in n for n in names)
        assert any("visible" in n for n in names)

    def test_hidden_classmethod_filtered_in_ismethod_loop(self):
        """Line 469-472: @hidden classmethod is filtered by is_hidden_method."""
        from nooa.agent import Agent
        from nooa.agentdoc import hidden as _hidden

        llm = FakeLLMClient()

        class _AgentHCM(Agent, llm=llm):
            @classmethod
            @_hidden
            def hidden_cm(cls) -> str:
                return "nope"

            def visible(self) -> int:
                return 1

        info = _AgentHCM.__type_info__()
        names = [m.name for m in info.methods]
        assert not any("hidden_cm" in n for n in names)

    def test_public_classmethod_passes_all_filters_in_ismethod_loop(self):
        """Lines 473-474: a public, non-hidden classmethod passes all filters.

        This classmethod should appear in ismethod but NOT isfunction,
        so it passes the dedup guard (line 462), the underscore check (line 464),
        and the hidden check (line 471), reaching lines 473-474.
        """
        from nooa.agent import Agent

        llm = FakeLLMClient()

        class _AgentCM(Agent, llm=llm):
            @classmethod
            def public_cm(cls) -> str:
                """A public classmethod that should be visible."""
                return "visible"

            def tool(self) -> int:
                return 1

        info = _AgentCM.__type_info__()
        names = [m.name for m in info.methods]
        assert any("public_cm" in n for n in names)
        assert any("tool" in n for n in names)

    def test_qualname_exception_in_source_order(self):
        """Lines 483-484: Exception in getattr __qualname__ triggers the fallback.

        We inject an object whose __getattr__ raises RuntimeError for
        __qualname__. Since getattr(obj, "__qualname__", None) propagates
        non-AttributeError exceptions, the except branch in __type_info__
        is triggered.
        """
        from nooa.agent import Agent

        llm = FakeLLMClient()

        class _QualnameBomb:
            """Object that raises RuntimeError on __qualname__ access."""

            def __getattr__(self, name):
                if name == "__qualname__":
                    raise RuntimeError("boom")
                raise AttributeError(name)

        class _AgentBD(Agent, llm=llm):
            def tool(self) -> int:
                return 1

        # Inject the bomb into the class dict so __type_info__ iterates over it
        bomb = _QualnameBomb()
        type.__setattr__(_AgentBD, "_bomb_val", bomb)

        # Should not raise — the except branch handles the RuntimeError
        info = _AgentBD.__type_info__()
        names = [m.name for m in info.methods]
        assert any("tool" in n for n in names)


# =============================================================================
# agents/summarization.py — None guards and edge cases
# =============================================================================


class TestSummarizationNoneGuards:
    """Cover target_event_manager-is-None guard lines in summarization.py."""

    def _make_summarizer(self):
        """Create a TokenBudgetSummarizer attached to a parent agent."""
        from nooa.agent import Agent
        from nooa.agents.summarization import TokenBudgetSummarizer
        from nooa.config.summarizer_config import TokenBudgetConfig

        llm = FakeLLMClient()

        class _Parent(Agent, llm=llm):
            async def chat(self, msg: str) -> str:
                """Chat."""
                ...

        agent = _Parent()
        config = TokenBudgetConfig(max_tokens=50_000)
        summarizer = TokenBudgetSummarizer.install(agent, config=config)
        return summarizer

    def test_install_raises_when_target_event_manager_is_none(self):
        """Line 151: _install raises ValueError when target_event_manager is None.

        We create a summarizer normally, then set target_event_manager to None
        and call _install directly.
        """
        summarizer = self._make_summarizer()
        # Uninstall existing subscriptions first
        summarizer._uninstall()
        # Force target_event_manager to None
        summarizer.target_event_manager = None
        with pytest.raises(ValueError, match="target_event_manager is None"):
            summarizer._install()

    def test_schedule_summarization_none_guard(self):
        """Line 278: _schedule_summarization returns early if target_event_manager is None."""
        summarizer = self._make_summarizer()
        summarizer.target_event_manager = None
        # Should return without error
        summarizer._schedule_summarization("t1", "t5")
        assert summarizer._pending_task is None

    def test_get_events_in_range_none_guard(self):
        """Line 353: _get_events_in_range returns [] if target_event_manager is None."""
        summarizer = self._make_summarizer()
        summarizer.target_event_manager = None
        result = summarizer._get_events_in_range("t1", "t5")
        assert result == []

    def test_render_range_to_markdown_none_guard(self):
        """Line 386: _render_range_to_markdown returns '' if target_event_manager is None."""
        summarizer = self._make_summarizer()
        summarizer.target_event_manager = None
        result = summarizer._render_range_to_markdown("t1", "t5")
        assert result == ""

    def test_render_range_to_markdown_empty_events(self):
        """Line 391: _render_range_to_markdown returns '' if events list is empty."""
        summarizer = self._make_summarizer()
        # Patch _get_events_in_range to return empty list
        summarizer._get_events_in_range = MagicMock(return_value=[])  # type: ignore[method-assign]
        result = summarizer._render_range_to_markdown("t1", "t5")
        assert result == ""

    def test_token_budget_compute_range_none_guard(self):
        """Line 514: TokenBudgetSummarizer._compute_range returns None when target_event_manager is None."""
        summarizer = self._make_summarizer()
        summarizer.target_event_manager = None
        mock_event = MagicMock()
        result = summarizer._compute_range(mock_event)
        assert result is None

    def test_method_summarizer_compute_range_none_guard(self):
        """Line 585: MethodSummarizer._compute_range returns None when target_event_manager is None."""
        from nooa.agent import Agent
        from nooa.agents.summarization import MethodSummarizer
        from nooa.config.summarizer_config import MethodSummarizerConfig

        llm = FakeLLMClient()

        class _Parent(Agent, llm=llm):
            async def chat(self, msg: str) -> str:
                """Chat."""
                ...

        agent = _Parent()
        summarizer = MethodSummarizer.install(agent, config=MethodSummarizerConfig())
        summarizer.target_event_manager = None
        mock_event = MagicMock()
        mock_event.metadata = {"call_id": "abc"}
        result = summarizer._compute_range(mock_event)
        assert result is None


class TestSummarizationKwargsOverride:
    """Line 127: kwargs override extraction in __init__."""

    def test_kwargs_override_sets_attribute(self):
        """SummarizationAgent.__init__ extracts class attributes from kwargs."""
        from nooa.agent import Agent
        from nooa.agents.summarization import SummarizationAgent

        llm = FakeLLMClient()

        class _Parent(Agent, llm=llm):
            async def chat(self, msg: str) -> str:
                """Chat."""
                ...

        agent = _Parent()
        # SummarizationAgent has 'config' as a class attribute;
        # passing it in kwargs should trigger the extraction path (line 127)
        summarizer = SummarizationAgent(agent, config="custom_config_value")
        assert summarizer.config == "custom_config_value"


class TestSummarizationPendingDoneAndNotAfterTurn:
    """Lines 198 and 201: pending task done check and not-AfterTurn check."""

    def _make_summarizer(self):
        from nooa.agent import Agent
        from nooa.agents.summarization import TokenBudgetSummarizer
        from nooa.config.summarizer_config import TokenBudgetConfig

        llm = FakeLLMClient()

        class _Parent(Agent, llm=llm):
            async def chat(self, msg: str) -> str:
                """Chat."""
                ...

        agent = _Parent()
        return TokenBudgetSummarizer.install(agent, config=TokenBudgetConfig())

    def test_before_turn_clears_done_pending_task(self):
        """_handle_before_turn is where pending-summary state gets cleared.

        AfterTurn intentionally doesn't touch pending state (that ordering
        is what prevents the stale-stats cascade), so clearing is the
        exclusive responsibility of BeforeTurn.
        """
        summarizer = self._make_summarizer()
        loop = asyncio.new_event_loop()
        try:
            future = loop.create_future()
            future.set_result(None)
            summarizer._pending_task = future  # type: ignore[assignment]

            from nooa.events import BeforeTurn

            event = BeforeTurn(
                method_name="chat",
                strategy="CodeAct",
                generation_id="gen-1",
                turn_number=1,
            )
            summarizer._handle_before_turn(event)
            assert summarizer._pending_task is None
        finally:
            loop.close()

    def test_after_turn_preserves_pending_state(self):
        """AfterTurn must not clear pending state — a done-but-unapplied
        summary needs to survive until BeforeTurn applies it."""
        summarizer = self._make_summarizer()
        loop = asyncio.new_event_loop()
        try:
            future = loop.create_future()
            future.set_result(None)
            summarizer._pending_task = future  # type: ignore[assignment]

            from nooa.events import AfterTurn

            event = AfterTurn(
                method_name="chat",
                strategy="CodeAct",
                generation_id="gen-1",
                turn_number=1,
                is_final=False,
            )
            summarizer._handle_after_turn(event)
            # Pending state untouched; BeforeTurn will handle it.
            assert summarizer._pending_task is future
        finally:
            loop.close()

    def test_non_after_turn_event_returns_early(self):
        """Line 201: _handle_after_turn returns early for non-AfterTurn events."""
        summarizer = self._make_summarizer()

        # Pass a generic event that is NOT AfterTurn
        mock_event = MagicMock()
        # Make it explicitly not an AfterTurn instance
        mock_event.__class__ = type("FakeEvent", (), {})

        # Should not raise — just return early
        summarizer._handle_after_turn(mock_event)


# =============================================================================
# runtime/debug_handler.py — traceback.print_stack exception
# =============================================================================


class TestDebugHandlerTracebackException:
    """Lines 279-280: traceback.print_stack raises an exception."""

    def test_print_stack_exception_handled(self, tmp_path):
        """When traceback.print_stack raises, the error is written to the file."""
        from nooa.runtime.debug_handler import _debug_signal_handler

        # Mock frame
        mock_frame = MagicMock()

        with (
            patch("nooa.runtime.debug_handler._get_debug_dump_path") as mock_path,
            patch("nooa.runtime.debug_handler._detect_llm_in_stack", return_value=[]),
            patch("nooa.runtime.debug_handler._pending_llm_calls", {}),
            patch("traceback.print_stack", side_effect=RuntimeError("frame is gone")),
            patch("nooa.runtime.debug_handler._dump_pending_llm_calls"),
            patch("nooa.runtime.debug_handler._dump_cell_code"),
        ):
            dump_path = str(tmp_path / "debug_dump.txt")
            mock_path.return_value = dump_path

            _debug_signal_handler(12, mock_frame)

            content = (tmp_path / "debug_dump.txt").read_text()
            assert "Error getting traceback: frame is gone" in content


# =============================================================================
# runtime/media_capture.py — PIL and matplotlib conversion paths
# =============================================================================


class TestMediaCapturePIL:
    """Line 113: PIL image conversion path in _try_auto_convert."""

    def test_pil_image_to_content_block(self):
        """PIL Image is converted to an image_url content block."""
        pytest.importorskip("PIL")
        from PIL import Image as PILImage

        from nooa.runtime.media_capture import _try_pil_to_content_block

        img = PILImage.new("RGB", (10, 10), color="red")
        result = _try_pil_to_content_block(img)
        assert result is not None
        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/png;base64,")

    def test_auto_convert_pil_returns_block(self):
        """_try_auto_convert (aliased as _to_content_block) returns PIL block."""
        pytest.importorskip("PIL")
        from PIL import Image as PILImage

        from nooa.runtime.media_capture import _to_content_block

        img = PILImage.new("RGB", (10, 10), color="blue")
        result = _to_content_block(img)
        assert result is not None
        assert result["type"] == "image_url"


class TestMediaCaptureMatplotlib:
    """Lines 145-148: matplotlib Figure conversion path."""

    def test_matplotlib_figure_to_content_block(self):
        """matplotlib Figure is converted to an image_url content block."""
        pytest.importorskip("matplotlib")
        from matplotlib.figure import Figure

        from nooa.runtime.media_capture import _try_matplotlib_to_content_block

        fig = Figure(figsize=(2, 2))
        ax = fig.add_subplot(111)
        ax.plot([1, 2, 3], [1, 2, 3])

        result = _try_matplotlib_to_content_block(fig)
        assert result is not None
        assert result["type"] == "image_url"
        assert result["image_url"]["url"].startswith("data:image/png;base64,")
        assert result["image_url"]["format"] == "image/png"

    def test_auto_convert_matplotlib_returns_block(self):
        """_try_auto_convert falls through PIL to matplotlib and returns block."""
        pytest.importorskip("matplotlib")
        from matplotlib.figure import Figure

        from nooa.runtime.media_capture import _to_content_block

        fig = Figure(figsize=(2, 2))
        result = _to_content_block(fig)
        assert result is not None
        assert result["type"] == "image_url"


# =============================================================================
# runtime/method_wrapper.py — resolved_strategy falsy check
# =============================================================================


class TestMethodWrapperResolvedStrategyFalsy:
    """Line 251: resolved_strategy falsy raises ValueError."""

    @pytest.mark.asyncio
    async def test_strategy_method_without_strategy_raises(self):
        """When needs_generation=True and args[0] is RuntimeServices but the
        resolved strategy is falsy, the wrapper raises ValueError (line 251).

        We patch get_default_strategy to return None to make auto-resolution
        produce a falsy strategy.
        """
        from nooa.runtime.method_wrapper import create_agent_method_wrapper
        from nooa.strategies.base import RuntimeServices

        async def my_method(self, runtime: RuntimeServices, x: int) -> int:
            """Do something."""
            ...

        # Create a mock that satisfies isinstance(obj, RuntimeServices) check.
        mock_runtime = MagicMock(spec=RuntimeServices)

        wrapper = create_agent_method_wrapper(
            my_method,
            needs_generation=True,
            needs_tracing=False,
            strategy=None,
        )

        # self must NOT have 'runtime' attribute to reach the elif branch
        class _BareObj:
            pass

        bare = _BareObj()

        # Patch get_default_strategy to return None so resolved_strategy stays falsy
        with patch("nooa.strategies.get_default_strategy", return_value=None):
            with pytest.raises(ValueError, match="requires strategy parameter"):
                await wrapper(bare, mock_runtime, 42)


# =============================================================================
# strategies/generated_code.py — TypeError in _is_pydantic_model issubclass
# =============================================================================


class TestIsPydanticModelEdgeCases:
    """_is_pydantic_model returns False for non-types and non-BaseModel types."""

    def test_non_type_returns_false(self):
        """A string is not a type — returns False without calling issubclass."""
        from nooa.strategies.generated_code import ReturnValueValidator

        validator = ReturnValueValidator()
        assert validator._is_pydantic_model("not_a_type") is False

    def test_plain_class_returns_false(self):
        """A plain class that isn't a BaseModel returns False."""
        from nooa.strategies.generated_code import ReturnValueValidator

        class NotAModel:
            pass

        validator = ReturnValueValidator()
        assert validator._is_pydantic_model(NotAModel) is False

    def test_pydantic_model_returns_true(self):
        """A BaseModel subclass returns True."""
        from pydantic import BaseModel

        from nooa.strategies.generated_code import ReturnValueValidator

        class MyModel(BaseModel):
            x: int

        validator = ReturnValueValidator()
        assert validator._is_pydantic_model(MyModel) is True


# =============================================================================
# strategies/prefill.py — ImportError for Media import
# =============================================================================


class TestPrefillMediaImportError:
    """Lines 30-31: _is_media returns False when Media import fails."""

    def test_is_media_false_on_import_error(self):
        """_is_media returns False when nooa.media cannot be imported."""
        from nooa.strategies.prefill import _is_media

        # Patch the import to fail
        with patch.dict(sys.modules, {"nooa.media": None}):
            # importlib will raise ImportError for None entries in sys.modules
            result = _is_media("some_string")
            assert result is False

    def test_is_media_false_for_non_media_object(self):
        """_is_media returns False for ordinary objects."""
        from nooa.strategies.prefill import _is_media

        assert _is_media(42) is False
        assert _is_media("hello") is False


# =============================================================================
# skill.py — _find_skill_md on non-directory
# =============================================================================


class TestFindSkillMdNonDirectory:
    """Line 42: _find_skill_md returns None for a non-directory path."""

    def test_non_directory_returns_none(self, tmp_path):
        """Passing a file (not directory) to _find_skill_md returns None."""
        from nooa.skill import _find_skill_md

        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("content")
        result = _find_skill_md(file_path)
        assert result is None

    def test_nonexistent_path_returns_none(self, tmp_path):
        """Passing a non-existent path to _find_skill_md returns None."""
        from nooa.skill import _find_skill_md

        result = _find_skill_md(tmp_path / "does_not_exist")
        assert result is None


# =============================================================================
# storage/snapshot.py — is_nosnapshot_value continue path
# =============================================================================


class TestSnapshotNosnaphotValue:
    """Line 107: from_agent skips attributes with __nosnapshot__ class."""

    def test_nosnapshot_value_attribute_is_skipped(self):
        """An attribute whose type has __nosnapshot__=True is excluded from snapshot."""
        from nooa.agent import Agent
        from nooa.storage.snapshot import AgentSnapshot

        llm = FakeLLMClient()

        class _NosnapObj:
            __nosnapshot__ = True

        class _TestAgent(Agent, llm=llm):
            value: int = 0

        agent = _TestAgent()
        agent.value = 42
        agent.__dict__["transient"] = _NosnapObj()

        snap = AgentSnapshot.from_agent(agent)
        assert "transient" not in snap.attributes
        assert snap.attributes["value"] == 42


# =============================================================================
# storage/sqlite.py — SessionAlreadyActiveError and close-on-connect-failure
# =============================================================================


class TestSQLiteSessionLocking:
    """Lines 352-355: SessionAlreadyActiveError when lock is held."""

    def test_session_already_active_error(self, tmp_path):
        """Opening the same db twice raises SessionAlreadyActiveError."""
        from nooa.storage.sqlite import SessionAlreadyActiveError, SQLiteStorageManager

        db_path = tmp_path / "test.db"
        session1 = SQLiteStorageManager(db_path=db_path)
        try:
            with pytest.raises(SessionAlreadyActiveError, match="already active"):
                SQLiteStorageManager(db_path=db_path)
        finally:
            session1.close()

    def test_session_already_active_error_reports_owner_pid(self, tmp_path):
        """The error object carries the owner pid so callers (and users) can
        identify which process is holding the lock and decide what to do.
        """
        from nooa.storage.sqlite import SessionAlreadyActiveError, SQLiteStorageManager

        db_path = tmp_path / "test.db"
        session1 = SQLiteStorageManager(db_path=db_path)
        try:
            with pytest.raises(SessionAlreadyActiveError) as exc_info:
                SQLiteStorageManager(db_path=db_path)
            err = exc_info.value
            # The lock file was just written by this same process.
            assert err.owner_pid == os.getpid()
            assert err.session_id == "test"
            assert f"pid {os.getpid()}" in str(err)
        finally:
            session1.close()

    def test_lock_file_truncates_prior_pid(self, tmp_path):
        """A shorter PID written over a longer predecessor must not leave
        trailing digits behind.

        Regression: the prior inline lock-acquisition path wrote without
        truncating, so a predecessor pid of ``157472`` overwritten by a
        three-digit pid produced a garbled ``999472`` that would later
        mislead diagnostics.
        """
        import fcntl

        from nooa.storage.sqlite import _acquire_session_lock

        lock_path = tmp_path / "truncation.lock"
        # Seed with a longer "old owner" PID.
        lock_path.write_bytes(b"99999999")

        fd = _acquire_session_lock(str(lock_path))
        try:
            assert lock_path.read_bytes() == str(os.getpid()).encode()
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def test_close_on_connect_failure(self, tmp_path):
        """Lines 365-367: close() is called when sqlite3.connect or schema setup fails."""
        from nooa.storage.sqlite import SQLiteStorageManager

        db_path = tmp_path / "test_fail.db"
        # Create a valid lock file so the lock succeeds,
        # then make connect fail by patching
        with patch(
            "nooa.storage.sqlite.sqlite3.connect",
            side_effect=RuntimeError("connect failed"),
        ):
            with pytest.raises(RuntimeError, match="connect failed"):
                SQLiteStorageManager(db_path=db_path)


class TestSQLiteModuleLevelAssertions:
    """Lines 48, 72: module-level assertions during import verify Event union structure."""

    def test_event_union_unwrap_assertion(self):
        """Line 48: _CONTEXT_BLOCKS_TYPES has >= 3 types from Event union."""
        import importlib
        import typing

        import nooa.storage.sqlite as sqlite_mod

        original_get_args = typing.get_args

        def fake_get_args(tp):
            # Return empty tuple for the inner call (Event union unwrap)
            result = original_get_args(tp)
            if len(result) >= 3:
                return result[:2]  # Force < 3 to trigger assertion
            return result

        try:
            with patch.object(typing, "get_args", side_effect=fake_get_args):
                with pytest.raises(
                    AssertionError, match="Failed to unwrap context_blocks Event union"
                ):
                    importlib.reload(sqlite_mod)
        finally:
            # Always restore module even if assertion doesn't match
            importlib.reload(sqlite_mod)

    def test_duplicate_event_type_assertion(self):
        """Line 72: duplicate event_type key in _CORE_TYPES raises AssertionError."""
        import importlib

        from nooa.events import Task

        # Patch Task's event_type default to collide with Message's "Message" key.
        # When sqlite.py module reloads, it iterates all event classes
        # and will find the duplicate.
        original_default = Task.model_fields["event_type"].default
        Task.model_fields["event_type"].default = "Message"  # collides with Message
        try:
            with pytest.raises(AssertionError, match="Duplicate event_type key"):
                import nooa.storage.sqlite as sqlite_mod

                importlib.reload(sqlite_mod)
        finally:
            Task.model_fields["event_type"].default = original_default
            # Reload to restore clean state
            import nooa.storage.sqlite as sqlite_mod

            importlib.reload(sqlite_mod)


# =============================================================================
# nemo_flow_middleware.py — model name extraction and _wrapper fallback
# =============================================================================


class TestNemoFlowMiddlewareModelExtraction:
    """Lines 99-101: model_name extracted from ctx.agent._llm.model."""

    @pytest.mark.asyncio
    async def test_model_name_from_agent_llm(self):
        """nemo_flow_llm_middleware extracts model_name from ctx.agent._llm.model."""
        fake_nemo_flow, _ = _make_fake_nemo_flow_for_test()

        # Ensure nooa.nemo_flow_middleware is imported
        import nooa.nemo_flow_middleware as _nm_ensure  # noqa: F401

        with patch.dict(
            sys.modules,
            {"nemo_relay": fake_nemo_flow},
        ):
            nm = sys.modules["nooa.nemo_flow_middleware"]
            importlib.reload(nm)
            try:
                mock_agent = MagicMock()
                mock_agent._llm = MagicMock()
                mock_agent._llm.model = "gpt-4-test"

                ctx = MagicMock()
                ctx.agent = mock_agent
                ctx.params = {"temperature": 0.5}
                ctx.messages = [{"role": "user", "content": "hi"}]
                ctx.response = None

                async def mock_nxt(c):
                    c.response = None
                    return c

                # The middleware should extract model_name = "gpt-4-test"
                # We verify it doesn't crash and nemo_flow.llm.execute is called
                # with the correct model_name
                await nm.nemo_flow_llm_middleware(ctx, mock_nxt)
                call_args = fake_nemo_flow.llm.execute.call_args
                assert call_args[0][0] == "gpt-4-test"  # first positional arg is model_name
            finally:
                importlib.reload(nm)


class TestNemoFlowWrapperFallbackReturn:
    """Line 169: _wrapper returns {} when resp has no recognized attributes."""

    @pytest.mark.asyncio
    async def test_wrapper_fallback_empty_dict(self):
        """When resp has no model_dump, raw_response, or assistant_message, return {}."""
        fake_nemo_flow, _ = _make_fake_nemo_flow_for_test()

        import nooa.nemo_flow_middleware as _nm_ensure  # noqa: F401

        captured_wrapper_result = {}

        async def llm_execute_capturing(*args, **kwargs):
            """Call the wrapper and capture its return value."""
            wrapper = args[2]
            request = args[1]
            result = await wrapper(request)
            captured_wrapper_result["result"] = result

        fake_nemo_flow.llm.execute = AsyncMock(side_effect=llm_execute_capturing)

        with patch.dict(sys.modules, {"nemo_relay": fake_nemo_flow}):
            nm = sys.modules["nooa.nemo_flow_middleware"]
            importlib.reload(nm)
            try:
                ctx = MagicMock()
                ctx.agent = None
                ctx.params = {}
                ctx.messages = []

                # The response object has none of the expected attributes
                plain_resp = types.SimpleNamespace()
                # no model_dump, no raw_response, no assistant_message

                async def mock_nxt(c):
                    c.response = plain_resp
                    return c

                await nm.nemo_flow_llm_middleware(ctx, mock_nxt)
                assert captured_wrapper_result["result"] == {}
            finally:
                importlib.reload(nm)


def _make_fake_nemo_flow_for_test():
    """Build a fake nemo_flow module for testing."""
    fake = MagicMock()
    fake_handle = MagicMock()
    fake_handle.uuid = "test-uuid"
    fake.scope.scope.return_value.__enter__ = MagicMock(return_value=fake_handle)
    fake.scope.scope.return_value.__exit__ = MagicMock(return_value=False)

    async def default_llm_execute(*args, **kwargs):
        # args: (model_name, request, wrapper)
        wrapper = args[2]
        request = args[1]
        await wrapper(request)

    fake.llm.execute = AsyncMock(side_effect=default_llm_execute)
    fake.scope.push.return_value = MagicMock()
    fake.scope.pop.return_value = None

    return fake, fake_handle


# =============================================================================
# tools/web_publisher.py — RichOutput and WebPublisher methods
# =============================================================================


class TestWebPublisherMethods:
    """Cover all WebPublisher methods and _post paths."""

    def test_rich_output_event_type(self):
        """RichOutput has event_type 'rich_output'."""
        from nooa.tools.web_publisher import RichOutput

        ro = RichOutput(payload={"kind": "html", "html": "<b>hi</b>"})
        assert ro.event_type == "RichOutput"
        assert ro.payload["kind"] == "html"

    def test_post_stores_event_when_event_manager_present(self):
        """_post adds a RichOutput event to the event_manager."""
        from nooa.tools.web_publisher import WebPublisher

        mock_em = MagicMock()
        wp = WebPublisher(event_manager=mock_em)
        # No NEMO_OO_RICH_URL set, so POST is skipped but event is stored
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEMO_OO_RICH_URL", None)
            wp._post({"kind": "html", "html": "<p>test</p>"})

        mock_em.add.assert_called_once()

    def test_post_skips_event_for_clear_kind(self):
        """_post does NOT store events for 'clear' kind."""
        from nooa.tools.web_publisher import WebPublisher

        mock_em = MagicMock()
        wp = WebPublisher(event_manager=mock_em)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEMO_OO_RICH_URL", None)
            wp._post({"kind": "clear"})

        mock_em.add.assert_not_called()

    def test_post_without_event_manager(self):
        """_post works when event_manager is None (no storage)."""
        from nooa.tools.web_publisher import WebPublisher

        wp = WebPublisher(event_manager=None)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEMO_OO_RICH_URL", None)
            # Should not raise
            wp._post({"kind": "markdown", "text": "hi"})

    def test_post_with_nemo_rich_url_httpx(self):
        """_post sends HTTP POST when NEMO_OO_RICH_URL is set and httpx is available."""
        from nooa.tools.web_publisher import WebPublisher

        wp = WebPublisher(event_manager=None)
        mock_httpx = MagicMock()
        with patch.dict(os.environ, {"NEMO_OO_RICH_URL": "http://localhost:1234/rich"}):
            with patch.dict(sys.modules, {"httpx": mock_httpx}):
                wp._post({"kind": "json", "data": {"x": 1}})
                mock_httpx.post.assert_called_once()

    def test_post_without_httpx_logs_warning(self, caplog):
        """_post logs a warning when httpx is not installed."""
        from nooa.tools.web_publisher import WebPublisher

        wp = WebPublisher(event_manager=None)
        with patch.dict(os.environ, {"NEMO_OO_RICH_URL": "http://localhost:1234/rich"}):
            # Make httpx import fail
            with patch("builtins.__import__", side_effect=_selective_import_error("httpx")):
                import logging

                with caplog.at_level(logging.WARNING, logger="nooa.tools.web_publisher"):
                    wp._post({"kind": "json", "data": {}})
                assert any("httpx" in r.message for r in caplog.records)

    def test_post_httpx_exception_logged(self):
        """_post catches and logs httpx POST exceptions."""
        from nooa.tools.web_publisher import WebPublisher

        wp = WebPublisher(event_manager=None)
        mock_httpx = MagicMock()
        mock_httpx.post.side_effect = ConnectionError("refused")
        with patch.dict(os.environ, {"NEMO_OO_RICH_URL": "http://localhost:1234/rich"}):
            with patch.dict(sys.modules, {"httpx": mock_httpx}):
                # Should not raise
                wp._post({"kind": "html", "html": "<b>test</b>"})

    def test_post_event_manager_add_exception_handled(self):
        """_post catches exceptions from event_manager.add."""
        from nooa.tools.web_publisher import WebPublisher

        mock_em = MagicMock()
        mock_em.add.side_effect = RuntimeError("storage full")
        wp = WebPublisher(event_manager=mock_em)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NEMO_OO_RICH_URL", None)
            # Should not raise
            wp._post({"kind": "html", "html": "<b>test</b>"})

    def test_plot_method(self):
        """plot() converts figure to JSON and posts."""
        from nooa.tools.web_publisher import WebPublisher

        wp = WebPublisher(event_manager=None)
        mock_fig = MagicMock()
        mock_fig.to_json.return_value = '{"data": [], "layout": {}}'

        with patch.object(wp, "_post") as mock_post:
            wp.plot(mock_fig, title="My Chart")
            mock_post.assert_called_once_with(
                {
                    "kind": "plotly",
                    "figure_json": '{"data": [], "layout": {}}',
                    "title": "My Chart",
                }
            )

    def test_plot_serialization_error(self):
        """plot() catches fig.to_json() exceptions."""
        from nooa.tools.web_publisher import WebPublisher

        wp = WebPublisher(event_manager=None)
        mock_fig = MagicMock()
        mock_fig.to_json.side_effect = TypeError("not serializable")

        with patch.object(wp, "_post") as mock_post:
            wp.plot(mock_fig)
            mock_post.assert_not_called()

    def test_html_method(self):
        """html() posts an HTML fragment."""
        from nooa.tools.web_publisher import WebPublisher

        wp = WebPublisher(event_manager=None)
        with patch.object(wp, "_post") as mock_post:
            wp.html("<p>hi</p>", title="Fragment")
            mock_post.assert_called_once_with(
                {
                    "kind": "html",
                    "html": "<p>hi</p>",
                    "title": "Fragment",
                }
            )

    def test_image_method(self):
        """image() posts an image src."""
        from nooa.tools.web_publisher import WebPublisher

        wp = WebPublisher(event_manager=None)
        with patch.object(wp, "_post") as mock_post:
            wp.image("data:image/png;base64,abc", alt="img", title="Photo")
            mock_post.assert_called_once_with(
                {
                    "kind": "image",
                    "src": "data:image/png;base64,abc",
                    "alt": "img",
                    "title": "Photo",
                }
            )

    def test_markdown_method(self):
        """markdown() posts markdown text."""
        from nooa.tools.web_publisher import WebPublisher

        wp = WebPublisher(event_manager=None)
        with patch.object(wp, "_post") as mock_post:
            wp.markdown("## Hello", title="MD")
            mock_post.assert_called_once_with(
                {
                    "kind": "markdown",
                    "text": "## Hello",
                    "title": "MD",
                }
            )

    def test_json_method(self):
        """json() posts JSON data."""
        from nooa.tools.web_publisher import WebPublisher

        wp = WebPublisher(event_manager=None)
        with patch.object(wp, "_post") as mock_post:
            wp.json({"key": "value"}, title="Data")
            mock_post.assert_called_once_with(
                {
                    "kind": "json",
                    "data": {"key": "value"},
                    "title": "Data",
                }
            )

    def test_clear_method(self):
        """clear() posts a clear payload."""
        from nooa.tools.web_publisher import WebPublisher

        wp = WebPublisher(event_manager=None)
        with patch.object(wp, "_post") as mock_post:
            wp.clear()
            mock_post.assert_called_once_with({"kind": "clear"})

    def test_constructor_without_event_manager(self):
        """WebPublisher can be constructed with no event_manager."""
        from nooa.tools.web_publisher import WebPublisher

        wp = WebPublisher()
        assert wp._event_manager is None

    def test_constructor_with_event_manager(self):
        """WebPublisher stores the provided event_manager."""
        from nooa.tools.web_publisher import WebPublisher

        mock_em = MagicMock()
        wp = WebPublisher(event_manager=mock_em)
        assert wp._event_manager is mock_em


def _selective_import_error(blocked_module: str):
    """Return an __import__ replacement that fails for the blocked module."""
    _real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    def _fake_import(name, *args, **kwargs):
        if name == blocked_module:
            raise ImportError(f"No module named '{blocked_module}'")
        return _real_import(name, *args, **kwargs)

    return _fake_import
