# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP adapter for the host-neutral NOOA coding agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

from acp import (
    PROTOCOL_VERSION,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
    RequestError,
    run_agent,
    text_block,
    update_agent_message,
    update_user_message,
)
from acp.helpers import update_available_commands
from acp.interfaces import Agent, Client
from acp.schema import (
    AgentCapabilities,
    AvailableCommand,
    AvailableCommandInput,
    CloseSessionResponse,
    HttpMcpServer,
    Implementation,
    ListSessionsResponse,
    McpCapabilities,
    McpServerStdio,
    SessionCapabilities,
    SessionCloseCapabilities,
    SessionListCapabilities,
    SseMcpServer,
    UnstructuredCommandInput,
)
from acp.schema import (
    SessionInfo as ACPSessionInfo,
)
from nooa_cli.coding import (
    CodingAgent,
    CodingSlashCommand,
    CodingSlashCommandRegistry,
    load_coding_skills_dirs,
)
from nooa_cli.coding.agent import _coding_constructed
from nooa_cli.sessions import (
    InvalidSessionIdError,
    SessionHandle,
    SessionNotFoundError,
    SessionStore,
)

from nooa.errors import GenerationError
from nooa.mcp import MCPManager, MCPTool
from nooa.runtime.model_binding import ModelBinding
from nooa.slash_dispatch import CoercionError
from nooa.storage.json_snapshot import snapshot_to_json
from nooa.strategies.codeact import MAX_ITERATIONS_MESSAGE, OUTPUT_TOKENS_EXHAUSTED_MESSAGE
from nooa.unifiedllm import UnifiedLLM
from nooa_acp._runtime import (
    SessionBusyError,
    SessionRuntime,
    SessionRuntimeClosedError,
    SessionRuntimePool,
    _acp_capture,
    _acp_join,
    _acp_task,
    _ACPOutcome,
)
from nooa_acp.dispatcher import InteractiveSessionDispatcher
from nooa_acp.event_bridge import ACPEventBridge

logger = logging.getLogger(__name__)

_SESSION_PAGE_SIZE = 50
_GENERATION_LIMIT_PREFIX = MAX_ITERATIONS_MESSAGE.partition("{")[0]


def _snapshot_blob(agent: CodingAgent) -> tuple[str, str]:
    """The agent's snapshot as JSON text, and its SHA-256 digest."""
    blob = json.dumps(snapshot_to_json(agent), sort_keys=True)
    return blob, hashlib.sha256(blob.encode()).hexdigest()


@dataclass(eq=False, slots=True)
class _ACPModelOwner:
    # Register the raw factory product before even binding validation can fail.
    raw: Any
    binding: ModelBinding | None = None
    completed: bool = False

    async def close(self) -> None:
        if not self.completed:
            if self.binding is None:
                await self.raw.aclose()
            else:
                await self.binding.aclose()
            self.completed = True


@dataclass(eq=False, slots=True)
class _ACPConstruction:
    """Strong ownership from factory return through runtime retirement."""

    handle: SessionHandle | None = None
    models: list[_ACPModelOwner] = field(default_factory=list)
    agents: list[CodingAgent] = field(default_factory=list)
    tools: dict[str, MCPTool] = field(default_factory=dict)
    dispatcher: InteractiveSessionDispatcher | None = None
    bridge: ACPEventBridge | None = None
    commands: CodingSlashCommandRegistry | None = None
    completed: set[tuple[str, int]] = field(default_factory=set)
    close_task: asyncio.Task[_ACPOutcome] | None = None
    failures: list[_ACPOutcome] = field(default_factory=list)
    release_started: bool = False
    release_completed: bool = False
    primary_failure: _ACPOutcome | None = None
    release_failure: _ACPOutcome | None = None

    def remember(self, outcome: _ACPOutcome) -> None:
        if outcome.failure is not None and all(
            retained.failure is not outcome.failure for retained in self.failures
        ):
            self.failures.append(outcome)

    @property
    def retryable(self) -> bool:
        resources = [
            ("bridge", self.bridge),
            ("commands", self.commands),
            ("dispatch", self.dispatcher),
        ]
        resources += [("agent", value) for value in self.agents]
        resources += [("model", value) for value in self.models]
        resources += [("mcp", value) for value in self.tools.values()]
        return any(
            value is not None and (label, id(value)) not in self.completed
            for label, value in resources
        ) or (self.handle is not None and not self.release_started)

    async def close(self) -> None:
        if self.close_task is None or (self.close_task.done() and self.retryable):
            try:
                self.close_task = _acp_task(
                    self._close_resources(), name="nooa-acp-construction-close"
                )
            except BaseException as error:
                self.remember(_ACPOutcome(failure=error, traceback=error.__traceback__))
                raise
        (await _acp_join(self.close_task)).unwrap()

    async def failed(self, error: BaseException) -> _ACPOutcome:
        """One cleanup attempt per failed operation, not per nested catch."""
        if self.primary_failure is None:
            self.primary_failure = _ACPOutcome(failure=error, traceback=error.__traceback__)
        if self.close_task is None:
            outcome = await _acp_capture(self.close())
        else:
            outcome = await _acp_join(self.close_task)
        self.remember(outcome)
        return outcome

    async def retire(self, label: str, resource: Any, method: str) -> None:
        if resource is None or (label, id(resource)) in self.completed:
            return

        async def invoke() -> None:
            result = getattr(resource, method)()
            if hasattr(result, "__await__"):
                await result

        outcome = await _acp_capture(invoke())
        if outcome.failure is None:
            self.completed.add((label, id(resource)))
        self.remember(outcome)

    async def _close_resources(self) -> None:
        await self.retire("bridge", self.bridge, "close")
        await self.retire("commands", self.commands, "close")
        await self.retire("dispatch", self.dispatcher, "cancel")
        for agent in self.agents:
            await self.retire("agent", agent, "close")
        for tool in self.tools.values():
            try:
                method = "aclose" if callable(getattr(tool, "aclose", None)) else "close"
                if callable(getattr(tool, method, None)):
                    await self.retire("mcp", tool, method)
                else:
                    self.completed.add(("mcp", id(tool)))
            except BaseException as error:
                self.remember(_ACPOutcome(failure=error, traceback=error.__traceback__))
        for owner in self.models:
            await self.retire("model", owner, "close")
        if self.handle is not None and not self.release_started:
            await self._release_handle()
        if self.release_failure is not None:
            self.remember(self.release_failure)
        if self.failures:
            self.failures[0].unwrap()

    async def _release_handle(self) -> None:
        # A setup/getter failure is retryable. Only the actual delegate call
        # crosses the protected storage protocol's terminal boundary.
        try:
            assert self.handle is not None
            close = self.handle.close
        except BaseException as error:
            self.remember(_ACPOutcome(failure=error, traceback=error.__traceback__))
            return
        self.release_started = True
        try:
            close()
        except BaseException as error:
            self.release_failure = _ACPOutcome(failure=error, traceback=error.__traceback__)
        else:
            self.release_completed = True


@dataclass(slots=True)
class _ACPSession:
    """Live resources owned by one ACP session runtime."""

    handle: SessionHandle
    agent: CodingAgent
    dispatcher: InteractiveSessionDispatcher
    bridge: ACPEventBridge
    commands: CodingSlashCommandRegistry
    construction: _ACPConstruction
    startup_warnings: tuple[str, ...] = ()
    cancel_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cancel_complete: asyncio.Event = field(default_factory=asyncio.Event)
    notification_tasks: set[asyncio.Task[_ACPOutcome]] = field(default_factory=set)
    notification_failure: _ACPOutcome | None = None
    notification_owner: asyncio.Task[_ACPOutcome] | None = None
    close_task: asyncio.Task[_ACPOutcome] | None = None
    commands_sent_on_prompt: bool = False
    snapshot_digest: str | None = None

    def __post_init__(self) -> None:
        self.cancel_complete.set()

    def checkpoint(self) -> None:
        """Save the agent's state if it changed since the last save.

        A failure is logged and the turn is not affected; the next turn tries
        again.
        """
        try:
            blob, digest = _snapshot_blob(self.agent)
            if digest == self.snapshot_digest:
                return
            self.handle.storage.save_snapshot_json(blob)
            self.snapshot_digest = digest
        except Exception:
            logger.warning(
                "Session %s: could not save the agent's state", self.handle.id, exc_info=True
            )

    @asynccontextmanager
    async def saving_state(self) -> AsyncIterator[None]:
        """Save the agent's state when the turn ends, unless it was cancelled.

        A cancel can stop a cell halfway, so its state is not saved; the
        session keeps the last turn that finished. ``cancel()`` sets
        ``cancel_complete``, which each turn clears when it starts.
        """
        cancelled = False
        try:
            yield
        except asyncio.CancelledError:
            cancelled = True
            raise
        finally:
            if not cancelled and not self.cancel_complete.is_set():
                self.checkpoint()

    async def close(self) -> None:
        if self.close_task is None or (self.close_task.done() and self.construction.retryable):
            self.close_task = _acp_task(self._close_resources(), name="nooa-acp-session-close")
        (await _acp_join(self.close_task)).unwrap()

    @property
    def _close_retryable(self) -> bool:
        return self.construction.retryable

    @property
    def _close_retained(self) -> bool:
        # The actual typed stage ledger, not repeated no-op close, establishes
        # settlement. Delegated release failure is terminal and still fenced.
        return self.construction.retryable or self.construction.release_failure is not None

    async def _close_resources(self) -> None:
        first = self.notification_failure
        tasks = set(self.notification_tasks)
        if self.notification_owner is not None:
            tasks.add(self.notification_owner)
        for task in tuple(tasks):
            try:
                task.cancel()
            except BaseException as exc:
                if first is None:
                    first = _ACPOutcome(failure=exc, traceback=exc.__traceback__)
        for task in tuple(tasks):
            outcome = await _acp_join(task)
            if outcome.failure is None and isinstance(outcome.value, _ACPOutcome):
                outcome = outcome.value
            if outcome.failure is not None and not isinstance(
                outcome.failure, asyncio.CancelledError
            ):
                if first is None:
                    first = outcome
        self.notification_tasks.clear()
        self.notification_owner = None
        outcome = await _acp_capture(self.construction.close())
        if first is None:
            first = outcome
        if first is not None:
            first.unwrap()


class CodingACPAdapter:
    def __init__(self, llm_factory: Callable[[], UnifiedLLM], *, model_alias: str) -> None:
        if not isinstance(model_alias, str) or not model_alias:
            raise ValueError("ACP model_alias must be a nonempty configured alias")
        self._model_alias = model_alias
        self._construction: set[_ACPConstruction] = set()
        self._llm_factory = llm_factory
        self._client: Client | None = None
        self._sessions: SessionRuntimePool[_ACPSession] = SessionRuntimePool()

    def on_connect(self, conn: Client) -> None:
        self._client = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        del protocol_version, client_capabilities, client_info, kwargs
        try:
            package_version = version("nooa-acp")
        except PackageNotFoundError:
            package_version = "0.0.0"
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(
                load_session=True,
                # McpCapabilities defaults to all-false, and a client that
                # honours the handshake then filters its HTTP/SSE servers out of
                # session/new — so the agent receives no MCP servers at all,
                # however the user configured them. _create_mcp_tools connects
                # both transports, so say so. `acp` stays off: it is unstable in
                # the spec and not implemented here.
                mcp_capabilities=McpCapabilities(http=True, sse=True),
                session_capabilities=SessionCapabilities(
                    list=SessionListCapabilities(),
                    close=SessionCloseCapabilities(),
                ),
            ),
            auth_methods=[],
            agent_info=Implementation(
                name="nooa-acp",
                title="NVIDIA Labs Object Oriented Agents (NOOA)",
                version=package_version,
            ),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[Any] | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        del kwargs
        root = self._validate_workspace(cwd, additional_directories)
        construction = _ACPConstruction()
        self._construction.add(construction)
        handle: SessionHandle | None = None
        runtime: SessionRuntime[_ACPSession] | None = None
        try:
            llm = self._new_model(construction)
            handle = self._store(root).create(
                model=self._model_alias,
                agent="CodingAgent",
                working_directory=str(root),
                origin="acp",
                check_same_thread=False,
            )
            construction.handle = handle
            runtime = await self._create_runtime(
                handle, root, mcp_servers, llm=llm, _construction=construction
            )
            self._defer_bootstrap_updates(runtime.value)
        except BaseException as error:
            if construction.primary_failure is None:
                construction.primary_failure = _ACPOutcome(
                    failure=error, traceback=error.__traceback__
                )
            if runtime is not None:
                cleanup = await _acp_capture(
                    self._sessions.remove(runtime.session_id, include_unavailable=True)
                )
                construction.remember(cleanup)
            else:
                cleanup = await construction.failed(error)
            if cleanup.failure is None:
                self._construction.discard(construction)
                if handle is not None:
                    deletion = await _acp_capture(self._delete_failed_session(root, handle.id))
                    if deletion.failure is not None:
                        construction.remember(deletion)
                        self._construction.add(construction)
            raise
        return NewSessionResponse(session_id=handle.id)

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        mcp_servers: list[Any] | None = None,
        additional_directories: list[str] | None = None,
        **kwargs: Any,
    ) -> LoadSessionResponse:
        del kwargs
        root = self._validate_workspace(cwd, additional_directories)
        # Check the actual owner ledgers before allocating a second writer.
        # Opening and registration below contain no intervening await.
        path = self._store(root).path_for(session_id).resolve()
        if await self._sessions.owns(session_id) or any(
            owner.handle is not None and owner.handle.path.resolve() == path
            for owner in self._construction
        ):
            raise RequestError.invalid_request(
                {"sessionId": session_id, "reason": "Session is already owned"}
            )
        try:
            handle = self._store(root).open(session_id, check_same_thread=False)
        except (InvalidSessionIdError, SessionNotFoundError):
            raise RequestError.resource_not_found(session_id) from None
        construction = _ACPConstruction(handle=handle)
        self._construction.add(construction)
        runtime: SessionRuntime[_ACPSession] | None = None
        try:
            if handle.info.model and handle.info.model != self._model_alias:
                raise RequestError.invalid_request(
                    {
                        "sessionId": session_id,
                        "reason": "Saved model alias differs from configured ACP model_alias",
                    }
                )
            runtime = await self._create_runtime(
                handle, root, mcp_servers, available=False, _construction=construction
            )
            # After the replay, never during it: replay writes straight to the
            # client while the bridge pump drains bootstrap updates, so every
            # await here would otherwise let a commands update or an MCP warning
            # land in the middle of the restored conversation.
            await self._replay_session(handle)
            await self._sessions.publish(session_id)
            self._defer_bootstrap_updates(runtime.value)
        except BaseException as error:
            if construction.primary_failure is None:
                construction.primary_failure = _ACPOutcome(
                    failure=error, traceback=error.__traceback__
                )
            if runtime is not None:
                cleanup = await _acp_capture(
                    self._sessions.remove(session_id, include_unavailable=True)
                )
                construction.remember(cleanup)
            else:
                cleanup = await construction.failed(error)
                if cleanup.failure is None:
                    self._construction.discard(construction)
            raise
        return LoadSessionResponse()

    async def list_sessions(
        self,
        cwd: str | None = None,
        cursor: str | None = None,
        **kwargs: Any,
    ) -> ListSessionsResponse:
        del kwargs
        root = self._validate_workspace(cwd or str(Path.cwd()), None)
        try:
            offset = int(cursor) if cursor is not None else 0
        except ValueError:
            raise RequestError.invalid_params(
                {"cursor": cursor, "reason": "Invalid cursor"}
            ) from None
        if offset < 0:
            raise RequestError.invalid_params({"cursor": cursor, "reason": "Invalid cursor"})

        found = self._store(root).list(limit=offset + _SESSION_PAGE_SIZE + 1)
        page = found[offset : offset + _SESSION_PAGE_SIZE]
        sessions = [
            ACPSessionInfo(
                session_id=info.id,
                cwd=info.working_directory or str(root),
                title=info.title,
                updated_at=datetime.fromtimestamp(info.last_active, UTC).isoformat(),
            )
            for info in page
        ]
        next_cursor = str(offset + len(page)) if len(found) > offset + len(page) else None
        return ListSessionsResponse(sessions=sessions, next_cursor=next_cursor)

    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse:
        del kwargs
        try:
            await self._sessions.remove(session_id)
        except KeyError:
            raise RequestError.resource_not_found(session_id) from None
        return CloseSessionResponse()

    async def prompt(self, session_id: str, prompt: list[Any], **kwargs: Any) -> PromptResponse:
        del kwargs
        runtime = await self._get_runtime(session_id)
        text = self._prompt_text(prompt)
        try:
            async with runtime.turn(), runtime.value.saving_state():
                session = runtime.value
                session.handle.record_user_message(text)
                session.cancel_complete.clear()
                try:
                    if not session.commands_sent_on_prompt:
                        session.bridge.publish(
                            _available_commands_update(session.commands.commands())
                        )
                        session.commands_sent_on_prompt = True
                    slash = self._slash_invocation(session.commands, text)
                    if slash is None:
                        result = await session.dispatcher.submit(text)
                    else:
                        name, raw_args = slash
                        try:
                            submission = await session.dispatcher.invoke_slash(
                                session.commands,
                                name,
                                raw_args,
                            )
                        except CoercionError as exc:
                            message = f"/{name}: {exc.message}"
                            if exc.hint:
                                message += f"\n\nUsage: `/{name} {exc.hint}`"
                            session.agent.message(message)
                            await session.bridge.flush()
                            return PromptResponse(stop_reason="end_turn")
                        except GenerationError:
                            # Subclasses Exception, so the catch-all below would
                            # swallow it and lose the stop reason the outer
                            # handler maps. Generation limits are the runtime's
                            # to report, not a command failure.
                            raise
                        except Exception as exc:
                            # Command bodies are third-party code from workspace
                            # and installed skills. Letting one raise turns the
                            # whole prompt into a JSON-RPC internal_error, and
                            # the user's turn is already durably recorded — so
                            # the session replays a question with no answer.
                            # The same failure inside execute_python is caught by
                            # the strategy and shown to the model; this path had
                            # no equivalent.
                            logger.warning(
                                "Slash command /%s failed in session %s",
                                name,
                                session_id,
                                exc_info=True,
                            )
                            session.agent.message(f"/{name} failed: {exc}")
                            await session.bridge.flush()
                            return PromptResponse(stop_reason="end_turn")
                        if submission is None:
                            result = None
                        else:
                            slash_result, result = submission
                            if not slash_result.output_to_agent:
                                message = str(slash_result)
                                if message:
                                    session.agent.message(message)
                                await session.bridge.flush()
                                return PromptResponse(stop_reason="end_turn")
                except GenerationError as exc:
                    # The strategy does not guarantee a PythonOutput for a call
                    # it already announced, so a turn ending on a generation
                    # limit can leave its card in_progress. Nothing else closes
                    # it before session close, and a later cancel would retitle
                    # this turn's stale card "Cancelled".
                    await session.bridge.fail_open_tools("Did not finish.", title="Unfinished")
                    await session.bridge.flush()
                    message = str(exc)
                    if message == OUTPUT_TOKENS_EXHAUSTED_MESSAGE:
                        return PromptResponse(stop_reason="max_tokens")
                    # MAX_ITERATIONS_MESSAGE and the max_retries message share
                    # this prefix; the limit name tells them from other errors.
                    if message.startswith(_GENERATION_LIMIT_PREFIX) and (
                        "max_iterations=" in message or "max_retries=" in message
                    ):
                        return PromptResponse(stop_reason="max_turn_requests")
                    raise RequestError(-32603, message, {"details": message}) from exc
                if result is None:
                    await session.cancel_complete.wait()
                    # stop_reason and the tool card both carry the outcome, but
                    # a collapsed card shows nothing and the turn just goes
                    # quiet. Record it as a real message so the conversation —
                    # and the durable transcript on resume — says what happened.
                    session.agent.message("Stopped at your request.")
                    await session.bridge.flush()
                    return PromptResponse(stop_reason="cancelled")
                await session.bridge.flush()
                return PromptResponse(stop_reason="end_turn")
        except SessionBusyError:
            raise RequestError.invalid_request(
                {"sessionId": session_id, "reason": "A prompt is already running"}
            ) from None
        except SessionRuntimeClosedError:
            # The session was closed between _get_runtime and the turn claim.
            # It is gone as far as the client is concerned, so say so rather
            # than letting this escape as an opaque internal error.
            raise RequestError.resource_not_found(session_id) from None

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        del kwargs
        runtime = await self._get_runtime(session_id)
        session = runtime.value
        async with session.cancel_lock:
            try:
                if await session.dispatcher.cancel():
                    await session.bridge.fail_open_tools("Cancelled by user.", title="Cancelled")
                    await session.bridge.flush()
            finally:
                session.cancel_complete.set()

    async def _create_runtime(
        self,
        handle: SessionHandle,
        root: Path,
        mcp_servers: list[Any] | None,
        *,
        llm: UnifiedLLM | None = None,
        available: bool = True,
        _construction: _ACPConstruction | None = None,
    ) -> SessionRuntime[_ACPSession]:
        construction = _construction or _ACPConstruction(handle=handle)
        self._construction.add(construction)
        try:
            if self._client is None:
                raise RequestError.internal_error({"reason": "ACP client is not connected"})
            if llm is None:
                llm = self._new_model(construction)
            mcp, mcp_warnings = await self._create_mcp_tools(mcp_servers, _tools=construction.tools)
            agent, restored = await self._restore_agent(
                self._construct_agent(llm, root, handle, construction), root, handle, construction
            )
            registration_warnings: list[str] = []
            for name, tool in mcp.items():
                registry_name = f"mcp.{name}"
                try:
                    agent.skills.register(registry_name, tool)
                    agent.skills.activate([registry_name])
                except ValueError as exc:
                    # A server name can collide with a core agent attribute
                    # (`shell`, `repo`) or a reserved one (`runtime`). Skipping
                    # it keeps the session usable — the same contract the
                    # connect path already offers for an unreachable server —
                    # instead of failing session/new with an opaque error.
                    registration_warnings.append(f"MCP server {name!r} was not registered: {exc}")
            dispatcher = InteractiveSessionDispatcher.__new__(InteractiveSessionDispatcher)
            construction.dispatcher = dispatcher
            dispatcher.__init__(agent)
            bridge = ACPEventBridge.__new__(ACPEventBridge)
            construction.bridge = bridge
            bridge.__init__(agent, self._client, handle.id)
            commands = CodingSlashCommandRegistry.__new__(CodingSlashCommandRegistry)
            construction.commands = commands
            commands.__init__(agent)
            value = _ACPSession(
                handle,
                agent,
                dispatcher,
                bridge,
                commands,
                construction,
                startup_warnings=(*mcp_warnings, *registration_warnings),
            )
            if restored:
                # The state just restored is already saved: without a digest
                # the first unchanged turn would write a copy of it. A fresh
                # agent keeps none, so its first turn replaces a bad snapshot.
                with suppress(Exception):
                    value.snapshot_digest = _snapshot_blob(agent)[1]
            commands.set_on_change(
                lambda available: bridge.publish(_available_commands_update(available)),
            )
            try:
                runtime = await self._sessions.add(handle.id, value, available=available)
                self._construction.discard(construction)
                return runtime
            except BaseException as error:
                # A post-registration failure still belongs to this construction.
                # Retire its exact value atomically; never remove another writer.
                cleanup = await _acp_capture(self._sessions._remove_if_owned(handle.id, value))
                construction.remember(cleanup)
                if isinstance(error, ValueError):
                    raise RequestError.invalid_request(
                        {"sessionId": handle.id, "reason": "Session is already loaded"}
                    ) from None
                raise
        except BaseException as error:
            cleanup = await construction.failed(error)
            if cleanup.failure is None:
                self._construction.discard(construction)
            raise

    def _new_model(self, construction: _ACPConstruction) -> ModelBinding:
        owner = _ACPModelOwner(self._llm_factory())
        construction.models.append(owner)
        owner.binding = ModelBinding(owner.raw, owned=True)
        return owner.binding

    def _construct_agent(
        self, llm: Any, root: Path, handle: SessionHandle, construction: _ACPConstruction
    ) -> CodingAgent:
        token = _coding_constructed.set(construction.agents.append)
        try:
            agent = self._build_agent(llm, root, handle)
            if not any(item is agent for item in construction.agents):
                construction.agents.append(agent)
            return agent
        finally:
            _coding_constructed.reset(token)

    async def _delete_failed_session(self, root: Path, session_id: str) -> None:
        self._store(root).delete(session_id)

    @staticmethod
    def _build_agent(llm: UnifiedLLM, root: Path, handle: SessionHandle) -> CodingAgent:
        return CodingAgent(
            llm=llm,
            cwd=root,
            storage=handle.storage,
            # Same anchoring as the session store above: one ACP process
            # serves many workspaces, so project-local paths follow the
            # session's workspace rather than the process.
            libs_dir=root / ".nooa" / "skills",
            skills_dirs=load_coding_skills_dirs(root),
        )

    async def _restore_agent(
        self,
        agent: CodingAgent,
        root: Path,
        handle: SessionHandle,
        construction: _ACPConstruction,
    ) -> tuple[CodingAgent, bool]:
        """Restore the latest snapshot; return the agent to use and whether one was restored.

        If it cannot be restored (damaged, or from an older agent class), log
        it and return a fresh agent: the failed restore may have half-changed
        this one.
        """
        try:
            restored = handle.storage.restore_latest_snapshot(agent)
        except Exception:
            # Register the replacement before retiring any previous resource.
            # A diagnostic must never interrupt the independent cleanup stages.
            try:
                logger.warning(
                    "Session %s: could not restore its saved state; starting a fresh agent",
                    handle.id,
                    exc_info=True,
                )
            except BaseException as error:
                construction.remember(_ACPOutcome(failure=error, traceback=error.__traceback__))
            replacement = self._new_model(construction)
            rebuilt = self._construct_agent(replacement, root, handle, construction)
            await construction.retire("agent", agent, "close")
            for owner in construction.models[:-1]:
                await construction.retire("model", owner, "close")
            if construction.failures:
                construction.failures[0].unwrap()
            return rebuilt, False
        # The restore ran the agent's after_restore() (AgentSnapshot.restore),
        # which repairs what did not survive the restart.
        return agent, restored

    async def _create_mcp_tools(
        self,
        mcp_servers: list[Any] | None,
        *,
        _tools: dict[str, MCPTool] | None = None,
    ) -> tuple[dict[str, MCPTool], tuple[str, ...]]:
        servers = list(mcp_servers or [])
        supported_types = (McpServerStdio, HttpMcpServer, SseMcpServer)
        tools: dict[str, MCPTool] = {} if _tools is None else _tools
        warnings: list[str] = []
        seen_names: set[str] = set()
        for server in servers:
            name = getattr(server, "name", "<unnamed>")
            if not isinstance(server, supported_types):
                warnings.append(
                    f"MCP server {name!r} was not loaded: unsupported ACP server type "
                    f"{type(server).__name__}."
                )
                continue
            if name in seen_names:
                warnings.append(
                    f"MCP server {name!r} was not loaded: another server has the same name."
                )
                continue
            seen_names.add(name)
            try:
                if isinstance(server, McpServerStdio):
                    env = {item.name: item.value for item in server.env}
                    tools[name] = await MCPManager.create_stdio_server(
                        name,
                        command=server.command,
                        args=server.args,
                        env=env,
                    )
                elif isinstance(server, (HttpMcpServer, SseMcpServer)):
                    headers = {item.name: item.value for item in server.headers}
                    tools[name] = await MCPManager.create_url_server(
                        name,
                        server.url,
                        headers=headers,
                        transport=(
                            "streamable-http" if isinstance(server, HttpMcpServer) else "sse"
                        ),
                    )
            except Exception as exc:
                warnings.append(f"MCP server {name!r} was not loaded: {exc}")
        return tools, tuple(warnings)

    async def _get_runtime(self, session_id: str) -> SessionRuntime[_ACPSession]:
        try:
            return await self._sessions.get(session_id)
        except KeyError:
            raise RequestError.resource_not_found(session_id) from None

    @staticmethod
    def _defer_bootstrap_updates(session: _ACPSession) -> None:
        """Publish bootstrap updates after the session response can reach clients such as Zed."""

        async def _publish() -> None:
            await asyncio.sleep(0)
            session.bridge.publish_best_effort(
                _available_commands_update(session.commands.commands())
            )
            if session.startup_warnings:
                details = "\n".join(f"- {warning}" for warning in session.startup_warnings)
                session.bridge.publish_best_effort(
                    update_agent_message(
                        text_block(
                            "NOOA started without one or more MCP servers. The session is still "
                            f"usable.\n\n{details}"
                        )
                    )
                )
            await session.bridge.flush()

        task = _acp_task(_publish(), name="nooa-acp-bootstrap")
        # A registration/callback failure cannot orphan a live worker.
        session.notification_owner = task
        session.notification_tasks.add(task)

        def _finished(done: asyncio.Task[_ACPOutcome]) -> None:
            # Successful work has no history; a failure remains on its session.
            if done.cancelled():
                return
            outcome = done.result()
            if outcome.failure is not None and session.notification_failure is None:
                session.notification_failure = outcome
            session.notification_tasks.discard(done)
            if session.notification_owner is done:
                session.notification_owner = None

        task.add_done_callback(_finished)

    async def _replay_session(self, handle: SessionHandle) -> None:
        if self._client is None:
            return
        for turn in handle.turns():
            # Each update is a *chunk*, and ACP has no end-of-message marker —
            # a boundary is implied by a different update type arriving. Two
            # turns from the same speaker in a row therefore land in one bubble.
            # That happens whenever a turn produced no reply, as a cancelled one
            # used to, so several stopped prompts replayed as a single run-on
            # line. Terminate each turn so it keeps its own boundary.
            content = turn.content if turn.content.endswith("\n") else turn.content + "\n"
            block = text_block(content)
            update = (
                update_user_message(block) if turn.role == "user" else update_agent_message(block)
            )
            await self._client.session_update(handle.id, update)

    @staticmethod
    def _validate_workspace(cwd: str, additional_directories: list[str] | None) -> Path:
        if additional_directories:
            raise RequestError.invalid_params(
                {"reason": "Additional directories are not supported"}
            )
        root = Path(cwd).expanduser()
        if not root.is_absolute() or not root.is_dir():
            raise RequestError.invalid_params(
                {"cwd": cwd, "reason": "cwd must be an existing absolute directory"}
            )
        return root.resolve()

    @staticmethod
    def _store(root: Path) -> SessionStore:
        return SessionStore(root / ".nooa" / "sessions")

    @staticmethod
    def _prompt_text(prompt: list[Any]) -> str:
        parts: list[str] = []
        for block in prompt:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                parts.append(block.text)
            elif block_type == "resource_link":
                parts.append(f"Resource {block.name}: {block.uri}")
            else:
                raise RequestError.invalid_params(
                    {"reason": f"Unsupported prompt content type: {block_type!r}"}
                )
        text = "\n\n".join(parts)
        if not text.strip():
            raise RequestError.invalid_params({"reason": "Prompt text must not be empty"})
        return text

    @staticmethod
    def _slash_invocation(
        commands: CodingSlashCommandRegistry,
        text: str,
    ) -> tuple[str, str] | None:
        stripped = text.strip()
        if not stripped.startswith("/"):
            return None
        command_text = stripped[1:]
        parts = command_text.split(maxsplit=1)
        name = parts[0].lower() if parts else ""
        if not name or commands.get(name) is None:
            return None
        return name, parts[1] if len(parts) == 2 else ""

    async def close(self) -> None:
        first = await _acp_capture(self._sessions.close())
        for construction in tuple(self._construction):
            outcome = await _acp_capture(construction.close())
            if outcome.failure is None:
                self._construction.discard(construction)
            elif first.failure is None:
                first = outcome
        first.unwrap()


async def serve(llm_factory: Callable[[], UnifiedLLM], *, model_alias: str) -> None:
    adapter = CodingACPAdapter(llm_factory, model_alias=model_alias)
    primary: _ACPOutcome | None = None
    try:
        # session/close is registered by the router as unstable. initialize()
        # advertises the close capability, so without this flag the agent
        # promises a method that answers "method not found", and a client can
        # never release a session. session/list is stable and unaffected.
        await run_agent(cast(Agent, adapter), use_unstable_protocol=True)
    except BaseException as error:
        primary = _ACPOutcome(failure=error, traceback=error.__traceback__)
    cleanup = await _acp_capture(adapter.close())
    if primary is not None:
        primary.unwrap()
    # Retain the established best-effort policy for ordinary cleanup errors,
    # while public control failures propagate only after the close joins.
    if cleanup.failure is not None and not isinstance(cleanup.failure, Exception):
        cleanup.unwrap()


def _available_commands_update(commands: tuple[CodingSlashCommand, ...]):
    available: list[AvailableCommand] = []
    for command in commands:
        input_spec = (
            AvailableCommandInput(
                UnstructuredCommandInput(hint=command.argument_hint),
            )
            if command.argument_hint
            else None
        )
        available.append(
            AvailableCommand(
                name=command.name,
                description=command.description,
                input=input_spec,
            )
        )
    return update_available_commands(available)
