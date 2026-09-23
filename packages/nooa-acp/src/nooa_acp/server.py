# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP adapter for the host-neutral NOOA coding agent."""

from __future__ import annotations

import asyncio
import inspect
import logging
import signal
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, NamedTuple, cast

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
)
from nooa_cli.coding.factory import create_session_agent
from nooa_cli.coding.slash_commands import RESERVED_COMMAND_NAMES
from nooa_cli.interactive.controls import CONTROL_TYPES, behavior_commands
from nooa_cli.interactive.local_turn_policy import LocalTurnPolicy
from nooa_cli.interactive.options import (
    SessionOptions,
    configure_session_skills,
    connect_session_mcp,
)
from nooa_cli.interactive.session_paths import session_directory

from nooa.errors import GenerationError
from nooa.mcp import MCPManager, MCPTool
from nooa.sessions import (
    InvalidSessionIdError,
    SessionBusyError,
    SessionHandle,
    SessionNotFoundError,
    SessionResumed,
    SessionRuntime,
    SessionRuntimeClosedError,
    SessionRuntimePool,
    SessionStore,
)
from nooa.slash_dispatch import CoercionError
from nooa.storage.sqlite import (
    SessionAlreadyActiveError,
    _claim_path,
    claim_owner_is_confirmed_dead,
    is_sqlite_database_active,
)
from nooa.unifiedllm import UnifiedLLM
from nooa_acp._mcp_trace import MCPHandoffTrace
from nooa_acp.dispatcher import InteractiveSessionDispatcher
from nooa_acp.event_bridge import ACPEventBridge

logger = logging.getLogger(__name__)

_SESSION_PAGE_SIZE = 50
# Bound on prompt()'s wait for cancel_complete after a None turn result. A
# genuine cancel() RPC sets that event almost immediately; this only matters
# for the (rare, internal) case where the shared runtime settles a turn with
# no result outside of cancel() at all, which would otherwise wait forever.
_CANCEL_CONFIRMATION_TIMEOUT_SECONDS = 30


async def _close_in_order(*closers: Callable[[], Any] | None) -> None:
    """Run each closer in order, tolerating failures, then re-raise.

    Equivalent to nesting one ``try/finally`` per closer: every closer runs
    even if an earlier one fails, and the last failure propagates with the
    earlier ones chained as its ``__context__``.
    """
    pending: BaseException | None = None
    for close in closers:
        if close is None:
            continue
        try:
            result = close()
            if inspect.isawaitable(result):
                await result
        except BaseException as exc:
            if pending is not None:
                exc.__context__ = pending
            pending = exc
    if pending is not None:
        raise pending


class _SlashRequest(NamedTuple):
    """One parsed ``/name args`` request; ``command`` is None when unregistered."""

    name: str
    raw_args: str
    command: CodingSlashCommand | None


@dataclass(slots=True)
class _ACPSession:
    """Live resources owned by one ACP session runtime."""

    handle: SessionHandle
    agent: CodingAgent
    dispatcher: InteractiveSessionDispatcher
    bridge: ACPEventBridge
    commands: CodingSlashCommandRegistry
    startup_warnings: tuple[str, ...] = ()
    cancel_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cancel_complete: asyncio.Event = field(default_factory=asyncio.Event)
    notification_tasks: set[asyncio.Task[None]] = field(default_factory=set)
    commands_sent_on_prompt: bool = False
    restored: bool = False
    policy: LocalTurnPolicy | None = None
    # ACP requests must not interleave with the transcript replay on load.
    ready: bool = False
    # Set by prompt() right before admitting a non-slash turn, whose text it
    # already recorded synchronously (see prompt()). Lets the dequeue-
    # triggered callback below recognize and skip that one admission instead
    # of double-recording it, while still recording every other admission
    # path (e.g. a markdown skill's prepared next turn) exactly as before.
    _pending_synced_text: str | None = None

    def __post_init__(self) -> None:
        self.cancel_complete.set()

    async def close(self) -> None:
        for task in self.notification_tasks:
            task.cancel()
        if self.notification_tasks:
            await asyncio.gather(*self.notification_tasks, return_exceptions=True)
        self.notification_tasks.clear()
        try:
            try:
                if self.policy is not None:
                    await self.policy.shutdown()
            finally:
                try:
                    await self.dispatcher.runtime.cancel_work()
                finally:
                    self.handle.storage.save_snapshot(self.agent)
        finally:
            await self._close_resources()

    async def _close_resources(self) -> None:
        """Release every resource even if a checkpoint or earlier close fails."""
        await _close_in_order(
            self.bridge.close,
            self.commands.close,
            self.dispatcher.close,
            self.handle.close,
        )


class CodingACPAdapter:
    def __init__(
        self,
        llm_factory: Callable[[], UnifiedLLM],
        *,
        options_factory: Callable[[Path], SessionOptions] | None = None,
    ) -> None:
        self._llm_factory = llm_factory
        self._options_factory = options_factory or SessionOptions.load
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
        options = self._options_factory(root)
        llm = self._llm_factory()
        try:
            handle = self._store(root).create(
                model=llm.model,
                # Match create_session_agent()'s actual precedence exactly
                # (options.agent_spec and not options.legacy_agent): that
                # function's "or" here would persist agent_spec even when
                # legacy_agent overrides it, so a later resume's class-mismatch
                # check would compare against a class that was never built.
                agent=(
                    options.agent_spec
                    if (options.agent_spec and not options.legacy_agent)
                    else ("CodingAgent" if options.legacy_agent else "ExperimentalCodingAgent")
                ),
                working_directory=str(root),
                host="acp",
                check_same_thread=False,
            )
        except BaseException:
            await llm.aclose()
            raise
        try:
            runtime = await self._create_runtime(
                handle, root, mcp_servers, llm=llm, options=options
            )
        except BaseException:
            handle.close()
            self._store(root).delete(handle.id)
            raise
        runtime.value.ready = True
        self._defer_bootstrap_updates(runtime.value)
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
        try:
            handle = self._store(root).open(session_id, check_same_thread=False)
        except (InvalidSessionIdError, SessionNotFoundError):
            raise RequestError.resource_not_found(session_id) from None
        except SessionAlreadyActiveError as exc:
            # Clients may display only error.message, without error.data.
            raise RequestError(
                -32600,
                f"Session {session_id[:8]!r} is already open. "
                "Close it in the other client or tab, then try resuming again.",
                {"sessionId": session_id, "reason": str(exc)},
            ) from None
        runtime: SessionRuntime[_ACPSession] | None = None
        try:
            runtime = await self._create_runtime(handle, root, mcp_servers, restore=True)
            # After the replay, never during it: replay writes straight to the
            # client while the bridge pump drains bootstrap updates, so every
            # await here would otherwise let a commands update or an MCP warning
            # land in the middle of the restored conversation.
            await self._replay_session(handle)
            if runtime.is_closed:
                raise RequestError.resource_not_found(session_id)
            runtime.value.ready = True
            self._defer_bootstrap_updates(runtime.value)
        except BaseException:
            if runtime is not None:
                with suppress(KeyError):
                    await self._sessions.remove(session_id)
            else:
                handle.close()
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

        store = self._store(root)

        def _lock_status(session_id: str) -> Literal["free", "active", "orphaned"]:
            """ "orphaned" only when the claim's owner is provably dead."""
            db_path = store.path_for(session_id)
            if not is_sqlite_database_active(db_path):
                return "free"
            if claim_owner_is_confirmed_dead(_claim_path(db_path)):
                return "orphaned"
            return "active"

        # ACP has no standard field for disabling a busy entry in the picker,
        # so a session currently open elsewhere still must be omitted (like
        # native resume, also omitting startup-only sessions with no
        # conversation) -- opening it would just fail. But a claim whose
        # recorded owner process is confirmed dead (see
        # claim_owner_is_confirmed_dead) is never coming back and, unlike a
        # genuinely busy session, was otherwise permanently invisible with
        # no way to discover it for manual cleanup (claims never auto-expire
        # by design). Surface that case instead of hiding it, flagged via
        # ACP's _meta extension point so unaware clients still just work.
        # Filter before pagination so excluded sessions cannot hide later results.
        # The load-time lock still handles sessions opened after this check.
        def _list_and_filter() -> list[tuple[Any, bool]]:
            # store.list() and _locked_state() are pure synchronous filesystem
            # I/O (no awaits), one flock probe per non-empty session -- run
            # off the event loop so a workspace with many sessions can't
            # stall every other open session's prompt/cancel/response
            # delivery in this same process for the duration of the scan.
            result = []
            for info in store.list(limit=None):
                if info.turn_count == 0:
                    continue
                status = _lock_status(info.id)
                if status == "active":
                    continue
                result.append((info, status == "orphaned"))
            return result

        found = await asyncio.to_thread(_list_and_filter)
        page = found[offset : offset + _SESSION_PAGE_SIZE]
        sessions = [
            ACPSessionInfo(
                session_id=info.id,
                # Older native sessions persisted CLI values such as "." or
                # "../workspace". ACP requires an absolute cwd for every list
                # entry. Their workspace is the store's requested scope; using
                # the server process cwd would resolve relative paths twice.
                cwd=(
                    info.working_directory
                    if Path(info.working_directory).is_absolute()
                    else str(root)
                ),
                # Old ACP sessions never requested an automatic title. Give
                # pickers a nonempty label without rewriting durable metadata.
                title=info.title or f"Untitled session [{info.id[:8]}]",
                updated_at=datetime.fromtimestamp(info.last_active, UTC).isoformat(),
                field_meta={"dev.nooa/orphaned_claim": True} if orphaned else None,
            )
            for info, orphaned in page
        ]
        next_cursor = str(offset + len(page)) if len(found) > offset + len(page) else None
        return ListSessionsResponse(sessions=sessions, next_cursor=next_cursor)

    async def close_session(self, session_id: str, **kwargs: Any) -> CloseSessionResponse:
        del kwargs
        await self._get_runtime(session_id)
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
            async with runtime.turn():
                session = runtime.value
                session.cancel_complete.clear()
                try:
                    if not session.commands_sent_on_prompt:
                        session.bridge.publish(
                            _available_commands_update(session.commands.commands())
                        )
                        session.commands_sent_on_prompt = True
                    slash = self._slash_invocation(session.commands, text)
                    if slash is None or slash.command is None:
                        if slash is not None and slash.name in RESERVED_COMMAND_NAMES:
                            available = ", ".join(f"/{name}" for name in CONTROL_TYPES)
                            message = (
                                f"NOOA /{slash.name} is not available through ACP yet. "
                                f"Available behavior controls: {available}. "
                                "Use native NOOA for the other agent controls."
                            )
                            session.bridge.publish(update_agent_message(text_block(message)))
                            await session.bridge.flush()
                            return PromptResponse(stop_reason="end_turn")
                        # Recorded here, synchronously, instead of relying solely
                        # on the runtime's dequeue-triggered callback:
                        # dispatcher.submit() wraps admission in a freshly created
                        # asyncio.Task, and a cancel() arriving before the event
                        # loop ever schedules that task's first run cancels it
                        # without running any of its body -- the text would never
                        # even reach the queue, let alone get dequeued, and be
                        # lost with no trace. record_unless_already_synced()
                        # above skips the matching dequeue-triggered call so a
                        # normal (non-cancelled) turn isn't recorded twice.
                        session._pending_synced_text = text
                        session.handle.record_user_message(text)
                        result = await session.dispatcher.submit(text)
                    else:
                        name, raw_args, command = slash
                        if command._method is not None and not command.is_control:
                            session.handle.record_user_message(text)
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
                                if command is not None and command.is_control:
                                    if message:
                                        session.bridge.publish(
                                            update_agent_message(text_block(message))
                                        )
                                    session.handle.storage.save_snapshot(session.agent)
                                elif message:
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
                    if message.startswith(
                        "Empty response: the model used all available output tokens"
                    ):
                        return PromptResponse(stop_reason="max_tokens")
                    if message.startswith("Generation failed after ") and (
                        "max_iterations=" in message or "max_retries=" in message
                    ):
                        return PromptResponse(stop_reason="max_turn_requests")
                    raise RequestError(-32603, message, {"details": message}) from exc
                if result is None:
                    # A None result usually means an in-flight cancel() RPC,
                    # which always sets cancel_complete in its finally block
                    # (near-instantly relative to this wait). But the shared
                    # LocalAgentRunner can also settle a turn with no result
                    # on a purely internal completion path that never goes
                    # through cancel() at all -- nothing would ever set the
                    # event then. Bound the wait so that ambiguity can never
                    # turn into a permanently held turn lock; a real cancel
                    # is expected to finish this wait well within the bound.
                    confirmed = True
                    try:
                        async with asyncio.timeout(_CANCEL_CONFIRMATION_TIMEOUT_SECONDS):
                            await session.cancel_complete.wait()
                    except TimeoutError:
                        confirmed = False
                        logger.warning(
                            "Session %s: turn ended with no result and no cancel "
                            "confirmation within %ss; releasing the turn lock anyway.",
                            session_id,
                            _CANCEL_CONFIRMATION_TIMEOUT_SECONDS,
                        )
                        session.cancel_complete.set()
                    # stop_reason and the tool card both carry the outcome, but
                    # a collapsed card shows nothing and the turn just goes
                    # quiet. Record it as a real message so the conversation —
                    # and the durable transcript on resume — says what happened.
                    # The timeout above means this None result was never
                    # actually confirmed as a cancel, so don't claim one.
                    session.agent.message(
                        "Stopped at your request."
                        if confirmed
                        else "The turn ended without a result."
                    )
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
        options: SessionOptions | None = None,
        restore: bool = False,
    ) -> SessionRuntime[_ACPSession]:
        llm = llm or self._llm_factory()
        if self._client is None:
            await llm.aclose()
            raise RequestError.internal_error({"reason": "ACP client is not connected"})
        agent: CodingAgent | None = None
        commands: CodingSlashCommandRegistry | None = None
        dispatcher: InteractiveSessionDispatcher | None = None
        bridge: ACPEventBridge | None = None
        value: _ACPSession | None = None
        try:
            options = options or self._options_factory(root)
            mcp, mcp_warnings = await self._create_mcp_tools(mcp_servers)
            agent = create_session_agent(llm=llm, storage=handle.storage, options=options)
            # Computed from the freshly constructed agent's own class and the
            # saved session identity -- both known before any restore -- so
            # a class mismatch is detected before Snapshot.restore() setattrs
            # every stored attribute into this agent with no class check of
            # its own, not after. Restoring anyway despite a mismatch (only
            # warning, not refusing) is the existing, deliberate, tested
            # contract (see test_resume_warns_when_host_selects_a_different_
            # agent); only the detection now happens ahead of the restore.
            agent_mismatch_warning: str | None = None
            if restore and handle.info.agent:
                from nooa_cli.coding.identity import canonical_agent_spec

                saved = canonical_agent_spec(handle.info.agent)
                current = f"{type(agent).__module__}:{type(agent).__qualname__}"
                if saved not in {current, type(agent).__name__, options.agent_spec}:
                    agent_mismatch_warning = (
                        f"Session was created with agent {handle.info.agent!r}; "
                        f"resuming with {current!r} from the current host options."
                    )
            restored = handle.storage.restore_latest_snapshot(agent) if restore else False
            agent._session_manager = handle
            registration_warnings = configure_session_skills(agent, options)
            if agent_mismatch_warning is not None:
                registration_warnings.append(agent_mismatch_warning)
            registration_warnings.extend(await connect_session_mcp(agent, options))
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
            dispatcher = InteractiveSessionDispatcher(agent)
            bridge = ACPEventBridge(agent, self._client, handle.id)
            bridge.watch_session(handle)

            async def emit_status(status: Any) -> None:
                # ACP stop reasons represent normal turn completion.
                logger.debug("session %s: %s", handle.id, status)

            policy = LocalTurnPolicy(
                emit_output=emit_status,
            )

            async def checkpoint(current: Any, result: Any) -> None:
                await policy.after_handle(current, result)
                handle.storage.save_snapshot(current)

            dispatcher.runtime.set_dispatch_hooks(
                on_after_handle=checkpoint,
            )

            commands = CodingSlashCommandRegistry(agent, skills_dirs=options.skills_dirs)
            commands.set_controls(
                behavior_commands(
                    agent,
                    options,
                    workspace=root,
                    command_registry=commands,
                )
            )
            value = _ACPSession(
                handle,
                agent,
                dispatcher,
                bridge,
                commands,
                startup_warnings=(*mcp_warnings, *registration_warnings),
                restored=restored,
                policy=policy,
            )

            def record_unless_already_synced(text: str) -> None:
                if text is value._pending_synced_text:
                    value._pending_synced_text = None
                    return
                handle.record_user_message(text)

            dispatcher.runtime.set_user_message_accepted_callback(record_unless_already_synced)
            commands.set_on_change(
                lambda available: bridge.publish(_available_commands_update(available)),
            )
            try:
                runtime = await self._sessions.add(handle.id, value)
                return runtime
            except ValueError:
                raise RequestError.invalid_request(
                    {"sessionId": handle.id, "reason": "Session is already loaded"}
                ) from None
        except BaseException:
            if value is not None:
                await value.close()
            elif agent is not None:
                # Same order as _ACPSession._close_resources, restricted to
                # whatever got built before the failure. dispatcher.close()
                # closes the agent itself, so the bare agent is closed only
                # when no dispatcher wrapped it yet.
                await _close_in_order(
                    bridge.close if bridge is not None else None,
                    commands.close if commands is not None else None,
                    dispatcher.close if dispatcher is not None else agent.close,
                )
            else:
                await llm.aclose()
            raise

    async def _create_mcp_tools(
        self,
        mcp_servers: list[Any] | None,
    ) -> tuple[dict[str, MCPTool], tuple[str, ...]]:
        servers = list(mcp_servers or [])
        supported_types = (McpServerStdio, HttpMcpServer, SseMcpServer)
        tools: dict[str, MCPTool] = {}
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
            runtime = await self._sessions.get(session_id)
        except KeyError:
            raise RequestError.resource_not_found(session_id) from None
        if not runtime.value.ready:
            raise RequestError.resource_not_found(session_id)
        return runtime

    @staticmethod
    def _defer_bootstrap_updates(session: _ACPSession) -> None:
        """Publish bootstrap updates after the session response can reach clients such as Zed."""

        async def _publish() -> None:
            await asyncio.sleep(0)
            session.agent.event_manager.add(
                SessionResumed(session_id=session.handle.id, restored=session.restored)
            )
            session.bridge.publish_best_effort(
                _available_commands_update(session.commands.commands())
            )
            if session.startup_warnings:
                details = "\n".join(f"- {warning}" for warning in session.startup_warnings)
                session.bridge.publish_best_effort(
                    update_agent_message(
                        text_block(
                            "NOOA started with configuration warnings. The session is still "
                            f"usable.\n\n{details}"
                        )
                    )
                )
            await session.bridge.flush()

        task = asyncio.create_task(_publish(), name="nooa-acp-bootstrap")
        session.notification_tasks.add(task)

        def _finished(done: asyncio.Task[None]) -> None:
            session.notification_tasks.discard(done)
            if not done.cancelled():
                done.exception()

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
        return SessionStore(session_directory(root))

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
    ) -> _SlashRequest | None:
        """Parse ``/name args`` once, resolving the command if one is registered.

        Returns None for text that is not a slash request at all. A request
        naming an unregistered command still comes back (with ``command``
        None) so the caller can tell a reserved-but-unavailable control apart
        from a plain prompt without re-parsing the text.
        """
        stripped = text.strip()
        if not stripped.startswith("/"):
            return None
        parts = stripped[1:].split(maxsplit=1)
        name = parts[0].lower() if parts else ""
        if not name:
            return None
        return _SlashRequest(name, parts[1] if len(parts) == 2 else "", commands.get(name))

    async def close(self) -> None:
        await self._sessions.close()


async def serve(
    llm_factory: Callable[[], UnifiedLLM],
    *,
    options_factory: Callable[[Path], SessionOptions] | None = None,
) -> None:
    adapter = CodingACPAdapter(llm_factory, options_factory=options_factory)
    mcp_trace = MCPHandoffTrace.from_env()
    # ACP clients may terminate their subprocess instead of closing stdin.
    # Let normal teardown save snapshots and release shared-filesystem claims.
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    terminating = False

    def terminate() -> None:
        nonlocal terminating
        if not terminating and task is not None:
            terminating = True
            task.cancel()
            # A second SIGTERM must be able to kill the process outright even
            # if cleanup hangs (e.g. SessionRuntime._close_once() blocked on
            # the turn lock): the process's inherited SIGTERM handler could be
            # SIG_IGN or another custom handler, so simply restoring it
            # (finally, below) is not guaranteed to make a second signal
            # terminate anything. Install the default action right away.
            loop.remove_signal_handler(signal.SIGTERM)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)

    signal_installed = False
    try:
        loop.add_signal_handler(signal.SIGTERM, terminate)
        signal_installed = True
    except (NotImplementedError, RuntimeError):
        pass  # Non-Unix event loops or an embedded server outside the main thread.
    try:
        # session/close is registered by the router as unstable. initialize()
        # advertises the close capability, so without this flag the agent
        # promises a method that answers "method not found", and a client can
        # never release a session. session/list is stable and unaffected.
        await run_agent(
            cast(Agent, adapter),
            use_unstable_protocol=True,
            observers=[mcp_trace] if mcp_trace is not None else [],
        )
    except asyncio.CancelledError:
        if not terminating:
            raise
    finally:
        try:
            with suppress(Exception):
                await adapter.close()
        finally:
            if signal_installed:
                loop.remove_signal_handler(signal.SIGTERM)
                signal.signal(signal.SIGTERM, previous_sigterm)


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
