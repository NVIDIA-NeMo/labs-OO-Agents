# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCPRegistry — agent-facing MCP server management mirroring SkillRegistry."""

from __future__ import annotations

import asyncio
import copy
import fnmatch
import inspect
import keyword
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal

from nooa.agentdoc import hidden, spec
from nooa.skill import Skill, slash_command

from .mcp_approval import (
    MCPApprovalRequest,
    MCPApprovalRequired,
    MCPApprovalStore,
    _safe_display,
    build_approval_request,
    redact_approved_environment,
    resolve_approved_environment,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

_MAX_TOOL_NAMES = 8
_ALLOWED_CONNECT_KWARGS = {"tool_call_timeout"}
_FACTORY_CONFIG_FIELDS = (
    "command",
    "args",
    "env",
    "url",
    "headers",
    "transport",
    "oauth_client_id",
    "oauth_redirect_uri",
    "oauth_scope",
    "oauth_open_browser",
    "oauth_manual",
)


def _to_attr_name(name: str) -> str:
    """Convert common human-readable server-name separators for attribute access."""
    return name.replace("-", "_").replace(" ", "_")


class MCPRegistry(Skill):
    """Connect to MCP servers and surface their tools to the agent.

    Mirrors :class:`SkillRegistry`. An MCP server is *configured* (in
    ``.mcp.json`` or the TUI ``tui.mcp_servers`` settings.yaml block), *connected* (an
    authenticated client that lists the server's tools), and *activated* (its
    tools are listed as callable free functions in the ``<mcp>`` context
    block). A connected-but-deactivated server keeps its generated tool/client
    cached without flooding the agent's context; each tool invocation still
    opens and closes its own transport session. This mirrors the visibility
    rationale of ``SkillRegistry.activate/deactivate``.

    **The ``<mcp>`` context block is the single source of truth for what is
    callable.** Each *active* server contributes one free-function line per
    tool (signature + first docstring line); the agent calls them as
    ``self.<server>.<tool>(...)``. The ``self.<server>`` attribute is always
    hidden from ``doc(self)`` so the top-level surface never bloats —
    activation changes the block, not attribute visibility.

    ## Lifecycle

    ::

        self.mcp.discovered()              # configured server names
        await self.mcp.connect(["maas"])   # open session(s), attach as self.maas
        self.mcp.connected()               # connected server names
        self.mcp.activate(["maas"])        # list maas tools in <mcp> as free functions
        self.mcp.deactivate(["maas"])      # drop from <mcp> (client stays cached)
        self.mcp.status()                  # render the <mcp> block

    ``connect``/``activate``/``deactivate`` take fnmatch globs
    (``"maas"``, ``"*"``, ``"conf-*"``).

    ## User approval and environment secrets

    Configuration is discovery, not trust. Before any transport is created,
    the TUI requires a user-level approval for the fingerprint of the complete
    effective server definition. The human reviews and grants it through
    ``/mcp approve <name> <code>``; the agent cannot grant approval through this
    skill. Changes to the URL, command, args, headers, env, or OAuth settings
    invalidate approval. Only after the exact fingerprint matches does the TUI resolve
    ``${VAR}`` placeholders from the host environment. Core ``MCPManager`` keeps
    config placeholders literal.

    ## Adding a server

    Register an in-memory server entry, then connect it::

        self.mcp.register(
            "myserver",
            url="https://host/mcp",
            transport="streamable-http",
            headers={"Authorization": "Bearer ..."},
        )
        await self.mcp.connect(["myserver"])

    To persist a server, add a ``tui.mcp_servers.<name>`` block to
    ``.nooa/settings.yaml`` or a VS Code /
    Claude-style ``.mcp.json``; ``register`` is in-memory only.

    ## OAuth: what the AGENT can do vs. what the HUMAN must do

    HTTP servers that return 401 trigger an OAuth flow (RFC 9728 / dynamic
    client registration) on first connect. **OAuth consent is inherently a
    human action** — the agent cannot click "Approve" in a browser. Know which
    side of the line each step is on:

    **The agent CAN, unattended:**

    - ``register(...)`` a server and request ``connect``/``activate``/``deactivate``.
      A new or changed definition still needs the human's ``/mcp approve`` action.
    - Reconnect a server whose token is already **cached** (a prior successful
      auth in this environment) — no human needed; the cached token is reused.
    - Put auth settings such as ``oauth_client_id``/``oauth_scope`` or static
      authorization headers in the approved server definition. After exact-config
      approval, a static-key or pre-authorized server connects fully unattended.

    **The HUMAN MUST do (agent cannot substitute):**

    - **Grant first-time OAuth consent.** On a 401 with no cached token, a real
      person has to open the consent URL, approve, and let the callback return.
      The TUI surfaces the URL and collects the pasted code/callback in a masked
      in-app prompt, but the agent cannot approve on the user's behalf. Prefer
      letting the human drive ``/mcp connect <name>`` so a single flow runs
      start-to-finish — OAuth codes are single-use, and a half-finished
      agent-driven flow burns the code (causing a confusing 401 on retry).
    - **Be present for the browser handoff.** ``oauth_open_browser`` (default
      True) opens the system browser; it falls back to manual when none exists.
      Either way a human completes consent.

    **Timeout / no-hang guarantee.** ``connect`` bounds the OAuth wait with
    ``oauth_timeout`` (default 180s) — a never-returning browser callback raises
    a clear ``TimeoutError`` telling the user to retry, rather than wedging the
    agent indefinitely. Pass ``on_connecting=cb`` to surface "launching browser
    for <name>..." feedback *before* the wait begins (the UI must not look
    frozen while the user is sent to a browser).

    Tokens are cached with the dynamically registered client credentials, so
    later connects in the same environment skip the prompt — the *second* connect
    is something the agent can do unattended.

    ## Developer API

    ``MCPManager`` (``nooa.mcp``) stays a stateless, dependency-free
    factory — library code calls ``MCPManager.create_from_server(...)``
    directly. This registry wraps it to hold connection/activation state for an
    agent.
    """

    __nosnapshot__ = True
    context_block = ("mcp", "self.mcp.status()")

    _agent: Annotated[Any, hidden] = None

    @slash_command(
        "mcp-add",
        argument_hint="<server info: name, URL, transport, auth notes>",
        output_to_agent=True,
    )
    async def mcp_add_command(self, args: str) -> str:
        """Add a new MCP server: hand the details to the agent to wire it up.

        The user pastes whatever they have about a server — a name and URL, a
        ``claude mcp add ...`` line, a docs snippet, an OAuth client id, etc.
        This does NOT edit anything itself; it returns a task for the agent,
        which reads the details, writes the ``tui.mcp_servers.<name>`` block in
        ``.nooa/settings.yaml``, and guides the user through connecting
        (OAuth/host-browser handoff as needed).
        """
        details = args.strip()
        if not details:
            return (
                "Usage: /mcp-add <server info>\n"
                "Paste a name + URL (and transport/auth notes), or a "
                "`claude mcp add ...` line, e.g.:\n"
                "  /mcp-add maas-gdrive https://maas.prd.astra.nvidia.com/maas/gdrive/mcp "
                "streamable-http"
            )
        config_path = self._config_path()
        configured = ", ".join(self.discovered()) or "(none)"
        return (
            "The user wants to add a new MCP server. Here are the details they provided:\n\n"
            f"{details}\n\n"
            "Do the following:\n"
            "1. Parse the server name, URL, transport (default `streamable-http` for HTTP "
            "URLs), and any auth info (OAuth client_id, static API key/headers).\n"
            f"2. Add a `tui.mcp_servers.<name>` YAML block to the TUI config at `{config_path}` "
            "(create the file/section if missing; do NOT clobber existing servers). Use an "
            "environment placeholder in `headers` for a static API key, or `oauth_client_id` "
            "for a pre-provisioned OAuth client. Never write a secret value into project "
            "config. Don't set `oauth_manual` unless the server requires OOB.\n"
            "3. Tell the user to run `/mcp connect <name>` to review the exact config, then "
            "personally run the displayed `/mcp approve <name> <code>` command. The agent "
            "must not approve MCP config. Explain browser consent / masked code entry if the "
            "server needs OAuth.\n"
            "4. Confirm what you wrote and show the resulting config block without resolving "
            "or printing secret values.\n\n"
            f"Currently configured servers: {configured}.\n"
            f"Config file: {config_path}."
        )

    def _config_path(self) -> Path:
        """Return the project-local TUI settings.yaml path."""
        from nooa.paths import get_project_dir

        from .settings import SETTINGS_FILENAME

        return get_project_dir(SETTINGS_FILENAME)

    def __init__(
        self,
        mcp_file: Path | None = None,
        servers: dict[str, dict[str, Any]] | None = None,
        approval_path: Path | None = None,
        watch_settings: bool = False,
    ) -> None:
        """Initialize with optional config sources.

        Args:
            mcp_file: Path to a VS Code / Claude-style ``.mcp.json``.
            servers: Inline server config (from TUI ``tui.mcp_servers`` in settings.yaml).
            approval_path: Override the user approval store path (primarily for tests).
            watch_settings: Reload layered TUI settings before lifecycle commands.
                The production TUI enables this so agent-assisted config edits
                become available without restarting the process.
        """
        self._mcp_file = mcp_file
        self._servers: dict[str, dict[str, Any]] = copy.deepcopy(servers or {})
        self._watch_settings = watch_settings
        self._settings_server_names = set(self._servers)
        self._registered_server_names: set[str] = set()
        self._approval_store = MCPApprovalStore(approval_path)
        self._connected: dict[str, Any] = {}
        self._activated: set[str] = set()
        self._oauth_code_prompt: Callable[[str], Awaitable[str]] | None = None
        # Servers whose connect() is currently in-flight — guards against a
        # retry spawning a second concurrent create_from_server (and second
        # browser window / racing token-cache write) while the first to_thread
        # is still running after a wait_for timeout (the thread keeps going).
        self._pending: set[str] = set()
        super().__init__()

    def refresh_settings(self) -> list[str]:
        """Reload inline MCP definitions from the layered TUI settings.

        The agent-assisted ``/mcp-add`` workflow edits ``settings.yaml`` while
        the TUI is already running. Production registries opt into this refresh
        so the next ``/mcp`` lifecycle command sees the exact current config.
        Explicit in-memory registrations remain available until removed.

        Returns the names loaded from settings. Test/programmatic registries do
        nothing unless ``watch_settings=True`` was requested.
        """
        if not self._watch_settings:
            return []

        from nooa.layered_config import load_layered_yaml

        from .settings import SETTINGS_ENV_VAR, SETTINGS_FILENAME

        data = load_layered_yaml(SETTINGS_FILENAME, SETTINGS_ENV_VAR)
        tui = data.get("tui", {})
        raw_servers = tui.get("mcp_servers", {}) if isinstance(tui, dict) else {}
        if raw_servers is None:
            raw_servers = {}
        if not isinstance(raw_servers, dict):
            raise ValueError("tui.mcp_servers must be a mapping")

        fresh: dict[str, dict[str, Any]] = {}
        for name, definition in raw_servers.items():
            if not isinstance(name, str) or not isinstance(definition, dict):
                raise ValueError("each tui.mcp_servers entry must map a name to a mapping")
            fresh[name] = copy.deepcopy(definition)

        removed = self._settings_server_names - set(fresh)
        changed = {
            name
            for name, definition in fresh.items()
            if name in self._servers and self._servers[name] != definition
        }
        for name in sorted((removed | changed) & set(self._connected)):
            self.deactivate([name])
            self._detach(name)
        for name in removed:
            if name not in self._registered_server_names:
                self._servers.pop(name, None)
                self._activated.discard(name)
        self._servers.update(fresh)
        # A definition first registered in memory by `/mcp add` becomes
        # settings-owned as soon as the persisted layer contains it.
        self._registered_server_names.difference_update(fresh)
        self._settings_server_names = set(fresh)
        return sorted(fresh)

    def _bind_oauth_code_prompt(self, callback: Callable[[str], Awaitable[str]] | None) -> None:
        """Bind the host TUI's thread-safe manual OAuth input bridge."""
        self._oauth_code_prompt = callback

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def discovered(self) -> list[str]:
        """All configured server names (``.mcp.json`` + inline + registered).

        Returns an empty list if the ``mcp`` package is unavailable (the TUI
        package depends on it, but ``MCPManager`` lives in core where it stays
        optional), so the ``<mcp>`` context block renders cleanly instead of
        leaking an ``ImportError`` every turn.
        """
        try:
            from nooa.mcp import MCPManager
        except ImportError:
            return []
        return sorted(MCPManager.list_servers(self._mcp_file, servers=self._servers))

    def register(
        self,
        name: str,
        *,
        url: str | None = None,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        transport: Literal["stdio", "sse", "streamable-http"] | None = None,
        headers: dict[str, str] | None = None,
        oauth_client_id: str | None = None,
        oauth_redirect_uri: str | None = None,
        oauth_scope: str | None = None,
        oauth_open_browser: bool | None = None,
        oauth_manual: bool | None = None,
    ) -> None:
        """Add an in-memory server entry (not persisted to config).

        Use ``url``/``headers``/``transport`` for HTTP servers, or
        ``command``/``args``/``env`` for stdio servers. To persist, add a
        ``tui.mcp_servers.<name>`` block to settings.yaml instead.
        """
        if name in self._connected or name in self._pending:
            raise RuntimeError(f"Disconnect MCP server {name!r} before changing its configuration")
        entry: dict[str, Any] = {}
        for key, val in (
            ("url", url),
            ("command", command),
            ("args", args),
            ("env", env),
            ("transport", transport),
            ("headers", headers),
            ("oauth_client_id", oauth_client_id),
            ("oauth_redirect_uri", oauth_redirect_uri),
            ("oauth_scope", oauth_scope),
            ("oauth_open_browser", oauth_open_browser),
            ("oauth_manual", oauth_manual),
        ):
            if val is not None:
                entry[key] = val
        self._servers[name] = copy.deepcopy(entry)
        self._registered_server_names.add(name)

    # ------------------------------------------------------------------
    # User approval boundary
    # ------------------------------------------------------------------

    def _approval_request(self, name: str) -> MCPApprovalRequest:
        return build_approval_request(
            name,
            mcp_file=self._mcp_file,
            servers=self._servers,
        )

    def _is_approved(self, name: str) -> bool:
        return self._approval_store.is_approved(self._approval_request(name))

    def _approve(self, name: str, confirmation: str) -> MCPApprovalRequest:
        request = self._approval_request(name)
        if not request.accepts_confirmation(confirmation):
            raise ValueError(
                "Approval code does not match the current MCP configuration. "
                "Run `/mcp approve " + name + "` to review it again."
            )
        self._approval_store.approve(request)
        return request

    def _revoke_approvals(self, name: str) -> bool:
        return self._approval_store.revoke_server(name)

    def _prepared_connection(
        self, name: str
    ) -> tuple[
        MCPApprovalRequest,
        dict[str, dict[str, Any]],
        dict[str, Any],
        dict[str, str],
    ]:
        """Return an approved config as explicit, non-reexpanded factory arguments.

        The blank server override prevents either the selected ``.mcp.json`` or
        the process-default one from contributing fields after approval. Passing
        resolved values as explicit arguments also keeps older core releases from
        applying their legacy environment expansion a second time.
        """
        request = self._approval_request(name)
        if not self._approval_store.is_approved(request):
            raise MCPApprovalRequired(request)
        approved_environment = {
            variable: value
            for variable in request.variables
            if (value := os.environ.get(variable)) is not None
        }
        resolved = resolve_approved_environment(request, approved_environment)
        factory_config = {
            field: copy.deepcopy(resolved[field])
            for field in _FACTORY_CONFIG_FIELDS
            if field in resolved
        }
        return request, {name: {}}, factory_config, approved_environment

    # ------------------------------------------------------------------
    # Connecting
    # ------------------------------------------------------------------

    def connected(self) -> list[str]:
        """Currently connected server names."""
        return sorted(self._connected)

    async def connect(
        self,
        patterns: list[str],
        *,
        oauth_timeout: float = 180.0,
        on_connecting: Callable[[str], None] | None = None,
        activate: bool = True,
        **kwargs: Any,
    ) -> list[str]:
        """Connect to configured servers matching *patterns*.

        Each connected server is attached to the agent as ``self.<name>``
        (hyphens and spaces become underscores), hidden from ``doc(self)``. By default
        newly connected servers are also activated (listed in ``<mcp>``); pass
        ``activate=False`` to keep them connected but unlisted.

        Security-relevant connection settings, including OAuth options, must be
        part of the approved server definition. ``tool_call_timeout`` is the only
        operational factory override accepted through extra keyword arguments.

        Every configured server must first be approved by the user through
        ``/mcp approve``. The approval covers the complete effective config;
        environment placeholders are resolved only after it matches.

        Returns the list of server names connected by this call.
        """
        from nooa.mcp import MCPManager

        unexpected = sorted(set(kwargs) - _ALLOWED_CONNECT_KWARGS)
        if unexpected:
            names = ", ".join(unexpected)
            raise TypeError(
                "Connection-time MCP endpoint/config overrides are not allowed by the TUI "
                f"approval boundary: {names}. Register a new server definition instead."
            )

        matched = self._match(patterns, set(self.discovered()))
        newly: list[str] = []
        for name in sorted(matched):
            if name in self._connected:
                continue
            # In-flight guard: prevent a retry from starting a SECOND concurrent
            # connect for the same server while a prior attempt is still waiting
            # on OAuth — that would race token-cache writes, duplicate client
            # registrations, or open two browser windows.
            if name in self._pending:
                raise RuntimeError(
                    f"Connect to {name!r} is already in progress (a prior attempt "
                    f"may still be waiting on OAuth). Wait for it to finish or time "
                    f"out before retrying."
                )
            self._validate_attach_name(name)
            request, prepared_servers, approved_kwargs, approved_environment = (
                self._prepared_connection(name)
            )
            # Feedback BEFORE the (possibly long, browser-launching) OAuth wait —
            # otherwise the UI looks frozen while the user is sent to a browser.
            if on_connecting is not None:
                on_connecting(name)
            self._pending.add(name)
            worker: asyncio.Task[Any] | None = None
            try:
                # Outer to_thread keeps the prompt_toolkit UI loop painting during
                # OAuth waits (see !373). The OAuth wait itself is bounded by a
                # SINGLE authoritative timeout owned by the OAuth layer: the local
                # callback server polls every 1s and exits on it, so there is no
                # orphaned thread — we pass oauth_timeout straight down rather than
                # stacking a second wait_for here.
                create_kwargs = {**approved_kwargs, **kwargs}
                if oauth_timeout is not None:
                    create_kwargs["oauth_timeout"] = oauth_timeout
                worker = asyncio.create_task(
                    asyncio.to_thread(
                        MCPManager.create_from_server,
                        name,
                        mcp_file=self._mcp_file,
                        servers=prepared_servers,
                        oauth_code_prompt=self._oauth_code_prompt,
                        **create_kwargs,
                    )
                )
                tool = await asyncio.shield(worker)
            except asyncio.CancelledError:
                assert worker is not None

                def _finish_cancelled_connect(
                    finished: asyncio.Task[Any], server_name: str = name
                ) -> None:
                    self._pending.discard(server_name)
                    if not finished.cancelled():
                        # Retrieve a late worker exception so asyncio does not
                        # report it as an unhandled background task.
                        finished.exception()

                worker.add_done_callback(_finish_cancelled_connect)
                raise
            except Exception as exc:
                message = redact_approved_environment(request, exc, approved_environment)
                raise RuntimeError(f"MCP server {name!r} connection failed: {message}") from None
            finally:
                if worker is None or worker.done():
                    self._pending.discard(name)
            self._attach(name, tool)
            newly.append(name)
        if activate and newly:
            self.activate(newly)
        return newly

    async def disconnect(self, patterns: list[str]) -> list[str]:
        """Close connected servers matching *patterns* and detach them.

        Returns the list of server names disconnected.
        """
        matched = self._match(patterns, set(self._connected))
        for name in sorted(matched):
            self.deactivate([name])
            self._detach(name)
        return sorted(matched)

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------

    def activated(self) -> list[str]:
        """Currently activated (listed in ``<mcp>``) server names."""
        return sorted(self._activated)

    def activate(self, patterns: list[str]) -> None:
        """List connected servers' tools as free functions in ``<mcp>``."""
        matched = self._match(patterns, set(self._connected))
        self._activated.update(matched)

    def deactivate(self, patterns: list[str]) -> None:
        """Drop a connected server's tools from ``<mcp>`` (session stays open)."""
        matched = self._match(patterns, set(self._activated))
        self._activated.difference_update(matched)

    # ------------------------------------------------------------------
    # Status / context block
    # ------------------------------------------------------------------

    def status(self) -> str:
        """Render the ``<mcp>`` block, mirroring ``SkillRegistry.status()``.

        Three sections of uniform ``self.<attr>   <one-line summary>`` rows so a
        server is never invisible while its client is still alive:

        * **Active** — connected and listed for the agent (callable now).
        * **Connected (inactive)** — session/client still open, hidden from the
          agent until re-activated.
        * **Configured** — known but not connected; how to connect.

        Matches the ``<skills>`` block so the two read identically.
        """
        configured = self.discovered()
        if not configured:
            return "No MCP servers configured."

        lines: list[str] = []

        active = [n for n in configured if n in self._activated]
        if active:
            lines.append(
                "Active MCP servers (use via self.<attr>, docs via doc(self.<attr>),"
                " deactivate via self.mcp.deactivate(['name'])):"
            )
            for name in active:
                attr = _to_attr_name(name)
                lines.append(f"  self.{attr:22s} {self._server_summary(name)}")

        inactive = [n for n in configured if n in self._connected and n not in self._activated]
        if inactive:
            if lines:
                lines.append("")
            lines.append(
                "Connected but inactive (client/token still live;"
                " activate with self.mcp.activate(['name'])):"
            )
            for name in inactive:
                attr = _to_attr_name(name)
                lines.append(f"  self.{attr:22s} {self._server_summary(name)}")

        available = [n for n in configured if n not in self._connected]
        if available:
            if lines:
                lines.append("")
            lines.append("Configured MCP servers (connect with self.mcp.connect(['name'])):")
            for name in available:
                safe_name = _safe_display(name)
                lines.append(f"  {safe_name:24s} {self._server_summary(name)}")

        return "\n".join(lines)

    def _server_summary(self, name: str) -> str:
        """One-line summary for a server row (tool names if connected, else endpoint)."""
        tool = self._connected.get(name)
        if tool is not None:
            method_names = sorted(getattr(type(tool), "_tool_method_names", ()) or [])
            if not method_names:
                method_names = sorted(
                    m
                    for m in dir(type(tool))
                    if not m.startswith("_")
                    and callable(getattr(tool, m, None))
                    and getattr(getattr(type(tool), m, None), "__qualname__", "").startswith(
                        type(tool).__name__
                    )
                )
            n = len(method_names)
            shown = ", ".join(method_names[:_MAX_TOOL_NAMES])
            extra = n - min(n, _MAX_TOOL_NAMES)
            tools = shown + (f", +{extra} more" if extra > 0 else "")
            return f"{tools} ({n} tool{'s' if n != 1 else ''})" if n else "(no tools)"
        try:
            request = self._approval_request(name)
        except Exception:
            return "(invalid MCP configuration)"
        suffix = "" if self._approval_store.is_approved(request) else " [approval required]"
        return f"{request.target} ({request.transport}){suffix}"

    # ------------------------------------------------------------------
    # Access
    # ------------------------------------------------------------------

    def __getitem__(self, name: str) -> Any:
        """Access a connected server's tool by name."""
        if name not in self._connected:
            raise KeyError(f"MCP server {name!r} is not connected")
        return self._connected[name]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _match(patterns: list[str], names: set[str]) -> set[str]:
        """fnmatch *patterns* against *names* (same semantics as SkillRegistry)."""
        matched: set[str] = set()
        for pat in patterns:
            matched.update(n for n in names if fnmatch.fnmatch(n, pat))
        return matched

    def _attach(self, name: str, tool: Any) -> None:
        """Attach a connected tool to the agent, hidden from doc(self)."""
        self._validate_attach_name(name)
        self._connected[name] = tool
        if self._agent is not None:
            try:
                self._bind_agent_attr(name, tool)
            except BaseException:
                self._connected.pop(name, None)
                raise

    def _validate_attach_name(self, name: str) -> None:
        """Reject unusable names and collisions before any transport side effect."""
        attr = _to_attr_name(name)
        if not attr.isidentifier() or keyword.iskeyword(attr) or attr.startswith("_"):
            raise ValueError(f"MCP server name {name!r} does not map to a safe agent attribute")
        for connected_name in self._connected:
            if connected_name != name and _to_attr_name(connected_name) == attr:
                raise ValueError(
                    f"MCP server {name!r} conflicts with connected server {connected_name!r}"
                )
        if self._agent is not None:
            missing = object()
            existing = inspect.getattr_static(self._agent, attr, missing)
            if existing is not missing:
                raise ValueError(
                    f"MCP server {name!r} would overwrite existing agent attribute self.{attr}"
                )

    def _bind_agent_attr(self, name: str, tool: Any) -> None:
        attr = _to_attr_name(name)
        setattr(self._agent, attr, tool)
        try:
            spec(self._agent, attr, hidden=True)
        except Exception:
            logger.debug("Failed to hide MCP attr self.%s", attr, exc_info=True)

    def _detach(self, name: str) -> None:
        """Detach a connected tool from the agent."""
        attr = _to_attr_name(name)
        self._connected.pop(name, None)
        if self._agent is not None and hasattr(self._agent, attr):
            try:
                delattr(self._agent, attr)
            except AttributeError:
                pass

    def attach(self, agent: Any) -> None:
        """Wire into the agent and self-register the ``<mcp>`` context block."""
        previous_agent = self._agent
        self._agent = agent
        bound: list[str] = []
        try:
            for name, tool in self._connected.items():
                self._validate_attach_name(name)
                self._bind_agent_attr(name, tool)
                bound.append(name)
        except BaseException:
            for name in bound:
                attr = _to_attr_name(name)
                if hasattr(agent, attr):
                    delattr(agent, attr)
            self._agent = previous_agent
            raise
        cm = getattr(agent, "context_manager", None)
        if cm is not None and self.context_block:
            key, expr = self.context_block
            if key not in cm.protected_keys:
                cm.set_dynamic(key, expr)

    def detach(self) -> None:
        """Disconnect all servers and remove the context block."""
        for name in list(self._connected):
            self.deactivate([name])
            self._detach(name)
        agent = self._agent
        if agent is not None and self.context_block:
            cm = getattr(agent, "context_manager", None)
            key = self.context_block[0]
            if cm is not None and key in cm and key not in cm.protected_keys:
                cm.pop(key, None)
        self._agent = None
