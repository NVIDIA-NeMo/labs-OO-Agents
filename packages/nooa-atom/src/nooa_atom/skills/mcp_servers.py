# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCPServers: the MCP servers an Atom agent can connect to.

Held by the ``SkillManager``; the model reaches MCP servers through
``self.skills``. A server is *configured* (``.mcp.json``, the shared
``coding.mcp_servers`` block in settings.yaml, or registered in memory),
*connected* (its tools are known and ``self.<server>`` is set on the agent)
and *active* (listed as active in the ``<skills>`` block).

Configuration is discovery, not trust. Before any transport is created the
person must approve the fingerprint of the complete effective definition
with ``/mcp approve <name> <code>``; the agent cannot. Any change to the
URL, command, arguments, headers, environment or OAuth settings needs a new
approval. ``${VAR}`` placeholders are resolved from the host environment
only after the fingerprint matches.

HTTP servers that need OAuth sign in through the MCP SDK's provider
(``mcp_auth``). When a sign-in is needed, ``connect`` raises
``MCPSignInRequired`` with the link; the connection waits in the background
until the person pastes the address the browser ended on
(``/mcp auth <name> <address>``, which calls ``complete_sign_in``). Tokens
are stored per user, so later connections need no sign-in.
"""

from __future__ import annotations

import asyncio
import copy
import fnmatch
import inspect
import json
import keyword
import logging
import os
import textwrap
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal

from nooa.agentdoc import spec
from nooa.mcp import MCPManager
from nooa_atom.skills.mcp_auth import (
    DEFAULT_REDIRECT_URI,
    AuthorizedHTTPClient,
    FileTokenStorage,
    PastedSignIn,
    oauth_provider,
)
from nooa_atom.workspace.mcp_approval import (
    MCPApprovalRequest,
    MCPApprovalRequired,
    MCPApprovalStore,
    _safe_display,
    build_approval_request,
    redact_approved_environment,
    resolve_approved_environment,
)

logger = logging.getLogger(__name__)

_ALLOWED_CONNECT_KWARGS = {"tool_call_timeout"}
# How long /mcp auth waits for the token exchange and tool listing to finish.
_FINISH_SIGN_IN_SECONDS = 60.0

State = Literal["available", "active", "connected", "needs-auth", "failed"]


def _file_state(path: Path) -> tuple[int, int, int] | None:
    """What tells a changed file from an unchanged one: inode, mtime, size."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return (stat.st_ino, stat.st_mtime_ns, stat.st_size)


def attr_name(name: str) -> str:
    """The agent attribute of a server: hyphens and spaces become underscores."""
    return name.replace("-", "_").replace(" ", "_")


def tool_names(tool: Any) -> list[str]:
    """The tool methods of a connected server's tool object."""
    return sorted(getattr(type(tool), "_tool_method_names", ()) or ())


class MCPSignInRequired(RuntimeError):
    """A server needs the person to sign in; ``url`` is the sign-in link."""

    def __init__(self, name: str, url: str) -> None:
        self.name = name
        self.url = url
        super().__init__(
            f"MCP server {name!r} needs a sign-in. Open this link, sign in, then copy the "
            f"address the browser ends on (the page may fail to load) and run:\n"
            f"  /mcp auth {name} <address>\n\nSign-in link:\n{url}"
        )


class _PendingSignIn:
    def __init__(self, signin: PastedSignIn, task: asyncio.Task[Any], activate: bool) -> None:
        self.signin = signin
        self.task = task
        self.activate = activate


class MCPServers:
    """Configured MCP servers: approval, connection, sign-in and activation."""

    def __init__(
        self,
        mcp_file: Path | None = None,
        servers: dict[str, dict[str, Any]] | None = None,
        approval_path: Path | None = None,
        watch_settings: bool = False,
        project_dir: Path | None = None,
        token_path: Path | None = None,
    ) -> None:
        """Initialize with the configuration sources.

        Args:
            mcp_file: Path to a VS Code / Claude-style ``.mcp.json``.
            servers: Inline definitions (``coding.mcp_servers`` in settings.yaml).
            approval_path: The user approval store (tests override it).
            watch_settings: Reload the layered settings before lifecycle
                commands, so servers added to settings.yaml while the host
                runs are seen.
            project_dir: The workspace's ``.nooa`` directory.
            token_path: The OAuth token file (tests override it).
        """
        self.mcp_file = mcp_file
        self.project_dir = project_dir
        self.agent: Any = None
        self._notify: Callable[[str], None] | None = None
        self._servers: dict[str, dict[str, Any]] = copy.deepcopy(servers or {})
        self._watch_settings = watch_settings
        self._settings_server_names = set(self._servers)
        self._registered_server_names: set[str] = set()
        self._approval_store = MCPApprovalStore(approval_path)
        self._token_path = token_path
        self._discovered_cache: tuple[tuple[Any, ...], list[str]] | None = None
        self._connected: dict[str, Any] = {}
        self._activated: set[str] = set()
        self._connecting: set[str] = set()
        self._signins: dict[str, _PendingSignIn] = {}
        self._errors: dict[str, str] = {}

    def bind(self, agent: Any, notify: Callable[[str], None] | None = None) -> None:
        """Set the agent that gets ``self.<server>``, and where notices for it go."""
        self.agent = agent
        self._notify = notify

    def config_path(self) -> Path:
        """The workspace settings.yaml where servers are saved."""
        from nooa.paths import get_project_dir
        from nooa_atom.workspace.settings import SETTINGS_FILENAME

        if self.project_dir:
            return self.project_dir / SETTINGS_FILENAME
        return get_project_dir(SETTINGS_FILENAME)

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def refresh_settings(self) -> list[str]:
        """Reload inline definitions from the layered settings; return their names.

        Servers whose saved definition was removed or changed are
        disconnected. In-memory registrations stay until the settings hold
        the same name. Does nothing unless ``watch_settings`` was set.
        """
        if not self._watch_settings:
            return []

        from nooa.layered_config import load_layered_yaml
        from nooa_atom.workspace.settings import SETTINGS_ENV_VAR, SETTINGS_FILENAME

        data = load_layered_yaml(SETTINGS_FILENAME, SETTINGS_ENV_VAR, project_dir=self.project_dir)
        legacy = data.get("tui", {})
        tui = dict(legacy) if isinstance(legacy, dict) else {}
        coding = data.get("coding", {})
        if "mcp_servers" in tui and not (isinstance(coding, dict) and "mcp_servers" in coding):
            logger.warning("Reading legacy tui.mcp_servers; use coding.mcp_servers")
        if isinstance(coding, dict):
            tui.update(coding)
        raw_servers = tui.get("mcp_servers", {}) if isinstance(tui, dict) else {}
        if raw_servers is None:
            raw_servers = {}
        if not isinstance(raw_servers, dict):
            raise ValueError("coding.mcp_servers must be a mapping")

        fresh: dict[str, dict[str, Any]] = {}
        for name, definition in raw_servers.items():
            if not isinstance(name, str) or not isinstance(definition, dict):
                raise ValueError("each coding.mcp_servers entry must map a name to a mapping")
            fresh[name] = copy.deepcopy(definition)

        removed = self._settings_server_names - set(fresh)
        changed = {
            name
            for name, definition in fresh.items()
            if name in self._servers and self._servers[name] != definition
        }
        for name in sorted((removed | changed) & set(self._connected)):
            self._detach(name)
        for name in removed:
            if name not in self._registered_server_names:
                self._servers.pop(name, None)
        self._servers.update(fresh)
        # A definition first registered in memory becomes settings-owned as
        # soon as the saved settings contain it.
        self._registered_server_names.difference_update(fresh)
        self._settings_server_names = set(fresh)
        return sorted(fresh)

    def adopt(self, name: str, definition: dict[str, Any]) -> None:
        """Use ``definition`` for ``name`` as saved, without treating it as a change."""
        self._servers[name] = copy.deepcopy(definition)

    def discovered(self) -> list[str]:
        """All configured server names (``.mcp.json``, settings, registered).

        The ``<skills>`` block reads this every turn, so the answer is kept
        until the config file or the inline definitions change.
        """
        key = (
            _file_state(self.mcp_file or Path(".mcp.json")),
            json.dumps(self._servers, sort_keys=True, default=str),
        )
        if self._discovered_cache is None or self._discovered_cache[0] != key:
            names = sorted(MCPManager.list_servers(self.mcp_file, servers=self._servers))
            self._discovered_cache = (key, names)
        return list(self._discovered_cache[1])

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
    ) -> None:
        """Add an in-memory server definition (not saved; still needs approval)."""
        if name in self._connected or name in self._connecting or name in self._signins:
            raise RuntimeError(f"Disconnect MCP server {name!r} before changing its configuration")
        entry = {
            key: value
            for key, value in (
                ("url", url),
                ("command", command),
                ("args", args),
                ("env", env),
                ("transport", transport),
                ("headers", headers),
                ("oauth_client_id", oauth_client_id),
                ("oauth_redirect_uri", oauth_redirect_uri),
                ("oauth_scope", oauth_scope),
            )
            if value is not None
        }
        self._servers[name] = copy.deepcopy(entry)
        self._registered_server_names.add(name)

    # ------------------------------------------------------------------
    # Approval
    # ------------------------------------------------------------------

    def _approval_scope(self) -> str:
        return str((self.project_dir or Path.cwd()).resolve())

    def approval_request(self, name: str) -> MCPApprovalRequest:
        """The review material for the current definition of ``name``."""
        return build_approval_request(
            name, mcp_file=self.mcp_file, servers=self._servers, scope=self._approval_scope()
        )

    def is_approved(self, name: str) -> bool:
        return self._approval_store.is_approved(self.approval_request(name))

    def approve(self, name: str, confirmation: str) -> MCPApprovalRequest:
        """Approve the current definition; the code comes from its review text."""
        request = self.approval_request(name)
        if not request.accepts_confirmation(confirmation):
            raise ValueError(
                "Approval code does not match the current MCP configuration. "
                "Run `/mcp approve " + name + "` to review it again."
            )
        self._approval_store.approve(request)
        return request

    def revoke(self, name: str) -> bool:
        """Revoke ``name`` in this workspace only; other workspaces keep theirs."""
        try:
            fingerprint: str | None = self.approval_request(name).fingerprint
        except ValueError:
            fingerprint = None  # an invalid config; its scope still matches
        return self._approval_store.revoke_server(
            name, scope=self._approval_scope(), fingerprint=fingerprint
        )

    def _approved_config(
        self, name: str
    ) -> tuple[MCPApprovalRequest, dict[str, Any], dict[str, str]]:
        """The approved definition with its placeholders resolved, and the values used."""
        request = self.approval_request(name)
        if not self._approval_store.is_approved(request):
            raise MCPApprovalRequired(request)
        environment = {
            variable: value
            for variable in request.variables
            if (value := os.environ.get(variable)) is not None
        }
        config = resolve_approved_environment(request, environment)
        config["transport"] = request.transport
        return request, config, environment

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def connected(self) -> list[str]:
        """Connected server names (active or not)."""
        return sorted(self._connected)

    def activated(self) -> list[str]:
        """Active server names."""
        return sorted(self._activated)

    def state(self, name: str) -> State:
        if name in self._activated:
            return "active"
        if name in self._connected:
            return "connected"
        if name in self._signins:
            return "needs-auth"
        if name in self._errors:
            return "failed"
        return "available"

    def error(self, name: str) -> str | None:
        """Why the last connection of ``name`` failed, if it did."""
        return self._errors.get(name)

    def sign_in_url(self, name: str) -> str | None:
        """The link of a sign-in waiting for the person, if any."""
        pending = self._signins.get(name)
        return pending.signin.url if pending is not None else None

    def tool(self, name: str) -> Any:
        """The connected tool object of ``name``, or ``None``."""
        return self._connected.get(name)

    def tool_names(self, name: str) -> list[str]:
        tool = self._connected.get(name)
        return tool_names(tool) if tool is not None else []

    def description(self, name: str) -> str:
        """One line: transport and target, and the tools once connected."""
        try:
            request = self.approval_request(name)
            text = f"{request.transport} MCP server {request.target}"
        except ValueError:
            text = "MCP server with an invalid configuration"
        names = self.tool_names(name)
        if names:
            text += "; tools: " + ", ".join(names)
        return _safe_display(text)

    # ------------------------------------------------------------------
    # Connecting
    # ------------------------------------------------------------------

    async def connect(
        self, patterns: list[str], *, activate: bool = True, **kwargs: Any
    ) -> list[str]:
        """Connect configured servers matching ``patterns``; return the names connected.

        Each connected server is set on the agent as ``self.<name>`` (hyphens
        and spaces become underscores) and, unless ``activate`` is false,
        made active. ``tool_call_timeout`` is the only extra keyword allowed;
        everything else must be in the approved definition.

        Raises ``MCPApprovalRequired`` for a definition that is not approved
        and ``MCPSignInRequired`` when the person must sign in; that
        connection then waits for ``complete_sign_in``.
        """
        unexpected = sorted(set(kwargs) - _ALLOWED_CONNECT_KWARGS)
        if unexpected:
            raise TypeError(
                "MCP connection settings must be part of the approved server definition: "
                + ", ".join(unexpected)
            )
        self.refresh_settings()
        newly: list[str] = []
        for name in sorted(self._match(patterns, set(self.discovered()))):
            if name in self._connected:
                continue
            if name in self._connecting or name in self._signins:
                raise RuntimeError(
                    f"MCP server {name!r} is already connecting"
                    + (" and waits for the sign-in (/mcp auth)" if name in self._signins else "")
                )
            self._validate_attach_name(name)
            request, config, environment = self._approved_config(name)
            self._errors.pop(name, None)
            self._connecting.add(name)
            try:
                tool = await self._open(name, request, config, environment, activate, **kwargs)
            finally:
                self._connecting.discard(name)
            self._finish(name, request, tool, activate)
            newly.append(name)
        return newly

    async def _open(
        self,
        name: str,
        request: MCPApprovalRequest,
        config: dict[str, Any],
        environment: dict[str, str],
        activate: bool,
        tool_call_timeout: timedelta = timedelta(seconds=60),
    ) -> Any:
        """Start the connection; return its tool, or raise ``MCPSignInRequired``."""
        signin = PastedSignIn()
        opening = asyncio.create_task(self._discover(name, config, signin, tool_call_timeout))
        waiting = asyncio.create_task(signin.url_ready.wait())
        try:
            await asyncio.wait({opening, waiting}, return_when=asyncio.FIRST_COMPLETED)
        except asyncio.CancelledError:
            opening.cancel()
            raise
        finally:
            waiting.cancel()
        if not opening.done():
            self._signins[name] = _PendingSignIn(signin, opening, activate)
            opening.add_done_callback(
                lambda task: self._signed_in(name, request, environment, task)
            )
            raise MCPSignInRequired(name, signin.url or "")
        try:
            return opening.result()
        except Exception as exc:
            message = redact_approved_environment(request, exc, environment)
            self._errors[name] = message
            raise RuntimeError(f"MCP server {name!r} connection failed: {message}") from None

    async def _discover(
        self,
        name: str,
        config: dict[str, Any],
        signin: PastedSignIn,
        tool_call_timeout: timedelta,
    ) -> Any:
        """Connect once to sign in if needed and list the tools; return the tool object."""
        transport = config["transport"]
        if transport == "stdio":
            return await MCPManager.create_stdio_server(
                name,
                command=config["command"],
                args=config.get("args"),
                env=config.get("env"),
                tool_call_timeout=tool_call_timeout,
            )
        url = config["url"]
        headers = dict(config.get("headers") or {})
        storage = FileTokenStorage(
            url,
            self._token_path,
            client_id=config.get("oauth_client_id"),
            redirect_uri=config.get("oauth_redirect_uri") or DEFAULT_REDIRECT_URI,
        )
        client = AuthorizedHTTPClient(
            url,
            provider=oauth_provider(signin, storage, scope=config.get("oauth_scope")),
            headers=headers,
            transport=transport,
            tool_call_timeout=tool_call_timeout,
        )
        # The first connection signs in when the server asks for it.
        async with client.connect_to_server():
            pass
        tokens = await storage.get_tokens()
        if tokens is not None and tokens.access_token:
            headers["Authorization"] = f"Bearer {tokens.access_token}"
        listed = await MCPManager.create_url_server(
            name, url, headers=headers, transport=transport, tool_call_timeout=tool_call_timeout
        )
        # Same generated class, but its calls go through the OAuth provider,
        # which refreshes tokens and signs in again when the server asks.
        return type(listed)(client, name)

    def _signed_in(
        self,
        name: str,
        request: MCPApprovalRequest,
        environment: dict[str, str],
        task: asyncio.Task[Any],
    ) -> None:
        """A connection that waited for a sign-in ended: attach it or record why not."""
        pending = self._signins.pop(name, None)
        if task.cancelled():
            return
        error = task.exception()
        if error is None:
            try:
                self._finish(name, request, task.result(), pending.activate if pending else True)
            except Exception as exc:
                error = exc
            else:
                tools = ", ".join(self.tool_names(name)) or "none"
                self._tell(
                    f"MCP server {name!r} is connected after the sign-in. Its tools are "
                    f"methods of self.{attr_name(name)}: {tools}."
                )
                return
        message = redact_approved_environment(request, error, environment)
        self._errors[name] = message
        self._tell(f"MCP server {name!r} did not connect after the sign-in: {message}")

    def _finish(self, name: str, request: MCPApprovalRequest, tool: Any, activate: bool) -> None:
        """Attach a connected tool if its definition and approval did not change meanwhile."""
        self.refresh_settings()
        try:
            current = self.approval_request(name)
        except ValueError:
            current = None
        if (
            current is None
            or current.fingerprint != request.fingerprint
            or not self._approval_store.is_approved(current)
        ):
            raise RuntimeError(
                f"MCP server {name!r} configuration or approval changed while connecting. "
                "Review and approve the current configuration before connecting again."
            )
        self._attach(name, tool)
        if activate:
            self._activated.add(name)

    async def complete_sign_in(self, name: str, pasted: str) -> str:
        """Finish a waiting sign-in with the address the browser ended on.

        Raises ``ValueError`` when no sign-in waits or the address has no
        code; the sign-in keeps waiting in the latter case.
        """
        pending = self._signins.get(name)
        if pending is None:
            raise ValueError(
                f"No sign-in is waiting for MCP server {name!r}; connect it first "
                f"(/mcp approve {name})."
            )
        pending.signin.complete(pasted)
        try:
            await asyncio.wait_for(asyncio.shield(pending.task), _FINISH_SIGN_IN_SECONDS)
        except Exception:
            pass
        if name in self._connected:
            tools = ", ".join(self.tool_names(name)) or "none"
            return f"Signed in; MCP server {name!r} is connected (tools: {tools})."
        raise RuntimeError(self._errors.get(name) or f"MCP server {name!r} did not connect")

    async def disconnect(self, patterns: list[str]) -> list[str]:
        """Detach connected servers matching ``patterns``; stop waiting sign-ins."""
        names = set(self._connected) | set(self._signins)
        matched = sorted(self._match(patterns, names))
        for name in matched:
            pending = self._signins.pop(name, None)
            if pending is not None:
                pending.task.cancel()
            self._detach(name)
        return matched

    def activate(self, patterns: list[str]) -> list[str]:
        """Make connected servers matching ``patterns`` active; return them."""
        matched = self._match(patterns, set(self._connected))
        self._activated.update(matched)
        return sorted(matched)

    async def deactivate(self, patterns: list[str]) -> list[str]:
        """Make active servers inactive; they stay connected. Return them."""
        matched = self._match(patterns, self._activated)
        self._activated.difference_update(matched)
        return sorted(matched)

    async def aclose(self) -> None:
        """Stop waiting sign-ins and detach every server."""
        for pending in self._signins.values():
            pending.task.cancel()
        self._signins.clear()
        for name in list(self._connected):
            self._detach(name)

    # ------------------------------------------------------------------
    # Host display
    # ------------------------------------------------------------------

    def details(self) -> str:
        """Every configured server with its state and endpoint, for ``/mcp status``."""
        configured = self.discovered()
        if not configured:
            return "No MCP servers configured."
        rows = []
        for name in configured:
            try:
                request = self.approval_request(name)
                target = f"{request.target} ({request.transport})"
                if not self._approval_store.is_approved(request):
                    target += " [approval required]"
            except ValueError:
                target = "(invalid MCP configuration)"
            state = self.state(name)
            names = self.tool_names(name)
            tools = f"; tools: {', '.join(names)}" if names else ""
            rows.append(f"  {_safe_display(name):24s} {state:10s} {target}{tools}")
        return "\n".join(
            ["MCP servers (name, state, endpoint):"]
            + [textwrap.shorten(row, 400, placeholder=" ...") for row in rows]
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tell(self, text: str) -> None:
        if self._notify is not None:
            self._notify(text)

    @staticmethod
    def _match(patterns: list[str], names: set[str]) -> set[str]:
        matched: set[str] = set()
        for pattern in patterns:
            matched.update(name for name in names if fnmatch.fnmatch(name, pattern))
        return matched

    def _attach(self, name: str, tool: Any) -> None:
        """Set a connected tool on the agent, hidden from ``doc(self)``."""
        self._validate_attach_name(name)
        self._connected[name] = tool
        if self.agent is None:
            return
        attr = attr_name(name)
        setattr(self.agent, attr, tool)
        try:
            spec(self.agent, attr, hidden=True)
        except Exception:
            logger.debug("Could not hide MCP attribute self.%s", attr, exc_info=True)

    def _validate_attach_name(self, name: str) -> None:
        """Refuse unusable names and collisions before any transport starts."""
        attr = attr_name(name)
        if not attr.isidentifier() or keyword.iskeyword(attr) or attr.startswith("_"):
            raise ValueError(f"MCP server name {name!r} does not map to a safe agent attribute")
        for connected_name in self._connected:
            if connected_name != name and attr_name(connected_name) == attr:
                raise ValueError(
                    f"MCP server {name!r} conflicts with connected server {connected_name!r}"
                )
        if self.agent is not None and name not in self._connected:
            missing = object()
            if inspect.getattr_static(self.agent, attr, missing) is not missing:
                raise ValueError(
                    f"MCP server {name!r} would overwrite existing agent attribute self.{attr}"
                )

    def _detach(self, name: str) -> None:
        self._activated.discard(name)
        self._connected.pop(name, None)
        attr = attr_name(name)
        if self.agent is not None and attr in vars(self.agent):
            delattr(self.agent, attr)
