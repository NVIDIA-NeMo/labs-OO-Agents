# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""User-owned approvals for TUI MCP server configurations.

Repository configuration is not a trust grant.  Before the TUI connects to an
MCP server, this module fingerprints the complete effective server definition
and requires that exact fingerprint to be present in a user-level approval
store.  Environment placeholders are resolved only after that check succeeds.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import re
import shlex
import stat
import tempfile
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_plus, urlsplit, urlunsplit

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_APPROVALS_FILENAME = "mcp_approvals.json"
_STORE_VERSION = 1
_CONFIRMATION_LENGTH = 16
_TARGET_PREVIEW_LENGTH = 240
_SUPPORTED_CONFIG_FIELDS = {
    "args",
    "command",
    "env",
    "headers",
    "oauth_client_id",
    "oauth_manual",
    "oauth_open_browser",
    "oauth_redirect_uri",
    "oauth_scope",
    "transport",
    "url",
}


def _safe_display(value: Any) -> str:
    """Escape terminal control characters in untrusted config-derived text."""
    out: list[str] = []
    for character in str(value):
        codepoint = ord(character)
        if codepoint < 32 or 127 <= codepoint <= 159:
            out.append(f"\\x{codepoint:02x}")
        elif unicodedata.category(character) == "Cf":
            escape = f"\\u{codepoint:04x}" if codepoint <= 0xFFFF else f"\\U{codepoint:08x}"
            out.append(escape)
        else:
            out.append(character)
    return "".join(out)


def _safe_preview(value: Any, limit: int = _TARGET_PREVIEW_LENGTH) -> str:
    """Escape untrusted text and bound how much a config can render."""
    rendered = _safe_display(value)
    if len(rendered) <= limit:
        return rendered
    return rendered[:limit] + f"… ({len(rendered) - limit} more characters)"


def _load_server_config(
    server_name: str,
    *,
    mcp_file: Path | None,
    servers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Load one effective config using the same whole-entry precedence as core."""
    configured: dict[str, Any] = {}
    path = mcp_file or Path(".mcp.json")
    if path.exists():
        try:
            root = json.loads(path.read_text())
            file_servers = root.get("mcpServers", {}) if isinstance(root, dict) else {}
            if isinstance(file_servers, dict):
                configured.update(file_servers)
        except (json.JSONDecodeError, OSError):
            # Match MCPManager: a missing or malformed compatibility file is
            # treated as empty, while inline TUI servers remain usable.
            pass
    configured.update(servers)
    config = configured.get(server_name)
    if not isinstance(config, dict):
        raise ValueError(f"MCP server {server_name!r} has no valid configuration mapping")
    return copy.deepcopy(config)


def _placeholder_bindings(value: Any, path: str = "") -> list[tuple[str, str]]:
    """Return ``(variable, config-path)`` pairs without reading the environment."""
    found: list[tuple[str, str]] = []
    if isinstance(value, str):
        found.extend(
            (match.group(1), path or "<value>") for match in _ENV_VAR_PATTERN.finditer(value)
        )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child = f"{path}[{index}]" if path else f"[{index}]"
            found.extend(_placeholder_bindings(item, child))
    elif isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            found.extend(_placeholder_bindings(item, child))
    return sorted(set(found))


def _fingerprint(server_name: str, config: dict[str, Any]) -> str:
    """Hash the server name and complete literal config deterministically."""
    try:
        canonical = json.dumps(
            {"server": server_name, "config": config},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"MCP server {server_name!r} must contain only JSON-compatible values"
        ) from exc
    return hashlib.sha256(canonical.encode()).hexdigest()


def _safe_http_target(raw_url: Any) -> str:
    """Show the destination without echoing userinfo, query values, or fragments."""
    if not isinstance(raw_url, str) or not raw_url:
        return "(missing URL)"
    try:
        parsed = urlsplit(raw_url)
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        port = f":{parsed.port}" if parsed.port is not None else ""
        netloc = hostname + port
        query = "<redacted>" if parsed.query else ""
        return _safe_display(urlunsplit((parsed.scheme, netloc, parsed.path, query, "")))
    except (TypeError, ValueError):
        return "(configured URL; inspect the config file)"


@dataclass(frozen=True)
class MCPApprovalRequest:
    """Review material bound to one exact MCP server definition."""

    server_name: str
    fingerprint: str
    transport: str
    target: str
    bindings: tuple[tuple[str, str], ...]
    config: dict[str, Any] = field(repr=False, compare=False)

    @property
    def confirmation(self) -> str:
        """Short fingerprint a user must echo to approve this definition."""
        return self.fingerprint[:_CONFIRMATION_LENGTH]

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(sorted({variable for variable, _path in self.bindings}))

    @property
    def approval_command(self) -> str:
        """Exact user-owned command that approves this configuration."""
        quoted_name = _safe_display(shlex.quote(self.server_name))
        return f"/mcp approve {quoted_name} {self.confirmation}"

    def accepts_confirmation(self, value: str) -> bool:
        return hmac.compare_digest(value, self.confirmation) or hmac.compare_digest(
            value, self.fingerprint
        )

    def review_text(self) -> str:
        """Render a secret-safe review and a second-step confirmation command."""
        lines = [
            f"Approval required for MCP server {_safe_display(self.server_name)!r}.",
            "",
            f"Transport: {_safe_display(self.transport)}",
            f"Target: {self.target}",
            f"Config fingerprint: sha256:{self.fingerprint}",
        ]
        if self.transport == "stdio":
            lines.extend(
                ("", "Warning: approval allows this configuration to execute a local process.")
            )
        headers = self.config.get("headers") or {}
        if headers:
            lines.append(f"Header names: {_safe_preview(', '.join(sorted(headers)))}")
        child_environment = self.config.get("env") or {}
        if child_environment:
            lines.append(
                "Child environment keys: " + _safe_preview(", ".join(sorted(child_environment)))
            )
        if self.config.get("oauth_redirect_uri"):
            lines.append("OAuth redirect: " + _safe_http_target(self.config["oauth_redirect_uri"]))
        if self.config.get("oauth_scope"):
            lines.append("OAuth scope: " + _safe_preview(self.config["oauth_scope"]))
        if self.config.get("oauth_client_id"):
            lines.append("OAuth client ID: " + _safe_preview(self.config["oauth_client_id"]))
        if self.bindings:
            lines.extend(("", "Host environment values that will be copied after approval:"))
            for variable, location in self.bindings:
                lines.append(f"  {variable} -> {_safe_display(location)}")
        else:
            lines.extend(("", "No host environment placeholders were found."))
        lines.extend(
            (
                "",
                "Approval is user-level and applies only to this exact configuration. Any change",
                "to its URL, command, arguments, headers, environment, or OAuth settings requires approval again.",
                "",
                "Review the source config, then approve and connect with:",
                f"  {self.approval_command}",
            )
        )
        return "\n".join(lines)


def build_approval_request(
    server_name: str,
    *,
    mcp_file: Path | None,
    servers: dict[str, dict[str, Any]],
) -> MCPApprovalRequest:
    """Build review material for the current effective server definition."""
    config = _load_server_config(server_name, mcp_file=mcp_file, servers=servers)
    if not all(isinstance(field_name, str) for field_name in config):
        raise ValueError(f"MCP server {server_name!r} field names must be strings")
    unsupported = sorted(set(config) - _SUPPORTED_CONFIG_FIELDS)
    if unsupported:
        names = ", ".join(unsupported)
        raise ValueError(f"MCP server {server_name!r} has unsupported field(s): {names}")
    if not config.get("transport") and config.get("url"):
        # TUI configs have always documented URL entries as HTTP servers. Make
        # that effective value explicit before fingerprinting and before core
        # sees the definition, instead of falling through to core's stdio
        # default and failing with a missing command.
        config["transport"] = "streamable-http"
    transport = str(config.get("transport") or "stdio")
    if transport not in ("stdio", "sse", "streamable-http"):
        raise ValueError(f"Unsupported MCP transport {transport!r}")
    if transport == "stdio" and (
        not isinstance(config.get("command"), str) or not config["command"].strip()
    ):
        raise ValueError(f"MCP stdio server {server_name!r} requires a string command")
    if transport in ("sse", "streamable-http") and (
        not isinstance(config.get("url"), str) or not config["url"].strip()
    ):
        raise ValueError(f"MCP HTTP server {server_name!r} requires a string URL")
    args = config.get("args")
    if args is not None and (
        not isinstance(args, list) or not all(isinstance(item, str) for item in args)
    ):
        raise ValueError(f"MCP server {server_name!r} args must be a list of strings")
    for field_name in ("env", "headers"):
        values = config.get(field_name)
        if values is not None and (
            not isinstance(values, dict)
            or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in values.items()
            )
        ):
            raise ValueError(f"MCP server {server_name!r} {field_name} must map strings to strings")
    for field_name in ("oauth_client_id", "oauth_redirect_uri", "oauth_scope"):
        value = config.get(field_name)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"MCP server {server_name!r} {field_name} must be a string")
    for field_name in ("oauth_open_browser", "oauth_manual"):
        value = config.get(field_name)
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"MCP server {server_name!r} {field_name} must be a boolean")
    if transport in ("sse", "streamable-http"):
        target = _safe_http_target(config.get("url"))
    else:
        command = config.get("command")
        args = config.get("args")
        invocation = shlex.join([command, *(args or [])])
        target = _safe_preview(invocation)
    return MCPApprovalRequest(
        server_name=server_name,
        fingerprint=_fingerprint(server_name, config),
        transport=transport,
        target=target,
        bindings=tuple(_placeholder_bindings(config)),
        config=config,
    )


def resolve_approved_environment(
    request: MCPApprovalRequest, environment: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """Resolve placeholders in an already-approved request, failing atomically."""
    source = os.environ if environment is None else environment
    missing = sorted(variable for variable in request.variables if variable not in source)
    if missing:
        names = ", ".join(missing)
        raise ValueError(
            f"Approved MCP server {request.server_name!r} requires unset environment "
            f"variable(s): {names}"
        )

    def _resolve(value: Any) -> Any:
        if isinstance(value, str):
            return _ENV_VAR_PATTERN.sub(lambda match: source[match.group(1)], value)
        if isinstance(value, list):
            return [_resolve(item) for item in value]
        if isinstance(value, dict):
            return {key: _resolve(item) for key, item in value.items()}
        return value

    return _resolve(request.config)


def redact_approved_environment(
    request: MCPApprovalRequest,
    value: Any,
    environment: Mapping[str, str] | None = None,
) -> str:
    """Remove approved environment values from an error before displaying it."""
    rendered = str(value)
    replacements: list[tuple[str, str]] = []
    source = os.environ if environment is None else environment
    for variable in request.variables:
        secret = source.get(variable)
        if not secret:
            continue
        marker = f"${{{variable}}}"
        candidates = {secret, quote(secret, safe=""), quote_plus(secret, safe="")}
        replacements.extend((candidate, marker) for candidate in candidates if candidate)
    for candidate, marker in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        rendered = rendered.replace(candidate, marker)
    return _safe_display(rendered)


class MCPApprovalRequired(RuntimeError):
    """Raised before transport creation when a definition is not approved."""

    def __init__(self, request: MCPApprovalRequest) -> None:
        self.request = request
        super().__init__(request.review_text())


class MCPApprovalStore:
    """A fail-closed, user-level store containing config fingerprints only."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            from nooa.paths import get_user_dir

            path = get_user_dir(_APPROVALS_FILENAME)
        self.path = path

    def _load(self) -> dict[str, Any]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {"version": _STORE_VERSION, "approvals": {}}
        if not isinstance(data, dict) or data.get("version") != _STORE_VERSION:
            return {"version": _STORE_VERSION, "approvals": {}}
        approvals = data.get("approvals")
        if not isinstance(approvals, dict):
            return {"version": _STORE_VERSION, "approvals": {}}
        return {"version": _STORE_VERSION, "approvals": approvals}

    def is_approved(self, request: MCPApprovalRequest) -> bool:
        return request.fingerprint in self._load()["approvals"]

    def approve(self, request: MCPApprovalRequest) -> None:
        data = self._load()
        data["approvals"][request.fingerprint] = {
            "server": request.server_name,
            "variables": list(request.variables),
            "approved_at": datetime.now(UTC).isoformat(),
        }
        self._write(data)

    def revoke_server(self, server_name: str) -> bool:
        data = self._load()
        approvals = data["approvals"]
        remove = [
            fingerprint
            for fingerprint, record in approvals.items()
            if isinstance(record, dict) and record.get("server") == server_name
        ]
        for fingerprint in remove:
            approvals.pop(fingerprint, None)
        if remove:
            self._write(data)
        return bool(remove)

    def _write(self, data: dict[str, Any]) -> None:
        """Atomically replace the store and keep it readable only by the user."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_temp = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        temp = Path(raw_temp)
        try:
            os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
            self.path.chmod(stat.S_IRUSR | stat.S_IWUSR)
        except BaseException:
            try:
                os.close(fd)
            except OSError:
                pass
            temp.unlink(missing_ok=True)
            raise


__all__ = [
    "MCPApprovalRequest",
    "MCPApprovalRequired",
    "MCPApprovalStore",
    "build_approval_request",
    "redact_approved_environment",
    "resolve_approved_environment",
]
