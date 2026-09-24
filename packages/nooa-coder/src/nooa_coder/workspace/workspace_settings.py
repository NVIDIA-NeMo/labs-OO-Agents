# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-facing workspace preferences for NOOA interactive hosts."""

import copy
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from nooa.skill import Skill

_ENV_PLACEHOLDER_ONLY = re.compile(r"^\$\{[A-Za-z_][A-Za-z0-9_]*\}$")
# Command-line flags whose value is a credential, e.g. --token X or --api-key=X.
_SECRET_FLAG = re.compile(
    r"^--?(?:[a-z0-9]+[-_])*(?:token|api[-_]?key|apikey|password|passwd|secret|auth)$",
    re.IGNORECASE,
)
# URL query parameters whose value is a credential, e.g. ?api_key=X or ?sig=X.
_SECRET_QUERY_KEY = re.compile(
    r"(?:token|key|secret|password|passwd|auth|credential|signature|^sig$)", re.IGNORECASE
)


def _is_placeholder(value: str) -> bool:
    return bool(_ENV_PLACEHOLDER_ONLY.match(value))


def _literal_credentials(definition: dict[str, Any]) -> list[str]:
    """Name each place in an MCP definition that holds a literal credential.

    Only a value that is entirely one ``${VAR}`` placeholder (resolved from the
    environment at connect time) may be written to the committed workspace
    settings file.
    """
    found = []
    for field_name in ("headers", "env"):
        for key, value in (definition.get(field_name) or {}).items():
            if isinstance(value, str) and not _is_placeholder(value):
                found.append(f"{field_name}[{key!r}]")
    url = definition.get("url")
    if isinstance(url, str):
        parts = urlsplit(url)
        if parts.password is not None and not _is_placeholder(parts.password):
            found.append("url password")
        if parts.username and parts.password is None and not _is_placeholder(parts.username):
            # A lone userinfo value (https://TOKEN@host) is a bearer credential.
            found.append("url userinfo")
        for key, value in parse_qsl(parts.query, keep_blank_values=True):
            if value and _SECRET_QUERY_KEY.search(key) and not _is_placeholder(value):
                found.append(f"url query {key!r}")
    args = definition.get("args") or []
    for index, arg in enumerate(args):
        if not isinstance(arg, str):
            continue
        flag, separator, inline = arg.partition("=")
        if not _SECRET_FLAG.match(flag):
            continue
        if separator:
            value = inline
        elif index + 1 < len(args) and isinstance(args[index + 1], str):
            value = args[index + 1]
        else:
            continue
        if not _is_placeholder(value):
            found.append(f"args value for {flag}")
    return found


class WorkspaceSettings(Skill):
    """Manage saved defaults for interactive sessions in this workspace.

    Preferences belong to this workspace's .nooa/settings.yaml. They apply to
    fresh agents in either client; other live agents retain their current state.
    Named operations cover skills, MCP startup, and model defaults.
    Ordinary self.skills.load/activate remains session-local.
    """

    def __init__(self, options: Any):
        super().__init__()
        self._workspace = Path(options.working_dir).expanduser().resolve()
        self._agent_spec = options.agent_spec
        self._legacy_agent = options.legacy_agent

    def _options(self):
        from .options import SessionOptions

        options = SessionOptions.load(self._workspace)
        options.agent_spec = self._agent_spec
        options.legacy_agent = self._legacy_agent
        return options

    async def remember_skill(self, skill_id: str, directory: str | None = None) -> str:
        """Activate a skill here and remember it for future workspace sessions.

        Use an exact ID from self.skills.discovered(). For a skill from a local
        repository, supply its directory so fresh agents can discover it too;
        relative paths resolve against this workspace. An already active skill
        can still be remembered. Installation is separate. A save failure raises
        an error describing any changes already applied to the live session.
        """
        if directory is not None:
            await self._run_control("skills", ["add", directory])
        await self._run_control("skills", ["activate", skill_id])
        return f"Remembered `{skill_id}` in {self._workspace / '.nooa' / 'settings.yaml'}."

    async def forget_skill(self, skill_id: str) -> str:
        """Deactivate a skill here and disable its automatic workspace activation.

        Saves the same preference as /skills deactivate. Other live sessions
        keep their state; source directories, installed packages, and saved
        session data are retained. A save failure raises an error.
        """
        await self._run_control("skills", ["deactivate", skill_id])
        return f"Forgot `{skill_id}` in {self._workspace / '.nooa' / 'settings.yaml'}."

    def set_default_model(self, model: str) -> str:
        """Save a NOOA model alias or provider/model ID for future sessions.

        Does not switch the running agent or alter a resumed session's model.
        Explicit launch overrides still take precedence; the current ACP CLI
        requires --model or NOOA_MODEL, so it continues to use that override.
        Model availability and
        credentials are checked when the model is used, not by this operation.
        """
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty alias or provider/model ID")
        path = self._save({("coding", "default_model"): model.strip()})
        return f"Saved default model in {path}; the running model is unchanged."

    def remember_mcp(self, name: str, auto_connect: bool = True) -> str:
        """Save a registered NOOA MCP definition and its startup preference.

        First register the server with self.mcp.register(...), or use an existing
        NOOA-configured server. Use environment placeholders for credentials.
        This saves the definition without connecting, approving, or authenticating
        it. Existing exact-definition approvals still apply when connecting.
        Pool-supplied servers are not automatically copied into NOOA settings.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a nonempty MCP server name")
        if not isinstance(auto_connect, bool):
            raise ValueError("auto_connect must be a boolean")
        registry = self._agent.mcp
        registry.refresh_settings()
        # Reuse the registry's exact, unresolved definition. The request only
        # describes configuration; it neither grants approval nor reads secrets.
        definition = registry._approval_request(name).config
        # This definition is about to be written to the workspace's committed
        # .nooa/settings.yaml, not the user-level approval store -- a literal
        # secret here would land in version control.
        literals = _literal_credentials(definition)
        if literals:
            raise ValueError(
                f"MCP server {name!r} {', '.join(literals)} must be a ${{VAR}} "
                "placeholder, not a literal value, to be saved in the workspace "
                "settings file"
            )
        names = [item for item in self._project_auto_connect() if item != name]
        if auto_connect:
            names.append(name)
        path = self._save(
            {
                ("coding", "mcp_servers", name): definition,
                ("coding", "mcp_auto_connect"): names,
            }
        )
        if name in registry._servers:
            # The saved form is normalized (e.g. a bare url gains its transport).
            # Adopt it first so the refresh does not see a "changed" definition
            # and disconnect a live server.
            registry._servers[name] = copy.deepcopy(definition)
        registry.refresh_settings()
        return f"Saved MCP server `{name}` in {path}; auto-connect={auto_connect}. Connection approvals are unchanged."

    def forget_mcp(self, name: str) -> str:
        """Disable startup connection and remove this workspace's MCP definition.

        A null entry masks an inherited definition. An existing connection is
        managed separately through self.mcp.disconnect; this does not revoke
        approvals or remove credential caches. Client-supplied servers are outside
        this preference and may still be supplied by that client on startup.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a nonempty MCP server name")
        path = self._save(
            {
                ("coding", "mcp_servers", name): None,
                ("coding", "mcp_auto_connect"): [
                    n for n in self._project_auto_connect() if n != name
                ],
            }
        )
        return f"Forgot MCP startup preference `{name}` in {path}."

    def status(self) -> dict[str, Any]:
        """Inspect effective saved defaults and current state without credentials.

        Saved values include layered configuration; explicit launch overrides may
        supersede them. MCP output contains names only, never connection secrets.
        """
        options = self._options()
        return {
            "workspace": str(self._workspace),
            "settings_file": str(self._workspace / ".nooa" / "settings.yaml"),
            "saved": {
                "active_skills": options.active_skills,
                "inactive_skills": options.inactive_skills,
                "skills_directories": [str(p) for p in options.skills_dirs],
                "default_model": options.default_model,
                "mcp_servers": sorted(options.mcp_servers),
                "mcp_auto_connect": options.mcp_auto_connect,
            },
            "current": {
                "active_skills": self._agent.skills.activated(),
                "model": getattr(getattr(self._agent, "_llm", None), "model", None),
                "connected_mcp": self._agent.mcp.connected(),
                "active_mcp_skills": [
                    name for name in self._agent.skills.activated() if name.startswith("mcp.")
                ],
            },
        }

    def _project_auto_connect(self) -> list[str]:
        """Read mcp_auto_connect from the project-scope settings file only.

        _save() always writes project scope. Basing an add/remove on the
        fully layered options.mcp_auto_connect (user+project merged) would
        copy any server name from a user's personal settings into the
        shared, committed project file -- the same cross-scope leak
        test_skills_control_settings_scope.py guards for skills.
        """
        import yaml

        from .settings import resolve_behavior_settings, settings_path

        path = settings_path("project", workspace=self._workspace)
        if not path.exists():
            return []
        try:
            data = yaml.safe_load(path.read_text())
        except (OSError, UnicodeError, yaml.YAMLError):
            return []
        if not isinstance(data, dict):
            return []
        return list(resolve_behavior_settings(data).get("mcp_auto_connect", []))

    def _save(self, updates: dict[tuple[str, ...], Any]) -> Path:
        from .settings import write_settings_updates

        path, _ = write_settings_updates(updates, workspace=self._workspace)
        return path

    async def _run_control(self, name: str, args: list[str]) -> str:
        from .controls import CONTROL_TYPES, ControlMessage

        control = CONTROL_TYPES[name](
            self._agent,
            self._options(),
            workspace=self._workspace,
            command_registry=getattr(self._agent, "_command_registry", None),
        )
        result = await control.run(args)
        if not result.success or any(
            isinstance(output, ControlMessage) and output.style == "warning"
            for output in result.outputs
        ):
            raise RuntimeError(str(result))
        return str(result)
