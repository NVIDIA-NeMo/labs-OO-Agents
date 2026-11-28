# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-facing workspace preferences for NOOA interactive hosts."""

import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from nooa.skill import Skill, slash_command

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


def _describe(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "unset"
    if summary["kind"] == "append":
        return f"append every {summary['every_n_turns']} turns"
    return summary["kind"]


def _policy_summary(policy: Any) -> dict[str, Any] | None:
    """``{"kind": ..., "every_n_turns": ...}`` for a live-context policy, ``None`` when unset."""
    if policy is None:
        return None
    return policy.model_dump()


class WorkspaceSettings(Skill):
    """Manage saved defaults for interactive sessions in this workspace.

    Preferences belong to this workspace's .nooa/settings.yaml. They apply to
    fresh agents in either client; other live agents retain their current state.
    Named operations cover skills, MCP startup, and model defaults.
    Ordinary self.skills.activate remains session-local.
    """

    def __init__(self, options: Any):
        super().__init__()
        self._workspace = Path(options.working_dir).expanduser().resolve()
        self._agent_spec = options.agent_spec

    def _options(self):
        from .options import AtomOptions

        options = AtomOptions.load(self._workspace)
        options.agent_spec = self._agent_spec
        return options

    async def remember_skill(self, skill_id: str, directory: str | None = None) -> str:
        """Activate a skill here and remember it for future workspace sessions.

        Use an exact name from self.skills.search(). For a skill from a local
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
        uses NOOA_MODEL when set, so that override still wins.
        Model availability and
        credentials are checked when the model is used, not by this operation.
        """
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty alias or provider/model ID")
        path = self._save({("atom", "default_model"): model.strip()})
        return f"Saved default model in {path}; the running model is unchanged."

    def remember_mcp(self, name: str, auto_connect: bool = True) -> str:
        """Save a configured MCP definition and its startup preference.

        The server comes from the workspace or user settings or from
        .mcp.json. Use environment placeholders for credentials. This saves the definition without connecting, approving, or authenticating
        it. Existing exact-definition approvals still apply when connecting.
        Pool-supplied servers are not automatically copied into NOOA settings.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a nonempty MCP server name")
        if not isinstance(auto_connect, bool):
            raise ValueError("auto_connect must be a boolean")
        servers = self._agent.skills.mcp
        servers.refresh_settings()
        # Reuse the exact, unresolved definition. The request only describes
        # configuration; it neither grants approval nor reads secrets.
        definition = servers.approval_request(name).config
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
                ("atom", "mcp_servers", name): definition,
                ("atom", "mcp_auto_connect"): names,
            }
        )
        # The saved form is normalized (e.g. a bare url gains its transport).
        # Adopt it first so the refresh does not see a "changed" definition
        # and disconnect a live server.
        servers.adopt(name, definition)
        servers.refresh_settings()
        return f"Saved MCP server `{name}` in {path}; auto-connect={auto_connect}. Connection approvals are unchanged."

    def forget_mcp(self, name: str) -> str:
        """Disable startup connection and remove this workspace's MCP definition.

        A null entry masks an inherited definition. An existing connection is
        managed separately through self.skills.deactivate; this does not revoke
        approvals or remove credential caches. Client-supplied servers are outside
        this preference and may still be supplied by that client on startup.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a nonempty MCP server name")
        path = self._save(
            {
                ("atom", "mcp_servers", name): None,
                ("atom", "mcp_auto_connect"): [
                    n for n in self._project_auto_connect() if n != name
                ],
            }
        )
        return f"Forgot MCP startup preference `{name}` in {path}."

    def set_live_context(self, kind: str, every_n_turns: int | None = None) -> str:
        """Choose how the live <context> block reaches the model, now and in future sessions.

        kind is "append" (a snapshot is appended to the history at least every
        every_n_turns turns; default 5), "dynamic" (regenerated on every request)
        or "off". every_n_turns is for "append" only. The running agent switches
        at its next model call; the saved setting applies to fresh sessions.
        """
        from nooa.context_blocks import RenderConfig, coerce_live_context

        if not isinstance(kind, str) or not kind.strip():
            raise ValueError('kind must be "append", "dynamic" or "off"')
        kind = kind.strip()
        if every_n_turns is not None and kind != "append":
            raise ValueError('every_n_turns applies to kind="append" only')
        request: Any = (
            {"kind": kind, "every_n_turns": every_n_turns} if every_n_turns is not None else kind
        )
        policy = coerce_live_context(request)
        path = self._save({("atom", "render_config", "live_context"): policy.model_dump()})
        agent = self._agent
        if agent is not None and hasattr(agent, "render_config"):
            agent.render_config = RenderConfig(live_context=policy)
            applied = "applied from the next model call"
        else:
            applied = "applies from the next session"
        detail = f" every {policy.every_n_turns} turns" if policy.kind == "append" else ""
        return f"Live context set to {policy.kind}{detail}; saved in {path}, {applied}."

    @slash_command(
        "live-context",
        argument_hint="<dynamic|off|append> [every_n_turns]",
        completions=("dynamic", "off", "append"),
        output_to_agent=False,
    )
    async def live_context_command(self, args: str) -> str:
        """Show or set how the live <context> block reaches the model."""
        words = args.split()
        if not words:
            status = self.status()
            saved, current = status["saved"]["live_context"], status["current"]["live_context"]
            return (
                f"Live context now: {_describe(current)}; saved for future sessions: "
                f"{_describe(saved) if saved else 'default (append every 5 turns)'}. "
                "Usage: /live-context <dynamic|off|append> [every_n_turns]"
            )
        if len(words) > 2:
            return "Usage: /live-context <dynamic|off|append> [every_n_turns]"
        try:
            every_n = int(words[1]) if len(words) == 2 else None
            return self.set_live_context(words[0], every_n_turns=every_n)
        except ValueError as exc:
            return f"Not changed: {exc}. Usage: /live-context <dynamic|off|append> [every_n_turns]"

    def clear_live_context(self) -> str:
        """Forget this workspace's live-context choice; fresh sessions use the default (append every 5)."""
        path = self._save({("atom", "render_config", "live_context"): None})
        return f"Cleared the live-context setting in {path}; the running agent is unchanged."

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
                "live_context": _policy_summary(
                    options.render_config.live_context if options.render_config else None
                ),
            },
            "current": {
                "active_skills": self._agent.skills.activated(),
                "model": getattr(getattr(self._agent, "llm", None), "model", None),
                "live_context": _policy_summary(
                    getattr(getattr(self._agent, "render_config", None), "live_context", None)
                ),
                "connected_mcp": self._agent.skills.mcp.connected(),
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
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
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
