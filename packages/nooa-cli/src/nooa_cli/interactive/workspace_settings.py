# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-facing workspace preferences for NOOA interactive hosts."""

from pathlib import Path
from typing import Any, Literal

from nooa.skill import Skill


class WorkspaceSettings(Skill):
    """Manage saved defaults for NOOA TUI and ACP sessions in this workspace.

    Preferences belong to this workspace's .nooa/settings.yaml. They apply to
    fresh agents in either client; other live agents retain their current state.
    Named operations cover skills, memory, reflection, MCP startup, and model
    defaults. Ordinary self.skills.load/activate remains session-local.
    """

    def __init__(self, options: Any, *, live_config: Any = None):
        super().__init__()
        self._workspace = Path(options.working_dir).expanduser().resolve()
        self._agent_spec = options.agent_spec
        self._legacy_agent = options.legacy_agent
        self._live_config = live_config if live_config is not None else options

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

    async def configure_memory(self, scope: Literal["off", "session", "project"]) -> str:
        """Apply and save this agent type's memory scope in this workspace.

        Session memory uses each session's own sidecar database. Project memory
        shares a store across workspace sessions. Other live agents retain their
        current memory configuration. Disabling memory does not delete its data.
        """
        modes = {"off": "off", "session": "local", "project": "on"}
        if scope not in modes:
            raise ValueError("Memory scope must be off, session, or project")
        return await self._run_control("memory", [modes[scope]])

    async def configure_reflection(self, enabled: bool) -> str:
        """Apply and save idle memory reflection for this agent type here.

        Requires attached memory when enabling. Other live agents retain their
        configuration. Use self.workspace_settings.configure_memory first.
        """
        if not isinstance(enabled, bool):
            raise ValueError("enabled must be a boolean")
        return await self._run_control("reflection", ["on" if enabled else "off"])

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
        options = self._options()
        names = [item for item in options.mcp_auto_connect if item != name]
        if auto_connect:
            names.append(name)
        path = self._save(
            {
                ("coding", "mcp_servers", name): definition,
                ("coding", "mcp_auto_connect"): names,
            }
        )
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
        options = self._options()
        path = self._save(
            {
                ("coding", "mcp_servers", name): None,
                ("coding", "mcp_auto_connect"): [n for n in options.mcp_auto_connect if n != name],
            }
        )
        return f"Forgot MCP startup preference `{name}` in {path}."

    def status(self) -> dict[str, Any]:
        """Inspect effective saved defaults and current state without credentials.

        Saved values include layered configuration; explicit launch overrides may
        supersede them. Memory/reflection preferences are scoped to this agent
        type. MCP output contains names only, never connection secrets.
        """
        from .memory import agent_memory_key

        options = self._options()
        key = agent_memory_key(self._agent, options)
        reflection = getattr(self._agent, "_reflection_runner", None)
        return {
            "workspace": str(self._workspace),
            "settings_file": str(self._workspace / ".nooa" / "settings.yaml"),
            "agent_key": key,
            "saved": {
                "active_skills": options.active_skills,
                "inactive_skills": options.inactive_skills,
                "skills_directories": [str(p) for p in options.skills_dirs],
                "memory": options.memory_agents.get(key, options.memory),
                "reflection": options.reflection_agents.get(key, options.reflection),
                "default_model": options.default_model,
                "mcp_servers": sorted(options.mcp_servers),
                "mcp_auto_connect": options.mcp_auto_connect,
            },
            "current": {
                "active_skills": self._agent.skills.activated(),
                "memory_attached": hasattr(self._agent, "memory"),
                "memory": self._live_config.memory_agents.get(key, self._live_config.memory),
                "reflection_enabled": bool(reflection and reflection.enabled),
                "model": getattr(getattr(self._agent, "_llm", None), "model", None),
                "connected_mcp": self._agent.mcp.connected(),
                "active_mcp_skills": [
                    name for name in self._agent.skills.activated() if name.startswith("mcp.")
                ],
            },
        }

    def _save(self, updates: dict[tuple[str, ...], Any]) -> Path:
        from .settings import write_settings_updates

        path, _ = write_settings_updates(updates, workspace=self._workspace)
        return path

    async def _run_control(self, name: str, args: list[str]) -> str:
        from .controls import CONTROL_TYPES, ControlMessage
        from .memory import configure_session_memory

        options = self._options()
        behavior_fields = [
            field
            for field in type(options).model_fields
            if field.startswith(("memory", "reflection"))
        ]
        # These controls act on the current agent. Defaults written by another
        # session must not implicitly switch its memory store or reflection.
        for field in behavior_fields:
            value = getattr(self._live_config, field)
            setattr(options, field, value.copy() if isinstance(value, dict) else value)

        def configure_memory():
            session = getattr(self._agent, "_session_manager", None)
            configure_session_memory(
                self._agent,
                options,
                agent_db=session.path if session is not None else None,
                session_id=session.id if session is not None else None,
            )
            for field in behavior_fields:
                setattr(self._live_config, field, getattr(options, field))

        control = CONTROL_TYPES[name](
            self._agent,
            options,
            workspace=self._workspace,
            configure_memory=configure_memory,
            command_registry=getattr(self._agent, "_command_registry", None),
        )
        result = await control.run(args)
        if not result.success or any(
            isinstance(output, ControlMessage) and output.style == "warning"
            for output in result.outputs
        ):
            raise RuntimeError(str(result))
        return str(result)
