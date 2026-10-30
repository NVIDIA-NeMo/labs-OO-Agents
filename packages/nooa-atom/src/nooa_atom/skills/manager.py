# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SkillManager: the Atom agent's code skills, text skills and MCP servers."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from nooa.agentdoc import doc as agentdoc
from nooa.agentdoc import hidden, spec
from nooa.events import Notification
from nooa.skill import Skill, TextSkill, slash_command
from nooa.skill_registry import SkillRegistry
from nooa_atom.skills.mcp_servers import (
    MCPServers,
    MCPSignInRequired,
    attr_name,
    tool_names,
)
from nooa_atom.workspace.mcp_approval import MCPApprovalRequired

Kind = Literal["code", "text", "mcp"]
_KIND_LABELS = {"code": "code", "text": "text", "mcp": "MCP"}
_DESCRIPTION_WIDTH = 100

# Timers and ticks are built into ``self.jobs``, and a file tail is a Shell command,
# not skills: a search for them points there instead of answering "No skills match".
_WORK_WORDS = frozenset(
    "timer timers schedule scheduler scheduled cron remind reminder later after every "
    "periodic periodically monitor watch tail follow delay sleep wait".split()
)
_BUILT_IN_WORK = (
    "self.jobs (built in, no skill needed): self.jobs.after(seconds, name=...) timer "
    "(a delay counts from now: subtract time you have already spent), "
    "self.jobs.every(seconds, name=...) ticks; watch a command or follow a file: "
    'self.jobs.launch(sh.run("tail -F app.log"), name=..., progress=Throttle(seconds=5))'
)


@dataclass(frozen=True)
class SkillEntry:
    """One skill of any kind, as search and the host controls show it.

    ``name`` is what the model passes to ``activate`` and ``doc``:
    the agent attribute of a code skill, the id of a text skill, the name of
    an MCP server. ``key`` is the name the skill is registered under
    (``nemo.shell``, ``cmd.review``, ``mcp.<client server>``, or the MCP
    server name); both are accepted wherever a name is.
    """

    name: str
    kind: Kind
    state: str
    description: str
    source: str
    key: str

    def line(self) -> str:
        state = self.state
        if self.kind == "mcp" and self.state in ("active", "connected"):
            # The call form is otherwise invisible: attached servers are hidden
            # from doc(self).
            state += f"; call: await self.{attr_name(self.name)}.<tool>(...)"
        text = f"{self.name} ({self.kind}, {state}): {self.description}"
        return text if len(text) <= 160 else text[:159] + "…"


def _first_line(text: str | None) -> str:
    return (text or "").strip().split("\n", 1)[0].strip()


class SkillManager(Skill):
    """Find, activate and inspect code/text skills and MCP servers.

    ``search(query)`` finds skills of every kind by name and description.
    ``await activate([...])`` turns skills on: a code skill is loaded and its
    object is ``self.<name>``; an MCP server is connected and its tools are
    methods of ``self.<server>``; a text skill's instructions arrive on the
    ``system_messages`` channel at the start of your next turn. Use
    ``print(self.skills.doc(name))`` for full text instructions, code APIs or
    MCP tools now. An MCP server may first need the person's approval or
    sign-in; ``activate`` says so, and you get a notice when it connects.
    """

    __nosnapshot__ = True

    registry: Annotated[SkillRegistry, hidden]
    mcp: Annotated[MCPServers | None, hidden]

    def __init__(self, agent: Any) -> None:
        super().__init__()
        self._owner = agent
        # The core registry discovers and loads code and text skills. It also
        # installs the ``skills`` context block, ``self.skills.status()``.
        self.registry = SkillRegistry(agent)
        self.mcp = None

    # ------------------------------------------------------------------
    # Model-facing
    # ------------------------------------------------------------------

    def search(self, query: str, limit: int = 10) -> str:
        """Find skills whose name or description contains any word of ``query``.

        One line per match: name, kind, state and a one-line description.
        Names that match come first. Searching loads nothing.
        """
        words = [word for word in query.lower().split() if word]
        hint = [_BUILT_IN_WORK] if any(w.strip(".,;:!?\"'()") in _WORK_WORDS for w in words) else []
        scored = []
        for entry in self.entries():
            name = entry.name.lower()
            text = f"{name} {entry.key.lower()} {entry.description.lower()}"
            hits = sum(word in text for word in words)
            if hits:
                scored.append((-sum(word in name for word in words), -hits, entry.name, entry))
        if not scored:
            return hint[0] if hint else f"No skills match {query!r}."
        scored.sort(key=lambda item: item[:3])
        lines = hint + [entry.line() for *_, entry in scored[:limit]]
        if len(scored) > limit:
            lines.append(f"... {len(scored) - limit} more; narrow the query or raise limit")
        return "\n".join(lines)

    async def activate(self, names: list[str]) -> str:
        """Turn on skills by name; return one line per skill saying what happened.

        Code skill: loaded and active as ``self.<name>``. MCP server:
        connected, its tools are methods of ``self.<server>``. Text skill:
        its instructions arrive on the ``system_messages`` channel in your
        next turn. Names may be globs (``"github*"``).
        """
        lines = []
        for pattern in names:
            matched = self._resolve(pattern)
            if not matched:
                lines.append(f"{pattern}: no such skill; try self.skills.search({pattern!r})")
            for entry in matched:
                lines.append(await self._activate(entry))
        return "\n".join(lines)

    async def deactivate(self, names: list[str]) -> str:
        """Turn off skills by name; an MCP server stays connected. One line per skill."""
        lines = []
        for pattern in names:
            matched = self._resolve(pattern)
            if not matched:
                lines.append(f"{pattern}: no such skill")
            for entry in matched:
                if entry.kind == "text":
                    lines.append(f"{entry.name}: a text skill has nothing to turn off")
                elif entry.kind == "mcp" and entry.key == entry.name and self.mcp is not None:
                    await self.mcp.deactivate([_literal(entry.key)])
                    lines.append(f"{entry.name}: inactive (still connected)")
                else:
                    self.registry.deactivate([_literal(entry.key)])
                    lines.append(f"{entry.name}: inactive")
        return "\n".join(lines)

    @hidden
    def read(self, name: str) -> str:
        """The instructions of the text skill ``name``, as text."""
        entry = self._one(name)
        if entry.kind != "text":
            raise ValueError(
                f"{entry.name} is a {entry.kind} skill, not a text skill; "
                f"use print(self.skills.doc({entry.name!r})) for its description"
            )
        skill = self.registry[entry.key]
        return (type(skill).__doc__ or "").split("\n---\n", 1)[-1].strip()

    def doc(self, name: str) -> str:
        """Return full text instructions, code APIs or MCP tools as str; use print(self.skills.doc(name)).

        An unconnected MCP skill returns a connection-status message instead of its tools.
        """
        entry = self._one(name)
        if entry.kind == "text":
            attr = _leaf(entry.key)
            return (
                f"{entry.name} (text skill): {entry.description}\n"
                f"Source: {entry.source}; its files: self.{attr}.read_file(path), "
                f"its scripts: self.{attr}.run_script(name)\n\n"
                f"Instructions:\n{self.read(entry.name)}"
            )
        header = entry.line()
        if entry.kind == "mcp":
            tool = self._mcp_tool(entry)
            if tool is None:
                state = self.mcp.error(entry.key) if self.mcp is not None else None
                return (
                    f"{header}\nnot connected{': ' + state if state else ''}; activate to connect"
                )
            return f"{header}\n\n{agentdoc(tool)}"
        if entry.key in self.registry.loaded():
            return f"{header}\n\n{agentdoc(self.registry[entry.key])}"
        # Asking for the full description of a code skill imports it.
        record = self.registry.entry(entry.key)
        target = record.entry_point.load() if record and record.entry_point else None
        return f"{header}\n\n{agentdoc(target)}" if target is not None else header

    # ------------------------------------------------------------------
    # The context block
    # ------------------------------------------------------------------

    def status(self) -> str:
        """Which skills are active now, and how many more there are (sync).

        Also the text of the ``<skills>`` block of agents that show one.
        """
        active, rest_kinds, rest = [], [], 0
        for name, kind, key in self._index():
            if self._is_active(kind, key):
                active.append(
                    f"{name} (mcp: await self.{attr_name(name)}.<tool>(...))"
                    if kind == "mcp"
                    else name
                )
            else:
                rest += 1
                if _KIND_LABELS[kind] not in rest_kinds:
                    rest_kinds.append(_KIND_LABELS[kind])
        mcp_last = sorted(active, key=lambda item: (" (mcp: await self." in item, item))
        lines = [f"Active: {', '.join(mcp_last) or 'none'}"]
        if rest:
            order = [label for label in _KIND_LABELS.values() if label in rest_kinds]
            lines.append(
                f"{rest} more ({', '.join(order)}): self.skills.search('query') (sync); "
                "print(self.skills.doc('name')) for full instructions/APIs "
                "(unconnected MCP: connection status); "
                "await self.skills.activate(['name']) (async)"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Host-facing
    # ------------------------------------------------------------------

    @hidden
    def set_mcp_servers(self, servers: MCPServers) -> None:
        """Use ``servers`` for the agent's configured MCP servers."""
        servers.bind(self._owner, self._notice)
        self.mcp = servers

    @hidden
    def share_from(self, parent: SkillManager) -> list[str]:
        """Use ``parent``'s connected MCP servers here too; return warnings.

        For a child session: the client-provided servers (``mcp.<name>``)
        and the configured servers ``parent`` has connected get the same
        tool objects here (no new connection), active where they are active
        in ``parent``. Closing this manager detaches them and closes nothing.
        """
        warnings: list[str] = []
        for key in parent.registry.loaded():
            if not key.startswith("mcp."):
                continue
            try:
                self.registry.register(key, parent.registry[key])
                if key in parent.registry.activated():
                    self.registry.activate([_literal(key)])
            except (KeyError, ValueError) as exc:
                warnings.append(f"MCP server {_leaf(key)!r} was not shared: {exc}")
        if self.mcp is not None and parent.mcp is not None:
            warnings.extend(self.mcp.share_from(parent.mcp))
        return warnings

    @hidden
    def after_restore(self) -> None:
        """Drop restored context blocks that need an attribute this agent lacks; say so.

        A snapshot saved before the skill manager holds the ``<mcp>`` block,
        ``self.mcp.status()``, and may hold blocks of skills that are not
        active now. Those would fail on every turn. They are removed and the
        agent gets a notice naming them.
        """
        import re

        from nooa.storage.snapshot import AgentSnapshot, DynamicContextBlock

        agent = self._owner
        stale = []
        for block in AgentSnapshot.from_agent(agent).context:
            if not isinstance(block, DynamicContextBlock):
                continue
            owner = re.match(r"\s*self\.(\w+)", block.expr)
            if owner and not hasattr(agent, owner.group(1)):
                stale.append(block.key)
                agent.context.pop(block.key, None)
        if stale:
            self._notice(
                "This session was saved by an older version. These skills and MCP servers "
                f"were not restored: {', '.join(stale)}. MCP servers this workspace connects "
                "at start are connected again; find the others with self.skills.search() and "
                "turn them on with self.skills.activate()."
            )

    @hidden
    def entries(self) -> list[SkillEntry]:
        """Every skill with its kind, state, one-line description and source."""
        return [self._entry(name, kind, key) for name, kind, key in self._index()]

    @hidden
    def entry(self, name: str) -> SkillEntry | None:
        """The skill called ``name`` (its name or registered key), or ``None``."""
        for entry_name, kind, key in self._index():
            if name in (entry_name, key):
                return self._entry(entry_name, kind, key)
        return None

    @hidden
    def register(self, name: str, skill: Any) -> None:
        """Register an object as a skill under ``name`` (core ``SkillRegistry`` rules).

        A name starting ``mcp.`` is an MCP server a client supplied.
        """
        self.registry.register(name, skill)

    @hidden
    def discovered(self) -> list[str]:
        """Registered names of the code and text skills found."""
        return self.registry.discovered()

    @hidden
    def loaded(self) -> list[str]:
        return self.registry.loaded()

    @hidden
    def activated(self) -> list[str]:
        """Registered names of the active code skills and client MCP servers."""
        return self.registry.activated()

    @hidden
    def load(self, patterns: list[str]) -> None:
        self.registry.load(patterns)

    @hidden
    def discover_skills_dirs(self, dirs: list[Any]) -> None:
        self.registry.discover_skills_dirs(dirs)

    @hidden
    def discover_libs(self, path: Any) -> None:
        self.registry.discover_libs(path)

    @hidden
    async def reload(self, name: str | None = None) -> str:
        return await self.registry.reload(name)

    @hidden
    def __getitem__(self, key: str) -> Any:
        return self.registry[key]

    @hidden
    async def aclose(self) -> None:
        """Stop MCP sign-ins, detach MCP servers, then the code and text skills."""
        try:
            if self.mcp is not None:
                await self.mcp.aclose()
        finally:
            await self.registry.aclose()

    @slash_command(
        "mcp-add",
        argument_hint="<server info: name, URL, transport, auth notes>",
        output_to_agent=True,
    )
    async def mcp_add_command(self, args: str) -> str:
        """Ask the agent to configure a new MCP server from the details you provide.

        The person pastes what they have about a server (a name and URL, a
        ``claude mcp add ...`` line, a docs snippet, an OAuth client id). This
        edits nothing; it returns a task for the agent, which writes the
        ``atom.mcp_servers.<name>`` block and guides the person through
        approval and sign-in.
        """
        details = args.strip()
        if not details:
            return (
                "Usage: /mcp-add <server info>\n"
                "Paste a name and URL (with transport and auth notes), or a "
                "`claude mcp add ...` line, for example:\n"
                "  /mcp-add docs https://example.com/mcp streamable-http"
            )
        if self.mcp is None:
            return "This agent has no MCP servers."
        config_path = self.mcp.config_path()
        configured = ", ".join(self.mcp.discovered()) or "(none)"
        return (
            "The user wants to add a new MCP server. Here are the details they provided:\n\n"
            f"{details}\n\n"
            "Do the following:\n"
            "1. Parse the server name, URL, transport (default `streamable-http` for HTTP "
            "URLs), and any auth info (OAuth client_id, static API key/headers).\n"
            f"2. Add an `atom.mcp_servers.<name>` YAML block to `{config_path}` (create the "
            "file or section if missing; do not remove existing servers). Use an environment "
            "placeholder in `headers` for a static API key, or `oauth_client_id` for a "
            "pre-registered OAuth client. Never write a secret value into project config.\n"
            "3. Tell the user to run `/mcp approve <name>` to review the exact config, then "
            "run the displayed `/mcp approve <name> <code>` command themselves. The agent "
            "must not approve MCP config. If the server needs a sign-in, the user opens the "
            "link shown and pastes the address the browser ends on with `/mcp auth <name> "
            "<address>`.\n"
            "4. Confirm what you wrote and show the resulting config block without resolving "
            "or printing secret values.\n\n"
            f"Currently configured servers: {configured}.\n"
            f"Config file: {config_path}."
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _index(self) -> list[tuple[str, Kind, str]]:
        """``(name, kind, key)`` for every skill, without loading or describing any.

        Code skills come first, then text skills, then MCP servers. A name
        another skill already has is shown under its key instead.
        """
        found: list[tuple[str, Kind, str]] = []
        taken: set[str] = set()

        def add(name: str, kind: Kind, key: str) -> None:
            shown = key if name in taken else name
            if shown in taken:
                return
            taken.add(shown)
            found.append((shown, kind, key))

        keys = self.registry.discovered()
        for key in keys:
            if not key.startswith(("cmd.", "mcp.")):
                add(_leaf(key), "code", key)
        for key in keys:
            if key.startswith("cmd."):
                add(key.split(".", 1)[1], "text", key)
        for key in keys:
            if key.startswith("mcp."):
                add(_leaf(key), "mcp", key)
        if self.mcp is not None:
            for server in self.mcp.discovered():
                add(server, "mcp", server)
        return found

    def _is_active(self, kind: Kind, key: str) -> bool:
        if kind == "text":
            return False
        if kind == "mcp" and not key.startswith("mcp."):
            return self.mcp is not None and self.mcp.state(key) == "active"
        return key in self.registry.activated()

    def _entry(self, name: str, kind: Kind, key: str) -> SkillEntry:
        if kind == "mcp" and not key.startswith("mcp."):
            assert self.mcp is not None
            return SkillEntry(
                name,
                kind,
                self.mcp.state(key),
                self.mcp.description(key),
                "MCP settings",
                key,
            )
        loaded = key in self.registry.loaded()
        skill = self.registry[key] if loaded else None
        state = "active" if self._is_active(kind, key) else "available"
        if kind == "text":
            description = skill.description if isinstance(skill, TextSkill) else ""
            source = str(skill.source_dir) if isinstance(skill, TextSkill) else ""
        elif kind == "mcp":
            names = tool_names(skill) if skill is not None else []
            description = "MCP server from the client" + (
                f"; tools: {', '.join(names)}" if names else ""
            )
            source = "client"
        else:
            description, source = self._code_description(key, skill)
        return SkillEntry(name, kind, state, description[:_DESCRIPTION_WIDTH], source, key)

    def _code_description(self, key: str, skill: Any) -> tuple[str, str]:
        """A code skill's one line and source, without importing it.

        A loaded skill describes itself (its class docstring). An installed
        one that is not loaded is described by its distribution's summary
        from the package metadata, which needs no import.
        """
        record = self.registry.entry(key)
        point = record.entry_point if record is not None else None
        source = getattr(point, "value", None) or (type(skill).__module__ if skill else "")
        if skill is not None:
            return _first_line(type(skill).__doc__), source
        dist = getattr(point, "dist", None)
        metadata = getattr(dist, "metadata", None)
        summary = metadata.get("Summary") if metadata is not None else None
        return _first_line(summary), source

    def _resolve(self, pattern: str) -> list[SkillEntry]:
        index = self._index()
        exact = [item for item in index if pattern in (item[0], item[2])]
        if not exact:
            exact = [
                item
                for item in index
                if fnmatch.fnmatchcase(item[0], pattern) or fnmatch.fnmatchcase(item[2], pattern)
            ]
        return [self._entry(*item) for item in exact]

    def _one(self, name: str) -> SkillEntry:
        entry = self.entry(name)
        if entry is None:
            raise KeyError(f"No skill called {name!r}; try self.skills.search({name!r})")
        return entry

    def _mcp_tool(self, entry: SkillEntry) -> Any:
        if entry.key.startswith("mcp."):
            return self.registry[entry.key] if entry.key in self.registry.loaded() else None
        return self.mcp.tool(entry.key) if self.mcp is not None else None

    async def _activate(self, entry: SkillEntry) -> str:
        if entry.kind == "text":
            channel = self._owner.queue_manager.get_channel("system_messages")
            channel.put(self.doc(entry.name))
            return (
                f"{entry.name}: its instructions arrive on the system_messages channel in "
                f"your next turn; print(self.skills.doc({entry.name!r})) shows them now"
            )
        if entry.kind == "mcp" and not entry.key.startswith("mcp."):
            return await self._connect(entry)
        self.registry.activate([_literal(entry.key)])
        if entry.key not in self.registry.activated():
            return f"{entry.name}: could not be activated; see the log"
        return f"{entry.name}: active; use self.{_leaf(entry.key)}"

    async def _connect(self, entry: SkillEntry) -> str:
        assert self.mcp is not None
        name = entry.key
        try:
            if name in self.mcp.connected():
                self.mcp.activate([_literal(name)])
            else:
                await self.mcp.connect([_literal(name)])
        except MCPApprovalRequired:
            return f"{name}: not approved; ask the person to run /mcp approve {name}"
        except MCPSignInRequired as needed:
            self._owner.message(str(needed))
            return (
                f"{name}: needs sign-in; the person was sent the link and finishes with "
                f"/mcp auth {name} <address>; you get a notice when it connects"
            )
        except Exception as exc:
            return f"{name}: failed: {exc}"
        tools = ", ".join(self.mcp.tool_names(name)) or "none"
        return f"{name}: active; its tools are methods of self.{attr_name(name)}: {tools}"

    def _notice(self, text: str) -> None:
        """Tell the agent something it reads with its next model call."""
        self._owner.event_manager.add(Notification(source="skills", description=text))


def _leaf(key: str) -> str:
    return key.rsplit(".", 1)[-1].replace("-", "_")


def _literal(name: str) -> str:
    """``name`` as a glob that matches only itself."""
    return "".join({"[": "[[]", "*": "[*]", "?": "[?]"}.get(c, c) for c in name)


# This inherited implementation directory is not the registry's discovery roots.
spec(SkillManager, "source_dir", hidden=True)
