# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Skill — base class and wrapper for agent-loadable capabilities."""

import inspect
import re
import shlex
from pathlib import Path
from typing import Annotated, Any

import yaml
from pydantic import BaseModel

from nooa.agentdoc import hidden

# ---------------------------------------------------------------------------
# @slash_command decorator
# ---------------------------------------------------------------------------

_SLASH_COMMAND_ATTR = "_slash_command_meta"


class SlashCommandMeta:
    """Metadata attached to a method by @slash_command."""

    __slots__ = ("name", "argument_hint", "completions", "output_to_agent")

    def __init__(
        self,
        name: str,
        argument_hint: str | None,
        completions: tuple[str, ...] = (),
        output_to_agent: bool = True,
    ):
        self.name = name
        self.argument_hint = argument_hint
        self.completions = completions
        self.output_to_agent = output_to_agent


def slash_command(
    name: str,
    *,
    argument_hint: str | None = None,
    completions: tuple[str, ...] | list[str] = (),
    output_to_agent: bool = True,
):
    """Mark a Skill method as a user-invocable slash command.

    The decorated method receives the raw argument string and returns a
    prompt string that is injected as a user message.

    Args:
        name: Command name (without leading /). E.g. "gl-ci" -> /gl-ci.
        argument_hint: Shown in help. E.g. "<pipeline-id>".
        completions: Subcommand names offered by tab-completion.
        output_to_agent: If True (default), the command's return value is fed to the
            agent as a turn (via the ``slash_commands`` queue). If False, the
            output is shown to the user in the TUI only — no agent turn is
            spent. Use ``output_to_agent=False`` for read-only commands (list/status)
            whose result the human wants to see without invoking the LLM.

    Usage::

        class GitLabSkill(Skill):
            @slash_command("gl-ci", argument_hint="<pipeline-id>")
            async def monitor_ci(self, args: str) -> str:
                '''Monitor a CI pipeline.'''
                return f"Monitor CI pipeline {args}."

            @slash_command("gl-status", output_to_agent=False)
            async def status(self, args: str) -> str:
                '''Show CI status to the user without spending an agent turn.'''
                return self._render_status()
    """

    def decorator(fn):
        setattr(
            fn,
            _SLASH_COMMAND_ATTR,
            SlashCommandMeta(name, argument_hint, tuple(completions), output_to_agent),
        )
        return fn

    return decorator


def get_slash_commands(skill) -> list[tuple[SlashCommandMeta, Any]]:
    """Extract all @slash_command methods from a Skill instance.

    Returns list of (meta, bound_method) pairs.
    """
    results = []
    for attr_name in dir(type(skill)):
        try:
            obj = getattr(type(skill), attr_name)
        except AttributeError:
            continue
        meta = getattr(obj, _SLASH_COMMAND_ATTR, None)
        if meta is not None:
            results.append((meta, getattr(skill, attr_name)))
    return results


# ---------------------------------------------------------------------------
# SKILL.md frontmatter parsing
# ---------------------------------------------------------------------------
class _SkillProperties(BaseModel):
    name: str
    description: str


def _find_skill_md(skill_dir: Path) -> Path | None:
    # Check for both case variants; on case-insensitive filesystems (macOS HFS+/APFS),
    # path.exists() may return True even if the actual filename differs in case.
    # We iterate through actual directory entries to get real filenames, then
    # prefer SKILL.md (uppercase) over skill.md when both exist.
    if not skill_dir.is_dir():
        return None
    matches: dict[str, Path] = {}
    for entry in skill_dir.iterdir():
        if entry.name in ("SKILL.md", "skill.md"):
            matches[entry.name] = entry
    return matches.get("SKILL.md") or matches.get("skill.md")


def _parse_frontmatter(content: str) -> tuple[dict[str, Any], str]:
    """Parse SKILL.md frontmatter compatible with Claude Code's lenient behaviour.

    Strategy: try ``yaml.safe_load`` on the whole block first — it handles
    multi-line values (e.g. ``metadata:`` sub-keys) correctly.  If the block
    contains an invalid YAML scalar (e.g. ``argument-hint: "<title>" [-p]``),
    fall back to a line-by-line regex that treats each value as a raw string,
    matching what Claude Code actually does.

    In both paths, YAML lists are coerced to strings (Claude Code v2.1.47+
    behaviour for ``argument-hint: [label]``).
    """
    if not content.startswith("---"):
        raise ValueError("SKILL.md must start with YAML frontmatter (---)")
    parts = content.split("---", 2)
    if len(parts) < 3:
        raise ValueError("SKILL.md frontmatter not properly closed with ---")

    fm_text = parts[1]
    body = parts[2].strip()

    # --- Fast path: strict YAML ---
    try:
        meta = yaml.safe_load(fm_text) or {}
        if not isinstance(meta, dict):
            raise ValueError("SKILL.md frontmatter must be a YAML mapping")
    except yaml.YAMLError:
        # --- Fallback: line-by-line regex (Claude Code-compatible) ---
        meta = {}
        for line in fm_text.splitlines():
            m = re.match(r"^([a-zA-Z][a-zA-Z0-9_-]*):\s*(.+)$", line)
            if not m:
                continue
            key, raw = m.group(1), m.group(2).strip()
            try:
                parsed = yaml.safe_load(raw)
                meta[key] = str(parsed) if isinstance(parsed, list) else parsed
            except yaml.YAMLError:
                meta[key] = raw  # invalid scalar — keep as raw string

    # Coerce any top-level list values to strings (e.g. argument-hint: [label])
    for key, val in meta.items():
        if isinstance(val, list):
            meta[key] = str(val)

    if "metadata" in meta and isinstance(meta["metadata"], dict):
        meta["metadata"] = {str(k): str(v) for k, v in meta["metadata"].items()}
    return meta, body


def _read_skill_properties(skill_dir: Path) -> _SkillProperties:
    skill_md = _find_skill_md(skill_dir)
    if skill_md is None:
        raise ValueError(f"SKILL.md not found in {skill_dir}")
    meta, _ = _parse_frontmatter(skill_md.read_text(encoding="utf-8"))
    if "name" not in meta:
        raise ValueError("Missing required frontmatter field: name")
    if "description" not in meta:
        raise ValueError("Missing required frontmatter field: description")
    return _SkillProperties(
        name=str(meta["name"]).strip(),
        description=str(meta["description"]).strip(),
    )


def _parse_skill_md(path: Path) -> tuple[str, str, str]:
    """Parse a skill directory into (skill_id, description, body).

    Returns:
        skill_id:    Normalized directory name (lowercase, hyphens)
        description: From frontmatter ``description:`` field
        body:        Skill content after the frontmatter block

    Raises:
        ValueError: SKILL.md not found.
    """
    skill_id = path.name.lower().replace(" ", "-").replace("_", "-")

    skill_md = _find_skill_md(path)
    if skill_md is None:
        raise ValueError(f"SKILL.md not found in {path}")

    props = _read_skill_properties(path)
    description = props.description or path.name
    _, body = _parse_frontmatter(skill_md.read_text())
    return skill_id, description, body.strip()


def _resolve_skill_path(skill_root: Path, relative: str) -> Path:
    """Resolve a relative path and verify it stays within skill_root.

    Raises:
        ValueError: If the path escapes the skill directory.
        FileNotFoundError: If the resolved path does not exist.
    """
    resolved = (skill_root / relative).resolve()
    if not resolved.is_relative_to(skill_root.resolve()):
        raise ValueError(
            f"Path {relative!r} escapes the skill directory. Only paths within {skill_root} are allowed."
        )
    if not resolved.exists():
        raise FileNotFoundError(f"{relative!r} not found in skill directory {skill_root}")
    return resolved


def _build_script_command(
    script: Path, args: tuple[str, ...], interpreter: str | None = None
) -> str:
    """Build the shell command to run a script.

    Priority: explicit interpreter > shebang line > direct execution.
    Pass ``interpreter`` explicitly for scripts without a shebang.
    """
    quoted_script = shlex.quote(str(script))
    quoted_args = " ".join(shlex.quote(a) for a in args)

    if interpreter:
        cmd = (
            f"{interpreter} {quoted_script} {quoted_args}"
            if quoted_args
            else f"{interpreter} {quoted_script}"
        )
        return cmd.strip()

    cmd = f"{quoted_script} {quoted_args}" if quoted_args else quoted_script
    return cmd.strip()


class Skill:
    """Base class for agent skills.

    Wraps a Python object or inline content so the LLM can discover it via
    ``doc(self.<skill>)``. Third-party libraries use ``Skill(obj)``; inline
    text uses ``Skill(content=...)``. For SKILL.md-based skills use
    ``TextSkill(path=...)``.

        class WritingSkill(Skill):
            \"\"\"Write and reuse helper methods on the agent.\"\"\"

        class MyAgent(Agent, llm=llm):
            self.pd = Skill(pd)  # registers pandas for discovery; use pd directly
    """

    # Tells agentdoc not to expand this type in "Referenced Types" sections.
    # Skills show a brief one-liner wherever they appear as a field type.
    __agentdoc_skip__ = True

    # Skills are reconstructed on agent init (e.g., LibraryManager rescans libs/).
    # They shouldn't be serialized in snapshots — mark them as nosnapshot.
    __nosnapshot__ = True

    # Skill dependencies — list of skill names (category/name) that must be
    # loaded before this skill can function. SkillRegistry resolves these
    # transitively when activate() is called.
    requires: Annotated[tuple[str, ...], hidden] = ()

    # If set, SkillRegistry.activate() registers a dynamic context block
    # with this (key, expr) pair. deactivate() removes it.
    context_block: Annotated[tuple[str, str] | None, hidden] = None

    _agent: Any = None
    _source_dir: Path

    def __init__(self, obj: Any = None, *, content: str | None = None, name: str | None = None):
        n_given = sum(x is not None for x in (obj, content))
        if n_given > 1:
            raise ValueError("Skill() accepts exactly one of: obj, content")
        if n_given == 0 and type(self) is Skill:
            raise ValueError("Skill() requires one of: obj=<object>, content=<str>")

        if obj is not None:
            self._skill_obj = obj
            cls_name = name or "Skill"
            self.__class__ = type(cls_name, (Skill,), {"__doc__": obj.__doc__ or ""})  # pyright: ignore[reportAttributeAccessIssue]
        elif content is not None:
            cls_name = name or "Skill"
            self.__class__ = type(cls_name, (Skill,), {"__doc__": content})  # pyright: ignore[reportAttributeAccessIssue]

    def attach(self, agent: Any) -> None:
        """Called when this skill is installed on an agent.

        Subclasses can override to perform setup that requires the agent reference.
        """
        self._agent = agent

    def detach(self) -> None:
        """Called when this skill is removed from an agent."""
        self._agent = None

    def __dir__(self) -> list[str]:
        # Forward dir() to wrapped object so the LLM can discover its attributes.
        base = list(super().__dir__())
        try:
            skill_obj = object.__getattribute__(self, "_skill_obj")
            return list(set(base) | set(dir(skill_obj)))
        except AttributeError:
            return base

    @property
    def source_dir(self) -> Path | None:
        """Directory containing this skill's source code, or None if unknown.

        Allows an agent to locate and edit the skill's implementation::

            path = self.my_skill.source_dir
            if path:
                await self.shell.view(str(path / "__init__.py"))
        """
        # Explicit override (set during library/skill loading)
        try:
            return object.__getattribute__(self, "_source_dir")
        except AttributeError:
            pass
        # Derive from the module that defines the concrete class
        cls = type(self)
        try:
            source_file = inspect.getfile(cls)
        except (TypeError, OSError):
            return None
        source_path = Path(source_file).resolve().parent
        return source_path


class TextSkill(Skill):
    """Skill loaded from a SKILL.md directory. Has id, description, run_script, read_file."""

    def __init__(self, *, path: Path | str, id: str | None = None):
        skill_id, description, body = _parse_skill_md(Path(path))
        docstring = f"{description}\n---\n{body}"
        class_name = "".join(word.capitalize() for word in skill_id.split("-"))
        self._skill_path = Path(path)
        self.__class__ = type(  # pyright: ignore[reportAttributeAccessIssue]
            class_name, (TextSkill,), {"__doc__": docstring, "_id": id or skill_id}
        )

    @property
    def id(self) -> str:
        return str(type(self)._id)  # pyright: ignore[reportAttributeAccessIssue]  # _id set dynamically in __init__

    @property
    def source_dir(self) -> Path | None:
        return self._skill_path.resolve()

    @property
    def description(self) -> str:
        doc = type(self).__doc__ or ""
        return doc.split("\n---\n")[0].split("\n")[0]

    async def run_script(
        self,
        name: str,
        *args: str,
        interpreter: str | None = None,
        timeout: float = 30.0,
    ) -> str:
        """Run a script from this skill's scripts/ directory.

        Supports any script type — Python, shell, Ruby, Node.js, Perl, etc.
        Scripts with a shebang line run directly. For scripts without one,
        pass ``interpreter`` explicitly.

            output = await self.my_skill.run_script("run_eval.py", "--limit", "10")
            output = await self.my_skill.run_script("report.sh")
            output = await self.my_skill.run_script("query.sql", interpreter="psql -f")
            output = await self.my_skill.run_script("process.js", "input.json")
        """
        script = _resolve_skill_path(self._skill_path, f"scripts/{name}")
        if not script.is_file():
            scripts_dir = self._skill_path / "scripts"
            available = (
                sorted(p.name for p in scripts_dir.iterdir() if p.is_file())
                if scripts_dir.is_dir()
                else []
            )
            raise FileNotFoundError(
                f"Script {name!r} not found in {scripts_dir}. Available: {available}"
            )

        from nooa.tools._bash_session import BashSession

        cmd = _build_script_command(script, args, interpreter=interpreter)
        session = BashSession(cwd=self._skill_path)
        try:
            await session.start()
            stdout, stderr, exit_code = await session.run(cmd, timeout=timeout)
        finally:
            await session.close()

        # Format output to match the previous BashResult.__str__() contract
        output = stdout
        if stderr:
            output += f"\n[stderr]\n{stderr}"
        if exit_code != 0:
            output += f"\n[exit code: {exit_code}]"
        return output

    def read_file(self, path: str) -> str:
        """Read a file from anywhere within this skill's directory.

        content = self.my_skill.read_file("scripts/run_eval.py")
        content = self.my_skill.read_file("assets/prompt.txt")
        content = self.my_skill.read_file("SKILL.md")
        """
        resolved = _resolve_skill_path(self._skill_path, path)
        if not resolved.is_file():
            raise ValueError(f"{path!r} is not a file.")
        return resolved.read_text()
