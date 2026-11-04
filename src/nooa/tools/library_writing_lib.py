# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SkillWriting — write the skills later sessions get: code packages or SKILL.md text."""

import ast
import inspect
import json
import re
import sys
import textwrap
import types
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nooa.library_manager import LibraryManager
from nooa.skill import Skill

# ---------------------------------------------------------------------------
# LintReport
# ---------------------------------------------------------------------------


@dataclass
class LintReport:
    """Result of linting a library file.

    written: True if the file was written to disk.
    loaded:  True if the library was (re-)loaded on the agent.
    errors:  Hard errors — E001 (forbidden builtins), E003 (star imports).
    warnings: Soft issues — E002 (import not in agent's allowed set).
    """

    written: bool = False
    loaded: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        if not self.written:
            status = "ERROR — file not written"
        elif not self.loaded:
            status = "WARNING — written but not loaded"
        else:
            status = "OK — written and loaded"

        issues = [f"  \u2717 {e}" for e in self.errors] + [f"  \u26a0 {w}" for w in self.warnings]
        if not issues:
            return status
        return "\n".join([status, *issues])


# ---------------------------------------------------------------------------
# SkillWriting
# ---------------------------------------------------------------------------


def older_skill_dirs(skills_dir: Path) -> list[Path]:
    """The older directories whose skills still load beside ``skills_dir``.

    Skills were saved in ``.nooa/libs`` before ``.nooa/skills``: for a save
    directory named ``skills``, its sibling ``libs`` (user data on disk).
    """
    skills_dir = Path(skills_dir)
    return [skills_dir.with_name("libs")] if skills_dir.name == "skills" else []


class SkillWriting(Skill):
    """Write skills that later sessions get: code skills or text instructions.

    A code skill is a Python package; a text skill is a SKILL.md with
    instructions. Use ``self.shell`` for file I/O (write, edit, view, grep);
    this skill handles scaffolding, linting, hot-reload, and testing.

    ## Skill contract

    Every code skill's ``__init__.py`` should export a ``Skill`` subclass::

        '''Grep-style search across multiple repos.'''
        from nooa.skill import Skill

        class MultiGrep(Skill):
            '''Grep across a configured list of repos.'''
            def __init__(self):
                self._roots = [...]

            def search(self, pattern: str) -> list[str]:
                ...

    The SkillRegistry auto-attaches the Skill as ``self.<name>``.
    Without a Skill subclass, the module is wrapped via ``Skill(module)``.

    ## Slash commands

    Skills can expose slash commands — user-typed ``/name`` actions that
    inject a prompt into the conversation. Slash commands serve two purposes:

    1. **Do work** — run code, query APIs, modify files, then return a
       result summary.
    2. **Prompt the agent** — the return value becomes a user message in the
       conversation. Use it to tell the agent what happened and what it can
       do next. Think of the return value as instructions for yourself.

    Declare them with ``@slash_command``::

        from nooa.skill import Skill, slash_command

        class MySkill(Skill):
            @slash_command("check", argument_hint="<target>", completions=("deps", "lint", "tests"))
            async def check_command(self, args: str) -> str:
                '''Run checks on the project.'''
                # Do work...
                return (
                    f"Ran {args} checks. Found 3 issues.\n\n"
                    "Fix them with `await self.shell.run(...)` or ask the user for guidance."
                )

    Key points:

    - **Return value = agent prompt.** Don't just dump data — frame it.
      Tell the agent what the data means and what actions are available.
    - **``completions``** — tuple of subcommand names for tab-completion.
      The TUI autocompletes ``/check <tab>`` → ``deps``, ``lint``, ``tests``.
    - **``argument_hint``** — shown in ``/help`` and tab-completion display.
    - **``output_to_agent=False``** — show the command's output to the user
      in the TUI only, without spending an agent turn (useful for read-only
      list/status commands).
    - Slash commands are discovered automatically when a skill is activated
      or hot-reloaded. No manual registration needed.

    ## Lifecycle

        # A text skill: instructions, saved and active at once
        await self.skill_writing.create("release", "How to cut a release.", text=steps)

        # A code skill: 1. scaffold the package
        await self.skill_writing.create("stats", "Statistical utilities.")

        # 2. Write code using self.shell
        await self.shell.write_file(self.skill_writing.path("stats", "stats.py"), source)

        # 3. Lint + reload
        await self.skill_writing.reload("stats")
        # -> lints, hot-reloads; self.stats is now available

        # 4. Edit using self.shell (replace(path, old, new); old must be unique)
        await self.shell.replace(self.skill_writing.path("stats", "stats.py"), old, new)
        await self.skill_writing.reload("stats")

        # 5. Test
        await self.skill_writing.run_tests("stats")

    ## Where skills are saved, and later sessions

    Each skill is a directory ``self.skill_writing.path(name)`` kept on disk
    across sessions: ``<workspace>/.nooa/skills`` for an Atom agent, else
    ``.nooa/skills`` under the project root (or ``$NEMO_OO_PROJECT_DIR/skills``).
    Skills in an older ``.nooa/libs`` directory still load; a name in
    ``.nooa/skills`` wins. A later session loads each one at start as
    ``self.<name>`` (``local.<name>`` in ``self.skills.search``), with nothing
    to import or activate, unless the agent already has that attribute (such
    as ``shell``): that one keeps its attribute and is not loaded.
    ``await self.skill_writing.list()`` names them all.

    ## Discovery

        await self.skill_writing.list()       # all skill names
        await self.skill_writing.repo_tree()  # directory tree

    ## Rules
    - Always provide a meaningful description when calling create()
    - Code skills are plain Python — no agent `self`, no `...` bodies, no async
    - Always run run_tests() after creating or editing before claiming done
    - Write a skill for logic or advice worth reusing; use inline code for one-offs
    - Always write a ``Skill`` subclass in a code skill's ``__init__.py``
    - Slash command return values are prompts — tell the agent what happened
      and what it can do next, don't just dump raw output
    """

    requires = ("nemo.shell",)

    def __init__(self, agent: Any, path: Path, *, also_load: Sequence[Path] = ()) -> None:
        """Save new skills in ``path``; also load the skills in ``also_load`` (older dirs)."""
        self._agent = agent
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._also_load = [Path(p) for p in also_load if Path(p) != self._path]
        # The save directory first: a skill there wins over one of the same
        # name in an older directory (the first one registered is kept).
        roots = [self._path, *(p for p in self._also_load if p.is_dir())]
        for root in reversed(roots):
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
        # Discover and register skills via the SkillRegistry if the agent has
        # one (an Atom agent's SkillManager wraps it as ``registry``).
        registry = self._registry()
        if registry is not None:
            for root in roots:
                registry.discover_libs(root, agent_written=True)
            registry.activate(["local.*"])
        else:
            for root in roots:
                LibraryManager.install(self._agent, libs_dir=root)
        super().__init__()

    def _registry(self) -> Any:
        """The agent's SkillRegistry, or None."""
        skills = getattr(self._agent, "skills", None)
        if skills is None:
            return None
        return getattr(skills, "registry", skills)

    def _dir(self, lib_name: str) -> Path:
        """The directory of a skill: the save directory's, else an older one's that exists."""
        primary = self._path / lib_name
        if primary.is_dir():
            return primary
        for root in self._also_load:
            if (root / lib_name).is_dir():
                return root / lib_name
        return primary

    def path(self, lib_name: str, relative: str = "") -> str:
        """Return the absolute path for a file within a skill.

        Args:
            lib_name: Skill name.
            relative: Relative path within the skill (e.g. "stats.py").
                      If empty, returns the skill's root directory.

        Returns:
            Absolute path as a string (for self.shell.write_file/replace/read).
        """
        p = self._dir(lib_name)
        if relative:
            p = p / relative
        return str(p)

    async def list(self) -> list[str]:
        """Return sorted names of all saved skills (code: pyproject.toml; text: SKILL.md)."""
        names: set[str] = set()
        for root in (self._path, *self._also_load):
            if not root.is_dir():
                continue
            for p in root.iterdir():
                if p.is_dir() and any(
                    (p / f).is_file() for f in ("pyproject.toml", "SKILL.md", "skill.md")
                ):
                    names.add(p.name)
        return sorted(names)

    async def create(self, lib_name: str, description: str, *, text: str | None = None) -> str:
        """Create a skill that later sessions get, in .nooa/skills/<lib_name>.

        With ``text``: a text skill, SKILL.md (front matter name and
        description, then ``text``), saved and active at once as
        ``self.<lib_name>``. Without: a code skill scaffold (pyproject.toml +
        ``__init__.py`` with a Skill subclass); nothing is loaded yet. Write
        code with self.shell.write_file() then call self.skill_writing.reload()
        to lint and activate.

        Args:
            lib_name: Skill name, a module name for a code skill (e.g. "stats").
            description: One-line description; the code skill's module docstring.
            text: The text skill's instructions (Markdown).

        Returns:
            Confirmation string with the skill path.
        """
        lib_dir = self._path / lib_name
        lib_dir.mkdir(parents=True, exist_ok=True)
        first_line = description.strip().split("\n")[0]

        if text is not None:
            # JSON strings are valid YAML scalars: quotes and colons stay safe.
            skill_md = (
                f"---\nname: {json.dumps(lib_name)}\n"
                f"description: {json.dumps(first_line)}\n---\n\n{text.strip()}\n"
            )
            (lib_dir / "SKILL.md").write_text(skill_md, encoding="utf-8")
            registry = self._registry()
            if registry is not None:
                if f"local.{lib_name}" in registry.loaded():
                    await registry.reload(f"local.{lib_name}")
                else:
                    registry.discover_libs(self._path, agent_written=True)
                registry.activate([f"local.{lib_name}"])
            return f"Created text skill '{lib_name}' at {lib_dir / 'SKILL.md'}"

        pyproject = textwrap.dedent(f"""\
            [project]
            name = "{lib_name}"
            version = "0.1.0"
            description = "{first_line}"
            dependencies = []
        """)
        (lib_dir / "pyproject.toml").write_text(pyproject)

        # Scaffold a Skill subclass so doc(self.<lib>) shows methods.
        class_name = "".join(w.capitalize() for w in lib_name.split("_"))
        init_content = (
            f'""" {description}"""\n'
            f"from nooa.skill import Skill\n\n\n"
            f"class {class_name}(Skill):\n"
            f'    """{first_line}"""\n'
        )
        (lib_dir / "__init__.py").write_text(init_content)

        return f"Created skill '{lib_name}' at {lib_dir}"

    async def reload(self, lib_name: str) -> str:
        """Lint and hot-reload a skill (a text skill: re-read its SKILL.md).

        Parses all .py files in the skill for syntax/security errors,
        then reloads the skill into the SkillRegistry if clean.

        Args:
            lib_name: Skill name to reload.

        Returns:
            LintReport string describing the result.
        """
        lib_dir = self._dir(lib_name)
        if not lib_dir.is_dir():
            return f"Skill '{lib_name}' not found at {lib_dir}"

        # Lint all .py files (except __init__.py)
        all_errors: list[str] = []
        all_warnings: list[str] = []

        for py_file in lib_dir.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
            source = py_file.read_text()
            report = self._lint_source(source)
            rel = py_file.relative_to(lib_dir)
            all_errors.extend(f"{rel}: {e}" for e in report.errors)
            all_warnings.extend(f"{rel}: {w}" for w in report.warnings)

        combined = LintReport(errors=all_errors, warnings=all_warnings)
        if combined.errors:
            return str(combined)

        combined.written = True
        status = await self._do_reload(lib_name)
        combined.loaded = status.startswith("Reloaded")
        if not combined.loaded:
            combined.errors.append(status)
        return str(combined)

    async def _do_reload(self, lib_name: str) -> str:
        """Hot-reload a library, using SkillRegistry if available.

        Returns the registry's status string (starts with "Reloaded" on success).
        """
        registry = self._registry()
        if registry is not None:
            lib_dir = self._dir(lib_name)
            if (
                f"local.{lib_name}" not in registry.loaded()
                and not (lib_dir / "pyproject.toml").exists()
            ):
                # A text skill (SKILL.md only): register it from its directory.
                registry.discover_libs(lib_dir.parent, agent_written=True)
                registry.activate([f"local.{lib_name}"])
                if f"local.{lib_name}" in registry.loaded():
                    return f"Reloaded local.{lib_name}"
            try:
                return await registry.reload(f"local.{lib_name}")
            except KeyError:
                # First lint+reload of a brand-new library: it isn't registered
                # yet, so skills.reload() raises. Fall through to the manager,
                # which imports and attaches it from disk.
                pass
        mgr = LibraryManager(self._agent, self._dir(lib_name).parent)
        mgr._reload(lib_name)
        return f"Reloaded local.{lib_name}"

    async def repo_tree(self, directory: str = ".") -> str:
        """Show the directory tree of the skills root (or a subdirectory).

        Args:
            directory: Subdirectory to show (relative to the skills root, default ".").

        Returns:
            Tree output as a string.
        """
        root = (self._path / directory) if directory != "." else self._path
        if not root.exists():
            return f"{root} does not exist"
        lines = []
        for p in sorted(root.rglob("*")):
            rel = p.relative_to(root)
            depth = len(rel.parts) - 1
            lines.append("  " * depth + p.name + ("/" if p.is_dir() else ""))
        return "\n".join(lines)

    async def run_tests(self, lib_name: str) -> str:
        """Run pytest on a code skill's tests/ directory.

        Args:
            lib_name: Skill name.

        Returns:
            Pytest output as a string.
        """
        import shlex
        import sys as _sys

        shell = self._agent.shell
        lib_dir = self._dir(lib_name)
        tests_dir = lib_dir / "tests"
        result = await shell.run(
            f"PYTHONPATH={shlex.quote(str(lib_dir.parent))}${{PYTHONPATH:+:$PYTHONPATH}} "
            f"{shlex.quote(_sys.executable)} -m pytest {shlex.quote(str(tests_dir))} -v",
            timeout=60,
        )
        return str(result)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lint_source(self, source: str) -> LintReport:
        """Run SecurityValidator on library source code."""
        from nooa.runtime.code_validator import SecurityValidator, ValidationContext

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return LintReport(errors=[f"SyntaxError: {e}"])

        context = ValidationContext(
            code=source,
            restricted_imports=frozenset(),
            blocked_modules=frozenset(),
        )

        issues = SecurityValidator().validate(tree, context)

        errors: list[str] = []
        warnings: list[str] = []
        for issue in issues:
            msg = f"{issue.code}: {issue.message} (line {issue.line})"
            if issue.code == "E001":
                errors.append(msg)
            else:
                warnings.append(msg)

        return LintReport(errors=errors, warnings=warnings)

    def _lint_pyproject(self, deps: Sequence[str]) -> LintReport:
        """Check declared dependencies against agent's importable modules."""
        importable = self._importable_modules()
        warnings = [
            f"E002: '{dep}' is declared but not in agent's importable modules"
            for dep in deps
            if dep not in importable
        ]
        return LintReport(warnings=warnings)

    def _importable_modules(self) -> set[str]:
        """Get the set of module names visible in the agent's module namespace."""
        agent_module = inspect.getmodule(type(self._agent))
        ns = agent_module.__dict__ if agent_module else {}
        return {obj.__name__ for obj in ns.values() if isinstance(obj, types.ModuleType)}

    def _get_declared_deps(self, lib_name: str) -> set[str]:
        """Parse declared dependencies from the library's pyproject.toml."""
        pyproject_path = self._path / lib_name / "pyproject.toml"
        if not pyproject_path.exists():
            return set()
        return set(self._parse_pyproject_deps(pyproject_path.read_text()))

    def _parse_pyproject_deps(self, content: str) -> Sequence[str]:
        """Extract dependency names from pyproject.toml content."""
        m = re.search(r"dependencies\s*=\s*\[(.*?)\]", content, re.DOTALL)
        if not m:
            return []
        deps: list[str] = []
        for line in m.group(1).splitlines():
            m2 = re.search(r'["\'](.*?)["\']', line)
            if m2:
                dep_spec = m2.group(1)
                dep_name = re.split(r"[>=<!,;]", dep_spec)[0].strip()
                if dep_name:
                    deps.append(dep_name)
        return deps
