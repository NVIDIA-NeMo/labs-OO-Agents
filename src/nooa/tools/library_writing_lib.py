# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SkillWriting — scaffold, lint, and manage persistent skill libraries."""

import ast
import inspect
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


class SkillWriting(Skill):
    """Scaffold, lint, and manage persistent skill libraries.

    Libraries are standard Python packages stored at a project-local path.
    Use ``self.shell`` for all file I/O (write, edit, view, grep); this
    skill handles scaffolding, linting, hot-reload, and testing.

    ## Skill contract

    Every library's ``__init__.py`` should export a ``Skill`` subclass::

        '''Grep-style search across multiple repos.'''
        from nooa.skill import Skill

        class MultiGrep(Skill):
            '''Grep across a configured list of repos.'''
            def __init__(self):
                self._roots = [...]

            def search(self, pattern: str) -> list[str]:
                ...

    The SkillRegistry auto-attaches the Skill as ``self.<lib_name>``.
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

        # 1. Scaffold the package
        await self.libs.create("stats", "Statistical utilities.")

        # 2. Write code using self.shell
        await self.shell.write_file(self.libs.path("stats", "stats.py"), source)

        # 3. Lint + reload
        await self.libs.reload("stats")
        # -> lints, hot-reloads; self.stats is now available

        # 4. Edit using self.shell (replace(path, old, new); old must be unique)
        await self.shell.replace(self.libs.path("stats", "stats.py"), old, new)
        await self.libs.reload("stats")

        # 5. Test
        await self.libs.run_tests("stats")

    ## Discovery

        await self.libs.list()       # all library names
        await self.libs.repo_tree()  # directory tree

    ## Rules
    - Always provide a meaningful description when calling create()
    - Library code is plain Python — no agent `self`, no `...` bodies, no async
    - Always run run_tests() after creating or editing before claiming done
    - Use a library for logic worth naming and reusing; use inline code for one-offs
    - Always write a ``Skill`` subclass in ``__init__.py``
    - Slash command return values are prompts — tell the agent what happened
      and what it can do next, don't just dump raw output
    """

    requires = ("nemo.shell",)

    def __init__(self, agent: Any, path: Path) -> None:
        self._agent = agent
        self._path = path
        self._path.mkdir(parents=True, exist_ok=True)
        libs_str = str(self._path)
        if libs_str not in sys.path:
            sys.path.insert(0, libs_str)
        # Discover and register libs via SkillRegistry if available
        if hasattr(agent, "skills"):
            agent.skills.discover_libs(self._path)
            agent.skills.activate(["local.*"])
        else:
            LibraryManager.install(self._agent, libs_dir=self._path)
        super().__init__()

    def path(self, lib_name: str, relative: str = "") -> str:
        """Return the absolute path for a file within a library.

        Args:
            lib_name: Library name.
            relative: Relative path within the library (e.g. "stats.py").
                      If empty, returns the library root directory.

        Returns:
            Absolute path as a string (suitable for self.shell.write/edit/view).
        """
        p = self._path / lib_name
        if relative:
            p = p / relative
        return str(p)

    async def list(self) -> list[str]:
        """Return sorted names of all libraries (directories with a pyproject.toml)."""
        return sorted(
            p.parent.name for p in self._path.glob("*/pyproject.toml") if p.parent.is_dir()
        )

    async def create(self, lib_name: str, description: str) -> str:
        """Scaffold a new library: writes pyproject.toml + __init__.py.

        Nothing is loaded yet. Write code with self.shell.write_file() then
        call self.libs.reload() to lint and activate.

        Args:
            lib_name: Module name for the library (e.g. "stats").
            description: Human-readable description stored as the module docstring.

        Returns:
            Confirmation string with the library path.
        """
        lib_dir = self._path / lib_name
        lib_dir.mkdir(parents=True, exist_ok=True)

        first_line = description.strip().split("\n")[0]
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

        return f"Created library '{lib_name}' at {lib_dir}"

    async def reload(self, lib_name: str) -> str:
        """Lint and hot-reload a library.

        Parses all .py files in the library for syntax/security errors,
        then reloads the library into the SkillRegistry if clean.

        Args:
            lib_name: Library name to reload.

        Returns:
            LintReport string describing the result.
        """
        lib_dir = self._path / lib_name
        if not lib_dir.is_dir():
            return f"Library '{lib_name}' not found at {lib_dir}"

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
        if hasattr(self._agent, "skills"):
            try:
                return await self._agent.skills.reload(f"local.{lib_name}")
            except KeyError:
                # First lint+reload of a brand-new library: it isn't registered
                # yet, so skills.reload() raises. Fall through to the manager,
                # which imports and attaches it from disk.
                pass
        mgr = LibraryManager(self._agent, self._path)
        mgr._reload(lib_name)
        return f"Reloaded local.{lib_name}"

    async def repo_tree(self, directory: str = ".") -> str:
        """Show the directory tree of the libraries root (or a subdirectory).

        Args:
            directory: Subdirectory to show (relative to libs root, default ".").

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
        """Run pytest on the library's tests/ directory.

        Args:
            lib_name: Library name.

        Returns:
            Pytest output as a string.
        """
        import shlex
        import sys as _sys

        shell = self._agent.shell
        tests_dir = self._path / lib_name / "tests"
        result = await shell.run(
            f"PYTHONPATH={shlex.quote(str(self._path))}:$PYTHONPATH "
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
