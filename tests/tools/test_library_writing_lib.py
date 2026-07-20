# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SkillWriting, LibraryManager, and Skill(module) fallback."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from nooa.library_manager import LibraryManager
from nooa.skill import Skill
from nooa.tools.library_writing_lib import LintReport, SkillWriting

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeShell:
    """Minimal ShellTools mock for SkillWriting tests."""

    async def edit(self, path, old_str, new_str):
        from pathlib import Path

        from nooa.tools._results import EditResult

        p = Path(path)
        content = p.read_text()
        if content.count(old_str) != 1:
            return EditResult(path=path, diff="", success=False, error="not unique")
        p.write_text(content.replace(old_str, new_str, 1))
        return EditResult(path=path, diff=f"-{old_str}\n+{new_str}", success=True)

    async def grep(self, pattern, path=".", **kwargs):
        import subprocess

        from nooa.tools._results import SearchResult

        r = subprocess.run(
            ["grep", "-rn", pattern, path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [line for line in r.stdout.strip().split("\n") if line]
        return SearchResult(matches=lines, total_matches=len(lines))

    async def find(self, pattern, path=".", **kwargs):
        import subprocess

        from nooa.tools._results import SearchResult

        r = subprocess.run(
            ["find", path, "-name", pattern, "-not", "-path", "*/.*"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = sorted(line for line in r.stdout.strip().split("\n") if line)
        return SearchResult(matches=lines, total_matches=len(lines))

    async def run(self, command, timeout=120.0):
        import subprocess

        from nooa.tools._results import RunResult

        r = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return RunResult(stdout=r.stdout.strip(), stderr=r.stderr.strip(), returncode=r.returncode)


class _FakeAgent:
    """Minimal agent whose module is this test module."""

    runtime = SimpleNamespace()
    shell = _FakeShell()

    def __setattr__(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)

    def __getattr__(self, name: str) -> object:
        raise AttributeError(name)


def _make_agent() -> _FakeAgent:
    return _FakeAgent()


SIMPLE_SOURCE = """\
def add(a: int, b: int) -> int:
    return a + b
"""

DESCRIPTION = "Simple arithmetic library."


def _make_lib_dir(
    tmp_path: Path, lib_name: str, source: str, description: str = DESCRIPTION
) -> Path:
    """Create a minimal library directory: pyproject.toml + __init__.py + a .py module."""
    lib_dir = tmp_path / lib_name
    lib_dir.mkdir()
    pyproject = f'[project]\nname = "{lib_name}"\nversion = "0.1.0"\ndependencies = []\n'
    (lib_dir / "pyproject.toml").write_text(pyproject)
    (lib_dir / "__init__.py").write_text(f'"""{description}"""\nfrom .{lib_name} import *\n')
    (lib_dir / f"{lib_name}.py").write_text(source)
    if str(tmp_path) not in sys.path:
        sys.path.insert(0, str(tmp_path))
    return lib_dir


# ---------------------------------------------------------------------------
# LintReport
# ---------------------------------------------------------------------------


def test_lint_report_ok():
    assert str(LintReport(written=True, loaded=True)) == "OK — written and loaded"


def test_lint_report_not_written():
    s = str(LintReport(written=False, errors=["E001: exec() is forbidden (line 1)"]))
    assert "ERROR" in s
    assert "E001" in s


def test_lint_report_written_not_loaded():
    s = str(LintReport(written=True, loaded=False, warnings=["E002: 'numpy' not in allowed set"]))
    assert "WARNING" in s
    assert "E002" in s


# ---------------------------------------------------------------------------
# LibraryManager._import_module
# ---------------------------------------------------------------------------


def test_import_module_registers_in_sys_modules(tmp_path: Path):
    """_import_module imports the package and registers it in sys.modules."""
    lib_dir = _make_lib_dir(tmp_path, "ts_load", SIMPLE_SOURCE)
    agent = _make_agent()
    mgr = LibraryManager(agent, tmp_path)
    mgr._import_module(lib_dir)
    assert "ts_load" in sys.modules
    assert sys.modules["ts_load"].add(1, 2) == 3


def test_import_module_cache_busts(tmp_path: Path):
    """A second _import_module picks up changes made to the package on disk."""
    lib_dir = _make_lib_dir(tmp_path, "ts_reload", SIMPLE_SOURCE)
    agent = _make_agent()
    mgr = LibraryManager(agent, tmp_path)
    mgr._import_module(lib_dir)

    (lib_dir / "ts_reload.py").write_text("def add(a, b): return a + b + 99\n")
    mgr._import_module(lib_dir)

    assert sys.modules["ts_reload"].add(0, 0) == 99


def test_skill_module_fallback_has_module_dir(tmp_path: Path):
    """Skill(module) fallback exposes module names via __dir__."""
    _make_lib_dir(tmp_path, "ts_dir", SIMPLE_SOURCE)
    agent = _make_agent()
    LibraryManager.install(agent, libs_dir=tmp_path)
    assert "add" in dir(agent.ts_dir)


def test_skill_module_fallback_description(tmp_path: Path):
    """Skill(module) fallback uses the module docstring."""
    _make_lib_dir(tmp_path, "ts_desc", SIMPLE_SOURCE, description="My description.")
    agent = _make_agent()
    LibraryManager.install(agent, libs_dir=tmp_path)
    assert "My description." in (type(agent.ts_desc).__doc__ or "")


# ---------------------------------------------------------------------------
# LibraryManager
# ---------------------------------------------------------------------------


def test_library_manager_discover_finds_pyproject_dirs(tmp_path: Path):
    """discover() returns names of directories that contain a pyproject.toml."""
    _make_lib_dir(tmp_path, "lm_a", SIMPLE_SOURCE)
    _make_lib_dir(tmp_path, "lm_b", SIMPLE_SOURCE)
    (tmp_path / "no_pyproject").mkdir()  # no pyproject.toml — must not appear

    assert LibraryManager.discover(tmp_path) == ["lm_a", "lm_b"]


def test_library_manager_discover_empty(tmp_path: Path):
    """discover() returns an empty list when no libraries exist."""
    assert LibraryManager.discover(tmp_path) == []


def test_library_manager_install_attaches_libraries(tmp_path: Path):
    """install() attaches each library as a Skill attribute on the agent."""
    _make_lib_dir(tmp_path, "lm_install", SIMPLE_SOURCE)
    agent = _make_agent()
    LibraryManager.install(agent, libs_dir=tmp_path)
    assert hasattr(agent, "lm_install")
    assert isinstance(agent.lm_install, Skill)


def test_library_manager_install_skips_missing_dir(tmp_path: Path):
    """install() does nothing when the libs directory does not exist."""
    agent = _make_agent()
    mgr = LibraryManager.install(agent, libs_dir=tmp_path / "nonexistent")
    assert mgr._installed == []


def test_library_manager_install_skips_existing_attrs(tmp_path: Path):
    """install() does not overwrite an attribute already present on the agent."""
    _make_lib_dir(tmp_path, "lm_skip", SIMPLE_SOURCE)
    agent = _make_agent()
    sentinel = object()
    agent.lm_skip = sentinel  # type: ignore[attr-defined]
    LibraryManager.install(agent, libs_dir=tmp_path)
    assert agent.lm_skip is sentinel


def test_library_manager_install_skips_bad_library(tmp_path: Path):
    """install() logs a warning and continues when a library fails to import."""
    bad = tmp_path / "lm_bad"
    bad.mkdir()
    (bad / "pyproject.toml").write_text('[project]\nname = "lm_bad"\n')
    (bad / "__init__.py").write_text("raise RuntimeError('broken')\n")
    _make_lib_dir(tmp_path, "lm_good", SIMPLE_SOURCE)

    agent = _make_agent()
    mgr = LibraryManager.install(agent, libs_dir=tmp_path)
    assert "lm_good" in mgr._installed
    assert "lm_bad" not in mgr._installed


def test_library_manager_reload(tmp_path: Path):
    """reload() re-imports the library and the agent sees updated code."""
    lib_dir = _make_lib_dir(tmp_path, "lm_reload", SIMPLE_SOURCE)
    agent = _make_agent()
    mgr = LibraryManager.install(agent, libs_dir=tmp_path)

    (lib_dir / "lm_reload.py").write_text("def add(a, b): return a + b + 100\n")
    mgr._reload("lm_reload")

    assert sys.modules["lm_reload"].add(1, 2) == 103


def test_library_manager_reload_all(tmp_path: Path):
    """reload_all() reloads every installed library."""
    dir_a = _make_lib_dir(tmp_path, "lm_all_a", SIMPLE_SOURCE)
    dir_b = _make_lib_dir(tmp_path, "lm_all_b", SIMPLE_SOURCE)
    agent = _make_agent()
    mgr = LibraryManager.install(agent, libs_dir=tmp_path)

    (dir_a / "lm_all_a.py").write_text("def add(a, b): return a + b + 10\n")
    (dir_b / "lm_all_b.py").write_text("def add(a, b): return a + b + 20\n")
    mgr.reload()

    assert sys.modules["lm_all_a"].add(0, 0) == 10
    assert sys.modules["lm_all_b"].add(0, 0) == 20


# ---------------------------------------------------------------------------
# SkillWriting.create
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_writes_pyproject_and_init(tmp_path: Path):
    """create() writes pyproject.toml (with name/version) and __init__.py (with docstring)."""
    libs = SkillWriting(_make_agent(), path=tmp_path)
    await libs.create("mylib", DESCRIPTION)

    pyproject = (tmp_path / "mylib" / "pyproject.toml").read_text()
    assert 'name = "mylib"' in pyproject
    assert 'version = "0.1.0"' in pyproject

    init_py = (tmp_path / "mylib" / "__init__.py").read_text()
    assert DESCRIPTION in init_py
    assert "from nooa.skill import Skill" in init_py
    assert "class Mylib(Skill):" in init_py


@pytest.mark.asyncio
async def test_create_does_not_write_source_file(tmp_path: Path):
    """create() only scaffolds the package — no code files are written."""
    libs = SkillWriting(_make_agent(), path=tmp_path)
    await libs.create("mylib", DESCRIPTION)
    py_files = [p for p in (tmp_path / "mylib").iterdir() if p.suffix == ".py"]
    assert py_files == [tmp_path / "mylib" / "__init__.py"]


# ---------------------------------------------------------------------------
# SkillWriting.list
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_returns_lib_names(tmp_path: Path):
    """list() returns sorted names of libraries detected by pyproject.toml."""
    libs = SkillWriting(_make_agent(), path=tmp_path)
    await libs.create("alpha", "Alpha lib")
    await libs.create("beta", "Beta lib")
    (tmp_path / "empty_dir").mkdir()  # no pyproject.toml — must not appear
    assert await libs.list() == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_list_empty_when_no_libs(tmp_path: Path):
    libs = SkillWriting(_make_agent(), path=tmp_path)
    assert (
        await libs.list() == []
    )  # ---------------------------------------------------------------------------


# SkillWriting.view_file
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SkillWriting.run_tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_tests_passes(tmp_path: Path):
    """run_tests() on a library with a passing test returns output containing 'passed'."""
    libs = SkillWriting(_make_agent(), path=tmp_path)
    await libs.create("rt_math", DESCRIPTION)
    (tmp_path / "rt_math" / "rt_math.py").write_text(SIMPLE_SOURCE)
    tests_dir = tmp_path / "rt_math" / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_rt_math.py").write_text(
        "from rt_math.rt_math import add\ndef test_add(): assert add(1, 2) == 3\n"
    )
    output = await libs.run_tests("rt_math")
    assert "passed" in output.lower(), output


# ---------------------------------------------------------------------------
# SkillWriting.reload — failure reporting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reload_reports_registry_failure(tmp_path: Path):
    """A reload that the registry rejects must NOT be reported as loaded."""

    class _FailingSkills:
        def discover_libs(self, path):
            pass

        def activate(self, patterns):
            pass

        async def reload(self, name: str) -> str:
            return f"Reload failed for {name}: boom"

    agent = _make_agent()
    agent.skills = _FailingSkills()
    libs = SkillWriting(agent, path=tmp_path)
    await libs.create("rl_fail", DESCRIPTION)
    (tmp_path / "rl_fail" / "rl_fail.py").write_text(SIMPLE_SOURCE)

    result = await libs.reload("rl_fail")

    assert "OK — written and loaded" not in result
    assert "not loaded" in result.lower()
    assert "boom" in result


@pytest.mark.asyncio
async def test_reload_reports_success(tmp_path: Path):
    """A reload the registry accepts is reported as loaded."""

    class _OkSkills:
        def discover_libs(self, path):
            pass

        def activate(self, patterns):
            pass

        async def reload(self, name: str) -> str:
            return f"Reloaded {name} (self.rl_ok)"

    agent = _make_agent()
    agent.skills = _OkSkills()
    libs = SkillWriting(agent, path=tmp_path)
    await libs.create("rl_ok", DESCRIPTION)
    (tmp_path / "rl_ok" / "rl_ok.py").write_text(SIMPLE_SOURCE)

    result = await libs.reload("rl_ok")

    assert "OK — written and loaded" in result
