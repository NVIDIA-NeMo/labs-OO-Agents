# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SkillRegistry — file-based discovery, deps, libs, reload."""

import sys
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nooa.skill import Skill
from nooa.skill_registry import SkillRegistry


class FakeSkill(Skill):
    """A test skill."""

    requires: tuple[str, ...] = ()


class DepSkill(Skill):
    """Skill that declares a dependency."""

    requires = ("nemo.base",)


class _FakeAgent:
    pass


@pytest.fixture
def agent():
    return _FakeAgent()


@pytest.fixture
def registry(agent):
    with patch("nooa.skill_registry.entry_points", return_value=[]):
        value = SkillRegistry(agent)
        yield value
        value.close()


# ---------------------------------------------------------------------------
# Tests: discover_skills_dirs
# ---------------------------------------------------------------------------


class TestDiscoverSkillsDirs:
    def test_packaged_and_text_skills_are_discovered_in_one_call(self, registry, tmp_path):
        lib_dir = tmp_path / "workflow_lib"
        lib_dir.mkdir()
        (lib_dir / "pyproject.toml").write_text(
            '[project]\nname = "workflow-lib"\n\n'
            '[project.entry-points."nooa.skills"]\n'
            '"nvzurich.workflow" = "workflow_lib:WorkflowSkill"\n'
        )
        (lib_dir / "__init__.py").write_text(
            "from nooa.skill import Skill\n\nclass WorkflowSkill(Skill):\n    pass\n"
        )
        text_dir = tmp_path / "root-cause"
        text_dir.mkdir()
        (text_dir / "SKILL.md").write_text(
            "---\nname: root-cause\ndescription: Diagnose a defect\n---\nFind the cause.\n"
        )

        registry.discover_skills_dirs([tmp_path])

        assert "nvzurich.workflow" in registry.loaded()
        assert "cmd.root-cause" in registry.loaded()

    def test_python_skill_file_discovered(self, registry, agent, tmp_path):
        """A .py file with a Skill subclass is discovered as ext.<name>."""
        skill_file = tmp_path / "my_tool.py"
        skill_file.write_text(
            textwrap.dedent("""
            from nooa.skill import Skill

            class MyTool(Skill):
                \"\"\"A custom tool.\"\"\"
                pass
        """)
        )
        registry.discover_skills_dirs([tmp_path])
        assert "ext.my_tool" in registry.loaded()
        assert hasattr(agent, "my_tool")

    @pytest.mark.asyncio
    async def test_standalone_python_skill_detaches_and_releases_module(self, tmp_path):
        marker = tmp_path / "detached"
        skill_file = tmp_path / "resource.py"
        skill_file.write_text(
            "from pathlib import Path\n"
            "from nooa.skill import Skill\n\n"
            "class ResourceSkill(Skill):\n"
            "    def detach(self):\n"
            f"        Path({str(marker)!r}).write_text('yes')\n"
            "        super().detach()\n"
        )
        value = SkillRegistry(_FakeAgent())
        value.discover_skills_dirs([tmp_path])
        module_name = type(value["ext.resource"]).__module__

        assert module_name in sys.modules
        await value.aclose()

        assert marker.read_text() == "yes"
        assert module_name not in sys.modules

    @pytest.mark.asyncio
    @pytest.mark.parametrize("first_to_close", [0, 1])
    async def test_standalone_python_modules_are_isolated_per_live_registry(
        self, tmp_path, first_to_close
    ):
        skill_file = tmp_path / "isolated.py"
        skill_file.write_text(
            "from nooa.skill import Skill\n\nclass IsolatedSkill(Skill):\n    value = 'live'\n"
        )
        registries = [SkillRegistry(_FakeAgent()), SkillRegistry(_FakeAgent())]
        for value in registries:
            value.discover_skills_dirs([tmp_path])
        modules = [type(value["ext.isolated"]).__module__ for value in registries]

        assert modules[0] != modules[1]
        await registries[first_to_close].aclose()
        remaining = 1 - first_to_close
        assert modules[first_to_close] not in sys.modules
        assert modules[remaining] in sys.modules
        assert registries[remaining]["ext.isolated"].value == "live"

        await registries[remaining].aclose()
        assert modules[remaining] not in sys.modules

    def test_underscore_files_skipped(self, registry, tmp_path):
        """Files starting with _ are not loaded."""
        skill_file = tmp_path / "_private.py"
        skill_file.write_text(
            textwrap.dedent("""
            from nooa.skill import Skill
            class Priv(Skill): pass
        """)
        )
        registry.discover_skills_dirs([tmp_path])
        assert "ext._private" not in registry.loaded()

    def test_text_skill_discovered(self, registry, agent, tmp_path):
        """A directory with SKILL.md is discovered as cmd.<id>."""
        skill_dir = tmp_path / "my-cmd"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: my-cmd\ndescription: A test command\n---\nDo the thing.\n"
        )
        registry.discover_skills_dirs([tmp_path])
        assert "cmd.my-cmd" in registry.loaded()

    def test_nonexistent_dir_skipped(self, registry):
        """Non-existent directories are silently skipped."""
        registry.discover_skills_dirs([Path("/nonexistent/path")])
        # Should not raise

    def test_broken_python_file_skipped(self, registry, tmp_path):
        """A .py file that fails to import is skipped with warning."""
        skill_file = tmp_path / "broken.py"
        skill_file.write_text("raise RuntimeError('boom')")
        registry.discover_skills_dirs([tmp_path])
        assert "ext.broken" not in registry.loaded()


# ---------------------------------------------------------------------------
# Tests: discover_libs
# ---------------------------------------------------------------------------


class TestDiscoverLibs:
    def test_agents_load_local_objects_from_different_checkouts(self, tmp_path):
        def write_checkout(root: Path, value: str) -> Path:
            lib_dir = root / "workflow-checkout"
            package = lib_dir / "src" / "shared_workflow"
            package.mkdir(parents=True)
            (lib_dir / "pyproject.toml").write_text(
                '[project]\nname = "workflow-distribution"\n\n'
                '[project.entry-points."nooa.skills"]\n'
                '"test.workflow" = "shared_workflow:WorkflowSkill"\n'
            )
            (package / "__init__.py").write_text(
                "from nooa.skill import Skill\n\n"
                "class WorkflowSkill(Skill):\n"
                f"    value = {value!r}\n"
            )
            return root

        first_root = write_checkout(tmp_path / "first", "first")
        second_root = write_checkout(tmp_path / "second", "second")
        first = SkillRegistry(_FakeAgent())
        second = SkillRegistry(_FakeAgent())
        try:
            first.discover_libs(first_root)
            second.discover_libs(second_root)

            assert first["test.workflow"].value == "first"
            assert second["test.workflow"].value == "second"
            assert first["test.workflow"] is not second["test.workflow"]
        finally:
            first.close()
            second.close()

    def test_same_checkout_creates_distinct_agent_local_objects(self, tmp_path):
        lib_dir = tmp_path / "workflow-checkout"
        package = lib_dir / "src" / "reference_workflow"
        package.mkdir(parents=True)
        (lib_dir / "pyproject.toml").write_text(
            '[project]\nname = "workflow"\n\n'
            '[project.entry-points."nooa.skills"]\n'
            '"test.workflow" = "reference_workflow:WorkflowSkill"\n'
        )
        (package / "__init__.py").write_text(
            "from nooa.skill import Skill\n\nclass WorkflowSkill(Skill):\n    value = 'shared'\n"
        )
        first = SkillRegistry(_FakeAgent())
        second = SkillRegistry(_FakeAgent())
        try:
            first.discover_libs(tmp_path)
            second.discover_libs(tmp_path)

            assert first["test.workflow"] is not second["test.workflow"]
            first.close()
            assert second["test.workflow"].value == "shared"
        finally:
            first.close()
            second.close()

    @pytest.mark.asyncio
    async def test_each_agent_reloads_its_own_package_object(self, tmp_path):
        lib_dir = tmp_path / "shared-checkout"
        package = lib_dir / "src" / "shared_reload_workflow"
        package.mkdir(parents=True)
        (lib_dir / "pyproject.toml").write_text(
            '[project]\nname = "shared-reload"\n\n'
            '[project.entry-points."nooa.skills"]\n'
            '"test.shared" = "shared_reload_workflow:SharedSkill"\n'
        )
        (package / "__init__.py").write_text(
            "from nooa.skill import Skill\nclass SharedSkill(Skill):\n    value = 'old'\n"
        )
        first = SkillRegistry(_FakeAgent())
        second = SkillRegistry(_FakeAgent())
        first.discover_libs(tmp_path)
        second.discover_libs(tmp_path)
        try:
            assert first["test.shared"].value == "old"
            assert second["test.shared"].value == "old"

            module = package / "__init__.py"
            module.write_text(
                "from nooa.skill import Skill\n"
                "class SharedSkill(Skill):\n    value = 'new-version'\n"
            )
            stat = module.stat()
            import os

            os.utime(module, (stat.st_atime + 2, stat.st_mtime + 2))

            assert await first.reload("test.shared") == "Reloaded test.shared (self.shared)"
            assert first["test.shared"].value == "new-version"
            assert second["test.shared"].value == "old"

            assert await second.reload("test.shared") == "Reloaded test.shared (self.shared)"
            assert first["test.shared"].value == "new-version"
            assert second["test.shared"].value == "new-version"
            assert first["test.shared"] is not second["test.shared"]
        finally:
            await first.aclose()
            await second.aclose()

    def test_direct_module_entry_point_layout_is_importable(self, registry, tmp_path):
        lib_dir = tmp_path / "workflow-checkout"
        lib_dir.mkdir()
        (lib_dir / "pyproject.toml").write_text(
            '[project]\nname = "direct-workflow"\n\n'
            '[project.entry-points."nooa.skills"]\n'
            '"test.direct" = "direct_workflow:DirectSkill"\n'
        )
        (lib_dir / "direct_workflow.py").write_text(
            "from nooa.skill import Skill\nclass DirectSkill(Skill):\n    value = 'direct'\n"
        )

        registry.discover_libs(tmp_path)

        assert registry["test.direct"].value == "direct"

    def test_package_with_same_named_module_preserves_package_precedence(self, registry, tmp_path):
        lib_dir = tmp_path / "worktrees"
        lib_dir.mkdir()
        (lib_dir / "pyproject.toml").write_text(
            '[project]\nname = "worktrees"\n\n'
            '[project.entry-points."nooa.skills"]\n'
            '"test.worktrees" = "worktrees.worktrees:Worktrees"\n'
        )
        (lib_dir / "__init__.py").write_text("")
        (lib_dir / "worktrees.py").write_text(
            "from nooa.skill import Skill\nclass Worktrees(Skill):\n    value = 'package'\n"
        )

        registry.discover_libs(tmp_path)

        assert registry["test.worktrees"].value == "package"
        assert type(registry["test.worktrees"]).__module__ == "worktrees.worktrees"

    def test_entry_point_target_controls_import_package_and_class(self, registry, agent, tmp_path):
        """Checkout, distribution, module, and Skill class names may all differ."""
        lib_dir = tmp_path / "workflow-checkout"
        package = lib_dir / "src" / "actual_workflow" / "commands"
        package.mkdir(parents=True)
        (lib_dir / "pyproject.toml").write_text(
            '[project]\nname = "workflow-distribution"\n\n'
            '[project.entry-points."nooa.skills"]\n'
            '"nvzurich.workflow" = "actual_workflow.commands:WorkflowSkill"\n'
        )
        (package.parent / "__init__.py").write_text("")
        (package / "__init__.py").write_text(
            "from nooa.skill import Skill\n\nclass WorkflowSkill(Skill):\n    pass\n"
        )

        registry.discover_libs(tmp_path)

        assert "nvzurich.workflow" in registry.loaded()
        assert type(registry["nvzurich.workflow"]).__module__ == "actual_workflow.commands"
        assert agent.workflow is registry["nvzurich.workflow"]

    def test_lib_with_pyproject_discovered(self, registry, agent, tmp_path):
        """A library with pyproject.toml and Skill subclass is registered."""
        lib_dir = tmp_path / "my_lib"
        lib_dir.mkdir()
        (lib_dir / "pyproject.toml").write_text(
            '[project]\nname = "my-lib"\n\n'
            '[project.entry-points."nooa.skills"]\n'
            '"local.my_lib" = "my_lib:MyLibSkill"\n'
        )
        (lib_dir / "__init__.py").write_text(
            'from nooa.skill import Skill\n\nclass MyLibSkill(Skill):\n    """A library skill."""\n'
        )
        registry.discover_libs(tmp_path)
        assert "local.my_lib" in registry.loaded()

    def test_dir_without_pyproject_skipped(self, registry, tmp_path):
        """Directories without pyproject.toml are skipped."""
        lib_dir = tmp_path / "no_pyproject"
        lib_dir.mkdir()
        (lib_dir / "__init__.py").write_text("x = 1")
        registry.discover_libs(tmp_path)
        assert registry.loaded() == []

    def test_nonexistent_libs_path(self, registry):
        """Non-existent libs_path is silently handled."""
        registry.discover_libs(Path("/nonexistent"))
        # Should not raise

    def test_already_loaded_lib_skipped(self, registry, agent, tmp_path):
        """A library already loaded is not re-imported."""
        lib_dir = tmp_path / "dup_lib"
        lib_dir.mkdir()
        (lib_dir / "pyproject.toml").write_text(
            textwrap.dedent("""
            [project]
            name = "dup-lib"
        """)
        )
        (lib_dir / "__init__.py").write_text(
            textwrap.dedent("""
            from nooa.skill import Skill
            class DupSkill(Skill): pass
        """)
        )
        # Pre-register to simulate already loaded
        registry.register("local.dup_lib", FakeSkill())
        registry.discover_libs(tmp_path)
        # Should not have re-loaded (still the FakeSkill instance)
        assert isinstance(agent.dup_lib, FakeSkill)


# ---------------------------------------------------------------------------
# Tests: _resolve_deps
# ---------------------------------------------------------------------------


class TestResolveDeps:
    def test_resolves_single_dependency(self, agent):
        """A skill with requires=('nemo.base',) triggers loading of its dep."""
        ep_base = MagicMock()
        ep_base.name = "nemo.base"
        ep_base.load.return_value = FakeSkill

        with patch("nooa.skill_registry.entry_points", return_value=[ep_base]):
            reg = SkillRegistry(agent)

        dep_skill = DepSkill()
        reg.register("nemo.dep", dep_skill)
        reg._resolve_deps("nemo.dep")
        assert "nemo.base" in reg.loaded()

    def test_cycle_detection(self, registry, agent):
        """Circular dependencies don't infinite-loop."""

        class CycleA(Skill):
            requires = ("nemo.cycle_b",)

        class CycleB(Skill):
            requires = ("nemo.cycle_a",)

        registry.register("nemo.cycle_a", CycleA())
        registry.register("nemo.cycle_b", CycleB())
        # Should not raise or loop forever
        registry._resolve_deps("nemo.cycle_a")

    def test_missing_dep_warns(self, registry, agent):
        """Missing dependency logs a warning but doesn't crash."""
        skill = DepSkill()  # requires ('nemo.base',)
        registry.register("nemo.needy", skill)
        # nemo.base is not discovered — should warn, not crash
        registry._resolve_deps("nemo.needy")
        assert "nemo.base" not in registry.loaded()


class TestDeterministicLoadOrder:
    def test_load_sorts_unrelated_matches(self):
        loaded: list[str] = []

        class RecordingEntryPoint:
            def __init__(self, name: str):
                self.name = name

            def load(self):
                loaded.append(self.name)
                return FakeSkill

        entries = [
            RecordingEntryPoint("test.zulu"),
            RecordingEntryPoint("test.alpha"),
        ]
        with patch("nooa.skill_registry.entry_points", return_value=entries):
            value = SkillRegistry(_FakeAgent())
        try:
            value.load(["test.*"])
            assert loaded == ["test.alpha", "test.zulu"]
        finally:
            value.close()


# ---------------------------------------------------------------------------
# Tests: reload
# ---------------------------------------------------------------------------


class TestReload:
    @pytest.mark.asyncio
    async def test_each_agent_reloads_its_own_text_skill(self, tmp_path):
        skill_dir = tmp_path / "demo"
        skill_dir.mkdir()
        skill_md = skill_dir / "SKILL.md"
        skill_md.write_text("---\nname: demo\ndescription: old description\n---\nold body\n")
        first = SkillRegistry(_FakeAgent())
        second = SkillRegistry(_FakeAgent())
        first.discover_skills_dirs([tmp_path])
        second.discover_skills_dirs([tmp_path])
        try:
            skill_md.write_text("---\nname: demo\ndescription: new description\n---\nnew body\n")

            assert await first.reload("cmd.demo") == "Reloaded cmd.demo (self.demo)"
            assert first["cmd.demo"].description == "new description"
            assert second["cmd.demo"].description == "old description"

            assert await second.reload("cmd.demo") == "Reloaded cmd.demo (self.demo)"
            assert second["cmd.demo"].description == "new description"
            assert first["cmd.demo"] is not second["cmd.demo"]
        finally:
            await first.aclose()
            await second.aclose()

    @pytest.mark.asyncio
    async def test_each_agent_reloads_its_own_standalone_python_skill(self, tmp_path):
        import os

        skill_file = tmp_path / "demo.py"
        skill_file.write_text(
            "from nooa.skill import Skill\nclass Demo(Skill):\n    value = 'old'\n"
        )
        first = SkillRegistry(_FakeAgent())
        second = SkillRegistry(_FakeAgent())
        first.discover_skills_dirs([tmp_path])
        second.discover_skills_dirs([tmp_path])
        first_module = type(first["ext.demo"]).__module__
        second_module = type(second["ext.demo"]).__module__
        try:
            skill_file.write_text(
                "from nooa.skill import Skill\nclass Demo(Skill):\n    value = 'new'\n"
            )
            stat = skill_file.stat()
            os.utime(skill_file, (stat.st_atime + 2, stat.st_mtime + 2))

            assert await first.reload("ext.demo") == "Reloaded ext.demo (self.demo)"
            assert first["ext.demo"].value == "new"
            assert second["ext.demo"].value == "old"
            assert first_module not in sys.modules
            assert second_module in sys.modules

            assert await second.reload("ext.demo") == "Reloaded ext.demo (self.demo)"
            assert second["ext.demo"].value == "new"
            assert second_module not in sys.modules
            assert first["ext.demo"] is not second["ext.demo"]
        finally:
            await first.aclose()
            await second.aclose()

    @pytest.mark.asyncio
    async def test_nested_entry_point_module_reloads_its_declared_skill(self, registry, tmp_path):
        lib_dir = tmp_path / "nested-checkout"
        package = lib_dir / "src" / "nested_reload_workflow"
        package.mkdir(parents=True)
        (lib_dir / "pyproject.toml").write_text(
            '[project]\nname = "nested-reload"\n\n'
            '[project.entry-points."nooa.skills"]\n'
            '"test.nested" = "nested_reload_workflow.commands:NestedSkill"\n'
        )
        (package / "__init__.py").write_text("")
        commands = package / "commands.py"
        commands.write_text(
            "from nooa.skill import Skill\nclass NestedSkill(Skill):\n    value = 'old'\n"
        )
        registry.discover_libs(tmp_path)
        assert registry["test.nested"].value == "old"
        commands.write_text(
            "from nooa.skill import Skill\nclass NestedSkill(Skill):\n    value = 'new-version'\n"
        )
        stat = commands.stat()
        commands.touch()
        import os

        os.utime(commands, (stat.st_atime + 2, stat.st_mtime + 2))

        result = await registry.reload("test.nested")

        assert result == "Reloaded test.nested (self.nested)"
        assert registry["test.nested"].value == "new-version"

    @pytest.mark.asyncio
    async def test_shutdown_detaches_later_skills_before_their_dependencies(self):
        seen: list[str] = []

        class Dependency(Skill):
            def detach(self):
                seen.append("dependency")
                super().detach()

        class Dependent(Skill):
            def detach(self):
                assert hasattr(self._agent, "dependency")
                seen.append("dependent")
                super().detach()

        agent = _FakeAgent()
        value = SkillRegistry(agent)
        value.register("test.dependency", Dependency())
        value.register("test.dependent", Dependent())

        await value.aclose()

        assert seen == ["dependent", "dependency"]

    @pytest.mark.asyncio
    async def test_shutdown_detaches_auto_loaded_dependent_before_requirement(self):
        seen: list[str] = []

        class Requirement(Skill):
            def detach(self):
                seen.append("requirement")
                super().detach()

        class Dependent(Skill):
            requires = ("test.requirement",)

            def detach(self):
                assert hasattr(self._agent, "requirement")
                seen.append("dependent")
                super().detach()

        requirement_ep = MagicMock()
        requirement_ep.name = "test.requirement"
        requirement_ep.load.return_value = Requirement
        agent = _FakeAgent()
        with patch("nooa.skill_registry.entry_points", return_value=[requirement_ep]):
            value = SkillRegistry(agent)
        value.register("test.dependent", Dependent())
        value.activate(["test.dependent"])
        assert value._load_order == ["test.dependent", "test.requirement"]

        await value.aclose()

        assert seen == ["dependent", "requirement"]

    @pytest.mark.asyncio
    async def test_failed_package_reload_restores_lazy_submodule_imports(self, registry, tmp_path):
        lib_dir = tmp_path / "lazy-checkout"
        package = lib_dir / "src" / "lazy_reload_workflow"
        package.mkdir(parents=True)
        (lib_dir / "pyproject.toml").write_text(
            '[project]\nname = "lazy-reload"\n\n'
            '[project.entry-points."nooa.skills"]\n'
            '"test.lazy" = "lazy_reload_workflow:LazySkill"\n'
        )
        init = package / "__init__.py"
        init.write_text(
            "from nooa.skill import Skill\n\n"
            "class LazySkill(Skill):\n"
            "    def value(self):\n"
            "        from lazy_reload_workflow.helper import VALUE\n"
            "        return VALUE\n"
        )
        (package / "helper.py").write_text("VALUE = 'still-works'\n")
        registry.discover_libs(tmp_path)
        old_skill = registry["test.lazy"]
        init.write_text("this is invalid Python !!!\n")

        result = await registry.reload("test.lazy")

        assert result.startswith("Reload failed for test.lazy:")
        assert registry["test.lazy"] is old_skill
        assert old_skill.value() == "still-works"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("constructor", ["runtime-error", "requires-argument"])
    async def test_failed_skill_swap_restores_previous_package_tree(
        self, registry, tmp_path, constructor
    ):
        lib_dir = tmp_path / "swap-checkout"
        package = lib_dir / "src" / "swap_reload_workflow"
        package.mkdir(parents=True)
        (lib_dir / "pyproject.toml").write_text(
            '[project]\nname = "swap-reload"\n\n'
            '[project.entry-points."nooa.skills"]\n'
            '"test.swap" = "swap_reload_workflow:SwapSkill"\n'
        )
        init = package / "__init__.py"
        init.write_text(
            "from nooa.skill import Skill\n\n"
            "class SwapSkill(Skill):\n"
            "    def value(self):\n"
            "        from swap_reload_workflow.helper import VALUE\n"
            "        return VALUE\n"
        )
        helper = package / "helper.py"
        helper.write_text("VALUE = 'old-code'\n")
        registry.discover_libs(tmp_path)
        old_skill = registry["test.swap"]
        assert old_skill.value() == "old-code"
        old_package = sys.modules["swap_reload_workflow"]
        old_helper = sys.modules["swap_reload_workflow.helper"]
        helper.write_text("VALUE = 'replacement-code'\n")
        if constructor == "runtime-error":
            failed_constructor = (
                "    def __init__(self):\n        raise RuntimeError('new constructor failed')\n"
            )
        else:
            failed_constructor = (
                "    def __init__(self, required):\n        self.required = required\n"
            )
        init.write_text(
            "from nooa.skill import Skill\n\nclass SwapSkill(Skill):\n" + failed_constructor
        )

        result = await registry.reload("test.swap")

        assert result.startswith("Reload failed for test.swap:")
        assert registry["test.swap"] is old_skill
        assert sys.modules["swap_reload_workflow"] is old_package
        assert sys.modules["swap_reload_workflow.helper"] is old_helper
        assert old_skill.value() == "old-code"

        init.write_text(
            "from nooa.skill import Skill\n\n"
            "class SwapSkill(Skill):\n"
            "    def value(self):\n"
            "        return 'recovered'\n"
        )
        stat = init.stat()
        import os

        os.utime(init, (stat.st_atime + 2, stat.st_mtime + 2))
        assert await registry.reload("test.swap") == "Reloaded test.swap (self.swap)"
        assert registry["test.swap"].value() == "recovered"

    @pytest.mark.asyncio
    async def test_failed_source_package_attach_restores_previous_package_tree(
        self, registry, tmp_path
    ):
        lib_dir = tmp_path / "attach-checkout"
        package = lib_dir / "src" / "attach_reload_workflow"
        package.mkdir(parents=True)
        (lib_dir / "pyproject.toml").write_text(
            '[project]\nname = "attach-reload"\n\n'
            '[project.entry-points."nooa.skills"]\n'
            '"test.attach" = "attach_reload_workflow:AttachSkill"\n'
        )
        init = package / "__init__.py"
        init.write_text(
            "from nooa.skill import Skill\nclass AttachSkill(Skill):\n    value = 'old'\n"
        )
        registry.discover_libs(tmp_path)
        old_skill = registry["test.attach"]
        old_package = sys.modules["attach_reload_workflow"]
        init.write_text(
            "from nooa.skill import Skill\n"
            "class AttachSkill(Skill):\n"
            "    value = 'new'\n"
            "    def attach(self, agent):\n"
            "        raise RuntimeError('attach failed')\n"
        )

        result = await registry.reload("test.attach")

        assert result == "Reload failed for test.attach: attach failed"
        assert registry["test.attach"] is old_skill
        assert sys.modules["attach_reload_workflow"] is old_package
        assert old_skill.value == "old"

    @pytest.mark.asyncio
    async def test_reload_not_loaded_raises(self, registry):
        """Reloading an unknown skill raises loudly instead of silently no-op'ing (issue 250)."""
        with pytest.raises(KeyError):
            await registry.reload("nemo.nonexistent")

    @pytest.mark.asyncio
    async def test_reload_all_loaded(self, registry, agent):
        """reload() without args reloads all loaded skills."""
        registry.register("nemo.a", FakeSkill())
        registry.register("nemo.b", FakeSkill())
        result = await registry.reload()
        # Should attempt to reload both — result is a string summary
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_reload_bare_leaf_resolves_to_fq_name(self, registry):
        """A bare leaf name resolves to its fully-qualified skill (issue 250)."""
        registry.register("nvzurich.agent_mesh", FakeSkill())
        called = {}

        async def fake(name):
            called["name"] = name
            return "ok"

        registry._reload_one = fake
        result = await registry.reload("agent_mesh")
        assert called["name"] == "nvzurich.agent_mesh"
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_reload_glob_resolves_single_match(self, registry):
        """An fnmatch glob that hits exactly one loaded skill is accepted."""
        registry.register("nvzurich.agent_mesh", FakeSkill())
        called = {}

        async def fake(name):
            called["name"] = name
            return "ok"

        registry._reload_one = fake
        await registry.reload("nvzurich.*")
        assert called["name"] == "nvzurich.agent_mesh"

    @pytest.mark.asyncio
    async def test_reload_ambiguous_leaf_raises(self, registry):
        """A leaf that matches more than one loaded skill fails loudly."""
        registry.register("nvzurich.shell", FakeSkill())
        registry.register("nemo.shell", FakeSkill())
        with pytest.raises(ValueError):
            await registry.reload("shell")

    @pytest.mark.asyncio
    async def test_reload_ambiguous_glob_raises(self, registry):
        """A glob that matches more than one loaded skill fails loudly."""
        registry.register("nvzurich.shell", FakeSkill())
        registry.register("nvzurich.agent_mesh", FakeSkill())
        with pytest.raises(ValueError):
            await registry.reload("nvzurich.*")

    @pytest.mark.asyncio
    async def test_reload_hyphenated_leaf_resolves(self, registry):
        """A hyphenated leaf query resolves a skill keyed with an underscore leaf (issue 250)."""
        registry.register("nvzurich.agent_mesh", FakeSkill())
        called = {}

        async def fake(name):
            called["name"] = name
            return "ok"

        registry._reload_one = fake
        await registry.reload("agent-mesh")
        assert called["name"] == "nvzurich.agent_mesh"

    @pytest.mark.asyncio
    async def test_reload_exact_fq_name_takes_precedence(self, registry):
        """An exact FQ name reloads that skill even when a leaf would be ambiguous."""
        registry.register("nvzurich.shell", FakeSkill())
        registry.register("nemo.shell", FakeSkill())
        called = {}

        async def fake(name):
            called["name"] = name
            return "ok"

        registry._reload_one = fake
        await registry.reload("nemo.shell")
        assert called["name"] == "nemo.shell"
