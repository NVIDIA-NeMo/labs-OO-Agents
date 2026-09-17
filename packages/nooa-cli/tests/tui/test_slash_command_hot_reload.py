"""Test that slash command hot-reload picks up new code after SkillRegistry.reload()."""

import sys
import types
from unittest.mock import patch

import pytest

from nooa.skill import Skill, slash_command
from nooa.skill_registry import SkillRegistry


def _make_skill_class(version: str):
    """Create a Skill subclass dynamically with a slash command."""
    ns = {"Skill": Skill, "slash_command": slash_command}
    exec(
        f"""
class TestSkill(Skill):
    @slash_command("greet", argument_hint="<name>")
    def greet(self, args: str) -> str:
        return "{version}: " + args
""",
        ns,
    )
    cls = ns["TestSkill"]
    cls.__module__ = "test_hot_skill"
    return cls


def test_stale_slash_command_without_refresh():
    """Prove the bug: after replacing a skill, CommandRegistry holds stale method refs.

    Without calling refresh_skill_commands(), the _user_skills dict still
    references the old bound method from the previous skill instance.
    """
    from nooa_cli.tui.commands import CommandRegistry

    class FakeAgent:
        pass

    agent = FakeAgent()

    # Create v1 skill
    SkillV1 = _make_skill_class("v1")
    skill_v1 = SkillV1()
    agent.test_hot_skill = skill_v1

    # Create command registry and discover initial commands
    cmd_registry = CommandRegistry.__new__(CommandRegistry)
    cmd_registry.agent = agent
    cmd_registry._commands = {}
    cmd_registry._user_skills = {}
    cmd_registry.skills_dirs = []

    fresh = cmd_registry._discover_skill_commands()
    cmd_registry._user_skills.update(fresh)

    assert "greet" in cmd_registry._user_skills
    assert cmd_registry._user_skills["greet"]._method("world") == "v1: world"

    # Replace skill with v2 (as _reload_one does via setattr)
    SkillV2 = _make_skill_class("v2")
    skill_v2 = SkillV2()
    agent.test_hot_skill = skill_v2

    # BUG: Without refresh, the command still uses v1's method
    stale = cmd_registry._user_skills["greet"]._method("world")
    assert stale == "v1: world", "Sanity check: without refresh, method is stale"

    # After refresh, it should use v2
    cmd_registry.refresh_skill_commands()
    fresh_result = cmd_registry._user_skills["greet"]._method("world")
    assert fresh_result == "v2: world"


@pytest.mark.asyncio
async def test_skill_registry_reload_one_calls_refresh():
    """_reload_one should trigger refresh_skill_commands on the CommandRegistry.

    This test verifies the fix: after _reload_one replaces the skill instance,
    it should call agent._command_registry.refresh_skill_commands().
    """

    class FakeAgent:
        pass

    agent = FakeAgent()
    registry = SkillRegistry(agent)

    # Create initial skill
    SkillV1 = _make_skill_class("v1")
    skill_v1 = SkillV1()
    agent.test_hot_skill = skill_v1
    registry._attr_map["local.test_hot_skill"] = "test_hot_skill"
    registry._loaded.add("local.test_hot_skill")

    # Track whether refresh_skill_commands was called
    refresh_called = []

    class FakeCommandRegistry:
        def refresh_skill_commands(self):
            refresh_called.append(True)

    agent._command_registry = FakeCommandRegistry()

    # Patch importlib.import_module so _reload_one can "reimport" our fake module
    SkillV2 = _make_skill_class("v2")
    mod_v2 = types.ModuleType("test_hot_skill")
    mod_v2.__file__ = "/fake/test_hot_skill/__init__.py"
    mod_v2.TestSkill = SkillV2

    # Put v1 module in sys.modules first (so _reload_one can find module name)
    mod_v1 = types.ModuleType("test_hot_skill")
    mod_v1.TestSkill = SkillV1
    sys.modules["test_hot_skill"] = mod_v1

    def fake_import(name):
        sys.modules[name] = mod_v2
        return mod_v2

    with patch("importlib.import_module", side_effect=fake_import):
        result = await registry._reload_one("local.test_hot_skill")

    # Cleanup
    sys.modules.pop("test_hot_skill", None)

    assert "Reloaded" in result, f"Reload failed: {result}"
    assert refresh_called, (
        "refresh_skill_commands was NOT called after _reload_one — this is the bug"
    )
