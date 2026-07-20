# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-module hot-reload for builtin framework tool skills (issue 225).

Builtin tool skills live under ``nooa`` / ``nooa_cli``, which
are in ``_NO_RELOAD`` because the package-purge reload would strand the live
framework. These tests pin the narrow leaf-module reload path that reloads ONLY
the skill's own module via ``importlib.reload`` and re-resolves the class by name.
"""

import asyncio
import importlib
import sys
import types

import pytest

from nooa.skill_registry import SkillRegistry


class _FakeAgent:
    pass


def _install_framework_module(mod_name: str, cls_name: str, version: str):
    """Create a module that lives under the ``nooa`` top package."""
    mod = types.ModuleType(mod_name)
    mod.__file__ = f"/fake/{mod_name.replace('.', '/')}.py"
    ns: dict = {}
    exec(
        f"""
class {cls_name}:
    version = {version!r}
    def __init__(self, cwd="."):
        self.cwd = cwd
    def attach(self, agent):
        self.agent = agent
    def detach(self):
        pass
""",
        ns,
    )
    cls = ns[cls_name]
    cls.__module__ = mod_name
    setattr(mod, cls_name, cls)
    sys.modules[mod_name] = mod
    return mod, cls


@pytest.mark.asyncio
async def test_framework_tool_skill_reloads_in_place(monkeypatch):
    """A skill in a _NO_RELOAD top package reloads via single-module path.

    Reproduces issue 225: previously this returned
    "Skill ... is not reloadable". The fix reloads only the leaf module.
    """
    mod_name = "nooa.tools._fake_shell_for_test"
    mod_v1, cls_v1 = _install_framework_module(mod_name, "FakeShell", "v1")

    agent = _FakeAgent()
    registry = SkillRegistry(agent)

    skill_v1 = cls_v1()
    skill_v1.attach(agent)
    agent.fake_shell = skill_v1
    registry._attr_map["nemo.fake_shell"] = "fake_shell"
    registry._loaded.add("nemo.fake_shell")

    # importlib.reload(mod) re-execs the module file. Simulate the on-disk edit by
    # having the reload swap in the v2 class definition.
    import importlib

    def fake_reload(mod):
        # Mirror real importlib.reload: re-exec into the SAME module object.
        assert mod is sys.modules[mod_name]
        ns: dict = {}
        exec(
            """
class FakeShell:
    version = "v2"
    def __init__(self, cwd="."):
        self.cwd = cwd
    def attach(self, agent):
        self.agent = agent
    def detach(self):
        pass
""",
            ns,
        )
        new_cls = ns["FakeShell"]
        new_cls.__module__ = mod_name
        mod.FakeShell = new_cls
        return mod

    monkeypatch.setattr(importlib, "reload", fake_reload)

    result = await registry.reload("nemo.fake_shell")

    sys.modules.pop(mod_name, None)

    assert "not reloadable" not in result, result
    assert "Reloaded" in result, result
    # The agent now holds a fresh instance built from the reloaded module.
    assert agent.fake_shell is not skill_v1
    assert agent.fake_shell.version == "v2"
    # Re-attached to the same agent.
    assert agent.fake_shell.agent is agent


@pytest.mark.asyncio
async def test_framework_skill_needing_ctor_args_returns_clear_message(monkeypatch):
    """If the reloaded class can't be built zero-arg, return a clear message, not a crash."""
    mod_name = "nooa.tools._fake_needs_args_for_test"
    mod = types.ModuleType(mod_name)
    mod.__file__ = f"/fake/{mod_name.replace('.', '/')}.py"
    ns: dict = {}
    exec(
        """
class NeedsArgs:
    def __init__(self, required):
        self.required = required
    def attach(self, agent):
        self.agent = agent
""",
        ns,
    )
    cls = ns["NeedsArgs"]
    cls.__module__ = mod_name
    mod.NeedsArgs = cls
    sys.modules[mod_name] = mod

    agent = _FakeAgent()
    registry = SkillRegistry(agent)
    skill = cls(required="x")
    agent.needs_args = skill
    registry._attr_map["nemo.needs_args"] = "needs_args"
    registry._loaded.add("nemo.needs_args")

    import importlib

    monkeypatch.setattr(importlib, "reload", lambda m: m)

    result = await registry.reload("nemo.needs_args")
    sys.modules.pop(mod_name, None)

    # No crash; the agent still has its original instance; message is informative.
    assert agent.needs_args is skill
    assert (
        "not reloadable" in result.lower()
        or "constructor" in result.lower()
        or "args" in result.lower()
    ), result


@pytest.mark.asyncio
async def test_user_skill_reload_path_unchanged(monkeypatch):
    """No regression: a skill in a non-framework top package still uses the package-purge path."""
    import sys as _sys

    from nooa.skill import Skill

    events = []
    ns = {"Skill": Skill, "events": events, "asyncio": asyncio}
    exec(
        """
class LibSkill(Skill):
    version = "v1"
    async def detach(self):
        events.append("old-detach")
        await asyncio.sleep(0)
""",
        ns,
    )
    SkillV1 = ns["LibSkill"]
    SkillV1.__module__ = "mylib"

    mod_v1 = types.ModuleType("mylib")
    mod_v1.LibSkill = SkillV1
    _sys.modules["mylib"] = mod_v1

    agent = _FakeAgent()
    registry = SkillRegistry(agent)
    skill_v1 = SkillV1()
    agent.mylib = skill_v1
    registry._attr_map["local.mylib"] = "mylib"
    registry._loaded.add("local.mylib")

    ns2 = {"Skill": Skill, "events": events}
    exec(
        """
class LibSkill(Skill):
    version = "v2"
    def attach(self, agent):
        events.append("new-attach")
        self._agent = agent
""",
        ns2,
    )
    SkillV2 = ns2["LibSkill"]
    SkillV2.__module__ = "mylib"
    mod_v2 = types.ModuleType("mylib")
    mod_v2.LibSkill = SkillV2

    def fake_import(name):
        events.append("package-import")
        _sys.modules[name] = mod_v2
        return mod_v2

    monkeypatch.setattr(importlib, "import_module", fake_import)

    result = await registry.reload("local.mylib")
    _sys.modules.pop("mylib", None)

    assert "Reloaded" in result, result
    assert events == ["old-detach", "package-import", "new-attach"]
    assert agent.mylib.version == "v2"


@pytest.mark.asyncio
async def test_underscore_top_package_uses_single_module_path(monkeypatch):
    """A skill whose top package starts with '_' takes the single-module reload path.

    Pins the deliberate routing of `top_pkg.startswith("_")` to the in-place
    leaf reload (never the package purge) — so e.g. dynamically-loaded skill
    modules (`_nooa_skill_*`) get a clear, accurate result instead of a
    framework-wide purge.
    """
    mod_name = "_nooa_skill_fake_for_test"
    mod = types.ModuleType(mod_name)
    mod.__file__ = f"/fake/{mod_name}.py"
    ns: dict = {}
    exec(
        """
class FakeUnderscore:
    version = "v1"
    def __init__(self):
        pass
    def attach(self, agent):
        self.agent = agent
""",
        ns,
    )
    cls = ns["FakeUnderscore"]
    cls.__module__ = mod_name
    mod.FakeUnderscore = cls
    sys.modules[mod_name] = mod

    agent = _FakeAgent()
    registry = SkillRegistry(agent)
    skill = cls()
    skill.attach(agent)
    agent.fake_u = skill
    registry._attr_map["ext.fake_u"] = "fake_u"
    registry._loaded.add("ext.fake_u")

    def fake_reload(m):
        assert m is sys.modules[mod_name]
        ns2: dict = {}
        exec(
            """
class FakeUnderscore:
    version = "v2"
    def __init__(self):
        pass
    def attach(self, agent):
        self.agent = agent
""",
            ns2,
        )
        nc = ns2["FakeUnderscore"]
        nc.__module__ = mod_name
        m.FakeUnderscore = nc
        return m

    monkeypatch.setattr(importlib, "reload", fake_reload)
    result = await registry.reload("ext.fake_u")
    sys.modules.pop(mod_name, None)

    assert "Reloaded" in result, result
    assert agent.fake_u.version == "v2"


@pytest.mark.asyncio
async def test_attach_failure_leaves_agent_untouched(monkeypatch):
    """If the reloaded skill's attach() raises, the agent keeps its old skill.

    Mutation (setattr + context-block re-registration) must happen only after a
    successful attach, so a failing reload can't leave a half-reloaded skill.
    """
    mod_name = "nooa.tools._fake_attach_raises_for_test"
    mod = types.ModuleType(mod_name)
    mod.__file__ = f"/fake/{mod_name.replace('.', '/')}.py"
    ns: dict = {}
    exec(
        """
class Boom:
    version = "v1"
    def __init__(self):
        pass
    def attach(self, agent):
        self.agent = agent
""",
        ns,
    )
    cls = ns["Boom"]
    cls.__module__ = mod_name
    mod.Boom = cls
    sys.modules[mod_name] = mod

    agent = _FakeAgent()
    registry = SkillRegistry(agent)
    original = cls()
    original.attach(agent)
    agent.boom = original
    registry._attr_map["nemo.boom"] = "boom"
    registry._loaded.add("nemo.boom")

    def fake_reload(m):
        assert m is sys.modules[mod_name]
        ns2: dict = {}
        exec(
            """
class Boom:
    version = "v2"
    def __init__(self):
        pass
    def attach(self, agent):
        raise RuntimeError("attach boom")
""",
            ns2,
        )
        nc = ns2["Boom"]
        nc.__module__ = mod_name
        m.Boom = nc
        return m

    monkeypatch.setattr(importlib, "reload", fake_reload)
    result = await registry.reload("nemo.boom")
    sys.modules.pop(mod_name, None)

    assert "Reload failed" in result, result
    # Agent must still hold the original, attached skill — not the broken v2.
    assert agent.boom is original
    assert agent.boom.version == "v1"


@pytest.mark.asyncio
async def test_reload_awaits_old_detach_before_new_attach(monkeypatch):
    """Hot-reload must tear down the old skill before starting the new one."""
    mod_name = "nooa.tools._fake_async_detach_for_test"
    mod = types.ModuleType(mod_name)
    mod.__file__ = f"/fake/{mod_name.replace('.', '/')}.py"
    events = []

    class FakeAsyncDetach:
        version = "v1"
        __module__ = mod_name

        def attach(self, agent):
            self.agent = agent

        async def detach(self):
            events.append("old-detach-start")
            await asyncio.sleep(0)
            events.append("old-detach-end")

    old_cls = FakeAsyncDetach
    mod.FakeAsyncDetach = old_cls
    sys.modules[mod_name] = mod

    agent = _FakeAgent()
    registry = SkillRegistry(agent)
    original = old_cls()
    original.attach(agent)
    agent.fake_async_detach = original
    registry._attr_map["nemo.fake_async_detach"] = "fake_async_detach"
    registry._loaded.add("nemo.fake_async_detach")

    def fake_reload(m):
        assert m is sys.modules[mod_name]

        class FakeAsyncDetach:
            version = "v2"
            __module__ = mod_name

            def attach(self, agent):
                events.append("new-attach")
                self.agent = agent

        m.FakeAsyncDetach = FakeAsyncDetach
        return m

    monkeypatch.setattr(importlib, "reload", fake_reload)
    result = await registry.reload("nemo.fake_async_detach")
    sys.modules.pop(mod_name, None)

    assert "Reloaded" in result, result
    assert events == ["old-detach-start", "old-detach-end", "new-attach"]
    assert agent.fake_async_detach is not original
    assert agent.fake_async_detach.version == "v2"


@pytest.mark.asyncio
async def test_reload_detaches_old_skill_before_reexecuting_module(monkeypatch):
    """Old cleanup must run before importlib.reload mutates module globals."""
    mod_name = "nooa.tools._fake_detach_before_reload_for_test"
    mod = types.ModuleType(mod_name)
    mod.__file__ = f"/fake/{mod_name.replace('.', '/')}.py"
    events = []

    class FakeDetachBeforeReload:
        version = "v1"
        __module__ = mod_name

        def attach(self, agent):
            self.agent = agent

        async def detach(self):
            events.append("old-detach")
            await asyncio.sleep(0)

    old_cls = FakeDetachBeforeReload
    mod.FakeDetachBeforeReload = old_cls
    sys.modules[mod_name] = mod

    agent = _FakeAgent()
    registry = SkillRegistry(agent)
    original = old_cls()
    original.attach(agent)
    agent.fake_detach_before_reload = original
    registry._attr_map["nemo.fake_detach_before_reload"] = "fake_detach_before_reload"
    registry._loaded.add("nemo.fake_detach_before_reload")

    def fake_reload(m):
        assert m is sys.modules[mod_name]
        events.append("module-reload")

        class FakeDetachBeforeReload:
            version = "v2"
            __module__ = mod_name

            def attach(self, agent):
                events.append("new-attach")
                self.agent = agent

        m.FakeDetachBeforeReload = FakeDetachBeforeReload
        return m

    monkeypatch.setattr(importlib, "reload", fake_reload)
    result = await registry.reload("nemo.fake_detach_before_reload")
    sys.modules.pop(mod_name, None)

    assert "Reloaded" in result, result
    assert events == ["old-detach", "module-reload", "new-attach"]
    assert agent.fake_detach_before_reload is not original
    assert agent.fake_detach_before_reload.version == "v2"


@pytest.mark.asyncio
async def test_reload_restores_old_skill_if_new_attach_fails_after_detach(monkeypatch):
    """If new attach fails after old detach, best-effort reattach the old skill."""
    mod_name = "nooa.tools._fake_restore_after_detach_for_test"
    mod = types.ModuleType(mod_name)
    mod.__file__ = f"/fake/{mod_name.replace('.', '/')}.py"
    events = []

    class FakeRestoreAfterDetach:
        version = "v1"
        __module__ = mod_name

        def attach(self, agent):
            events.append("old-attach")
            self.agent = agent

        async def detach(self):
            events.append("old-detach")
            await asyncio.sleep(0)

    old_cls = FakeRestoreAfterDetach
    mod.FakeRestoreAfterDetach = old_cls
    sys.modules[mod_name] = mod

    agent = _FakeAgent()
    registry = SkillRegistry(agent)
    original = old_cls()
    original.attach(agent)
    events.clear()
    agent.fake_restore_after_detach = original
    registry._attr_map["nemo.fake_restore_after_detach"] = "fake_restore_after_detach"
    registry._loaded.add("nemo.fake_restore_after_detach")

    def fake_reload(m):
        assert m is sys.modules[mod_name]

        class FakeRestoreAfterDetach:
            version = "v2"
            __module__ = mod_name

            def attach(self, agent):
                events.append("new-attach")
                raise RuntimeError("attach boom")

        m.FakeRestoreAfterDetach = FakeRestoreAfterDetach
        return m

    monkeypatch.setattr(importlib, "reload", fake_reload)
    result = await registry.reload("nemo.fake_restore_after_detach")
    sys.modules.pop(mod_name, None)

    assert "Reload failed" in result, result
    assert events == ["old-detach", "new-attach", "old-attach"]
    assert agent.fake_restore_after_detach is original


@pytest.mark.asyncio
async def test_reload_reports_new_attach_error_if_old_reattach_also_fails(monkeypatch):
    """The replacement attach failure is more useful than a restore failure."""
    mod_name = "nooa.tools._fake_restore_also_fails_for_test"
    mod = types.ModuleType(mod_name)
    mod.__file__ = f"/fake/{mod_name.replace('.', '/')}.py"
    events = []

    class FakeRestoreAlsoFails:
        version = "v1"
        __module__ = mod_name

        def attach(self, agent):
            events.append("old-attach")
            if events.count("old-attach") > 1:
                raise RuntimeError("restore boom")
            self.agent = agent

        async def detach(self):
            events.append("old-detach")
            await asyncio.sleep(0)

    old_cls = FakeRestoreAlsoFails
    mod.FakeRestoreAlsoFails = old_cls
    sys.modules[mod_name] = mod

    agent = _FakeAgent()
    registry = SkillRegistry(agent)
    original = old_cls()
    original.attach(agent)
    events.clear()
    agent.fake_restore_also_fails = original
    registry._attr_map["nemo.fake_restore_also_fails"] = "fake_restore_also_fails"
    registry._loaded.add("nemo.fake_restore_also_fails")

    def fake_reload(m):
        assert m is sys.modules[mod_name]

        class FakeRestoreAlsoFails:
            version = "v2"
            __module__ = mod_name

            def attach(self, agent):
                events.append("new-attach")
                raise RuntimeError("new attach boom")

        m.FakeRestoreAlsoFails = FakeRestoreAlsoFails
        return m

    monkeypatch.setattr(importlib, "reload", fake_reload)
    result = await registry.reload("nemo.fake_restore_also_fails")
    sys.modules.pop(mod_name, None)

    assert "Reload failed" in result, result
    assert "new attach boom" in result, result
    assert "restore boom" not in result, result
    assert events == ["old-detach", "new-attach", "old-attach"]
    assert agent.fake_restore_also_fails is original


@pytest.mark.asyncio
async def test_reload_failure_reattaches_old_skill_after_reload_raises(monkeypatch):
    """A failed module reload after old detach restores the original skill."""
    mod_name = "nooa.tools._fake_reload_raises_after_detach_for_test"
    mod = types.ModuleType(mod_name)
    mod.__file__ = f"/fake/{mod_name.replace('.', '/')}.py"
    events = []

    class FakeReloadRaisesAfterDetach:
        version = "v1"
        __module__ = mod_name

        def attach(self, agent):
            events.append("old-attach")
            self.agent = agent

        async def detach(self):
            events.append("old-detach")
            await asyncio.sleep(0)

    mod.FakeReloadRaisesAfterDetach = FakeReloadRaisesAfterDetach
    sys.modules[mod_name] = mod

    agent = _FakeAgent()
    registry = SkillRegistry(agent)
    original = FakeReloadRaisesAfterDetach()
    original.attach(agent)
    events.clear()
    agent.fake_reload_raises_after_detach = original
    registry._attr_map["nemo.fake_reload_raises_after_detach"] = "fake_reload_raises_after_detach"
    registry._loaded.add("nemo.fake_reload_raises_after_detach")

    def fake_reload(m):
        events.append("module-reload")
        raise RuntimeError("reload boom")

    monkeypatch.setattr(importlib, "reload", fake_reload)
    result = await registry.reload("nemo.fake_reload_raises_after_detach")
    sys.modules.pop(mod_name, None)

    assert "Reload failed" in result, result
    assert "reload boom" in result, result
    assert events == ["old-detach", "module-reload", "old-attach"]
    assert agent.fake_reload_raises_after_detach is original


@pytest.mark.asyncio
async def test_reload_failure_does_not_reload_when_old_detach_raises(monkeypatch):
    """If old detach fails, stop before module reload and return a controlled failure."""
    mod_name = "nooa.tools._fake_detach_raises_before_reload_for_test"
    mod = types.ModuleType(mod_name)
    mod.__file__ = f"/fake/{mod_name.replace('.', '/')}.py"
    events = []

    class FakeDetachRaisesBeforeReload:
        version = "v1"
        __module__ = mod_name

        def attach(self, agent):
            events.append("old-attach")
            self.agent = agent

        async def detach(self):
            events.append("old-detach")
            await asyncio.sleep(0)
            raise RuntimeError("detach boom")

    mod.FakeDetachRaisesBeforeReload = FakeDetachRaisesBeforeReload
    sys.modules[mod_name] = mod

    agent = _FakeAgent()
    registry = SkillRegistry(agent)
    original = FakeDetachRaisesBeforeReload()
    original.attach(agent)
    events.clear()
    agent.fake_detach_raises_before_reload = original
    registry._attr_map["nemo.fake_detach_raises_before_reload"] = "fake_detach_raises_before_reload"
    registry._loaded.add("nemo.fake_detach_raises_before_reload")

    def fake_reload(m):
        events.append("module-reload")
        return m

    monkeypatch.setattr(importlib, "reload", fake_reload)
    result = await registry.reload("nemo.fake_detach_raises_before_reload")
    sys.modules.pop(mod_name, None)

    assert "Reload failed" in result, result
    assert "detach boom" in result, result
    assert events == ["old-detach", "old-attach"]
    assert agent.fake_detach_raises_before_reload is original


@pytest.mark.asyncio
async def test_reload_restores_old_skill_if_new_constructor_raises(monkeypatch):
    """Constructor failures after pre-reload detach also restore the old skill."""
    mod_name = "nooa.tools._fake_constructor_raises_for_test"
    mod = types.ModuleType(mod_name)
    mod.__file__ = f"/fake/{mod_name.replace('.', '/')}.py"
    events = []

    class FakeConstructorRaises:
        version = "v1"
        __module__ = mod_name

        def attach(self, agent):
            events.append("old-attach")
            self.agent = agent

        async def detach(self):
            events.append("old-detach")
            await asyncio.sleep(0)

    mod.FakeConstructorRaises = FakeConstructorRaises
    sys.modules[mod_name] = mod

    agent = _FakeAgent()
    registry = SkillRegistry(agent)
    original = FakeConstructorRaises()
    original.attach(agent)
    events.clear()
    agent.fake_constructor_raises = original
    registry._attr_map["nemo.fake_constructor_raises"] = "fake_constructor_raises"
    registry._loaded.add("nemo.fake_constructor_raises")

    def fake_reload(m):
        events.append("module-reload")

        class FakeConstructorRaises:
            version = "v2"
            __module__ = mod_name

            def __init__(self):
                events.append("new-construct")
                raise RuntimeError("constructor boom")

        m.FakeConstructorRaises = FakeConstructorRaises
        return m

    monkeypatch.setattr(importlib, "reload", fake_reload)
    result = await registry.reload("nemo.fake_constructor_raises")
    sys.modules.pop(mod_name, None)

    assert "Reload failed" in result, result
    assert "constructor boom" in result, result
    assert events == ["old-detach", "module-reload", "new-construct", "old-attach"]
    assert agent.fake_constructor_raises is original
