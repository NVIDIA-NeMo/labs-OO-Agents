# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
from nooa_cli.tui.config import Config


def _configure_project(monkeypatch, tmp_path):
    project_dir = tmp_path / ".nooa"
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project_dir))
    import nooa_cli.tui.session_manager as session_manager

    monkeypatch.setattr(session_manager, "SESSIONS_DIR", project_dir / "sessions")
    return project_dir


def test_tui_memory_config_defaults_to_off():
    cfg = Config.load()
    assert cfg.tui.memory == "off"
    assert cfg.tui.memory_agents == {}
    assert cfg.tui.memory_path is None


def test_tui_memory_config_overrides_path(tmp_path):
    db = tmp_path / "custom-memory.db"
    cfg = Config.load(memory="session", memory_path=str(db))
    assert cfg.tui.memory == "session"
    assert cfg.tui.memory_path == db


def test_tui_memory_config_loads_per_agent_preferences():
    cfg = Config.load(memory_agents={"my.module:Agent": "session"})
    assert cfg.tui.memory == "off"
    assert cfg.tui.memory_agents == {"my.module:Agent": "session"}


def test_tui_memory_config_accepts_project_scope():
    cfg = Config.load(memory="project", memory_agents={"my.module:Agent": "project"})
    assert cfg.tui.memory == "project"
    assert cfg.tui.memory_agents == {"my.module:Agent": "project"}


@pytest.mark.asyncio
async def test_bootstrap_default_memory_off_does_not_activate_memory(tmp_path, monkeypatch):
    from nooa_cli.tui.bootstrap import bootstrap

    _configure_project(monkeypatch, tmp_path)
    cfg = Config()
    cfg.tui.default_model = "test-model"
    cfg.agent.summarization.policy = "none"
    result = await bootstrap(cfg)
    try:
        assert not hasattr(result.agent, "memory")
        assert "nemo.memory" not in result.agent.skills.activated()
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_bootstrap_per_agent_memory_is_session_scoped(tmp_path, monkeypatch):
    from nooa_cli.tui.bootstrap import bootstrap
    from nooa_memory.memory_skill import MemorySkill

    project_dir = _configure_project(monkeypatch, tmp_path)
    cfg = Config()
    cfg.tui.default_model = "test-model"
    cfg.tui.memory_agents["nooa_cli.tui.agent:TUIAgent"] = "session"
    cfg.agent.summarization.policy = "none"
    result = await bootstrap(cfg)
    try:
        assert isinstance(result.agent.memory, MemorySkill)
        assert result.agent.memory._mgr is not None
        assert result.session_id is not None
        assert result.agent.memory._mgr.store.path == str(
            project_dir / "sessions" / f"{result.session_id}-memory.db"
        )
        assert "nemo.memory" in result.agent.skills.activated()
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_bootstrap_session_memory_sets_owner_and_session_ref(tmp_path, monkeypatch):
    from nooa_cli.tui.bootstrap import bootstrap

    _configure_project(monkeypatch, tmp_path)
    cfg = Config()
    cfg.tui.default_model = "test-model"
    cfg.tui.memory = "session"
    cfg.agent.summarization.policy = "none"
    result = await bootstrap(cfg)
    try:
        mgr = result.agent.memory._mgr
        # hierarchical owner: role (class name) @ 8-hex session instance
        assert mgr.owner == f"TUIAgent@{result.session_id[:8]}"
        assert mgr.role == "TUIAgent"
        assert mgr.session_ref == result.session_id
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_bootstrap_project_memory_uses_working_dir_store(tmp_path, monkeypatch):
    from nooa_cli.tui.bootstrap import bootstrap

    _configure_project(monkeypatch, tmp_path)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    cfg = Config()
    cfg.tui.default_model = "test-model"
    cfg.tui.memory = "project"
    cfg.agent.working_dir = str(work_dir)
    cfg.agent.summarization.policy = "none"
    result = await bootstrap(cfg)
    try:
        expected = (work_dir / ".nooa" / "memory" / "memory.sqlite").resolve()
        mgr = result.agent.memory._mgr
        assert mgr.store.path == str(expected)
        assert expected.exists()
        assert mgr.owner == f"TUIAgent@{result.session_id[:8]}"
        assert mgr.session_ref == result.session_id
        assert "nemo.memory" in result.agent.skills.activated()
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_configure_tui_memory_repoints_session_scope(tmp_path, monkeypatch):
    from nooa_cli.tui.bootstrap import bootstrap, configure_tui_memory

    project_dir = _configure_project(monkeypatch, tmp_path)
    cfg = Config()
    cfg.tui.default_model = "test-model"
    cfg.tui.memory = "session"
    cfg.agent.summarization.policy = "none"
    result = await bootstrap(cfg)
    try:
        assert result.agent.memory._mgr.store.path == str(
            project_dir / "sessions" / f"{result.session_id}-memory.db"
        )
        new_id = "22222222-2222-4222-8222-222222222222"
        new_db = project_dir / "sessions" / f"{new_id}.db"
        configure_tui_memory(result.agent, cfg, agent_db=new_db, session_id=new_id)
        assert result.agent.memory._mgr.store.path == str(
            project_dir / "sessions" / f"{new_id}-memory.db"
        )
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


def test_session_manager_ignores_and_deletes_memory_sidecars(tmp_path, monkeypatch):
    from nooa_cli.tui.session_manager import SessionManager

    project_dir = _configure_project(monkeypatch, tmp_path)
    sessions_dir = project_dir / "sessions"
    sm = SessionManager.create(model="m", agent_cls="A", working_dir=str(tmp_path))
    session_id = sm.session_id
    sm.close()
    sidecar = sessions_dir / f"{session_id}-memory.db"
    sidecar.write_text("memory sidecar")

    assert SessionManager.find_by_prefix(session_id[:8]) == [session_id]
    assert [m.id for m in SessionManager.list_sessions()] == [session_id]

    assert SessionManager.delete_session(session_id) is True
    assert not (sessions_dir / f"{session_id}.db").exists()
    assert not sidecar.exists()


def test_delete_session_never_touches_project_memory_db(tmp_path, monkeypatch):
    from nooa_cli.tui.session_manager import SessionManager

    project_dir = _configure_project(monkeypatch, tmp_path)
    sessions_dir = project_dir / "sessions"
    sm = SessionManager.create(model="m", agent_cls="A", working_dir=str(tmp_path))
    session_id = sm.session_id
    sm.close()
    sidecar = sessions_dir / f"{session_id}-memory.db"
    sidecar.write_text("session sidecar")

    project_db = tmp_path / "work" / ".nooa" / "memory" / "memory.sqlite"
    project_db.parent.mkdir(parents=True, exist_ok=True)
    project_db.write_text("project store")

    assert SessionManager.delete_session(session_id) is True
    # Only the session DB + its session-derived sidecars are unlinked.
    assert not (sessions_dir / f"{session_id}.db").exists()
    assert not sidecar.exists()
    assert project_db.exists()
    assert project_db.read_text() == "project store"


@pytest.mark.asyncio
async def test_memory_command_on_off_persists_to_settings_yaml_and_reloads(tmp_path, monkeypatch):
    from nooa_cli.tui.bootstrap import bootstrap, resolve_tui_memory_scope
    from nooa_cli.tui.commands import CommandRegistry
    from nooa_cli.tui.settings import SETTINGS_FILENAME

    project_dir = _configure_project(monkeypatch, tmp_path)
    agent_key = "nooa_cli.tui.agent:TUIAgent"
    cfg = Config()
    cfg.tui.default_model = "test-model"
    cfg.agent.summarization.policy = "none"
    result = await bootstrap(cfg)
    try:
        registry = CommandRegistry(
            config=cfg.tui,
            agent=result.agent,
            frontend=object(),
            skills_dirs=[],
            session_manager=result.session_manager,
            root_config=cfg,
        )
        cmd = registry._commands["memory"]

        on_result = await cmd.execute(["on"])
        assert on_result.success
        assert cfg.tui.memory == "project"  # on == the shared project store
        assert cfg.tui.memory_agents[agent_key] == "project"
        expected = (Path(cfg.agent.working_dir) / ".nooa" / "memory" / "memory.sqlite").resolve()
        assert result.agent.memory._mgr.store.path == str(expected)

        # Persistence lands in settings.yaml — the file the loader actually reads —
        # NOT the phantom config.toml that no loader reads.
        settings_file = project_dir / SETTINGS_FILENAME
        assert settings_file.exists()
        assert not (project_dir / "config.toml").exists()

        # Round-trip: a fresh Config.load() (which applies settings.yaml) sees the
        # preference, so it survives the restart the command prompts for.
        reloaded = Config.load()
        assert reloaded.tui.memory_agents.get(agent_key) == "project"
        assert resolve_tui_memory_scope(result.agent, reloaded) == "project"

        off_result = await cmd.execute(["off"])
        assert off_result.success
        assert cfg.tui.memory == "off"
        assert cfg.tui.memory_agents[agent_key] == "off"
        assert not hasattr(result.agent, "memory")

        reloaded_off = Config.load()
        assert reloaded_off.tui.memory_agents.get(agent_key) == "off"
        assert resolve_tui_memory_scope(result.agent, reloaded_off) == "off"
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_memory_command_local_scope(tmp_path, monkeypatch):
    from nooa_cli.tui.bootstrap import bootstrap
    from nooa_cli.tui.commands import CommandRegistry

    _configure_project(monkeypatch, tmp_path)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    cfg = Config()
    cfg.tui.default_model = "test-model"
    cfg.agent.working_dir = str(work_dir)
    cfg.agent.summarization.policy = "none"
    result = await bootstrap(cfg)
    try:
        registry = CommandRegistry(
            config=cfg.tui,
            agent=result.agent,
            frontend=object(),
            skills_dirs=[],
            session_manager=result.session_manager,
            root_config=cfg,
        )
        cmd = registry._commands["memory"]

        local_result = await cmd.execute(["local"])
        assert local_result.success
        assert cfg.tui.memory == "session"  # 'local' = per-session scope
        assert cfg.tui.memory_agents["nooa_cli.tui.agent:TUIAgent"] == "session"
        mgr = result.agent.memory._mgr
        assert f"{result.session_id}-memory.db" in mgr.store.path
        assert mgr.session_ref == result.session_id

        status = await cmd.execute(["status"])
        assert status.success
        content = status.outputs[0].content
        assert content.startswith("Memory: local (this session only)")
        assert f"you are TUIAgent@{result.session_id[:8]}" in content
        assert "store:" in content

        # the old scope words are no longer part of the command vocabulary
        assert cmd.validate_args(["project"])[0] is False
        assert cmd.validate_args(["session"])[0] is False
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_bootstrap_rejects_memory_path_escape(tmp_path, monkeypatch):
    from nooa_cli.tui.bootstrap import bootstrap

    _configure_project(monkeypatch, tmp_path)
    cfg = Config()
    cfg.tui.default_model = "test-model"
    cfg.tui.memory = "session"
    cfg.tui.memory_path = tmp_path / "outside.db"
    cfg.agent.summarization.policy = "none"

    result = await bootstrap(cfg)
    try:
        assert not hasattr(result.agent, "memory")
        assert any(
            "tui.memory_path must be relative" in getattr(m, "content", "") for m in result.messages
        )
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_bootstrap_rejects_memory_path_parent_escape(tmp_path, monkeypatch):
    from nooa_cli.tui.bootstrap import bootstrap

    _configure_project(monkeypatch, tmp_path)
    cfg = Config()
    cfg.tui.default_model = "test-model"
    cfg.tui.memory = "session"
    cfg.tui.memory_path = Path("../outside.db")
    cfg.agent.summarization.policy = "none"

    result = await bootstrap(cfg)
    try:
        assert not hasattr(result.agent, "memory")
        assert any(
            "tui.memory_path must stay under the project directory" in getattr(m, "content", "")
            for m in result.messages
        )
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_owner_resolution_override_chain(tmp_path, monkeypatch):
    """per-agent override > global memory_owner > class-name default."""
    from nooa_cli.tui.bootstrap import bootstrap

    _configure_project(monkeypatch, tmp_path)
    cfg = Config()
    cfg.tui.default_model = "test-model"
    cfg.tui.memory = "project"
    cfg.tui.memory_owner = "team"
    cfg.agent.working_dir = str(tmp_path / "work")
    Path(cfg.agent.working_dir).mkdir()
    cfg.agent.summarization.policy = "none"
    result = await bootstrap(cfg)
    try:
        mgr = result.agent.memory._mgr
        assert mgr.owner == f"team@{result.session_id[:8]}"  # global override wins
        assert mgr.role == "team"

        cfg.tui.memory_owner_agents["nooa_cli.tui.agent:TUIAgent"] = "planner"
        from nooa_cli.tui.bootstrap import configure_tui_memory

        configure_tui_memory(
            result.agent,
            cfg,
            agent_db=result.session_manager.agent_db_path,
            session_id=result.session_manager.session_id,
        )
        assert result.agent.memory._mgr.role == "planner"  # per-agent beats global
        assert result.agent.memory._mgr.owner.startswith("planner@")
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_legacy_owner_rows_are_healed_on_configure(tmp_path, monkeypatch):
    """Rows written under the module:qualname key fold into the class name."""
    from nooa_cli.tui.bootstrap import bootstrap
    from nooa_memory import Memory
    from nooa_memory.embeddings import HashingEmbedder
    from nooa_memory.store import MemoryStore

    _configure_project(monkeypatch, tmp_path)
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    legacy_key = "nooa_cli.tui.agent:TUIAgent"
    store_path = work_dir / ".nooa" / "memory" / "memory.sqlite"
    seed_store = MemoryStore(store_path)
    emb = HashingEmbedder(dim=256)
    m = Memory(content="written by the legacy spelling", owner=legacy_key)
    seed_store.add(m, emb.embed(m.embedding_text()))
    seed_store.close()

    cfg = Config()
    cfg.tui.default_model = "test-model"
    cfg.tui.memory = "project"
    cfg.agent.working_dir = str(work_dir)
    cfg.agent.summarization.policy = "none"
    result = await bootstrap(cfg)
    try:
        mgr = result.agent.memory._mgr
        healed = mgr.store.get(m.id)
        assert healed.owner == "TUIAgent"  # folded into the canonical short name
        history = mgr.store.maintenance_history(5)
        rename_rows = [h for h in history if h["kind"] == "rename_owner"]
        assert rename_rows and rename_rows[0]["report"] == {
            "from": legacy_key,
            "to": "TUIAgent",
            "rows": 1,
        }
        # the healed row is now recallable in the agent's default own-scope
        assert mgr.recall("legacy spelling")
    finally:
        if result.session_manager is not None:
            result.session_manager.close()


@pytest.mark.asyncio
async def test_generative_reflection_hooks_wiring(tmp_path, monkeypatch):
    """/reflection on wires the LLM reconciler+reasoner by default; the
    tui.reflection_generative knob and reflection-off both unwire them."""
    from nooa_cli.tui.bootstrap import bootstrap, configure_tui_memory

    _configure_project(monkeypatch, tmp_path)
    cfg = Config()
    cfg.tui.default_model = "test-model"
    cfg.tui.memory = "project"
    cfg.tui.reflection = True
    cfg.agent.working_dir = str(tmp_path / "work")
    Path(cfg.agent.working_dir).mkdir()
    cfg.agent.summarization.policy = "none"
    result = await bootstrap(cfg)
    try:
        mgr = result.agent.memory._mgr
        assert mgr._reconciler is not None and mgr._reasoner is not None
        assert mgr.config.reflection.trigger == "manual"  # idle runner owns it

        cfg.tui.reflection_generative = False
        configure_tui_memory(
            result.agent,
            cfg,
            agent_db=result.session_manager.agent_db_path,
            session_id=result.session_manager.session_id,
        )
        mgr = result.agent.memory._mgr
        assert mgr._reconciler is None and mgr._reasoner is None

        cfg.tui.reflection_generative = True
        cfg.tui.reflection = False  # reflection off -> no generative hooks either
        configure_tui_memory(
            result.agent,
            cfg,
            agent_db=result.session_manager.agent_db_path,
            session_id=result.session_manager.session_id,
        )
        assert result.agent.memory._mgr._reconciler is None
    finally:
        if result.session_manager is not None:
            result.session_manager.close()
