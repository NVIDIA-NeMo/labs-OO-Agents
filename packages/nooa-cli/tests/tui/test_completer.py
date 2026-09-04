# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared completion engine (completer.py).

Verifies that both TUI and web get identical completion behavior.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nooa_cli.tui.completer import Completer


@pytest.fixture
def mock_registry():
    reg = MagicMock()
    reg.agent.cwd = Path.cwd()
    _completions = {
        "/help": "Show all commands",
        "/edit": "Edit a file",
        "/session <new|resume|delete|export|rename>": "Manage sessions",
        "/show-python [status|on|off]": "Configure Python display",
        "/exit": "Exit",
    }
    # The completer uses get_active_help() (keys may include argument hints).
    # For built-in commands there are no hints, so this is the same dict.
    reg.get_active_help.return_value = _completions
    reg.get_completions.return_value = _completions
    return reg


@pytest.fixture
def completer(mock_registry):
    return Completer(registry=mock_registry)


def test_event_summary_previews_python_cell_code():
    event = MagicMock()
    event.name = "python_cell"
    event.arguments = {"code": "# setup\nprint('experimental')"}

    assert Completer._event_summary(event, "ToolCallEvent") == (
        "python_cell — print('experimental')"
    )


# ---------------------------------------------------------------------------
# Slash command completion
# ---------------------------------------------------------------------------


def test_slash_prefix_returns_all_commands(completer):
    items = completer.complete("/")
    texts = [i.text for i in items]
    assert "/help" in texts
    assert "/edit" in texts
    assert "/exit" in texts


def test_slash_partial_filters(completer):
    items = completer.complete("/he")
    assert len(items) == 1
    assert items[0].text == "/help"
    assert items[0].description == "Show all commands"


def test_slash_session_partial(completer):
    items = completer.complete("/session ")
    assert [item.text for item in items] == [
        "/session new",
        "/session resume",
        "/session delete",
        "/session export",
        "/session rename",
    ]


def test_slash_session_subcommands_filter_partial(completer):
    assert [item.text for item in completer.complete("/session re")] == [
        "/session resume",
        "/session rename",
    ]
    assert completer.complete("/session list") == []


def test_session_id_completion_excludes_empty_sessions(completer, monkeypatch):
    from nooa_cli.tui.session_manager import SessionManager, SessionMeta

    monkeypatch.setattr(
        SessionManager,
        "list_sessions",
        classmethod(
            lambda cls, limit=50: [
                SessionMeta("empty-session", "model", "Agent", 1.0, 2.0, turn_count=0),
                SessionMeta("used-session", "model", "Agent", 1.0, 2.0, turn_count=2),
            ]
        ),
    )

    assert [item.text for item in completer.complete("/session resume ")] == [
        "/session resume used-ses"
    ]
    assert [item.text for item in completer.complete("/session delete ")] == [
        "/session delete used-ses"
    ]
    # The /resume alias completes session ids like /session resume.
    assert [item.text for item in completer.complete("/resume ")] == ["/resume used-ses"]
    assert [item.text for item in completer.complete("/resume used")] == ["/resume used-ses"]


def test_theme_completion_uses_discovered_theme_ids(completer, monkeypatch) -> None:
    from nooa_cli.tui import theme

    monkeypatch.setattr(theme, "theme_names", lambda: ("mocha", "custom-night"))

    assert [item.text for item in completer.complete("/theme ")][:3] == [
        "/theme picker",
        "/theme gallery",
        "/theme update",
    ]
    assert [item.text for item in completer.complete("/theme g")] == ["/theme gallery"]
    assert [item.text for item in completer.complete("/theme custom")] == ["/theme custom-night"]


def test_slash_case_insensitive(completer):
    items = completer.complete("/HE")
    assert len(items) == 1
    assert items[0].text == "/help"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "/skills ",
            [
                "/skills list",
                "/skills add",
                "/skills commands",
                "/skills activate",
                "/skills deactivate",
            ],
        ),
        (
            "/show-python ",
            ["/show-python status", "/show-python on", "/show-python off"],
        ),
        (
            "/show-diffs ",
            ["/show-diffs status", "/show-diffs on", "/show-diffs off"],
        ),
        (
            "/reasoning ",
            [
                "/reasoning off",
                "/reasoning low",
                "/reasoning medium",
                "/reasoning high",
            ],
        ),
        ("/toolbar ", ["/toolbar reset", "/toolbar set"]),
        ("/memory ", ["/memory on", "/memory local", "/memory off"]),
        (
            "/reflection ",
            ["/reflection on", "/reflection off", "/reflection now"],
        ),
        (
            "/keep-going ",
            ["/keep-going on", "/keep-going off", "/keep-going model"],
        ),
    ],
)
def test_compact_help_command_families_complete_actions(completer, text, expected):
    assert [item.text for item in completer.complete(text)] == expected


def test_model_completion_does_not_include_backend_setup(completer):
    items = completer.complete("/model dis")

    assert [item.text for item in items] == []


def test_connect_completes_deduplicated_api_bases(completer, monkeypatch):
    import nooa.unifiedllm as unifiedllm

    monkeypatch.setattr(
        unifiedllm,
        "MODELS",
        {
            "alpha": {"api_base": "https://inference.example.test/v1/"},
            "beta": {"api_base": "https://inference.example.test/v1/models"},
            "gamma": {"api_base": "https://other.example.test/api"},
            "provider-default": {"model_name": "anthropic/model"},
            "unsafe": {"api_base": "https://user:password@example.test/v1"},
        },
    )

    items = completer.complete("/connect ")

    assert [item.text for item in items] == [
        "/connect https://inference.example.test/v1",
        "/connect https://other.example.test/api",
    ]
    assert items[0].display == "https://inference.example.test/v1"
    assert items[0].description == "Used by alpha, beta"


def test_connect_endpoint_completion_filters_partial(completer, monkeypatch):
    import nooa.unifiedllm as unifiedllm

    monkeypatch.setattr(
        unifiedllm,
        "MODELS",
        {
            "alpha": {"api_base": "https://inference.example.test/v1"},
            "beta": {"api_base": "https://other.example.test/v1"},
        },
    )

    assert [item.text for item in completer.complete("/connect https://oth")] == [
        "/connect https://other.example.test/v1"
    ]


def test_connect_provider_completion_no_hardcoded_shortcuts(completer):
    assert list(completer.complete("/connect ant")) == []


def test_skills_add_completes_directories_only(tmp_path, mock_registry):
    (tmp_path / "skill-pack").mkdir()
    (tmp_path / "skill-pack" / "nested").mkdir()
    (tmp_path / "not-a-directory.md").touch()
    mock_registry.agent.cwd = tmp_path
    completer = Completer(mock_registry)

    assert [item.text for item in completer.complete("/skills add ")] == ["/skills add skill-pack/"]
    assert [item.text for item in completer.complete("/skills add skill-pack/")] == [
        "/skills add skill-pack/nested/"
    ]


def test_toolbar_set_completes_unselected_items(completer):
    first = completer.complete("/toolbar set ")
    assert first

    selected = first[0].text.removeprefix("/toolbar set ")
    remaining = completer.complete(f"/toolbar set {selected} ")
    assert all(item.text.startswith(f"/toolbar set {selected} ") for item in remaining)
    assert all(item.text != f"/toolbar set {selected} {selected}" for item in remaining)


# ---------------------------------------------------------------------------
# Path completion (/edit)
# ---------------------------------------------------------------------------


def test_edit_path_completion(completer):
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some files
        Path(tmpdir, "foo.py").touch()
        Path(tmpdir, "bar.txt").touch()
        Path(tmpdir, "subdir").mkdir()

        items = completer.complete(f"/edit {tmpdir}/")
        texts = [i.text for i in items]
        displays = [i.display for i in items]

        assert any("foo.py" in t for t in texts)
        assert any("bar.txt" in t for t in texts)
        assert any("subdir/" in d for d in displays)

        # Every item.text must start with "/edit " so the web UI
        # can use it as a full input replacement without losing the command.
        for item in items:
            assert item.text.startswith("/edit "), (
                f"Path completion item should be a full replacement: {item.text!r}"
            )


def test_edit_path_partial_filter(completer):
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "alpha.py").touch()
        Path(tmpdir, "beta.py").touch()

        items = completer.complete(f"/edit {tmpdir}/al")
        texts = [i.text for i in items]
        assert len(items) == 1
        assert any("alpha.py" in t for t in texts)


# ---------------------------------------------------------------------------
# Bang commands
# ---------------------------------------------------------------------------


def test_bang_prefix_returns_no_builtins(completer):
    items = completer.complete("!")
    texts = [i.text for i in items]
    # No bang builtins — !<cmd> just runs bash
    assert texts == []


# ---------------------------------------------------------------------------
# No completions for regular text
# ---------------------------------------------------------------------------


def test_regular_text_no_completions(completer):
    assert completer.complete("hello") == []
    assert completer.complete("") == []
    assert completer.complete("how are you") == []


# ---------------------------------------------------------------------------
# User-invocable skills in completer — real registry, not mock
# ---------------------------------------------------------------------------


@pytest.fixture
def user_skill_registry(tmp_path):
    """A real CommandRegistry with one install-as:command skill."""
    from unittest.mock import MagicMock

    from nooa_cli.tui.commands import CommandRegistry

    skill_dir = tmp_path / "myskill"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: myskill\ndescription: My test skill\n"
        "argument-hint: <action>\ninstall-as: command\n---\nDo the thing.\n"
    )

    agent = MagicMock()
    agent.get_summarization_status = MagicMock(
        return_value={
            "active_events": 0,
            "policy": "auto",
            "has_summarizer": False,
            "max_tokens": 100000,
            "current_tokens": 0,
            "preserve_recent": 5,
            "summary_count": 0,
            "summary_tags": [],
        }
    )
    config = MagicMock()
    config.default_model = "test-model"
    frontend = MagicMock()

    return CommandRegistry(
        config=config,
        agent=agent,
        frontend=frontend,
        skills_dirs=[tmp_path],
        mcp_file=None,
    )


def test_user_skill_appears_in_completer_via_real_registry(user_skill_registry):
    """User skill shows up in Completer.complete() with a REAL registry, not a mock.

    This catches the gap where registry.get_completions() works but the skill
    fails to reach the Tab-completion dropdown in practice.
    """
    completer = Completer(registry=user_skill_registry)
    items = completer.complete("/")
    texts = [i.text for i in items]
    assert any("myskill" in t for t in texts), f"Expected '/myskill' in completions, got: {texts}"


def test_user_skill_is_not_flattened_into_builtin_help(user_skill_registry):
    """Generic /help stays portable; /skills commands owns extension discovery."""
    assert any(key.startswith("/myskill") for key in user_skill_registry.get_active_help())
    assert not any(key.startswith("/myskill") for key in user_skill_registry.get_builtin_help())


def test_user_skill_partial_completion(user_skill_registry):
    """Partial typing '/mys' narrows to the user skill."""
    completer = Completer(registry=user_skill_registry)
    items = completer.complete("/mys")
    texts = [i.text for i in items]
    assert any("myskill" in t for t in texts)
    # Built-in commands like /model don't start with /mys
    assert not any("model" in t for t in texts)


def test_opted_out_skill_not_in_completer(tmp_path):
    """A skill with user-invocable: false does NOT appear in Tab completions.

    CC default: user-invocable is true. Opt out with user-invocable: false.
    """
    from unittest.mock import MagicMock

    from nooa_cli.tui.commands import CommandRegistry

    skill_dir = tmp_path / "internal"
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(
        "---\nname: internal\ndescription: Internal tool\nuser-invocable: false\n---\nBody.\n"
    )

    agent = MagicMock()
    agent.get_summarization_status = MagicMock(
        return_value={
            "active_events": 0,
            "policy": "auto",
            "has_summarizer": False,
            "max_tokens": 100000,
            "current_tokens": 0,
            "preserve_recent": 5,
            "summary_count": 0,
            "summary_tags": [],
        }
    )
    config = MagicMock()
    config.default_model = "test"
    frontend = MagicMock()

    registry = CommandRegistry(
        config=config,
        agent=agent,
        frontend=frontend,
        skills_dirs=[tmp_path],
        mcp_file=None,
    )
    completer = Completer(registry=registry)
    items = completer.complete("/")
    texts = [i.text for i in items]
    assert not any("internal" in t for t in texts), (
        f"Opted-out skill should not appear in completions, but found in: {texts}"
    )


def _make_skill_registry(tmp_path, skill_name: str, skill_md_text: str):
    """Helper: create a CommandRegistry with a single skill."""
    from unittest.mock import MagicMock

    from nooa_cli.tui.commands import CommandRegistry

    skill_dir = tmp_path / skill_name
    skill_dir.mkdir()
    (skill_dir / "SKILL.md").write_text(skill_md_text)

    agent = MagicMock()
    agent.get_summarization_status = MagicMock(
        return_value={
            "active_events": 0,
            "policy": "auto",
            "has_summarizer": False,
            "max_tokens": 100000,
            "current_tokens": 0,
            "preserve_recent": 5,
            "summary_count": 0,
            "summary_tags": [],
        }
    )
    config = MagicMock()
    config.default_model = "test"
    frontend = MagicMock()

    return CommandRegistry(
        config=config,
        agent=agent,
        frontend=frontend,
        skills_dirs=[tmp_path],
        mcp_file=None,
    )


def test_bracket_hint_not_inserted(tmp_path):
    """[label] style argument hints must NOT be inserted when Tab-completing.

    /wtf-status has argument-hint: [label].  Selecting the completion should
    insert '/wtf-status' only — not '/wtf-status [label]'.
    """
    registry = _make_skill_registry(
        tmp_path,
        "wtf-status",
        "---\nname: wtf-status\ndescription: Show project status\n"
        "argument-hint: [label]\n---\nShow status.\n",
    )
    completer = Completer(registry=registry)
    items = completer.complete("/wtf-status")

    assert items, "Expected at least one completion for /wtf-status"
    item = next((i for i in items if "wtf-status" in i.text), None)
    assert item is not None, f"No wtf-status item found in: {[i.text for i in items]}"
    assert item.text == "/wtf-status", f"Hint must not be inserted; got text={item.text!r}"


def test_bracket_hint_shown_in_display(tmp_path):
    """[label] style argument hints must appear in the display (menu) string."""
    registry = _make_skill_registry(
        tmp_path,
        "wtf-status",
        "---\nname: wtf-status\ndescription: Show project status\n"
        "argument-hint: [label]\n---\nShow status.\n",
    )
    completer = Completer(registry=registry)
    items = completer.complete("/wtf")

    item = next((i for i in items if "wtf-status" in i.text), None)
    assert item is not None, f"No wtf-status item found in: {[i.text for i in items]}"
    assert "[label]" in item.display, (
        f"Hint should be visible in the dropdown; got display={item.display!r}"
    )


def test_angle_hint_shown_in_display(tmp_path):
    """<action> style argument hints must appear in the display (menu) string."""
    registry = _make_skill_registry(
        tmp_path,
        "wtf",
        "---\nname: wtf\ndescription: Manage issues\nargument-hint: <action>\n---\nManage.\n",
    )
    completer = Completer(registry=registry)
    items = completer.complete("/wtf")

    item = next((i for i in items if i.text == "/wtf"), None)
    assert item is not None, f"No /wtf item found in: {[i.text for i in items]}"
    assert "<action>" in item.display, (
        f"Hint should be visible in the dropdown; got display={item.display!r}"
    )


# ---------------------------------------------------------------------------
# Inline @ file/dir mentions
# ---------------------------------------------------------------------------


def test_mention_completion_at_start(completer):
    """An @ at the start of the buffer completes files/dirs under the typed path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "notes.md").touch()
        Path(tmpdir, "sub").mkdir()

        items = completer.complete(f"@{tmpdir}/")
        displays = [i.display for i in items]
        assert any("notes.md" in d for d in displays)
        assert any("sub/" in d for d in displays)

        # Every replacement keeps everything up to and including the @.
        for item in items:
            assert item.text.startswith(f"@{tmpdir}/"), item.text


def test_mention_completion_inline(completer):
    """An @ typed mid-sentence completes against the path after it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "alpha.py").touch()
        Path(tmpdir, "beta.py").touch()

        text = f"please read @{tmpdir}/al"
        items = completer.complete(text)
        assert len(items) == 1
        # Replacement preserves the leading sentence and the @.
        assert items[0].text == f"please read @{tmpdir}/alpha.py"


def test_mention_completion_in_slash_command_uses_agent_cwd(tmp_path, mock_registry):
    """An explicit @ argument overrides slash routing and resolves from the workspace."""
    (tmp_path / "alpha.py").touch()
    mock_registry.agent.cwd = tmp_path
    completer = Completer(mock_registry)

    items = completer.complete("/plan @al")

    assert [item.text for item in items] == ["/plan @alpha.py"]


def test_mention_only_last_token(completer):
    """An earlier @ in the buffer must not hijack completion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "x.py").touch()
        # Cursor is right after the trailing space — no active mention.
        items = completer.complete(f"@{tmpdir}/x.py ")
        assert items == []


def test_mention_requires_boundary(completer):
    """An @ not at a word boundary (e.g. an email) is not a mention."""
    assert completer.complete("user@example") == []


# ---------------------------------------------------------------------------
# Mention expansion (submit-time markdown substitution)
# ---------------------------------------------------------------------------


def test_expand_mentions_file():
    """A submitted @file mention becomes a Markdown link to its absolute path."""
    from nooa_cli.tui.completer import expand_mentions

    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir, "blah.md")
        f.touch()
        rel = str(f)
        out = expand_mentions(f"see @{rel} for details")
        assert out == f"see [{rel}](<{f.resolve()}>) for details"


def test_expand_mentions_relative_to_agent_cwd(tmp_path):
    """Relative mentions resolve from the configured workspace, not process cwd."""
    from nooa_cli.tui.completer import expand_mentions

    f = tmp_path / "notes.md"
    f.touch()

    assert expand_mentions("read @notes.md", base_dir=tmp_path) == (
        f"read [notes.md](<{f.resolve()}>)"
    )


def test_expand_mentions_directory_strips_trailing_slash():
    """A directory mention drops its trailing slash in the link label."""
    from nooa_cli.tui.completer import expand_mentions

    with tempfile.TemporaryDirectory() as tmpdir:
        d = Path(tmpdir, "docs")
        d.mkdir()
        out = expand_mentions(f"@{d}/")
        assert out == f"[{d}](<{d.resolve()}>)"


def test_expand_mentions_nonexistent_untouched():
    """A mention that resolves to no file/dir is left verbatim."""
    from nooa_cli.tui.completer import expand_mentions

    text = "ping @nope/does/not/exist now"
    assert expand_mentions(text) == text


def test_expand_mentions_email_untouched():
    """An email address (@ not at a word boundary) is not treated as a mention."""
    from nooa_cli.tui.completer import expand_mentions

    text = "contact user@example.com please"
    assert expand_mentions(text) == text


def test_expand_mentions_multiple():
    """Every resolvable @ mention in a line is expanded independently."""
    from nooa_cli.tui.completer import expand_mentions

    with tempfile.TemporaryDirectory() as tmpdir:
        a = Path(tmpdir, "a.txt")
        b = Path(tmpdir, "b.txt")
        a.touch()
        b.touch()
        out = expand_mentions(f"@{a} and @{b}")
        assert f"[{a}](<{a.resolve()}>)" in out
        assert f"[{b}](<{b.resolve()}>)" in out


# ---------------------------------------------------------------------------
# Bang command-name completion
# ---------------------------------------------------------------------------


def test_bang_command_completes_executables(completer, monkeypatch, tmp_path):
    """The first token of !cmd completes $PATH executables, not files."""
    import stat

    exe = tmp_path / "mytool"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    # A non-executable file with the same prefix must be excluded.
    (tmp_path / "mytool.txt").write_text("x")

    monkeypatch.setenv("PATH", str(tmp_path))
    items = completer.complete("!myto")
    displays = [i.display for i in items]
    assert "mytool" in displays
    assert "mytool.txt" not in displays
    # Selecting inserts the command plus a trailing space, ready for args.
    item = next(i for i in items if i.display == "mytool")
    assert item.text == "!mytool "


def test_bang_argument_completes_paths(completer):
    """After the command + a space, completion switches to filesystem paths."""
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "data.csv").touch()
        items = completer.complete(f"!cat {tmpdir}/")
        displays = [i.display for i in items]
        assert any("data.csv" in d for d in displays)


# ---------------------------------------------------------------------------
# Regression: empty-arg bang completion + trailing-punctuation mentions
# ---------------------------------------------------------------------------


def test_bang_empty_arg_completes_cwd_paths(completer, tmp_path, monkeypatch):
    """'!ls ' (command + trailing space, no arg yet) completes cwd paths.

    Regression for the branch that sent any space-free command token to
    $PATH completion: a trailing space ends the command token, so the next
    Tab must offer filesystem paths, not nothing.
    """
    (tmp_path / "data.csv").touch()
    monkeypatch.chdir(tmp_path)
    items = completer.complete("!ls ")
    displays = [i.display for i in items]
    assert any("data.csv" in d for d in displays), displays


def test_expand_mentions_trailing_punctuation():
    """A mention followed by sentence punctuation still resolves."""
    from nooa_cli.tui.completer import expand_mentions

    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir, "file.md")
        f.touch()
        out = expand_mentions(f"see @{f}. thanks")
        # The path resolves; the trailing '.' stays outside the link.
        assert f"[{f}](<{f.resolve()}>)." in out, out


def test_expand_mentions_angle_brackets_target():
    """The link target is angle-bracketed so a ')' in a path can't break out."""
    from nooa_cli.tui.completer import expand_mentions

    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir, "weird(name).md")
        f.touch()
        out = expand_mentions(f"@{f}")
        assert f"(<{f.resolve()}>)" in out, out


# ---------------------------------------------------------------------------
# MCP server-name completion (/mcp connect | /mcp disconnect)
# ---------------------------------------------------------------------------


def _mcp_registry(servers, connected=()):
    """A registry whose agent.mcp is a stub MCPRegistry (discovered/connected)."""
    from unittest.mock import MagicMock

    reg = MagicMock()
    mcp = MagicMock()
    mcp.discovered.return_value = sorted(servers)
    mcp.connected.return_value = sorted(connected)
    mcp._is_approved.return_value = False
    reg.agent.mcp = mcp
    return reg


def test_mcp_lists_lifecycle_actions():
    completer = Completer(registry=_mcp_registry([]))
    items = completer.complete("/mcp ")

    assert [item.text for item in items] == [
        "/mcp list",
        "/mcp add",
        "/mcp remove",
        "/mcp connect",
        "/mcp approve",
        "/mcp revoke",
        "/mcp disconnect",
    ]
    assert all(item.description for item in items)


def test_mcp_action_completion_filters_partial_input():
    completer = Completer(registry=_mcp_registry([]))

    assert [item.text for item in completer.complete("/mcp ap")] == ["/mcp approve"]


def test_mcp_connect_lists_configured_servers():
    reg = _mcp_registry(["maas-jira", "maas-confluence"])
    completer = Completer(registry=reg)
    items = completer.complete("/mcp connect ")
    texts = [i.text for i in items]
    assert "/mcp connect maas-jira" in texts
    assert "/mcp connect maas-confluence" in texts


def test_mcp_connect_partial_filters():
    reg = _mcp_registry(["maas-jira", "maas-confluence", "langfuse"])
    completer = Completer(registry=reg)
    items = completer.complete("/mcp connect maas-c")
    texts = [i.text for i in items]
    assert texts == ["/mcp connect maas-confluence"]


def test_mcp_connect_marks_connected_status():
    reg = _mcp_registry(["maas-jira", "langfuse"], connected=["langfuse"])
    completer = Completer(registry=reg)
    by_text = {i.text: i.description for i in completer.complete("/mcp connect ")}
    assert by_text["/mcp connect langfuse"] == "connected"
    assert by_text["/mcp connect maas-jira"] == "approval required"


def test_mcp_disconnect_only_offers_connected():
    reg = _mcp_registry(["maas-jira", "langfuse"], connected=["langfuse"])
    completer = Completer(registry=reg)
    texts = [i.text for i in completer.complete("/mcp disconnect ")]
    assert texts == ["/mcp disconnect langfuse"]


def test_mcp_remove_offers_configured_servers():
    reg = _mcp_registry(["maas-jira", "langfuse"], connected=["langfuse"])
    completer = Completer(reg)
    texts = [item.text for item in completer.complete("/mcp remove ")]
    assert texts == ["/mcp remove langfuse", "/mcp remove maas-jira"]


def test_mcp_approve_and_revoke_complete_configured_servers():
    reg = _mcp_registry(["maas-jira"])
    completer = Completer(registry=reg)
    assert [item.text for item in completer.complete("/mcp approve ")] == ["/mcp approve maas-jira"]
    assert [item.text for item in completer.complete("/mcp revoke ")] == ["/mcp revoke maas-jira"]


def test_mcp_completion_marks_approved_server():
    reg = _mcp_registry(["maas-jira"])
    reg.agent.mcp._is_approved.return_value = True
    item = Completer(registry=reg).complete("/mcp connect ")[0]
    assert item.description == "approved"


def test_mcp_completion_missing_command_is_safe():
    from unittest.mock import MagicMock

    reg = MagicMock()
    reg.agent.mcp = None
    completer = Completer(registry=reg)
    assert completer.complete("/mcp connect ") == []


def test_mcp_completion_quotes_server_names_with_spaces_and_omits_unsafe_names():
    reg = _mcp_registry(["safe-server", "evil\x1b[2J", "MaaS Jira"])
    items = Completer(registry=reg).complete("/mcp connect ")

    assert [item.text for item in items] == [
        "/mcp connect 'MaaS Jira'",
        "/mcp connect safe-server",
    ]
    assert all("\x1b" not in item.display for item in items)
