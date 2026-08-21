# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Display-mode resolution and compatibility tests."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from nooa_cli.commands.tui import command
from nooa_cli.tui.config import Config, DisplayMode, TUIConfig, resolve_display_mode
from pydantic import ValidationError

from .tui_app_harness import TUIHarness


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    user = tmp_path / "user"
    project = tmp_path / "project"
    user.mkdir()
    project.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(project))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (TUIConfig(), DisplayMode.FULLSCREEN),
        (TUIConfig(full_screen=True), DisplayMode.NATIVE_REPLAY),
        (TUIConfig(full_screen=False), DisplayMode.NATIVE),
        (TUIConfig(display_mode="native-replay"), DisplayMode.NATIVE_REPLAY),
        (TUIConfig(display_mode="native"), DisplayMode.NATIVE),
        (TUIConfig(display_mode="fullscreen"), DisplayMode.FULLSCREEN),
    ],
)
def test_resolution_matrix(config: TUIConfig, expected: DisplayMode) -> None:
    assert resolve_display_mode(config) is expected


def test_new_mode_wins_conflicting_deprecated_boolean_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    config = TUIConfig(display_mode=DisplayMode.FULLSCREEN, full_screen=False)

    with caplog.at_level(logging.WARNING, logger="nooa_cli.tui.config"):
        assert resolve_display_mode(config) is DisplayMode.FULLSCREEN
        assert resolve_display_mode(config) is DisplayMode.FULLSCREEN

    conflicts = [record for record in caplog.records if "full_screen" in record.message]
    assert len(conflicts) == 1
    assert "display_mode" in conflicts[0].message


def test_compatible_deprecated_boolean_warns_once(caplog: pytest.LogCaptureFixture) -> None:
    config = TUIConfig(display_mode=DisplayMode.NATIVE, full_screen=False)

    with caplog.at_level(logging.WARNING, logger="nooa_cli.tui.config"):
        assert resolve_display_mode(config) is DisplayMode.NATIVE
        assert resolve_display_mode(config) is DisplayMode.NATIVE

    warnings = [record for record in caplog.records if "full_screen" in record.message]
    assert len(warnings) == 1
    assert "conflict" not in warnings[0].message


def test_settings_dump_canonicalizes_default_without_deprecated_boolean() -> None:
    from nooa_cli.tui.settings import settings_to_dict

    data = settings_to_dict(Config())

    assert data["tui"]["display_mode"] == "fullscreen"
    assert "full_screen" not in data["tui"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("tui:\n  full_screen: true\n", DisplayMode.NATIVE_REPLAY),
        ("tui:\n  full_screen: false\n", DisplayMode.NATIVE),
        (
            "tui:\n  display_mode: native\n  full_screen: true\n",
            DisplayMode.NATIVE,
        ),
    ],
)
def test_settings_round_trip_migrates_to_one_canonical_mode(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    source: str,
    expected: DisplayMode,
) -> None:
    from nooa_cli.tui.settings import dump_settings

    settings_path = tmp_path / "user" / "settings.yaml"
    settings_path.write_text(source)
    loaded = Config.load()
    assert resolve_display_mode(loaded.tui) is expected

    canonical = dump_settings(loaded)
    assert "full_screen" not in canonical
    assert f"display_mode: {expected.value}" in canonical

    caplog.clear()
    settings_path.write_text(canonical)
    reloaded = Config.load()
    with caplog.at_level(logging.WARNING, logger="nooa_cli.tui.config"):
        assert resolve_display_mode(reloaded.tui) is expected
    assert not [record for record in caplog.records if "full_screen" in record.message]


def test_invalid_constructor_mode_fails_validation() -> None:
    with pytest.raises(ValidationError, match="display_mode"):
        TUIConfig(display_mode="not-a-mode")


def test_layered_display_mode_beats_deprecated_boolean(tmp_path: Path, monkeypatch) -> None:
    user = tmp_path / "user"
    (user / "settings.yaml").write_text("tui:\n  full_screen: false\n  display_mode: fullscreen\n")

    config = Config.load()

    assert config.tui.model_fields_set >= {"full_screen", "display_mode"}
    assert resolve_display_mode(config.tui) is DisplayMode.FULLSCREEN


def test_invalid_layered_mode_fails_clearly(tmp_path: Path) -> None:
    (tmp_path / "user" / "settings.yaml").write_text("tui:\n  display_mode: bogus\n")

    with pytest.raises(ValueError, match="bogus"):
        Config.load()


def test_cli_display_mode_has_highest_precedence(tmp_path: Path) -> None:
    (tmp_path / "project" / "settings.yaml").write_text(
        "tui:\n  display_mode: native\n  full_screen: false\n"
    )
    seen: list[DisplayMode] = []

    async def fake_main(*, config, **_kwargs):
        seen.append(resolve_display_mode(config.tui))

    with patch("nooa_cli.tui.main.main", fake_main):
        result = CliRunner().invoke(command, ["--display-mode", "fullscreen"])

    assert result.exit_code == 0, result.output
    assert seen == [DisplayMode.FULLSCREEN]


def test_omitted_cli_mode_preserves_settings_mode(tmp_path: Path) -> None:
    (tmp_path / "project" / "settings.yaml").write_text("tui:\n  display_mode: native\n")
    seen: list[DisplayMode] = []

    async def fake_main(*, config, **_kwargs):
        seen.append(resolve_display_mode(config.tui))

    with patch("nooa_cli.tui.main.main", fake_main):
        result = CliRunner().invoke(command, [])

    assert result.exit_code == 0, result.output
    assert seen == [DisplayMode.NATIVE]


def test_cli_rejects_invalid_display_mode() -> None:
    result = CliRunner().invoke(command, ["--display-mode", "bogus"])

    assert result.exit_code == 2
    assert "Invalid value for '--display-mode'" in result.output


def test_cli_help_documents_modes_and_restart_requirement() -> None:
    result = CliRunner().invoke(command, ["--help"])

    assert result.exit_code == 0
    assert "--display-mode" in result.output
    assert "native-replay" in result.output
    assert "fullscreen" in result.output
    assert "restart" in result.output.lower()


def test_programmatic_override_is_typed_and_highest_precedence(tmp_path: Path) -> None:
    (tmp_path / "project" / "settings.yaml").write_text("tui:\n  display_mode: native\n")

    config = Config.load(display_mode="native-replay")

    assert config.tui.display_mode is DisplayMode.NATIVE_REPLAY
    assert resolve_display_mode(config.tui) is DisplayMode.NATIVE_REPLAY


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "legacy_replay"),
    [
        (DisplayMode.NATIVE_REPLAY, True),
        (DisplayMode.NATIVE, False),
    ],
)
async def test_native_modes_keep_inline_prompt_toolkit_shell(
    mode: DisplayMode, legacy_replay: bool
) -> None:
    async with TUIHarness(display_mode=mode) as harness:
        assert harness.app.display_mode is mode
        assert harness.app.full_screen is legacy_replay
        assert harness.app._app.full_screen is False


def test_fullscreen_selects_alternate_screen_renderer() -> None:
    from nooa_cli.tui.tui_application import TUIApplication
    from prompt_toolkit.application import create_app_session
    from prompt_toolkit.input import DummyInput
    from prompt_toolkit.output import DummyOutput

    with create_app_session(input=DummyInput(), output=DummyOutput()):
        app = TUIApplication(display_mode=DisplayMode.FULLSCREEN)

    assert app._app.full_screen is True
    assert app.full_screen is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("legacy", "expected"),
    [
        (True, DisplayMode.NATIVE_REPLAY),
        (False, DisplayMode.NATIVE),
    ],
)
async def test_deprecated_application_constructor_maps_exactly(
    legacy: bool,
    expected: DisplayMode,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="nooa_cli.tui.config"):
        async with TUIHarness(full_screen=legacy) as harness:
            assert harness.app.display_mode is expected
            assert harness.app._app.full_screen is False

    warnings = [record for record in caplog.records if "full_screen" in record.message]
    assert len(warnings) == 1
