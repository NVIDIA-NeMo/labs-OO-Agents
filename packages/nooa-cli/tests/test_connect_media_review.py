# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real rendered choice menus and keyboard input, without a worker pool or HTTP."""

import click
import pytest
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands import _connect_wizard as wizard
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.shortcuts.choice_input import ChoiceInput

from nooa.unifiedllm.connect import ConnectResult


@pytest.fixture
def terminal_menu(monkeypatch):
    """Send keys only after the initial menu actually reaches the renderer."""
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(prompts.sys.stdin, "isatty", lambda: True)
    original = ChoiceInput._create_application

    def run(keys, callback):
        screens = []
        with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):

            def create(choice):
                app = original(choice)
                sent = False

                def rendered(_):
                    nonlocal sent
                    screen = app.renderer.last_rendered_screen
                    if screen is None:  # The final render clears the cached screen.
                        return
                    screens.append(
                        "\n".join(
                            "".join(row[x].char for x in sorted(row))
                            for _, row in sorted(screen.data_buffer.items())
                        )
                    )
                    if not sent:
                        sent = True
                        pipe.send_text(keys)

                app.after_render += rendered
                return app

            monkeypatch.setattr(ChoiceInput, "_create_application", create)
            result = callback()
        return result, screens

    return run


@pytest.mark.parametrize("current", [[], ["image", "pdf"], ["audio", "video", "image"]])
@pytest.mark.parametrize(
    "keys,expected", [("\r", "keep"), ("\x1b[B\r", "manual"), ("\x1b[B\x1b[A\r", "keep")]
)
def test_rendered_menu_labels_and_arrow_selection(terminal_menu, current, keys, expected):
    result, screens = terminal_menu(keys, lambda: prompts.review_media_inputs(current))
    assert result == expected
    initial = screens[0]
    assert "How would you like to continue?" in initial
    label = "Continue with " + (", ".join(current) or "text only")
    assert label in initial and "Choose media inputs manually" in initial
    assert ">  1. " + label in initial  # The visible selection is the Enter default.
    for screen in screens:
        assert "keep" not in screen and "Enter to use" not in screen
        assert "→ to edit" not in screen and "Review media inputs:" not in screen
        assert "manual:" not in screen


def test_rendered_menu_ctrl_c_aborts(terminal_menu):
    with pytest.raises(click.Abort):
        terminal_menu("\x03", lambda: prompts.review_media_inputs(["image"]))


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("\n", "keep"),
        ("1\n", "keep"),
        ("keep\n", "keep"),
        ("2\n", "manual"),
        ("manual\n", "manual"),
        ("bad\n2\n", "manual"),
    ],
)
@pytest.mark.parametrize("current", [[], ["image", "pdf"]])
def test_pipe_menu_labels_and_compatible_answers(answer, expected, current):
    selected = []

    @click.command()
    def command():
        selected.append(prompts.review_media_inputs(current))

    result = CliRunner().invoke(command, input=answer)
    assert result.exit_code == 0, result.output
    assert selected == [expected]
    assert "How would you like to continue?" in result.output
    assert "1 (keep): Continue with " + (", ".join(current) or "text only") in " ".join(
        result.output.split()
    )
    assert "2 (manual): Choose media inputs manually" in " ".join(result.output.split())


def review_state(path, current, *, yes=False):
    return wizard.WizardState(
        alias="local",
        model="openai/test",
        api_style="chat",
        path=path,
        no_probe=True,
        yes=yes,
        test_media_inputs=["image"],
        result=ConnectResult(
            alias="local",
            entry={
                "model_name": "openai/test",
                "api_style": "chat",
                "max_tokens": 1000,
                "media_inputs": list(current),
                "provenance": {"probes": {}},
            },
        ),
    )


@pytest.mark.parametrize("current", [[], ["image", "pdf"]])
@pytest.mark.parametrize("keys,manual", [("\r", False), ("\x1b[B\r", True)])
def test_tty_review_only_manual_opens_format_prompt(
    terminal_menu, monkeypatch, tmp_path, current, keys, manual
):
    path = tmp_path / "models.yaml"
    state = review_state(path, current)
    declarations = []
    approvals = []

    def choose(previous):
        declarations.append(list(previous))
        return ["video"]

    def decline(text, **kwargs):
        approvals.append(text)
        return False

    monkeypatch.setattr(prompts, "choose_media_inputs", choose)
    monkeypatch.setattr(prompts, "confirm", decline)
    result, _ = terminal_menu(keys, lambda: wizard.save_model(state))
    assert result is True
    assert declarations == ([current] if manual else [])
    assert state.result.entry["media_inputs"] == (["video"] if manual else current)
    assert len(approvals) == 1 and "Write model entry" in approvals[0]
    assert not path.exists()


def test_tty_ctrl_c_review_preserves_existing_file(terminal_menu, tmp_path):
    path = tmp_path / "models.yaml"
    original = "models: {}\n"
    path.write_text(original)
    state = review_state(path, ["image"])
    with pytest.raises(click.Abort):
        terminal_menu("\x03", lambda: wizard.save_model(state))
    assert path.read_text() == original
    assert state.result.entry["media_inputs"] == ["image"]


@pytest.mark.parametrize(
    "answer,expected", [("\nn\n", ["image"]), ("manual\nvideo,pdf\nn\n", ["video", "pdf"])]
)
def test_pipe_save_confirmation_still_required(tmp_path, answer, expected):
    path = tmp_path / "models.yaml"
    state = review_state(path, ["image"])

    @click.command()
    def command():
        wizard.save_model(state)

    result = CliRunner().invoke(command, input=answer)
    assert result.exit_code == 0, result.output
    assert state.result.entry["media_inputs"] == expected
    assert "Write model entry" in result.output
    assert ("Enable media inputs manually" in result.output) == ("manual" in answer)
    assert not path.exists()


def test_yes_skips_review_and_preserves_enabled_inputs(monkeypatch, tmp_path):
    import yaml

    path = tmp_path / "models.yaml"
    state = review_state(path, ["audio", "image"], yes=True)

    def forbidden(*args, **kwargs):
        pytest.fail("--yes must bypass both review and format prompts")

    monkeypatch.setattr(prompts, "review_media_inputs", forbidden)
    monkeypatch.setattr(prompts, "choose_media_inputs", forbidden)
    assert wizard.save_model(state)
    assert yaml.safe_load(path.read_text())["models"]["local"]["media_inputs"] == ["audio", "image"]


@pytest.mark.parametrize(
    "editing,keys,expected",
    [
        (False, "\r", "test"),
        (False, "\x1b[B\r", "manual"),
        (False, "\x1b[B\x1b[A\r", "test"),
        (True, "\r", "manual"),
        (True, "\x1b[A\r", "test"),
        (True, "\x1b[A\x1b[B\r", "manual"),
    ],
)
def test_initial_rendered_menu_defaults_labels_and_arrows(terminal_menu, editing, keys, expected):
    result, screens = terminal_menu(keys, lambda: prompts.choose_media_mode(editing=editing))
    assert result == expected
    initial = screens[0]
    assert "How would you like to configure media inputs?" in initial
    assert "Test all media inputs (generated samples)" in initial
    assert "Choose media inputs manually" in initial
    selected = "2. Choose media inputs manually" if editing else "1. Test all media inputs"
    assert ">  " + selected in initial
    for screen in screens:
        assert "Enter to use" not in screen and "→ to edit" not in screen
        assert "Media inputs:" not in screen
        # Labels can mention tests/manual selection, but internal lowercase tokens
        # must not appear as standalone values or editable placeholders.
        assert "test" not in screen.split() and "manual" not in screen.split()
        assert "test:" not in screen and "manual:" not in screen


@pytest.mark.parametrize("editing", [False, True])
def test_initial_tty_ctrl_c_preserves_file_and_configuration(terminal_menu, tmp_path, editing):
    path = tmp_path / "models.yaml"
    original = "models: {}\n"
    path.write_text(original)
    previous = {"media_inputs": ["audio"]} if editing else None
    state = wizard.WizardState(path=path, editing=previous, approval="all", yes=False)
    with pytest.raises(click.Abort):
        terminal_menu("\x03", lambda: wizard.configure_media_inputs(state))
    assert path.read_text() == original
    assert state.media_inputs is None and state.test_media_inputs is None
    if editing:
        assert previous == {"media_inputs": ["audio"]}


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize(
    "answer,expected",
    [
        ("\n", None),
        ("1\n", "test"),
        ("test\n", "test"),
        ("2\n", "manual"),
        ("manual\n", "manual"),
        ("bad\n2\n", "manual"),
    ],
)
def test_initial_pipe_menu_defaults_and_compatible_answers(editing, answer, expected):
    selected = []

    @click.command()
    def command():
        selected.append(prompts.choose_media_mode(editing=editing))

    result = CliRunner().invoke(command, input=answer)
    assert result.exit_code == 0, result.output
    assert selected == [expected or ("manual" if editing else "test")]
    assert "How would you like to configure media inputs?" in result.output
    assert "1 (test): Test all media inputs (generated samples)" in " ".join(result.output.split())
    assert "2 (manual): Choose media inputs manually" in " ".join(result.output.split())
    assert f"Choice [{2 if editing else 1}]" in result.output


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize("mode", ["test", "manual"])
def test_initial_tty_wizard_only_manual_asks_for_formats(terminal_menu, monkeypatch, editing, mode):
    previous = {"media_inputs": ["audio"]} if editing else None
    state = wizard.WizardState(editing=previous, approval="all", yes=False)
    declarations = []

    def choose(current):
        declarations.append(list(current))
        return ["pdf"]

    monkeypatch.setattr(prompts, "choose_media_inputs", choose)
    default = "manual" if editing else "test"
    keys = "\r" if mode == default else ("\x1b[A\r" if editing else "\x1b[B\r")
    result, _ = terminal_menu(keys, lambda: wizard.configure_media_inputs(state))
    assert result is True
    assert declarations == ([(["audio"] if editing else [])] if mode == "manual" else [])
    assert state.media_inputs == (["pdf"] if mode == "manual" else ["audio"] if editing else [])
    assert state.test_media_inputs == (
        ["image", "audio", "video", "pdf"] if mode == "test" else None
    )
    if editing:
        assert previous == {"media_inputs": ["audio"]}


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize("explicit_tests", [False, True])
def test_yes_skips_initial_menu_and_format_questions(monkeypatch, editing, explicit_tests):
    def forbidden(*args, **kwargs):
        pytest.fail("--yes must bypass initial media questions")

    monkeypatch.setattr(prompts, "choose_media_mode", forbidden)
    monkeypatch.setattr(prompts, "choose_media_inputs", forbidden)
    state = wizard.WizardState(
        editing={"media_inputs": ["audio"]} if editing else None,
        approval="all",
        yes=True,
        test_media_inputs="image,pdf" if explicit_tests else None,
    )
    assert wizard.configure_media_inputs(state)
    assert state.media_inputs == (["audio"] if editing else [])
    assert state.test_media_inputs == (["image", "pdf"] if explicit_tests else None)


@pytest.mark.parametrize(
    "editing,answers,manual",
    [
        (False, "\n", False),
        (False, "test\n", False),
        (False, "1\n", False),
        (False, "manual\npdf\n", True),
        (False, "2\npdf\n", True),
        (True, "\n\n", True),
        (True, "manual\n\n", True),
        (True, "test\n", False),
        (True, "1\n", False),
    ],
)
def test_scripted_initial_wizard_paths_do_not_add_format_quiz(editing, answers, manual):
    state = wizard.WizardState(
        editing={"media_inputs": ["audio"]} if editing else None,
        approval="all",
        yes=False,
    )

    @click.command()
    def command():
        assert wizard.configure_media_inputs(state)

    result = CliRunner().invoke(command, input=answers)
    assert result.exit_code == 0, result.output
    assert result.output.count("How would you like to configure media inputs?") == 1
    assert result.output.count("Enable media inputs manually") == int(manual)
    assert "Formats to TEST" not in result.output
    assert state.media_inputs == (["audio"] if editing else ["pdf"] if manual else [])
    assert state.test_media_inputs == (None if manual else ["image", "audio", "video", "pdf"])


@pytest.mark.parametrize("current", [[], ["image"]])
@pytest.mark.parametrize("action", ["text", "manual", "test", "default"])
def test_rendered_decision_image_actions_and_defaults(terminal_menu, current, action):
    actions = (["keep"] if current else []) + ["text", "manual", "test"]
    selected = ("keep" if current else "text") if action == "default" else action
    keys = "\x1b[B" * actions.index(selected) + "\r"
    before = current.copy()
    result, screens = terminal_menu(keys, lambda: prompts.choose_decision_images(current))
    assert result == selected and current == before
    initial = screens[0]
    assert "Should this decision model receive images?" in initial
    for label in (
        "Leave images off",
        "Enable images without testing",
        "Test images before enabling",
    ):
        assert label in initial
    default_label = "Keep current image settings" if current else "Leave images off"
    assert ">  1. " + default_label in initial
    assert ("Keep current image settings" in initial) is bool(current)
    for screen in screens:
        for jargon in ("text:", "manual:", "test:", "keep:", "unverified", "native", "wire"):
            assert jargon not in screen


@pytest.mark.parametrize("current", [[], ["image"]])
def test_rendered_decision_image_ctrl_c_preserves_inputs(terminal_menu, current):
    before = current.copy()
    with pytest.raises(click.Abort):
        terminal_menu("\x03", lambda: prompts.choose_decision_images(current))
    assert current == before


@pytest.mark.parametrize("current", [[], ["image"]])
@pytest.mark.parametrize("action", ["text", "manual", "test", "default"])
@pytest.mark.parametrize("numeric", [False, True])
def test_pipe_decision_image_actions(current, action, numeric):
    actions = (["keep"] if current else []) + ["text", "manual", "test"]
    expected = ("keep" if current else "text") if action == "default" else action
    answer = (
        "" if action == "default" else str(actions.index(expected) + 1) if numeric else expected
    )
    selected = []

    @click.command()
    def command():
        selected.append(prompts.choose_decision_images(current))

    result = CliRunner().invoke(command, input=answer + "\n")
    assert result.exit_code == 0, result.output
    assert selected == [expected]
    assert "Image tests use a generated sample." in result.output
    assert "approved check budget" not in result.output
    for index, token in enumerate(actions, 1):
        assert f"{index} ({token}):" in result.output
    assert current == (["image"] if current else [])


@pytest.mark.parametrize("answer", ["keep\n", "1\n"])
def test_pipe_decision_images_explicit_keep(answer):
    selected = []

    @click.command()
    def command():
        selected.append(prompts.choose_decision_images(["image"]))

    result = CliRunner().invoke(command, input=answer)
    assert result.exit_code == 0 and selected == ["keep"]


@pytest.mark.parametrize("previous", [[], ["image"]])
@pytest.mark.parametrize("action", ["keep", "text", "manual", "test"])
def test_decision_image_wizard_token_mapping(monkeypatch, tmp_path, previous, action):
    from copy import deepcopy

    from nooa.unifiedllm import connect
    from tests.unifiedllm.connect.connect_http import mock_http

    mock_http(monkeypatch, lambda request: pytest.fail("Configuring inputs must not send HTTP"))
    existing = connect.plan(
        "local",
        "model",
        "openai",
        "https://api.test/v1/decisions",
        "",
        media_inputs=previous,
    ).entry
    original = deepcopy(existing)
    state = wizard.WizardState(
        alias="local",
        model="model",
        api_style="openai",
        endpoint="https://api.test/v1/decisions",
        api_key_env="",
        editing=existing,
        approval="all",
        budget_tokens=9000,
        path=tmp_path / "models.yaml",
        media_hints=None,
    )
    seen = []

    def choose(current):
        seen.append(list(current))
        return action

    monkeypatch.setattr(prompts, "choose_decision_images", choose)
    assert wizard.configure_media_inputs(state)
    assert seen == [previous]
    assert state.media_inputs == (
        previous if action == "keep" else ["image"] if action == "manual" else []
    )
    assert state.test_media_inputs == (["image"] if action == "test" else None)
    assert existing == original
    assert not state.path.exists()


@pytest.mark.parametrize(
    "keys,action",
    [
        ("\r", "test"),
        ("\x1b[B\r", "suggested"),
        ("\x1b[B\x1b[B\r", "manual"),
        ("\x1b[B\x1b[B\x1b[B\r", "none"),
    ],
)
def test_listed_media_menu_requested_order_and_default(terminal_menu, monkeypatch, keys, action):
    from nooa_cli.commands._connect_advertised_media import offer_suggestions

    state = wizard.WizardState(
        model="model",
        api_style="chat",
        approval="all",
        guided=True,
        media_hints={"id": "model", "input_modalities": ["image"]},
    )
    monkeypatch.setattr(prompts, "choose_media_inputs", lambda previous: ["video"])
    accepted, screens = terminal_menu(keys, lambda: offer_suggestions(state, []))
    assert accepted is True
    labels = [
        "Enable listed inputs with test",
        "Enable listed inputs without testing",
        "Select media inputs manually",
        "Turn media inputs off",
    ]
    initial = screens[0]
    assert [initial.index(label) for label in labels] == sorted(
        initial.index(label) for label in labels
    )
    assert ">  1. Enable listed inputs with test" in initial
    assert "Keep current media" not in initial
    assert state.test_media_inputs == (["image"] if action == "test" else None)
    assert (
        state.media_inputs
        == {"test": [], "suggested": ["image"], "manual": ["video"], "none": []}[action]
    )


@pytest.mark.parametrize(
    "answer,action",
    [
        ("\n", "test"),
        ("1\n", "test"),
        ("test\n", "test"),
        ("2\n", "suggested"),
        ("3\n", "manual"),
        ("4\n", "none"),
    ],
)
def test_listed_media_pipe_order_and_default(monkeypatch, answer, action):
    from nooa_cli.commands._connect_advertised_media import offer_suggestions

    state = wizard.WizardState(
        model="model",
        api_style="chat",
        approval="all",
        guided=True,
        media_hints={"id": "model", "input_modalities": ["image"]},
    )
    monkeypatch.setattr(prompts, "choose_media_inputs", lambda previous: ["video"])

    @click.command()
    def choose():
        assert offer_suggestions(state, [])

    result = CliRunner().invoke(choose, input=answer)
    assert result.exit_code == 0, result.output
    assert "1 (test): Enable listed inputs with test" in " ".join(result.output.split())
    assert state.test_media_inputs == (["image"] if action == "test" else None)
    assert (
        state.media_inputs
        == {"test": [], "suggested": ["image"], "manual": ["video"], "none": []}[action]
    )
