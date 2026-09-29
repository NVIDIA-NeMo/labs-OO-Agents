# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the real terminal editor, including completion and cursor keys."""

import pytest
from nooa_cli.commands import _connect_prompts as prompts
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput


@pytest.fixture(autouse=True)
def terminal_type(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")


def answer(monkeypatch, keys, **kwargs):
    import prompt_toolkit
    from prompt_toolkit.application import get_app

    monkeypatch.setattr(prompts.sys.stdin, "isatty", lambda: True)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        if "\t" in keys:
            # Wait for the menu, as a user would, before selecting a completion.
            before, after = keys.split("\t", 1)
            original = prompt_toolkit.prompt

            def run(*args, **options):
                def ready():
                    buffer = get_app().current_buffer

                    def menu_opened(_):
                        if buffer.complete_state:
                            buffer.on_completions_changed -= menu_opened
                            pipe.send_text("\t" + after)

                    buffer.on_completions_changed += menu_opened
                    pipe.send_text(before)

                return original(*args, pre_run=ready, **options)

            monkeypatch.setattr(prompt_toolkit, "prompt", run)
        else:
            pipe.send_text(keys)
        return prompts.prompt("Choose", **kwargs)


def test_model_completion_matches_inside_long_ids(monkeypatch):
    assert (
        answer(monkeypatch, "qwen\t\r", choices=["vendor/model-one", "vendor/qwen-model"])
        == "vendor/qwen-model"
    )


def test_environment_completion_only_uses_names(monkeypatch):
    monkeypatch.setenv("CONNECT_COMPLETION_KEY", "secret-never-completed")
    names = prompts.environment_names(["ANOTHER_KEY"])
    assert "CONNECT_COMPLETION_KEY" in names
    assert "ANOTHER_KEY" in names
    assert "secret-never-completed" not in names
    assert (
        answer(monkeypatch, "CONNECT_COMPLETION\t\r", suggestions=names) == "CONNECT_COMPLETION_KEY"
    )


@pytest.mark.parametrize(
    "keys,expected",
    [
        ("\x1b[C\x1b[D\x1b[DZ\r", "abZcd"),  # Right accepts the default for editing.
        ("\x1b[C\x1b[H\x1b[C\x1b[3~\x1b[F!\r", "acd!"),  # Home, Right, Delete, End.
    ],
)
def test_prefilled_text_is_editable(monkeypatch, keys, expected):
    assert answer(monkeypatch, keys, default="abcd") == expected


def test_enter_accepts_default_but_typing_replaces_it(monkeypatch):
    assert answer(monkeypatch, "\r", default="default-model") == "default-model"
    assert answer(monkeypatch, "my-model\r", default="default-model") == "my-model"


def test_alias_collision_hint_tracks_input_and_default(monkeypatch):
    import prompt_toolkit
    from prompt_toolkit.application import get_app
    from prompt_toolkit.formatted_text import to_plain_text

    monkeypatch.setattr(prompts.sys.stdin, "isatty", lambda: True)
    original = prompt_toolkit.prompt
    hints = []
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):

        def run(*args, **kwargs):
            hint = kwargs["rprompt"]

            def ready():
                hints.append(to_plain_text(hint()))  # The default name already exists.
                get_app().current_buffer.text = "fresh-alias"
                hints.append(to_plain_text(hint()))
                pipe.send_text("\r")

            return original(*args, pre_run=ready, **kwargs)

        monkeypatch.setattr(prompt_toolkit, "prompt", run)
        assert prompts.prompt("Alias", default="taken", existing=("taken",)) == "fresh-alias"
    assert "already exists" in hints[0]
    assert hints[1] == ""


def test_completion_does_not_submit_a_confirmation(monkeypatch):
    monkeypatch.setattr(prompts.sys.stdin, "isatty", lambda: True)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text("\r")
        assert prompts.confirm("Spend?", default=False) is False


def test_confirmation_accepts_yes_without_deleting_default(monkeypatch):
    monkeypatch.setattr(prompts.sys.stdin, "isatty", lambda: True)
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):
        pipe.send_text("y\r")
        assert prompts.confirm("Spend?", default=False) is True


def test_secret_prompt_has_no_completer_or_history(monkeypatch):
    import prompt_toolkit

    monkeypatch.setattr(prompts.sys.stdin, "isatty", lambda: True)
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return "secret"

    monkeypatch.setattr(prompt_toolkit, "prompt", capture)
    assert prompts.prompt("Key", hide_input=True, suggestions=["must-not-appear"]) == "secret"
    assert seen["is_password"] is True
    assert seen["completer"] is None
    assert list(seen["history"].get_strings()) == []


def test_cancel_uses_click_abort(monkeypatch):
    import click

    with pytest.raises(click.Abort):
        answer(monkeypatch, "\x03")


@pytest.mark.parametrize("urls", [False, True])
def test_menu_opens_and_arrow_enter_selects_without_typing(monkeypatch, urls):
    import prompt_toolkit
    from prompt_toolkit.application import get_app

    monkeypatch.setattr(prompts.sys.stdin, "isatty", lambda: True)
    original = prompt_toolkit.prompt
    with create_pipe_input() as pipe, create_app_session(input=pipe, output=DummyOutput()):

        def run(*args, **kwargs):
            open_menu = kwargs.pop("pre_run")

            def ready():
                buffer = get_app().current_buffer

                def menu_opened(_):
                    if buffer.complete_state:
                        buffer.on_completions_changed -= menu_opened
                        pipe.send_text("\x1b[B\r")

                buffer.on_completions_changed += menu_opened
                open_menu()

            return original(*args, pre_run=ready, **kwargs)

        monkeypatch.setattr(prompt_toolkit, "prompt", run)
        options = (
            {"suggestions": ("https://first.example/v1", "https://second.example/v1")}
            if urls
            else {
                "choices": ("nvidia", "custom"),
                "labels": {"nvidia": "NVIDIA · build.nvidia.com", "custom": "Custom endpoint"},
            }
        )
        value = prompts.prompt(
            "Model server URL" if urls else "Provider",
            **options,
            open_menu=True,
        )
    assert value == ("https://first.example/v1" if urls else "nvidia")


def test_help_can_be_toggled_without_losing_edited_text(monkeypatch):
    # F1, F1 closes help; the input buffer survives both transitions.
    assert answer(monkeypatch, "abc\x1bOP\x1bOPd\r") == "abcd"
