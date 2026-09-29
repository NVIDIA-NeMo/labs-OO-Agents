# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Editable terminal prompts; the onboarding library has no terminal dependencies.

Only suggested names enter completion, never environment values. Each prompt
has disposable history, and secrets have neither history nor completion. Pipes
use Click's line input so scripted answers continue to work without a terminal.
"""

import os
import sys

import click


def environment_names(defaults=()):
    """Suggest variable names, including conventional names not yet exported."""
    return sorted(set(os.environ).union(defaults))


def prompt(
    text,
    *,
    default=None,
    choices=(),
    suggestions=(),
    hide_input=False,
    show_default=True,
    labels=None,
    open_menu=False,
    existing=(),
):
    """Read one editable answer with a scrolling, single-column completion menu."""
    choices = tuple(choices)
    if not sys.stdin.isatty():
        if labels:
            for value, label in labels.items():
                click.echo(f"  {value}: {label}")
        while True:
            value = click.prompt(
                text, default=default, hide_input=hide_input, show_default=show_default
            )
            if not choices or value in choices:
                return value
            # Click.Choice includes every model ID in its prompt and error.
            click.echo(
                "Choose an available value; check the spelling of the model or provider name.",
                err=True,
            )

    from prompt_toolkit import prompt as terminal_prompt
    from prompt_toolkit.application import get_app
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.history import DummyHistory
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.output import ColorDepth
    from prompt_toolkit.styles import Style
    from prompt_toolkit.validation import Validator

    words = list(dict.fromkeys(choices or suggestions))
    completer = (
        WordCompleter(
            words, ignore_case=True, match_middle=True, sentence=True, display_dict=labels
        )
        if words and not hide_input
        else None
    )
    default_text = "" if default is None else str(default)

    def effective(value):
        return value or default_text

    validator = Validator.from_callable(
        lambda value: (
            effective(value) in choices if choices else bool(effective(value)) or default == ""
        ),
        error_message="Choose a matching value." if choices else "Enter a value.",
        move_cursor_to_end=False,
    )
    help_open = False
    bindings = KeyBindings()

    @bindings.add(
        "right",
        filter=Condition(
            lambda: bool(default_text) and not hide_input and not get_app().current_buffer.text
        ),
    )
    def edit_default(event):
        event.current_buffer.insert_text(default_text)

    existing = frozenset(existing)

    def collision_hint():
        if effective(get_app().current_buffer.text) in existing:
            return FormattedText([("class:warning", "Name already exists · confirmation required")])
        return ""

    @bindings.add("f1")
    def help_toggle(event):
        nonlocal help_open
        help_open = not help_open
        event.app.invalidate()

    def toolbar():
        if help_open:
            return FormattedText(
                [
                    (
                        "class:help",
                        " No paid calls: restart with --no-probe\n"
                        " Manual setup: docs/model-configuration.md\n"
                        " Agent skill: nooa-model-configuration\n"
                        " Enter uses the suggested default. Typing replaces it. Right Arrow edits it.\n"
                        " Arrow keys / Home / End edit. Ctrl-U clears. F1 closes help.",
                    )
                ]
            )
        return FormattedText(
            [
                (
                    "class:hint",
                    " Enter Continue   ←→ Edit   Ctrl-C Cancel   F1 Help"
                    if hide_input
                    else " ↑↓ Choose   Tab Complete   Enter Continue   Ctrl-C Cancel   F1 Help",
                )
            ]
        )

    def start_menu():
        get_app().current_buffer.start_completion(select_first=False)

    try:
        value = terminal_prompt(
            FormattedText([("class:prompt", text + ": ")]),
            default="",
            placeholder=FormattedText(
                [("class:hint", default_text + "  (Enter to use · → to edit)")]
            )
            if default_text and show_default and not hide_input
            else None,
            rprompt=collision_hint,
            completer=completer,
            complete_while_typing=True,
            reserve_space_for_menu=8,
            history=DummyHistory(),
            is_password=hide_input,
            validator=validator,
            validate_while_typing=False,
            show_frame=True,
            bottom_toolbar=toolbar,
            key_bindings=bindings,
            style=Style.from_dict(
                {
                    "prompt": "ansicyan bold",
                    "frame.border": "ansibrightblack",
                    "completion-menu.completion": "bg:ansiblack ansiwhite",
                    "completion-menu.completion.current": "bg:ansicyan ansiblack bold",
                    "bottom-toolbar": "noreverse",
                    "hint": "ansibrightblack",
                    "help": "ansicyan",
                    "warning": "ansiyellow",
                }
            )
            if "NO_COLOR" not in os.environ
            else Style.from_dict({}),
            color_depth=ColorDepth.DEPTH_1_BIT if "NO_COLOR" in os.environ else None,
            **({"pre_run": start_menu} if open_menu and not hide_input else {}),
        )
        return effective(value)
    except (EOFError, KeyboardInterrupt):
        raise click.Abort() from None


def confirm(text, *, default):
    """Keep approval explicit; completion never submits an answer."""
    if not sys.stdin.isatty():
        return click.confirm(text, default=default)
    value = prompt(
        text + (" [Y/n]" if default else " [y/N]"),
        default="",
        choices=("yes", "no", "y", "n", ""),
    )
    return default if not value else value in {"yes", "y"}


def edit_model_details(model):
    """Edit a detached copy of published settings; no request or file writes."""
    from copy import deepcopy

    edited = deepcopy(model)
    click.echo(
        "Edit the settings below. Enter keeps a suggestion; - leaves it unknown. Ctrl-C cancels setup."
    )

    def count(label, current):
        while True:
            value = prompt(label, default=str(current) if current else "-").strip()
            if value == "-":
                return None
            if value.isascii() and value.isdecimal() and int(value) > 0:
                return int(value)
            click.echo("Enter a positive whole number, or - for unknown.", err=True)

    edited["context_length"] = count("Context window (tokens)", model.get("context_length"))
    edited["top_provider"] = dict(model.get("top_provider") or {})
    edited["top_provider"]["max_completion_tokens"] = count(
        "Reported reply ceiling (tokens; metadata, not a per-reply budget)",
        edited["top_provider"].get("max_completion_tokens"),
    )
    reasoning = edited["reasoning"] = dict(model.get("reasoning") or {})
    while True:
        value = prompt(
            "Reasoning levels (comma-separated)",
            default=", ".join(reasoning.get("supported_efforts") or []) or "-",
        )
        levels = [] if value.strip() == "-" else [v.strip() for v in value.split(",")]
        if all(levels) and len(set(levels)) == len(levels):
            break
        click.echo("Enter distinct level names separated by commas, or - for unknown.", err=True)
    reasoning["supported_efforts"] = levels
    reasoning["default_effort"] = None
    if levels:
        previous = (model.get("reasoning") or {}).get("default_effort")
        value = prompt(
            "Default reasoning level",
            choices=(*levels, "-"),
            default=previous if previous in levels else "-",
        )
        reasoning["default_effort"] = None if value == "-" else value
    return edited


def choose_reply_limit(suggested, ceiling=None, *, output_ceiling=None, source="connect_default"):
    """Choose a real request budget, independent of the capability ceiling.

    ``ceiling`` is the strict upper bound every choice must respect (the
    tighter of context window and known max output, when both are known).
    ``output_ceiling`` is specifically the model's own declared max output —
    never inferred from context window — offered as an explicit "Model
    maximum" choice so picking exactly that value never requires --custom.
    """
    click.echo("\n  Room for each reply, including thinking and the final answer.")
    click.echo("  Short replies use fewer tokens; this limit does not make replies longer.\n")
    origin = {
        "connect_default": "NOOA default",
        "catalogue_recommendation": "catalogue recommendation",
    }.get(source, "current setting")
    smaller = {name: cap for name, cap in (("smaller", 8192), ("short", 2048)) if cap < suggested}
    larger = {
        name: cap
        for name, cap in (("high", 65536), ("extended", 131072))
        if cap > suggested and (ceiling is None or cap <= ceiling)
    }
    maximum = (
        {"max": output_ceiling}
        if output_ceiling is not None
        and output_ceiling > suggested
        and output_ceiling not in larger.values()
        and (ceiling is None or output_ceiling <= ceiling)
        else {}
    )
    selected = prompt(
        "Reply budget",
        choices=("recommended", *larger, *maximum, *smaller, "custom"),
        default="recommended",
        open_menu=True,
        labels={
            "recommended": f"Recommended — {suggested:,} tokens ({origin})",
            **{
                name: f"{'High' if name == 'high' else 'Extended'} reasoning budget — {cap:,} tokens"
                for name, cap in larger.items()
            },
            **{name: f"Model maximum — {cap:,} tokens" for name, cap in maximum.items()},
            **{name: f"{cap:,} tokens" for name, cap in smaller.items()},
            "custom": "Custom…",
        },
    )
    if selected != "custom":
        return {**smaller, **larger, **maximum}.get(selected, suggested)
    if ceiling is not None:
        click.echo(f"  Known upper limit: {ceiling:,} tokens.")
    while True:
        value = prompt("Maximum tokens per reply", default=str(suggested))
        if value.isascii() and value.isdecimal() and int(value) > 0:
            cap = int(value)
            if ceiling is None or cap <= ceiling:
                return cap
        click.echo(
            f"Enter a positive whole number{f' at most {ceiling:,}' if ceiling else ''}.", err=True
        )
