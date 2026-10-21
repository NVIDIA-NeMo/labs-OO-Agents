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

from . import _connect_view as view


def choose_structured_output_mode(current=None):
    """Explicit connection profile, never provider catalogue metadata."""
    view.note(
        "Native Anthropic structured output is opt-in; gateway acceptance must be tested. None/auto use the existing carrier."
    )
    choice = labelled_choice(
        "Structured output profile",
        (
            ("none", "None (legacy carrier)"),
            ("auto", "Auto (legacy carrier)"),
            ("anthropic", "Anthropic native (output_config.format)"),
        ),
        default=current or "none",
    )
    return None if choice == "none" else choice


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
    searchable=False,
):
    """Use labelled menus for finite TTY choices, completion for searchable/editable input.

    Line input retains its original defaults, token answers and validation.
    """
    choices = tuple(choices)
    if choices and not searchable and not hide_input and sys.stdin.isatty():
        return labelled_choice(
            text,
            tuple((value, (labels or {}).get(value, choice_label(value))) for value in choices),
            default=default,
        )
    if not sys.stdin.isatty():
        if labels:
            for value, label in labels.items():
                view.fields([(value, label)])
        while True:
            label = text
            if show_default and default is not None and not hide_input:
                label += f" [{default}]"
            value = click.prompt(
                view.prompt_text(label), default=default, hide_input=hide_input, show_default=False
            )
            if not choices or value in choices:
                return value
            # Click.Choice includes every model ID in its prompt and error.
            view.note(
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
                        " Skip tests: restart with --no-probe\n"
                        " Manual setup: docs/model-configuration.md\n"
                        " Agent workflow: nooa-connect\n"
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
        return click.confirm(
            view.prompt_text(text + (" [Y/n]" if default else " [y/N]")),
            default=default,
            show_default=False,
        )
    value = prompt(
        text + (" [Y/n]" if default else " [y/N]"),
        default="",
        choices=("yes", "no", "y", "n", ""),
        searchable=True,
    )
    return default if not value else value in {"yes", "y"}


def edit_model_details(model):
    """Edit a detached copy of published settings; no request or file writes."""
    from copy import deepcopy

    edited = deepcopy(model)
    view.note(
        "Edit the settings below. Enter keeps a suggestion; - leaves it unknown. Ctrl-C cancels setup."
    )

    def count(label, current):
        while True:
            value = prompt(label, default=str(current) if current else "-").strip()
            if value == "-":
                return None
            if value.isascii() and value.isdecimal() and int(value) > 0:
                return int(value)
            view.note("Enter a positive whole number, or - for unknown.", err=True)

    edited["context_length"] = count("Context window (tokens)", model.get("context_length"))
    edited["top_provider"] = dict(model.get("top_provider") or {})
    edited["top_provider"]["max_completion_tokens"] = count(
        "Model reply maximum (tokens)",
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
        view.note("Enter distinct level names separated by commas, or - for unknown.", err=True)
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
    view.note("\n  Room for each reply, including thinking and the final answer.")
    view.note("  Choose a smaller limit for short replies, or leave more room for thinking.\n")
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
        "Reply limit",
        choices=("recommended", *larger, *maximum, *smaller, "custom"),
        default="recommended",
        open_menu=True,
        labels={
            "recommended": f"Recommended — {suggested:,} tokens ({origin})",
            **{
                name: f"{'High' if name == 'high' else 'Extended'} reply limit — {cap:,} tokens"
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
        view.note(f"  Known upper limit: {ceiling:,} tokens.")
    while True:
        value = prompt("Maximum tokens per reply", default=str(suggested))
        if value.isascii() and value.isdecimal() and int(value) > 0:
            cap = int(value)
            if ceiling is None or cap <= ceiling:
                return cap
        view.note(
            f"Enter a positive whole number{f' at most {ceiling:,}' if ceiling else ''}.", err=True
        )


def parse_media_inputs(value):
    """Parse an explicit CLI declaration, never a catalogue capability."""
    from nooa.unifiedllm.media_inputs import validate_media_inputs

    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(validate_media_inputs(value))
    return list(
        validate_media_inputs(
            []
            if not value.strip() or value.strip() == "-"
            else [item.strip() for item in value.split(",")]
        )
    )


def choose_media_inputs(current=()):
    """User-declared formats; no network or model metadata involved."""
    if sys.stdin.isatty():
        from itertools import combinations

        formats = ("image", "audio", "video", "pdf")
        values = [subset for size in range(5) for subset in combinations(formats, size)]
        selected = labelled_choice(
            "Choose media inputs without testing",
            tuple(
                (
                    ",".join(subset) or "-",
                    ", ".join("PDF" if kind == "pdf" else kind.capitalize() for kind in subset)
                    or "Text only",
                )
                for subset in values
            ),
            default=",".join(kind for kind in formats if kind in current) or "-",
        )
        parsed = parse_media_inputs(selected)
        return list(current) if set(parsed) == set(current) else parsed
    while True:
        value = prompt(
            "Enable media inputs manually (not tested; image,audio,video,pdf; - to turn all off)",
            default=",".join(current) or "-",
        )
        try:
            return parse_media_inputs(value)
        except ValueError as exc:
            view.note(str(exc), err=True)


def choice_label(value):
    """Human labels for finite choices; custom reasoning names remain verbatim."""
    return {
        "discover": "Discover request interfaces",
        "systemone": "System One Decisions",
        "openai": "OpenAI Decisions",
        "chat": "OpenAI-compatible Chat",
        "responses": "OpenAI Responses",
        "anthropic": "Anthropic Messages",
        "-": "Unknown / not configured",
        "": "Continue without catalogue settings",
    }.get(value, value)


def labelled_choice(message, options, *, default):
    """Show every action immediately; retain numeric and token answers for pipes."""
    if not sys.stdin.isatty():
        view.section(message)
        answers = {}
        view.fields(
            [(f"{number} ({value})", label) for number, (value, label) in enumerate(options, 1)]
        )
        for number, (value, _label) in enumerate(options, 1):
            answers[str(number)] = answers[value] = value
        default_number = str(next(i for i, (value, _) in enumerate(options, 1) if value == default))
        hint = " or ".join(f"{i} ({value})" for i, (value, _) in enumerate(options, 1))
        while True:
            value = (
                click.prompt(
                    view.prompt_text(f"Choice [{default_number}]"),
                    default=default_number,
                    show_default=False,
                )
                .strip()
                .lower()
            )
            if value in answers:
                return answers[value]
            view.note(f"Enter {hint}.", err=True)

    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.output import ColorDepth
    from prompt_toolkit.shortcuts.choice_input import ChoiceInput
    from prompt_toolkit.styles import Style

    class ConnectChoiceInput(ChoiceInput):
        def _create_application(self):
            app = super()._create_application()
            if "NO_COLOR" in os.environ:
                app._color_depth = ColorDepth.DEPTH_1_BIT
            return app

    def menu_toolbar(help_open):
        text = (
            "F1 Close help\nSkip tests: restart with --no-probe\nManual setup: docs/model-configuration.md\nAgent workflow: nooa-connect\n↑↓ Choose · Enter Continue · Ctrl-C Cancel"
            if help_open
            else "↑↓ Choose · Enter Continue\nCtrl-C Cancel · F1 Help"
        )
        width = max(8, view.shutil.get_terminal_size((80, 24)).columns - 2)
        return "\n".join(
            part
            for line in text.splitlines()
            for part in view.textwrap.wrap(line, width=width, break_on_hyphens=False)
        )

    help_open = False
    bindings = KeyBindings()

    @bindings.add("f1")
    def toggle_help(event):
        nonlocal help_open
        help_open = not help_open
        chooser.bottom_toolbar = menu_toolbar(help_open)
        event.app.invalidate()

    try:
        chooser = ConnectChoiceInput(
            message=FormattedText([("class:prompt", message)]),
            options=((None, "Choose an option…"), *options) if default is None else options,
            default=default,
            show_frame=True,
            bottom_toolbar=menu_toolbar(False),
            key_bindings=bindings,
            style=Style.from_dict({})
            if "NO_COLOR" in os.environ
            else Style.from_dict(
                {
                    "prompt": "ansicyan bold",
                    "selected-option": "ansicyan bold",
                    "frame.border": "ansibrightblack",
                    "bottom-toolbar": "noreverse",
                }
            ),
        )
        while True:
            value = chooser.prompt()
            if value is not None:
                return value
    except (EOFError, KeyboardInterrupt):
        raise click.Abort() from None


def review_media_inputs(current=()):
    """Offer labelled actions, not editable internal tokens, after media checks."""
    return labelled_choice(
        "How would you like to continue?",
        (
            ("keep", "Continue with " + (", ".join(current) or "text only")),
            ("manual", "Choose media inputs manually"),
        ),
        default="keep",
    )


def choose_media_mode(*, editing=False):
    """Offer generated tests or manual declarations with a visible Enter default."""
    return labelled_choice(
        "How would you like to configure media inputs?",
        (
            ("test", "Test all media inputs (generated samples)"),
            ("manual", "Choose media inputs manually"),
        ),
        default="manual" if editing else "test",
    )


def choose_decision_images(current=()):
    """Human actions with unchanged image setup tokens and Enter defaults."""
    view.note("Image tests use a generated sample.")
    options = (("keep", "Keep current image settings"),) if current else ()
    return labelled_choice(
        "Should this decision model receive images?",
        (
            *options,
            ("text", "Leave images off"),
            ("manual", "Enable images without testing"),
            ("test", "Test images before enabling"),
        ),
        default="keep" if current else "text",
    )
