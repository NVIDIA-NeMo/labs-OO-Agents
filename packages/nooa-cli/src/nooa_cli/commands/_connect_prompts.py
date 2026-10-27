# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminal prompts for the connect wizard.

The shared library ``nooa.unifiedllm.connect`` never reads the terminal; all
questions to the user live here. In an interactive terminal the prompts use
prompt_toolkit (menus, completion, checkboxes). When standard input is not a
terminal (a pipe or script), they fall back to Click's plain line input and
accept the same typed answers as before, so scripted runs keep working.

Completion offers only names (such as environment variable names), never
environment variable values. Prompts keep no history between questions, and
prompts for secrets have neither history nor completion.
"""

import os
import sys

import click

from . import _connect_view as view


def choose_structured_output_mode(current=None):
    """Ask how NOOA should request structured (typed JSON) answers from this endpoint.

    The answer is always the user's choice; it is never taken from catalogue
    information. Returns None for the standard request format, else
    ``"auto"`` or ``"anthropic"`` (Anthropic's native JSON output format).
    """
    view.note(
        "Choose how NOOA requests typed JSON answers. Native JSON uses Anthropic’s format; unknown servers must be tested. Standard uses the usual request format."
    )
    choice = labelled_choice(
        "Answer format",
        (
            ("none", "Standard JSON"),
            ("auto", "Auto"),
            ("anthropic", "Native JSON (Anthropic format)"),
        ),
        default=current or "none",
    )
    return None if choice == "none" else choice


def environment_names(defaults=()):
    """Return sorted environment variable names to offer, plus ``defaults`` even if unset."""
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
    """Ask one question and return the answer as a string.

    In a terminal, a fixed list of ``choices`` is shown as a menu (unless
    ``searchable`` or ``hide_input`` is set); other questions use an editable
    text field with completion from ``choices`` or ``suggestions``. Enter
    accepts ``default``; Right Arrow copies the default into the field for
    editing; F1 shows help. ``existing`` lists names that are already taken,
    for which a warning is shown. ``hide_input`` masks the answer (for keys).

    Without a terminal, Click's line input is used with the same defaults,
    accepted answers and validation as before. Ctrl-C or end of input raises
    ``click.Abort``.
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
            # Checked here instead of with click.Choice, which would print every
            # model ID in its prompt and error message.
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
    """Ask a yes/no question and return the answer as a bool.

    Empty input returns ``default``. Completion only fills in text; the user
    must still press Enter, so an answer is never submitted automatically.
    """
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


def manual_model_limits(context=None, input_limit=None, *, edit=False):
    """Ask for the context window and input token limit; return ``(context, input_limit)``.

    Each value must be a whole number of at least 2 tokens. A known value
    (greater than 1) is kept without asking unless ``edit`` is set. Both values
    are collected before either is returned, so cancelling half-way applies
    nothing.
    """
    view.note("Enter your server limits. Ctrl-C cancels setup.")

    def count(label, current):
        if not edit and isinstance(current, int) and not isinstance(current, bool) and current > 1:
            return current
        while True:
            value = prompt(
                label, default=str(current) if current else "", show_default=bool(current)
            ).strip()
            if value.isascii() and value.isdecimal() and int(value) > 0:
                number = int(value)
                if number > 1:
                    return number
                view.note(
                    "The context bound must leave room for input and a positive reply: enter at least 2 tokens.",
                    err=True,
                )
                continue
            view.note("Enter a positive whole number.", err=True)

    return (
        count("Context window (tokens)", context),
        count("Input token limit (tokens)", input_limit),
    )


def edit_model_details(model, *, reasoning=True):
    """Let the user correct the published details of a model; return an edited copy.

    ``model`` is a catalogue-style dict (``context_length``,
    ``top_provider.max_completion_tokens``, ``reasoning``). It is copied, so the
    original is not changed. ``-`` marks a value as unknown. Reasoning levels
    are only asked for when ``reasoning`` is true. No requests are sent and no
    files are written.
    """
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
    if not reasoning:
        return edited
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
    """Ask for the reply cap to save (maximum reply tokens per request) and return it.

    The reply cap is what each request asks for; it may be lower than what
    the model allows. ``suggested`` is the recommended value and ``source``
    says where it came from (shown in its label).

    ``ceiling`` is the upper bound every choice must stay within: the smaller
    of the context window and the model's maximum reply length, when known.
    ``output_ceiling`` is the model's own published maximum reply length (never
    derived from the context window). It is offered as a "Model maximum" menu
    item so the user can pick exactly that value without typing a custom one.
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
    """Parse a ``--media-inputs`` value into a list of media kinds, or None if not given.

    Accepts a comma-separated string or a list. An empty string or ``-``
    means no media. Raises ``ValueError`` for unknown kinds. This only parses
    what the user typed; it does not consult catalogue information.
    """
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


def choose_media_inputs(current=(), *, available=("image", "audio", "video", "pdf")):
    """Let the user pick media inputs to enable without testing; return the chosen kinds.

    ``current`` is pre-selected and ``available`` limits the choice. A terminal
    shows checkboxes; without a terminal a comma-separated answer is read
    (``-`` turns all media off). Ctrl-C raises ``click.Abort``.
    """
    if not sys.stdin.isatty():
        while True:
            value = prompt(
                "Enable media inputs manually (not tested; "
                + ",".join(available)
                + "; - to turn all off)",
                default=",".join(current) or "-",
            )
            try:
                parsed = parse_media_inputs(value)
                if any(kind not in available for kind in parsed):
                    raise ValueError("Choose only: " + ", ".join(available))
                return parsed
            except ValueError as exc:
                view.note(str(exc), err=True)

    from prompt_toolkit.application import Application
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout
    from prompt_toolkit.output import ColorDepth
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import CheckboxList, Frame, Label

    checks = CheckboxList(
        values=[(kind, view.media_label(kind)) for kind in available],
        default_values=[kind for kind in current if kind in available],
    )
    help_open = False

    def toolbar():
        text = (
            "F1 Close help\nNone checked = text only\nSkip tests: restart with --no-probe\n"
            "Manual setup: docs/model-configuration.md\nAgent workflow: nooa-connect\n"
            if help_open
            else "None checked = text only\n"
        ) + "↑↓ Move · Space Toggle · Enter Confirm\nCtrl-C Cancel · F1 Help"
        width = max(8, view.shutil.get_terminal_size((80, 24)).columns - 2)
        return "\n".join(
            part
            for line in text.splitlines()
            for part in view.textwrap.wrap(line, width=width, break_on_hyphens=False)
        )

    bindings = KeyBindings()

    @bindings.add("enter", eager=True)
    def accept(event):
        selected = checks.current_values
        # Keep previously enabled kinds in their order, then add newly checked ones.
        event.app.exit(
            result=[kind for kind in current if kind in selected]
            + [kind for kind in available if kind in selected and kind not in current]
        )

    @bindings.add("c-c")
    def cancel(event):
        event.app.exit(exception=KeyboardInterrupt())

    @bindings.add("f1")
    def toggle_help(event):
        nonlocal help_open
        help_open = not help_open
        event.app.invalidate()

    app = Application(
        layout=Layout(
            HSplit(
                [
                    Frame(checks, title="Choose media inputs"),
                    Label(toolbar),
                ]
            ),
            focused_element=checks,
        ),
        key_bindings=bindings,
        style=Style.from_dict({})
        if "NO_COLOR" in os.environ
        else Style.from_dict(
            {
                "checkbox-selected": "ansicyan bold",
                "frame.border": "ansibrightblack",
            }
        ),
        color_depth=ColorDepth.DEPTH_1_BIT if "NO_COLOR" in os.environ else None,
        full_screen=False,
    )
    try:
        return app.run()
    except (EOFError, KeyboardInterrupt):
        raise click.Abort() from None


def choice_label(value):
    """Return the menu label for a choice value; unknown values (e.g. reasoning levels) as is."""
    return {
        "discover": "Find what works (sends a few small test requests)",
        "systemone": "System One Decisions",
        "openai": "OpenAI Decisions",
        "chat": "OpenAI-compatible Chat",
        "responses": "OpenAI Responses",
        "anthropic": "Anthropic Messages",
        "-": "Unknown / not configured",
        "": "Continue without catalogue settings",
    }.get(value, value)


def key_source(default, names):
    """Ask where the API key comes from and return the answer.

    Returns an environment variable name, ``"new"`` (paste a key now) or
    ``"-"`` (no key). In a terminal, a menu offers the variables in ``names``
    and ``default`` that are set, plus the other options. Without a terminal
    the answer is typed in, as before.
    """
    if not sys.stdin.isatty():
        return prompt(
            "Key environment variable (new to enter a key; - for no authentication)",
            default=default,
            suggestions=environment_names([*names, "-", "new"]),
        )
    found = [name for name in dict.fromkeys([*names, default]) if os.environ.get(name)]
    options = [(name, f"Use the key in {name}") for name in found]
    options += [
        ("new", "Paste a key now"),
        ("other", "Use another environment variable"),
        ("-", "No key needed"),
    ]
    choice = labelled_choice("API key", tuple(options), default=found[0] if found else "new")
    if choice == "other":
        return prompt("Environment variable name", default=default, suggestions=environment_names())
    return choice


def labelled_choice(message, options, *, default):
    """Show a menu of ``(value, label)`` options and return the chosen value.

    In a terminal all options are visible at once and Enter picks
    ``default``; if ``default`` is None the user must choose one. Without a
    terminal the options are printed with numbers, and the answer may be the
    number or the value. Ctrl-C raises ``click.Abort``.
    """
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
        """prompt_toolkit choice menu that also honours ``NO_COLOR``."""

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
    """After media tests, ask whether to keep the result or choose by hand.

    Returns ``"keep"`` or ``"manual"``.
    """
    return labelled_choice(
        "How would you like to continue?",
        (
            ("keep", "Continue with " + (", ".join(current) or "text only")),
            ("manual", "Choose media inputs manually"),
        ),
        default="keep",
    )


def choose_media_mode(*, editing=False):
    """Ask whether to test all media inputs or choose them by hand.

    Returns ``"test"`` or ``"manual"``. The default is ``"manual"`` when editing
    an existing entry and ``"test"`` otherwise.
    """
    return labelled_choice(
        "How would you like to configure media inputs?",
        (
            ("test", "Test all media inputs (generated samples)"),
            ("manual", "Choose media inputs manually"),
        ),
        default="manual" if editing else "test",
    )


def choose_decision_images(current=()):
    """Ask whether a decision model should receive images.

    Returns ``"keep"`` (only offered when images are already set up),
    ``"text"`` (no images), ``"manual"`` (enable without testing) or ``"test"``.
    """
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
