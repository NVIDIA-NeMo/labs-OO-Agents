# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reusable two-pane full-screen browser layout for terminal pickers and explorers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.layout import (
    BufferControl,
    ConditionalContainer,
    DynamicContainer,
    Float,
    FloatContainer,
    HSplit,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import FormattedTextControl, UIControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType, MouseModifier
from prompt_toolkit.widgets import Frame

from .terminal_safety import (
    project_prompt_toolkit_ansi,
    sanitize_live_text,
    sanitize_transcript_ansi,
)


def build_fullscreen_browser(
    *,
    app: Any,
    title_control: UIControl,
    help_control: UIControl,
    controls: Any,
    list_header_control: UIControl,
    list_control: UIControl,
    preview_header_control: UIControl,
    preview_control: UIControl,
    active_rail: Callable[[str], Any],
    small_control: UIControl,
    small_text: Callable[[], Any],
    list_height: Dimension | None = None,
    floats: list[Float] | None = None,
) -> DynamicContainer:
    """Build the shared Resume-style title/search/list/preview browser shell."""

    def separator() -> Window:
        return Window(char="─", height=1, style="class:fullscreen-browser.separator")

    def area(name: str, body: Any) -> VSplit:
        return VSplit(
            [
                Window(
                    FormattedTextControl(lambda: active_rail(name)),
                    width=1,
                    style="class:fullscreen-browser.active-rail",
                ),
                body,
            ],
            padding=0,
        )

    main_body = HSplit(
        [
            Window(title_control, height=1),
            separator(),
            Window(help_control, height=1),
            area(
                "list",
                HSplit(
                    [
                        controls,
                        Window(list_header_control, height=1),
                        separator(),
                        Window(
                            list_control,
                            height=list_height or Dimension(min=1, preferred=4, max=5),
                            wrap_lines=False,
                        ),
                    ],
                    padding=0,
                ),
            ),
            separator(),
            area(
                "preview",
                HSplit(
                    [
                        Window(preview_header_control, height=1),
                        Window(
                            preview_control, height=Dimension(min=2, weight=1), wrap_lines=False
                        ),
                    ],
                    padding=0,
                ),
            ),
            separator(),
        ],
        padding=0,
    )
    main = FloatContainer(content=main_body, floats=floats or [])
    small = HSplit(
        [
            Window(small_control, height=1),
            Window(FormattedTextControl(small_text), height=1),
        ]
    )

    def responsive() -> Any:
        size = app.output.get_size()
        return main if size.columns >= 48 and size.rows >= 13 else small

    return DynamicContainer(responsive)


class _BrowserPaneControl(FormattedTextControl):
    def __init__(self, browser: ExplorerBrowser, pane: str) -> None:
        self.browser = browser
        self.pane = pane
        self.viewport = (1, 1)
        super().__init__(self._text, focusable=True, show_cursor=False)

    def create_content(self, width: int, height: int):
        self.viewport = (max(1, width), max(1, height or 1))
        self._fragment_cache.clear()
        return super().create_content(width, height)

    def _text(self):
        return (
            self.browser.list_text(*self.viewport)
            if self.pane == "list"
            else self.browser.preview_text(*self.viewport)
        )

    def mouse_handler(self, mouse_event: MouseEvent):
        if (
            MouseModifier.ALT in mouse_event.modifiers
            or MouseModifier.SHIFT in mouse_event.modifiers
        ):
            return NotImplemented
        if mouse_event.event_type is MouseEventType.SCROLL_UP:
            self.browser.mouse_scroll(self.pane, -3)
            return None
        if mouse_event.event_type is MouseEventType.SCROLL_DOWN:
            self.browser.mouse_scroll(self.pane, 3)
            return None
        if (
            mouse_event.event_type is MouseEventType.MOUSE_DOWN
            and mouse_event.button is MouseButton.LEFT
        ):
            self.browser.activate_control(self.pane)
            if self.pane == "list":
                self.browser.select_visible(mouse_event.position.y)
            return None
        return NotImplemented


class _BrowserSearchControl(BufferControl):
    def __init__(self, browser: ExplorerBrowser, buffer: Buffer) -> None:
        self.browser = browser
        super().__init__(buffer)

    def mouse_handler(self, mouse_event: MouseEvent):
        if mouse_event.event_type is MouseEventType.MOUSE_DOWN:
            self.browser.activate_control("list")
        return super().mouse_handler(mouse_event)


class _BrowserOptionControl(FormattedTextControl):
    def __init__(self, browser: ExplorerBrowser, index: int) -> None:
        self.browser = browser
        self.index = index
        super().__init__(self._text, focusable=False, show_cursor=False)

    def _text(self):
        option = self.browser.view.options[self.index]
        focused = self.browser.option_cursor == self.index
        style = "class:fullscreen-browser.control" + (
            " class:fullscreen-browser.control-focused" if focused else ""
        )
        fragments = []
        if focused and getattr(option, "dropdown", False):
            fragments.append(("[SetMenuPosition]", ""))
        fragments.append((style, f"[{option.label}: {option.display_value}]"))
        return fragments

    def mouse_handler(self, mouse_event: MouseEvent):
        if (
            mouse_event.event_type is MouseEventType.MOUSE_DOWN
            and mouse_event.button is MouseButton.LEFT
        ):
            self.browser.option_cursor = self.index
            option = self.browser.view.options[self.index]
            if not getattr(option, "dropdown", False):
                option.activate()
                self.browser.list_offset = 0
                self.browser.close_options()
            self.browser.invalidate()
            return None
        return NotImplemented


class _BrowserDropdownControl(FormattedTextControl):
    """Expanded radio choices for the active dropdown option."""

    def __init__(self, browser: ExplorerBrowser, option_index: int) -> None:
        self.browser = browser
        self.option_index = option_index
        super().__init__(self._text, focusable=False, show_cursor=False)

    @property
    def option(self) -> Any | None:
        if self.browser.option_cursor != self.option_index:
            return None
        option = self.browser.view.options[self.option_index]
        return option if getattr(option, "dropdown", False) else None

    def _text(self):
        option = self.option
        if option is None:
            return []
        fragments: list[tuple[str, str]] = []
        for index, (value, label) in enumerate(option.choices):
            multi_select = bool(getattr(option, "multi_select", False))
            selected = option.is_checked(value) if multi_select else value == option.value
            focused = index == option.choice_cursor
            style = "class:fullscreen-browser.dropdown"
            if focused:
                style += " class:fullscreen-browser.control-focused"
            if multi_select:
                indicator = "☑" if selected else "☐"
            else:
                indicator = "◉" if selected else "○"
            fragments.append((style, f"  {indicator} {label}"))
            if index + 1 < len(option.choices):
                fragments.append(("", "\n"))
        return fragments

    def mouse_handler(self, mouse_event: MouseEvent):
        option = self.option
        if (
            option is not None
            and mouse_event.event_type is MouseEventType.MOUSE_DOWN
            and mouse_event.button is MouseButton.LEFT
            and 0 <= mouse_event.position.y < len(option.choices)
        ):
            option.choice_cursor = mouse_event.position.y
            option.activate()
            self.browser.list_offset = 0
            if not getattr(option, "multi_select", False):
                self.browser.close_options()
            else:
                self.browser.invalidate()
            return None
        return NotImplemented


class ExplorerBrowser:
    """Prompt-toolkit adapter that hosts a configured explorer in the shared shell."""

    pending_input: str | None = None

    @property
    def mouse_support(self) -> bool:
        return bool(getattr(self.view, "mouse_support", True))

    def __init__(self, view: Any, app: Any) -> None:
        self.view = view
        self.app = app
        self.title = view.title
        self.active_control = "list"
        self.option_cursor: int | None = None
        self.list_offset = 0
        self._list_offset_detached = False
        self.buffer = Buffer(multiline=False)
        self.buffer.text = str(getattr(view.model, "query", ""))
        self.buffer.on_text_changed += lambda _: self._query_changed()
        self.query_control = _BrowserSearchControl(self, self.buffer)
        self.query_window = Window(
            self.query_control,
            width=Dimension(min=4, weight=1),
            height=1,
            style=lambda: (
                "class:fullscreen-browser.control-focused"
                if self.active_control == "list" and self.option_cursor is None
                else ""
            ),
        )
        self.list_control = _BrowserPaneControl(self, "list")
        self.preview_control = _BrowserPaneControl(self, "preview")
        self.option_controls = [
            _BrowserOptionControl(self, index) for index in range(len(view.options))
        ]
        option_windows: list[Any] = []
        self.option_windows: list[Window] = []
        for index, control in enumerate(self.option_controls):
            if index:
                option_windows.append(Window(FormattedTextControl(" "), width=1, height=1))
            option_window = Window(control, width=Dimension(min=12, preferred=22), height=1)
            self.option_windows.append(option_window)
            option_windows.append(option_window)
        search = VSplit(
            [
                Window(FormattedTextControl(self._search_label), width=9, height=1),
                self.query_window,
                Window(FormattedTextControl(self._search_close), width=1, height=1),
            ],
            padding=0,
        )
        controls = VSplit(
            [search, Window(FormattedTextControl(" "), width=1, height=1), VSplit(option_windows)],
            padding=0,
        )
        self.dropdown_controls: list[_BrowserDropdownControl] = []
        self.dropdown_floats: list[Float] = []
        dropdown_floats = self.dropdown_floats
        for index, option in enumerate(view.options):
            if not getattr(option, "dropdown", False):
                continue
            control = _BrowserDropdownControl(self, index)
            self.dropdown_controls.append(control)
            menu_width = max(
                18,
                min(44, max(len(label) for _value, label in option.choices) + 6),
            )
            dropdown_floats.append(
                Float(
                    content=ConditionalContainer(
                        Frame(
                            Window(
                                control,
                                height=lambda option=option: len(option.choices),
                                style="class:fullscreen-browser.dropdown",
                            ),
                            style="class:fullscreen-browser.dropdown-frame",
                        ),
                        filter=Condition(lambda index=index: self.option_cursor == index),
                    ),
                    width=menu_width,
                    height=lambda option=option: len(option.choices) + 2,
                    xcursor=True,
                    ycursor=True,
                    attach_to_window=self.option_windows[index],
                    z_index=10,
                )
            )
        self.title_control = FormattedTextControl(self._title)
        self.help_control = FormattedTextControl(
            self._help_text, style="class:fullscreen-browser.footer"
        )
        self.list_header_control = FormattedTextControl(self._list_header)
        self.preview_header_control = FormattedTextControl(self._preview_header)
        self.small_control = FormattedTextControl(
            [("class:fullscreen-browser.too-small", "Terminal too small")], focusable=True
        )
        self.explorer_list_height = Dimension(min=1, preferred=5, weight=1)
        self.container = build_fullscreen_browser(
            app=app,
            title_control=self.title_control,
            help_control=self.help_control,
            controls=controls,
            list_header_control=self.list_header_control,
            list_control=self.list_control,
            preview_header_control=self.preview_header_control,
            preview_control=self.preview_control,
            active_rail=self._active_rail,
            small_control=self.small_control,
            small_text=self._small_text,
            # Explorer lists should consume the full list pane. The Resume
            # Picker keeps the shell's compact five-row default.
            list_height=self.explorer_list_height,
            floats=dropdown_floats,
        )

    @property
    def active_dropdown(self) -> Any | None:
        if self.option_cursor is None:
            return None
        option = self.view.options[self.option_cursor]
        return option if getattr(option, "dropdown", False) else None

    @property
    def model(self) -> Any:
        return self.view.model

    def _query_changed(self) -> None:
        self.model.edit_query(self.buffer.text)
        self.model.search_active = bool(self.buffer.text.strip())
        self.list_offset = 0
        self._list_offset_detached = False
        self.invalidate()

    def _title(self):
        count = len(self.model.matches)
        noun = getattr(self.view, "item_name", self.title.lower())
        return [
            (
                "class:fullscreen-browser.title",
                f"{self.title} · {count} {noun}{'' if count == 1 else 's'}",
            )
        ]

    def _help_text(self):
        if self.option_cursor is not None:
            return "Options · ←→ select · ↑↓/Space change · Enter/Esc done"
        actions = " · ".join(self.view.config.actions.values())
        suffix = f" · {actions}" if actions else ""
        return f"Type search · Ctrl-O options · Tab panes · ↑↓ move · Esc close{suffix}"

    def _search_label(self):
        style = "class:fullscreen-browser.search-label"
        if self.active_control == "list" and self.option_cursor is None:
            style += " class:fullscreen-browser.control-focused"
        return [(style, "[Search: ")]

    def _search_close(self):
        style = "class:fullscreen-browser.search-label"
        if self.active_control == "list" and self.option_cursor is None:
            style += " class:fullscreen-browser.control-focused"
        return [(style, "]")]

    def _active_rail(self, area: str):
        active = self.active_control == area
        style = (
            "class:fullscreen-browser.active-rail-active"
            if active
            else "class:fullscreen-browser.active-rail"
        )
        glyph = "▌" if active else "│"
        return [(style, "\n".join([glyph] * max(1, self.app.output.get_size().rows)))]

    def _list_header(self):
        count = len(self.model.matches)
        position = self.model.cursor + 1 if count else 0
        heading = getattr(self.view, "list_heading", "Items")
        return [("class:fullscreen-browser.heading", f"{heading} · {position}/{count}")]

    def _preview_header(self):
        row = self.model.current
        title = getattr(row, "title", None) or getattr(row, "event_type", None) or "No selection"
        return [("class:fullscreen-browser.heading", f"Preview · {title}")]

    def _small_text(self):
        size = self.app.output.get_size()
        return [
            ("class:fullscreen-browser.muted", f"Need 48 x 13; now {size.columns} x {size.rows}")
        ]

    def focus_initial(self) -> None:
        size = self.app.output.get_size()
        self.app.layout.focus(
            self.query_control if size.columns >= 48 and size.rows >= 13 else self.small_control
        )

    def invalidate(self) -> None:
        for control in [
            self.list_control,
            self.preview_control,
            *self.dropdown_controls,
            *self.option_controls,
            self.title_control,
            self.help_control,
            self.list_header_control,
            self.preview_header_control,
        ]:
            control._fragment_cache.clear()
        self.app.invalidate()

    def activate_control(self, name: str) -> None:
        self.active_control = name
        self.model.focus = (
            "list" if name == "list" else getattr(self.view, "detail_focus", "detail")
        )
        self.app.layout.focus(self.query_control if name == "list" else self.preview_control)
        self.invalidate()

    def focus_next(self, delta: int = 1) -> None:
        self.close_options()
        self.activate_control("preview" if self.active_control == "list" else "list")

    def toggle_options(self) -> None:
        if not self.view.options:
            return
        self.active_control = "list"
        self.option_cursor = 0 if self.option_cursor is None else None
        self.app.layout.focus(
            self.list_control if self.option_cursor is not None else self.query_control
        )
        self.invalidate()

    def close_options(self) -> bool:
        if self.option_cursor is None:
            return False
        self.option_cursor = None
        self.app.layout.focus(self.query_control)
        self.invalidate()
        return True

    def move_option(self, delta: int) -> None:
        if self.option_cursor is not None and self.view.options:
            self.option_cursor = (self.option_cursor + delta) % len(self.view.options)
            self.invalidate()

    def change_option(self, delta: int = 1) -> None:
        if self.option_cursor is not None:
            option = self.view.options[self.option_cursor]
            option.move(delta)
            if delta == 0:
                option.activate()
            self.list_offset = 0
            self.invalidate()

    def _visible(self, height: int) -> list[tuple[int, int]]:
        count = len(self.model.matches)
        self.list_offset = min(max(self.list_offset, 0), max(0, count - max(1, height)))
        if not self._list_offset_detached:
            if self.model.cursor < self.list_offset:
                self.list_offset = self.model.cursor
            elif self.model.cursor >= self.list_offset + height:
                self.list_offset = self.model.cursor - height + 1
        return [
            (i, self.model.matches[i])
            for i in range(self.list_offset, min(count, self.list_offset + height))
        ]

    def list_text(self, width: int, height: int):
        output: list[tuple[str, str]] = []
        for visible_index, row_index in self._visible(height):
            if output:
                output.append(("", "\n"))
            selected = visible_index == self.model.cursor
            base = "class:fullscreen-browser.row" + (
                " class:fullscreen-browser.selected" if selected else ""
            )
            marker = "❯ " if selected else "  "
            text = sanitize_live_text(
                marker + self.view.format_row(self.model.rows[row_index], max(1, width - 2))
            )
            text = text.replace("\n", " ").replace("\r", "")[:width]
            query = self.buffer.text.casefold().strip()
            if not query:
                output.append((base, text))
                continue
            folded = text.casefold()
            cursor = 0
            while True:
                found = folded.find(query, cursor)
                if found < 0:
                    output.append((base, text[cursor:]))
                    break
                output.append((base, text[cursor:found]))
                output.append(
                    (base + " class:fullscreen-browser.match", text[found : found + len(query)])
                )
                cursor = found + len(query)
        return output or [("class:fullscreen-browser.empty", "No matching items")]

    def preview_text(self, width: int, height: int):
        row = self.model.current
        if row is None:
            return [("class:fullscreen-browser.empty", "No item selected")]
        lines = self.view.detail_lines(row, max(1, width))
        self.model._last_detail_line_count = len(lines)
        self.model._last_detail_visible_lines = max(1, height)
        self.model.clamp_detail_offset(height)
        selected = lines[self.model.detail_offset : self.model.detail_offset + height]
        return ANSI(project_prompt_toolkit_ansi(sanitize_transcript_ansi("\n".join(selected))))

    def navigate_vertical(self, delta: int) -> None:
        if self.option_cursor is not None:
            self.change_option(delta)
        elif self.active_control == "list":
            self._list_offset_detached = False
            self.model.move_or_scroll(delta)
        else:
            self.model.move_or_scroll(delta)
        self.invalidate()

    def page(self, delta: int) -> None:
        amount = (
            self.list_control.viewport[1]
            if self.active_control == "list"
            else self.preview_control.viewport[1]
        )
        if self.active_control == "list":
            self._list_offset_detached = False
            self.model.move(delta * max(1, amount))
        else:
            self.model.scroll_detail(delta * max(1, amount))
        self.invalidate()

    def select_visible(self, y: int) -> None:
        index = self.list_offset + y
        if 0 <= index < len(self.model.matches):
            self.model.cursor = index
            self.model.detail_offset = 0
            self._list_offset_detached = False
            self.invalidate()

    def mouse_scroll(self, pane: str, delta: int) -> None:
        if pane == "list":
            maximum = max(0, len(self.model.matches) - self.list_control.viewport[1])
            self.list_offset = min(maximum, max(0, self.list_offset + delta))
            self._list_offset_detached = True
        else:
            self.model.scroll_detail(delta)
        self.invalidate()

    def handle_key(self, action: str, value: str = ""):
        if action == "escape":
            if self.close_options():
                return "handled"
            return "close"
        if action == "options":
            self.toggle_options()
        elif self.option_cursor is not None and action not in {
            "left",
            "right",
            "up",
            "down",
            "space",
            "enter",
            "tab",
            "s-tab",
        }:
            return "handled"
        elif action in {"left", "right"}:
            if self.option_cursor is not None:
                self.move_option(-1 if action == "left" else 1)
            elif self.active_control == "list":
                if action == "left":
                    self.buffer.cursor_left(count=1)
                else:
                    self.buffer.cursor_right(count=1)
        elif action in {"down", "up"}:
            self.navigate_vertical(1 if action == "down" else -1)
        elif action == "space":
            if self.option_cursor is not None:
                self.view.options[self.option_cursor].activate()
                self.invalidate()
            elif self.active_control == "list":
                self.buffer.insert_text(" ")
        elif action == "enter":
            if self.active_dropdown is not None:
                if not getattr(self.active_dropdown, "multi_select", False):
                    self.active_dropdown.activate()
                    self.list_offset = 0
                self.close_options()
            elif not self.close_options():
                result = self.view.handle_action("enter", self.model.current)
                if result == "close":
                    self.pending_input = getattr(self.view, "pending_input", None)
                    return "close"
        elif action in {"tab", "s-tab"}:
            self.focus_next(-1 if action == "s-tab" else 1)
        elif action == "page_down":
            self.page(1)
        elif action == "page_up":
            self.page(-1)
        elif action == "home":
            self.model.jump_home()
            self.invalidate()
        elif action == "end":
            self.model.jump_end()
            self.invalidate()
        elif action in {"scroll_down", "scroll_up"}:
            delta = 3 if action == "scroll_down" else -3
            self.mouse_scroll(self.active_control, delta)
        elif action == "backspace":
            if self.active_control == "list":
                self.buffer.delete_before_cursor()
        elif action in {"quit", "resume", "slash", "j", "k"}:
            text = {"quit": "q", "resume": "r", "slash": "/", "j": "j", "k": "k"}[action]
            if self.active_control == "list":
                self.buffer.insert_text(text)
            elif action == "quit":
                return "close"
            else:
                return self.view.handle_key("text", text)
        elif action == "text" and value and value.isprintable():
            if self.active_control == "preview":
                return self.view.handle_key("text", value)
            self.buffer.insert_text(value)
        else:
            return self.view.handle_key(action, value)
        return "handled"

    def on_open(self) -> None:
        self.view.on_open()
        self.focus_initial()

    def on_close(self) -> None:
        self.view.on_close()
