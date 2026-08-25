# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rich console wrapper with spinners and styled output.

Uses Catppuccin Mocha theme from https://catppuccin.com/palette/
"""

from rich.console import Console
from rich.live import Live
from rich.rule import Rule
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from .copyable_markdown import TerminalMarkdown
from .theme import CATPPUCCIN_THEME, COLORS


class TUIConsole:
    """Rich console wrapper with Catppuccin styling for the NOOA TUI."""

    def __init__(self) -> None:
        self.console = Console(theme=CATPPUCCIN_THEME)
        self._live_spinner: Live | None = None
        self._spinner: Spinner | None = None

    def replace_console(self, console: Console) -> None:
        """Swap the underlying Rich Console (e.g. to redirect output).

        ``Session`` uses this to route slash-command rendering through
        the ``TUIApplication`` block queue instead of the real stdout —
        without this seam, callers would have to patch ``self.console``
        directly, a private on a foreign object.
        """
        self.console = console

    def start_spinner(self, message: str = "thinking...") -> None:
        """Start the thinking spinner with an empty input prompt.

        The Live display renders as a Group:
          <spinner>
          ❯ █
        """
        if self._live_spinner is not None:
            return

        from rich.console import Group

        self._spinner = Spinner(
            "dots",
            text=Text(message, style=f"{COLORS['subtext1']}"),
            style=COLORS["mauve"],
        )
        prompt = Text("\u276f ", style=COLORS["subtext0"])
        prompt.append(" ", style=f"reverse {COLORS['subtext0']}")
        self._live_spinner = Live(
            Group(self._spinner, prompt),
            console=self.console,
            refresh_per_second=10,
            transient=True,
        )
        self._live_spinner.start()

    def stop_spinner(self) -> None:
        """Stop the thinking spinner if running."""
        if self._live_spinner is not None:
            self._live_spinner.stop()
            self._live_spinner = None
            self._spinner = None

    def print_agent(self, message: str, *, show_rule: bool = True, soft_wrap: bool = False) -> None:
        """Print an agent response, optionally with terminal-managed wrapping."""
        import textwrap

        # Aggressive cleanup for markdown rendering:
        # 1. Replace any non-breaking spaces with regular spaces
        cleaned = message.replace("\u00a0", " ").replace("\xa0", " ")
        # 2. Dedent and strip
        cleaned = textwrap.dedent(cleaned).strip()
        # 3. Remove common leading whitespace from all lines
        lines = cleaned.split("\n")
        non_empty_lines = [line for line in lines if line.strip()]
        if non_empty_lines:
            min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)
            if min_indent > 0:
                lines = [line[min_indent:] if len(line) >= min_indent else line for line in lines]
        # 4. Normalize line endings and rejoin
        cleaned = "\n".join(line.rstrip() for line in lines)

        if show_rule:
            self.console.print(
                Rule(title="[agent]OO[/agent]", style=COLORS["surface2"], align="left")
            )
        self.console.print(TerminalMarkdown(cleaned), soft_wrap=soft_wrap)

    def print_error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(f"[error]Error:[/error] {message}")

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self.console.print(f"[warning]Warning:[/warning] {message}")

    def print_status(self, message: str) -> None:
        """Print a dim status message."""
        self.console.print(f"[status]{message}[/status]")

    def print_success(self, message: str) -> None:
        """Print a success message."""
        self.console.print(f"[success]\u2713[/success] {message}")

    def print_info(self, message: str) -> None:
        """Print an info message."""
        self.console.print(f"[info]\u2139[/info] {message}")

    def print_table(
        self,
        title: str,
        columns: list[str],
        rows: list[list[str]],
        *,
        show_header: bool = True,
    ) -> None:
        """Print a formatted table with Catppuccin styling.

        ``show_header=False`` drops the column-header row — use it for
        key/value tables whose columns are unlabelled (otherwise Rich draws
        an empty header band above the first row).
        """
        table = Table(
            title=title or None,
            title_style=f"bold {COLORS['lavender']}",
            border_style=COLORS["surface2"],
            header_style=f"bold {COLORS['lavender']}",
            show_header=show_header,
        )
        for col in columns:
            table.add_column(col, style=COLORS["text"])
        for row in rows:
            table.add_row(*row)
        self.console.print(table)

    def print_help(self, commands: dict[str, str]) -> None:
        """Print help for available commands."""
        self.console.print(f"\n[bold {COLORS['mauve']}]Available Commands:[/]\n")
        for cmd, desc in commands.items():
            self.console.print(
                f"  [bold {COLORS['sapphire']}]{cmd}[/] - [{COLORS['subtext1']}]{desc}[/]"
            )
        self.console.print()
