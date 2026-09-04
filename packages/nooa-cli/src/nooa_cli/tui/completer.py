# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared completion engine for NOOA frontends.

``Completer`` is frontend-agnostic: it takes a text buffer and returns a list
of ``CompletionItem`` objects.  The terminal (prompt_toolkit adapter) uses
this engine, so completion behavior is identical whether run natively or
through the PTY web terminal (``nooa-term``).
"""

import keyword
import logging
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .commands import CommandRegistry

logger = logging.getLogger(__name__)


# An inline @ file/dir mention: an "@" at start-of-string or after whitespace,
# followed by a no-whitespace path fragment. Anchored with \Z by _active_mention
# (token being typed at the cursor) and used unanchored by expand_mentions (all
# mentions in a submitted line). input_handler imports _MENTION_PATTERN to gate
# its keybindings on the same definition.
_MENTION_PATTERN = r"(?:^|(?<=\s))@(\S+?)"


@dataclass(frozen=True)
class CompletionItem:
    """A single completion candidate."""

    text: str  # what gets inserted (the full replacement text)
    display: str  # what shows in the menu
    description: str = ""  # help text / description


class Completer:
    """Frontend-agnostic completion engine.

    Args:
        registry: CommandRegistry to pull slash-command completions from.
    """

    def __init__(
        self,
        registry: "CommandRegistry",
    ) -> None:
        self._registry = registry

    def complete(self, text: str) -> list[CompletionItem]:
        """Return completion candidates for *text*."""
        if text.startswith("/"):
            # An explicit @ token is unambiguous and composes with prompt-like
            # slash commands (for example, ``/plan @src/foo.py``).  Keep slash
            # command completion as the fallback so command-specific argument
            # completers retain their existing behavior otherwise.
            mention = self._active_mention(text)
            if mention is not None:
                return self._mention_completions(text, mention)
            return self._slash_completions(text)
        if text.startswith("!"):
            return self._bang_completions(text)
        # Inline @ file/dir mention — may appear anywhere in the buffer,
        # so this is checked for non-slash/bang input too.
        mention = self._active_mention(text)
        if mention is not None:
            return self._mention_completions(text, mention)
        return []

    # ------------------------------------------------------------------
    # Inline @ file/dir mentions
    # ------------------------------------------------------------------

    # The token being *typed* at the cursor: anchor the shared pattern at the
    # end of the buffer and allow an empty fragment ("@" alone opens the menu).
    _ACTIVE_MENTION_RE = re.compile(r"(?:^|(?<=\s))@(\S*)\Z")

    @classmethod
    def _active_mention(cls, text: str) -> "re.Match | None":
        """Return the @-mention token being typed at the end of *text*, or None."""
        return cls._ACTIVE_MENTION_RE.search(text)

    def _mention_completions(self, text: str, match: "re.Match") -> list[CompletionItem]:
        """Path completions for an inline ``@path`` token (prefix = buffer up to the @)."""
        partial = match.group(1)
        prefix = text[: match.start()] + "@"
        return self._path_completions(
            partial,
            prefix=prefix,
            base_dir=self._agent_working_dir(),
        )

    # ------------------------------------------------------------------
    # Slash commands
    # ------------------------------------------------------------------

    def _slash_completions(self, text: str) -> list[CompletionItem]:
        # Path completion after "/edit "
        lower = text.lower()
        if lower.startswith("/edit "):
            return self._path_completions(text[6:], prefix="/edit ")

        # Job name completion after "/jobs "
        if lower.startswith("/jobs "):
            return self._job_name_completions(text)

        # Theme name completion
        if lower.startswith("/theme "):
            return self._theme_completions(text)

        # Model completion
        if lower.startswith("/connect "):
            return self._model_registry_endpoint_completions(text)
        if lower.startswith("/model ") or lower.startswith("/keep-going model "):
            return self._model_completions(text)

        # Directory-only completion for adding a live skill root.
        if lower.startswith("/skills add "):
            return self._path_completions(
                text[len("/skills add ") :],
                prefix="/skills add ",
                directories_only=True,
                base_dir=self._agent_working_dir(),
            )

        # Toolbar item completion after the `set` action.
        if lower.startswith("/toolbar set "):
            return self._toolbar_item_completions(text)

        if lower.startswith("/keep-going "):
            prefix = "/keep-going "
            partial = text[len(prefix) :].lower()
            actions = {
                "on": "Enable stop auditing",
                "off": "Disable stop auditing",
                "model": "Configure the judge model",
            }
            return [
                CompletionItem(
                    text=prefix + action,
                    display=prefix + action,
                    description=description,
                )
                for action, description in actions.items()
                if action.startswith(partial)
            ]

        action_sets = {
            "/session ": {
                "new": "Start a new session",
                "resume": "Search for or resume a past session",
                "delete": "Delete a session",
                "export": "Export the current session as Markdown",
                "rename": "Rename the current session",
            },
            "/memory ": {
                "on": "Use project-wide memory",
                "local": "Use memory only for this session",
                "off": "Disable memory",
            },
            "/reflection ": {
                "on": "Enable idle memory reflection",
                "off": "Disable idle memory reflection",
                "now": "Run memory reflection immediately",
            },
            "/skills ": {
                "list": "List available skills",
                "add": "Add a skills directory",
                "commands": "Show slash commands supplied by skills",
                "activate": "Activate a skill",
                "deactivate": "Deactivate a skill",
            },
            "/show-python ": {
                "status": "Show whether Python execution display is on or off",
                "on": "Enable Python execution display",
                "off": "Suppress Python execution display",
            },
            "/show-diffs ": {
                "status": "Show whether inline file-edit diffs are on or off",
                "on": "Show inline file-edit diffs",
                "off": "Suppress inline file-edit diffs",
            },
            "/reasoning ": {
                "off": "Disable model reasoning",
                "low": "Use low reasoning effort",
                "medium": "Use medium reasoning effort",
                "high": "Use high reasoning effort",
            },
            "/toolbar ": {
                "reset": "Restore the default toolbar items",
                "set": "Choose the toolbar items to display",
            },
            "/mcp ": {
                "list": "Show configured, approved, and connected MCP servers",
                "add": "Add an MCP server to project settings",
                "remove": "Remove an inline project MCP server",
                "connect": "Connect an approved MCP server",
                "approve": "Review or approve one exact MCP configuration",
                "revoke": "Disconnect a server and revoke its approvals",
                "disconnect": "Disconnect an MCP server",
            },
        }
        for prefix, actions in action_sets.items():
            if lower.startswith(prefix):
                partial = text[len(prefix) :].lower()
                if any(char.isspace() for char in partial):
                    continue
                return [
                    CompletionItem(
                        text=prefix + action,
                        display=prefix + action,
                        description=description,
                    )
                    for action, description in actions.items()
                    if action.startswith(partial)
                ]

        # Skill ID completion for /skills activate and /skills deactivate
        for prefix in ("/skills activate ", "/skills deactivate "):
            if lower.startswith(prefix.lower()):
                return self._skill_id_completions(text, prefix)

        # Session ID completion
        for prefix in ("/session resume ", "/session delete ", "/resume "):
            if lower.startswith(prefix.lower()):
                return self._session_id_completions(text, prefix)

        # MCP server-name completion for actions that target existing servers.
        for prefix in (
            "/mcp connect ",
            "/mcp approve ",
            "/mcp revoke ",
            "/mcp disconnect ",
            "/mcp remove ",
        ):
            if lower.startswith(prefix.lower()):
                return self._mcp_server_completions(text, prefix)

        # Event tag completion — always return a non-empty list to
        # prevent prompt_toolkit's CompletionsMenu from crashing on
        # max() of an empty iterable.
        if lower.startswith("/events "):
            return self._event_tag_completions(text) or [
                CompletionItem(text=text, display=text, description="(no matching tags)")
            ]

        # Subcommand completion for @slash_command(completions=[...])
        items = self._skill_subcommand_completions(text)
        if items:
            return items

        # Top-level slash commands + subcommands.
        # get_active_help() keys may include argument hints ("/wtf-status [label]").
        # We strip the hint for text (insertion) but keep it for display.
        commands = self._registry.get_active_help()
        seen: set[str] = set()
        items: list[CompletionItem] = []
        for display_cmd, desc in commands.items():
            clean = re.split(r"\s+(?=[<\[])", display_cmd, maxsplit=1)[0].strip()
            if clean in seen:
                continue
            if clean.lower().startswith(text.lower()):
                seen.add(clean)
                items.append(CompletionItem(text=clean, display=display_cmd, description=desc))
        return items

    # ------------------------------------------------------------------
    # Skill subcommand completion
    # ------------------------------------------------------------------

    def _skill_subcommand_completions(self, text: str) -> list[CompletionItem]:
        """Complete subcommands for @slash_command methods that declare completions."""
        lower = text.lower()
        # Check each user skill for a matching prefix with completions
        for name, skill in self._registry._user_skills.items():
            cmd_prefix = f"/{name} "
            if not lower.startswith(cmd_prefix):
                continue
            if not skill.completions:
                return []
            partial = text[len(cmd_prefix) :]
            items = []
            for sub in skill.completions:
                if sub.lower().startswith(partial.lower()):
                    items.append(
                        CompletionItem(
                            text=cmd_prefix + sub,
                            display=cmd_prefix + sub,
                            description="",
                        )
                    )
            return items
        return []

    # ------------------------------------------------------------------
    # Theme completion
    # ------------------------------------------------------------------

    def _theme_completions(self, text: str) -> list[CompletionItem]:
        from .theme import theme_names

        prefix = "/theme "
        partial = text[len(prefix) :]
        items = []
        choices = (
            ("picker", "Browse installed themes"),
            ("gallery", "Browse and install remote themes"),
            ("update", "Download or refresh the remote catalog"),
            *((name, f"Switch to {name} theme") for name in theme_names()),
        )
        for name, description in choices:
            if name.startswith(partial.lower()):
                items.append(
                    CompletionItem(
                        text=prefix + name,
                        display=prefix + name,
                        description=description,
                    )
                )
        return items

    # ------------------------------------------------------------------
    # Model completion
    # ------------------------------------------------------------------

    def _model_completions(self, text: str) -> list[CompletionItem]:
        try:
            from nooa.unifiedllm import MODELS
        except Exception:
            return []

        prefix = (
            "/keep-going model " if text.lower().startswith("/keep-going model ") else "/model "
        )
        partial = text[len(prefix) :]
        description = (
            "Use {name} as keep-going judge"
            if prefix == "/keep-going model "
            else "Switch to {name}"
        )
        items = []
        for name in sorted(MODELS.keys()):
            if name.lower().startswith(partial.lower()):
                items.append(
                    CompletionItem(
                        text=prefix + name,
                        display=prefix + name,
                        description=description.format(name=name),
                    )
                )
        return items

    def _model_registry_endpoint_completions(self, text: str) -> list[CompletionItem]:
        """Complete deduplicated, credential-free API bases from loaded aliases."""
        try:
            from nooa.unifiedllm import MODELS

            from .model_catalog import ModelCatalogError, normalize_catalog_endpoint
        except Exception:
            return []

        prefix = "/connect "
        partial = text[len(prefix) :].casefold()
        aliases_by_endpoint: dict[str, list[str]] = {}
        for alias, config in MODELS.items():
            if not isinstance(config, dict):
                continue
            raw_endpoint = config.get("api_base")
            if not isinstance(raw_endpoint, str):
                continue
            try:
                endpoint, _models_url = normalize_catalog_endpoint(raw_endpoint)
            except ModelCatalogError:
                continue
            aliases_by_endpoint.setdefault(endpoint, []).append(str(alias))

        items: list[CompletionItem] = []
        for endpoint, aliases in sorted(aliases_by_endpoint.items()):
            if partial and not endpoint.casefold().startswith(partial):
                continue
            aliases.sort(key=str.casefold)
            if len(aliases) <= 3:
                description = "Used by " + ", ".join(aliases)
            else:
                description = f"Used by {len(aliases)} registry aliases"
            items.append(
                CompletionItem(
                    text=prefix + endpoint,
                    display=endpoint,
                    description=description,
                )
            )
        return items

    def _toolbar_item_completions(self, text: str) -> list[CompletionItem]:
        from .toolbar import ToolbarRegistry

        prefix = "/toolbar set "
        selected, _, partial = text[len(prefix) :].rpartition(" ")
        selected_items = selected.split() if selected else []
        replacement_prefix = prefix + (selected + " " if selected else "")
        return [
            CompletionItem(
                text=replacement_prefix + name,
                display=name,
                description="Toolbar item",
            )
            for name in ToolbarRegistry().names()
            if name not in selected_items and name.lower().startswith(partial.lower())
        ]

    # ------------------------------------------------------------------
    # Skill ID completion
    # ------------------------------------------------------------------

    def _skill_id_completions(self, text: str, prefix: str) -> list[CompletionItem]:
        agent = getattr(self._registry, "agent", None)
        if agent is None:
            return []
        skills_reg = getattr(agent, "skills", None)
        if skills_reg is None:
            return []

        partial = text[len(prefix) :]
        items = []
        try:
            for sid in sorted(skills_reg.discovered()):
                if partial and not sid.startswith(partial):
                    continue
                entry = skills_reg.entry(sid)
                desc = getattr(entry, "category", "") if entry else ""
                items.append(
                    CompletionItem(
                        text=prefix + sid,
                        display=prefix + sid,
                        description=desc,
                    )
                )
        except Exception:
            return []
        return items

    # ------------------------------------------------------------------
    # Session ID completion
    # ------------------------------------------------------------------

    def _session_id_completions(self, text: str, prefix: str) -> list[CompletionItem]:
        from .session_manager import SessionManager

        partial = text[len(prefix) :]
        try:
            sessions = SessionManager.list_sessions(limit=50)
        except Exception:
            return []

        items: list[CompletionItem] = []
        for meta in sessions:
            if meta.turn_count <= 0:
                continue
            sid = meta.id
            short = sid[:8]
            if partial and not short.startswith(partial) and not sid.startswith(partial):
                continue
            items.append(
                CompletionItem(
                    text=prefix + short,
                    display=prefix + short,
                    description=meta.name or sid,
                )
            )
        return items

    def _mcp_server_completions(self, text: str, prefix: str) -> list[CompletionItem]:
        """Complete safe configured names for MCP lifecycle operations.

        Reads the live ``self.mcp`` ``MCPRegistry`` so the candidates reflect both
        ``.mcp.json`` and inline ``settings.yaml`` servers,
        and annotates which are currently connected. ``disconnect`` only offers
        connected servers; ``connect`` and ``remove`` offer all configured servers.
        """
        partial = text[len(prefix) :]
        registry = self._registry
        mcp = getattr(getattr(registry, "agent", None), "mcp", None)
        if mcp is None:
            return []

        try:
            servers = mcp.discovered()
            connected = set(mcp.connected())
        except Exception:
            logger.debug("MCP server completion failed", exc_info=True)
            return []

        is_disconnect = prefix.strip().endswith("disconnect")
        names = sorted(connected) if is_disconnect else sorted(servers)

        items: list[CompletionItem] = []
        for name in names:
            attr_name = name.replace("-", "_").replace(" ", "_")
            if (
                not attr_name.isidentifier()
                or keyword.iskeyword(attr_name)
                or attr_name.startswith("_")
                or any(ord(char) < 32 or 127 <= ord(char) <= 159 for char in name)
            ):
                continue
            if partial and not name.lower().startswith(partial.lower()):
                continue
            if name in connected:
                status = "connected"
            else:
                try:
                    status = "approved" if mcp._is_approved(name) else "approval required"
                except Exception:
                    status = "configured"
            items.append(
                CompletionItem(
                    text=prefix + shlex.quote(name),
                    display=prefix + shlex.quote(name),
                    description=status,
                )
            )
        return items

    def _job_name_completions(self, text: str) -> list[CompletionItem]:
        prefix = "/jobs "
        partial = text[len(prefix) :].lower()
        agent = getattr(self._registry, "agent", None)
        qm = getattr(agent, "queue_manager", None) if agent else None
        if qm is None:
            return []

        items: list[CompletionItem] = []
        icons = {"running": "⏳", "done": "✅", "failed": "❌", "cancelled": "⏹"}
        for handle in sorted(qm.handles(), key=lambda item: item.job_id):
            candidates = (handle.job_id.lower(), handle.name.lower(), handle.label.lower())
            if partial and not any(candidate.startswith(partial) for candidate in candidates):
                continue
            buf = f" ({len(handle.values)} delivered)" if handle.values else ""
            items.append(
                CompletionItem(
                    text=prefix + handle.job_id,
                    display=prefix + handle.job_id,
                    description=(
                        f"{icons.get(handle.state, '?')} {handle.state} · "
                        f"{handle.label} → {handle.name}{buf}"
                    ),
                )
            )
        return items

    # ------------------------------------------------------------------
    # Event tag completion
    # ------------------------------------------------------------------

    def _event_tag_completions(self, text: str) -> list[CompletionItem]:
        prefix = "/events "
        partial = text[len(prefix) :]
        agent = getattr(self._registry, "agent", None)
        em = getattr(agent, "event_manager", None) if agent else None
        if em is None:
            return []

        icons = {
            "ToolCallEvent": "\U0001f527",
            "PythonOutput": "\U0001f4e4",
            "Task": "\U0001f4cb",
            "SessionUserMessage": "\U0001f4ac",
            "TUIUserInput": "\U0001f4ac",
            "SessionStarted": "\U0001f680",
            "TUISessionStart": "\U0001f680",
            "Summary": "\U0001f4dd",
            "Error": "\u274c",
        }

        items: list[CompletionItem] = []
        for tag in em.keys():
            if partial and not tag.startswith(partial):
                continue
            if tag == partial:
                continue  # skip exact match — already typed
            event = em.get(tag)
            if event is None:
                continue
            etype = getattr(event, "event_type", type(event).__name__)
            icon = icons.get(etype, "\U0001f4cc")

            # One-line summary
            summary = self._event_summary(event, etype)
            items.append(
                CompletionItem(
                    text=prefix + tag,
                    display=prefix + tag,
                    description=f"{icon} {summary}",
                )
            )
        return items

    @staticmethod
    def _event_summary(event, etype: str) -> str:
        """One-line summary for event tag completion."""
        if etype == "ToolCallEvent":
            name = getattr(event, "name", "?")
            args = getattr(event, "arguments", {})
            if (
                name in {"execute_python", "python_cell"}
                and isinstance(args, dict)
                and "code" in args
            ):
                for line in args["code"].split("\n"):
                    s = line.strip()
                    if s and not s.startswith("#"):
                        return f"{name} \u2014 {s[:50]}"
            return name
        elif etype == "PythonOutput":
            status = str(getattr(event, "execution_status", "?"))
            label = status.rsplit(".", 1)[-1]
            icon = "\u2705" if label == "complete" else "\u274c"
            stdout = getattr(event, "stdout", "") or ""
            error = getattr(event, "error", "") or ""
            if label != "complete" and error:
                return f"{icon} {error.strip().splitlines()[-1][:50]}"
            elif stdout:
                return f"{icon} {stdout.strip().splitlines()[0][:50]}"
            return f"{icon} {label}"
        elif etype in {"SessionUserMessage", "TUIUserInput"}:
            field = "content" if etype == "SessionUserMessage" else "text"
            text = getattr(event, field, "") or ""
            return text[:50] + ("..." if len(text) > 50 else "")
        elif etype in {"SessionStarted", "TUISessionStart"}:
            return f"model={getattr(event, 'model', '?')}"
        elif etype == "Task":
            prompt = getattr(event, "prompt", "") or ""
            first = prompt.strip().splitlines()[0] if prompt.strip() else ""
            return first[:50]
        return str(event)[:50]

    # ------------------------------------------------------------------
    # Bang commands
    # ------------------------------------------------------------------

    def _bang_completions(self, text: str) -> list[CompletionItem]:
        items: list[CompletionItem] = []

        rest = text[1:]  # strip leading !
        if " " not in rest:
            # Still typing the command name (single unterminated token, no
            # space yet). Bare "!" offers nothing; "!gi" completes $PATH
            # executables (git, gibo, …) rather than files named gi*.
            # A space ends the command token and switches to path completion.
            return items + self._command_completions(rest)

        # An argument is being typed (command token is followed by a space) →
        # complete filesystem paths on the last token, keeping the command +
        # earlier args as the prefix.
        head, _, last = rest.rpartition(" ")
        prefix = "!" + head + " "
        items.extend(self._path_completions(last, prefix=prefix))
        return items

    @staticmethod
    def _command_completions(partial: str) -> list[CompletionItem]:
        """Complete executable names from $PATH for the command token of !cmd."""
        if not partial:
            return []
        seen: set[str] = set()
        items: list[CompletionItem] = []
        for d in os.environ.get("PATH", "").split(os.pathsep):
            if not d:
                continue
            try:
                entries = sorted(os.listdir(d))
            except OSError:
                continue
            for name in entries:
                if name in seen or not name.startswith(partial):
                    continue
                full = os.path.join(d, name)
                if not (os.path.isfile(full) and os.access(full, os.X_OK)):
                    continue
                seen.add(name)
                items.append(CompletionItem(text="!" + name + " ", display=name, description=""))
        return items

    # ------------------------------------------------------------------
    # File path completion
    # ------------------------------------------------------------------

    def _path_completions(
        self,
        partial: str,
        prefix: str = "",
        *,
        directories_only: bool = False,
        base_dir: Path | None = None,
    ) -> list[CompletionItem]:
        """Complete filesystem paths from *partial*.

        Args:
            partial: The path fragment the user has typed so far.
            prefix: Command prefix to prepend to each item's ``text`` so that
                    the result is a full input replacement (e.g. ``"/edit "``).
            directories_only: Exclude files from the candidates.
            base_dir: Resolve relative lookup paths from this directory while
                      preserving the user's relative path in inserted text.
        """
        if not partial:
            partial = "."

        # Expand ~ to home dir
        expanded = os.path.expanduser(partial)
        p = Path(expanded)
        lookup = p if p.is_absolute() or base_dir is None else base_dir / p

        # Determine directory to list and filter prefix
        if lookup.is_dir():
            parent = lookup
            name_filter = ""
        else:
            parent = lookup.parent
            name_filter = lookup.name

        if not parent.exists():
            return []

        items: list[CompletionItem] = []
        try:
            for entry in sorted(parent.iterdir()):
                if directories_only and not entry.is_dir():
                    continue
                name = entry.name
                if name_filter and not name.lower().startswith(name_filter.lower()):
                    continue
                # Skip hidden files unless the user typed a dot
                if name.startswith(".") and not name_filter.startswith("."):
                    continue

                display = name + ("/" if entry.is_dir() else "")
                # Build the path portion
                if lookup.is_dir():
                    path_text = str(Path(partial) / name)
                else:
                    path_text = str(Path(partial).parent / name)

                if entry.is_dir():
                    path_text += "/"

                # Full input replacement = command prefix + path
                items.append(
                    CompletionItem(
                        text=prefix + path_text,
                        display=display,
                        description="",
                    )
                )
        except PermissionError:
            pass

        return items

    def _agent_working_dir(self) -> Path:
        agent = getattr(self._registry, "agent", None)
        return Path(getattr(agent, "cwd", Path.cwd())).expanduser().resolve()


# ---------------------------------------------------------------------------
# Mention expansion (used at submit time, shared by all frontends)
# ---------------------------------------------------------------------------

# Submit-time form of the mention: greedy so it grabs the whole path token.
# Trailing sentence punctuation is peeled back off the captured token below so
# "see @docs/file.md." still resolves to docs/file.md.
_MENTION_TOKEN = re.compile(r"(?:^|(?<=\s))@(\S+)")

# Punctuation that is almost always sentence-trailing rather than part of a
# real filename, stripped from the right of a captured mention before resolving.
_TRAILING_PUNCT = ".,;:!?)]}\"'"


def expand_mentions(text: str, *, base_dir: str | Path | None = None) -> str:
    """Expand inline ``@path`` mentions into Markdown links before sending.

    ``@docs/blah.md`` becomes ``[docs/blah.md](</abs/docs/blah.md>)`` so the
    agent receives an unambiguous absolute path while the user keeps the short
    form they typed. Only mentions resolving to an existing file/dir expand;
    emails (``a@b``), nonexistent paths, etc. are left untouched. Trailing
    sentence punctuation is peeled off before resolving. The link target is
    angle-bracketed so a path containing ``)`` can't break out of the link.
    """

    resolved_base = (
        Path(base_dir).expanduser().resolve() if isinstance(base_dir, (str, os.PathLike)) else None
    )

    def _sub(m: "re.Match") -> str:
        raw = m.group(1)
        # Peel trailing punctuation, but try the full token first so a real
        # filename that legitimately ends in such a char still resolves.
        stripped = raw.rstrip(_TRAILING_PUNCT)
        for candidate in (raw, stripped):
            p = Path(os.path.expanduser(candidate))
            resolved = p if p.is_absolute() or resolved_base is None else resolved_base / p
            if resolved.exists():
                trailer = raw[len(candidate) :]
                label = candidate.rstrip("/")
                return f"[{label}](<{resolved.resolve()}>){trailer}"
        return m.group(0)

    return _MENTION_TOKEN.sub(_sub, text)
