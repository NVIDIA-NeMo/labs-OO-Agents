# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Slash command parser and handlers for the NOOA TUI/Web hosts.

Commands return structured ``CommandResult`` objects whose ``outputs`` list
is rendered by the active ``Frontend``.  No command calls the console or
frontend directly for its results — it only uses ``self.frontend`` for
interactive I/O that must happen *during* execution (spinners, prompts).
"""

import abc
import asyncio
import datetime
import logging
import os
import re
import shlex
import urllib.parse
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

logger = logging.getLogger(__name__)

from .output import (  # noqa: E402
    AgentMessage,
    ClearScreen,
    CodeExecution,
    DiffOutput,
    HelpOutput,
    Output,
    TableOutput,
    TextOutput,
    _RichReplayPayload,
)
from .session_manager import SessionManager, build_resume_outputs  # noqa: E402

if TYPE_CHECKING:
    from nooa import Agent

    from .config import TUIConfig
    from .frontend import Frontend


def _mcp_auto_connect_names(value: object) -> list[str]:
    """Return validated MCP auto-connect names from config/test doubles."""
    if isinstance(value, list):
        return [v for v in value if isinstance(v, str)]
    return []


def _mcp_oauth_markdown_link(auth_url: str) -> str | None:
    """Return a safe Markdown link whose visible label is the complete URL."""
    safe_url = auth_url.strip()
    try:
        parsed = urllib.parse.urlsplit(safe_url)
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or any(
            character.isspace()
            or character in "<>"
            or ord(character) < 0x20
            or 0x7F <= ord(character) <= 0x9F
            for character in safe_url
        )
    ):
        return None
    label = re.sub(r"([\\`*_\[\]{}])", r"\\\1", safe_url)
    return f"[{label}](<{safe_url}>)"


def _batch_render_ctx(frontend: "Frontend"):
    """Return a context manager that batches a command's outputs into one block.

    Tolerant of frontends without ``batch_render`` and of test doubles whose
    ``batch_render()`` returns a Mock/coroutine instead of a real context
    manager — in either case we fall back to a no-op context. The real
    ``TerminalFrontend.batch_render()`` holds the ``_EmitStream`` flush so a
    multi-output command renders as a single ``run_in_terminal`` hop.
    """
    from contextlib import nullcontext

    factory = getattr(frontend, "batch_render", None)
    if not callable(factory):
        return nullcontext()
    ctx = factory()
    if hasattr(ctx, "__enter__") and hasattr(ctx, "__exit__"):
        return ctx
    # Mock / coroutine / anything non-context — don't try to enter it.
    if asyncio.iscoroutine(ctx):
        ctx.close()
    return nullcontext()


async def render_command_outputs(frontend: "Frontend", outputs: list[Output]) -> None:
    """Render command outputs, preserving Rich replay sentinel handling.

    ``_RichReplayPayload`` is an internal sentinel, not a public frontend output.
    CommandHandler normally intercepts it before rendering; Session uses this
    helper when output rendering is deferred until after the durable done line.
    """
    import os as _os

    _rich_url = (
        _os.environ.get("NEMO_OO_RICH_URL")
        if any(isinstance(o, _RichReplayPayload) for o in outputs)
        else None
    )
    with _batch_render_ctx(frontend):
        for output in outputs:
            if isinstance(output, _RichReplayPayload):
                if _rich_url:
                    try:
                        import httpx as _httpx

                        await asyncio.to_thread(
                            _httpx.post,
                            _rich_url,
                            json={**output.payload, "_replay": True},
                            timeout=5.0,
                        )
                    except Exception as exc:
                        logger.debug("replay POST to %s failed: %s", _rich_url, exc)
            else:
                await frontend.render(output)


def _detect_language(suffix: str) -> str:
    """Map a file extension to a language name for editor/diff rendering."""
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".sh": "bash",
        ".bash": "bash",
        ".toml": "toml",
        ".html": "html",
        ".css": "css",
        ".rs": "rust",
        ".go": "go",
        ".c": "c",
        ".cpp": "cpp",
        ".java": "java",
        ".rb": "ruby",
        ".sql": "sql",
    }.get(suffix.lower(), "plaintext")


# ---------------------------------------------------------------------------
# CommandResult
# ---------------------------------------------------------------------------


@dataclass
class CommandResult:
    """Result from a command execution."""

    success: bool
    outputs: list[Output] = field(default_factory=list)
    exit: bool = False
    # When set, Session.run() replaces the active SessionManager with this one.
    new_session_manager: "SessionManager | None" = None
    # Set by CompactCommand to signal that auto-renaming should be retried.
    compact_done: bool = False
    # Optional text the terminal host should place in an empty input buffer
    # after rendering command output. Used for explicit user-confirmation
    # steps such as MCP approval; prefill never submits the command.
    input_prefill: str | None = None
    # When set, Session.run() passes this as the user message for an agent turn.
    agent_message: str | None = None
    # Structured slash command result — passed to the agent on a queue.
    slash_result: "Any | None" = None
    # Optional callback Session runs after acknowledged cancellation and session
    # swap, on the agent loop when available. Used by /clear, /session new, and
    # /session resume for agent-state mutations that must not run on the UI loop
    # while a turn is active.
    post_session_swap: "Any | None" = None

    # Convenience constructors -------------------------------------------

    @classmethod
    def ok(cls, *outputs: "Output") -> "CommandResult":
        return cls(success=True, outputs=list(outputs))

    @classmethod
    def err(cls, message: str) -> "CommandResult":
        return cls(success=False, outputs=[TextOutput(message, "error")])

    @classmethod
    def bye(cls) -> "CommandResult":
        return cls(
            success=True,
            outputs=[TextOutput("Goodbye! Stay vibing.", "status")],
            exit=True,
        )


# ---------------------------------------------------------------------------
# Command base class
# ---------------------------------------------------------------------------


class Command(abc.ABC):
    """Abstract base class for all slash commands."""

    # Agent attributes that must be present for this command to be registered.
    required_capabilities: ClassVar[frozenset[str]] = frozenset()

    def __init__(
        self,
        frontend: "Frontend",
        config: "TUIConfig",
        agent: "Agent",
        session_manager: "SessionManager | None" = None,
        **kwargs,
    ):
        if agent is None:
            raise ValueError("agent cannot be None.")
        self.frontend = frontend
        self.config = config
        self.agent: Any = agent
        self.session_manager = session_manager
        self._registry: Any = kwargs.get("registry")
        self._root_config: Any = kwargs.get("root_config")
        # Dispatch operations to the agent thread. Set by CommandRegistry
        # after construction. Falls back to inline execution if not wired.
        self._agent_run = kwargs.get("agent_run")
        # Awaitable variant — set alongside _agent_run. Commands that run on
        # the UI loop must use this so they never block it (see
        # TUIApplication.agent_run_async).
        self._agent_run_async = kwargs.get("agent_run_async")

    def agent_run(self, fn):
        """Run *fn* on the agent thread. Falls back to inline if not wired."""
        if self._agent_run is not None:
            return self._agent_run(fn)
        return fn()

    async def agent_run_async(self, fn):
        """Awaitable ``agent_run`` — never blocks the calling loop.

        Falls back to inline execution when no async dispatcher is wired
        (tests / pre-startup), where there is no UI loop to protect.
        """
        if self._agent_run_async is not None:
            return await self._agent_run_async(fn)
        result = fn()
        if asyncio.iscoroutine(result):
            return await result
        return result

    @abc.abstractmethod
    async def execute(self, args: list[str]) -> "CommandResult":
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def help_text(cls) -> dict[str, str]:
        raise NotImplementedError

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if len(args) > 0:
            return False, f"Usage: /{self.name}"
        return True, None

    def _persist_tui_setting(self, key: str, value: object) -> Path:
        from .settings import write_settings_updates

        path, _data = write_settings_updates({("tui", key): value})
        return path


# ---------------------------------------------------------------------------
# Concrete commands
# ---------------------------------------------------------------------------


class HelpCommand(Command):
    def __init__(self, frontend, config, agent, **kwargs):
        super().__init__(frontend, config, agent, **kwargs)
        self._registry = kwargs.get("registry")

    @property
    def name(self) -> str:
        return "help"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {"/help": "Show this help message"}

    async def execute(self, args: list[str]) -> "CommandResult":
        commands_dict = (
            self._registry.get_builtin_help() if self._registry else CommandRegistry.get_help()
        )
        return CommandResult.ok(HelpOutput(commands_dict))


class ExitCommand(Command):
    @property
    def name(self) -> str:
        return "exit"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {
            "/exit": "Exit the TUI",
            "/quit": "Exit the TUI (alias for /exit)",
        }

    async def execute(self, args: list[str]) -> "CommandResult":
        return CommandResult.bye()


async def _reset_agent_working_state(agent: "Agent") -> None:
    """Reset the agent's in-memory working state for a fresh ``/clear``.

    The event manager is already pointed at the new (empty) storage via
    ``_swap_session_manager`` — that handles conversation history. Here
    we clear the *snapshotable* fields that live on the agent instance
    itself and don't get reset by storage swap alone.

    Guarded by ``hasattr`` / duck-typed checks so agents without a
    particular skill keep working.
    """
    # 1. TodoManager
    todo = getattr(agent, "todo", None)
    if todo is not None and hasattr(todo, "clear") and callable(todo.clear):
        try:
            todo.clear()
        except Exception:
            pass

    # 2. Persistent vars (self.v / agent.vars)
    if hasattr(agent, "vars") and isinstance(agent.vars, dict):
        agent.vars.clear()

    # 3. User-set context blocks (preserve framework-protected ones)
    cm = getattr(agent, "context_manager", None)
    if cm is not None:
        user_keys = [k for k in cm.keys() if k not in cm.protected_keys]
        for k in user_keys:
            try:
                del cm[k]
            except Exception:
                pass

    # 4. Shell session — kill and restart so env vars / cwd / aliases
    #    from the old session don't leak.
    shell = getattr(agent, "shell", None)
    if shell is not None and hasattr(shell, "reset") and callable(shell.reset):
        try:
            await shell.reset()
        except Exception:
            pass

    # 5. Notify skills the working state was just reset so they can
    #    re-initialize session-scoped state (symmetric with SessionResumed).
    em = getattr(agent, "event_manager", None)
    if em is not None:
        try:
            from nooa.sessions import SessionCleared

            em.register_event_type(SessionCleared)
            sid = None
            sm = getattr(agent, "_session_manager", None)
            if sm is not None:
                sid = getattr(sm, "session_id", None)
            em.add(SessionCleared(session_id=sid))
        except Exception:
            logger.debug("Failed to emit SessionCleared", exc_info=True)


class ClearCommand(Command):
    @property
    def name(self) -> str:
        return "clear"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {"/clear": "Start a new session (preserves old session history)"}

    async def execute(self, args: list[str]) -> "CommandResult":
        # Create a fresh SessionManager so subsequent turns go to a new file.
        # NOTE: do NOT call agent.event_manager.clear() here — that would
        # destroy the old session's SQLite data before _swap_session_manager()
        # can close and preserve it.  The new storage starts empty; the agent's
        # event_manager property will return the new backend after the swap.
        new_sm: SessionManager | None = None
        if self.session_manager is not None:
            try:
                new_sm = SessionManager.create(
                    model=self.session_manager.model,
                    agent_cls=self.session_manager.agent_cls,
                    working_dir=self.session_manager.working_dir,
                )
            except Exception:
                pass

        # Session._run_command performs acknowledged turn cancellation and
        # loop-affine QueueManager shutdown/flush during the actual session
        # swap. Do not call queue_manager.shutdown() here: spawned jobs live on
        # the agent loop, while commands run on the UI loop.

        # Session._run_command applies the agent working-state reset after
        # acknowledged turn cancellation and after the storage/session swap, on
        # the agent loop when available. Doing it here would run on the UI loop
        # and can race or deadlock with the active turn.

        outputs: list[Output] = [
            ClearScreen(),
            _RichReplayPayload(payload={"kind": "clear"}),  # type: ignore[list-item]
        ]
        if self._registry and self._registry.startup_info:
            outputs.append(self._registry.startup_info)
        outputs.append(TextOutput("Started new session. Previous session saved.", "success"))
        result = CommandResult(success=True, outputs=outputs)
        result.new_session_manager = new_sm
        result.post_session_swap = lambda: _reset_agent_working_state(self.agent)
        return result


class ModelCommand(Command):
    @property
    def name(self) -> str:
        return "model"

    def help_text(self) -> dict[str, str]:  # type: ignore[override]
        short = self.config.default_model.split("/")[-1] if hasattr(self, "config") else "?"
        return {
            "/model [NAME]": f"Switch and save a model (currently {short})",
        }

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if len(args) > 1:
            return False, "Usage: /model [name]"
        return True, None

    async def execute(self, args: list[str]) -> "CommandResult":
        if not args:
            return CommandResult.ok(
                TextOutput(f"Current model: {self.config.default_model}", "info")
            )
        selected = args[0]
        try:
            from nooa.interactive import apply_model_limits
            from nooa_cli.tui.config import UnresolvedModelError, get_llm_for_model
            from nooa_cli.tui.health_check import probe_llm

            try:
                candidate = get_llm_for_model(selected)
            except UnresolvedModelError:
                from nooa_cli.tui.health_check import unresolved_model_health

                health = unresolved_model_health(selected)
                return CommandResult(
                    success=False,
                    outputs=[
                        TextOutput(
                            f"Could not switch to model '{selected}': {health.error_message}",
                            "error",
                        ),
                        TextOutput(health.fix_hint or "", "info"),
                    ],
                )
            health = await probe_llm(candidate)
            if not health.ok:
                outputs = [
                    TextOutput(
                        f"Could not switch to model '{selected}': {health.error_message}", "error"
                    )
                ]
                if health.fix_hint:
                    outputs.append(TextOutput(health.fix_hint, "info"))
                return CommandResult(success=False, outputs=outputs)

            def _switch():
                self.agent.set_llm(candidate)
                apply_model_limits(self.agent)

            await self.agent_run_async(_switch)
        except Exception as e:
            return CommandResult.err(f"Failed to switch model: {e}")
        self.config.default_model = selected
        if self._registry is not None:
            self._registry.blocking_llm_health = None
            startup_info = getattr(self._registry, "startup_info", None)
            if startup_info is not None:
                from nooa_cli.tui.session import _short_model_name

                startup_info.model = selected
                startup_info.short_model = _short_model_name(selected)
                startup_info.llm_ready = True
                startup_info.llm_status = "ready"
        try:
            settings_path = self._persist_tui_setting("default_model", selected)
        except Exception as e:
            logger.warning("Failed to persist selected TUI model %r: %s", selected, e)
            return CommandResult.ok(
                TextOutput(f"Switched to model: {selected}", "success"),
                TextOutput(f"Could not save the project default: {e}", "warning"),
            )
        return CommandResult.ok(
            TextOutput(f"Switched to model: {selected}", "success"),
            TextOutput(f"Saved as the project default in {settings_path}", "status"),
        )

    async def _add_to_registry(self, server_url: str) -> "CommandResult":
        """Run the host-rendered workflow for an OpenAI-compatible server."""
        from nooa.paths import get_project_dir
        from nooa.unifiedllm import resolve_api_key_from_config

        from .model_catalog import (
            ModelCatalogError,
            default_alias,
            fetch_model_catalog,
            is_anthropic_endpoint,
            lookup_model_token_limits,
            normalize_catalog_endpoint,
            probe_ollama_backend,
            registry_entry,
            suggested_api_key_env,
            validate_alias,
            validate_api_key_env,
        )

        prompt_text = getattr(self.frontend, "prompt_text", None)
        prompt_choice = getattr(self.frontend, "prompt_choice", None)
        prompt_sensitive = getattr(self.frontend, "prompt_sensitive", None)
        if not callable(prompt_text) or not callable(prompt_choice):
            return CommandResult.err("This frontend does not support interactive model setup.")

        try:
            api_base, _models_url = normalize_catalog_endpoint(server_url)
        except ModelCatalogError as exc:
            return CommandResult.err(str(exc))

        if is_anthropic_endpoint(api_base):
            return await self._add_native_provider("anthropic")

        try:
            api_key_env = validate_api_key_env(suggested_api_key_env(api_base))
        except ModelCatalogError as exc:
            return CommandResult.err(str(exc))
        api_key = (
            resolve_api_key_from_config("model-catalog", {"api_key_env": api_key_env})
            if api_key_env
            else None
        )
        pending_secret: tuple[str, str] | None = None
        if api_key_env and not api_key:
            if not callable(prompt_sensitive):
                return CommandResult.err(
                    f"{api_key_env} is not set and this frontend cannot collect masked secrets. "
                    "Export it and run the command again."
                )
            api_key = await prompt_sensitive(
                "Model server API key",
                f"Paste the API key for {api_base}. "
                f"It will be saved as {api_key_env} in project secrets.yaml.",
            )
            if not api_key:
                return CommandResult.ok(TextOutput("Model setup cancelled.", "info"))
            pending_secret = (api_key_env, api_key)

        try:
            api_base, models = await asyncio.to_thread(
                fetch_model_catalog, server_url, api_key=api_key
            )
        except ModelCatalogError as exc:
            auth_rejected = "rejected authentication" in str(exc).lower()
            if not auth_rejected or not callable(prompt_sensitive):
                return CommandResult.err(str(exc))
            if not api_key_env:
                api_key_env = await prompt_text(
                    "Model server authentication",
                    "API key environment variable to save this backend's key.",
                    "MODEL_API_KEY",
                )
                try:
                    api_key_env = validate_api_key_env(api_key_env)
                except ModelCatalogError as env_exc:
                    return CommandResult.err(str(env_exc))
                if not api_key_env:
                    return CommandResult.err(str(exc))
            api_key = await prompt_sensitive(
                "Model server API key",
                f"Paste the API key for {api_base}. "
                f"It will replace {api_key_env} in project secrets.yaml.",
            )
            if not api_key:
                return CommandResult.ok(TextOutput("Model setup cancelled.", "info"))
            try:
                api_base, models = await asyncio.to_thread(
                    fetch_model_catalog, server_url, api_key=api_key
                )
            except ModelCatalogError as retry_exc:
                return CommandResult.err(str(retry_exc))
            pending_secret = (api_key_env, api_key)

        ollama = not api_key_env and await asyncio.to_thread(probe_ollama_backend, api_base)

        selected_id = await prompt_choice(
            "Add model",
            f"Found {len(models):,} models at {api_base}. Type to filter, then choose one.",
            [model.id for model in models],
        )
        if not selected_id:
            return CommandResult.ok(TextOutput("Model setup cancelled.", "info"))
        selected = next((model for model in models if model.id == selected_id), None)
        if selected is None:
            return CommandResult.ok(TextOutput("Model setup cancelled.", "info"))

        inferred_context: int | None = None
        inferred_output: int | None = None
        if selected.context_window is None or selected.max_tokens is None:
            inferred_context, inferred_output = await asyncio.to_thread(
                lookup_model_token_limits, selected.id
            )

        try:
            alias = validate_alias(default_alias(selected.id))
            entry = registry_entry(
                selected.id,
                api_base,
                api_key_env,
                context_window=selected.context_window or inferred_context,
                max_tokens=selected.max_tokens or inferred_output,
                ollama=ollama,
            )
        except ModelCatalogError as exc:
            return CommandResult.err(str(exc))

        return await self._finalize_alias_and_switch(
            alias,
            entry,
            get_project_dir("llm_config.yaml"),
            prompt_choice,
            pending_secret=pending_secret,
        )

    async def _add_native_provider(self, provider: str) -> "CommandResult":
        """Run the host-rendered workflow for a native LiteLLM provider."""
        from nooa.paths import get_project_dir
        from nooa.unifiedllm import resolve_api_key_from_config

        from .model_catalog import (
            ModelCatalogError,
            default_alias,
            fetch_native_provider_models,
            lookup_model_token_limits,
            native_provider_api_key_env,
            native_provider_registry_entry,
            normalize_native_provider,
            validate_alias,
        )

        normalized = normalize_native_provider(provider)
        if normalized is None:
            return CommandResult.err(f"Unsupported provider '{provider}'.")

        prompt_text = getattr(self.frontend, "prompt_text", None)
        prompt_choice = getattr(self.frontend, "prompt_choice", None)
        prompt_sensitive = getattr(self.frontend, "prompt_sensitive", None)
        if not callable(prompt_text) or not callable(prompt_choice):
            return CommandResult.err("This frontend does not support interactive model setup.")

        provider_label = normalized.title()
        api_key_env = native_provider_api_key_env(normalized)
        api_key = resolve_api_key_from_config(
            f"{normalized}-model-setup", {"api_key_env": api_key_env}
        )
        pending_secret: tuple[str, str] | None = None
        if not api_key:
            if not callable(prompt_sensitive):
                return CommandResult.err(
                    f"{api_key_env} is not set and this frontend cannot collect masked secrets. "
                    "Export it and run the command again."
                )
            api_key = await prompt_sensitive(
                f"{provider_label} API key",
                f"Paste the API key for {provider_label}. "
                f"It will be saved as {api_key_env} in project secrets.yaml.",
            )
            if not api_key:
                return CommandResult.ok(TextOutput("Model setup cancelled.", "info"))
            pending_secret = (api_key_env, api_key)

        try:
            discovered_models = await asyncio.to_thread(
                fetch_native_provider_models, normalized, api_key
            )
        except ModelCatalogError as exc:
            auth_rejected = "rejected authentication" in str(exc).lower()
            if not auth_rejected or not callable(prompt_sensitive):
                return CommandResult.err(str(exc))
            api_key = await prompt_sensitive(
                f"{provider_label} API key",
                f"Paste the API key for {provider_label}. "
                f"It will replace {api_key_env} in project secrets.yaml.",
            )
            if not api_key:
                return CommandResult.ok(TextOutput("Model setup cancelled.", "info"))
            try:
                discovered_models = await asyncio.to_thread(
                    fetch_native_provider_models, normalized, api_key
                )
            except ModelCatalogError as retry_exc:
                return CommandResult.err(str(retry_exc))
            pending_secret = (api_key_env, api_key)

        model_choices = [model.id for model in discovered_models]
        model_choices.append("Custom model...")
        selected_id = await prompt_choice(
            f"Add {provider_label} model",
            f"Found {len(discovered_models):,} models. Type to filter, then choose one.",
            model_choices,
        )
        if not selected_id:
            return CommandResult.ok(TextOutput("Model setup cancelled.", "info"))
        if selected_id == "Custom model...":
            selected_id = await prompt_text(
                f"{provider_label} model",
                "Enter a LiteLLM-compatible model name.",
                model_choices[0],
            )
            if not selected_id:
                return CommandResult.ok(TextOutput("Model setup cancelled.", "info"))

        inferred_context, inferred_output = await asyncio.to_thread(
            lookup_model_token_limits, selected_id
        )

        try:
            alias = validate_alias(default_alias(selected_id))
            entry = native_provider_registry_entry(
                normalized,
                selected_id,
                api_key_env,
                context_window=inferred_context,
                max_tokens=inferred_output,
            )
        except ModelCatalogError as exc:
            return CommandResult.err(str(exc))

        return await self._finalize_alias_and_switch(
            alias,
            entry,
            get_project_dir("llm_config.yaml"),
            prompt_choice,
            pending_secret=pending_secret,
        )

    async def _persist_pending_secret(self, pending: tuple[str, str]) -> "CommandResult | None":
        """Write a validated secret to project secrets.yaml; return err on failure."""
        from nooa.paths import get_project_dir

        from .model_catalog import ModelCatalogError, write_secret_env

        name, value = pending
        try:
            await asyncio.to_thread(write_secret_env, get_project_dir("secrets.yaml"), name, value)
        except (ModelCatalogError, OSError, ValueError) as exc:
            return CommandResult.err(f"Could not save {name} to secrets.yaml: {exc}")
        return None

    async def _finalize_alias_and_switch(
        self,
        alias: str,
        entry: dict[str, Any],
        registry_path: Path,
        prompt_choice: Callable[..., Awaitable[str | None]],
        *,
        pending_secret: tuple[str, str] | None = None,
    ) -> "CommandResult":
        """Persist confirmed setup, reload the registry, and switch unless replace-only."""
        from .model_catalog import ModelCatalogError, model_alias_exists, write_model_alias

        try:
            alias_exists = await asyncio.to_thread(model_alias_exists, registry_path, alias)
        except (ModelCatalogError, OSError, ValueError) as exc:
            return CommandResult.err(f"Could not inspect {registry_path}: {exc}")
        action = "Use now"
        if alias_exists:
            action = await prompt_choice(
                "Replace model alias",
                f"Alias '{alias}' already exists in {registry_path}.",
                ["Replace and use now", "Replace only", "Cancel"],
            )
            if not action or action == "Cancel":
                return CommandResult.ok(TextOutput("Model setup cancelled.", "info"))

        try:
            await asyncio.to_thread(
                write_model_alias, registry_path, alias, entry, replace=alias_exists
            )
            if pending_secret is not None:
                error = await self._persist_pending_secret(pending_secret)
                if error is not None:
                    return error
            await asyncio.to_thread(self._reload_model_registry)
        except (ModelCatalogError, OSError, ValueError) as exc:
            return CommandResult.err(f"Could not update {registry_path}: {exc}")

        added = TextOutput(f"Saved model '{alias}' to {registry_path}", "success")
        if action == "Replace only":
            return CommandResult.ok(
                added,
                TextOutput(f"Use /model {alias} when you are ready to switch.", "info"),
            )
        switched = await ModelCommand.execute(self, [alias])
        switched.outputs.insert(0, added)
        return switched

    def _reload_model_registry(self) -> None:
        """Reload discovered and explicitly supplied registry layers in place."""
        from nooa.llm_config import llm_config_chain
        from nooa.unifiedllm import reload_registry

        paths = llm_config_chain()
        explicit = getattr(self._root_config, "llm_config_paths", None) or []
        paths.extend(Path(path) for path in explicit if Path(path) not in paths)
        reload_registry(*paths)


class ConnectCommand(ModelCommand):
    """Friendly model-backend setup entry point."""

    @property
    def name(self) -> str:
        return "connect"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {"/connect [server-url]": "Connect a model backend by URL"}

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if len(args) > 1:
            return False, "Usage: /connect [server-url]"
        return True, None

    async def execute(self, args: list[str]) -> "CommandResult":
        server_url = args[0] if args else ""
        if not server_url:
            prompt_text = getattr(self.frontend, "prompt_text", None)
            if not callable(prompt_text):
                return CommandResult.err("Usage: /connect <server-url>")
            server_url = await prompt_text(
                "Connect model backend",
                "API base URL (Anthropic, Ollama, and OpenAI-compatible servers are auto-detected).",
                "http://localhost:11434",
            )
            if not server_url:
                return CommandResult.ok(TextOutput("Model setup cancelled.", "info"))
        return await self._add_to_registry(server_url)


class ModelsCommand(Command):
    @property
    def name(self) -> str:
        return "models"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {"/models": "List available models from registry"}

    async def execute(self, args: list[str]) -> "CommandResult":
        from nooa.unifiedllm import MODELS

        rows: list[list[str]] = []
        for model_name in sorted(MODELS.keys()):
            marker = "\u25c0" if model_name == self.config.default_model else ""
            rows.append([model_name, marker])

        return CommandResult.ok(
            TableOutput(
                title="Available Models",
                columns=["Model", ""],
                rows=rows,
                footer="Use /model <name> to switch. Any model supported by litellm works, not just these aliases.",
            )
        )


class ReasoningCommand(Command):
    """Toggle reasoning mode for the current model."""

    @property
    def name(self) -> str:
        return "reasoning"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {"/reasoning [off|low|medium|high]": "Toggle reasoning mode for the current model"}

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if len(args) > 1:
            return False, "Usage: /reasoning [off|low|medium|high]"
        if args and args[0].lower() not in ("off", "low", "medium", "high"):
            return False, "Usage: /reasoning [off|low|medium|high]"
        return True, None

    def _get_reasoning_state(self) -> tuple[str, str]:
        """Return (effort_level, client_type) for the current model.

        effort_level is 'off', 'low', 'medium', or 'high'.
        client_type is 'responses' or 'completion'.
        """
        from nooa.unifiedllm.unifiedllm import ResponsesClient

        llm = self.agent.llm
        is_responses = isinstance(llm, ResponsesClient)
        client_type = "responses" if is_responses else "completion"

        if is_responses:
            reasoning = llm.config.get("reasoning")
            if reasoning and isinstance(reasoning, dict):
                return reasoning.get("effort", "off"), client_type
            return "off", client_type
        else:
            effort = llm.config.get("reasoning_effort")
            if effort:
                return str(effort), client_type
            return "off", client_type

    def _set_reasoning(self, level: str) -> None:
        """Set reasoning level on the live LLM client config."""
        from nooa.unifiedllm.unifiedllm import ResponsesClient

        llm = self.agent.llm
        is_responses = isinstance(llm, ResponsesClient)

        if level == "off":
            if is_responses:
                llm.config.pop("reasoning", None)
            else:
                llm.config.pop("reasoning_effort", None)
        else:
            if is_responses:
                llm.config["reasoning"] = {"effort": level}
            else:
                llm.config["reasoning_effort"] = level
                # Ensure the gateway knows this param is allowed
                allowed = llm.config.get("allowed_openai_params")
                if isinstance(allowed, list):
                    if "reasoning_effort" not in allowed:
                        allowed.append("reasoning_effort")
                else:
                    llm.config["allowed_openai_params"] = ["reasoning_effort"]

    async def execute(self, args: list[str]) -> "CommandResult":
        if not args:
            effort, client_type = self._get_reasoning_state()
            model = self.config.default_model
            status = f"**{effort}**" if effort != "off" else "off"
            return CommandResult.ok(
                TextOutput(
                    f"Reasoning for {model} ({client_type}): {status}",
                    "info",
                )
            )

        level = args[0].lower()

        def _apply():
            self._set_reasoning(level)

        try:
            await self.agent_run_async(_apply)
        except Exception as e:
            return CommandResult.err(f"Failed to set reasoning: {e}")
        model = self.config.default_model
        if level == "off":
            return CommandResult.ok(TextOutput(f"Reasoning disabled for {model}", "success"))
        return CommandResult.ok(TextOutput(f"Reasoning set to {level} for {model}", "success"))


class ThemeCommand(Command):
    """Switch the color theme."""

    THEMES = ("mocha", "latte", "vsdark", "vslight")

    @property
    def name(self) -> str:
        return "theme"

    def help_text(self) -> dict[str, str]:  # type: ignore[override]
        from . import theme as theme_module

        current = theme_module.get_theme() if hasattr(self, "config") else "?"
        return {
            "/theme [name]": f"Switch theme (currently {current})",
        }

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if len(args) > 1:
            return False, f"Usage: /theme [{'|'.join(self.THEMES)}]"
        if len(args) == 1 and args[0].lower() not in self.THEMES:
            return False, f"Theme must be one of: {', '.join(self.THEMES)}"
        return True, None

    async def execute(self, args: list[str]) -> "CommandResult":
        from . import theme as theme_module

        if not args:
            current = theme_module.get_theme()
            others = ", ".join(t for t in self.THEMES if t != current)
            return CommandResult.ok(
                TextOutput(f"Current theme: {current}  (available: {others})", "info")
            )

        name = args[0].lower()
        theme_module.set_theme(name)

        # Replace the base theme in Rich Console's ThemeStack
        # We can't just push - we need to replace the base entry
        if hasattr(self.frontend, "_console") and hasattr(self.frontend._console, "console"):  # type: ignore[attr-defined]
            console = self.frontend._console.console  # type: ignore[attr-defined]
            new_theme = theme_module.create_theme()

            # Directly replace the base theme in the stack
            # This is the only way to actually change colors since Theme snapshots
            # the COLORS dict at creation time
            console._theme_stack._entries[0] = new_theme.styles
            console._theme_stack.get = console._theme_stack._entries[-1].get

        # Rebuild prompt_toolkit style for the new theme
        if hasattr(self.frontend, "_input_handler") and self.frontend._input_handler is not None:  # type: ignore[attr-defined]
            self.frontend._input_handler.refresh_style()  # type: ignore[attr-defined]
        app = getattr(self.frontend, "_app", None)
        refresh_app_style = getattr(app, "refresh_style", None)
        if callable(refresh_app_style):
            refresh_app_style()

        return CommandResult.ok(TextOutput(f"Switched to {name} theme", "success"))


# ---------------------------------------------------------------------------
# Skills commands
# ---------------------------------------------------------------------------


class SkillsCommand(Command):
    required_capabilities: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, frontend, config, agent, **kwargs):
        super().__init__(frontend, config, agent, **kwargs)
        self.skills_dirs = kwargs.get("skills_dirs")
        self._registry: CommandRegistry | None = kwargs.get("registry")

    @property
    def name(self) -> str:
        return "skills"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {
            "/skills <list|add DIR|commands|activate ID|deactivate ID>": (
                "List and manage skills or show their slash commands"
            ),
        }

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if not args:
            return False, "Usage: /skills <list|add|activate|deactivate|commands>"
        if args[0].lower() not in ("list", "add", "activate", "deactivate", "commands"):
            return False, f"Unknown subcommand `{args[0]}`"
        if args[0].lower() == "add" and len(args) != 2:
            return False, "Usage: /skills add <directory>"
        if args[0].lower() in ("activate", "deactivate") and len(args) < 2:
            return False, f"Usage: /skills {args[0]} <skill_id>"
        return True, None

    async def execute(self, args: list[str]) -> "CommandResult":
        subcmd = args[0].lower()
        subargs = args[1:]

        if subcmd == "add":
            if self._registry is None:
                return CommandResult.err("The command registry is unavailable.")
            raw_path = Path(subargs[0]).expanduser()
            base = Path(getattr(self.agent, "cwd", Path.cwd()))
            path = (base / raw_path).resolve() if not raw_path.is_absolute() else raw_path.resolve()
            if not path.is_dir():
                return CommandResult.err(f"Skills directory not found: {path}")

            before_skills = set(
                getattr(getattr(self.agent, "skills", None), "discovered", lambda: [])()
            )
            before_commands = set(self._registry._user_skills)
            try:
                added = self._registry.add_skills_dir(path)
            except Exception as exc:
                return CommandResult.err(f"Failed to add skills directory {path}: {exc}")

            persisted = list(
                dict.fromkeys(
                    Path(item).expanduser().resolve()
                    for item in getattr(self.config, "additional_skills_dirs", [])
                )
            )
            if path not in persisted:
                persisted.append(path)
                self.config.additional_skills_dirs = persisted
            try:
                settings_path = self._persist_tui_setting(
                    "additional_skills_dirs", [str(item) for item in persisted]
                )
            except Exception as exc:
                return CommandResult.ok(
                    TextOutput(f"Added skills directory: {path}", "success"),
                    TextOutput(f"Could not save the skills directory: {exc}", "warning"),
                )

            after_skills = set(
                getattr(getattr(self.agent, "skills", None), "discovered", lambda: [])()
            )
            after_commands = set(self._registry._user_skills)
            detail = (
                f"Discovered {len(after_skills - before_skills)} skill(s) and "
                f"{len(after_commands - before_commands)} slash command(s)."
            )
            verb = "Added" if added else "Already using"
            return CommandResult.ok(
                TextOutput(f"{verb} skills directory: {path}", "success"),
                TextOutput(detail, "info"),
                TextOutput(f"Saved in {settings_path}", "status"),
            )

        if subcmd == "commands":
            user_skills = self._registry._user_skills if self._registry else {}
            rows_cmd = [
                [f"/{name}", skill.argument_hint or "", skill.description]
                for name, skill in sorted(user_skills.items())
            ]
            if rows_cmd:
                return CommandResult.ok(
                    TableOutput(
                        columns=["Command", "Args", "Description"],
                        rows=rows_cmd,
                        title="Skill slash commands",
                    ),
                    TextOutput(f"Searched: {self.skills_dirs}", "status"),
                )
            return CommandResult.ok(
                TextOutput("No user-invocable skill commands found.", "info"),
                TextOutput(f"Searched: {self.skills_dirs}", "status"),
            )

        from nooa.skill_registry import SkillRegistry

        registry = getattr(self.agent, "skills", None)
        if not isinstance(registry, SkillRegistry):
            return CommandResult.err(
                "Agent has no SkillRegistry. Skills require self.skills = SkillRegistry(self)."
            )

        if subcmd == "list":
            all_names = registry.discovered()
            activated = set(registry.activated())
            if not all_names:
                return CommandResult.ok(TextOutput("No skills found", "info"))
            rows = [[name, "\u2713" if name in activated else "", ""] for name in all_names]
            return CommandResult.ok(
                TableOutput(columns=["ID", "Active", "Description"], rows=rows, title="Skills"),
            )

        if subcmd == "activate":
            skill_id = subargs[0]
            if skill_id in registry.activated():
                return CommandResult.err(f"Skill `{skill_id}` already active")
            if skill_id not in registry.discovered():
                return CommandResult.err(f"Skill `{skill_id}` not found. Use /skills list.")
            try:
                registry.activate([skill_id])
            except Exception as e:
                return CommandResult.err(f"Failed to activate `{skill_id}`: {e}")
            active = list(dict.fromkeys([*self.config.active_skills, skill_id]))
            self.config.active_skills = active
            try:
                self._persist_tui_setting("active_skills", active)
            except Exception as exc:
                return CommandResult.ok(
                    TextOutput(f"Skill `{skill_id}` activated", "success"),
                    TextOutput(f"Could not save skill activation: {exc}", "warning"),
                )
            return CommandResult.ok(TextOutput(f"Skill `{skill_id}` activated", "success"))

        # deactivate
        skill_id = subargs[0]
        if skill_id not in registry.activated():
            return CommandResult.err(f"`{skill_id}` not active. Use /skills list.")
        try:
            registry.deactivate([skill_id])
        except Exception as e:
            return CommandResult.err(f"Failed to deactivate `{skill_id}`: {e}")
        active = [name for name in self.config.active_skills if name != skill_id]
        self.config.active_skills = active
        try:
            self._persist_tui_setting("active_skills", active)
        except Exception as exc:
            return CommandResult.ok(
                TextOutput(f"Skill `{skill_id}` deactivated", "success"),
                TextOutput(f"Could not save skill deactivation: {exc}", "warning"),
            )
        return CommandResult.ok(TextOutput(f"Skill `{skill_id}` deactivated", "success"))


# ---------------------------------------------------------------------------
# Todo commands
# ---------------------------------------------------------------------------


class ContextCommand(Command):
    """Show context window utilization stats."""

    @property
    def name(self) -> str:
        return "context"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {"/context": "Show context window utilization"}

    async def execute(self, args: list[str]) -> "CommandResult":
        stats = getattr(self.agent, "context_stats", None)
        if stats is None:
            return CommandResult.ok(
                TextOutput("No context stats yet — run a generation first.", "info")
            )
        return CommandResult.ok(TextOutput(stats.format(), "info"))


# ---------------------------------------------------------------------------
# Compact command
# ---------------------------------------------------------------------------


class CompactCommand(Command):
    """Summarize and compact conversation history."""

    @property
    def name(self) -> str:
        return "compact"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {"/compact": "Summarize conversation history into a compact block (frees tokens)"}

    async def execute(self, args: list[str]) -> "CommandResult":
        if not hasattr(self.agent, "event_manager"):
            return CommandResult.err("Agent does not support history management.")

        tags = self.agent.event_manager.keys()
        events_before = len(tags)
        if events_before == 0:
            return CommandResult.ok(
                TextOutput("Nothing to compact \u2014 history is already empty.", "info")
            )

        tokens_before = 0
        stats = self.agent.context_stats
        if stats:
            tokens_before = stats.total_tokens or 0
        summarizers = getattr(self.agent, "_summarizers", [])

        if summarizers:
            summarizer = summarizers[0]
            start_tag = tags[0]
            end_tag = tags[-1]
            # A static status line — not start_thinking(). The latter spins a
            # rich.live.Live (~10 fps) layered on top of prompt_toolkit's
            # full-screen app; each repaint hops through emit_block /
            # run_in_terminal and fights pt's own status spinner, which the
            # user sees as flicker. Compaction runs on the UI loop (a command,
            # not an agent turn) so the native "thinking…" status never shows
            # anyway — one status block plus the result block is enough.
            await self.frontend.render(TextOutput("Summarizing history\u2026", "info"))
            try:
                history_md = summarizer._render_range_to_markdown(start_tag, end_tag)
                target_chars = getattr(getattr(summarizer, "config", None), "target_chars", 2000)
                summary_text = await summarizer.summarize(history_md, target_chars)
                await self.agent_run_async(
                    lambda: self.agent.event_manager.collapse(start_tag, end_tag, summary_text)
                )
                events_after = len(self.agent.event_manager.keys())
                tok_sfx = f" (~{tokens_before:,} tokens freed)" if tokens_before else ""
                result = CommandResult.ok(
                    TextOutput(
                        f"Compacted {events_before} events \u2192 {events_after} (summary block){tok_sfx}.",
                        "success",
                    )
                )
                result.compact_done = True
                return result
            except Exception as e:
                return CommandResult.ok(
                    TextOutput(
                        f"Summarization failed ({e}); kept {events_before} events intact.",
                        "warning",
                    )
                )

        await self.agent_run_async(lambda: self.agent.event_manager.clear())
        tok_sfx = f" (~{tokens_before:,} tokens freed)" if tokens_before else ""
        result = CommandResult.ok(
            TextOutput(f"Cleared {events_before} history events{tok_sfx}.", "success")
        )
        result.compact_done = True
        return result


# ---------------------------------------------------------------------------
# Display toggles
# ---------------------------------------------------------------------------


class ShowPythonCommand(Command):
    """Toggle display of the Python execution panel."""

    @property
    def name(self) -> str:
        return "show-python"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {"/show-python [status|on|off]": "Configure Python execution display"}

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if len(args) > 1 or (args and args[0].lower() not in ("status", "on", "off")):
            return False, f"Unknown subcommand `{args[0]}`"
        return True, None

    async def execute(self, args: list[str]) -> "CommandResult":
        subcmd = args[0].lower() if args else "status"

        if subcmd == "status":
            state = "on" if self.config.show_python else "off"
            return CommandResult.ok(TextOutput(f"Python execution display: {state}", "info"))

        enabled = subcmd == "on"
        self.config.show_python = enabled
        label = "enabled" if enabled else "suppressed"
        try:
            path = self._persist_tui_setting("show_python", enabled)
        except Exception as e:
            return CommandResult.ok(
                TextOutput(f"Python execution display {label}.", "success"),
                TextOutput(f"Could not save the display preference: {e}", "warning"),
            )
        return CommandResult.ok(
            TextOutput(f"Python execution display {label}.", "success"),
            TextOutput(f"Saved in {path}", "status"),
        )


class ShowDiffsCommand(Command):
    """Toggle inline rendering of semantic file-edit diffs."""

    @property
    def name(self) -> str:
        return "show-diffs"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {"/show-diffs [status|on|off]": "Configure inline file-edit diffs"}

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if len(args) > 1 or (args and args[0].lower() not in ("status", "on", "off")):
            return False, f"Unknown subcommand `{args[0]}`"
        return True, None

    async def execute(self, args: list[str]) -> "CommandResult":
        subcmd = args[0].lower() if args else "status"
        if subcmd == "status":
            state = "on" if self.config.show_diffs else "off"
            return CommandResult.ok(TextOutput(f"Inline file-edit diffs: {state}", "info"))

        enabled = subcmd == "on"
        self.config.show_diffs = enabled
        label = "enabled" if enabled else "suppressed"
        try:
            path = self._persist_tui_setting("show_diffs", enabled)
        except Exception as e:
            return CommandResult.ok(
                TextOutput(f"Inline file-edit diffs {label}.", "success"),
                TextOutput(f"Could not save the display preference: {e}", "warning"),
            )
        return CommandResult.ok(
            TextOutput(f"Inline file-edit diffs {label}.", "success"),
            TextOutput(f"Saved in {path}", "status"),
        )


def _set_agent_settings_preference(agent: "Agent | None", field: str, value: object) -> Path:
    """Persist one per-agent TUI preference in layered ``settings.yaml``."""
    from .settings import write_settings_updates

    key = getattr(agent, "_tui_memory_key", None)
    if key is None and agent is not None:
        key = f"{type(agent).__module__}:{type(agent).__qualname__}"
    path, _data = write_settings_updates({("tui", field, key or "default"): value})
    return path


_MEMORY_MODES = {"on": "project", "local": "session", "off": "off"}
_SCOPE_LABELS = {
    "project": "on (shared across sessions, project-wide)",
    "session": "local (this session only)",
    "off": "off",
}


class MemoryCommand(Command):
    """Configure long-term memory for this agent."""

    @property
    def name(self) -> str:
        return "memory"

    def help_text(self) -> dict[str, str]:  # type: ignore[override]
        return {
            "/memory [on|local|off]": (
                "Configure long-term memory: on shares a project store; "
                f"local uses this session (currently {self._scope_label()})"
            )
        }

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if len(args) > 1 or (args and args[0].lower() not in {*_MEMORY_MODES, "status"}):
            return False, "Usage: /memory [on|local|off]"
        return True, None

    def _agent_key(self) -> str:
        return getattr(
            self.agent,
            "_tui_memory_key",
            f"{type(self.agent).__module__}:{type(self.agent).__qualname__}",
        )

    def _scope(self) -> str:
        return self.config.memory_agents.get(self._agent_key(), self.config.memory)

    def _scope_label(self) -> str:
        scope = self._scope()
        return _SCOPE_LABELS.get(scope, scope)

    def _configure(self) -> None:
        from .bootstrap import configure_tui_memory

        root_config = getattr(self._registry, "_root_config", None)
        if root_config is None:
            raise RuntimeError("the TUI root configuration is unavailable")
        session_manager = self.session_manager or getattr(self._registry, "session_manager", None)
        configure_tui_memory(
            self.agent,
            root_config,
            agent_db=session_manager.agent_db_path if session_manager is not None else None,
            session_id=session_manager.session_id if session_manager is not None else None,
        )

    async def execute(self, args: list[str]) -> "CommandResult":
        if not args or args[0].lower() == "status":
            line = f"Memory: {self._scope_label()}"
            skill = getattr(self.agent, "memory", None)
            manager = getattr(skill, "_mgr", None) if skill is not None else None
            if manager is not None:
                line += f" — you are {manager.owner} · store: {manager.store.path}"
            return CommandResult.ok(TextOutput(line, "info"))

        scope = _MEMORY_MODES[args[0].lower()]
        self.config.memory = scope
        self.config.memory_agents[self._agent_key()] = scope
        _set_agent_settings_preference(self.agent, "memory_agents", scope)
        try:
            await self.agent_run_async(self._configure)
        except Exception as exc:
            return CommandResult.err(f"Failed to configure memory: {exc}")

        if scope == "off":
            return CommandResult.ok(TextOutput("Memory disabled for this agent.", "success"))
        return CommandResult.ok(
            TextOutput(f"Memory {_SCOPE_LABELS[scope]} enabled for this agent.", "success")
        )


class ReflectionCommand(MemoryCommand):
    """Configure idle consolidation for the current agent's memory."""

    @property
    def name(self) -> str:
        return "reflection"

    def help_text(self) -> dict[str, str]:  # type: ignore[override]
        state = "on" if self._enabled() else "off"
        return {
            "/reflection [on|off|now]": (
                f"Configure idle memory reflection (currently {state}); now runs immediately"
            )
        }

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if len(args) > 1 or (args and args[0].lower() not in {"on", "off", "status", "now"}):
            return False, "Usage: /reflection [on|off|now]"
        return True, None

    def _enabled(self) -> bool:
        return bool(self.config.reflection_agents.get(self._agent_key(), self.config.reflection))

    def _runner(self):
        return getattr(self.agent, "_tui_reflection_runner", None)

    def _status_output(self) -> TextOutput:
        state = "on" if self._enabled() else "off"
        runner = self._runner()
        if runner is None:
            return TextOutput(f"Idle reflection: {state} (memory is not attached)", "info")
        line = f"Idle reflection: {state} | dirty: {runner.dirty}"
        report = runner.last_report
        if report is not None:
            stopped = f"interrupted @ {report.stopped_in}, " if report.interrupted else ""
            line += (
                f" | last: merged {report.merged}, +{report.edges_added} edges, "
                f"rescored {report.rescored}, pruned {report.pruned}, "
                f"created {report.created} ({stopped}{report.duration_ms / 1000:.1f}s)"
            )
        return TextOutput(line, "info")

    async def execute(self, args: list[str]) -> "CommandResult":
        if not args or args[0].lower() == "status":
            return CommandResult.ok(self._status_output())

        if args[0].lower() == "now":
            runner = self._runner()
            if runner is None:
                return CommandResult.err("Memory is not attached. Enable it with /memory first.")
            started = await self.agent_run_async(runner.run_now)
            if not started:
                return CommandResult.ok(TextOutput("A reflection run is already pending.", "info"))
            return CommandResult.ok(
                TextOutput("Reflection started; /reflection shows the report.", "success")
            )

        enabled = args[0].lower() == "on"
        if enabled and getattr(self.agent, "memory", None) is None:
            return CommandResult.err("Memory is not attached. Enable it with /memory first.")

        self.config.reflection = enabled
        self.config.reflection_agents[self._agent_key()] = enabled
        _set_agent_settings_preference(self.agent, "reflection_agents", enabled)
        runner = self._runner()
        if runner is not None and not enabled:
            await self.agent_run_async(runner.interrupt)
        try:
            await self.agent_run_async(self._configure)
        except Exception as exc:
            return CommandResult.err(f"Failed to configure reflection: {exc}")
        state = "enabled" if enabled else "disabled"
        return CommandResult.ok(TextOutput(f"Idle reflection {state} for this agent.", "success"))


class KeepGoingCommand(Command):
    """Toggle stop auditing and autonomous continuation for unfinished work."""

    _VAR_KEY = "tui_keep_going"
    _MODEL_VAR_KEY = "tui_keep_going_model"

    @property
    def name(self) -> str:
        return "keep-going"

    def help_text(self) -> dict[str, str]:  # type: ignore[override]
        state = "on" if self._enabled() else "off"
        model = self._model() or "not configured"
        return {
            "/keep-going [on|off]": (
                f"Audit completed turns and continue unfinished work (currently {state}; "
                f"model: {model})"
            ),
            "/keep-going model <name>": f"Set the keep-going judge model (currently {model})",
        }

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if not args:
            return True, None
        subcommand = args[0].lower()
        if subcommand in {"on", "off"} and len(args) == 1:
            return True, None
        if subcommand == "model" and len(args) == 2 and args[1].strip():
            return True, None
        return False, "Usage: /keep-going [on|off] or /keep-going model <name>"

    async def execute(self, args: list[str]) -> "CommandResult":
        if not args:
            state = "on" if self._enabled() else "off"
            model = self._model() or "not configured"
            return CommandResult.ok(TextOutput(f"Keep-going mode: {state}; model: {model}", "info"))

        if args[0].lower() == "model":
            model = args[1].strip()
            self.config.keep_going_model = model
            vars_obj = getattr(self.agent, "vars", None)
            if vars_obj is not None:
                vars_obj[self._MODEL_VAR_KEY] = model
            self._persist_tui_setting("keep_going_model", model)
            return CommandResult.ok(TextOutput(f"Keep-going model set to {model}.", "success"))

        enabled = args[0].lower() == "on"
        if enabled and not self._model():
            return CommandResult.err(
                "Keep-going model is not configured. Run /keep-going model <model-id> first."
            )
        self.config.keep_going = enabled
        vars_obj = getattr(self.agent, "vars", None)
        if vars_obj is not None:
            vars_obj[self._VAR_KEY] = enabled
        self._persist_tui_setting("keep_going", enabled)
        state = "enabled" if enabled else "disabled"
        return CommandResult.ok(TextOutput(f"Keep-going mode {state}.", "success"))

    def _enabled(self) -> bool:
        vars_obj = getattr(self.agent, "vars", None)
        if vars_obj is not None and self._VAR_KEY in vars_obj:
            return bool(vars_obj.get(self._VAR_KEY))
        return bool(getattr(self.config, "keep_going", False))

    def _model(self) -> str | None:
        vars_obj = getattr(self.agent, "vars", None)
        if vars_obj is not None and self._MODEL_VAR_KEY in vars_obj:
            value = vars_obj.get(self._MODEL_VAR_KEY)
        else:
            value = getattr(self.config, "keep_going_model", None)
        if value is None:
            return None
        model = str(value).strip()
        return model or None


class ToolbarCommand(Command):
    """Configure ordered, named toolbar items."""

    @property
    def name(self) -> str:
        return "toolbar"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {"/toolbar [reset|set <items...>]": "Show or configure toolbar items"}

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        return True, None

    async def execute(self, args: list[str]) -> "CommandResult":
        from .toolbar import ToolbarRegistry

        available = ToolbarRegistry().names()
        if not args:
            active = " · ".join(self.config.toolbar_items)
            return CommandResult.ok(
                TextOutput(
                    f"Toolbar: {active}\nAvailable items: {', '.join(available)}",
                    "info",
                )
            )

        if args[0].lower() == "reset":
            self.config.toolbar_items = ["time", "model", "context", "session"]
            return CommandResult.ok(
                TextOutput("Toolbar reset to time · model · context · session.", "success")
            )

        requested = args[1:] if args[0].lower() == "set" else args
        requested = list(dict.fromkeys(item.lower() for item in requested))
        if not requested:
            return CommandResult.err("Usage: /toolbar set <item> [item ...]")
        unknown = [item for item in requested if item not in available]
        if unknown:
            return CommandResult.err(
                f"Unknown toolbar item(s): {', '.join(unknown)}. Available: {', '.join(available)}"
            )
        self.config.toolbar_items = requested
        return CommandResult.ok(TextOutput(f"Toolbar set to: {' · '.join(requested)}", "success"))


# ---------------------------------------------------------------------------
# Edit command
# ---------------------------------------------------------------------------


class EditCommand(Command):
    """Open a file in $EDITOR and show the diff on save."""

    @property
    def name(self) -> str:
        return "edit"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {
            "/edit <file>": "Open file in $EDITOR \u2014 shows diff on save",
        }

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if not args:
            return False, "Usage: /edit <file>"
        return True, None

    async def execute(self, args: list[str]) -> "CommandResult":
        import difflib
        from pathlib import Path

        path = Path(args[0]).expanduser().resolve()
        original = path.read_text(errors="replace") if path.exists() else ""
        language = _detect_language(path.suffix)

        from .frontend import ExternalEditorUnavailableError

        try:
            new_content = await self.frontend.open_editor(str(path), original, language)
        except ExternalEditorUnavailableError as exc:
            return CommandResult.err(str(exc))
        if new_content is None:
            return CommandResult.ok(TextOutput("Edit cancelled.", "info"))
        if new_content == original:
            return CommandResult.ok(TextOutput("No changes.", "info"))

        # Write the file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_content)

        # Compute unified diff
        diff_lines = list(
            difflib.unified_diff(
                original.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f"a/{path.name}",
                tofile=f"b/{path.name}",
                lineterm="\n",
            )
        )
        diff_text = "".join(diff_lines)

        outputs: list[Output] = [TextOutput(f"Saved {path}.", "success")]
        if diff_text:
            outputs.append(DiffOutput(diff=diff_text, filename=str(path)))
        return CommandResult.ok(*outputs)


class SessionCommand(Command):
    """List, resume, and manage conversation sessions."""

    @property
    def name(self) -> str:
        return "session"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {
            "/session list": "Open the session explorer",
            "/session new": "Start a new session (current history cleared)",
            "/session resume <id>": "Resume a past session (injects history as context)",
            "/session delete <id>": "Delete a session",
            "/session export": "Export current session as Markdown",
            "/session rename <name>": "Rename the current session",
        }

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if not args:
            return False, "Usage: /session <list|new|resume|delete|export|rename>"
        if args[0].lower() not in ("list", "new", "resume", "delete", "export", "rename"):
            return False, f"Unknown subcommand `{args[0]}`"
        if args[0].lower() in ("resume", "delete") and len(args) < 2:
            return False, f"Usage: /session {args[0]} <session_id>"
        if args[0].lower() == "rename" and len(args) < 2:
            return False, "Usage: /session rename <name>"
        return True, None

    async def execute(self, args: list[str]) -> "CommandResult":
        subcmd = args[0].lower()

        if subcmd == "list":
            open_explorer = getattr(self.frontend, "open_session_explorer", None)
            has_real_explorer = getattr(
                type(self.frontend), "open_session_explorer", None
            ) is not None or "open_session_explorer" in getattr(self.frontend, "__dict__", {})
            if has_real_explorer and callable(open_explorer):
                try:
                    await open_explorer()
                except Exception as exc:
                    return CommandResult.err(f"Session explorer failed: {exc}")
                return CommandResult.ok(TextOutput("Session explorer closed.", "status"))

            sessions = [s for s in SessionManager.list_sessions() if s.turn_count > 0]
            if not sessions:
                return CommandResult.ok(TextOutput("No sessions found.", "info"))
            rows = []
            for s in sessions:
                dt = datetime.datetime.fromtimestamp(s.last_active).strftime("%m/%d %H:%M")
                name_display = s.name[:28] if s.name else ""
                rows.append(
                    [s.id[:8], dt, s.model.split("/")[-1][:20], str(s.turn_count), name_display]
                )
            return CommandResult.ok(
                TableOutput(
                    title="Recent Sessions",
                    columns=["ID", "Last Active", "Model", "Turns", "Name"],
                    rows=rows,
                )
            )

        if subcmd == "export":
            if self.session_manager is None:
                return CommandResult.err("No active session manager.")
            md = self.session_manager.as_markdown()
            fname = f"session-{self.session_manager.session_id[:8]}-{datetime.date.today()}.md"
            try:
                Path(fname).write_text(md)
                return CommandResult.ok(TextOutput(f"Session exported to {fname}", "success"))
            except Exception as e:
                return CommandResult.ok(TextOutput(f"Export failed: {e}\n\n{md[:500]}", "warning"))

        if subcmd == "resume":
            session_id = args[1]
            matches = SessionManager.find_by_prefix(session_id)
            if not matches:
                return CommandResult.err(f"Session '{session_id}' not found. Use /session list.")
            if len(matches) > 1:
                ids = ", ".join(m[:8] for m in matches)
                return CommandResult.err(f"Ambiguous session prefix '{session_id}' matches: {ids}")
            full_id = matches[0]

            import os as _os

            from .session_manager import SESSIONS_DIR as _SESSIONS_DIR

            _session_db_path = _SESSIONS_DIR / f"{full_id}.db"
            _in_nemo_term = bool(_os.environ.get("NEMO_OO_RICH_URL"))

            from nooa.storage.sqlite import SessionAlreadyActiveError

            try:
                new_sm = SessionManager.open(full_id)
            except SessionAlreadyActiveError as e:
                return CommandResult.err(str(e))

            try:
                outputs = build_resume_outputs(
                    _session_db_path, full_id, in_nemo_term=_in_nemo_term
                )
                if not outputs:
                    new_sm.close()
                    return CommandResult.err(f"Session '{session_id}' is empty.")

                # Session._run_command executes this after acknowledged turn
                # cancellation and after the storage/session swap, on the agent
                # loop when available. restore_latest_snapshot mutates agent
                # vars/todos/context/tool state and must not race an active turn.
                async def _restore_and_emit() -> list[Output]:
                    restored = new_sm._storage.restore_latest_snapshot(self.agent)
                    try:
                        from nooa.sessions import SessionResumed

                        self.agent.event_manager.register_event_type(SessionResumed)
                        self.agent.event_manager.add(
                            SessionResumed(session_id=full_id, restored=restored)
                        )
                    except Exception:
                        logger.debug("Failed to emit SessionResumed", exc_info=True)
                    if restored:
                        return [
                            TextOutput(
                                f"Agent state restored from session {full_id[:8]}.", "status"
                            )
                        ]
                    return []

                result = CommandResult.ok(*outputs)
                result.new_session_manager = new_sm
                result.post_session_swap = _restore_and_emit
                return result
            except Exception as e:
                new_sm.close()
                return CommandResult.err(f"Could not restore session: {e}")

        if subcmd == "delete":
            session_id = args[1]
            matches = SessionManager.find_by_prefix(session_id)
            if not matches:
                return CommandResult.err(f"Session '{session_id}' not found.")
            if len(matches) > 1:
                ids = ", ".join(m[:8] for m in matches)
                return CommandResult.err(f"Ambiguous session prefix '{session_id}' matches: {ids}")
            SessionManager.delete_session(matches[0])
            return CommandResult.ok(TextOutput(f"Session {matches[0][:8]} deleted.", "success"))

        if subcmd == "rename":
            if self.session_manager is None:
                return CommandResult.err("No active session.")

            name = " ".join(args[1:]).strip()
            self.session_manager.rename(name, user_named=True)
            return CommandResult.ok(TextOutput(f"Session renamed to: {name}", "success"))

        if subcmd == "new":
            # NOTE: do NOT call agent.event_manager.clear() here — same
            # reasoning as ClearCommand: it would wipe the old session's
            # SQLite data before _swap_session_manager() preserves it.

            # Create a fresh SessionManager so subsequent turns go to a new file.
            new_sm: SessionManager | None = None
            if self.session_manager is not None:
                try:
                    new_sm = SessionManager.create(
                        model=self.session_manager.model,
                        agent_cls=self.session_manager.agent_cls,
                        working_dir=self.session_manager.working_dir,
                    )
                except Exception:
                    pass

            # Session._run_command performs acknowledged turn cancellation and
            # loop-affine QueueManager shutdown/flush during the actual session
            # swap. Do not call queue_manager.shutdown() here: spawned jobs live on
            # the agent loop, while commands run on the UI loop.

            # Session._run_command applies the agent working-state reset after
            # acknowledged turn cancellation and after the storage/session swap,
            # on the agent loop when available.

            outputs: list[Output] = [
                ClearScreen(),
                _RichReplayPayload(payload={"kind": "clear"}),  # type: ignore[list-item]
            ]
            if self._registry and self._registry.startup_info:
                outputs.append(self._registry.startup_info)
            outputs.append(TextOutput("Started new session. History cleared.", "success"))
            result = CommandResult(success=True, outputs=outputs)
            result.new_session_manager = new_sm
            result.post_session_swap = lambda: _reset_agent_working_state(self.agent)
            return result

        return CommandResult.err(f"Unknown subcommand `{subcmd}`")


# ---------------------------------------------------------------------------
# Registry and handler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _UserSkill:
    """Metadata for a user-invocable skill slash command."""

    name: str
    body: str
    description: str
    argument_hint: str | None = None
    completions: tuple[str, ...] = ()
    output_to_agent: bool = True
    _method: Any = field(default=None, repr=False)

    def help_entry(self) -> tuple[str, str]:
        hint = self.argument_hint or ""
        key = f"/{self.name} {hint}".strip()
        return key, self.description

    def make_agent_message(self, args: list[str]) -> str:
        body = self.body
        if args:
            joined = " ".join(args)
            if "$ARGUMENTS" in body:
                return body.replace("$ARGUMENTS", joined)
            return f"{body}\n\nArguments: {joined}"
        return body


# ---------------------------------------------------------------------------
# Jobs command
# ---------------------------------------------------------------------------


class JobsCommand(Command):
    """Open the job explorer."""

    @property
    def name(self) -> str:
        return "jobs"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {"/jobs": "Open the job explorer"}

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if args:
            return False, "Usage: /jobs"
        return True, None

    async def execute(self, args: list[str]) -> "CommandResult":
        open_explorer = getattr(self.frontend, "open_job_explorer", None)
        if callable(open_explorer):
            try:
                await open_explorer()
            except Exception as exc:
                return CommandResult.err(f"Job explorer failed: {exc}")
            return CommandResult.ok(TextOutput("Job explorer closed.", "status"))
        return CommandResult.err("The job explorer requires the terminal TUI.")


# ---------------------------------------------------------------------------
# Memories command
# ---------------------------------------------------------------------------


class MemoriesCommand(Command):
    """Open the in-app long-term memory explorer."""

    @property
    def name(self) -> str:
        return "memories"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {"/memories": "Open the long-term memory explorer"}

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if args:
            return False, "Usage: /memories"
        return True, None

    async def execute(self, args: list[str]) -> "CommandResult":
        if getattr(self.agent, "memory", None) is None:
            return CommandResult.err("Memory is not attached. Enable it with /memory first.")
        open_explorer = getattr(self.frontend, "open_memory_explorer", None)
        if not callable(open_explorer):
            return CommandResult.err("The memory explorer requires the terminal TUI.")
        try:
            await open_explorer()
        except Exception as exc:
            return CommandResult.err(f"Memory explorer failed: {exc}")
        return CommandResult.ok(TextOutput("Memory explorer closed.", "status"))


class TraceUrlCommand(Command):
    """Print the full viewer URL for the current session's trace."""

    @property
    def name(self) -> str:
        return "trace-url"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {
            "/trace-url": "Print the full URL to the trace of the current session",
        }

    async def execute(self, args: list[str]) -> "CommandResult":
        try:
            from nooa.tracing import get_session
        except ImportError:
            return CommandResult.err("Tracing package not installed.")

        session_name = get_session()
        if not session_name:
            return CommandResult.err("No active trace session.")

        # Determine the viewer base URL from the OTLP endpoint
        endpoint = os.environ.get("OTLP_ENDPOINT", "http://localhost:5001/v1/traces")
        # Strip /v1/traces or /v1 suffix to get the viewer base
        base = endpoint.rstrip("/")
        for suffix in ("/v1/traces", "/v1"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break

        url = f"{base}/traces/view?session_id={urllib.parse.quote(session_name)}"
        return CommandResult.ok(TextOutput(url, "info"))


# ---------------------------------------------------------------------------
# Show last Python command
# ---------------------------------------------------------------------------


class EventsCommand(Command):
    """Open the in-app event explorer."""

    @property
    def name(self) -> str:
        return "events"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {"/events": "Open the event explorer"}

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if args:
            return False, "Usage: /events"
        return True, None

    async def execute(self, args: list[str]) -> "CommandResult":
        if args:
            return CommandResult.err("Usage: /events")
        em = getattr(self.agent, "event_manager", None)
        if em is None:
            return CommandResult.err("No event manager available.")

        return await self._open_explorer(em)

    async def _open_explorer(self, em) -> "CommandResult":
        """Open the event explorer as an in-app TUI view."""
        open_explorer = getattr(self.frontend, "open_event_explorer", None)
        if not callable(open_explorer):
            return CommandResult.err("The event explorer requires the terminal TUI.")
        try:
            await open_explorer(em)
        except Exception as exc:
            return CommandResult.err(f"Event explorer failed: {exc}")
        return CommandResult.ok(TextOutput("Event explorer closed.", "status"))


class ActivityCommand(Command):
    """Report what the agent is doing right now: executing Python, waiting on
    an LLM call, or idle."""

    @property
    def name(self) -> str:
        return "activity"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {"/activity": "Show what the agent is doing right now (python / LLM / idle)"}

    async def execute(self, args: list[str]) -> "CommandResult":
        result = await self._activity_result()
        if not result.success:
            return result
        open_overlay = getattr(self.frontend, "open_activity_overlay", None)
        has_real_overlay = getattr(
            type(self.frontend), "open_activity_overlay", None
        ) is not None or "open_activity_overlay" in getattr(self.frontend, "__dict__", {})
        if has_real_overlay and callable(open_overlay):
            try:
                await open_overlay(result.outputs)
            except Exception as exc:
                return CommandResult.err(f"Activity overlay failed: {exc}")
            return CommandResult.ok(TextOutput("Activity overlay closed.", "status"))
        return result

    async def _activity_result(self) -> "CommandResult":
        try:
            from nooa.runtime.debug_handler import get_activity
        except ImportError:
            return CommandResult.err("Activity tracking not available in this build.")

        activity = get_activity()
        phase = activity["phase"]
        code_execs = activity["code_execs"]
        llm_calls = activity["llm_calls"]

        labels = {
            "executing_python": "Executing Python",
            "waiting_llm": "Waiting on LLM call",
            # "idle" only reaches the table in the rare idle-with-live-turn case;
            # idle-with-nothing-in-flight early-returns a one-liner below.
            "idle": "Idle",
        }
        headline = labels.get(phase, phase)

        # Where in the code is the agent suspended? Walk the agent turn task's
        # await stack. This runs on the agent loop (asyncio.Task is not safe to
        # introspect cross-thread), so dispatch via agent_run_async().
        location = await self._locate_agent()
        cell = location.innermost_cell if location is not None else None
        highlight = location.highlight if location is not None else None

        # Idle with nothing in flight: a one-liner, not a table.
        if phase == "idle" and (location is None or not location.stack):
            return CommandResult.ok(TextOutput("Agent idle — nothing in flight.", "status"))

        rows: list[list[str]] = [["Phase", headline]]
        for ex in code_execs:
            detail = ex.get("preview") or "(code cell)"
            rows.append([f"  python ({ex['elapsed']:.1f}s)", detail])
        for call in llm_calls:
            model = call.get("model", "unknown")
            rows.append([f"  llm ({call['elapsed']:.1f}s)", model])

        if location is not None and location.stack:
            rows.append(["", ""])
            rows.append(["Suspended at", location.task_name or "agent turn"])
            for frame in location.stack:
                if frame is highlight:
                    marker = "→ "  # the frame /activity points its source block at
                elif frame is location.innermost:
                    marker = "↳ "  # deepest plumbing the highlighted frame is blocked in
                else:
                    marker = "  "
                rows.append(["", f"{marker}{frame.short()}"])

        if phase == "executing_python" and llm_calls:
            footer = "Running a code cell that is itself blocked on an LLM call."
        elif phase == "executing_python" and cell is not None:
            footer = f"In a code cell — parked at {cell.short()}."
        elif phase == "executing_python":
            footer = "In a code cell — not waiting on the model."
        elif phase == "waiting_llm":
            footer = "Blocked waiting for the model to respond."
        elif location is not None and location.innermost is not None:
            footer = f"Awaiting at {location.innermost.short()}."
        else:
            footer = "Nothing in flight."

        outputs: list[Output] = [
            # Unlabelled key/value table: no title, no header row (Rich would
            # otherwise draw an empty header band above the first row).
            TableOutput(
                columns=["", ""],
                rows=rows,
                footer=footer,
                show_header=False,
            )
        ]

        # Render the suspend-point source as a syntax-highlighted code block
        # (the frontend lexes/colours CodeExecution.code). For a CodeAct/REPL
        # cell this is the cell text; the suspend line is the last context line.
        code_output = self._suspend_code_output(location)
        if code_output is not None:
            outputs.append(code_output)

        return CommandResult.ok(*outputs)

    @staticmethod
    def _suspend_code_output(location) -> "CodeExecution | None":
        """Build a highlighted code block for the suspend-point source, or None.

        The snippet is numbered from its real file offset (``start_line``) and
        the suspend line is tinted via ``highlight_line``, so the parked line
        stands out and the line numbers match the await stack.
        """
        if location is None:
            return None
        frame = location.highlight
        if frame is None or not frame.context:
            return None
        code = "\n".join(src.text for src in frame.context)
        return CodeExecution(
            tool_call_id=f"activity:{frame.filename}:{frame.lineno}",
            code=code,
            start_line=frame.context[0].lineno,
            highlight_line=frame.lineno,
        )

    async def _locate_agent(self):
        """Snapshot where the agent coroutine is suspended (or None if idle).

        Runs the probe on the agent loop via ``agent_run_async`` —
        ``asyncio.Task`` is not safe to walk from the command (UI) thread, and
        the blocking ``agent_run`` would freeze the UI loop while the agent is
        busy (causing status-bar flicker and stalling ``self.message()``
        output). Awaiting yields control so the UI keeps painting.
        """
        from .agent_location import locate_agent_on_loop

        try:
            return await self.agent_run_async(locate_agent_on_loop)
        except Exception:
            return None


class TodosCommand(Command):
    """Open the todo explorer."""

    @property
    def name(self) -> str:
        return "todos"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {"/todos": "Open the todo explorer"}

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if args:
            return False, "Usage: /todos"
        return True, None

    async def execute(self, args: list[str]) -> "CommandResult":
        open_explorer = getattr(self.frontend, "open_todo_explorer", None)
        if callable(open_explorer):
            try:
                await open_explorer()
            except Exception as exc:
                return CommandResult.err(f"Todo explorer failed: {exc}")
            return CommandResult.ok(TextOutput("Todo explorer closed.", "status"))
        return CommandResult.err("The todo explorer requires the terminal TUI.")


class MCPCommand(Command):
    """User-owned MCP configuration, approval, and connection lifecycle."""

    required_capabilities: ClassVar[frozenset[str]] = frozenset({"mcp"})

    @property
    def name(self) -> str:
        return "mcp"

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {
            "/mcp <list|add|remove|connect|approve|revoke|disconnect>": (
                "Configure, approve, and connect MCP servers"
            ),
        }

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        if not args:
            return True, None
        action = args[0].lower()
        allowed = {"list", "add", "remove", "connect", "approve", "revoke", "disconnect"}
        if action not in allowed:
            return False, f"Unknown MCP action {action!r}"
        if action == "list":
            return (len(args) == 1, "Usage: /mcp list")
        if action == "add":
            return (len(args) == 3, "Usage: /mcp add <server> <url-or-command>")
        if len(args) < 2:
            return False, f"Usage: /mcp {action} <server>"
        if action == "approve" and len(args) > 3:
            return False, "Usage: /mcp approve <server> [confirmation-code]"
        if action != "approve" and len(args) > 2:
            return False, f"Usage: /mcp {action} <server>"
        return True, None

    async def execute(self, args: list[str]) -> "CommandResult":
        from .mcp_approval import MCPApprovalRequired

        mcp = self.agent.mcp
        refresh = getattr(mcp, "refresh_settings", None)
        if callable(refresh):
            try:
                refresh()
            except Exception as exc:
                return CommandResult.err(f"Failed to refresh MCP settings: {exc}")
        action = args[0].lower() if args else "list"
        if action == "list":
            return CommandResult.ok(TextOutput(mcp.status(), "info"))

        server = args[1]
        configured = set(mcp.discovered())

        if action == "add":
            if server in configured:
                return CommandResult.err(
                    f"Server {server!r} is already configured; edit its source configuration."
                )
            value = args[2]
            entry = (
                {"url": value, "transport": "streamable-http"}
                if value.startswith(("http://", "https://"))
                else {"command": value}
            )
            path = None
            try:
                mcp._validate_attach_name(server)
                from .settings import write_settings_updates

                path, _data = write_settings_updates({("tui", "mcp_servers", server): entry})
                mcp.register(server, **entry)
            except Exception as exc:
                if path is not None:
                    from .settings import delete_settings_value

                    delete_settings_value(("tui", "mcp_servers", server))
                return CommandResult.err(f"Failed to add {server!r}: {exc}")
            kind = "HTTP" if "url" in entry else "stdio"
            return CommandResult.ok(
                TextOutput(
                    f"Added {kind} MCP server {server!r} to {path}. "
                    f"Review it with /mcp approve {server}.",
                    "success",
                )
            )

        if action == "remove":
            if server not in mcp._servers:
                if server in configured:
                    return CommandResult.err(
                        f"Server {server!r} comes from {mcp._mcp_file}; remove it from that file."
                    )
                return CommandResult.err(f"Server {server!r} is not configured. Try /mcp list.")
            if server in mcp.connected():
                await mcp.disconnect([server])
            from .settings import delete_settings_value

            path, _data, deleted = delete_settings_value(("tui", "mcp_servers", server))
            if not deleted:
                return CommandResult.err(
                    f"Server {server!r} is not present in project settings at {path}."
                )
            mcp._servers.pop(server, None)
            mcp._activated.discard(server)
            mcp._revoke_approvals(server)
            return CommandResult.ok(
                TextOutput(f"Removed MCP server {server!r} from {path}.", "success")
            )

        if action != "revoke" and server not in configured:
            return CommandResult.err(f"Server {server!r} not found. Try /mcp list.")

        if action == "connect":
            try:
                connected = await mcp.connect([server])
            except MCPApprovalRequired as exc:
                result = CommandResult.ok(TextOutput(str(exc), "warning"))
                result.input_prefill = exc.request.approval_command
                return result
            except Exception as exc:
                return CommandResult.err(f"Failed to connect {server!r}: {exc}")
            if not connected:
                return CommandResult.ok(
                    TextOutput(f"{server!r} is already connected.\n\n{mcp.status()}", "info")
                )
            return CommandResult.ok(
                TextOutput(f"Connected {server!r}.\n\n{mcp.status()}", "success")
            )

        if action == "approve":
            if len(args) == 2:
                try:
                    request = mcp._approval_request(server)
                    review = request.review_text()
                except Exception as exc:
                    return CommandResult.err(f"Cannot review {server!r}: {exc}")
                result = CommandResult.ok(TextOutput(review, "warning"))
                result.input_prefill = request.approval_command
                return result
            try:
                mcp._approve(server, args[2])
            except Exception as exc:
                return CommandResult.err(f"Failed to approve {server!r}: {exc}")
            try:
                connected = await mcp.connect([server])
            except Exception as exc:
                return CommandResult.err(f"Approved {server!r}, but connection failed: {exc}")
            state = "Approved and connected" if connected else "Approved"
            return CommandResult.ok(TextOutput(f"{state} {server!r}.\n\n{mcp.status()}", "success"))

        if action == "revoke":
            if server in mcp.connected():
                await mcp.disconnect([server])
            revoked = mcp._revoke_approvals(server)
            if revoked:
                return CommandResult.ok(
                    TextOutput(f"Disconnected {server!r} and revoked its approvals.", "success")
                )
            return CommandResult.ok(TextOutput(f"{server!r} had no stored approvals.", "info"))

        if server not in mcp.connected():
            return CommandResult.ok(
                TextOutput(f"{server!r} is not connected. Try /mcp list.", "info")
            )
        try:
            await mcp.disconnect([server])
        except Exception as exc:
            return CommandResult.err(f"Failed to disconnect {server!r}: {exc}")
        return CommandResult.ok(
            TextOutput(f"Disconnected {server!r}.\n\n{mcp.status()}", "success")
        )


class CommandRegistry:
    """Registry of command instances."""

    _command_classes: dict[str, type[Command]] = {
        "help": HelpCommand,
        "exit": ExitCommand,
        "quit": ExitCommand,
        "clear": ClearCommand,
        "compact": CompactCommand,
        "context": ContextCommand,
        "edit": EditCommand,
        "connect": ConnectCommand,
        "model": ModelCommand,
        "models": ModelsCommand,
        "theme": ThemeCommand,
        "skills": SkillsCommand,
        "show-python": ShowPythonCommand,
        "show-diffs": ShowDiffsCommand,
        "keep-going": KeepGoingCommand,
        "memory": MemoryCommand,
        "memories": MemoriesCommand,
        "reflection": ReflectionCommand,
        "session": SessionCommand,
        "jobs": JobsCommand,
        "events": EventsCommand,
        "todos": TodosCommand,
        "mcp": MCPCommand,
        "trace-url": TraceUrlCommand,
        "toolbar": ToolbarCommand,
        "activity": ActivityCommand,
        "reasoning": ReasoningCommand,
    }

    def __init__(
        self,
        config: "TUIConfig",
        agent: "Agent",
        frontend: "Frontend",
        skills_dirs: list[Path] | None = None,
        mcp_file: Path | None = None,
        session_manager: "SessionManager | None" = None,
        root_config: "Any | None" = None,
    ):
        self.config = config
        self.agent = agent
        self.frontend = frontend
        self.skills_dirs = skills_dirs
        self.mcp_file = mcp_file
        self.session_manager = session_manager
        self._root_config = root_config
        self.startup_info: Output | None = None  # set by main after bootstrap
        self.blocking_llm_health: Any | None = None
        self._bind_mcp_oauth_prompt()
        self._commands: dict[str, Command] = self._register()
        self._discover_directory_skills()
        self._user_skills: dict[str, _UserSkill] = self._discover_user_skills()

    def _register(self) -> dict[str, Command]:
        commands: dict[str, Command] = {}
        kwargs: dict[str, Any] = {
            "skills_dirs": self.skills_dirs,
            "registry": self,
            "session_manager": self.session_manager,
            "root_config": self._root_config,
        }
        for name, cls in self.get_all_command_classes().items():
            if not all(hasattr(self.agent, cap) for cap in cls.required_capabilities):
                continue
            commands[name] = cls(self.frontend, self.config, self.agent, **kwargs)
        return commands

    def _bind_mcp_oauth_prompt(self) -> None:
        """Bridge worker-thread OAuth input onto the host UI event loop."""
        mcp = getattr(self.agent, "mcp", None)
        bind = getattr(mcp, "_bind_oauth_code_prompt", None)
        if not callable(bind):
            return
        try:
            ui_loop = asyncio.get_running_loop()
        except RuntimeError:
            bind(None)
            return

        async def _ask_on_ui(auth_url: str) -> str:
            prompt_sensitive = getattr(self.frontend, "prompt_sensitive", None)
            if not callable(prompt_sensitive):
                raise RuntimeError("This frontend cannot collect MCP OAuth codes securely")

            # Put a full, clickable Markdown link in terminal scrollback before
            # opening the bounded modal. This survives tmux redraws and remains
            # selectable when OSC 52 clipboard forwarding is unavailable.
            markdown_link = _mcp_oauth_markdown_link(auth_url)
            if markdown_link is not None:
                await self.frontend.render(
                    AgentMessage(
                        "Open the MCP OAuth authorization URL in a browser:\n\n"
                        f"{markdown_link}\n\n"
                        "If the link is not clickable (for example through tmux), "
                        "copy and paste the displayed URL.",
                        show_rule=False,
                        soft_wrap=True,
                    )
                )

            return await prompt_sensitive(
                "MCP OAuth authorization",
                "Open the authorization URL shown in scrollback, authorize the server, "
                "then paste the authorization code or callback URL.",
                link_url=auth_url,
            )

        async def _prompt(auth_url: str) -> str:
            try:
                current_loop = asyncio.get_running_loop()
            except RuntimeError:
                current_loop = None
            if current_loop is ui_loop:
                return await _ask_on_ui(auth_url)
            future = asyncio.run_coroutine_threadsafe(_ask_on_ui(auth_url), ui_loop)
            return await asyncio.wrap_future(future)

        bind(_prompt)

    async def auto_connect_mcp(self) -> None:
        """Connect approved startup servers after the live TUI is ready."""
        await asyncio.sleep(0)
        names = _mcp_auto_connect_names(getattr(self.config, "mcp_auto_connect", None))
        if not names:
            return
        registry = getattr(self.agent, "mcp", None)
        if registry is None:
            logger.warning("MCP auto-connect requested but self.mcp is not available")
            return

        try:
            configured = set(registry.discovered())
        except Exception as exc:
            logger.warning("MCP auto-connect skipped: failed to read config: %s", exc)
            return

        unknown = [n for n in names if n not in configured]
        for name in unknown:
            logger.warning("MCP auto-connect skipped unknown server %r", name)
        wanted = [n for n in names if n in configured]
        if not wanted:
            return

        from .mcp_approval import MCPApprovalRequired

        for server_name in wanted:
            try:
                await registry.connect([server_name])
                logger.info("MCP server %r auto-connected", server_name)
            except MCPApprovalRequired as exc:
                logger.warning("MCP auto-connect needs approval for %r", server_name)
                await self.frontend.render(TextOutput(str(exc), "warning"))
            except Exception as exc:
                logger.warning("Failed to auto-connect MCP server %r: %s", server_name, exc)

    def _discover_user_skills(self) -> "dict[str, _UserSkill]":
        """Scan skills dirs for install-as:command skills and register them as slash commands.

        Uses rglob to match SkillRegistry.discover_skills_dirs() — finds skills at any depth.
        Parses SKILL.md frontmatter inline to avoid depending on private nooa
        internals that may not be present in older installed versions.
        """
        skills: dict[str, _UserSkill] = {}
        if not self.skills_dirs:
            return skills
        try:
            import yaml
        except ImportError:
            return skills
        for skills_dir in self.skills_dirs:
            skills_dir = Path(skills_dir)
            if not skills_dir.is_dir():
                continue
            for skill_md in sorted(skills_dir.rglob("SKILL.md")):
                entry = skill_md.parent
                try:
                    content = skill_md.read_text(encoding="utf-8")
                    if not content.startswith("---"):
                        continue
                    parts = content.split("---", 2)
                    if len(parts) < 3:
                        continue
                    try:
                        meta = yaml.safe_load(parts[1]) or {}
                        if not isinstance(meta, dict):
                            raise ValueError("not a mapping")
                    except Exception:
                        # Fallback: line-by-line regex for invalid-YAML values like
                        # argument-hint: "<action>" [issue-id]  (Claude Code style).
                        # Parse each scalar individually so "false" → False (not "false").
                        import re

                        meta = {}
                        for line in parts[1].splitlines():
                            m = re.match(r"^([a-zA-Z][a-zA-Z0-9_-]*):\s*(.+)$", line)
                            if not m:
                                continue
                            raw = m.group(2).strip()
                            try:
                                parsed = yaml.safe_load(raw)
                                meta[m.group(1)] = (
                                    str(parsed) if isinstance(parsed, list) else parsed
                                )
                            except Exception:
                                meta[m.group(1)] = raw
                    if not isinstance(meta, dict):
                        continue
                    # CC convention: user-invocable defaults to true.
                    # Opt out with user-invocable: false.
                    # install-as: command is honored for backward compat.
                    if meta.get("user-invocable") is False:
                        continue
                    raw_name = str(meta.get("name") or "").strip()
                    cmd_name = raw_name.lower()
                    if not cmd_name or cmd_name in self._commands or cmd_name in skills:
                        continue
                    description = str(meta.get("description", "")).strip()
                    body = parts[2].strip()
                    hint = meta.get("argument-hint")
                    if isinstance(hint, list):
                        # YAML parses [label] as a list; reconstruct bracket notation
                        hint = "[" + ", ".join(str(x) for x in hint) + "]"
                    elif hint is not None:
                        hint = str(hint)
                    skills[cmd_name] = _UserSkill(
                        name=cmd_name,
                        body=body,
                        description=description,
                        argument_hint=hint,
                    )
                except Exception as e:
                    logger.warning("Failed to load skill from %s: %s", entry, e)
        # Also discover @slash_command methods from loaded Skills
        skills.update(self._discover_skill_commands())
        return skills

    def _discover_skill_commands(self) -> "dict[str, _UserSkill]":
        """Discover @slash_command methods from loaded Skill instances on the agent."""
        skills: dict[str, _UserSkill] = {}
        try:
            from nooa.skill import get_slash_commands
        except ImportError:
            return skills

        from nooa.skill import Skill

        for attr_name in dir(self.agent):
            if attr_name.startswith("_"):
                continue
            try:
                obj = getattr(self.agent, attr_name)
            except Exception:
                continue
            if not isinstance(obj, Skill):
                continue
            for meta, method in get_slash_commands(obj):
                cmd_name = meta.name.lower()
                if cmd_name in self._commands or cmd_name in skills:
                    continue
                description = (method.__doc__ or "").strip().split("\n")[0]
                skills[cmd_name] = _UserSkill(
                    name=cmd_name,
                    body="",
                    description=description,
                    argument_hint=meta.argument_hint,
                    completions=getattr(meta, "completions", ()),
                    output_to_agent=getattr(meta, "output_to_agent", True),
                    _method=method,
                )
        return skills

    def _discover_directory_skills(self) -> None:
        """Load directory skills without making them model-visible by default."""
        if not self.skills_dirs:
            return
        try:
            from nooa.skill_registry import SkillRegistry

            registry = getattr(self.agent, "skills", None)
            if isinstance(registry, SkillRegistry):
                registry.discover_skills_dirs(self.skills_dirs)
        except ImportError:
            pass
        except Exception as e:
            logger.warning("Failed to auto-install skills: %s", e)

    def add_skills_dir(self, path: Path) -> bool:
        """Add one live skill root and refresh model and slash-command discovery."""
        path = Path(path).expanduser().resolve()
        existing = {Path(item).expanduser().resolve() for item in (self.skills_dirs or [])}
        added = path not in existing
        if added:
            if self.skills_dirs is None:
                self.skills_dirs = []
            self.skills_dirs.append(path)
            if path not in self.config.skills_dirs:
                self.config.skills_dirs.append(path)

        try:
            from nooa.skill_registry import SkillRegistry

            registry = getattr(self.agent, "skills", None)
            if isinstance(registry, SkillRegistry):
                registry.discover_skills_dirs([path])
        except ImportError:
            pass

        self._user_skills = self._discover_user_skills()
        return added

    def get_command(self, name: str) -> "Command | None":
        return self._commands.get(name.lower())

    def commands(self) -> "list[Command]":
        """Return the list of registered ``Command`` instances.

        Public surface so callers (e.g. ``Session._swap_session_manager``)
        don't have to reach into ``_commands`` directly.
        """
        return list(self._commands.values())

    def get_user_skill(self, name: str) -> "_UserSkill | None":
        return self._user_skills.get(name.lower())

    def refresh_skill_commands(self) -> None:
        """Re-discover @slash_command methods from agent skills (hot-reload support).

        Called by LibraryManager after reloading a library so newly-added
        slash commands become available and removed ones are deregistered
        without TUI restart.
        """
        fresh = self._discover_skill_commands()
        # Remove stale @slash_command entries (those with _method set);
        # preserve text-skill entries (SKILL.md, _method is None).
        self._user_skills = {k: v for k, v in self._user_skills.items() if v._method is None}
        self._user_skills.update(fresh)

    @classmethod
    def get_all_command_classes(cls) -> dict[str, type[Command]]:
        return cls._command_classes.copy()

    @classmethod
    def get_help(cls) -> dict[str, str]:
        commands: dict[str, str] = {}
        seen: set[type[Command]] = set()
        for cmd_cls in cls._command_classes.values():
            if cmd_cls not in seen:
                seen.add(cmd_cls)
                try:
                    commands.update(cmd_cls.help_text())
                except TypeError:
                    # Instance-method help_text() overrides can't be called on the class
                    pass
        return commands

    def get_active_help(self) -> dict[str, str]:
        commands = self.get_builtin_help()
        for skill in self._user_skills.values():
            key, desc = skill.help_entry()
            commands[key] = desc
        return commands

    def get_builtin_help(self) -> dict[str, str]:
        """Return active built-ins without environment-supplied skill commands."""
        commands: dict[str, str] = {}
        seen: set[type[Command]] = set()
        for cmd in self._commands.values():
            cls = type(cmd)
            if cls not in seen:
                seen.add(cls)
                commands.update(cmd.help_text())
        return commands

    def get_completions(self) -> dict[str, str]:
        help_text = self.get_active_help()
        completions: dict[str, str] = {}
        for cmd, desc in help_text.items():
            # Strip both <action> and [label] style argument hints
            clean = re.split(r"\s+(?=[<\[])", cmd, maxsplit=1)[0].strip()
            if clean and clean not in completions:
                completions[clean] = desc
        return dict(sorted(completions.items()))


class CommandHandler:
    """Parses slash-command input and dispatches to registered commands."""

    def __init__(
        self,
        registry: "CommandRegistry",
        frontend: "Frontend",
        agent_run_async: Any | None = None,
    ) -> None:
        self.registry = registry
        self.frontend = frontend
        self._agent_run_async = agent_run_async

    def _expand_agent_mentions(self, text: str) -> str:
        """Expand @paths in skill output immediately before it becomes an agent turn."""
        from .completer import expand_mentions

        agent = getattr(self.registry, "agent", None)
        return expand_mentions(text, base_dir=getattr(agent, "cwd", None))

    async def handle(self, input_text: str, *, render_outputs: bool = True) -> "CommandResult":
        if not input_text.startswith("/"):
            return CommandResult(False)

        try:
            parts = shlex.split(input_text[1:])
        except ValueError:
            parts = input_text[1:].split()

        if not parts:
            result = CommandResult.err("Empty command. Type /help for available commands.")
            if render_outputs:
                for output in result.outputs:
                    await self.frontend.render(output)
            return result

        cmd_name = parts[0].lower()
        args = parts[1:]

        # Check user-invocable skills before falling through to unknown-command error
        skill = self.registry.get_user_skill(cmd_name)
        if skill is not None:
            if skill._method is not None:
                import inspect

                from nooa.slash_dispatch import (
                    CoercionError,
                    SlashCommandResult,
                    parse_typed_args,
                )

                raw_args = " ".join(args)
                try:
                    kwargs = parse_typed_args(skill._method, raw_args)
                except CoercionError as e:
                    msg = f"/{cmd_name}: {e.message}"
                    if e.hint:
                        msg += f"\nUsage: /{cmd_name} {e.hint}"
                    result = CommandResult.err(msg)
                    if render_outputs:
                        for output in result.outputs:
                            await self.frontend.render(output)
                    return result

                def _call_skill_method():
                    return skill._method(**kwargs)

                if self._agent_run_async is not None:
                    result_val = await self._agent_run_async(_call_skill_method)
                else:
                    result_val = _call_skill_method()
                    if inspect.isawaitable(result_val):
                        result_val = await result_val

                result_text = str(result_val) if result_val is not None else None
                if result_text is not None and skill.output_to_agent:
                    result_text = self._expand_agent_mentions(result_text)
                slash_result = SlashCommandResult(
                    command=cmd_name,
                    args=raw_args,
                    value=result_val,
                    text=result_text,
                    output_to_agent=skill.output_to_agent,
                )
                return CommandResult(success=True, slash_result=slash_result)
            agent_message = self._expand_agent_mentions(skill.make_agent_message(args))
            return CommandResult(success=True, agent_message=agent_message)

        command = self.registry.get_command(cmd_name)
        if not command:
            all_classes = self.registry.get_all_command_classes()
            if cmd_name in all_classes:
                msg = f"/{cmd_name} is not available with this agent."
            else:
                suggestions = [c for c in all_classes if c.startswith(cmd_name[:2])][:3]
                suffix = (
                    f" Did you mean: {', '.join(f'/{s}' for s in suggestions)}?"
                    if suggestions
                    else ""
                )
                msg = f"Unknown command: /{cmd_name}.{suffix} Type /help."
            result = CommandResult.err(msg)
            if render_outputs:
                for output in result.outputs:
                    await self.frontend.render(output)
            return result

        is_valid, error_msg = command.validate_args(args)
        if not is_valid:
            result = CommandResult.err(error_msg or "Invalid arguments")
            if render_outputs:
                for output in result.outputs:
                    await self.frontend.render(output)
            return result

        try:
            result = await command.execute(args)
        except Exception as exc:
            result = CommandResult.err(f"Command failed: {exc}")
        if render_outputs:
            await render_command_outputs(self.frontend, result.outputs)
        return result
