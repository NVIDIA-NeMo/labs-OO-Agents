# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main entry point for the NOOA TUI (terminal frontend).

Thin wrapper around the shared bootstrap.  Creates a ``TerminalFrontend``,
calls ``bootstrap()``, wires them together, and runs the session.

The ``main()`` coroutine keeps its original signature so that callers like
``examples/tools_agent_tui/example.py`` continue to work unchanged::

    await main(config=config, agent=agent)
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nooa import Agent

    from .config import Config


def _prepare_splash(config, frontend) -> list:
    """Route fullscreen splash output through the app; print native splash now."""
    if config.no_splash:
        return []

    from .config import DisplayMode, resolve_display_mode
    from .output import SplashScreen
    from .splash import show_splash

    if resolve_display_mode(config.tui) is DisplayMode.FULLSCREEN:
        return [SplashScreen()]
    show_splash(frontend.raw_console)
    return []


async def main(
    config: "Config | None" = None,
    agent: "Agent | None" = None,
    continue_last: bool = False,
    resume_session_id: str | None = None,
) -> None:
    """Main entry point for the TUI.

    Args:
        config: Optional Config instance. If None, load layered defaults.
        agent: Optional NOOA agent. If None, a TUIAgent (or custom class
               from ``config.tui.agent_spec``) is created from ``config``.
               Custom hosts must implement the InteractiveAgent queue contract.
        resume_session_id: Explicit session ID (or prefix) to resume.
    """
    from .config import Config
    from .frontend import TerminalFrontend

    if config is None:
        config = Config.load()

    # Terminal-specific: create frontend and splash screen
    frontend = TerminalFrontend(config)
    _splash_outputs = _prepare_splash(config, frontend)

    if agent is None:
        from .session import run_deferred_bootstrap

        await run_deferred_bootstrap(
            frontend=frontend,
            config=config,
            splash_outputs=_splash_outputs,
            continue_last=continue_last,
            resume_session_id=resume_session_id,
        )
        return

    from .bootstrap import bootstrap, build_initial_outputs, build_registry, build_session

    # Shared bootstrap: tracing, LLM, storage, agent, session manager
    result = await bootstrap(
        config,
        continue_last=continue_last,
        resume_session_id=resume_session_id,
        agent=agent,
    )

    _initial_outputs = build_initial_outputs(
        result,
        _splash_outputs,
        continue_last=continue_last,
    )

    # Wire frontend → registry → session
    registry = build_registry(result, frontend)
    frontend.init_input(registry)  # terminal-specific: prompt_toolkit completions
    session = build_session(result, frontend, registry, initial_outputs=_initial_outputs)

    await session.run()
