# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main entry point for NeMo OO Agents TUI (terminal frontend).

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


async def main(
    config: "Config | None" = None,
    agent: "Agent | None" = None,
    continue_last: bool = False,
    resume_session_id: str | None = None,
) -> None:
    """Main entry point for the TUI.

    Args:
        config: Optional Config instance. If None, load layered defaults.
        agent: Optional NeMo OO Agents agent.  If None, a TUIAgent (or custom class
               from ``config.tui.agent_spec``) is created from ``config``.
               Custom hosts must implement the InteractiveAgent queue contract.
        resume_session_id: Explicit session ID (or prefix) to resume.
    """
    from .bootstrap import bootstrap, build_registry, build_session, build_startup_info
    from .config import Config
    from .frontend import TerminalFrontend
    from .output import TextOutput, _RichReplayPayload
    from .session_manager import SESSIONS_DIR, build_resume_outputs
    from .splash import show_splash

    if config is None:
        config = Config.load()

    # Terminal-specific: create frontend and splash screen
    frontend = TerminalFrontend(config)
    if not config.no_splash:
        show_splash(frontend.raw_console)

    # Shared bootstrap: tracing, LLM, storage, agent, session manager
    result = await bootstrap(
        config,
        continue_last=continue_last,
        resume_session_id=resume_session_id,
        agent=agent,
    )

    _startup_info = build_startup_info(result)
    _initial_outputs = [*result.messages, _startup_info]

    # Show resumed session history (interleaved with any rich content). Terminal
    # text/markdown outputs are deferred until Session.run(), after the frontend
    # console is redirected through TUIApplication.emit_block; that makes them
    # part of fullscreen resize replay instead of one-off pre-app writes.
    if result.resumed and result.session_id is not None:
        import os as _os

        _in_nemo_term = bool(_os.environ.get("NEMO_OO_RICH_URL"))
        _db_path = SESSIONS_DIR / f"{result.session_id}.db"
        _resume_outputs = build_resume_outputs(
            _db_path, result.session_id, in_nemo_term=_in_nemo_term
        )
        if _resume_outputs:
            _rich_url = _os.environ.get("NEMO_OO_RICH_URL") if _in_nemo_term else None
            for _item in _resume_outputs:
                if isinstance(_item, _RichReplayPayload):
                    if _rich_url:
                        try:
                            import httpx as _httpx

                            _httpx.post(
                                _rich_url,
                                json={**_item.payload, "_replay": True},
                                timeout=5.0,
                            )
                        except Exception:
                            pass
                else:
                    _initial_outputs.append(_item)
            _initial_outputs.append(
                TextOutput(f"Session {result.session_id[:8]} resumed.", "status")
            )
        else:
            _initial_outputs.append(TextOutput("No previous session with turns found.", "info"))
    elif continue_last:
        _initial_outputs.append(TextOutput("No previous session with turns found.", "info"))

    # Wire frontend → registry → session
    registry = build_registry(result, frontend)
    registry.startup_info = _startup_info
    frontend.init_input(registry)  # terminal-specific: prompt_toolkit completions
    session = build_session(result, frontend, registry, initial_outputs=_initial_outputs)

    await session.run()
