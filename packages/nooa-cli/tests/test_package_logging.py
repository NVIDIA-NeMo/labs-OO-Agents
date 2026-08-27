# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``nooa_cli`` must not log to stderr unless the application configures logging.

Without a NullHandler on the package root logger, ``logging.lastResort``
prints every ``nooa_cli.*`` WARNING straight to stderr with no level or
logger name attached. The TUI captures stray stderr into its scrollback, so
those records surface as transcript noise (e.g. RepoTools' tree-sitter
fallback notice). ``nooa`` already installs one; this pins the CLI package to
the same contract.
"""

import io
import logging
from contextlib import contextmanager


@contextmanager
def _no_application_logging(buffer: io.StringIO):
    """Strip root handlers and route ``lastResort`` into *buffer*.

    Reproduces a bare process — pytest's logging plugin normally attaches root
    handlers, which would satisfy ``callHandlers`` and mask the regression.
    """
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_last_resort = logging.lastResort
    root.handlers = []
    logging.lastResort = logging.StreamHandler(buffer)
    try:
        yield
    finally:
        root.handlers = saved_handlers
        logging.lastResort = saved_last_resort


def test_package_logger_has_null_handler():
    import nooa_cli  # noqa: F401

    logger = logging.getLogger("nooa_cli")
    assert any(isinstance(handler, logging.NullHandler) for handler in logger.handlers)


def test_importing_a_submodule_installs_the_handler():
    """Importing ``nooa_cli.tools.repo_tools`` imports the package root first."""
    import nooa_cli.tools.repo_tools  # noqa: F401

    assert logging.getLogger("nooa_cli.tools.repo_tools").hasHandlers()


def test_repo_tools_tree_sitter_notice_stays_off_stderr(monkeypatch):
    from nooa_cli.tools import repo_tools

    monkeypatch.setattr(repo_tools, "_tree_sitter_available", lambda: False)

    buffer = io.StringIO()
    with _no_application_logging(buffer):
        repo_tools.RepoTools(root=".")

    assert buffer.getvalue() == ""


def test_last_resort_still_fires_for_unhandled_loggers():
    """Guard the harness itself: the fixture must not suppress everything."""
    buffer = io.StringIO()
    with _no_application_logging(buffer):
        logging.getLogger("some_other_package.module").warning("leaked")

    assert "leaked" in buffer.getvalue()
