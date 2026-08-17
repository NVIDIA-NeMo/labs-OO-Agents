# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""_system_browser_available detects launcher executables so loopback OAuth is used.

Regression: on hosts where webbrowser.get() raises (no registered browser) but a
launcher like xdg-open / sensible-browser IS on PATH (e.g. a sandbox that
forwards to the host browser), OAuth fell back to the manual OOB paste flow.
The loopback-callback flow (pop browser + local listener, no paste) should be
used instead.
"""

import webbrowser

import pytest

import nooa.mcp.oauth as oauth


@pytest.fixture(autouse=True)
def _clear_remote_runtime_signals(monkeypatch):
    """Keep host runtime markers from leaking into browser-detection tests."""
    for name in ("SANDBOX_VM_ID", "SBX_NO_DISPLAY", "SSH_CONNECTION"):
        monkeypatch.delenv(name, raising=False)


def test_uses_launcher_when_webbrowser_get_fails(monkeypatch):
    # Simulate "no registered browser".
    def _raise(*_a, **_k):
        raise webbrowser.Error("could not locate runnable browser")

    monkeypatch.setattr(webbrowser, "get", _raise)
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    # Pretend xdg-open is on PATH.
    monkeypatch.setattr(
        oauth.shutil, "which", lambda name: "/usr/bin/xdg-open" if name == "xdg-open" else None
    )
    registered = {}
    monkeypatch.setattr(
        webbrowser,
        "register",
        lambda name, klass, instance=None, *, preferred=False: registered.setdefault(
            name, preferred
        ),
    )

    assert oauth._system_browser_available() is True
    assert "xdg-open" in registered


def test_false_when_no_browser_and_no_launcher(monkeypatch):
    def _raise(*_a, **_k):
        raise webbrowser.Error("none")

    monkeypatch.setattr(webbrowser, "get", _raise)
    monkeypatch.setattr(oauth.shutil, "which", lambda name: None)
    assert oauth._system_browser_available() is False


def test_true_when_webbrowser_get_succeeds(monkeypatch):
    monkeypatch.delenv("SSH_CONNECTION", raising=False)
    monkeypatch.setattr(webbrowser, "get", lambda: object())
    assert oauth._system_browser_available() is True


def test_false_over_ssh_without_display_even_when_xdg_open_exists(monkeypatch):
    """A bare xdg-open executable is not a browser in a headless SSH session."""

    def _raise(*_a, **_k):
        raise webbrowser.Error("could not locate runnable browser")

    # Simulate xdg-open having already been registered by an earlier probe:
    # the headless check must run before this apparent browser success.
    monkeypatch.setattr(webbrowser, "get", lambda *a, **k: object())
    monkeypatch.setattr(oauth.os, "name", "posix")
    monkeypatch.setenv("SSH_CONNECTION", "client 123 server 22")
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("WAYLAND_DISPLAY", raising=False)
    monkeypatch.setattr(
        oauth.shutil, "which", lambda name: "/usr/bin/xdg-open" if name == "xdg-open" else None
    )

    assert oauth._system_browser_available() is False


def test_false_in_headless_container_even_with_forwarded_wayland(monkeypatch):
    """Forwarded host-display metadata must not imply a reachable loopback browser."""
    monkeypatch.setattr(webbrowser, "get", lambda *a, **k: object())
    monkeypatch.setenv("SANDBOX_VM_ID", "agent-example")
    monkeypatch.setenv("SBX_NO_DISPLAY", "1")
    monkeypatch.setenv("WAYLAND_DISPLAY", "wayland-0")
    monkeypatch.delenv("SSH_CONNECTION", raising=False)

    assert oauth._system_browser_available() is False
