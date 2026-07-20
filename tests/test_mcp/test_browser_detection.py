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

import nooa.mcp.oauth as oauth


def test_uses_launcher_when_webbrowser_get_fails(monkeypatch):
    # Simulate "no registered browser".
    def _raise(*_a, **_k):
        raise webbrowser.Error("could not locate runnable browser")

    monkeypatch.setattr(webbrowser, "get", _raise)
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
    monkeypatch.setattr(webbrowser, "get", lambda: object())
    assert oauth._system_browser_available() is True
