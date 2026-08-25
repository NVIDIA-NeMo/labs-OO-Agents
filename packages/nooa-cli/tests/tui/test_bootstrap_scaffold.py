# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for first-run settings scaffolding.

These cover ``_scaffold_settings``, which writes the starter
``~/.config/nooa/settings.yaml`` the first time the TUI launches (when no
settings file exists in any layer). The template contains literal braces
(e.g. ``${MAAS_API_KEY}`` in the inline-MCP example), so it must be rendered
with ``str.replace`` — ``str.format`` would raise ``KeyError`` on the
unrelated brace and crash startup before any config is written.
"""

from nooa_cli.tui.bootstrap import _scaffold_settings
from nooa_cli.tui.config import Config
from nooa_cli.tui.settings import SETTINGS_TEMPLATE, render_settings_template


def _isolate(tmp_path, monkeypatch):
    """Point user + project dirs at empty temp dirs; clear the env override."""
    user = tmp_path / "user"
    proj = tmp_path / "proj"
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(proj))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    return user


def test_scaffold_writes_settings_with_model_and_preserves_literal_braces(tmp_path, monkeypatch):
    """First-run scaffold writes settings.yaml without choking on literal braces."""
    user = _isolate(tmp_path, monkeypatch)

    config = Config()
    config.tui.default_model = "nvidia/llama-3.3"
    _scaffold_settings(config)

    settings_path = user / "settings.yaml"
    assert settings_path.exists()
    written = settings_path.read_text()

    # The {default_model} placeholder is substituted...
    assert "nvidia/llama-3.3" in written
    assert "{default_model}" not in written
    assert "# theme: mocha" in written
    # ...while literal braces from the inline-MCP example survive verbatim.
    assert "${MAAS_API_KEY}" in written


def test_scaffold_skips_when_settings_present(tmp_path, monkeypatch):
    """An existing settings.yaml (any layer) is never overwritten."""
    user = _isolate(tmp_path, monkeypatch)
    user.mkdir(parents=True)
    settings_path = user / "settings.yaml"
    settings_path.write_text("# user edits\n")

    _scaffold_settings(Config())

    assert settings_path.read_text() == "# user edits\n"


def test_template_has_only_the_default_model_field():
    """Guard: {default_model} is the only single-brace field in the template.

    If a future edit adds another ``{field}`` it would silently break the
    plain .replace() substitution, so flag it here.
    """
    import re

    # Drop ${...} shell-style refs, then the known placeholder, then assert no
    # other single-brace {token} survives.
    stripped = re.sub(r"\$\{[^}]*\}", "", SETTINGS_TEMPLATE)
    stripped = stripped.replace("{default_model}", "")
    assert re.search(r"(?<!\{)\{[A-Za-z_][A-Za-z0-9_]*\}", stripped) is None


def test_render_does_not_raise_on_literal_braces():
    """render_settings_template must not str.format() the ${...} examples."""
    out = render_settings_template(Config())
    assert "${MAAS_API_KEY}" in out
