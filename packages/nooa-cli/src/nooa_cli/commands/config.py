# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``nooa config`` — inspect and customize the project's config files.

Subcommands:

- ``show``   — print the resolved layers for ``llm_config.yaml``,
               ``settings.yaml``, and ``secrets.yaml`` (secret values
               redacted — only key names shown).
- ``path``   — print the user-level ``llm_config.yaml`` path.

All three files share the same directories and precedence (user →
project → env-var override); see :mod:`nooa.layered_config`.

The config chain (last wins):

1. User config at ``~/.config/nooa/llm_config.yaml``
   (overridable via ``NEMO_OO_USER_DIR``).
2. Project-local ``.nooa/llm_config.yaml``.
3. ``NEMO_OO_LLM_CONFIG`` env var — comma-separated YAML paths;
   the global override, highest priority.
"""

import os
from pathlib import Path

import click


def _user_config_path() -> Path:
    """Where the registry's user layer reads from."""
    from nooa.paths import get_user_dir

    return get_user_dir("llm_config.yaml")


def _project_config_path() -> Path:
    """Where the registry's project-local layer reads from."""
    from nooa.paths import get_project_dir

    return get_project_dir("llm_config.yaml")


def _count_aliases(path: Path) -> int | None:
    """Return the count of ``models:`` entries in a YAML file, or None on error."""
    try:
        import yaml

        data = yaml.safe_load(path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return 0
    models = data.get("models")
    if not isinstance(models, dict):
        return 0
    # Count non-None entries (null entries are removal directives, not aliases).
    return sum(1 for v in models.values() if v is not None)


@click.group()
def command():
    """Inspect and customize the LLM-registry config chain."""


@command.command("show")
def show_cmd():
    """Print the resolved config-file chain.

    Useful for debugging "why is my model alias not found" — shows each
    layer, whether it's present, and how many aliases it contributes.

    For each candidate layer, output includes a "(deduplicated …)" note
    if the file's resolved path is *also* reachable through a
    higher-priority layer — :func:`llm_config_chain` keeps only the
    higher slot, so the lower one isn't actually loaded.
    """
    from nooa.llm_config import llm_config_chain

    # Resolve the chain once up-front so each candidate line below can
    # accurately note whether it survived deduplication.
    resolved_chain = llm_config_chain()
    resolved_set = {p.resolve() for p in resolved_chain}

    def _dedup_suffix(candidate_path: Path) -> str:
        """Return ``""`` if loaded, else a "(deduplicated …)" annotation."""
        try:
            return (
                ""
                if candidate_path.resolve() in resolved_set
                else "  (deduplicated — same resolved path as a higher-priority layer)"
            )
        except OSError:
            return ""

    # Layer 1 — user
    user_path = _user_config_path()
    click.echo(f"User config ({user_path}):")
    if user_path.exists():
        n = _count_aliases(user_path)
        suffix = f"  ({n} aliases)" if n is not None else "  (unreadable)"
        click.echo(f"  present{suffix}{_dedup_suffix(user_path)}")
    else:
        click.echo("  not present")

    # Layer 2 — project-local
    project_path = _project_config_path()
    click.echo(f"Project config ({project_path}):")
    if project_path.exists():
        n = _count_aliases(project_path)
        suffix = f"  ({n} aliases)" if n is not None else "  (unreadable)"
        click.echo(f"  present{suffix}{_dedup_suffix(project_path)}")
    else:
        click.echo("  not present")

    # Layer 4 — env var (global override, highest priority)
    env_val = os.environ.get("NEMO_OO_LLM_CONFIG", "").strip()
    click.echo("NEMO_OO_LLM_CONFIG (global override):")
    if not env_val:
        click.echo("  not set")
    else:
        for raw in env_val.split(","):
            raw = raw.strip()
            if not raw:
                continue
            p = Path(raw).expanduser().resolve()
            if p.exists():
                n = _count_aliases(p)
                suffix = f"  ({n} aliases)" if n is not None else "  (unreadable)"
                click.echo(f"  {p}{suffix}{_dedup_suffix(p)}")
            else:
                click.echo(f"  {p}  (does not exist)")

    # Total — populate MODELS via the same resolved chain so the count
    # is consistent with the dedup-annotated layers above.
    from nooa.unifiedllm import MODELS, reload_registry

    reload_registry(*resolved_chain)
    click.echo("")
    click.echo(f"Total: {len(MODELS)} model aliases registered")

    # Surface which API key env vars the loaded aliases reference, and
    # whether each is set. This is the "why isn't my key working"
    # diagnostic.
    referenced = {
        cfg["api_key_env"]
        for cfg in MODELS.values()
        if isinstance(cfg, dict) and cfg.get("api_key_env")
    }
    if referenced:
        click.echo("")
        click.echo("API key env vars referenced:")
        for var in sorted(referenced):
            present = "set" if os.environ.get(var) else "NOT SET"
            click.echo(f"  {var:30s}  ({present})")

    # ── Settings & secrets (same dirs, same precedence) ───────────────
    click.echo("")
    _show_layered_file("Settings", "settings.yaml", "NEMO_OO_SETTINGS", _summarize_settings)
    click.echo("")
    _show_layered_file("Secrets", "secrets.yaml", "NEMO_OO_SECRETS", _summarize_secrets)


def _summarize_settings(path: Path) -> str:
    """One-line summary of a settings.yaml: top-level sections + key count."""
    try:
        import yaml

        data = yaml.safe_load(path.read_text())
    except Exception:
        return "(unreadable)"
    if not isinstance(data, dict) or not data:
        return "(empty)"
    parts = []
    for section in ("tui", "agent"):
        sect = data.get(section)
        if isinstance(sect, dict):
            parts.append(f"{section}: {len(sect)} keys")
    return ", ".join(parts) if parts else f"{len(data)} keys"


def _summarize_secrets(path: Path) -> str:
    """One-line summary of a secrets.yaml: env var NAMES only (values redacted)."""
    try:
        import yaml

        data = yaml.safe_load(path.read_text())
    except Exception:
        return "(unreadable)"
    if not isinstance(data, dict):
        return "(empty)"
    env_map = data.get("env")
    if not isinstance(env_map, dict) or not env_map:
        return "(no env: keys)"
    names = sorted(str(k) for k in env_map)
    return "env: " + ", ".join(names) + "  (values redacted)"


def _show_layered_file(label: str, filename: str, env_var: str, summarize) -> None:
    """Print the user → project → env-var layers for *filename*.

    *summarize* is called per existing file to produce a one-line,
    value-safe description. The highest-priority occurrence of a resolved
    path wins; lower occurrences are flagged as shadowed.
    """
    from nooa.layered_config import layered_paths
    from nooa.paths import get_project_dir, get_user_dir

    resolved_set = set(layered_paths(filename, env_var))

    def _summary(path: Path) -> str:
        try:
            note = "" if path.resolve() in resolved_set else "  (shadowed by a higher layer)"
        except OSError:
            note = ""
        return f"present — {summarize(path)}{note}"

    click.echo(f"{label} ({filename}):")

    user_path = get_user_dir(filename)
    project_path = get_project_dir(filename)
    for layer, path in (("User", user_path), ("Project", project_path)):
        click.echo(f"  {layer} ({path}):")
        click.echo(f"    {_summary(path)}" if path.exists() else "    not present")

    env_val = os.environ.get(env_var, "").strip()
    click.echo(f"  {env_var} (override):")
    if not env_val:
        click.echo("    not set")
        return
    for raw in env_val.split(","):
        raw = raw.strip()
        if not raw:
            continue
        p = Path(raw).expanduser()
        click.echo(f"    {p}: {_summary(p)}" if p.exists() else f"    {p}  (does not exist)")


@command.command("path")
def path_cmd():
    """Print the user-level ``llm_config.yaml`` path."""
    click.echo(str(_user_config_path()))
