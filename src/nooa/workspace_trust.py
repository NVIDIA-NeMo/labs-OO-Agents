# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workspace trust: whether a folder may run its own code.

A folder is *trusted* when it, or one of its parents, is listed under
``trusted_folders`` in the user-level ``settings.yaml``
(``nooa.paths.get_user_dir("settings.yaml")``) or in a file named by
``NEMO_OO_SETTINGS`` (how CI and scripts trust a folder)::

    trusted_folders:
      - ~/src/my-project

Entries must be absolute paths (``~`` is expanded); relative entries are
ignored with a warning. The project layer (``<workspace>/.nooa/settings.yaml``)
is never read here: a repository must not be able to trust itself.

Interactive hosts (the coding agent behind ``nooa-acp`` and Atom) do not run
code from an untrusted workspace (code skills in ``.nooa/skills``,
``.claude/skills`` and similar directories). A path counts as workspace code
when it was found in the workspace or points into it (symlinks), unless it is
under one of the user's own skill roots (``user_skill_roots``). Plain agents
and the skill registry used directly are not affected.
"""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SETTINGS_FILENAME = "settings.yaml"
SETTINGS_ENV_VAR = "NEMO_OO_SETTINGS"
TRUSTED_FOLDERS_KEY = "trusted_folders"
USER_SKILL_DIRS = (Path(".agents/skills"), Path(".claude/skills"), Path(".claude/commands"))
"""The user's own skill roots, relative to the home folder."""
HOME_REFUSAL = (
    "Open a project folder instead; trusting your home folder would trust every repository in it."
)


class WorkspaceTrustError(ValueError):
    """``trust()`` refused: the folder cannot be trusted, or the settings file is not valid."""


def untrusted_notice(folder: str | Path, subject: str = "Code skills") -> str:
    """The notice a host shows when ``subject`` in the untrusted ``folder`` was skipped."""
    notice = f"{subject} in this folder are off because it is not trusted."
    if can_trust(folder):
        return f"{notice} Type /trust to allow."
    return f"{notice} {HOME_REFUSAL}"


def user_settings_file() -> Path:
    """The user-level ``settings.yaml`` that ``trust()`` writes."""
    from nooa.paths import get_user_dir

    return get_user_dir(SETTINGS_FILENAME)


def _trust_files() -> Iterator[Path]:
    """The files that may trust a folder: the user file, then ``NEMO_OO_SETTINGS`` files."""
    yield user_settings_file()
    for entry in os.environ.get(SETTINGS_ENV_VAR, "").split(","):
        if entry.strip():
            yield Path(entry.strip()).expanduser()


def _read_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def _resolve(folder: str | Path) -> Path:
    return Path(folder).expanduser().resolve()


def _absolute(path: str | Path) -> Path:
    """``path`` made absolute without following symlinks."""
    return Path(os.path.abspath(Path(path).expanduser()))


def _under(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def trusted_folders() -> list[Path]:
    """Every folder listed under ``trusted_folders`` in the user and ``NEMO_OO_SETTINGS`` files."""
    folders: list[Path] = []
    for path in _trust_files():
        value = _read_mapping(path).get(TRUSTED_FOLDERS_KEY)
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, str) or not item.strip():
                continue
            folder = Path(item.strip()).expanduser()
            if not folder.is_absolute():
                logger.warning(
                    "Ignoring relative trusted_folders entry %r in %s: use an absolute path",
                    item,
                    path,
                )
                continue
            folders.append(folder.resolve())
    return folders


def is_trusted(folder: str | Path) -> bool:
    """Whether ``folder`` (resolved) or one of its parents is a trusted folder."""
    target = _resolve(folder)
    return any(_under(target, item) for item in trusted_folders())


def can_trust(folder: str | Path) -> bool:
    """False for the home folder and filesystem roots, which ``trust()`` refuses."""
    target = _resolve(folder)
    return target != _resolve(Path.home()) and target.parent != target


def trust(folder: str | Path) -> None:
    """Add ``folder`` (resolved) to ``trusted_folders`` in the user ``settings.yaml``.

    Raises ``WorkspaceTrustError`` for the home folder or a filesystem root
    (``HOME_REFUSAL``), and when the settings file exists but is not a YAML
    mapping (it is left unchanged). When the file has no ``trusted_folders``
    yet, the entry is appended, so its comments stay; otherwise the file is
    rewritten from its data. The write holds the file's lock and replaces the
    file atomically.
    """
    import yaml

    from nooa._filelock import file_write_lock

    if not can_trust(folder):
        raise WorkspaceTrustError(HOME_REFUSAL)
    target = str(_resolve(folder))
    with file_write_lock(user_settings_file()) as path:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        try:
            data = yaml.safe_load(text) if text.strip() else {}
        except yaml.YAMLError as exc:
            raise WorkspaceTrustError(
                f"{path} is not valid YAML, so the folder was not trusted; fix the file first ({exc})"
            ) from None
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise WorkspaceTrustError(
                f"{path} is not a YAML mapping, so the folder was not trusted; fix the file first"
            )
        current = data.get(TRUSTED_FOLDERS_KEY)
        if isinstance(current, str):
            current = [current]
        folders = (
            [item for item in current if isinstance(item, str)] if isinstance(current, list) else []
        )
        if target in folders:
            return
        new_data = {**data, TRUSTED_FOLDERS_KEY: [*folders, target]}
        new_text = yaml.safe_dump(new_data, sort_keys=False)
        if TRUSTED_FOLDERS_KEY not in data:
            # Append, keeping the file's comments, if that gives the same data.
            appended = text + ("\n" if text and not text.endswith("\n") else "")
            appended += yaml.safe_dump({TRUSTED_FOLDERS_KEY: [target]}, sort_keys=False)
            try:
                if yaml.safe_load(appended) == new_data:
                    new_text = appended
            except yaml.YAMLError:
                pass
        fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(new_text)
            if path.exists():
                os.chmod(temporary, path.stat().st_mode & 0o777)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)


def launch_project_trusted() -> bool:
    """Whether the process's own project directory (``nooa.paths.get_project_dir``) is trusted.

    Process-start loading (``secrets.yaml``, ``llm_config.yaml``) takes its
    project layer from there, not from a session workspace.
    """
    from nooa.paths import get_project_dir

    return is_trusted(get_project_dir())


def user_skill_roots(extra: Iterable[str | Path] = ()) -> list[Path]:
    """The user's own skill roots: ``~/.agents/skills``, ``~/.claude/skills``, ``~/.claude/commands``.

    ``extra`` adds directories from user-level configuration (the user
    ``settings.yaml`` and ``NEMO_OO_SETTINGS``); only absolute ones (after
    ``~`` expansion) count, since a relative one names a workspace folder.
    """
    roots = [Path.home() / path for path in USER_SKILL_DIRS]
    for path in extra:
        candidate = Path(path).expanduser()
        if candidate.is_absolute():
            roots.append(candidate)
    return [_absolute(root) for root in roots]


def is_inside(path: str | Path, root: str | Path) -> bool:
    """Whether ``path`` is in ``root``: where it was found, or where it points (symlinks)."""
    found = _absolute(path)
    roots = {_absolute(root), _resolve(root)}
    return any(_under(candidate, item) for candidate in (found, found.resolve()) for item in roots)


def code_allowed_outside(
    root: str | Path, *, user_roots: Iterable[str | Path] = ()
) -> Callable[[Path], bool]:
    """A ``SkillRegistry.code_allowed`` predicate refusing code in ``root``.

    A path is allowed when it is under one of ``user_roots`` (the user's own
    skill roots, also when ``root`` contains them), or when neither where it
    was found nor where it points is in ``root`` (``is_inside``).
    """
    allowed_roots = [_absolute(item) for item in user_roots]

    def allowed(path: Path) -> bool:
        found = _absolute(path)
        if any(
            _under(found, item) or _under(found.resolve(), item.resolve()) for item in allowed_roots
        ):
            return True
        return not is_inside(found, root)

    return allowed


__all__ = [
    "HOME_REFUSAL",
    "WorkspaceTrustError",
    "can_trust",
    "code_allowed_outside",
    "is_inside",
    "is_trusted",
    "launch_project_trusted",
    "trust",
    "trusted_folders",
    "untrusted_notice",
    "user_settings_file",
    "user_skill_roots",
]
