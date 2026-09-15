# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Composable instruction discovery for interactive coding agents."""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Literal

from nooa.layered_config import load_layered_yaml
from nooa.paths import get_user_dir

logger = logging.getLogger(__name__)

_SETTINGS_FILENAME = "settings.yaml"
_SETTINGS_ENV_VAR = "NEMO_OO_SETTINGS"
_DEVELOPER_INSTRUCTIONS_FILENAME = "developer-instructions.yaml"
_DEVELOPER_INSTRUCTIONS_ENV_VAR = "NEMO_OO_DEVELOPER_INSTRUCTIONS"


@dataclass(frozen=True, slots=True)
class ResolvedInstructionProfile:
    """One selected instruction overlay, independent of how it was selected."""

    name: str | None
    model: str
    files: tuple[Path, ...] = ()
    selection: Literal["explicit", "model", "none"] = "none"
    model_pattern: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedDeveloperOverlay:
    """Private instructions selected for one repository root."""

    repository: Path
    files: tuple[Path, ...] = ()
    selection: Literal["repository", "none"] = "none"
    repository_pattern: str | None = None
    config_file: Path | None = None


@dataclass(frozen=True, slots=True)
class ResolvedInstructionStack:
    """Inspectable repository, developer, and profile instruction stack."""

    repository_files: tuple[Path, ...]
    developer: ResolvedDeveloperOverlay
    profile: ResolvedInstructionProfile

    @property
    def files(self) -> tuple[Path, ...]:
        """Return effective files in prompt precedence order."""
        return (*self.repository_files, *self.developer.files, *self.profile.files)

    def format_debug(self) -> str:
        """Return a compact, deterministic explanation of the resolved stack."""
        lines = [f"model: {self.profile.model}"]
        if self.profile.name is None:
            lines.append("instruction profile: none")
        else:
            lines.append(f"instruction profile: {self.profile.name} ({self.profile.selection})")
        if self.profile.model_pattern is not None:
            lines.append(f"matched model pattern: {self.profile.model_pattern}")
        lines.append("repository instructions:")
        lines.extend(f"  - {path}" for path in self.repository_files)
        if not self.repository_files:
            lines.append("  - none")
        lines.append("developer overlay:")
        if self.developer.repository_pattern is not None:
            lines.append(f"  matched repository: {self.developer.repository_pattern}")
        if self.developer.config_file is not None:
            lines.append(f"  config: {self.developer.config_file}")
        lines.extend(f"  - {path}" for path in self.developer.files)
        if not self.developer.files:
            lines.append("  - none")
        lines.append("profile overlay:")
        lines.extend(f"  - {path}" for path in self.profile.files)
        if not self.profile.files:
            lines.append("  - none")
        return "\n".join(lines)


def discover_agent_instruction_files(working_directory: str | Path) -> tuple[Path, ...]:
    """Return applicable ``AGENTS.md`` files from repository root to cwd.

    A file in a deeper directory is appended after its parent instruction file,
    so the resulting context naturally gives the most local instructions the
    final word. Discovery stops at the nearest Git worktree root. Outside a Git
    worktree, only the working directory itself is considered.
    """
    cwd = Path(working_directory).resolve()
    root = _git_root(cwd)
    directories = [cwd]
    if root is not None:
        distance = len(cwd.relative_to(root).parts)
        directories = list(reversed((cwd, *cwd.parents[:distance])))

    return tuple(path for directory in directories if (path := directory / "AGENTS.md").is_file())


def resolve_instruction_profile(
    working_directory: str | Path,
    *,
    model: str,
    explicit_profile: str | None = None,
) -> ResolvedInstructionProfile:
    """Resolve the instruction overlay for ``model`` in one workspace.

    Selection precedence is deterministic:

    1. ``explicit_profile`` names an entry under ``instructions.profiles``.
    2. An exact key under ``instructions.models``.
    3. The glob with the most literal characters, then lexical order as a tie-breaker.
    4. No profile overlay.

    A model entry may contain paths directly, which becomes an implicit
    profile named ``model:<pattern>``, or refer to a named profile with a
    string value. Relative paths are anchored at the Git worktree root (or the
    supplied directory outside Git).
    """
    cwd = Path(working_directory).resolve()
    workspace = _git_root(cwd) or cwd
    settings = load_layered_yaml(
        _SETTINGS_FILENAME,
        _SETTINGS_ENV_VAR,
        project_dir=workspace / ".nooa",
    )
    instructions = settings.get("instructions")
    if instructions is None:
        if explicit_profile is not None:
            raise ValueError(
                f"Instruction profile {explicit_profile!r} was requested, but no "
                "instructions.profiles are configured"
            )
        return ResolvedInstructionProfile(name=None, model=model)
    if not isinstance(instructions, Mapping):
        raise ValueError("settings.yaml instructions must be a mapping")

    profiles = _profile_configs(instructions.get("profiles"))
    if explicit_profile is not None:
        if explicit_profile not in profiles:
            raise ValueError(f"Unknown instruction profile {explicit_profile!r}")
        return ResolvedInstructionProfile(
            name=explicit_profile,
            model=model,
            files=_resolve_profile_files(profiles[explicit_profile], workspace, explicit_profile),
            selection="explicit",
        )

    models = _model_configs(instructions.get("models"))
    matched_pattern = _match_model_pattern(model, models)
    if matched_pattern is None:
        return ResolvedInstructionProfile(name=None, model=model)

    configured = models[matched_pattern]
    if isinstance(configured, str):
        if configured not in profiles:
            raise ValueError(
                f"Model instruction pattern {matched_pattern!r} references unknown "
                f"profile {configured!r}"
            )
        name = configured
        paths = profiles[configured]
    else:
        name = f"model:{matched_pattern}"
        paths = configured
    return ResolvedInstructionProfile(
        name=name,
        model=model,
        files=_resolve_profile_files(paths, workspace, name),
        selection="model",
        model_pattern=matched_pattern,
    )


def resolve_developer_overlay(
    working_directory: str | Path,
    *,
    config_file: str | Path | None = None,
) -> ResolvedDeveloperOverlay:
    """Resolve private developer instructions for one repository.

    Configuration is read only from a user-owned file, never from the
    repository. By default that file is
    ``~/.config/nooa/developer-instructions.yaml``. The
    ``NEMO_OO_DEVELOPER_INSTRUCTIONS`` environment variable or ``config_file``
    argument can select another file explicitly.

    ``repositories`` maps absolute repository-root paths or path globs to
    ordered lists of instruction files. Exact paths win over globs; otherwise
    the glob with the most literal characters wins, with lexical order as the
    tie-breaker. Relative instruction paths are anchored at the private config
    file's directory.
    """
    cwd = Path(working_directory).resolve()
    repository = _git_root(cwd) or cwd
    path, required = _developer_config_path(config_file)
    try:
        exists = path.is_file()
    except OSError as exc:
        raise ValueError(f"Could not inspect developer instruction config {path}: {exc}") from exc
    if not exists:
        if required:
            raise ValueError(f"Developer instruction config {path} does not exist")
        return ResolvedDeveloperOverlay(repository=repository)

    config = _read_developer_config(path)
    repositories = _repository_configs(config.get("repositories"), path)
    matched_pattern = _match_repository_pattern(str(repository), repositories)
    if matched_pattern is None:
        return ResolvedDeveloperOverlay(repository=repository, config_file=path)

    return ResolvedDeveloperOverlay(
        repository=repository,
        files=_resolve_configured_files(
            repositories[matched_pattern],
            path.parent,
            f"developer repository {matched_pattern!r}",
        ),
        selection="repository",
        repository_pattern=matched_pattern,
        config_file=path,
    )


def resolve_instruction_stack(
    working_directory: str | Path,
    *,
    model: str,
    explicit_profile: str | None = None,
    developer_config: str | Path | None = None,
) -> ResolvedInstructionStack:
    """Resolve repository, private developer, and model-profile instructions."""
    return ResolvedInstructionStack(
        repository_files=discover_agent_instruction_files(working_directory),
        developer=resolve_developer_overlay(
            working_directory,
            config_file=developer_config,
        ),
        profile=resolve_instruction_profile(
            working_directory,
            model=model,
            explicit_profile=explicit_profile,
        ),
    )


def _developer_config_path(config_file: str | Path | None) -> tuple[Path, bool]:
    if config_file is not None:
        return Path(config_file).expanduser().resolve(), True
    configured = os.environ.get(_DEVELOPER_INSTRUCTIONS_ENV_VAR, "").strip()
    if configured:
        return Path(configured).expanduser().resolve(), True
    return get_user_dir(_DEVELOPER_INSTRUCTIONS_FILENAME).expanduser().resolve(), False


def _read_developer_config(path: Path) -> Mapping[str, Any]:
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read developer instruction config {path}: {exc}") from exc
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Developer instruction config {path} must be a YAML mapping")
    return data


def _repository_configs(value: Any, config_file: Path) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Developer instruction config {config_file} repositories must be a mapping"
        )
    result: dict[str, tuple[str, ...]] = {}
    for raw_pattern, paths in value.items():
        if not isinstance(raw_pattern, str) or not raw_pattern.strip():
            raise ValueError(
                f"Developer instruction config {config_file} repository keys must be "
                "non-empty path strings"
            )
        pattern = _normalize_repository_pattern(raw_pattern)
        if pattern in result:
            raise ValueError(
                f"Developer instruction config {config_file} contains duplicate normalized "
                f"repository pattern {pattern!r}"
            )
        result[pattern] = _path_list(
            paths,
            f"developer instruction repositories.{raw_pattern}",
            source=str(config_file),
        )
    return result


def _profile_configs(value: Any) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("settings.yaml instructions.profiles must be a mapping")
    return {
        str(name): _path_list(paths, f"instructions.profiles.{name}")
        for name, paths in value.items()
    }


def _model_configs(value: Any) -> dict[str, str | tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("settings.yaml instructions.models must be a mapping")
    result: dict[str, str | tuple[str, ...]] = {}
    for pattern, configured in value.items():
        key = str(pattern)
        result[key] = (
            configured
            if isinstance(configured, str)
            else _path_list(configured, f"instructions.models.{key}")
        )
    return result


def _path_list(
    value: Any,
    setting: str,
    *,
    source: str = "settings.yaml",
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{source} {setting} must be a list of paths")
    if not all(isinstance(path, str) and path.strip() for path in value):
        raise ValueError(f"{source} {setting} must contain only non-empty path strings")
    return tuple(value)


def _match_model_pattern(
    model: str,
    models: Mapping[str, str | tuple[str, ...]],
) -> str | None:
    if model in models:
        return model
    matches = [pattern for pattern in models if fnmatchcase(model, pattern)]
    if not matches:
        return None
    return min(matches, key=lambda pattern: (-_literal_character_count(pattern), pattern))


def _literal_character_count(pattern: str) -> int:
    """Return a stable glob-specificity score without interpreting the pattern."""
    return sum(character not in "*?[]" for character in pattern)


def _normalize_repository_pattern(pattern: str) -> str:
    expanded = os.path.expanduser(pattern.strip())
    if not Path(expanded).is_absolute():
        raise ValueError(f"Developer repository pattern {pattern!r} must be an absolute path")
    normalized = os.path.normpath(expanded)
    if not any(character in normalized for character in "*?["):
        return str(Path(normalized).resolve())
    return normalized


def _match_repository_pattern(
    repository: str,
    repositories: Mapping[str, tuple[str, ...]],
) -> str | None:
    if repository in repositories:
        return repository
    matches = [pattern for pattern in repositories if fnmatchcase(repository, pattern)]
    if not matches:
        return None
    return min(matches, key=lambda pattern: (-_literal_character_count(pattern), pattern))


def _resolve_profile_files(
    configured: Sequence[str],
    workspace: Path,
    profile: str,
) -> tuple[Path, ...]:
    return _resolve_configured_files(configured, workspace, f"profile {profile!r}")


def _resolve_configured_files(
    configured: Sequence[str],
    base_directory: Path,
    owner: str,
) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for raw_path in configured:
        candidate = Path(raw_path).expanduser()
        path = (candidate if candidate.is_absolute() else base_directory / candidate).resolve()
        if path in seen:
            continue
        try:
            exists = path.is_file()
        except OSError as exc:
            raise ValueError(
                f"Could not inspect instruction file {path} for {owner}: {exc}"
            ) from exc
        if not exists:
            raise ValueError(f"Instruction file {path} for {owner} does not exist")
        seen.add(path)
        result.append(path)
    return tuple(result)


#: Per-file and combined caps on repository instructions. The content is
#: workspace-controlled and lands in a prefix context block on every turn, so an
#: unbounded read is a memory and context-window risk at session setup.
_MAX_INSTRUCTION_FILE_CHARS = 100_000
_MAX_INSTRUCTION_TOTAL_CHARS = 200_000
_SECTION_SEPARATOR = "\n\n---\n\n"


def render_agent_instructions(working_directory: str | Path) -> str:
    """Render applicable repository instructions as one bounded context block.

    Reads are bounded rather than truncated after the fact: the content is
    workspace-controlled and lands in a prefix context block on every turn, so
    reading a huge file in full before discarding most of it would still cost
    the memory. The budget covers the rendered text — headers, separators and
    truncation markers included — not just the retained file content.
    """
    return _render_instruction_files(discover_agent_instruction_files(working_directory))


def render_instruction_profile(profile: ResolvedInstructionProfile) -> str:
    """Render one already-resolved profile overlay as a bounded context block."""
    return _render_instruction_files(profile.files, strict=True)


def render_developer_overlay(overlay: ResolvedDeveloperOverlay) -> str:
    """Render one resolved private developer overlay as a bounded context block."""
    return _render_instruction_files(overlay.files, strict=True)


def _render_instruction_files(paths: Sequence[Path], *, strict: bool = False) -> str:
    sections: list[str] = []
    remaining = _MAX_INSTRUCTION_TOTAL_CHARS
    for path in paths:
        if remaining <= 0:
            logger.warning("Skipping repository instructions from %s: total limit reached", path)
            continue
        budget = min(_MAX_INSTRUCTION_FILE_CHARS, remaining)
        try:
            with path.open("r", encoding="utf-8") as stream:
                # One char past the budget is enough to know it was cut.
                content = stream.read(budget + 1)
        except (OSError, UnicodeError) as exc:
            if strict:
                raise ValueError(
                    f"Could not read configured instruction file {path}: {exc}"
                ) from exc
            logger.warning("Skipping unreadable repository instructions from %s: %s", path, exc)
            continue
        truncated = len(content) > budget
        if truncated:
            content = content[:budget]
            logger.warning("Truncating repository instructions from %s at %d chars", path, budget)
        content = content.strip()
        if not content:
            continue
        if truncated:
            content += "\n\n[... truncated ...]"
        section = f"Instructions from {path}:\n\n{content}"
        remaining -= len(section) + len(_SECTION_SEPARATOR)
        sections.append(section)
    return _SECTION_SEPARATOR.join(sections)


def _git_root(cwd: Path) -> Path | None:
    for directory in (cwd, *cwd.parents):
        if (directory / ".git").exists():
            return directory
    return None


__all__ = [
    "ResolvedDeveloperOverlay",
    "ResolvedInstructionProfile",
    "ResolvedInstructionStack",
    "discover_agent_instruction_files",
    "render_agent_instructions",
    "render_developer_overlay",
    "render_instruction_profile",
    "resolve_developer_overlay",
    "resolve_instruction_profile",
    "resolve_instruction_stack",
]
