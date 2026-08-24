from __future__ import annotations

from importlib import resources
from pathlib import Path

from nooa.skill import Skill


class ErlangOtpBehaviors(Skill):
    'Use when oTP behaviors including gen_server for stateful processes, gen_statem for state machines, supervisors for fault tolerance, gen_event for event handling, and building robust, production-ready Erlang applications with proven patterns.\n\nPackage-native skill translated from a traditional TextSkill.\nUse the public Python methods on this skill instead of invoking scripts or subprocesses.\n\nNo public script APIs were inferred. This package only carries private bundled resources for package code.'

    def _resource_root(self):
        return resources.files(__package__) / "resources"

    def _list_resources(self) -> list[str]:
        """Return all bundled resource paths."""
        root = self._resource_root()
        return sorted(
            path.relative_to(root).as_posix()
            for path in Path(root).rglob("*")
            if path.is_file()
        )

    def _read_resource(self, path: str) -> str:
        """Read a bundled resource as text."""
        root = Path(self._resource_root()).resolve()
        resolved = (root / path).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Path {path!r} escapes package resources")
        if not resolved.is_file():
            raise FileNotFoundError(path)
        return resolved.read_text()


