from __future__ import annotations

from importlib import resources
from pathlib import Path

from nooa.skill import Skill


class SegmentCombiner(Skill):
    'Combine multiple segment detection results into a unified list. Use when you need to merge segments from different detectors, prepare removal lists for video processing, or consolidate detection outputs.\n\nPackage-native skill translated from a traditional TextSkill.\nUse the public Python methods on this skill instead of invoking scripts or subprocesses.\n\nPublic APIs:\n- combine_segments(segment_files: object) -> object: returns the Python value from the translated implementation.'

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


    def combine_segments(self, segment_files: object) -> object:
        'Combine segments from multiple detection files.'
        from ._impl import _scripts_combine_segments as module
        return module.combine_segments(segment_files=segment_files)

