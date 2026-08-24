from __future__ import annotations

from importlib import resources
from pathlib import Path

from nooa.skill import Skill


class CachePolicyComparison(Skill):
    'Compare and implement eviction policies (LRU, LFU, FIFO, S3FIFO, ARC) for bounded-capacity caches. Use when choosing or implementing an eviction policy for a buffer pool, page cache, CDN edge, or LLM KV cache, or when writing a replay simulator that supports multiple policies. Clarifies recency vs frequency semantics, queue topology, saturating counters, ghost buffers, and the second-chance rule that distinguishes modern FIFO-family policies from classic LRU.\n\nPackage-native skill translated from a traditional TextSkill.\nUse the public Python methods on this skill instead of invoking scripts or subprocesses.\n\nNo public script APIs were inferred. This package only carries private bundled resources for package code.'

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


