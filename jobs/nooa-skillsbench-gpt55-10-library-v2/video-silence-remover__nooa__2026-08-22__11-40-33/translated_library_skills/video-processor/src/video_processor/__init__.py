from __future__ import annotations

from importlib import resources
from pathlib import Path

from nooa.skill import Skill


class VideoProcessor(Skill):
    'Process videos by removing segments and concatenating remaining parts. Use when you need to remove detected pauses/openings from videos, create highlight reels, or batch process segment removals using ffmpeg filter_complex.\n\nPackage-native skill translated from a traditional TextSkill.\nUse the public Python methods on this skill instead of invoking scripts or subprocesses.\n\nPublic APIs:\n- get_video_duration(video_path: object) -> object: returns the Python value from the translated implementation.\n- load_segments(segment_files: object) -> object: returns the Python value from the translated implementation.\n- calculate_keep_segments(remove_segments: object, total_duration: object) -> object: returns the Python value from the translated implementation.\n- build_ffmpeg_filter(keep_segments: object) -> object: returns the Python value from the translated implementation.\n- process_video(input_path: object, output_path: object, keep_segments: object) -> object: returns the Python value from the translated implementation.'

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


    def get_video_duration(self, video_path: object) -> object:
        'Get video duration in seconds.'
        from ._impl import _scripts_process_video as module
        return module.get_video_duration(video_path=video_path)

    def load_segments(self, segment_files: object) -> object:
        'Load segments from one or more JSON files.'
        from ._impl import _scripts_process_video as module
        return module.load_segments(segment_files=segment_files)

    def calculate_keep_segments(self, remove_segments: object, total_duration: object) -> object:
        'Calculate segments to keep (inverse of remove segments).'
        from ._impl import _scripts_process_video as module
        return module.calculate_keep_segments(remove_segments=remove_segments, total_duration=total_duration)

    def build_ffmpeg_filter(self, keep_segments: object) -> object:
        'Build ffmpeg filter_complex for segment processing.'
        from ._impl import _scripts_process_video as module
        return module.build_ffmpeg_filter(keep_segments=keep_segments)

    def process_video(self, input_path: object, output_path: object, keep_segments: object) -> object:
        'Process video using ffmpeg.'
        from ._impl import _scripts_process_video as module
        return module.process_video(input_path=input_path, output_path=output_path, keep_segments=keep_segments)

