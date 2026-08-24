from __future__ import annotations

from importlib import resources
from pathlib import Path

from nooa.skill import Skill


class AudioExtractor(Skill):
    'Extract audio from video files to WAV format. Use when you need to analyze audio from video, prepare audio for energy calculation, or convert video audio to standard format for processing.\n\nPackage-native skill translated from a traditional TextSkill.\nUse the public Python methods on this skill instead of invoking scripts or subprocesses.\n\nPublic APIs:\n- extract_audio(video: str, output: str, sample_rate: int | None = 16000, duration: int | None = None) -> str: returns captured text output.\n- extract_audio_2(video_path: object, output_path: object, sample_rate: object = 16000, duration: object = None) -> object: returns the Python value from the translated implementation.'

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


    def extract_audio(self, video: str, output: str, sample_rate: int | None = 16000, duration: int | None = None) -> str:
        """Run the translated package implementation and return captured text output."""
        from ._impl import _scripts_extract_audio as module
        import contextlib
        import io
        import os
        import types
        args = types.SimpleNamespace(video=video, output=output, sample_rate=sample_rate, duration=duration)
        buffer = io.StringIO()
        cwd = os.getcwd()
        try:
            os.chdir(self._resource_root())
            with contextlib.redirect_stdout(buffer):
                print(f'Extracting audio from: {video}')
                print(f'Sample rate: {sample_rate} Hz')
                if duration:
                    print(f'Duration limit: {duration}s')
                module.extract_audio(video, output, sample_rate, duration)
                size_mb = module.os.path.getsize(output) / (1024 * 1024)
                print(f'\nAudio extracted to: {output}')
                print(f'File size: {size_mb:.2f} MB')
        finally:
            os.chdir(cwd)
        return buffer.getvalue().rstrip('\n')

    def extract_audio_2(self, video_path: object, output_path: object, sample_rate: object = 16000, duration: object = None) -> object:
        'Extract audio from video to WAV format.'
        from ._impl import _scripts_extract_audio as module
        return module.extract_audio(video_path=video_path, output_path=output_path, sample_rate=sample_rate, duration=duration)

