from __future__ import annotations

from importlib import resources
from pathlib import Path

from nooa.skill import Skill


class ReportGenerator(Skill):
    'Generate compression reports for video processing. Use when you need to create structured JSON reports with duration statistics, compression ratios, and segment details after video processing.\n\nPackage-native skill translated from a traditional TextSkill.\nUse the public Python methods on this skill instead of invoking scripts or subprocesses.\n\nPublic APIs:\n- generate_report(original: str, compressed: str, segments: str | None = None, output: str) -> str: returns captured text output.\n- get_duration(video_path: object) -> object: returns the Python value from the translated implementation.\n- generate_report_2(original_path: object, compressed_path: object, segments_path: object = None) -> object: returns the Python value from the translated implementation.'

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


    def generate_report(self, original: str, compressed: str, output: str, segments: str | None = None) -> str:
        """Run the translated package implementation and return captured text output."""
        from ._impl import _scripts_generate_report as module
        import contextlib
        import io
        import os
        import types
        args = types.SimpleNamespace(original=original, compressed=compressed, segments=segments, output=output)
        buffer = io.StringIO()
        cwd = os.getcwd()
        try:
            os.chdir(self._resource_root())
            with contextlib.redirect_stdout(buffer):
                print('Generating compression report...')
                print(f'  Original: {original}')
                print(f'  Compressed: {compressed}')
                report = module.generate_report(original, compressed, segments)
                with open(output, 'w') as f:
                    module.json.dump(report, f, indent=2)
                print(f"\nOriginal: {report['original_duration_seconds']}s")
                print(f"Compressed: {report['compressed_duration_seconds']}s")
                print(f"Removed: {report['removed_duration_seconds']}s")
                print(f"Compression: {report['compression_percentage']:.1f}%")
                print(f'Report saved to: {output}')
        finally:
            os.chdir(cwd)
        return buffer.getvalue().rstrip('\n')

    def get_duration(self, video_path: object) -> object:
        'Get video duration using ffprobe.'
        from ._impl import _scripts_generate_report as module
        return module.get_duration(video_path=video_path)

    def generate_report_2(self, original_path: object, compressed_path: object, segments_path: object = None) -> object:
        'Generate compression report.'
        from ._impl import _scripts_generate_report as module
        return module.generate_report(original_path=original_path, compressed_path=compressed_path, segments_path=segments_path)

