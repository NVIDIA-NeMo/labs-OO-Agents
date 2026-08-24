from __future__ import annotations

from importlib import resources
from pathlib import Path

from nooa.skill import Skill


class PauseDetector(Skill):
    'Detect pauses and silence in audio using local dynamic thresholds. Use when you need to find natural pauses in lectures, board-writing silences, or breaks between sections. Uses local context comparison to avoid false positives from volume variation.\n\nPackage-native skill translated from a traditional TextSkill.\nUse the public Python methods on this skill instead of invoking scripts or subprocesses.\n\nPublic APIs:\n- detect_pauses(energies: str, output: str, start_time: int | None = 0, threshold_ratio: float | None = 0.5, min_duration: int | None = 2, window_size: int | None = 30) -> str: returns captured text output.\n- detect_pauses_2(energies: object, start_time: object = 0, threshold_ratio: object = 0.5, min_duration: object = 2, window_size: object = 30) -> object: returns the Python value from the translated implementation.'

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


    def detect_pauses(self, energies: str, output: str, start_time: int | None = 0, threshold_ratio: float | None = 0.5, min_duration: int | None = 2, window_size: int | None = 30) -> str:
        """Run the translated package implementation and return captured text output."""
        from ._impl import _scripts_detect_pauses as module
        import contextlib
        import io
        import os
        import types
        args = types.SimpleNamespace(energies=energies, output=output, start_time=start_time, threshold_ratio=threshold_ratio, min_duration=min_duration, window_size=window_size)
        buffer = io.StringIO()
        cwd = os.getcwd()
        try:
            os.chdir(self._resource_root())
            with contextlib.redirect_stdout(buffer):
                print(f'Detecting pauses from: {energies}')
                print(f'Parameters: start={start_time}s, ratio={threshold_ratio}, min={min_duration}s, window={window_size}s')
                with open(energies) as f:
                    energy_data = module.json.load(f)
                energies = energy_data['energies']
                segments = module.detect_pauses(energies, start_time, threshold_ratio, min_duration, window_size)
                total_duration = sum((s['duration'] for s in segments))
                result = {'method': 'local_dynamic_threshold', 'segments': segments, 'total_segments': len(segments), 'total_duration_seconds': total_duration, 'parameters': {'threshold_ratio': threshold_ratio, 'window_size': window_size, 'min_duration': min_duration, 'start_time': start_time}}
                with open(output, 'w') as f:
                    module.json.dump(result, f, indent=2)
                print(f'\nFound {len(segments)} pauses totaling {total_duration}s ({total_duration / 60:.2f} min)')
                print(f'Results saved to: {output}')
        finally:
            os.chdir(cwd)
        return buffer.getvalue().rstrip('\n')

    def detect_pauses_2(self, energies: object, start_time: object = 0, threshold_ratio: object = 0.5, min_duration: object = 2, window_size: object = 30) -> object:
        'Detect pauses using local dynamic threshold.'
        from ._impl import _scripts_detect_pauses as module
        return module.detect_pauses(energies=energies, start_time=start_time, threshold_ratio=threshold_ratio, min_duration=min_duration, window_size=window_size)

