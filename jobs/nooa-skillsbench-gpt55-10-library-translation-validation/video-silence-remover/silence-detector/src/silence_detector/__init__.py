from __future__ import annotations

from importlib import resources
from pathlib import Path

from nooa.skill import Skill


class SilenceDetector(Skill):
    'Detect initial silence segments in audio/video using energy-based analysis. Use when you need to find low-energy periods at the start of recordings (title slides, setup time, pre-roll silence).\n\nPackage-native skill translated from a traditional TextSkill.\nUse the public Python methods on this skill instead of invoking scripts or subprocesses.\n\nPublic APIs:\n- detect_silence(energies: str, output: str, threshold_multiplier: float | None = 1.5, initial_window: int | None = 60, smoothing_window: int | None = 30) -> str: returns captured text output.\n- detect_initial_silence(energies: object, threshold_multiplier: object = 1.5, initial_window: object = 60, smoothing_window: object = 30) -> object: returns the Python value from the translated implementation.'

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


    def detect_silence(self, energies: str, output: str, threshold_multiplier: float | None = 1.5, initial_window: int | None = 60, smoothing_window: int | None = 30) -> str:
        """Run the translated package implementation and return captured text output."""
        from ._impl import _scripts_detect_silence as module
        import contextlib
        import io
        import os
        import types
        args = types.SimpleNamespace(energies=energies, output=output, threshold_multiplier=threshold_multiplier, initial_window=initial_window, smoothing_window=smoothing_window)
        buffer = io.StringIO()
        cwd = os.getcwd()
        try:
            os.chdir(self._resource_root())
            with contextlib.redirect_stdout(buffer):
                print(f'Detecting initial silence from: {energies}')
                print(f'Parameters: multiplier={threshold_multiplier}, initial={initial_window}s, smoothing={smoothing_window}s')
                with open(energies) as f:
                    energy_data = module.json.load(f)
                energies = energy_data['energies']
                total_seconds = energy_data['total_seconds']
                silence_end, analysis = module.detect_initial_silence(energies, threshold_multiplier, initial_window, smoothing_window)
                segments = []
                if silence_end > 0:
                    segments.append({'start': 0, 'end': silence_end, 'duration': silence_end})
                result = {'method': 'energy_threshold', 'segments': segments, 'total_segments': len(segments), 'total_duration_seconds': silence_end if silence_end > 0 else 0, 'parameters': {'threshold_multiplier': threshold_multiplier, 'initial_window': initial_window, 'smoothing_window': smoothing_window}, 'analysis': analysis}
                with open(output, 'w') as f:
                    module.json.dump(result, f, indent=2)
                print(f'\nInitial silence detected: {silence_end}s ({silence_end / 60:.2f} min)')
                print(f'Results saved to: {output}')
        finally:
            os.chdir(cwd)
        return buffer.getvalue().rstrip('\n')

    def detect_initial_silence(self, energies: object, threshold_multiplier: object = 1.5, initial_window: object = 60, smoothing_window: object = 30) -> object:
        'Detect initial silence using energy threshold method.'
        from ._impl import _scripts_detect_silence as module
        return module.detect_initial_silence(energies=energies, threshold_multiplier=threshold_multiplier, initial_window=initial_window, smoothing_window=smoothing_window)

