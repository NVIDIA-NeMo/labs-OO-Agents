from __future__ import annotations

from importlib import resources
from pathlib import Path

from nooa.skill import Skill


class EnergyCalculator(Skill):
    'Calculate per-second RMS energy from audio files. Use when you need to analyze audio volume patterns, prepare data for silence/pause detection, or create an energy profile for audio analysis tasks.\n\nPackage-native skill translated from a traditional TextSkill.\nUse the public Python methods on this skill instead of invoking scripts or subprocesses.\n\nPublic APIs:\n- calc_energy(audio: str, output: str, window_seconds: float | None = 1) -> str: returns captured text output.\n- calculate_energy(audio_path: object, window_seconds: object = 1) -> object: returns the Python value from the translated implementation.'

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


    def calc_energy(self, audio: str, output: str, window_seconds: float | None = 1) -> str:
        """Run the translated package implementation and return captured text output."""
        from ._impl import _scripts_calc_energy as module
        import contextlib
        import io
        import os
        import types
        args = types.SimpleNamespace(audio=audio, output=output, window_seconds=window_seconds)
        buffer = io.StringIO()
        cwd = os.getcwd()
        try:
            os.chdir(self._resource_root())
            with contextlib.redirect_stdout(buffer):
                print(f'Calculating energy from: {audio}')
                print(f'Window size: {window_seconds}s')
                result = module.calculate_energy(audio, window_seconds)
                with open(output, 'w') as f:
                    module.json.dump(result, f, indent=2)
                print(f"\nEnergy calculated for {result['total_seconds']}s of audio")
                print(f"Energy range: {result['stats']['min']:.1f} - {result['stats']['max']:.1f}")
                print(f"Mean energy: {result['stats']['mean']:.1f}")
                print(f'Results saved to: {output}')
        finally:
            os.chdir(cwd)
        return buffer.getvalue().rstrip('\n')

    def calculate_energy(self, audio_path: object, window_seconds: object = 1) -> object:
        'Calculate per-second RMS energy from audio file.'
        from ._impl import _scripts_calc_energy as module
        return module.calculate_energy(audio_path=audio_path, window_seconds=window_seconds)

