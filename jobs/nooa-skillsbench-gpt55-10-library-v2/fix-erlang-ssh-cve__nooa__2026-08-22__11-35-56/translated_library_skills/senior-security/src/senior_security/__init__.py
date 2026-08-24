from __future__ import annotations

from importlib import resources
from pathlib import Path

from nooa.skill import Skill


class SeniorSecurity(Skill):
    'Comprehensive security engineering skill for application security, penetration testing, security architecture, and compliance auditing. Includes security assessment tools, threat modeling, crypto implementation, and security automation. Use when designing security architecture, conducting penetration tests, implementing cryptography, or performing security audits.\n\nPackage-native skill translated from a traditional TextSkill.\nUse the public Python methods on this skill instead of invoking scripts or subprocesses.\n\nPublic APIs:\n- pentest_automator(target: str, verbose: bool = False, json: bool = False, output: str | None = None) -> str: returns captured text output.\n- security_auditor(target: str, verbose: bool = False, json: bool = False, output: str | None = None) -> str: returns captured text output.\n- threat_modeler(target: str, verbose: bool = False, json: bool = False, output: str | None = None) -> str: returns captured text output.'

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


    def pentest_automator(self, target: str, verbose: bool = False, json: bool = False, output: str | None = None) -> str:
        """Run the translated package implementation and return captured text output."""
        from ._impl import _scripts_pentest_automator as module
        import contextlib
        import io
        import os
        import types
        args = types.SimpleNamespace(target=target, verbose=verbose, json=json, output=output)
        buffer = io.StringIO()
        cwd = os.getcwd()
        try:
            os.chdir(self._resource_root())
            with contextlib.redirect_stdout(buffer):
                tool = module.PentestAutomator(target, verbose=verbose)
                results = tool.run()
                if json:
                    output = json.dumps(results, indent=2)
                    if output:
                        with open(output, 'w') as f:
                            f.write(output)
                        print(f'Results written to {output}')
                    else:
                        print(output)
        finally:
            os.chdir(cwd)
        return buffer.getvalue().rstrip('\n')

    def security_auditor(self, target: str, verbose: bool = False, json: bool = False, output: str | None = None) -> str:
        """Run the translated package implementation and return captured text output."""
        from ._impl import _scripts_security_auditor as module
        import contextlib
        import io
        import os
        import types
        args = types.SimpleNamespace(target=target, verbose=verbose, json=json, output=output)
        buffer = io.StringIO()
        cwd = os.getcwd()
        try:
            os.chdir(self._resource_root())
            with contextlib.redirect_stdout(buffer):
                tool = module.SecurityAuditor(target, verbose=verbose)
                results = tool.run()
                if json:
                    output = json.dumps(results, indent=2)
                    if output:
                        with open(output, 'w') as f:
                            f.write(output)
                        print(f'Results written to {output}')
                    else:
                        print(output)
        finally:
            os.chdir(cwd)
        return buffer.getvalue().rstrip('\n')

    def threat_modeler(self, target: str, verbose: bool = False, json: bool = False, output: str | None = None) -> str:
        """Run the translated package implementation and return captured text output."""
        from ._impl import _scripts_threat_modeler as module
        import contextlib
        import io
        import os
        import types
        args = types.SimpleNamespace(target=target, verbose=verbose, json=json, output=output)
        buffer = io.StringIO()
        cwd = os.getcwd()
        try:
            os.chdir(self._resource_root())
            with contextlib.redirect_stdout(buffer):
                tool = module.ThreatModeler(target, verbose=verbose)
                results = tool.run()
                if json:
                    output = json.dumps(results, indent=2)
                    if output:
                        with open(output, 'w') as f:
                            f.write(output)
                        print(f'Results written to {output}')
                    else:
                        print(output)
        finally:
            os.chdir(cwd)
        return buffer.getvalue().rstrip('\n')

