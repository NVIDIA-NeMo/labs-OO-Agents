# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate installed Windows wheels without importing the source checkout.

Copy this file, smoke_acp_coding.py, and _acp_smoke_support.py together outside
the checkout. Prepare cl100k_base with tiktoken during installation, with
TIKTOKEN_CACHE_DIR set to the dedicated cache directory. Run with the wheel
venv's Python, for example:

    uv run --no-project --no-sync --python VENV/Scripts/python.exe python -I
        smoke_installed_wheels.py --tokenizer-cache TOKENIZER_CACHE --report wheel-smoke.json

Both installed commands perform an actual coding workflow through ACP and a
deterministic loopback model. No credentials, pytest, or optional CLI extras
are needed. The script does not build packages or change the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

_PACKAGES = {"nooa": "nooa", "nooa-cli": "nooa_cli", "nooa-acp": "nooa_acp"}
_WORKSPACE_NAME = "中文 repo's [literal]"
_TOKENIZER_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
_STDOUT_MARKER = "NOOA wheel smoke stdout"


def _companion(filename: str) -> ModuleType:
    """Load one adjacent support file without exposing checkout roots to Python."""
    path = Path(__file__).resolve().with_name(filename)
    spec = importlib.util.spec_from_file_location(f"_wheel_smoke_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load acceptance companion: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inside_venv(path: Path, prefix: Path, label: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(prefix) or not resolved.is_file():
        raise RuntimeError(f"{label} is not an installed file inside {prefix}: {resolved}")
    return resolved


def _distribution_evidence(name: str, module_name: str, prefix: Path) -> dict[str, str]:
    """Require wheel metadata and the import selected by Python to agree."""
    distribution = importlib.metadata.distribution(name)
    direct_url = distribution.read_text("direct_url.json")
    if direct_url and "dir_info" in json.loads(direct_url):
        raise RuntimeError(f"{name} is a source-directory/editable install; install its wheel")
    files = distribution.files
    if not files:
        raise RuntimeError(f"{name} has no wheel installation RECORD")
    recorded = {path.as_posix(): path for path in files}
    package_file = f"{module_name}/__init__.py"
    if package_file not in recorded:
        raise RuntimeError(f"{name} RECORD does not contain {package_file}")
    metadata_files = {
        path.name: path
        for path in files
        if len(path.parts) == 2 and path.parts[0].endswith(".dist-info")
    }
    if not {"WHEEL", "METADATA", "RECORD"} <= metadata_files.keys():
        raise RuntimeError(f"{name} is missing wheel installation metadata")
    metadata_path = _inside_venv(
        Path(distribution.locate_file(metadata_files["METADATA"])), prefix, f"{name} metadata"
    )
    for filename in ("WHEEL", "RECORD"):
        _inside_venv(
            Path(distribution.locate_file(metadata_files[filename])), prefix, f"{name} {filename}"
        )
    installed_path = _inside_venv(
        Path(distribution.locate_file(recorded[package_file])), prefix, f"{name} package"
    )
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"Cannot locate the installed {module_name} module")
    module_path = _inside_venv(Path(spec.origin), prefix, f"{module_name} import")
    if module_path != installed_path:
        raise RuntimeError(f"{module_name} import disagrees with {name} wheel RECORD")
    return {
        "version": distribution.version,
        "module": module_name,
        "module_path": str(module_path),
        "metadata_path": str(metadata_path),
    }


def inspect_installation() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    prefix = Path(sys.prefix).resolve()
    if prefix == Path(sys.base_prefix).resolve():
        raise RuntimeError(
            "Run this acceptance script with a separate wheel-only virtual environment"
        )
    distributions = {
        name: _distribution_evidence(name, module, prefix) for name, module in _PACKAGES.items()
    }
    scripts = Path(sysconfig.get_path("scripts"))
    commands = {
        name: str(_inside_venv(scripts / f"{name}.exe", prefix, f"{name} console script"))
        for name in ("nooa", "nooa-acp")
    }
    return distributions, commands


def _prepared_tokenizer_cache(path: Path) -> Path:
    """Fail before imports if the immutable resource was not prepared at install time."""
    cache = path.resolve()
    # tiktoken indexes downloaded resources by SHA-1 of their URL, and verifies
    # the actual content against its own pinned SHA-256 when loading the data.
    resource = cache / hashlib.sha1(_TOKENIZER_URL.encode(), usedforsecurity=False).hexdigest()
    if not resource.is_file() or resource.stat().st_size == 0:
        raise RuntimeError(
            f"Prepared cl100k_base tokenizer data is missing from {cache}. "
            "During installation, set TIKTOKEN_CACHE_DIR to this directory and run "
            "tiktoken.get_encoding('cl100k_base') with the wheel venv's Python. "
            "Wheel acceptance keeps runtime network access blocked."
        )
    return cache


@contextmanager
def _isolated_process_environment(environment: dict[str, str]) -> Iterator[None]:
    """Apply the same isolation while importing the installed smoke dependencies."""
    previous = dict(os.environ)
    os.environ.clear()
    os.environ.update(environment)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(previous)


def _check_commands(
    commands: dict[str, str], workspace: Path, environment: dict[str, str]
) -> list[dict[str, Any]]:
    checks = []
    for name, arguments, expected in (
        ("nooa", ["--help"], "acp"),
        ("nooa", ["acp", "--help"], "Serve the NOOA coding agent over ACP"),
        ("nooa-acp", ["--help"], "Serve the NOOA coding agent over ACP"),
    ):
        process = subprocess.run(
            [commands[name], *arguments],
            cwd=workspace,
            env=environment,
            capture_output=True,
            timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        stdout = process.stdout.decode("utf-8", errors="replace")
        if process.returncode != 0 or expected not in stdout:
            stderr = process.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"{name} {' '.join(arguments)} failed ({process.returncode}): {stderr[:2000]}"
            )
        checks.append({"command": [commands[name], *arguments], "exit_code": process.returncode})
    return checks


def _fenced_blocks(text: str) -> list[str]:
    return [
        match.group("body").rstrip("\n")
        for match in re.finditer(
            r"(?ms)^(?P<fence>`{3,}|~{3,})[^\n]*\n(?P<body>.*?)"
            r"^(?P=fence)[ \t]*(?=\n|\Z)",
            text.replace("\r\n", "\n"),
        )
    ]


def _command_cards(updates: list[object]) -> list[dict[str, Any]]:
    """Require shell outcomes in the standard content that clients actually render."""
    from acp.schema import (
        ContentToolCallContent,
        TextContentBlock,
        ToolCallProgress,
        ToolCallStart,
    )

    def visible_text(update: ToolCallStart | ToolCallProgress) -> str:
        content = update.content or []
        if (
            len(content) != 1
            or not isinstance(content[0], ContentToolCallContent)
            or not isinstance(content[0].content, TextContentBlock)
        ):
            raise RuntimeError("Command card is missing its standard text content")
        return content[0].content.text.replace("\r\n", "\n")

    starts = [
        update
        for update in updates
        if isinstance(update, ToolCallStart)
        and isinstance(update.raw_input, dict)
        and "command" in update.raw_input
    ]
    if len(starts) != 2 or len({start.tool_call_id for start in starts}) != 2:
        raise RuntimeError(f"Expected two distinct command cards, received {len(starts)}")
    evidence = []
    for start, exit_code in zip(starts, (1, 0), strict=True):
        if start.kind != "other" or start.title != "Running command":
            raise RuntimeError("Command card must use the generic kind and visible running title")
        command = start.raw_input["command"]
        if not isinstance(command, str) or command not in _fenced_blocks(visible_text(start)):
            raise RuntimeError("Command card start does not show the full fenced command")
        progress = [
            update
            for update in updates
            if isinstance(update, ToolCallProgress) and update.tool_call_id == start.tool_call_id
        ]
        if not progress:
            raise RuntimeError("Command card has no progress or completion update")
        for update in progress:
            if update.kind not in (None, "other"):
                raise RuntimeError("Command card switched away from the generic kind")
            if command not in _fenced_blocks(visible_text(update)):
                raise RuntimeError("Command card updates do not retain the full fenced command")
        finished = progress[-1]
        expected_status = "failed" if exit_code else "completed"
        expected_title = (
            f"Command failed (exit {exit_code})" if exit_code else "Ran command (exit 0)"
        )
        if (
            not isinstance(finished.raw_output, dict)
            or finished.raw_output.get("exit_code") != exit_code
            or finished.status != expected_status
            or finished.title != expected_title
        ):
            raise RuntimeError("Command card status and title do not match its actual exit code")
        text = visible_text(finished)
        blocks = _fenced_blocks(text)
        if "Stdout:" not in text or "Stderr:" not in text:
            raise RuntimeError("Command card does not label its stdout and stderr")
        if _STDOUT_MARKER not in blocks:
            raise RuntimeError("Command card is missing its fenced stdout")
        diagnostics = "FAILED (failures=2)" if exit_code else "OK"
        if not any(
            "Ran 4 tests" in block and diagnostics in block.splitlines() for block in blocks
        ):
            raise RuntimeError("Command card is missing its fenced stderr test diagnostics")
        if f"Exit code: {exit_code}" not in text:
            raise RuntimeError("Command card does not show its exit code in standard content")
        evidence.append(
            {
                "kind": start.kind,
                "exit_code": exit_code,
                "status": finished.status,
                "command_visible": True,
                "stdout_visible": True,
                "stderr_visible": True,
                "exit_code_visible": True,
            }
        )
    return evidence


def _workflow(distributions: dict[str, dict[str, str]]) -> str:
    # The check runs inside the real ACP process, proving its console launcher
    # selected the same interpreter and wheel imports as this acceptance runner.
    attestations = (
        "import importlib\n"
        "import importlib.metadata\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"assert Path(sys.prefix).resolve() == Path({sys.prefix!r}).resolve(), sys.prefix\n"
        f"assert Path(sys.executable).resolve() == Path({sys.executable!r}).resolve()\n"
    )
    for distribution, evidence in distributions.items():
        attestations += (
            f"module = importlib.import_module({evidence['module']!r})\n"
            f"assert Path(module.__file__).resolve() == Path({evidence['module_path']!r})\n"
            f"assert importlib.metadata.version({distribution!r}) == {evidence['version']!r}\n"
        )
    executable = "'" + sys.executable.replace("'", "''") + "'"
    command = f"Write-Output '{_STDOUT_MARKER}'; & {executable} -X utf8 -B verify.py"
    return (
        attestations
        + "assert self.shell.shell == 'powershell'\n"
        + f"before = await self.shell.run({command!r})\n"
        + "assert before.returncode == 1, repr(before)\n"
        + "definitions = await self.repo.symbols(query='add')\n"
        + "assert len(definitions) == 1, repr(definitions)\n"
        + "await self.shell.replace('calculator.py', 'return a - b', 'return a + b')\n"
        + f"after = await self.shell.run({command!r})\n"
        + "assert after.success, repr(after)\n"
        + "return_result(Done(message='已修复相加实现；四项本地测试通过。', "
        + "explanation='Installed wheel imports and native coding workflow verified.', "
        + "evidence=[after.stdout, after.stderr]))"
    )


async def run_acceptance(timeout: float, tokenizer_cache: Path | None = None) -> dict[str, Any]:
    started = time.monotonic()
    report: dict[str, Any] = {
        "passed": False,
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "venv": sys.prefix,
        "working_directory": str(Path.cwd()),
        "workspace_name": _WORKSPACE_NAME,
        "workflows": [],
    }
    try:
        if sys.platform != "win32":
            raise RuntimeError("This acceptance script exercises native Windows wheels")
        distributions, commands = inspect_installation()
        report["distributions"] = distributions
        cache = _prepared_tokenizer_cache(
            tokenizer_cache or Path(__file__).resolve().parent.parent / "tokenizer-cache"
        )
        report["tokenizer_cache"] = str(cache)
        support = _companion("_acp_smoke_support.py")
        with tempfile.TemporaryDirectory(prefix="nooa-wheel-smoke-") as directory:
            root = Path(directory).resolve()
            workspace = root / _WORKSPACE_NAME
            workspace.mkdir()
            # Imports and --help are offline too: this server rejects any HTTP.
            with support.ScriptedModelServer([]) as setup:
                environment = support.child_environment(
                    workspace, root / "config", setup.url, tokenizer_cache=cache
                )
                with _isolated_process_environment(environment):
                    smoke = _companion("smoke_acp_coding.py")
                    report["commands"] = _check_commands(commands, workspace, environment)
                    for entrypoint in ("nooa-acp", "nooa"):
                        with support.ScriptedModelServer([_workflow(distributions)]) as model:

                            def child_environment(workspace, state, model_name):
                                env = support.child_environment(
                                    workspace, state / "config", model.url, tokenizer_cache=cache
                                )
                                return env, (support._TEST_KEY,)

                            observed_updates: list[object] = []
                            flow = await smoke.run_smoke(
                                support._ALIAS,
                                entrypoint,
                                timeout,
                                environment_factory=child_environment,
                                workspace_name=_WORKSPACE_NAME,
                                executable=commands[entrypoint],
                                updates_observer=observed_updates.extend,
                            )
                            flow["command_cards_verified"] = False
                            try:
                                flow["command_cards"] = _command_cards(observed_updates)
                            except RuntimeError as error:
                                flow["command_card_error"] = str(error)
                            else:
                                flow["command_cards_verified"] = True
                            flow["model_requests"] = len(model.requests)
                            flow["remaining_model_responses"] = model.remaining
                            flow["model_errors"] = list(model.errors)
                            flow["passed"] = (
                                flow["passed"]
                                and not model.errors
                                and len(model.requests) == 1
                                and model.remaining == 0
                                and flow.get("terminal_exit_codes") == [1, 0]
                                and flow["command_cards_verified"]
                            )
                            flow["child_wheel_imports_verified"] = flow["passed"]
                            report["workflows"].append(flow)
                report["setup_network_errors"] = list(setup.errors)
            report["passed"] = (
                not report["setup_network_errors"]
                and len(report["workflows"]) == 2
                and all(flow["passed"] for flow in report["workflows"])
            )
    except Exception as error:
        report.update(passed=False, error_type=type(error).__name__, error=str(error))
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=90, help="Seconds for each ACP workflow")
    parser.add_argument(
        "--tokenizer-cache",
        type=Path,
        help="Prepared tiktoken cache directory (default: ../tokenizer-cache beside the bundle)",
    )
    parser.add_argument("--report", type=Path, help="Optional JSON result file")
    args = parser.parse_args()
    report = asyncio.run(run_acceptance(args.timeout, args.tokenizer_cache))
    serialized = json.dumps(report, ensure_ascii=True, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
