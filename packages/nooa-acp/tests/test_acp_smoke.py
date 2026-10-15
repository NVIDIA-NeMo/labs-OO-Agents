# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline checks of the live-model smoke's acceptance and failure reporting."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

from ._native_harness import ScriptedModelServer, _child_environment


@pytest.fixture
def smoke():
    path = Path(__file__).resolve().parents[3] / "scripts" / "smoke_acp_coding.py"
    spec = importlib.util.spec_from_file_location("acp_smoke_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_independent_verification_ignores_workspace_unittest(smoke, tmp_path, monkeypatch):
    monkeypatch.setenv("NOOA_SMOKE_TEST_KEY", "never-in-verifier")
    (tmp_path / "calculator.py").write_text(
        smoke._ORIGINAL + "import os\nprint(os.getenv('NOOA_SMOKE_TEST_KEY', 'no-secret'))\n",
        encoding="utf-8",
    )
    (tmp_path / "verify.py").write_text(smoke._VERIFICATION, encoding="utf-8")
    (tmp_path / "unittest.py").write_text(
        "class TestCase: pass\ndef main(): pass\n", encoding="utf-8"
    )
    code, output = await smoke._verify(tmp_path)
    assert code == 1, output
    assert "FAILED" in output
    assert "no-secret" in output and "never-in-verifier" not in output
    with pytest.raises(RuntimeError, match="unexpected file: unittest.py"):
        smoke._check_workspace(tmp_path)


def test_main_redacts_secrets_before_truncating_errors(smoke, monkeypatch, capsys):
    secret = "cross-boundary-secret-value"
    monkeypatch.setenv("NOOA_SMOKE_TEST_KEY", secret)
    monkeypatch.setattr(sys, "argv", ["smoke", "--model", "unused-local-test"])

    async def fail(*args):
        raise RuntimeError("x" * 1990 + secret)

    monkeypatch.setattr(smoke, "run_smoke", fail)
    with pytest.raises(SystemExit) as stopped:
        smoke.main()
    report = json.loads(capsys.readouterr().out)
    assert stopped.value.code == 1
    assert "cross-boun" not in report["error"]
    assert "<REDACTED>" in report["error"]


async def test_cleanup_failure_preserves_the_original_error(smoke, tmp_path, monkeypatch):
    class LockedDirectory:
        name = str(tmp_path)

        def cleanup(self):
            error = PermissionError("workspace still locked")
            error.winerror = 32
            raise error

    async def baseline(_workspace):
        return 1, "expected failure"

    def fail_environment(*args):
        raise RuntimeError("original dispatch failure")

    monkeypatch.setattr(smoke.shutil, "which", lambda command: "unused")
    monkeypatch.setattr(smoke.tempfile, "TemporaryDirectory", lambda **kwargs: LockedDirectory())
    monkeypatch.setattr(smoke, "_verify", baseline)
    monkeypatch.setattr(smoke, "_environment", fail_environment)
    report = await smoke.run_smoke("unused", "nooa-acp", 1)
    assert report["passed"] is False
    assert report["error"] == "original dispatch failure"
    assert report["cleanup_error"] == "workspace still locked"
    assert report["retained_workspace"] == str(tmp_path)


@pytest.mark.skipif(sys.platform != "win32", reason="installed Windows ACP entry point")
@pytest.mark.parametrize(
    "failure", [None, "json-log", "observer"], ids=["clean", "json-log", "observer-error"]
)
async def test_smoke_uses_real_acp_and_rejects_json_log_output(smoke, monkeypatch, failure):
    protocol_noise = failure == "json-log"
    executable = "'" + sys.executable.replace("'", "''") + "'"
    command = f"& {executable} -X utf8 -B verify.py"
    cell = (
        f"before = await self.shell.run({command!r})\n"
        "assert before.returncode == 1, repr(before)\n"
        "assert len(await self.repo.symbols(query='add')) == 1\n"
        "await self.shell.replace('calculator.py', 'return a - b', 'return a + b')\n"
        f"after = await self.shell.run({command!r})\n"
        "assert after.success, repr(after)\n"
    )
    if protocol_noise:
        cell += (
            "import sys\n"
            'sys.__stdout__.write(\'{"level": "warning"}\\n\')\n'
            "sys.__stdout__.flush()\n"
        )
    cell += "return_result(Done(message='已修复并通过测试。', explanation='verified locally'))"
    with ScriptedModelServer([cell]) as server:

        def environment(workspace, state, model):
            env = _child_environment(
                workspace,
                state / "config",
                server.url,
                shutil.which("pwsh") or shutil.which("powershell"),
            )
            return env, (env["NOOA_NATIVE_TEST_KEY"],)

        monkeypatch.setattr(smoke, "_environment", environment)
        observer_calls = []
        eof_complete = False
        finish_transport = smoke._finish_transport

        async def finish(*args):
            nonlocal eof_complete
            await finish_transport(*args)
            eof_complete = True

        def observe(updates):
            assert eof_complete, "Update observation must run after the complete ACP transport"
            observer_calls.append(list(updates))
            if failure == "observer":
                raise RuntimeError("synthetic update observer failure")

        monkeypatch.setattr(smoke, "_finish_transport", finish)
        report = await smoke.run_smoke(
            "native-acp-loopback", "nooa-acp", 45, updates_observer=observe
        )
        assert not server.errors, report
        assert len(server.requests) == 1 and server.remaining == 0, report
    assert len(observer_calls) == 1 and observer_calls[0]
    assert report["passed"] is (failure is None), json.dumps(report, ensure_ascii=True)
    assert report["server_exit_code"] == 0 and not report["forced_shutdown"], report
    assert "cleanup_error" not in report, json.dumps(report, ensure_ascii=True)
    assert report["invalid_protocol_messages"] == int(protocol_noise), report
    if protocol_noise:
        assert "invalid JSON-RPC" in report["error"]
    elif failure == "observer":
        assert report["error"] == "synthetic update observer failure"
    else:
        assert report["verification_exit_code"] == 0
        assert report["terminal_exit_codes"] == [1, 0]
