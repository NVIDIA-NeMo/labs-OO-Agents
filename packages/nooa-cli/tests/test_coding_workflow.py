# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real native coding tools driven through CodeAct by scripted model responses."""

from __future__ import annotations

import json
import shutil
import sys
from functools import partial
from pathlib import Path

import pytest
from nooa_cli.coding import CodingAgent
from nooa_cli.coding.activity import (
    FileEdit,
    TerminalCommandFinished,
    TerminalCommandOutput,
    TerminalCommandStarted,
)

from nooa.errors import GenerationError
from nooa.events import PythonOutput
from nooa.interactive import Done, SummarizationConfig
from nooa.tools import PowerShellSession
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="native Windows coding workflow")


def _quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


@pytest.fixture(params=["pwsh", "powershell"])
def powershell_executable(request, monkeypatch):
    executable = shutil.which(request.param)
    if executable is None:
        pytest.skip(f"{request.param} is not installed")
    # Only pin which installed host to launch; session, tools, and CodeAct
    # execution remain real. CodingAgent uses the normal automatic selection.
    monkeypatch.setattr(
        "nooa.tools._powershell_session.PowerShellSession",
        partial(PowerShellSession, executable=executable),
    )
    return executable


def _scripted_llm(cells: list[str]) -> FakeLLMClient:
    return FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id=f"workflow_{index}",
                        name="execute_python",
                        arguments=json.dumps({"code": code}),
                    )
                ],
            )
            for index, code in enumerate(cells)
        ],
        strict_exhaustion=True,
    )


@pytest.mark.timeout(60)
async def test_search_edit_verify_and_activity_through_native_codeact(
    tmp_path, powershell_executable
):
    workspace = tmp_path / "工作 repo"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    nested = workspace / "嵌套 space's [literal]"
    nested.mkdir()

    source_name = "问候 helper.py"
    source = workspace / source_name
    original = "def greet(name):\n    return '你好 ' + name\n\n\nmessage = greet('旧用户')\n"
    source.write_text(original, encoding="utf-8")
    decoy = nested / source_name
    decoy.write_text("# This file must remain untouched.\n", encoding="utf-8")

    test_name = "验证 tests.py"
    (workspace / test_name).write_text(
        "import os\n"
        "import runpy\n"
        "import unittest\n"
        "from pathlib import Path\n\n"
        "class MessageTests(unittest.TestCase):\n"
        "    def test_message(self):\n"
        "        self.assertEqual(os.environ['NOOA_WORKFLOW_PROBE'], 'kept')\n"
        f"        namespace = runpy.run_path(str(Path(__file__).with_name({source_name!r})))\n"
        "        self.assertEqual(namespace['message'], '你好 世界')\n"
        "        print('验证完成：你好 世界')\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    # Child Python writes UTF-8 to the terminal independently of the parent's
    # file encoding, including when this suite is run with -X utf8=0.
    test_command = f"& {_quote(sys.executable)} -X utf8 -B {_quote(test_name)}"
    before_command = "$env:NOOA_WORKFLOW_PROBE = 'kept'\n" + test_command
    move_command = f"Set-Location -LiteralPath {_quote(nested)}\n(Get-Location).ProviderPath"
    replacement = "message = greet('世界')\n"
    cells = [
        "definitions = await self.repo.symbols(query='greet')\n"
        "references = await self.repo.refs('greet')\n"
        "assert len(definitions) == 1, str(definitions)\n"
        "assert len(references) == 1, str(references)\n"
        "print(definitions)\n"
        "print(references)\n"
        f"before = await self.shell.run({before_command!r})\n"
        "assert before.returncode == 1, repr(before)\n"
        "print(before)",
        f"moved = await self.shell.run({move_command!r})\n"
        "assert moved.success, repr(moved)\n"
        "print(moved)",
        # The repository remains rooted in the original workspace; the saved
        # Match still identifies its source even with a same-named file in cwd.
        "definitions_after_cd = await self.repo.symbols(query='greet')\n"
        "assert len(definitions_after_cd) == 1, str(definitions_after_cd)\n"
        "assert definitions_after_cd[0].resolved_path == definitions[0].resolved_path\n"
        f"print(await self.shell.replace(references[0], {replacement!r}))",
        f"checks = await self.shell.run({test_command!r}, cwd='..')\nprint(checks)",
        "assert checks.success, repr(checks)\n"
        "assert '验证完成：你好 世界' in checks.stdout\n"
        "return_result(Done(message='已修复问候并通过本地测试。', "
        "explanation='verified greeting fix', "
        "evidence=[checks.stdout.strip(), checks.stderr.strip()]))",
    ]
    llm = _scripted_llm(cells)
    agent = CodingAgent(
        llm=llm,
        cwd=workspace,
        libs_dir=tmp_path / "libs",
        summarization=SummarizationConfig(policy="none"),
    )
    events = []
    agent.event_manager.on("*", events.append)
    try:
        assert isinstance(agent.shell.session, PowerShellSession)
        assert agent.shell.shell == "powershell"
        assert agent.shell.session is agent._base_shell.session is agent.repo.session

        try:
            result = await agent.handle({"user_messages": ["让问候指向世界，并运行测试验证。"]})
        except GenerationError:
            errors = [
                event.error for event in events if isinstance(event, PythonOutput) and event.error
            ]
            if errors:
                pytest.fail("CodeAct workflow failed:\n" + "\n\n".join(errors))
            raise

        assert isinstance(result, Done)
        assert result.message == "已修复问候并通过本地测试。"
        assert "Ran 1 test" in "\n".join(result.evidence)
        assert "OK" in "\n".join(result.evidence)
        assert llm.call_count == len(cells)
        assert llm.remaining_responses == 0
        assert source.read_text(encoding="utf-8") == original.replace("旧用户", "世界")
        assert decoy.read_text(encoding="utf-8") == "# This file must remain untouched.\n"
        assert agent.repo.root == workspace
        assert agent.shell.cwd == agent.shell.session.cwd == nested

        started = [event for event in events if isinstance(event, TerminalCommandStarted)]
        finished = [event for event in events if isinstance(event, TerminalCommandFinished)]
        assert [event.working_directory for event in started] == [str(workspace)] * 3
        assert [event.exit_code for event in finished] == [1, 0, 0]
        assert [event.command_id for event in finished] == [event.command_id for event in started]
        assert len({event.command_id for event in started}) == 3
        assert all(
            not event.error and not event.timed_out and not event.cancelled for event in finished
        )

        [edit] = [event for event in events if isinstance(event, FileEdit)]
        assert edit.path == str(source)
        assert edit.old_text == "message = greet('旧用户')\n"
        assert edit.new_text == replacement
        assert "-message = greet('旧用户')" in edit.diff
        assert "+message = greet('世界')" in edit.diff
        assert edit.content_complete and edit.diff_complete
        assert events.index(edit) < events.index(started[-1]) < events.index(finished[-1])

        [output] = [
            event
            for event in events
            if isinstance(event, TerminalCommandOutput)
            and event.command_id == finished[-1].command_id
        ]
        assert "验证完成：你好 世界" in output.stdout
        assert "Ran 1 test" in output.stderr
        assert "OK" in output.stderr
        assert not output.truncated
    finally:
        await agent.close()
