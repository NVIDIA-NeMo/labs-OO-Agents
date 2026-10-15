# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native CLI-to-ACP coding workflow with a real loopback model-client transport."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from acp import PROTOCOL_VERSION, text_block
from acp.schema import (
    AgentMessageChunk,
    ContentToolCallContent,
    FileEditToolCallContent,
    TextContentBlock,
    ToolCallProgress,
    ToolCallStart,
)

from ._native_harness import launch_native_acp

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="native Windows ACP workflow")


class _RecordingClient:
    def __init__(self) -> None:
        self.updates: list[tuple[str, Any]] = []

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append((session_id, update))


def _quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


@pytest.fixture(params=["pwsh", "powershell"])
def native_shell(request):
    executable = shutil.which(request.param)
    if executable is None:
        pytest.skip(f"{request.param} is not installed")
    return executable, 7 if request.param == "pwsh" else 5


@pytest.mark.parametrize("entrypoint", ["nooa-acp", "nooa acp"])
@pytest.mark.timeout(90)
async def test_installed_entrypoint_reports_real_native_edits_and_checks(
    tmp_path, entrypoint, native_shell
):
    executable, major_version = native_shell
    workspace = tmp_path / "原生 repo's [literal]"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    nested = workspace / "子目录 space's [literal]"
    nested.mkdir()
    source_name = "问候 helper.py"
    source = workspace / source_name
    original = "def greet(name):\n    return '你好 ' + name\n\n\nmessage = greet('旧用户')\n"
    source.write_text(original, encoding="utf-8")
    decoy = nested / source_name
    decoy.write_text("# Leave the same-named file in the new cwd untouched.\n", encoding="utf-8")
    test_name = "验证 tests.py"
    (workspace / test_name).write_text(
        "import os\n"
        "import runpy\n"
        "import sys\n"
        "import unittest\n"
        "from pathlib import Path\n\n"
        "class MessageTests(unittest.TestCase):\n"
        "    def test_message(self):\n"
        "        self.assertEqual(os.environ['NOOA_ACP_WORKFLOW_PROBE'], 'kept')\n"
        f"        namespace = runpy.run_path(str(Path(__file__).with_name({source_name!r})))\n"
        "        self.assertEqual(namespace['message'], '你好 世界')\n"
        "        print('标准输出：你好 世界')\n"
        "        print('标准错误：校验通过', file=sys.stderr)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n",
        encoding="utf-8",
    )
    test_command = f"& {_quote(sys.executable)} -X utf8 -B {_quote(test_name)}"
    before_command = "$env:NOOA_ACP_WORKFLOW_PROBE = 'kept'\n" + test_command
    move_command = f"Set-Location -LiteralPath {_quote(nested)}\n(Get-Location).ProviderPath"
    replacement = "message = greet('世界')\n"
    message = "已修复问候；本地测试通过。"
    cells = [
        "assert self.shell.shell == 'powershell'\n"
        "version = await self.shell.run('$PSVersionTable.PSVersion.Major')\n"
        f"assert int(version.stdout) == {major_version}, repr(version)\n"
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
        "definitions_after_cd = await self.repo.symbols(query='greet')\n"
        "assert len(definitions_after_cd) == 1, str(definitions_after_cd)\n"
        "assert definitions_after_cd[0].resolved_path == definitions[0].resolved_path\n"
        f"print(await self.shell.replace(references[0], {replacement!r}))",
        f"checks = await self.shell.run({test_command!r}, cwd='..')\nprint(checks)",
        "assert checks.success, repr(checks)\n"
        "assert '标准输出：你好 世界' in checks.stdout\n"
        "assert '标准错误：校验通过' in checks.stderr\n"
        f"return_result(Done(message={message!r}, explanation='verified native fix', "
        "evidence=[checks.stdout, checks.stderr]))",
    ]
    client = _RecordingClient()
    async with launch_native_acp(
        client,
        workspace,
        cells,
        entrypoint=entrypoint,
        shell_executable=executable,
        user_dir=tmp_path / "isolated-user",
    ) as (connection, process, model):
        initialized = await asyncio.wait_for(connection.initialize(PROTOCOL_VERSION), timeout=30)
        assert initialized.agent_info is not None
        assert initialized.agent_info.name == "nooa-acp"
        session = await asyncio.wait_for(connection.new_session(str(workspace)), timeout=30)
        response = await asyncio.wait_for(
            connection.prompt(session.session_id, [text_block("让问候指向世界，并运行测试验证。")]),
            timeout=45,
        )

        assert response.stop_reason == "end_turn"
        assert {session_id for session_id, _ in client.updates} == {session.session_id}
        assert source.read_text(encoding="utf-8") == original.replace("旧用户", "世界")
        assert decoy.read_text(encoding="utf-8").startswith("# Leave the same-named file")
        updates = [update for _, update in client.updates]
        assert any(
            isinstance(update, AgentMessageChunk) and update.content.text == message
            for update in updates
        )

        terminals = [
            update
            for update in updates
            if isinstance(update, ToolCallStart)
            and isinstance(update.raw_input, dict)
            and "command" in update.raw_input
        ]
        assert all(update.kind == "other" for update in terminals)
        terminal_ids = [update.tool_call_id for update in terminals]
        assert len(terminal_ids) == len(set(terminal_ids)) == 4
        assert all(update.raw_input["working_directory"] == str(workspace) for update in terminals)
        finished = [
            update
            for update in updates
            if isinstance(update, ToolCallProgress)
            and update.tool_call_id in terminal_ids
            and update.status in {"completed", "failed"}
        ]
        assert [update.tool_call_id for update in finished] == terminal_ids
        assert [update.status for update in finished] == [
            "completed",
            "failed",
            "completed",
            "completed",
        ]
        assert [update.raw_output["exit_code"] for update in finished] == [0, 1, 0, 0]
        assert all(
            not update.raw_output["timed_out"] and not update.raw_output["output_truncated"]
            for update in finished
        )
        final_content = finished[-1].content[0]
        assert isinstance(final_content, ContentToolCallContent)
        assert isinstance(final_content.content, TextContentBlock)
        assert "标准输出：你好 世界" in final_content.content.text
        assert "标准错误：校验通过" in final_content.content.text
        assert "Ran 1 test" in final_content.content.text
        assert "OK" in final_content.content.text

        [edit] = [
            update
            for update in updates
            if isinstance(update, ToolCallStart) and update.kind == "edit"
        ]
        assert edit.status == "completed"
        assert edit.locations[0].path == str(source)
        assert edit.locations[0].line == 4
        [diff] = edit.content
        assert isinstance(diff, FileEditToolCallContent)
        assert diff.path == str(source)
        assert diff.old_text == "message = greet('旧用户')\n"
        assert diff.new_text == replacement
        assert updates.index(finished[1]) < updates.index(edit) < updates.index(terminals[-1])

        assert len(model.requests) == len(cells)
        assert model.remaining == 0
        assert all(request["model"] == "nooa-native-test" for request in model.requests)
        assert "Shell command language: powershell." in json.dumps(model.requests[0])
        assert {tool["function"]["name"] for tool in model.requests[0]["tools"]} == {
            "execute_python",
            "return_result",
        }

    assert process.returncode == 0
    assert not model.forced_shutdown
    assert not model.errors
