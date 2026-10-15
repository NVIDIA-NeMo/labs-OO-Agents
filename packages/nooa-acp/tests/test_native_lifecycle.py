# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native ACP cancellation and durable recovery through the installed CLI."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from contextlib import suppress
from pathlib import Path

import pytest
from acp import PROTOCOL_VERSION, text_block
from acp.schema import AgentMessageChunk, ToolCallProgress, ToolCallStart, UserMessageChunk
from nooa_cli.sessions import SessionStore

from ._native_harness import launch_native_acp

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="native Windows ACP lifecycle")


def _quote(value: str | Path) -> str:
    return "'" + str(value).replace("'", "''") + "'"


@pytest.fixture(params=["pwsh", "powershell"])
def shell_executable(request):
    executable = shutil.which(request.param)
    if executable is None:
        pytest.skip(f"{request.param} is not installed")
    return executable


class _Client:
    def __init__(self):
        self.updates: list[tuple[str, object]] = []
        self.changed = asyncio.Event()

    async def session_update(self, session_id, update, **kwargs):
        self.updates.append((session_id, update))
        self.changed.set()

    async def wait_message(self, text: str):
        async with asyncio.timeout(30):
            while True:
                self.changed.clear()
                if any(
                    isinstance(update, AgentMessageChunk) and update.content.text == text
                    for _, update in self.updates
                ):
                    return
                await self.changed.wait()


def _turns(updates):
    return [
        ("user" if isinstance(update, UserMessageChunk) else "agent", update.content.text)
        for _, update in updates
        if isinstance(update, (UserMessageChunk, AgentMessageChunk))
    ]


def _card_text(update):
    return "\n".join(
        str(getattr(getattr(item, "content", None), "text", "")) for item in (update.content or [])
    )


class _ProcessHandle:
    """Retain identity while checking exit or cleaning up a failed test."""

    def __init__(self, pid: int):
        import _winapi

        self._api = _winapi
        self._handle = _winapi.OpenProcess(0x100000 | 0x0001, False, pid)

    @property
    def alive(self):
        return self._api.WaitForSingleObject(self._handle, 0) == 258

    async def wait_dead(self):
        async with asyncio.timeout(10):
            while self.alive:
                await asyncio.sleep(0.02)

    def close(self):
        try:
            if self.alive:
                self._api.TerminateProcess(self._handle, 1)
        finally:
            self._api.CloseHandle(self._handle)


async def _read_identities(path: Path):
    async with asyncio.timeout(30):
        while not path.exists():
            await asyncio.sleep(0.02)
    return json.loads(path.read_text(encoding="utf-8"))


def _shell_turn(command: str, message: str) -> str:
    return (
        "assert self.shell.shell == 'powershell'\n"
        f"result = await self.shell.run({command!r}, timeout=30)\n"
        "assert result.success and not result.timed_out, repr(result)\n"
        f"assert {message!r} in result.stdout, repr(result)\n"
        f"return_result(Done(message={message!r}, explanation='verified native shell output'))"
    )


@pytest.mark.timeout(120)
async def test_cancel_running_process_tree_then_prompt_same_session(tmp_path, shell_executable):
    workspace = tmp_path / "工作 repo's [literal]"
    workspace.mkdir()
    identities = workspace / "running.json"
    script = workspace / "待取消.py"
    ready = "本机进程树已启动：中文输出"
    script.write_text(
        "import json,os,pathlib,subprocess,sys,time\n"
        "child=subprocess.Popen([sys._base_executable,'-c','import time;time.sleep(120)'],"
        "creationflags=subprocess.CREATE_NO_WINDOW)\n"
        "data={'shell':int(os.environ['NOOA_TEST_SHELL_PID']),"
        "'parent':os.getpid(),'grandchild':child.pid,"
        "'powershell_major':int(os.environ['NOOA_TEST_PS_MAJOR'])}\n"
        f"identities=pathlib.Path({str(identities)!r})\n"
        "pending=identities.with_suffix('.tmp')\n"
        "pending.write_text(json.dumps(data),encoding='utf-8');pending.replace(identities)\n"
        f"print({ready!r},flush=True)\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )
    command = (
        "$env:NOOA_TEST_SHELL_PID = [string]$PID;"
        "$env:NOOA_TEST_PS_MAJOR = [string]$PSVersionTable.PSVersion.Major;"
        f"& {_quote(sys.executable)} -u {_quote(script)}"
    )
    blocked_cell = (
        "from contextlib import aclosing\n"
        "captured = ''\nannounced = False\n"
        f"async with aclosing(self.shell.run_stream({command!r}, timeout=120)) as stream:\n"
        "    async for item in stream:\n"
        "        if item.kind == 'stdout':\n"
        "            captured += item.text\n"
        f"            if {ready!r} in captured and not announced:\n"
        f"                self.message({ready!r})\n"
        "                announced = True\n"
        "return_result(Done(message='UNEXPECTED SUCCESS', explanation='must be cancelled'))"
    )
    recovered = "取消后同一会话已恢复"
    recovery_cell = _shell_turn(f"[Console]::WriteLine({_quote(recovered)})", recovered)
    client = _Client()
    handles = []
    try:
        async with launch_native_acp(
            client,
            workspace,
            [blocked_cell, recovery_cell],
            shell_executable=shell_executable,
        ) as (connection, process, model):
            await connection.initialize(PROTOCOL_VERSION)
            session = await connection.new_session(str(workspace))
            prompting = asyncio.create_task(
                connection.prompt(session.session_id, [text_block("启动长任务，然后等待取消。")])
            )
            try:
                pids = await _read_identities(identities)
                for name in ("shell", "parent", "grandchild"):
                    handles.append(_ProcessHandle(pids[name]))
                expected_major = 7 if Path(shell_executable).stem == "pwsh" else 5
                assert pids["powershell_major"] == expected_major
                # ToolCallStart precedes actual subprocess startup. Require both
                # native process identities and consumed stdout before cancelling.
                await client.wait_message(ready)
                assert all(handle.alive for handle in handles)
                await connection.cancel(session.session_id)
                response = await asyncio.wait_for(prompting, timeout=30)
            finally:
                if not prompting.done():
                    with suppress(Exception):
                        await connection.cancel(session.session_id)
                        await asyncio.wait_for(asyncio.shield(prompting), timeout=10)
                    prompting.cancel()
                    await asyncio.gather(prompting, return_exceptions=True)

            assert response.stop_reason == "cancelled"
            await client.wait_message("Stopped at your request.")
            await asyncio.gather(*(handle.wait_dead() for handle in handles))
            assert process.returncode is None, "cancel must leave the ACP server running"
            cards = [update for _, update in client.updates if isinstance(update, ToolCallStart)]
            assert {card.kind for card in cards} >= {"other", "execute"}
            for card in cards:
                changes = [
                    update
                    for _, update in client.updates
                    if isinstance(update, ToolCallProgress)
                    and update.tool_call_id == card.tool_call_id
                ]
                assert changes and changes[-1].status == "failed"
                assert all(update.status != "completed" for update in changes)
                assert "Cancelled by user." in _card_text(changes[-1])
                assert "CancelledError" not in _card_text(changes[-1])
                if card.kind == "other":
                    assert changes[-1].title == "Cancelled"
            terminal = next(card for card in cards if card.kind == "execute")
            terminal_end = next(
                update
                for _, update in reversed(client.updates)
                if isinstance(update, ToolCallProgress)
                and update.tool_call_id == terminal.tool_call_id
            )
            assert ready in _card_text(terminal_end), (
                "cancelled cards retain already-written output"
            )
            assert terminal_end.raw_output["exit_code"] != 0
            assert terminal_end.raw_output["timed_out"] is False
            assert ("agent", "Stopped at your request.") in _turns(client.updates)
            assert ("agent", "UNEXPECTED SUCCESS") not in _turns(client.updates)

            result = await asyncio.wait_for(
                connection.prompt(
                    session.session_id, [text_block("继续执行，验证 shell 可以恢复。")]
                ),
                timeout=30,
            )
            assert result.stop_reason == "end_turn"
            await client.wait_message(recovered)
            assert ("agent", recovered) in _turns(client.updates)
            assert {session_id for session_id, _ in client.updates} == {session.session_id}
            assert len(model.requests) == 2 and model.remaining == 0
            await connection.close_session(session.session_id)
        assert process.returncode == 0 and not model.forced_shutdown
    finally:
        for handle in handles:
            handle.close()


def _stage_command(artifact, pidfile, value, message, *, previous=None):
    check = (
        f"if ([IO.File]::ReadAllText({_quote(artifact)}) -ne {_quote(previous)}) "
        "{ throw 'Previous native result was not preserved' };"
        if previous is not None
        else ""
    )
    return (
        check + f"[IO.File]::WriteAllText({_quote(artifact)}, {_quote(value)},"
        "[Text.UTF8Encoding]::new($false));"
        f"[IO.File]::WriteAllText({_quote(pidfile)}, [string]$PID);"
        f"[Console]::WriteLine({_quote(message)})"
    )


@pytest.mark.timeout(150)
async def test_closed_and_restarted_sessions_replay_unicode_and_continue(
    tmp_path, shell_executable
):
    workspace = tmp_path / "持久 工作区"
    workspace.mkdir()
    user_dir = tmp_path / "user config"
    artifact, pidfile = workspace / "进度.txt", workspace / "shell.pid"
    first_prompt = "会话记忆：你好 世界\n保留第二行与标点。"
    second_prompt = "关闭会话后继续：确认第一步仍在。"
    third_prompt = "重启进程后继续：确认第二步仍在。"
    messages = ["第一步已写入：中文", "重新加载后第二步完成", "重启后第三步完成"]
    values = ["步骤一·你好 世界", "步骤二·再次执行", "步骤三·恢复成功"]
    cells = [
        _shell_turn(
            _stage_command(
                artifact,
                pidfile,
                value,
                message,
                previous=values[index - 1] if index else None,
            ),
            message,
        )
        for index, (value, message) in enumerate(zip(values, messages, strict=True))
    ]
    client = _Client()
    handles = []
    try:
        async with launch_native_acp(
            client,
            workspace,
            cells[:2],
            shell_executable=shell_executable,
            user_dir=user_dir,
        ) as (connection, first_process, first_model):
            await connection.initialize(PROTOCOL_VERSION)
            session = await connection.new_session(str(workspace))
            response = await asyncio.wait_for(
                connection.prompt(session.session_id, [text_block(first_prompt)]), timeout=30
            )
            assert response.stop_reason == "end_turn"
            assert artifact.read_text(encoding="utf-8") == values[0]
            first_shell = _ProcessHandle(int(pidfile.read_text(encoding="utf-8")))
            handles.append(first_shell)
            assert first_shell.alive
            await connection.close_session(session.session_id)
            await first_shell.wait_dead()

            replay_start = len(client.updates)
            await connection.load_session(str(workspace), session.session_id)
            await client.wait_message(messages[0] + "\n")
            assert _turns(client.updates[replay_start:]) == [
                ("user", first_prompt + "\n"),
                ("agent", messages[0] + "\n"),
            ]
            response = await asyncio.wait_for(
                connection.prompt(session.session_id, [text_block(second_prompt)]), timeout=30
            )
            assert response.stop_reason == "end_turn"
            assert artifact.read_text(encoding="utf-8") == values[1]
            second_shell = _ProcessHandle(int(pidfile.read_text(encoding="utf-8")))
            handles.append(second_shell)
            assert second_shell.alive
            assert len(first_model.requests) == 2 and first_model.remaining == 0
            # Leave a live, idle session for EOF shutdown to release itself.
        assert first_process.returncode == 0 and not first_model.forced_shutdown
        await second_shell.wait_dead()

        resumed_client = _Client()
        async with launch_native_acp(
            resumed_client,
            workspace,
            cells[2:],
            entrypoint="nooa acp",
            shell_executable=shell_executable,
            user_dir=user_dir,
        ) as (connection, second_process, second_model):
            await connection.initialize(PROTOCOL_VERSION)
            listed = await connection.list_sessions(str(workspace))
            assert [item.session_id for item in listed.sessions] == [session.session_id]
            await connection.load_session(str(workspace), session.session_id)
            await resumed_client.wait_message(messages[1] + "\n")
            assert _turns(resumed_client.updates) == [
                ("user", first_prompt + "\n"),
                ("agent", messages[0] + "\n"),
                ("user", second_prompt + "\n"),
                ("agent", messages[1] + "\n"),
            ]
            response = await asyncio.wait_for(
                connection.prompt(session.session_id, [text_block(third_prompt)]), timeout=30
            )
            assert response.stop_reason == "end_turn"
            await resumed_client.wait_message(messages[2])
            assert artifact.read_text(encoding="utf-8") == values[2]
            assert ("agent", messages[2]) in _turns(resumed_client.updates)
            assert len(second_model.requests) == 1 and second_model.remaining == 0
            # Durable history must reach the newly created agent as well as the UI.
            model_context = json.dumps(second_model.requests[0]["messages"], ensure_ascii=False)
            assert first_prompt.splitlines()[0] in model_context
            assert second_prompt in model_context
            assert messages[0] in model_context and messages[1] in model_context
            await connection.close_session(session.session_id)
        assert second_process.returncode == 0 and not second_model.forced_shutdown

        store = SessionStore(workspace / ".nooa" / "sessions")
        assert [(turn.role, turn.content) for turn in store.load_turns(session.session_id)] == [
            ("user", first_prompt),
            ("agent", messages[0]),
            ("user", second_prompt),
            ("agent", messages[1]),
            ("user", third_prompt),
            ("agent", messages[2]),
        ]
        assert store.get(session.session_id).origin == "acp"
    finally:
        for handle in handles:
            handle.close()
