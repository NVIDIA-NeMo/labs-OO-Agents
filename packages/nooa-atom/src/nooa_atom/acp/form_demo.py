# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Human-driven native-form demo: real Atom/Session/ACP, strictly fake provider.

Run as ``python -m nooa_atom.acp.form_demo``. This deliberately uses one process,
not the production router. No client workspace or forwarded MCP tools are used.
The guards are defense in depth for this fixed script, not an OS sandbox for
arbitrary Python, malicious dependencies, or an untrusted ACP peer.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import sys
import tempfile
from importlib.metadata import entry_points
from pathlib import Path
from unittest.mock import patch

_FORM_CELL = """
return_result(NeedInputForm(
    heading="FAKE provider — native form demo",
    reason="No model calls or external tools. Enter only harmless demo values.",
    questions=[
        TextQuestion(id="name", label="Name", required=False, max_length=80),
        IntegerQuestion(id="count", label="Count", minimum=1, maximum=5, default=2),
        NumberQuestion(id="ratio", label="Ratio", minimum=0, maximum=1, default=0.5),
        BooleanQuestion(id="enabled", label="Enabled", default=False),
        ChoiceQuestion(id="target", label="Target", default="dev", choices=[
            FormChoice(value="dev", title="Development"),
            FormChoice(value="prod", title="Production"),
        ]),
        MultiChoiceQuestion(id="tags", label="Tags", required=False,
            default=["fast"], max_items=3, choices=[
                FormChoice(value="fast", title="Fast"),
                FormChoice(value="safe", title="Safe"),
            ]),
    ],
))
"""

_SUMMARY_CELL = r"""
import json
answer = notification["user_messages"][0]
if isinstance(answer, FormResponse):
    content = answer.content
    types = None if content is None else {
        key: ("list[str]" if isinstance(value, list) else type(value).__name__)
        for key, value in content.items()
    }
    summary = {
        "action": answer.action,
        "content": content,
        "native_types": types,
        "optional_omissions": (
            [key for key in ("name", "tags") if key not in content]
            if content is not None else None
        ),
    }
    text = "Native FormResponse (no generated Other keys):"
else:
    summary = {"raw_input": answer, "native_form_response": False}
    text = "Plain-text fallback — raw input, not coerced into FormResponse:"
self.message(text + "\n\n```json\n" + json.dumps(summary, ensure_ascii=False, indent=2)
             + "\n```\n\nStart a fresh external-agent thread for the next scenario.")
return_result(Done(explanation="fake form demo complete"))
"""

# Atom eagerly loads installed skill entry points. Fail closed BEFORE constructing
# it if anything beyond the workspace's known built-ins is installed. memory/web
# are skipped by Atom itself; no project skill discovery/configuration runs here.
_BUILTIN_SKILLS = {
    "nemo.shell": "nooa.tools.shell_tools:ShellTools",
    "nemo.todo": "nooa.tools.todo:TodoManager",
    "nemo.context": "nooa.runtime.context:ContextApi",
    "nemo.events": "nooa.runtime.events:EventsApi",
    "nemo.producers": "nooa.runtime.producers_skill:ProducersSkill",
    "nemo.web": "nooa.tools.web_publisher:WebPublisher",
    "nemo.libwriting": "nooa.tools.library_writing_lib:SkillWriting",
    "nemo.methodwriting": "nooa.tools.method_writing_lib:MethodWriting",
    "nemo.repo": "nooa_cli.tools.repo_tools:RepoTools",
    "nemo.memory": "nooa_memory.memory_skill:MemorySkill",
}


def _guard(event: str, args: tuple) -> None:
    """Process-local stdlib audit guard; never installed in production entrypoints."""
    if event in {"subprocess.Popen", "os.system", "os.posix_spawn", "os.spawn"}:
        raise PermissionError("Form demo prohibits subprocesses/external tools")
    if event == "socket.__new__" and args[1] in (socket.AF_INET, socket.AF_INET6):
        raise PermissionError("Form demo prohibits network sockets (including loopback)")
    if event in {"socket.connect", "socket.bind", "socket.sendto", "socket.getaddrinfo"}:
        raise PermissionError("Form demo prohibits network/connected external tools")


def _isolate(root: Path) -> Path:
    """Use only a freshly allocated, demo-owned tree; accept no state path argument."""
    # Do not carry keys, model selectors, trace destinations, shell init or config
    # overrides into imports. PATH is only for platform/library discovery; launches
    # are prohibited. Python sources have already been resolved by the interpreter.
    keep = {key: os.environ[key] for key in ("PATH", "SYSTEMROOT") if key in os.environ}
    os.environ.clear()
    os.environ.update(keep)
    workspace = root / "workspace"
    for name in ("home", "user", "cache", "config", "data", "tmp", "workspace"):
        (root / name).mkdir(mode=0o700)
    os.environ.update(
        {
            "HOME": str(root / "home"),
            "NEMO_OO_USER_DIR": str(root / "user"),
            "NEMO_OO_PROJECT_DIR": str(workspace / ".nooa"),
            "XDG_CACHE_HOME": str(root / "cache"),
            "XDG_CONFIG_HOME": str(root / "config"),
            "XDG_DATA_HOME": str(root / "data"),
            "TMPDIR": str(root / "tmp"),
            "TIKTOKEN_CACHE_DIR": str(root / "cache"),
            "LITELLM_LOCAL_MODEL_COST_MAP": "True",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    sys.dont_write_bytecode = True
    os.chdir(workspace)
    sys.addaudithook(_guard)
    return workspace


def main() -> None:
    """Start the shipped demo, without nooa CLI secrets/config/provider bootstrap."""
    if len(sys.argv) != 1:
        raise SystemExit("form_demo takes no arguments; it is a single-process stdio demo")
    with tempfile.TemporaryDirectory(prefix="nooa-form-demo-") as state:
        root = Path(state)
        workspace = _isolate(root)
        # This light import reserves stdout BEFORE any framework/library imports.
        from nooa_atom.acp.cli import reserve_stdio_for_acp

        input_fd, output_fd = reserve_stdio_for_acp()
        for ep in entry_points(group="nooa.skills"):
            if _BUILTIN_SKILLS.get(ep.name) != ep.value:
                raise RuntimeError(
                    "Form demo refuses non-builtin installed skills; use locked setup"
                )

        import nooa.agent as agent_module

        agent_module._auto_tracing_attempted = True
        # InteractiveAgent's module-level default eagerly constructs a real LLM.
        # Disable ONLY that import-time bootstrap in this demo process. The module's
        # existing exception path creates a fake default; every real session below
        # receives its own strict scripted fake. No production server/SDK patch.
        with patch(
            "nooa.unifiedllm.get_llm_client", side_effect=RuntimeError("demo: no providers")
        ):
            from nooa.interactive import SummarizationConfig
            from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
            from nooa_atom.acp.server import AtomACPAgent
            from nooa_atom.agent.agent import AtomAgent
            from nooa_atom.session.registry import SessionRegistry

        from acp import run_agent

        from nooa_atom.acp.protocol import open_stdio

        def cell(code: str, index: int) -> LLMResponse:
            return LLMResponse(
                raw_response=None,
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCall(
                        id=f"demo_{index}", name="python_cell", arguments=json.dumps({"code": code})
                    )
                ],
            )

        def fake_factory(alias, cwd):
            return FakeLLMClient(
                [cell(_FORM_CELL, 1), cell(_SUMMARY_CELL, 2)], strict_exhaustion=True
            )

        def agent_factory(options, storage):
            # Deliberately bypass workspace settings/skill configuration, not Atom
            # or Session execution. Built-in shell is lazy and NEVER called by script.
            return AtomAgent(
                llm=options.llm,
                storage=storage,
                cwd=workspace,
                skills_dirs=[],
                libs_dir=workspace / ".nooa" / "libs",
                summarization=SummarizationConfig(policy="none"),
            )

        class DemoAdapter(AtomACPAgent):
            """Private workspace and no MCP, on the unmodified real form adapter."""

            async def new_session(self, cwd, additional_directories=None, mcp_servers=None, **kw):
                return await super().new_session(str(workspace), mcp_servers=mcp_servers, **kw)

            async def load_session(self, cwd, session_id, **kw):
                kw.pop("additional_directories", None)
                return await super().load_session(str(workspace), session_id, **kw)

            async def list_sessions(self, cwd=None, **kw):
                return await super().list_sessions(str(workspace), **kw)

            def registry_for(self, cwd):
                return super().registry_for(workspace)

            def _log_llm_config(self, cwd):
                pass  # No workspace model discovery, even for diagnostics.

            def _config_options(self, session):
                return None  # Never enumerate real model aliases/config files.

            async def _prepare_tools(self, session, mcp_servers):
                # Neither session.prepare_tools nor _create_mcp_tools is called.
                return ["FAKE demo: client workspace ignored; external MCP/tools disabled."]

        adapter = DemoAdapter(
            lambda store: SessionRegistry(
                store, agent_factory=agent_factory, llm_factory=fake_factory
            ),
            sessions_dir=root / "sessions",
            agent_spec="nooa_atom.agent:AtomAgent",
            model="fake",
        )

        async def serve_demo():
            loop = asyncio.get_running_loop()
            serving = asyncio.current_task()
            loop.add_signal_handler(signal.SIGTERM, serving.cancel)
            try:
                reader, writer = await open_stdio(input_fd, output_fd)
                await run_agent(
                    adapter, input_stream=writer, output_stream=reader, use_unstable_protocol=True
                )
            except asyncio.CancelledError:
                pass
            finally:
                await adapter.close()
                loop.remove_signal_handler(signal.SIGTERM)

        print(
            "FAKE form demo: private scratch workspace; no external tools or network.",
            file=sys.stderr,
        )
        asyncio.run(serve_demo())


if __name__ == "__main__":
    main()
