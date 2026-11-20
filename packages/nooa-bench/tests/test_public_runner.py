# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark runner reaches the real public SDK path and releases its owner."""

import pytest
from nooa_bench import runner

from nooa import Agent
from tests.runtime.public_llm_fixtures import native_models as native_models
from tests.runtime.test_public_llm_factory_calls import generate


@pytest.mark.parametrize("constructor_failure", [False, True])
async def test_runner_owned_public_view_native_http_and_constructor_cleanup(
    native_models, monkeypatch, tmp_path, constructor_failure
):
    root = native_models.make()
    requested = []

    def loader(model, **settings):
        requested.append((model, settings))
        return root

    class RuntimeAgent(Agent):
        def __init__(self, **kw):
            if constructor_failure:
                raise RuntimeError("constructor sentinel")
            super().__init__(**kw)

        async def _run_evaluation(self, task_input):
            await generate(self, max_tokens=250)
            return {"success": True, "response": "native wire"}

        async def close(self):
            await self.aclose()

    monkeypatch.setattr("nooa.unifiedllm.load_llm", loader)
    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(runner, "_import_agent_class", lambda _: RuntimeAgent)
    monkeypatch.setattr(runner, "_write_answer", lambda _: None)
    if constructor_failure:
        with pytest.raises(RuntimeError, match="constructor sentinel"):
            await runner._run("task", "configured", "bench", None)
        assert not native_models.seen
    else:
        assert await runner._run("task", "configured", "bench", None) == 0
        assert native_models.seen[0][1]["max_completion_tokens"] == 250
    assert requested[0][0] == "configured"
    assert root._closed
    assert native_models.sdks[0].max_retries == 9
    assert not native_models.http[0].is_closed


class _FourthRunnerControl(BaseException):
    pass


def _fourth_runner_alias(native_models, monkeypatch):
    from nooa.unifiedllm import registry

    baseline = native_models.make()
    monkeypatch.setitem(
        registry.MODELS,
        "fourth-runner",
        {
            "adapter": "openai-chat",
            "model": "fourth-owned-model",
            "api_key": "fixture-key",
            "base_url": "https://fixture.invalid/v1",
            "client": native_models.sdks[-1],
            "context_window": 12800,
            "max_tokens": 100,
        },
    )
    return baseline


@pytest.mark.parametrize("primary_phase", ["none", "constructor", "evaluation", "result"])
@pytest.mark.parametrize("cleanup_kind", ["ordinary", "cancel", "custom", "systemexit"])
async def test_runner_fourth_real_alias_primary_identity_and_both_stage_controls(
    native_models, monkeypatch, tmp_path, primary_phase, cleanup_kind
):
    import asyncio

    from nooa import no_trace

    baseline = _fourth_runner_alias(native_models, monkeypatch)
    primary = RuntimeError("original benchmark failure")
    secondary = {
        "ordinary": RuntimeError,
        "cancel": asyncio.CancelledError,
        "custom": _FourthRunnerControl,
        "systemexit": SystemExit,
    }[cleanup_kind]("agent cleanup failure")
    model_secondary = (
        _FourthRunnerControl("later model cleanup failure")
        if cleanup_kind != "ordinary"
        else RuntimeError("ordinary model cleanup failure")
    )
    clients, agents, stages = [], [], []

    def install_model_gate(client):
        actual = client.aclose

        async def dispose():
            stages.append("model")
            monkeypatch.setattr(client, "aclose", actual)
            await actual()
            raise model_secondary

        monkeypatch.setattr(client, "aclose", dispose)

    actual_write_result = runner._write_result

    def write_result(*args, **kwargs):
        if primary_phase == "result":
            raise primary
        return actual_write_result(*args, **kwargs)

    monkeypatch.setattr(runner, "_write_result", write_result)

    class RuntimeAgent(Agent):
        def __init__(self, **kwargs):
            client = kwargs["llm"]
            clients.append(client)
            if primary_phase == "constructor":
                install_model_gate(client)
                raise primary
            super().__init__(**kwargs)
            agents.append(self)

        async def _run_evaluation(self, task_input):
            await generate(self, max_tokens=250)
            if primary_phase == "evaluation":
                raise primary
            return {"success": True, "response": "native wire"}

        @no_trace
        async def close(self):
            stages.append("agent")
            await self.aclose()
            install_model_gate(clients[0])
            raise secondary

    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(runner, "_import_agent_class", lambda _: RuntimeAgent)
    monkeypatch.setattr(runner, "_write_answer", lambda _: None)
    if primary_phase != "none":
        with pytest.raises(RuntimeError) as caught:
            await runner._run("task", "fourth-runner", "bench", None)
        assert caught.value is primary
    elif cleanup_kind == "ordinary":
        assert await runner._run("task", "fourth-runner", "bench", None) == 0
    else:
        with pytest.raises(BaseException) as caught:
            await runner._run("task", "fourth-runner", "bench", None)
        assert caught.value is secondary
    assert stages == (["model"] if primary_phase == "constructor" else ["agent", "model"])
    assert clients[0]._closed and clients[0]._lease.leases == 0
    assert clients[0]._lease is not baseline._lease
    assert not baseline._closed and baseline._lease.leases == 1
    assert all(agent.llms["main"]._references == 0 for agent in agents)
    assert len(native_models.seen) == (0 if primary_phase == "constructor" else 1)
    if native_models.seen:
        assert native_models.seen[-1][1]["max_completion_tokens"] == 250
    assert native_models.sdks[-1].max_retries == 9 and not native_models.http[-1].is_closed


@pytest.mark.parametrize("primary_kind", ["cancel", "custom", "systemexit"])
async def test_runner_fourth_active_primary_control_original_object(
    native_models, monkeypatch, tmp_path, primary_kind
):
    import asyncio

    from nooa import no_trace

    baseline = _fourth_runner_alias(native_models, monkeypatch)
    primary = {
        "cancel": asyncio.CancelledError,
        "custom": _FourthRunnerControl,
        "systemexit": SystemExit,
    }[primary_kind]("primary control")
    clients = []

    class RuntimeAgent(Agent):
        def __init__(self, **kwargs):
            clients.append(kwargs["llm"])
            super().__init__(**kwargs)

        async def _run_evaluation(self, task_input):
            await generate(self, max_tokens=250)
            raise primary

        @no_trace
        async def close(self):
            await self.aclose()
            raise asyncio.CancelledError("secondary close cancellation")

    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(runner, "_import_agent_class", lambda _: RuntimeAgent)
    monkeypatch.setattr(runner, "_write_answer", lambda _: None)
    with pytest.raises(BaseException) as caught:
        await runner._run("task", "fourth-runner", "bench", None)
    assert caught.value is primary
    assert clients[0]._closed and clients[0]._lease.leases == 0
    assert not baseline._closed and baseline._lease.leases == 1
    assert native_models.seen[-1][1]["max_completion_tokens"] == 250


@pytest.mark.parametrize("active_primary", [False, True])
async def test_runner_fourth_repeated_caller_cancel_joins_both_stages_before_precedence(
    native_models, monkeypatch, tmp_path, active_primary
):
    import asyncio

    from nooa import no_trace

    baseline = _fourth_runner_alias(native_models, monkeypatch)
    primary = RuntimeError("evaluation original")
    secondary = _FourthRunnerControl("secondary model control")
    entered, release = asyncio.Event(), asyncio.Event()
    clients, stages = [], []

    def install_model_gate(client):
        actual = client.aclose

        async def gated():
            entered.set()
            await release.wait()
            monkeypatch.setattr(client, "aclose", actual)
            await actual()
            stages.append("model")
            raise secondary

        monkeypatch.setattr(client, "aclose", gated)

    class RuntimeAgent(Agent):
        def __init__(self, **kwargs):
            client = kwargs["llm"]
            clients.append(client)
            super().__init__(**kwargs)

        async def _run_evaluation(self, task_input):
            await generate(self, max_tokens=250)
            if active_primary:
                raise primary
            return {"success": False}

        @no_trace
        async def close(self):
            await self.aclose()
            stages.append("agent")
            install_model_gate(clients[0])
            raise asyncio.CancelledError("secondary agent control")

    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(runner, "_import_agent_class", lambda _: RuntimeAgent)
    monkeypatch.setattr(runner, "_write_answer", lambda _: None)
    caller = asyncio.create_task(runner._run("task", "fourth-runner", "bench", None))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        caller.cancel("first cleanup join cancellation")
        await asyncio.sleep(0)
        caller.cancel("later cleanup join cancellation")
        await asyncio.sleep(0)
        assert not caller.done() and not clients[0]._closed
        release.set()
        if active_primary:
            with pytest.raises(RuntimeError) as caught:
                await caller
            assert caught.value is primary
        else:
            with pytest.raises(asyncio.CancelledError, match="first cleanup join cancellation"):
                await caller
        assert stages == ["agent", "model"]
        assert clients[0]._closed and clients[0]._lease.leases == 0
        assert not baseline._closed and baseline._lease.leases == 1
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
