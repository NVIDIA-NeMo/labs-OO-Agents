# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded pattern validation without blocking external answer paths."""

import asyncio
import subprocess
import time

import pytest

import nooa.forms as forms
from nooa.forms import FormResponse, NeedInputForm, TextQuestion


def test_pattern_defaults_nesting_and_reconstruction_reuse_immutable_results(monkeypatch):
    forms._PATTERN_CACHE.clear()
    calls = []
    original = subprocess.run

    def run(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    q = TextQuestion(id="q", label="Q", pattern="^cached-test$", default="cached-test")
    need = NeedInputForm(heading="H", questions=[q])
    NeedInputForm.model_validate_json(need.model_dump_json())
    need.validate_response(FormResponse(action="accept", content={"q": "cached-test"}))
    assert len(calls) == 1  # Syntax + default together, no identical nested work.


async def test_pathological_async_match_heartbeat_timeout_and_no_sync_second_match(monkeypatch):
    need = NeedInputForm(
        heading="H", questions=[TextQuestion(id="q", label="Q", pattern="^(a+)+$")]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("async answer validation must never call subprocess.run")

    monkeypatch.setattr(subprocess, "run", forbidden)
    start = time.monotonic()
    task = asyncio.create_task(
        forms._validate_form_response(
            need, FormResponse(action="accept", content={"q": "a" * 10000 + "!"})
        )
    )
    await asyncio.sleep(0.02)
    assert time.monotonic() - start < 0.5 and not task.done()
    with pytest.raises(ValueError, match="time budget"):
        await task


async def test_cancellation_reaps_actual_worker_and_releases_slot(monkeypatch):
    started = asyncio.Event()
    processes = []
    original = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        started.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    task = asyncio.create_task(forms._pattern_matches_async("^(a+)+$", "a" * 12000 + "!"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert processes[0].returncode is not None
    assert await forms._pattern_matches_async("^after-cancel$", "after-cancel")


async def test_many_fields_are_sequential_and_global_worker_budget_is_bounded(monkeypatch):
    questions = [TextQuestion(id=f"q{i}", label="Q", pattern=f"^a{{{i + 1}}}$") for i in range(12)]
    need = NeedInputForm(heading="H", questions=questions)
    active = peak = 0
    original = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        nonlocal active, peak
        process = await original(*args, **kwargs)
        active += 1
        peak = max(peak, active)
        original_wait = process.wait
        counted = False

        async def wait():
            nonlocal active, counted
            result = await original_wait()
            if not counted:
                active -= 1
                counted = True
            return result

        process.wait = wait
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    answer = FormResponse(
        action="accept", content={q.id: "a" * (i + 1) for i, q in enumerate(questions)}
    )
    assert await forms._validate_form_response(need, answer) == answer
    assert peak == 1 and active == 0
    acquired = [forms._PATTERN_SLOTS.acquire(blocking=False) for _ in range(4)]
    assert all(acquired)
    try:
        with pytest.raises(ValueError, match="concurrency budget"):
            await forms._pattern_matches_async("uncached-concurrency", "test")
    finally:
        for _ in acquired:
            forms._PATTERN_SLOTS.release()


async def test_pattern_and_input_limits_apply_before_spawn(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("oversized inputs must not spawn")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", forbidden)
    for pattern, value in [("a" * 4097, ""), ("a", "x" * 65537)]:
        with pytest.raises(ValueError, match="size budget"):
            forms._pattern_matches(pattern, value)
        with pytest.raises(ValueError, match="size budget"):
            await forms._pattern_matches_async(pattern, value)
    with pytest.raises(ValueError):
        NeedInputForm(
            heading="H", questions=[TextQuestion(id=str(i), label="Q") for i in range(65)]
        )


async def test_cancel_during_spawn_also_reaps_actual_process(monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    processes = []
    original = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        entered.set()
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    task = asyncio.create_task(forms._pattern_matches_async("^(a+)+$", "a" * 13000 + "!"))
    await entered.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert processes[0].returncode is not None


async def test_worker_creation_failure_is_safe_and_releases_budget(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("process budget unavailable")

    async def unavailable_async(*args, **kwargs):
        raise OSError("process budget unavailable")

    monkeypatch.setattr(subprocess, "run", unavailable)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", unavailable_async)
    with pytest.raises(ValueError, match="worker unavailable"):
        forms._pattern_matches("uncached-spawn-failure", "x")
    with pytest.raises(ValueError, match="worker unavailable"):
        await forms._pattern_matches_async("uncached-spawn-failure", "x")
    acquired = [forms._PATTERN_SLOTS.acquire(blocking=False) for _ in range(4)]
    try:
        assert all(acquired)
    finally:
        for success in acquired:
            if success:
                forms._PATTERN_SLOTS.release()


async def test_four_actual_concurrent_workers_and_fifth_fail_fast(monkeypatch):
    entered = asyncio.Queue()
    release = asyncio.Event()
    processes = []
    original = asyncio.create_subprocess_exec

    async def spawn(*args, **kwargs):
        process = await original(*args, **kwargs)
        processes.append(process)
        entered.put_nowait(process)
        await release.wait()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    tasks = [
        asyncio.create_task(forms._pattern_matches_async(f"^concurrent{i}$", f"concurrent{i}"))
        for i in range(4)
    ]
    try:
        for _ in tasks:
            await asyncio.wait_for(entered.get(), 5)
        with pytest.raises(ValueError, match="concurrency budget"):
            await forms._pattern_matches_async("^fifth-concurrent$", "fifth-concurrent")
        assert len(processes) == 4
    finally:
        release.set()
        assert all(await asyncio.gather(*tasks))
    assert all(process.returncode is not None for process in processes)


async def test_async_search_semantics_invalid_values_and_optional_default_omission():
    need = NeedInputForm(
        heading="H",
        questions=[
            TextQuestion(
                id="q", label="Q", pattern="search-me", default="search-me", required=False
            )
        ],
    )
    omitted = FormResponse(action="accept", content={})
    assert await forms._validate_form_response(need, omitted) == omitted
    valid = FormResponse(action="accept", content={"q": "prefix-search-me-suffix"})
    assert await forms._validate_form_response(need, valid) == valid
    with pytest.raises(ValueError, match="does not match pattern"):
        await forms._validate_form_response(
            need, FormResponse(action="accept", content={"q": "no"})
        )
