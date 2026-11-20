# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime publication follows real all-file metadata commit, never vice versa."""

import sqlite3

import pytest
from public_model_fixtures import native_models as native_models
from public_model_fixtures import public_registry as public_registry

from tests.storage.test_multi_backend_batch import InstrumentedConnection, state


@pytest.fixture
def instrument_connections(monkeypatch):
    connect = sqlite3.connect
    monkeypatch.setattr(
        sqlite3, "connect", lambda *a, **kw: connect(*a, **kw, factory=InstrumentedConnection)
    )


def snapshot(sessions):
    return (
        state([session.handle.storage for session in sessions]),
        [session.info.model_dump() for session in sessions],
        [session.handle.info.model_dump() for session in sessions],
        [session.options for session in sessions],
        [session._pending_model for session in sessions],
        [session._agent.llm for session in sessions],
    )


@pytest.mark.parametrize("stage", ["begin", "write", "commit"])
async def test_reasoning_failure_real_transaction_preserves_all_sharers(
    public_registry, instrument_connections, stage
):
    registry, options, made = public_registry
    root = await registry.create(options)
    child = await registry.create(root.options.inherit(kept=True), parent_id=root.id)
    sessions = [root, child]
    before = snapshot(sessions)
    sentinel = RuntimeError("transaction sentinel")
    coordinator = min(
        sessions, key=lambda s: str(s.handle.storage._canonical_path)
    ).handle.storage._conn
    coordinator.failure, coordinator.stage = sentinel, stage
    try:
        with pytest.raises(RuntimeError) as caught:
            await root.set_reasoning("high")
        assert caught.value is sentinel
        assert snapshot(sessions) == before
        for session in sessions:
            assert registry.store.get(session.id).reasoning == session.info.reasoning
    finally:
        coordinator.stage = None
    await root.set_reasoning("high")
    assert all(s.info.reasoning == registry.store.get(s.id).reasoning == "high" for s in sessions)
    assert root._agent.llm is child._agent.llm


async def test_pending_replacement_commit_failure_preserves_previous_selection(
    public_registry, instrument_connections
):
    registry, options, made = public_registry
    root = await registry.create(options)
    await root.set_model("pending-valid")
    before = snapshot([root])
    conn = root.handle.storage._conn
    sentinel = RuntimeError("commit sentinel")
    conn.failure, conn.stage = sentinel, "commit"
    try:
        with pytest.raises(RuntimeError) as caught:
            await root.set_model("rejected")
        assert caught.value is sentinel
        assert snapshot([root]) == before
        assert made[-1]._closed
        assert not made[-2]._closed
        assert registry.store.get(root.id).model == "pending-valid"
    finally:
        conn.stage = None
    await root._apply_pending_model()
    assert root._agent.llm is made[-2]


async def test_pending_reasoning_failure_and_failed_application_preserve_pending(
    public_registry, instrument_connections, monkeypatch
):
    registry, options, made = public_registry
    root = await registry.create(options)
    await root.set_model("pending-valid")
    before = snapshot([root])
    sentinel = RuntimeError("preparation sentinel")
    monkeypatch.setattr(root._agent, "_set_llm_failure", sentinel)
    with pytest.raises(RuntimeError) as caught:
        await root._apply_pending_model()
    assert caught.value is sentinel
    assert snapshot([root]) == before
    conn = root.handle.storage._conn
    conn.failure, conn.stage = sentinel, "begin"
    try:
        with pytest.raises(RuntimeError) as caught:
            await root.set_reasoning("high")
        assert caught.value is sentinel
        assert snapshot([root]) == before
        assert not made[-1]._closed
    finally:
        conn.stage = None


async def test_capacity_refuses_before_any_session_publication(public_registry):
    registry, options, made = public_registry
    root = await registry.create(options)
    child = await registry.create(root.options.inherit(kept=True), parent_id=root.id)
    sessions = [root, child]
    coordinator = min(
        sessions, key=lambda s: str(s.handle.storage._canonical_path)
    ).handle.storage._conn
    old = coordinator.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, 0)
    before = snapshot(sessions)
    try:
        with pytest.raises(Exception, match="[Cc]apacity|[Aa]ttachment|[Ll]imit"):
            await root.set_reasoning("high")
        assert snapshot(sessions) == before
    finally:
        coordinator.setlimit(sqlite3.SQLITE_LIMIT_ATTACHED, old)


@pytest.mark.parametrize("failure_kind", ["metadata", "selection-race"])
async def test_selected_create_failure_preserves_parent_and_closes_unpublished_owner(
    public_registry, monkeypatch, failure_kind
):
    from nooa_atom.session.store import SessionHandle

    registry, options, made = public_registry
    parent = await registry.create(options)
    await parent.set_reasoning("high")
    before = snapshot([parent])
    binding = parent._selected_binding()
    references = binding._references
    sentinel = RuntimeError("child-only write failure")
    if failure_kind == "metadata":
        original = SessionHandle.set_reasoning

        def fail(handle, level):
            if handle.id != parent.id:
                raise sentinel
            return original(handle, level)

        monkeypatch.setattr(SessionHandle, "set_reasoning", fail)

    async def selection_race(child):
        await parent.set_model("pending-race")

    with pytest.raises(RuntimeError) as caught:
        await registry.create(
            parent.options.inherit(kept=True),
            parent_id=parent.id,
            prepare=selection_race if failure_kind == "selection-race" else None,
        )
    if failure_kind == "metadata":
        assert caught.value is sentinel
        assert snapshot([parent]) == before
    else:
        assert "selection changed" in str(caught.value)
        assert parent.info.model == "pending-race" and parent._pending_model is not None
        assert parent._agent.llms["main"] is binding
    assert binding._references == references
    assert not parent._agent.llm._closed


@pytest.mark.parametrize("operation", ["create", "open"])
@pytest.mark.parametrize("original_cancel", [False, True])
async def test_unpublished_first_failure_wins_and_open_preserves_record(
    public_registry, monkeypatch, operation, original_cancel
):
    import asyncio

    registry, options, made = public_registry
    original_session = await registry.create(options) if operation == "open" else None
    if original_session is not None:
        session_id = original_session.id
        await original_session.close()
        assert session_id not in registry.sessions
        persisted = registry.store.get(session_id).model_dump()
    entered, release = asyncio.Event(), asyncio.Event()
    sentinel = (
        asyncio.CancelledError("first cancellation") if original_cancel else RuntimeError("prepare")
    )
    secondary = RuntimeError("close failed after native cleanup")
    unpublished = []

    async def prepare(session):
        unpublished.append(session)
        close = session.close

        async def rejected_close():
            entered.set()
            await release.wait()
            await close()
            raise secondary

        monkeypatch.setattr(session, "close", rejected_close)
        raise sentinel

    task = asyncio.create_task(
        registry.load(session_id, prepare=prepare)
        if operation == "open"
        else registry.create(options, prepare=prepare)
    )
    try:
        await asyncio.wait_for(entered.wait(), 3)
        for _ in range(2):
            task.cancel("later cancellation")
            await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(BaseException) as caught:
            await task
        assert caught.value is sentinel
        assert not registry._reserved and unpublished[0].id not in registry.sessions
        assert unpublished[0]._agent.llms["main"]._references == 0
        if operation == "open":
            assert registry.store.get(session_id).model_dump() == persisted
            assert registry.store.path_for(session_id).exists()
        else:
            assert not registry.store.path_for(unpublished[0].id).exists()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


async def test_create_discard_attempts_store_delete_after_handle_error(
    public_registry, monkeypatch
):
    registry, options, made = public_registry
    sentinel = RuntimeError("prepare wins")
    deleted = []
    delete = registry.store.delete

    def record_delete(session_id):
        deleted.append(session_id)
        return delete(session_id)

    monkeypatch.setattr(registry.store, "delete", record_delete)
    unpublished = []

    async def prepare(session):
        unpublished.append(session)
        original = session.handle.close

        def failed_close():
            original()
            raise RuntimeError("handle-close secondary")

        monkeypatch.setattr(session.handle, "close", failed_close)
        raise sentinel

    with pytest.raises(RuntimeError) as caught:
        await registry.create(options, prepare=prepare)
    assert caught.value is sentinel
    assert deleted == [unpublished[0].id] and not registry._reserved
    assert not registry.store.path_for(unpublished[0].id).exists()


class _FourthSessionControl(BaseException):
    pass


@pytest.mark.parametrize(
    "stage",
    ["pause", "stop_starting", "children", "outcomes", "queues", "agent", "record", "update"],
)
@pytest.mark.parametrize("kind", ["custom", "systemexit", "cancel", "ordinary"])
async def test_session_fourth_all_close_stages_and_terminal_result(
    public_registry, monkeypatch, stage, kind
):
    import asyncio
    import inspect

    registry, options, made = public_registry
    session = await registry.create(options)
    failure = {
        "custom": _FourthSessionControl,
        "systemexit": SystemExit,
        "cancel": asyncio.CancelledError,
        "ordinary": RuntimeError,
    }[kind]("close sentinel")
    owner, method = {
        "pause": (session._agent.turns, "pause"),
        "stop_starting": (session._agent.turns, "stop_starting"),
        "children": (session, "_before_close"),
        "outcomes": (session, "_resolve_all"),
        "queues": (session._agent.queue_manager, "shutdown"),
        "agent": (session._agent, "aclose"),
        "record": (session.handle, "close"),
        "update": (session, "_emit"),
    }[stage]
    actual = getattr(owner, method)
    calls = []

    def failing(*args, **kwargs):
        if stage == "agent":
            vars(owner).pop(method, None)
        else:
            monkeypatch.setattr(owner, method, actual)
        calls.append(stage)
        result = actual(*args, **kwargs)
        if inspect.isawaitable(result):

            async def finish():
                await result
                raise failure

            return finish()
        raise failure

    if stage == "agent":
        vars(owner)[method] = failing
    else:
        monkeypatch.setattr(owner, method, failing)
    try:
        for _ in range(2):
            if kind in ("custom", "systemexit"):
                with pytest.raises(BaseException) as caught:
                    await session.close()
                assert caught.value is failure
            else:
                await session.close()
        assert session._closed and not registry.store.is_active(session.id)
        assert session._close_task.done() and not session._close_task.cancelled()
        assert session._close_task.result() is (
            failure if kind in ("custom", "systemexit") else None
        )
        assert made[0]._closed and made[0]._lease.leases == 0
        assert session._cleanup_errors.report()["error_count"] >= 1
        assert session._model_binding._references == 0
        # Runtime teardown may call queue shutdown too; the retained close itself never repeats.
        before = list(calls)
        if kind in ("custom", "systemexit"):
            with pytest.raises(BaseException) as caught:
                await session.close()
            assert caught.value is failure
        else:
            await session.close()
        assert calls == before
    finally:
        if stage == "agent":
            vars(owner).pop(method, None)
        else:
            monkeypatch.setattr(owner, method, actual)
        if session._closed:
            # Fully disposed: don't ask the fixture to report this intentional custom control again.
            registry.sessions.pop(session.id, None)


async def test_session_fourth_loop_ended_sync_failure_still_retains_worker(
    public_registry, monkeypatch
):
    from nooa.runtime.turn_loop import TurnLoopEnded

    registry, options, made = public_registry
    session = await registry.create(options)
    actual = session._resolve_all
    failure = _FourthSessionControl("synchronous outcome callback")

    def failing(error):
        actual(error)
        raise failure

    monkeypatch.setattr(session, "_resolve_all", failing)
    session._on_loop_ended(TurnLoopEnded(message="fixture terminal"))
    assert session._close_task is not None and session._closing
    try:
        for _ in range(2):
            with pytest.raises(_FourthSessionControl) as caught:
                await session.close()
            assert caught.value is failure
        assert made[0]._closed and made[0]._lease.leases == 0
        assert not registry.store.is_active(session.id) and session._close_task.result() is failure
    finally:
        monkeypatch.setattr(session, "_resolve_all", actual)
        if session._closed:
            registry.sessions.pop(session.id, None)


async def test_session_fourth_caller_cancel_immediate_but_owned_worker_finishes(
    public_registry, monkeypatch
):
    import asyncio

    registry, options, made = public_registry
    session = await registry.create(options)
    entered, release = asyncio.Event(), asyncio.Event()
    actual = session._agent.aclose

    async def gated():
        entered.set()
        await release.wait()
        vars(session._agent).pop("aclose", None)
        await actual()

    vars(session._agent)["aclose"] = gated
    caller = asyncio.create_task(session.close())
    try:
        await asyncio.wait_for(entered.wait(), 3)
        worker = session._close_task
        caller.cancel("first caller")
        with pytest.raises(asyncio.CancelledError, match="first caller"):
            await caller
        assert not worker.done() and not made[0]._closed
        second = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        second.cancel("second caller")
        with pytest.raises(asyncio.CancelledError, match="second caller"):
            await second
        assert session._close_task is worker and not worker.done()
        release.set()
        await session.close()
        assert worker.done() and worker.result() is None
        assert made[0]._closed and made[0]._lease.leases == 0
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
        await session.close()


async def test_session_fourth_close_reentrant_guard_keeps_single_worker(
    public_registry, monkeypatch
):
    registry, options, made = public_registry
    session = await registry.create(options)
    actual = session._before_close
    seen = []

    async def reentrant():
        seen.append(session._close_task)
        await session.close()
        await actual()

    monkeypatch.setattr(session, "_before_close", reentrant)
    await session.close()
    await session.close()
    assert seen == [session._close_task]
    assert made[0]._closed and made[0]._lease.leases == 0


@pytest.mark.parametrize("operation", ["children", "all"])
@pytest.mark.parametrize("control_type", [BaseException, SystemExit, KeyboardInterrupt])
async def test_fifth_registry_fanout_control_data_finishes_entire_snapshot(
    public_registry, operation, control_type
):
    registry, options, made = public_registry
    root = await registry.create(options)
    children = [
        await registry.create(root.options.inherit(kept=True), parent_id=root.id) for _ in range(3)
    ]
    order = []
    control = control_type("finite fanout control")

    for index, child in enumerate(children):

        async def closing(index=index):
            order.append(index)
            if index == 2:
                raise control

        child._agent.event_manager.on_close(closing)

    try:
        with pytest.raises(control_type) as caught:
            if operation == "children":
                await registry._close_children(root.id)
            else:
                await registry.close_all()
        assert caught.value is control
        assert sorted(order) == [0, 1, 2]
        assert all(child._closed and child.id not in registry.sessions for child in children)
        assert not registry._reserved
        if operation == "children":
            assert root.id in registry.sessions and not root._closed
            await root.close()
        else:
            assert root._closed and not registry.sessions
        assert all(model._closed and model._lease.leases == 0 for model in made)
        # Global close is a snapshot, not a permanent session-creation fence.
        later = await registry.create(options)
        assert not later._closed
        await registry.close_all()
        assert later._closed and not registry.sessions
    finally:
        await registry.close_all()


async def test_fifth_registry_preflight_rejects_before_any_sibling_fence(public_registry):
    from nooa.runtime.event_manager import _CloseDependency

    registry, options, made = public_registry
    root = await registry.create(options)
    children = [
        await registry.create(root.options.inherit(kept=True), parent_id=root.id) for _ in range(2)
    ]
    dependency = _CloseDependency()
    dependency.attach(children[-1]._agent.event_manager)
    token = dependency.enter()
    try:
        with pytest.raises(RuntimeError, match="active"):
            await registry.close_all()
        assert all(not session.closing for session in (root, *children))
        assert all(session._close_task is None for session in (root, *children))
        assert all(not model._closed for model in made)
    finally:
        dependency.reset(token)
        dependency.finish()
    await registry.close_all()
    assert not registry.sessions


class _SixthPendingControl(BaseException):
    pass


async def _sixth_pending_dispatch(client):
    from nooa.unifiedllm.request import LLMRequest

    response = await client.acall(
        LLMRequest(messages=[{"role": "user", "content": "sixth pending native"}])
    )
    assert response is not None


@pytest.mark.parametrize(
    "control_type", [SystemExit, KeyboardInterrupt, _SixthPendingControl, RuntimeError]
)
async def test_sixth_pending_listener_sees_coherent_commit_and_retained_data_worker(
    public_registry, native_models, control_type
):
    registry, options, made = public_registry
    session = await registry.create(options)
    await session.set_model("sixth-A")
    owner = session._pending_model[1]
    await _sixth_pending_dispatch(owner.client)
    control = control_type("listener after durable commit")
    observed = []

    def listener(update):
        if getattr(update, "model", None) != "sixth-B":
            return
        retirement = next(r for r in session._retired_llms if r.owner is owner)
        assert retirement.task is not None
        assert session.info.model == session.options.model == "sixth-B"
        assert session._pending_model[0] == registry.store.get(session.id).model == "sixth-B"
        assert session.options.model_binding is session._pending_model[1]
        observed.append(retirement)
        raise control

    unsubscribe = session.subscribe(listener)
    try:
        # Direct public await safely tests real interpreter controls; the only
        # cleanup Task carries controls as DATA, not Task-terminal exceptions.
        if control_type is RuntimeError:
            await session.set_model("sixth-B")
        else:
            with pytest.raises(control_type) as caught:
                await session.set_model("sixth-B")
            assert caught.value is control
        await _sixth_pending_dispatch(session._pending_model[1].client)
        worker = observed[0].task
        await session.close()
        await session.close()
        assert worker.done() and not worker.cancelled()
        assert worker.exception() is None and worker.result() is None
        assert not session._retired_llms
        assert made[1]._closed and made[2]._closed
        assert len(native_models.seen) == 2
        assert all(sdk.max_retries == 9 for sdk in native_models.sdks)
        assert all(not http.is_closed for http in native_models.http)
    finally:
        unsubscribe()
        await session.close()


async def test_sixth_pending_scheduling_failure_retains_empty_slot_and_committed_B(
    public_registry, native_models, monkeypatch
):
    import asyncio
    import inspect

    from nooa_atom.session.session import Session

    registry, options, made = public_registry
    session = await registry.create(options)
    await session.set_model("sixth-schedule-A")
    owner = session._pending_model[1]
    await _sixth_pending_dispatch(owner.client)
    original = asyncio.ensure_future
    control = _SixthPendingControl("task factory failure")
    unscheduled, observed = [], []

    def fail(coroutine, *args, **kwargs):
        # Observe only the actual Session retirement scheduling site.
        frame = inspect.currentframe().f_back
        try:
            if frame.f_code is Session._start_retirement.__code__:
                retirement = next(r for r in session._retired_llms if r.owner is owner)
                assert retirement.task is None
                assert session.info.model == session.options.model == "sixth-schedule-B"
                assert registry.store.get(session.id).model == session._pending_model[0]
                unscheduled.append(coroutine)
                raise control
        finally:
            del frame
        return original(coroutine, *args, **kwargs)

    unsubscribe = session.subscribe(
        lambda update: (
            observed.append(update)
            if getattr(update, "model", None) == "sixth-schedule-B"
            else None
        )
    )
    try:
        with monkeypatch.context() as patch:
            patch.setattr(asyncio, "ensure_future", fail)
            with pytest.raises(_SixthPendingControl) as caught:
                await session.set_model("sixth-schedule-B")
        assert caught.value is control
        retirement = next(r for r in session._retired_llms if r.owner is owner)
        assert retirement.task is None and not made[1]._closed and not observed
        assert len(unscheduled) == 1 and unscheduled[0].cr_frame is None
        await _sixth_pending_dispatch(session._pending_model[1].client)
        await session.close()
        assert retirement.task.done() and retirement.task.result() is None
        assert made[1]._closed and made[2]._closed and not session._retired_llms
        assert all(not http.is_closed for http in native_models.http)
    finally:
        unsubscribe()
        await session.close()


@pytest.mark.parametrize("stage", ["validation", "storage"])
async def test_sixth_pending_precommit_failure_preserves_A_and_live_retirement_list(
    public_registry, instrument_connections, monkeypatch, stage
):
    registry, options, made = public_registry
    session = await registry.create(options)
    await session.set_model("sixth-precommit-A")
    await _sixth_pending_dispatch(session._pending_model[1].client)
    old_pending, old_info, old_options = session._pending_model, session.info, session.options
    old_retired = session._retired_llms
    control = _SixthPendingControl("precommit failure")
    conn = session.handle.storage._conn
    if stage == "validation":

        def fail(_binding):
            raise control

        monkeypatch.setattr(session, "_metadata", fail)
    else:
        conn.failure, conn.stage = control, "commit"
    try:
        with pytest.raises(_SixthPendingControl) as caught:
            await session.set_model("sixth-rejected-B")
        assert caught.value is control
        assert session._pending_model is old_pending
        assert session.info is old_info and session.options is old_options
        assert session._retired_llms is old_retired and not old_retired
        assert registry.store.get(session.id).model == "sixth-precommit-A"
        assert not made[1]._closed and made[2]._closed
    finally:
        conn.stage = None
    await session.close()
    assert made[1]._closed


@pytest.mark.parametrize(
    "failure_type", [None, SystemExit, KeyboardInterrupt, _SixthPendingControl]
)
async def test_sixth_pending_cancellation_and_failure_reuse_one_strong_data_worker(
    public_registry, native_models, monkeypatch, failure_type
):
    import asyncio
    import weakref

    from nooa.runtime.model_binding import _cleanup_result

    registry, options, made = public_registry
    session = await registry.create(options)
    await session.set_model("sixth-gated-A")
    owner = session._pending_model[1]
    await _sixth_pending_dispatch(owner.client)
    weak_owner = weakref.ref(owner)
    entered, release = asyncio.Event(), asyncio.Event()
    actual = owner.aclose
    calls = []
    failure = failure_type("retirement cleanup control") if failure_type is not None else None

    async def gated():
        calls.append(1)
        entered.set()
        await release.wait()
        await actual()
        if failure is not None:
            raise failure

    monkeypatch.setattr(owner, "aclose", gated)
    caller = asyncio.create_task(session.set_model("sixth-gated-B"))
    try:
        await asyncio.wait_for(entered.wait(), 3)
        retirement = next(r for r in session._retired_llms if r.owner is owner)
        worker = retirement.task
        # Existing rejection cleanup suppresses logical cancellation but keeps
        # the actual cleanup obligation alive and discoverable.
        caller.cancel("postcommit caller cancellation")
        await caller
        await _sixth_pending_dispatch(session._pending_model[1].client)
        for _ in range(2):
            joining = asyncio.create_task(session._dispose_rejected(owner))
            await asyncio.sleep(0)
            joining.cancel("repeated join cancellation")
            await joining
        assert retirement.task is worker and not worker.done() and calls == [1]
        assert weak_owner() is retirement.owner
        closing = asyncio.create_task(_cleanup_result(session.close))
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        if failure is None:
            await closing
            assert not session._retired_llms
        else:
            assert await closing is failure
            assert closing.exception() is None
            assert retirement in session._retired_llms and retirement.owner is weak_owner()
            # Settled failure remains DATA and is never replaced by a second worker.
            await session._dispose_rejected(owner)
            with pytest.raises(failure_type) as caught:
                await session.close()
            assert caught.value is failure
            assert session._cleanup_errors.report()["error_count"] >= 1
        assert retirement.task is worker and worker.done() and worker.exception() is None
        assert worker.result() is failure and calls == [1]
        assert made[1]._closed and made[2]._closed
        assert all(not http.is_closed for http in native_models.http)
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
        if session._close_task is not None:
            await asyncio.shield(session._close_task)
        if session._closed:
            registry.sessions.pop(session.id, None)
        else:
            await session.close()


# D2 maintenance proof uses real Atom turns, kept native-backed delegation and
# public-loader SDK HTTP. It does not manufacture settlement or accounting events.
def _d2_atom_native_payload(family, *, code=None, content=None, input_tokens=11, empty=False):
    import json

    from tests.unifiedllm.test_native_load_llm import paid_payload

    data = paid_payload(family, input_tokens=input_tokens, output_tokens=3, empty=empty)
    if code is not None:
        arguments = json.dumps({"code": code})
        if family == "openai-chat":
            data["choices"][0]["message"].update(
                content=None,
                tool_calls=[
                    {
                        "id": "atom_cell",
                        "type": "function",
                        "function": {"name": "python_cell", "arguments": arguments},
                    }
                ],
            )
            data["choices"][0]["finish_reason"] = "tool_calls"
        elif family == "openai-responses":
            data["output"] = [
                {
                    "id": "atom_cell",
                    "type": "function_call",
                    "call_id": "atom_cell",
                    "name": "python_cell",
                    "arguments": arguments,
                    "status": "completed",
                }
            ]
        else:
            data["content"] = [
                {
                    "type": "tool_use",
                    "id": "atom_cell",
                    "name": "python_cell",
                    "input": {"code": code},
                }
            ]
            data["stop_reason"] = "tool_use"
    elif content is not None:
        if family == "openai-chat":
            data["choices"][0]["message"]["content"] = content
        elif family == "openai-responses":
            data["output"][0]["content"][0]["text"] = content
        else:
            data["content"][0]["text"] = content
    return data


async def _d2_atom_idle(*sessions):
    import asyncio

    async def wait():
        while any(session._agent.turns.running for session in sessions):
            await asyncio.sleep(0.001)

    await asyncio.wait_for(wait(), 10)


def _d2_maintenance_rows(store, session_id, kind="MaintenanceUsageRecorded"):
    return [row for _, row in store.load_rows(session_id, frozenset({kind}))]


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
@pytest.mark.parametrize(
    "plan_mode", ["accepted", "paid-rejected", "paid-parse-failed", "paid-truncated"]
)
async def test_d2_native_atom_kept_child_maintenance_http_retry_own_attributed_reload(
    native_models,
    tmp_path,
    family,
    plan_mode,
):
    import asyncio
    import json

    import httpx
    from nooa_atom import SessionOptions, SessionStore
    from nooa_atom.agent.agent import AtomAgent
    from nooa_atom.session.registry import SessionRegistry

    from nooa.unifiedllm.request import LLMRequest
    from tests.unifiedllm.test_native_load_llm import zero_delay

    foreground, reflection = [], []
    attempts = {}
    restarted_child_request = asyncio.Event()

    def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "native-wire-model"
        if any(
            body.get(key) == 2048
            for key in ("max_tokens", "max_output_tokens", "max_completion_tokens")
        ):
            reflection.append(body)
            serialized = json.dumps(body)
            label = "child" if "child accepted" in serialized else "root"
            attempts[label] = attempts.get(label, 0) + 1
            if attempts[label] == 1:
                # An actual SDK transient retry; no invented paid usage on the
                # failed HTTP response. Structured parse failures do not retry.
                return httpx.Response(
                    500, json={"error": {"message": "retry", "type": "api_error"}}
                )
            plan = {
                "summary": "",
                "useful_findings": "native maintenance finding",
                "decisions": "",
                "outstanding_work": "",
                "drop_names": [] if plan_mode == "accepted" else ["not_a_candidate"],
            }
            data = _d2_atom_native_payload(
                family,
                content=(
                    None
                    if plan_mode == "paid-truncated"
                    else "not valid JSON"
                    if plan_mode == "paid-parse-failed"
                    else json.dumps(plan)
                ),
                input_tokens=47,
                empty=plan_mode == "paid-truncated",
            )
            if plan_mode == "paid-truncated":
                if family == "openai-chat":
                    data["choices"][0]["finish_reason"] = "length"
                elif family == "openai-responses":
                    data.update(
                        status="incomplete", incomplete_details={"reason": "max_output_tokens"}
                    )
                else:
                    data["stop_reason"] = "max_tokens"
            return httpx.Response(200, json=data)
        foreground.append(body)
        if len(foreground) == 6:
            restarted_child_request.set()
        if len(foreground) == 1:
            code = (
                "if not self.v.get('delegation_completed', False):\n"
                "    review = self.jobs.launch(Subagent('review'), name='review')\n"
                "    result = await review.send('native kept review')\n"
                "    assert result.solution_description == 'native child result'\n"
                "    self.v.delegation_completed = True\n"
                "return_result(Done(message='root reply', explanation='root accepted'))"
            )
        elif len(foreground) == 2:
            code = (
                "return_result(Done(result=TaskResult("
                "solution_description='native child result', evidence='native SDK', "
                "how_to_verify='normal child turn'), explanation='child accepted'))"
            )
        elif len(foreground) == 3:
            # Retire the kept request resource in a normal parent turn before
            # checkpoint/reload; no rehydrated job is allowed to repeat work.
            code = (
                "review_job = self.jobs.get('review')\n"
                "if review_job is not None:\n"
                "    await review_job.cancel()\n"
                "return_result(Done(message='retired', explanation='retired kept review'))"
            )
        else:
            code = (
                "return_result(Done(message='next reply', explanation='next accepted', "
                "result=TaskResult(solution_description='native child result', "
                "evidence='native SDK', how_to_verify='normal restarted turn')))"
            )
        return httpx.Response(200, json=_d2_atom_native_payload(family, code=code))

    assert set(LLMRequest.model_fields) == {"messages", "tools", "tool_choice", "output_type"}
    view = native_models.make(
        family,
        handler=handler,
        model="native-wire-model",
        retry_config=zero_delay(empty=True),
    )
    store = SessionStore(tmp_path / "sessions")
    from nooa_atom.agent.factory import create_session_agent

    registry = SessionRegistry(store, agent_factory=create_session_agent)
    options = SessionOptions(
        workspace=tmp_path,
        sessions_dir=tmp_path / "sessions",
        agent_spec="nooa_atom.agent:AtomAgent",
        model="configured-native-alias",
        llm=view,
        experimental_done_reflection=True,
        experimental_reflection_min_turns=1,
        experimental_reflection_min_seconds=0,
        kept=True,
    )
    root = child = None
    try:
        root = await registry.create(options)
        result = await asyncio.wait_for(root.prompt("native root delegation"), 15)
        assert result.explanation == "root accepted"
        children = [s for s in registry.sessions.values() if s.info.parent_id == root.id]
        assert len(children) == 1
        child = children[0]
        await _d2_atom_idle(root, child)
        assert isinstance(root._agent, AtomAgent) and isinstance(child._agent, AtomAgent)
        assert root._agent.llm is child._agent.llm
        assert len(foreground) == 2 and len(reflection) == 4
        assert attempts == {"child": 2, "root": 2}
        own_root = _d2_maintenance_rows(store, root.id)
        own_child = _d2_maintenance_rows(store, child.id)
        attributed = _d2_maintenance_rows(store, root.id, "MaintenanceUsageAttributed")
        assert len(own_root) == len(own_child) == len(attributed) == 1
        for row in own_root + own_child:
            expected = {
                "input_tokens": 47,
                "output_tokens": 3,
                "cached_input_tokens": None if family == "openai-chat" else 0,
                "cache_write_input_tokens": 0 if family == "anthropic-messages" else None,
                "reasoning_tokens": 0 if family == "openai-responses" else None,
                "total_tokens": 50,
                "cost_usd": None,
                # The actual first HTTP failure has no usage proof. Measured
                # zeros in the second attempt must not make that attempt known.
                "complete_fields": [],
            }
            assert row["usage"] == expected
            assert row["attempt_count"] == 2
        assert attributed[0]["source_session_id"] == child.id
        assert attributed[0]["logical_call_id"] == own_child[0]["logical_call_id"]
        assert attributed[0]["usage"] == own_child[0]["usage"]
        for session in (root, child):
            assert session.info.usage.input_tokens == 58
            assert session.info.usage.last_input_tokens == (
                11 if plan_mode in {"paid-parse-failed", "paid-truncated"} else 47
            )
            notes = session._agent.context.get("done_reflection_notes")
            assert bool(notes) is (plan_mode == "accepted")
            turns = store.load_rows(session.id, frozenset({"TurnEnded"}))
            assert len(turns) == 1 and turns[0][1]["usage"]["input_tokens"] == 11
        assert root.info.usage.attributed_input_tokens == 58
        root._agent._done_reflection = False
        assert (
            await root.prompt("retire the completed kept review")
        ).explanation == "retired kept review"
        await _d2_atom_idle(root, child)
        assert _d2_maintenance_rows(store, root.id) == own_root
        assert _d2_maintenance_rows(store, child.id) == own_child
        saved_root, saved_child = root.info.usage.model_copy(), child.info.usage.model_copy()
        root_id, child_id = root.id, child.id
        await registry.close_all()
        assert all(not http.is_closed for http in native_models.http)
        assert all(sdk.max_retries == 9 for sdk in native_models.sdks)
        lifecycle = store.load_rows(
            child_id, frozenset({"ItemAdmitted", "ItemConsumed", "ItemWithdrawn", "ItemDiscarded"})
        )
        admitted_ids = {row["item_id"] for kind, row in lifecycle if kind == "ItemAdmitted"}
        finished_ids = {row["item_id"] for kind, row in lifecycle if kind != "ItemAdmitted"}
        assert admitted_ids <= finished_ids, lifecycle
        loaded_root = await registry.load(root_id, llm=view, experimental_done_reflection=False)
        assert loaded_root.info.usage.input_tokens == saved_root.input_tokens
        assert loaded_root.info.usage.attributed_input_tokens == saved_root.attributed_input_tokens
        assert loaded_root.info.usage.last_input_tokens == 0
        assert loaded_root._agent.v.delegation_completed is True
        assert (await loaded_root.prompt("next ordinary generation")).explanation == "next accepted"
        await _d2_atom_idle(loaded_root)
        assert len(foreground) == 4
        assert loaded_root.info.usage.input_tokens == saved_root.input_tokens + 11
        assert loaded_root.info.usage.attributed_input_tokens == saved_root.attributed_input_tokens
        assert _d2_maintenance_rows(store, root_id) == own_root
        assert _d2_maintenance_rows(store, child_id) == own_child
        assert _d2_maintenance_rows(store, root_id, "MaintenanceUsageAttributed") == attributed
        # Loading a child emits a real restart notification and may start a new
        # ordinary worker turn. Check durable reload before that independent
        # turn rather than mislabelling its foreground usage as maintenance.
        loaded_child = await registry.load(child_id, llm=view, experimental_done_reflection=False)
        assert loaded_child.info.usage.input_tokens == saved_child.input_tokens
        assert (
            loaded_child.info.usage.attributed_input_tokens == saved_child.attributed_input_tokens
        )
        assert loaded_child.info.usage.last_input_tokens == 0
        await asyncio.wait_for(restarted_child_request.wait(), 10)
        await _d2_atom_idle(loaded_root, loaded_child)
        # Child restart and its delivered completion notification each cause
        # one legitimate foreground turn; neither is a maintenance prediction.
        assert len(foreground) == 6 and len(reflection) == 4
        assert loaded_child.info.usage.input_tokens == saved_child.input_tokens + 11
        assert loaded_root.info.usage.input_tokens == saved_root.input_tokens + 22
        assert (
            loaded_root.info.usage.attributed_input_tokens
            == saved_root.attributed_input_tokens + 11
        )
        assert _d2_maintenance_rows(store, root_id) == own_root
        assert _d2_maintenance_rows(store, child_id) == own_child
        assert _d2_maintenance_rows(store, root_id, "MaintenanceUsageAttributed") == attributed
    finally:
        await registry.close_all()


@pytest.mark.parametrize("stage", ["begin", "write", "commit"])
async def test_d2_manual_native_receipt_real_transaction_all_ancestors_retry(
    public_registry,
    instrument_connections,
    stage,
    monkeypatch,
):
    """Manual receipt/transaction seam; NOT normal Atom reflection integration."""
    from nooa.unifiedllm._accounting import _accounting_receiver
    from nooa.unifiedllm.request import LLMRequest

    # The coordinator can be any batch index; fault an actual active-tag
    # write rather than assuming that nooa_batch_1 is an attached database.
    original_write = InstrumentedConnection.executemany

    def write_seam(connection, sql, *args):
        if connection.stage == "write" and ".active_tags" in sql:
            raise connection.failure
        return original_write(connection, sql, *args)

    monkeypatch.setattr(InstrumentedConnection, "executemany", write_seam)
    registry, options, made = public_registry
    root = await registry.create(options)
    child = await registry.create(root.options.inherit(kept=True), parent_id=root.id)
    leaf = await registry.create(child.options.inherit(kept=True), parent_id=child.id)
    sessions = (root, child, leaf)
    owner = leaf._maintenance
    pending = owner.begin(leaf._maintenance_ancestors())
    assert set(pending.ancestor_ids) == {root.id, child.id}
    with _accounting_receiver(owner.capture):
        response = await leaf._agent.llm.acall(
            LLMRequest(messages=[{"role": "user", "content": "actual native receipt"}])
        )
    pending.response = response
    owner.end(pending)
    assert response.usage.input_tokens == 7
    assert leaf.info.usage.input_tokens == 7
    assert leaf.info.usage.last_input_tokens == 7
    before = state([session.handle.storage for session in sessions])
    notifications = []
    unsubscribe = [session.subscribe(notifications.append) for session in sessions]
    sentinel = RuntimeError(f"maintenance {stage} sentinel")
    coordinator = min(
        sessions, key=lambda s: str(s.handle.storage._canonical_path)
    ).handle.storage._conn
    coordinator.failure, coordinator.stage = sentinel, stage
    try:
        with pytest.raises(RuntimeError) as caught:
            await owner.finish()
        assert caught.value is sentinel
        assert state([session.handle.storage for session in sessions]) == before
        assert owner.passes == [pending] and owner.handle is leaf.handle
        assert not leaf.handle.closed
        assert leaf.info.usage.input_tokens == 7
        assert root.info.usage.attributed_input_tokens == 0
        assert child.info.usage.attributed_input_tokens == 0
        assert notifications == []
        with pytest.raises(RuntimeError, match="unresolved maintenance writer"):
            registry.store.open(leaf.id)
    finally:
        coordinator.stage = None
    # Explicit retry joins the original owned writer; sticky first failure is
    # still observable although durable persistence now succeeds exactly once.
    with pytest.raises(RuntimeError) as caught:
        await owner.finish()
    assert caught.value is sentinel
    assert owner.passes == []
    own = _d2_maintenance_rows(registry.store, leaf.id)
    assert len(own) == 1
    assert leaf.info.usage.input_tokens == 7
    for ancestor in (root, child):
        rows = _d2_maintenance_rows(registry.store, ancestor.id, "MaintenanceUsageAttributed")
        assert len(rows) == 1
        assert rows[0]["logical_call_id"] == own[0]["logical_call_id"]
        assert rows[0]["source_session_id"] == leaf.id
        assert ancestor.info.usage.attributed_input_tokens == 7
    assert len(notifications) == 3
    assert all(type(update).__name__ == "MaintenanceUsageUpdate" for update in notifications)
    committed = state([session.handle.storage for session in sessions])
    with pytest.raises(RuntimeError) as caught:
        await owner.finish()
    assert caught.value is sentinel
    assert state([session.handle.storage for session in sessions]) == committed
    assert len(notifications) == 3
    for stop in unsubscribe:
        stop()


async def _d2_manual_paid_receipt(session):
    from nooa.unifiedllm._accounting import _accounting_receiver
    from nooa.unifiedllm.request import LLMRequest

    owner = session._maintenance
    pending = owner.begin(session._maintenance_ancestors())
    try:
        with _accounting_receiver(owner.capture):
            pending.response = await session._agent.llm.acall(
                LLMRequest(messages=[{"role": "user", "content": "paid receipt"}])
            )
    finally:
        owner.end(pending)
    return owner


async def test_d2_manual_pending_write_close_retains_actual_writer_outside_publication(
    public_registry,
    instrument_connections,
):
    """Manual transaction/release qualification, not reflection integration."""
    registry, options, made = public_registry
    root = await registry.create(options)
    child = await registry.create(root.options.inherit(kept=True), parent_id=root.id)
    owner = await _d2_manual_paid_receipt(child)
    handle = child.handle
    coordinator = min(
        (root, child), key=lambda s: str(s.handle.storage._canonical_path)
    ).handle.storage._conn
    sentinel = RuntimeError("pending OWN close commit")
    coordinator.failure, coordinator.stage = sentinel, "commit"
    try:
        with pytest.raises(RuntimeError) as caught:
            await registry.close(child.id)
        assert caught.value is sentinel
        assert child.id not in registry.sessions
        assert child._closed and not handle.closed
        assert owner.handle is handle and owner.passes
        assert child._close_task.done() and child._closing
        assert _d2_maintenance_rows(registry.store, child.id) == []
        assert child.info.usage.input_tokens == 7
        with pytest.raises(RuntimeError, match="unresolved maintenance writer"):
            registry.store.open(child.id)
    finally:
        coordinator.stage = None
    with pytest.raises(RuntimeError) as caught:
        await registry.close(child.id)
    assert caught.value is sentinel
    assert handle.closed and not owner.passes
    assert len(_d2_maintenance_rows(registry.store, child.id)) == 1
    assert root.info.usage.attributed_input_tokens == 7
    assert child.info.usage.input_tokens == 7
    await registry.close(child.id)


async def test_d2_manual_delegated_close_failure_is_terminal_retained_identity(
    public_registry,
    monkeypatch,
):
    from nooa_atom.session.store import SessionHandle, _maintenance_key, _maintenance_owners

    registry, options, made = public_registry
    root = await registry.create(options)
    owner = await _d2_manual_paid_receipt(root)
    handle = root.handle
    original_close = SessionHandle.close
    sentinel = RuntimeError("delegated release escaped after closing")
    calls = []

    def delegated_close(candidate):
        if candidate is handle:
            calls.append(candidate)
            original_close(candidate)
            raise sentinel
        return original_close(candidate)

    monkeypatch.setattr(SessionHandle, "close", delegated_close)
    try:
        with pytest.raises(RuntimeError) as caught:
            await registry.close(root.id)
        assert caught.value is sentinel
        assert root.id not in registry.sessions and root._closed
        assert handle.closed and owner.handle is handle
        assert owner.release_started and owner.release_failure.error is sentinel
        assert _maintenance_owners[_maintenance_key(handle)] is owner
        assert len(_d2_maintenance_rows(registry.store, root.id)) == 1
        for _ in range(2):
            with pytest.raises(RuntimeError) as caught:
                await registry.close(root.id)
            assert caught.value is sentinel
            with pytest.raises(RuntimeError, match="unresolved maintenance writer"):
                registry.store.open(root.id)
        assert calls == [handle]
    finally:
        # Test-only disposal of a terminal receipt after verifying that the
        # actual delegated close succeeded before raising. Not a runtime retry
        # or a resolution policy for genuinely unreleased resources.
        _maintenance_owners.pop(_maintenance_key(handle), None)


async def test_d2_manual_maintenance_close_repeated_waiter_cancel_joins_writer(
    public_registry,
    monkeypatch,
):
    """Held finalizer seam; actual owned writer closes only after it exits."""
    import asyncio

    registry, options, made = public_registry
    session = await registry.create(options)
    owner = await _d2_manual_paid_receipt(session)
    entered, release = asyncio.Event(), asyncio.Event()
    original_finish = owner._finish

    async def held_finish(release_writer):
        entered.set()
        await release.wait()
        return await original_finish(release_writer)

    monkeypatch.setattr(owner, "_finish", held_finish)
    closing = asyncio.create_task(registry.close(session.id))
    await asyncio.wait_for(entered.wait(), 5)
    try:
        closing.cancel("first waiter cancellation")
        await asyncio.sleep(0)
        closing.cancel("second waiter cancellation")
        await asyncio.sleep(0)
        assert not closing.done()
        assert owner.task is not None and not owner.task.done()
        assert not session.handle.closed and owner.handle is session.handle
        with pytest.raises(RuntimeError, match="unresolved maintenance writer"):
            registry.store.open(session.id)
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError) as caught:
        await closing
    assert caught.value.args == ("first waiter cancellation",)
    assert owner.task.done() and session._close_task.done()
    assert session.handle.closed
    assert len(_d2_maintenance_rows(registry.store, session.id)) == 1
    assert session.info.usage.input_tokens == 7


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
@pytest.mark.parametrize("measured", [False, True])
async def test_d2_manual_native_seven_fields_distinguish_unknown_measured_zero(
    native_models,
    public_registry,
    family,
    measured,
):
    """Actual native receipt seam; not a substitute for normal reflection turns."""
    import httpx
    from nooa_atom.session.items import USAGE_FIELDS

    def handler(request):
        data = _d2_atom_native_payload(family, content="answer", input_tokens=0)
        if measured:
            data["usage"].update(cost_usd=0.0)
            if family == "openai-chat":
                data["usage"].update(
                    completion_tokens=0,
                    total_tokens=0,
                    prompt_tokens_details={"cached_tokens": 0},
                    completion_tokens_details={"reasoning_tokens": 0},
                )
            else:
                data["usage"].update(
                    output_tokens=0,
                    total_tokens=0,
                    input_tokens_details={"cached_tokens": 0},
                    output_tokens_details={"reasoning_tokens": 0},
                )
        else:
            data["usage"] = {}
        return httpx.Response(200, json=data)

    view = native_models.make(family, handler=handler)
    registry, options, made = public_registry
    root = await registry.create(options.model_copy(update={"llm": view}))
    owner = await _d2_manual_paid_receipt(root)
    await owner.finish()
    rows = _d2_maintenance_rows(registry.store, root.id)
    assert len(rows) == 1
    usage = rows[0]["usage"]
    expected = dict.fromkeys(USAGE_FIELDS, 0 if measured else None)
    # Native OpenAI endpoints do not expose cache-write input usage.
    if family != "anthropic-messages" and measured:
        expected["cache_write_input_tokens"] = None
    expected["complete_fields"] = [field for field in USAGE_FIELDS if expected[field] is not None]
    assert usage == expected
    assert {field: getattr(root.info.usage, field) for field in USAGE_FIELDS} == dict.fromkeys(
        USAGE_FIELDS, 0
    )
    assert all(not client.is_closed for client in native_models.http)


async def test_d2_manual_native_offline_ancestry_reconciles_only_owned_writers(
    public_registry,
    monkeypatch,
):
    """Manual native receipt with genuinely unloaded ancestors and public reload."""
    registry, options, made = public_registry
    root = await registry.create(options)
    child = await registry.create(root.options.inherit(kept=True), parent_id=root.id)
    leaf = await registry.create(child.options.inherit(kept=True), parent_id=child.id)
    root_id, child_id, leaf_id = root.id, child.id, leaf.id
    await registry.close_all()
    # No ancestor is published and no ancestor writer is open during the call.
    leaf = await registry.load(leaf_id)
    assert set(registry.sessions) == {leaf_id}
    owner = await _d2_manual_paid_receipt(leaf)
    assert set(owner.passes[0].ancestor_ids) == {root_id, child_id}
    await owner.finish()
    own = _d2_maintenance_rows(registry.store, leaf_id)
    assert len(own) == 1
    assert _d2_maintenance_rows(registry.store, root_id, "MaintenanceUsageAttributed") == []
    assert _d2_maintenance_rows(registry.store, child_id, "MaintenanceUsageAttributed") == []
    await registry.close_all()
    for ancestor_id in (root_id, child_id):
        ancestor = await registry.load(ancestor_id)
        assert ancestor.handle is ancestor._maintenance.handle and not ancestor.handle.closed
        rows = _d2_maintenance_rows(registry.store, ancestor_id, "MaintenanceUsageAttributed")
        assert len(rows) == 1
        assert rows[0]["source_session_id"] == leaf_id
        assert rows[0]["logical_call_id"] == own[0]["logical_call_id"]
        assert rows[0]["usage"] == own[0]["usage"]
        assert ancestor.info.usage.attributed_input_tokens == 7
        await registry.close_all()
        again = await registry.load(ancestor_id)
        assert again.info.usage.attributed_input_tokens == 7
        assert (
            _d2_maintenance_rows(registry.store, ancestor_id, "MaintenanceUsageAttributed") == rows
        )
        await registry.close_all()


import os as _d2_atom_control_os  # noqa: E402 (additive integration test section)


class _D2MaintenanceControl(BaseException):
    pass


@pytest.mark.skipif(
    _d2_atom_control_os.environ.get("NOOA_D2_ISOLATED_CONTROLS") != "1",
    reason="interpreter controls run only in the private subprocess driver",
)
@pytest.mark.parametrize("control_type", [SystemExit, KeyboardInterrupt, _D2MaintenanceControl])
@pytest.mark.parametrize("stage", ["worker", "listener", "release"])
async def test_d2_isolated_manual_native_maintenance_complete_data_chain(
    public_registry,
    monkeypatch,
    control_type,
    stage,
):
    """Actual SDK receipt, manual finalizer fault; not normal reflection."""
    import asyncio

    from nooa_acp._runtime import _acp_capture, _ACPOutcome
    from nooa_atom.session.session import _MaintenanceFailure
    from nooa_atom.session.store import _maintenance_key, _maintenance_owners

    registry, options, made = public_registry
    session = await registry.create(options)
    owner = await _d2_manual_paid_receipt(session)
    handle = session.handle
    control = control_type("native maintenance " + stage)
    calls = []
    if stage == "worker":
        actual = owner._finish

        async def escaped_worker(release):
            monkeypatch.setattr(owner, "_finish", actual)
            calls.append(stage)
            await actual(release)
            raise control

        monkeypatch.setattr(owner, "_finish", escaped_worker)
    elif stage == "listener":

        def escaped_listener(update):
            if type(update).__name__ == "MaintenanceUsageUpdate":
                calls.append(stage)
                raise control

        session.subscribe(escaped_listener)
    else:
        actual = handle.close

        def escaped_release():
            monkeypatch.setattr(handle, "close", actual)
            calls.append(stage)
            actual()
            raise control

        monkeypatch.setattr(handle, "close", escaped_release)

    outer = asyncio.create_task(_acp_capture(registry.close(session.id)))
    outcome = await outer
    assert outer.exception() is None and isinstance(outcome, _ACPOutcome)
    assert outcome.failure is control
    assert session._closed and session.id not in registry.sessions
    assert handle.closed and owner.handle is handle
    assert owner.task.done() and owner.task.exception() is None
    assert isinstance(owner.task.result(), _MaintenanceFailure)
    assert owner.task.result().error is control
    assert session._close_task.done() and session._close_task.exception() is None
    assert session._close_task.result() is control
    assert len(_d2_maintenance_rows(registry.store, session.id)) == 1
    assert session.info.usage.input_tokens == 7
    try:
        for _ in range(2):
            again = await _acp_capture(owner.finish(release=True))
            assert again.failure is control
        assert calls == [stage]
        assert len(_d2_maintenance_rows(registry.store, session.id)) == 1
        if stage == "release":
            assert owner.release_failure.error is control
            assert _maintenance_owners[_maintenance_key(handle)] is owner
    finally:
        # Only test disposal of a receipt whose real delegated close completed.
        _maintenance_owners.pop(_maintenance_key(handle), None)


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
async def test_d2_manual_native_cancelled_receipts_freeze_until_actual_provider_exit(
    native_models,
    tmp_path,
    family,
):
    """Manual maintenance/lifetime qualification, not normal reflection."""
    import asyncio

    import httpx
    from nooa_atom import SessionOptions, SessionStore
    from nooa_atom.session.registry import SessionRegistry
    from public_model_fixtures import public_agent_factory

    from nooa.unifiedllm._accounting import _accounting_receiver
    from nooa.unifiedllm.admission import _admission_controller_scope
    from nooa.unifiedllm.request import LLMRequest

    entered, release, queued = (asyncio.Event() for _ in range(3))
    wire_tasks, snapshots = [], []
    active = 0
    slot = asyncio.Semaphore(1)

    class Permit:
        def release(self):
            nonlocal active
            active -= 1
            slot.release()

    class Controller:
        async def acquire(self, observer):
            nonlocal active
            if slot.locked():
                queued.set()
            await slot.acquire()
            active += 1
            observer({"outcome": "immediate"})
            return Permit()

    async def handler(request):
        wire_tasks.append(asyncio.current_task())
        entered.set()
        await release.wait()
        return httpx.Response(
            200,
            json=_d2_atom_native_payload(
                family,
                content="late paid result",
                input_tokens=900000,
            ),
        )

    model = native_models.make(family, handler=handler)
    registry = SessionRegistry(
        SessionStore(tmp_path / "sessions"),
        agent_factory=public_agent_factory,
        llm_factory=lambda alias, workspace: model,
    )
    session = await registry.create(
        SessionOptions(
            workspace=tmp_path, agent_spec="atom_test_agents:EchoAgent", model="cancel-native"
        )
    )
    handle, binding, owner = session.handle, session._model_binding, session._maintenance
    pending = owner.begin(session._maintenance_ancestors())
    calibration = model.adapter._route_calibration_record

    def receive(snapshot):
        snapshots.append(snapshot)
        owner.capture(snapshot)

    async def predict():
        async with binding.capture(owner=session._agent) as selected:
            return await selected.acall(
                LLMRequest(messages=[{"role": "user", "content": "cancel"}])
            )

    close_task = None
    try:
        with _accounting_receiver(receive), _admission_controller_scope(Controller()):
            first = asyncio.create_task(predict())
            await asyncio.wait_for(entered.wait(), 2)
            second = asyncio.create_task(predict())
            await asyncio.wait_for(queued.wait(), 2)
            second.cancel("queued")
            with pytest.raises(asyncio.CancelledError):
                await second
            first.cancel("entered")
            with pytest.raises(asyncio.CancelledError):
                await first
            owner.end(pending)
            await owner.finish()
        assert len(snapshots) == 2 and len(wire_tasks) == active == 1
        assert all(
            not attempt.response_received and attempt.finished
            for snapshot in snapshots
            for attempt in snapshot.attempts
        )
        frozen = tuple(snapshots)
        ratio = calibration.ratio()
        rows = _d2_maintenance_rows(registry.store, session.id)
        assert len(rows) == 2
        by_id = {snapshot.logical_id: snapshot for snapshot in snapshots}
        for row in rows:
            snapshot = by_id[row["logical_call_id"]]
            for name in (
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "cache_write_input_tokens",
                "reasoning_tokens",
                "total_tokens",
                "cost_usd",
            ):
                assert row["usage"].get(name) == snapshot.subtotal(name).known
                assert (name in row["usage"]["complete_fields"]) == snapshot.subtotal(name).complete
        assert session.info.usage.input_tokens == 0
        assert binding._native and any(
            record.owner is session._agent for record in binding._records
        )
        close_task = asyncio.create_task(registry.close(session.id))
        for _ in range(10):
            await asyncio.sleep(0)
        assert not close_task.done() and not handle.closed and not wire_tasks[0].done()
        assert active == 1 and not native_models.http[-1].is_closed
        release.set()
        await asyncio.wait_for(wire_tasks[0], 2)
        await asyncio.wait_for(close_task, 2)
        assert active == 0 and handle.closed and session._closed
        assert not binding._native and all(
            record.owner is not session._agent for record in binding._records
        )
        assert tuple(snapshots) == frozen and calibration.ratio() == ratio
        assert _d2_maintenance_rows(registry.store, session.id) == rows
        assert registry.store.get(session.id).usage.input_tokens == 0
        assert not native_models.http[-1].is_closed and native_models.sdks[-1].max_retries == 9
    finally:
        release.set()
        if close_task is not None:
            await close_task
        await registry.close_all()


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
@pytest.mark.parametrize("race", ["none", "changed-notes", "close"])
async def test_d2_native_one_shot_normal_reflection_admission_close_races(
    native_models, tmp_path, family, race
):
    """Normal one-shot delegation; real Predict delivery or close, never forced admission."""
    import asyncio
    import json

    import httpx
    from nooa_atom import SessionOptions, SessionStore
    from nooa_atom.agent.factory import create_session_agent
    from nooa_atom.session.registry import SessionRegistry

    foreground, reflection = [], []
    held, release = asyncio.Event(), asyncio.Event()

    async def handler(request):
        body = json.loads(request.content)
        assert body["model"] == "native-wire-model"
        if any(
            body.get(key) == 2048
            for key in ("max_tokens", "max_output_tokens", "max_completion_tokens")
        ):
            reflection.append(body)
            child_call = "child accepted" in json.dumps(body)
            if not child_call and race != "none":
                held.set()
                await release.wait()
            plan = {
                "summary": "",
                "drop_names": [],
                "useful_findings": "one-shot finding",
                "decisions": "",
                "outstanding_work": "",
            }
            return httpx.Response(
                200, json=_d2_atom_native_payload(family, content=json.dumps(plan), input_tokens=47)
            )
        foreground.append(body)
        code = (
            "result = await Subagent('one-shot review').run('native one-shot task')\n"
            "assert result.solution_description == 'one-shot result'\n"
            "return_result(Done(message='root reply', explanation='root accepted'))"
            if len(foreground) == 1
            else "return_result(Done(result=TaskResult(solution_description='one-shot result', "
            "evidence='actual SDK', how_to_verify='normal turn'), explanation='child accepted'))"
        )
        return httpx.Response(200, json=_d2_atom_native_payload(family, code=code))

    view = native_models.make(family, handler=handler, model="native-wire-model")
    store = SessionStore(tmp_path / "sessions")
    registry = SessionRegistry(store, agent_factory=create_session_agent)
    root = await registry.create(
        SessionOptions(
            workspace=tmp_path,
            sessions_dir=tmp_path / "sessions",
            agent_spec="nooa_atom.agent:AtomAgent",
            model="configured-native-alias",
            llm=view,
            experimental_done_reflection=True,
            experimental_reflection_min_turns=1,
            experimental_reflection_min_seconds=0,
            kept=True,
        )
    )
    prompt = asyncio.create_task(root.prompt("normal one-shot review"))
    close_task = None
    try:
        if race != "none":
            await asyncio.wait_for(held.wait(), 10)
            child = root
            if race == "changed-notes":
                child._agent.context["done_reflection_notes"] = "newer authoritative notes"
            else:
                close_task = asyncio.create_task(registry.close(root.id))
                await asyncio.sleep(0.02)
                assert not close_task.done()
                assert not child.handle._closed
            release.set()
        result = await asyncio.wait_for(prompt, 15)
        assert result.explanation == "root accepted"
        if close_task is not None:
            await asyncio.wait_for(close_task, 10)
            assert root._maintenance.failure is None
        await _d2_atom_idle(root)
        if registry._background:
            await asyncio.gather(*tuple(registry._background))
        child_id = next(
            info.id for info in store.list(roots_only=False) if info.parent_id == root.id
        )
        own_root = _d2_maintenance_rows(store, root.id)
        own_child = _d2_maintenance_rows(store, child_id)
        attr = _d2_maintenance_rows(store, root.id, "MaintenanceUsageAttributed")
        assert len(foreground) == 2 and len(reflection) == 1
        assert len(own_root) == 1
        assert own_child == attr == []
        # One-shot result delivery naturally closes the child before maintenance
        # admission. Never force admission just to manufacture a child receipt.
        assert store.get(child_id).usage.input_tokens == 11
        assert child_id not in registry.sessions
        assert store.get(child_id).status == "on_disk"
        assert own_root[0]["attempt_count"] == 1
        if race == "none":
            assert root._agent.context.get("done_reflection_notes")
        if race == "close":
            assert root.handle._closed
        turns = store.load_rows(root.id, frozenset({"TurnEnded"}))
        assert len(turns) == 1 and turns[0][1]["usage"]["input_tokens"] == 11
        if race != "close":
            assert own_root[0]["usage"]["input_tokens"] == 47
            assert root.info.usage.input_tokens == 58
            assert root.info.usage.last_input_tokens == 47
        if race == "changed-notes":
            assert root._agent.context["done_reflection_notes"] == "newer authoritative notes"
        assert all(not http.is_closed for http in native_models.http)
        assert all(sdk.max_retries == 9 for sdk in native_models.sdks)
    finally:
        release.set()
        await asyncio.gather(prompt, return_exceptions=True)
        if close_task is not None:
            await asyncio.gather(close_task, return_exceptions=True)
        await registry.close_all()


async def test_d2_legacy_duck_usage_receiver_has_no_maintenance_owner():
    from types import SimpleNamespace

    from nooa_atom.session.items import SessionInfo
    from nooa_atom.session.session import Session

    from nooa.llm_types import LLMResponse, LLMUsage

    published = []
    receiver = SimpleNamespace(
        id="duck",
        info=SessionInfo(id="duck"),
        handle=SimpleNamespace(
            update_usage=lambda usage: published.append(("handle", usage.model_copy()))
        ),
        _emit=lambda update: published.append(("update", update)),
    )
    Session._count_usage(
        receiver, LLMResponse(content="legacy", usage=LLMUsage(input_tokens=11, output_tokens=7))
    )
    assert not hasattr(receiver, "_maintenance")
    assert receiver.info.usage.input_tokens == 11 and receiver.info.usage.output_tokens == 7
    assert receiver.info.usage.last_input_tokens == 11
    assert [kind for kind, _ in published] == ["handle", "update"]


@pytest.mark.parametrize("known", [None, 0, 11])
async def test_d2_legacy_maintenance_live_known_fields_identity_only(public_registry, known):
    from nooa_atom.session.items import USAGE_FIELDS

    from nooa.llm_types import LLMResponse, LLMUsage

    registry, options, _ = public_registry
    session = await registry.create(options)
    session.info.usage.last_input_tokens = 23
    pending = session._maintenance.begin(())
    supplied = dict.fromkeys(USAGE_FIELDS, known)
    supplied["output_tokens"] = 7 if known == 11 else known
    response = LLMResponse(content="legacy response", usage=LLMUsage().model_copy(update=supplied))
    session._count_usage(response)
    frozen = tuple((field, value) for field, value in supplied.items() if value is not None)
    assert pending.legacy_responses == ((response, frozen),)
    expected = {field: value or 0 for field, value in supplied.items()}
    assert {field: getattr(session.info.usage, field) for field in USAGE_FIELDS} == expected
    assert session.info.usage.last_input_tokens == (23 if known is None else known)
    response.usage.input_tokens = 999999
    session._count_usage(response)
    assert pending.legacy_responses[0][1] == frozen
    assert session.info.usage.input_tokens == (known or 0)
    # Equal content/values are not logical-call identity: new response, new charge.
    second = LLMResponse(content="legacy response", usage=LLMUsage().model_copy(update=supplied))
    session._count_usage(second)
    assert len(pending.legacy_responses) == 2
    assert {field: getattr(session.info.usage, field) for field in USAGE_FIELDS} == {
        field: 2 * value for field, value in expected.items()
    }
    assert pending.snapshots == pending.records == () and not pending.charged
    session._maintenance.end(pending)
    await session._maintenance.finish()
    assert not session._maintenance.passes
    assert _d2_maintenance_rows(registry.store, session.id) == []
    # LIVE compatibility only: no invented durable maintenance record.


async def test_d2_legacy_live_publication_control_retains_once_charge(public_registry, monkeypatch):
    from nooa.llm_types import LLMResponse, LLMUsage

    registry, options, _ = public_registry
    session = await registry.create(options)
    owner = session._maintenance
    pending = owner.begin(())
    control = _D2MaintenanceControl("legacy live update control")
    actual = session.handle.update_usage
    calls = []

    def update(usage):
        calls.append(usage)
        actual(usage)
        raise control

    monkeypatch.setattr(session.handle, "update_usage", update)
    response = LLMResponse(content="legacy", usage=LLMUsage(input_tokens=11, output_tokens=7))
    session._count_usage(response)
    session._count_usage(response)
    assert len(calls) == 1 and session.info.usage.input_tokens == 11
    assert owner.failure.error is control and pending.snapshots == pending.records == ()
    owner.end(pending)
    from nooa_acp._runtime import _acp_capture

    result = await _acp_capture(owner.finish())
    assert result.failure is control
    assert not owner.passes and _d2_maintenance_rows(registry.store, session.id) == []
    # Intentional control was observed; fixture close must still own the actual writer.
    result = await _acp_capture(registry.close_all())
    assert result.failure is control and session.handle.closed


@pytest.mark.parametrize("family", ["openai-chat", "openai-responses", "anthropic-messages"])
@pytest.mark.parametrize("missing_usage", [False, True])
async def test_d2_native_maintenance_canonical_delivery_never_legacy_double_counts(
    native_models, public_registry, family, missing_usage
):
    import httpx

    from nooa.unifiedllm._accounting import (
        _accounting_receiver,
        _invocation_delivery,
        _owns_delivered_response,
    )
    from nooa.unifiedllm.request import LLMRequest

    def handler(request):
        body = _d2_atom_native_payload(family, content="canonical", input_tokens=7)
        if missing_usage:
            body.pop("usage", None)
        return httpx.Response(200, json=body)

    view = native_models.make(family, handler=handler)
    registry, options, _ = public_registry
    session = await registry.create(options.model_copy(update={"llm": view}))
    owner = session._maintenance
    pending = owner.begin(())
    try:
        with _accounting_receiver(owner.capture), _invocation_delivery():
            async with session._agent.llms["main"].capture(owner=session._agent) as held:
                response = await held.acall(
                    LLMRequest(messages=[{"role": "user", "content": "real canonical receipt"}])
                )
            assert _owns_delivered_response(response)
            session._count_usage(response)
            session._count_usage(response)
            assert pending.legacy_responses == ()
            assert len(pending.snapshots) == 1
    finally:
        owner.end(pending)
    await owner.finish()
    rows = _d2_maintenance_rows(registry.store, session.id)
    assert len(rows) == 1 and not owner.passes
    assert session.info.usage.input_tokens == (0 if missing_usage else 7)
    assert rows[0]["usage"]["input_tokens"] == (None if missing_usage else 7)
    await owner.finish()
    assert _d2_maintenance_rows(registry.store, session.id) == rows
    assert all(not client.is_closed for client in native_models.http)


async def test_d2_legacy_missing_usage_delivery_freezes_empty_once(public_registry):
    from nooa.llm_types import LLMResponse, LLMUsage

    registry, options, _ = public_registry
    session = await registry.create(options)
    pending = session._maintenance.begin(())
    response = LLMResponse(content="legacy unavailable", usage=None)
    session._count_usage(response)
    assert pending.legacy_responses == ((response, ()),)
    response.usage = LLMUsage(input_tokens=11, output_tokens=7)
    session._count_usage(response)
    assert session.info.usage.input_tokens == session.info.usage.output_tokens == 0
    session._maintenance.end(pending)
    await session._maintenance.finish()
    assert not session._maintenance.passes
    assert _d2_maintenance_rows(registry.store, session.id) == []


def test_d2_legacy_duck_maintenance_getter_error_is_not_suppressed():
    from nooa_atom.session.session import Session

    from nooa.llm_types import LLMResponse, LLMUsage

    error = RuntimeError("actual duck getter")

    class Receiver:
        @property
        def _maintenance(self):
            raise error

    with pytest.raises(RuntimeError) as caught:
        Session._count_usage(Receiver(), LLMResponse(content="legacy", usage=LLMUsage()))
    assert caught.value is error


async def test_d2_legacy_live_only_explicit_fields_no_absent_cache_or_price(public_registry):
    from nooa.llm_types import LLMResponse, LLMUsage

    registry, options, _ = public_registry
    session = await registry.create(options)
    session.info.usage.last_input_tokens = 23
    pending = session._maintenance.begin(())
    response = LLMResponse(content="legacy", usage=LLMUsage(output_tokens=7))
    session._count_usage(response)
    assert pending.legacy_responses == ((response, (("output_tokens", 7),)),)
    assert session.info.usage.last_input_tokens == 23
    assert session.info.usage.output_tokens == 7
    session._maintenance.end(pending)
    await session._maintenance.finish()
    assert _d2_maintenance_rows(registry.store, session.id) == []


import asyncio as _d2_boundary_asyncio  # noqa: E402 (additive integration test section)


@pytest.mark.parametrize(
    "control_type", [RuntimeError, _d2_boundary_asyncio.CancelledError, _D2MaintenanceControl]
)
async def test_d2_legacy_live_observer_controls_are_original_data(public_registry, control_type):
    from nooa_acp._runtime import _acp_capture
    from nooa_atom.session.items import UsageChangedUpdate

    from nooa.llm_types import LLMResponse, LLMUsage

    registry, options, _ = public_registry
    session = await registry.create(options)
    owner = session._maintenance
    pending = owner.begin(())
    control = control_type("actual legacy usage observer")
    observed = []

    def listener(update):
        assert isinstance(update, UsageChangedUpdate)
        assert session.info.usage.input_tokens == update.usage.input_tokens == 11
        raise control

    remove = session.subscribe(listener)
    session.subscribe(observed.append)
    response = LLMResponse(content="legacy", usage=LLMUsage(input_tokens=11, output_tokens=7))
    session._count_usage(response)
    remove()
    session._count_usage(response)
    assert len(observed) == 1 and owner.failure.error is control
    owner.end(pending)
    assert (await _acp_capture(owner.finish())).failure is control
    assert not owner.passes and _d2_maintenance_rows(registry.store, session.id) == []
    for _ in range(2):
        assert (await _acp_capture(session.close())).failure is control
    # Already-closed registry fanout remains an ordinary successful no-op.
    cleanup = await _acp_capture(registry.close_all())
    assert cleanup.failure is None
    assert session.handle.closed and owner.failure.error is control
