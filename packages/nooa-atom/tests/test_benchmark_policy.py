# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real AtomHarness/Session ownership with strict fake policy clients, no provider."""

import asyncio
import inspect
import json

import pytest
from atom_test_agents import CellLLM, cell
from nooa_atom.agent.base import AtomAgentBase
from nooa_atom.agent.done_reflection import DoneReflectionPlan
from nooa_atom.benchmark_policy import POLICY, BenchmarkPolicyGuard, BenchmarkPolicyViolation
from nooa_atom.gym import AtomHarness
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_bench.gym import _current_invocation, _RecordingClient, _ToolHooks

from nooa.runtime.hooks import hooks_scope

DONE = "return_result(Done(explanation='done', result=TaskResult(solution_description='s', evidence='e', how_to_verify='v', report='accepted')))"


@pytest.fixture(autouse=True)
def no_optional_viewer_probe(monkeypatch):
    monkeypatch.setattr("nooa.agent._auto_tracing_attempted", True)


class PolicyFake(CellLLM):
    bind_episode_policy = _RecordingClient.bind_episode_policy
    _check_episode_policy = _RecordingClient._check_episode_policy

    def __init__(self, responses):
        super().__init__(responses)
        self._episode_policy_guard = self._required_episode_policy = None
        self.requests = []

    async def acall(self, *args, **kwargs):
        self._check_episode_policy(kwargs.get("output_model"))
        self.requests.append((_current_invocation.get(), kwargs.get("output_model")))
        return await super().acall(*args, **kwargs)


def harness(tmp_path, responses, *, mode="session", **kwargs):
    client = PolicyFake(responses)
    result = AtomHarness(
        llm=client,
        workdir=tmp_path,
        sessions_dir=tmp_path / "sessions",
        benchmark_policy=POLICY,
        experimental_namespace=mode,
        experimental_done_reflection=False,
        benchmark_allow_session_titles=False,
        benchmark_policy_path=tmp_path / "episode.json.atom-policy.json",
        experimental_reflection_min_turns=1,
        experimental_reflection_min_seconds=0,
        **kwargs,
    )
    return result, client


async def solve(h):
    tools = []
    with hooks_scope(_ToolHooks(tools)):
        return await h.solve("test task")


@pytest.mark.parametrize("mode", ["legacy", "call", "session"])
async def test_ordinary_harness_has_explicit_coverage_and_persisted_invocation_map(tmp_path, mode):
    h, client = harness(tmp_path, [cell(DONE)], mode=mode)
    try:
        result = await solve(h)
    except BenchmarkPolicyViolation:
        pytest.fail(str(h.benchmark_policy_receipt()["violations"]))
    assert result.report == "accepted"
    receipt = h.benchmark_policy_receipt()
    assert receipt == json.loads(h.benchmark_guard.path.read_text())
    assert receipt["verdict"] == "pass" and receipt["phase"] == "final"
    [session] = receipt["sessions"]
    assert session["registered"] and session["ready"] and session["closed"]
    assert session["runtime_reflection"] is False and session["runtime_namespace"] == mode
    assert session["effective_options"]["benchmark_allow_session_titles"] is False
    assert session["initial"] == session["final"]
    mapping = {row["invocation_id"]: row["session_id"] for row in receipt["invocations"]}
    assert len(client.requests) == 1
    invocation, schema = client.requests[0]
    assert schema is None and mapping[invocation] == session["session_id"]
    assert invocation != session["session_id"]
    assert not h.benchmark_guard._unsubscribe


@pytest.mark.parametrize(
    "operation,code",
    [
        ("reflection", "await self._reflect_done_plan({})"),
        (
            "title_generation",
            "await self.name_session('DoneReflectionPlan _reflect_done_plan quoted')",
        ),
        ("rename", "await self.rename_session('renamed')"),
        ("rename", "await self.session.rename('renamed')"),
        ("rename", "await self.session._session.set_title('renamed', user_set=True)"),
        ("rename", "await self.session._session.set_title('')"),
    ],
)
@pytest.mark.parametrize("mode", ["legacy", "call", "session"])
async def test_caught_forbidden_attempt_cannot_be_overridden_by_successful_done(
    tmp_path, operation, code, mode
):
    h, client = harness(
        tmp_path, [cell("try:\n " + code + "\nexcept RuntimeError:\n pass\n" + DONE)], mode=mode
    )
    with pytest.raises(BenchmarkPolicyViolation):
        await solve(h)
    receipt = h.benchmark_policy_receipt()
    assert receipt["verdict"] == "fail"
    assert operation in {row["code"] for row in receipt["violations"]}
    assert len(client.requests) == 1 and client.requests[0][1] is None
    assert all(row["initial"] == row["final"] for row in receipt["sessions"])


@pytest.mark.parametrize("mode", ["legacy", "call", "session"])
async def test_child_description_never_becomes_title_and_child_is_covered(tmp_path, mode):
    h, client = harness(
        tmp_path,
        [
            cell(
                "r=await self.jobs.launch(Subagent('DO NOT TURN THIS DESCRIPTION INTO A TITLE').run('work'), name='kid')\nassert r.report == 'accepted'\n"
                + DONE
            ),
            cell(DONE),
        ],
        mode=mode,
    )
    try:
        result = await solve(h)
    except BenchmarkPolicyViolation:
        pytest.fail(str(h.benchmark_policy_receipt()["violations"]))
    assert result.report == "accepted"
    receipt = h.benchmark_policy_receipt()
    assert receipt["verdict"] == "pass" and len(receipt["sessions"]) == 2
    root, child = sorted(receipt["sessions"], key=lambda row: row["parent_id"] is not None)
    assert child["parent_id"] == root["session_id"]
    assert child["initial"]["name"] == "kid"
    assert root["initial"]["title"] is child["initial"]["title"] is None
    assert all(
        row["registered"] and row["ready"] and row["runtime_reflection"] is False
        for row in receipt["sessions"]
    )
    mapping = {row["invocation_id"]: row["session_id"] for row in receipt["invocations"]}
    assert {mapping[call_id] for call_id, _ in client.requests} == {
        root["session_id"],
        child["session_id"],
    }


CONFIG = {
    "benchmark_policy": POLICY,
    "experimental_namespace": "session",
    "experimental_done_reflection": False,
    "benchmark_allow_session_titles": False,
}


@pytest.mark.parametrize("key", list(CONFIG))
def test_missing_explicit_configuration_is_rejected_by_constructor_and_options(tmp_path, key):
    config = {k: v for k, v in CONFIG.items() if k != key}
    client = PolicyFake([])
    path = tmp_path / "policy.json"
    with pytest.raises(ValueError):
        AtomHarness(llm=client, benchmark_policy_path=path, **config)
    receipt = json.loads(path.read_text())
    assert receipt["phase"] == "final" and receipt["verdict"] == "fail"
    assert not client.requests
    with pytest.raises(ValueError):
        SessionOptions(workspace=tmp_path, agent_spec="unused", **config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("experimental_done_reflection", True),
        ("experimental_done_reflection", 0),
        ("experimental_done_reflection", "false"),
        ("benchmark_allow_session_titles", True),
        ("benchmark_allow_session_titles", 0),
        ("benchmark_allow_session_titles", "false"),
        ("benchmark_policy", "unknown"),
        ("experimental_namespace", "unknown"),
    ],
)
def test_invalid_policy_is_not_coerced_or_ignored(tmp_path, key, value):
    config = {**CONFIG, key: value}
    path = tmp_path / "policy.json"
    with pytest.raises(ValueError):
        AtomHarness(llm=PolicyFake([]), benchmark_policy_path=path, **config)
    assert json.loads(path.read_text())["verdict"] == "fail"
    with pytest.raises(ValueError):
        SessionOptions(workspace=tmp_path, agent_spec="unused", **config)


@pytest.mark.parametrize("mode", ["legacy", "call", "session"])
def test_policy_inheritance_is_explicit_and_preserves_routing_names(tmp_path, mode):
    root = SessionOptions(
        workspace=tmp_path,
        agent_spec="unused",
        name="host_root",
        **{**CONFIG, "experimental_namespace": mode},
    )
    child = root.inherit(name="worker_route", kept=True)
    assert child.benchmark_policy == POLICY and child.experimental_namespace == mode
    assert (
        child.experimental_done_reflection is False
        and child.benchmark_allow_session_titles is False
    )
    assert child.name == "worker_route" and root.name == "host_root"
    with pytest.raises(ValueError):
        root.inherit(experimental_done_reflection=True)


def test_worker_only_extras_remain_compatible_but_experiment_typos_fail(tmp_path):
    client = PolicyFake([])
    with pytest.raises(ValueError, match="unknown benchmark/experiment keys"):
        AtomHarness(
            llm=client,
            benchmark_policy_path=tmp_path / "policy.json",
            experimental_unknown=False,
            **CONFIG,
        )
    assert json.loads((tmp_path / "policy.json").read_text())["verdict"] == "fail"


@pytest.mark.parametrize("mode", ["legacy", "call", "session"])
@pytest.mark.parametrize(
    "method,operation",
    [
        ("_reflect_done_plan({})", "reflection"),
        ("name_session('quoted markers')", "title_generation"),
    ],
)
async def test_child_forbidden_generation_is_counted_before_predict(
    tmp_path, mode, method, operation
):
    h, client = harness(
        tmp_path,
        [
            cell(
                "await self.jobs.launch(Subagent('context only').run('work'), name='kid')\n" + DONE
            ),
            cell("try:\n await self." + method + "\nexcept RuntimeError:\n pass\n" + DONE),
        ],
        mode=mode,
    )
    with pytest.raises(BenchmarkPolicyViolation):
        await solve(h)
    receipt = h.benchmark_policy_receipt()
    assert receipt["observed_attempt_counts"][operation] == 1
    assert all(schema is None for _, schema in client.requests) and len(client.requests) == 2
    [child] = [row for row in receipt["sessions"] if row["parent_id"]]
    assert any(
        row["session_id"] == child["session_id"] and row["code"] == operation
        for row in receipt["violations"]
    )
    assert child["runtime_reflection"] is False and child["ready"]


@pytest.mark.parametrize("user_set", [False, True])
async def test_noop_and_rename_restore_attempts_are_counted_before_early_return(tmp_path, user_set):
    code = f"self.session._session.info.title_is_user_set = {user_set!r}\n"
    code += (
        "for title in (None, 'new', None):\n try:\n  await self.session._session.set_title(title)\n except RuntimeError:\n  pass\n"
        + DONE
    )
    h, _ = harness(tmp_path, [cell(code)])
    with pytest.raises(BenchmarkPolicyViolation):
        await solve(h)
    receipt = h.benchmark_policy_receipt()
    assert receipt["observed_attempt_counts"]["rename"] == 3
    assert receipt["sessions"][0]["initial"] == receipt["sessions"][0]["final"]


@pytest.mark.parametrize(
    "fault", ["owner", "agent_middleware", "llm_middleware", "observer", "runtime", "option"]
)
@pytest.mark.parametrize("child_only", [False, True])
async def test_missing_installation_and_drift_fail_before_inference_and_close_half_built(
    tmp_path, monkeypatch, fault, child_only
):
    original = BenchmarkPolicyGuard.install

    def broken(guard, session):
        original(guard, session)
        if child_only and session.parent_id is None:
            return
        if fault == "owner":
            session._benchmark_guard = None
        elif fault in {"agent_middleware", "llm_middleware"}:
            key = "agent_call" if fault == "agent_middleware" else "llm_call"
            session._agent.event_manager._middleware[key].clear()
        elif fault == "observer":
            session._agent.event_manager._handlers["BeforeTurn"].clear()
        elif fault == "runtime":
            session._agent._done_reflection = True
        else:
            session.options.experimental_done_reflection = True

    monkeypatch.setattr(BenchmarkPolicyGuard, "install", broken)
    code = (
        "try:\n await self.jobs.launch(Subagent('work').run('task'), name='kid')\nexcept RuntimeError:\n pass\n"
        + DONE
        if child_only
        else DONE
    )
    h, client = harness(tmp_path, [cell(code)])
    with pytest.raises(BenchmarkPolicyViolation):
        await solve(h)
    assert len(client.requests) == int(child_only)
    receipt = h.benchmark_policy_receipt()
    assert receipt["verdict"] == "fail"
    assert all(row["closed"] for row in receipt["sessions"])
    assert not h.benchmark_guard._unsubscribe


async def test_missing_fresh_context_readiness_blocks_first_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(BenchmarkPolicyGuard, "ready", lambda *args: None)
    h, client = harness(tmp_path, [cell(DONE)])
    with pytest.raises(BenchmarkPolicyViolation):
        await solve(h)
    assert not client.requests
    assert not h.benchmark_policy_receipt()["sessions"][0]["ready"]


async def test_missing_outer_invocation_hook_fails_closed(tmp_path):
    h, client = harness(tmp_path, [cell(DONE)])
    with pytest.raises(BenchmarkPolicyViolation):
        await h.solve("no outer correlation hook")
    assert not client.requests
    assert "model_invocation_uncovered" in {
        row["code"] for row in h.benchmark_policy_receipt()["violations"]
    }


@pytest.mark.parametrize(
    "mutation",
    [
        "self._done_reflection=True",
        "self._namespace_lifetime='call'",
        "self.session._session.options.experimental_done_reflection=True",
        "self.session._session._benchmark_guard=None",
    ],
)
async def test_done_cannot_hide_runtime_drift_after_last_model_call(tmp_path, mutation):
    h, client = harness(tmp_path, [cell(mutation + "\n" + DONE)])
    with pytest.raises(BenchmarkPolicyViolation):
        await solve(h)
    assert h.benchmark_policy_receipt()["verdict"] == "fail"
    assert len(client.requests) == 1


async def test_low_level_title_mutation_and_restore_are_observed_not_just_final_equality(
    tmp_path, monkeypatch
):
    original = BenchmarkPolicyGuard.install

    def fixed_host_metadata(guard, session):
        # Fixed trusted host metadata, not model/description titling. The
        # released harness itself leaves display titles unset by default.
        session.handle.set_title("fixed_host_title")
        session.info.title = "fixed_host_title"
        original(guard, session)

    monkeypatch.setattr(BenchmarkPolicyGuard, "install", fixed_host_metadata)
    code = (
        "owner=self.session._session\nowner.handle.set_title('changed')\nowner.handle.set_title('fixed_host_title')\n"
        + DONE
    )
    h, _ = harness(tmp_path, [cell(code)])
    with pytest.raises(BenchmarkPolicyViolation):
        await solve(h)
    receipt = h.benchmark_policy_receipt()
    assert receipt["observed_attempt_counts"]["title_mutation"] == 2
    assert receipt["sessions"][0]["initial"] == receipt["sessions"][0]["final"]


async def test_pre_first_turn_automatic_title_attempt_is_not_an_initialization_loophole(
    tmp_path, monkeypatch
):
    original = SessionRegistry.create

    async def old_style_prepare(registry, options, **kwargs):
        if kwargs.get("parent_id"):
            prepare = kwargs.get("prepare")

            async def attempt(child):
                try:
                    await child.set_title("automatic description title")
                except BenchmarkPolicyViolation:
                    pass
                if prepare:
                    await prepare(child)

            kwargs["prepare"] = attempt
        return await original(registry, options, **kwargs)

    monkeypatch.setattr(SessionRegistry, "create", old_style_prepare)
    code = (
        "try:\n await self.jobs.launch(Subagent('description').run('work'), name='kid')\nexcept RuntimeError:\n pass\n"
        + DONE
    )
    h, client = harness(tmp_path, [cell(code)])
    with pytest.raises(BenchmarkPolicyViolation):
        await solve(h)
    receipt = h.benchmark_policy_receipt()
    assert receipt["observed_attempt_counts"]["rename"] == 1
    assert len(client.requests) == 1
    assert all(
        row["initial"]["title"] is row["final"]["title"] is None for row in receipt["sessions"]
    )


@pytest.mark.parametrize("mode", ["legacy", "call", "session"])
async def test_ordinary_dynamic_provider_remains_available_under_policy(
    tmp_path, monkeypatch, mode
):
    original = BenchmarkPolicyGuard.install
    provider = type("OrdinaryBenchmarkDynamic", (), {"__module__": "dynamic_benchmark_fixture"})

    def install(guard, session):
        original(guard, session)
        session._agent.tool = provider()

    monkeypatch.setattr(BenchmarkPolicyGuard, "install", install)
    h, _ = harness(
        tmp_path,
        [cell("assert isinstance(self.tool, OrdinaryBenchmarkDynamic)\n" + DONE)],
        mode=mode,
        run_command=None,
        extra_tools=[],
        session_name="fixed_host_root",
    )
    assert (await solve(h)).report == "accepted"
    assert h.benchmark_policy_receipt()["sessions"][0]["initial"]["name"] == "fixed_host_root"


async def test_verdict_write_failure_and_missing_verdict_do_not_return_success(
    tmp_path, monkeypatch
):
    h, _ = harness(tmp_path, [cell(DONE)])
    original = BenchmarkPolicyGuard._write

    def fail_final(guard):
        if guard.finalized:
            raise OSError("simulated verdict write failure")
        return original(guard)

    monkeypatch.setattr(BenchmarkPolicyGuard, "_write", fail_final)
    with pytest.raises(OSError, match="simulated verdict"):
        await solve(h)
    assert json.loads(h.benchmark_guard.path.read_text())["phase"] == "pending"
    assert all(entry["session"]._closed for entry in h.benchmark_guard.sessions.values())
    assert not h.benchmark_guard._unsubscribe
    with pytest.raises(BenchmarkPolicyViolation):
        h.benchmark_policy_receipt()


async def test_late_attempt_invalidates_sealed_pass_and_receipts_are_detached_copies(tmp_path):
    h, _ = harness(tmp_path, [cell(DONE)])
    await solve(h)
    copy = h.benchmark_policy_receipt()
    copy["verdict"] = "not authoritative"
    assert h.benchmark_policy_receipt()["verdict"] == "pass"
    session = next(iter(h.benchmark_guard.sessions.values()))["session"]
    with pytest.raises(BenchmarkPolicyViolation):
        await session.set_title("late title")
    assert h.benchmark_policy_receipt()["verdict"] == "fail"
    assert json.loads(h.benchmark_guard.path.read_text())["observed_attempt_counts"]["rename"] == 1


async def test_model_error_closes_tree_and_still_writes_final_policy_receipt(tmp_path):
    h, _ = harness(tmp_path, [])
    with pytest.raises(RuntimeError):
        await solve(h)
    receipt = h.benchmark_policy_receipt()
    assert receipt["phase"] == "final" and receipt["outcome"] == "error"
    assert receipt["verdict"] == "pass"  # policy compliance is not task success
    assert all(row["closed"] for row in receipt["sessions"])
    assert not h.benchmark_guard._unsubscribe


async def test_cancellation_closes_owned_tree_and_preserves_cancellation(tmp_path):
    h, client = harness(tmp_path, [cell("await asyncio.sleep(1000)")])
    task = asyncio.create_task(solve(h))
    try:
        async with asyncio.timeout(20):
            while not client.requests:
                await asyncio.sleep(0.01)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    receipt = h.benchmark_policy_receipt()
    assert receipt["phase"] == "final" and receipt["outcome"] == "cancelled"
    assert all(row["closed"] for row in receipt["sessions"])
    assert not h.benchmark_guard._unsubscribe


class DirectRoutes:
    """Plain Python methods: deliberately outside Agent method middleware."""

    async def raw_rename(self, agent):
        await inspect.unwrap(AtomAgentBase.rename_session)(agent, "unwrapped title")

    async def raw_prediction(self, agent):
        await agent.llm.acall(
            [{"role": "user", "content": "direct plan request"}], output_model=DoneReflectionPlan
        )

    async def disabled_maintenance(self, agent):
        from nooa.interactive import Done

        await agent._reflect_accepted_done(Done(explanation="already delivered"))


@pytest.mark.parametrize("route,kind", [("raw_rename", "rename"), ("raw_prediction", "reflection")])
async def test_unwrapped_routes_use_owner_or_client_boundaries_not_swallowed_hooks(
    tmp_path, monkeypatch, route, kind
):
    original = BenchmarkPolicyGuard.install

    def install(guard, session):
        original(guard, session)
        session._agent.probe = DirectRoutes()

    monkeypatch.setattr(BenchmarkPolicyGuard, "install", install)
    h, client = harness(
        tmp_path,
        [cell(f"try:\n await self.probe.{route}(self)\nexcept RuntimeError:\n pass\n" + DONE)],
    )
    with pytest.raises(BenchmarkPolicyViolation):
        await solve(h)
    receipt = h.benchmark_policy_receipt()
    assert receipt["observed_attempt_counts"][kind] == 1
    assert len(client.requests) == 1 and client.requests[0][1] is None
    # There was no wrapped dispatch under these forbidden method names.
    assert not {"rename_session", "_reflect_done_plan"} & {
        row["method"] for row in receipt["invocations"]
    }


async def test_unwrapped_maintenance_is_disabled_without_generation_or_mutation(
    tmp_path, monkeypatch
):
    original = BenchmarkPolicyGuard.install

    def install(guard, session):
        original(guard, session)
        session._agent.probe = DirectRoutes()

    monkeypatch.setattr(BenchmarkPolicyGuard, "install", install)
    h, client = harness(tmp_path, [cell("await self.probe.disabled_maintenance(self)\n" + DONE)])
    assert (await solve(h)).report == "accepted"
    receipt = h.benchmark_policy_receipt()
    assert receipt["verdict"] == "pass" and len(client.requests) == 1
    assert all(count == 0 for count in receipt["observed_attempt_counts"].values())


async def test_unregistered_child_fails_closed_and_is_closed_before_publication(
    tmp_path, monkeypatch
):
    original = BenchmarkPolicyGuard.install
    missed = []

    def miss_child(guard, session):
        if session.parent_id:
            missed.append(session)
        else:
            original(guard, session)

    monkeypatch.setattr(BenchmarkPolicyGuard, "install", miss_child)
    h, client = harness(
        tmp_path,
        [
            cell(
                "try:\n await self.jobs.launch(Subagent('work').run('task'), name='kid')\nexcept RuntimeError:\n pass\n"
                + DONE
            )
        ],
    )
    with pytest.raises(BenchmarkPolicyViolation):
        await solve(h)
    assert len(client.requests) == 1 and len(missed) == 1 and missed[0]._closed
    assert h.benchmark_policy_receipt()["verdict"] == "fail"
    assert any(not row["registered"] for row in h.benchmark_policy_receipt()["sessions"])


async def test_receipt_bounds_fail_closed_instead_of_omitting_violations(tmp_path):
    code = (
        "for i in range(100):\n try:\n  await self.session.rename('denied')\n except RuntimeError:\n  pass\n"
        + DONE
    )
    h, _ = harness(tmp_path, [cell(code)])
    with pytest.raises(BenchmarkPolicyViolation):
        await solve(h)
    receipt = h.benchmark_policy_receipt()
    assert receipt["overflow"] and receipt["verdict"] == "fail"
    assert len(receipt["violations"]) == BenchmarkPolicyGuard.MAX_VIOLATIONS
    assert len(h.benchmark_guard.path.read_bytes()) <= BenchmarkPolicyGuard.MAX_BYTES


async def test_existing_or_missing_verdict_is_never_adopted_as_current_success(tmp_path):
    h, _ = harness(tmp_path, [cell(DONE)])
    await solve(h)
    original = h.benchmark_guard.path.read_bytes()
    with pytest.raises(FileExistsError):
        harness(tmp_path, [])
    assert h.benchmark_guard.path.read_bytes() == original
    h.benchmark_guard.path.unlink()
    with pytest.raises(FileNotFoundError):
        h.benchmark_policy_receipt()


async def test_repeated_cancellation_does_not_orphan_teardown(tmp_path, monkeypatch):
    from nooa_atom.session.session import Session

    closing, release = asyncio.Event(), asyncio.Event()
    original = Session._close_body

    async def slow_close(session):
        closing.set()
        await release.wait()
        await original(session)

    monkeypatch.setattr(Session, "_close_body", slow_close)
    h, client = harness(tmp_path, [cell("await asyncio.sleep(1000)")])
    task = asyncio.create_task(solve(h))
    try:
        async with asyncio.timeout(20):
            while not client.requests:
                await asyncio.sleep(0.01)
        task.cancel()
        await asyncio.wait_for(closing.wait(), 20)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    receipt = h.benchmark_policy_receipt()
    assert all(row["closed"] for row in receipt["sessions"])
    assert not h.benchmark_guard._unsubscribe


async def test_opaque_runtime_drift_still_has_bounded_final_failure_receipt(tmp_path):
    h, client = harness(tmp_path, [cell("self._done_reflection = object()\n" + DONE)])
    with pytest.raises(BenchmarkPolicyViolation):
        await solve(h)
    receipt = h.benchmark_policy_receipt()
    assert receipt["phase"] == "final" and receipt["verdict"] == "fail"
    assert receipt["sessions"][0]["runtime_reflection"] == "<invalid-or-over-bound>"
    assert len(client.requests) == 1


@pytest.mark.parametrize("phase", ["install", "readback", "prepare"])
@pytest.mark.parametrize("child", [False, True], ids=["root", "child"])
@pytest.mark.parametrize("cancellations", [1, 2])
async def test_cancel_joins_unpublished_close_before_final_receipt(
    tmp_path, monkeypatch, phase, child, cancellations
):
    from nooa_atom.session.session import Session

    closing, release = asyncio.Event(), asyncio.Event()
    seen, registries, checks = [], [], {}
    original_install = BenchmarkPolicyGuard.install
    original_check = BenchmarkPolicyGuard.check_session
    original_create = SessionRegistry.create
    original_close = Session._close_body

    def target(session):
        return bool(session.parent_id) is child

    def broken_install(guard, session):
        original_install(guard, session)
        if target(session) and phase == "install":
            session._benchmark_guard = None

    def readback(guard, session, **kwargs):
        original_check(guard, session, **kwargs)
        checks[session.id] = checks.get(session.id, 0) + 1
        if target(session) and phase == "readback" and checks[session.id] == 2:
            guard.refuse("review_readback_failure", session_id=session.id)

    async def create(registry, options, **kwargs):
        registries.append(registry)
        if phase == "prepare" and bool(kwargs.get("parent_id")) is child:

            async def fail_prepare(session):
                session._benchmark_guard.refuse("review_prepare_failure", session_id=session.id)

            kwargs["prepare"] = fail_prepare
        return await original_create(registry, options, **kwargs)

    async def delayed_close(session):
        if target(session):
            seen.append(session)
            closing.set()
            await release.wait()
        await original_close(session)

    monkeypatch.setattr(BenchmarkPolicyGuard, "install", broken_install)
    monkeypatch.setattr(BenchmarkPolicyGuard, "check_session", readback)
    monkeypatch.setattr(SessionRegistry, "create", create)
    monkeypatch.setattr(Session, "_close_body", delayed_close)
    code = "await self.jobs.launch(Subagent('work').run('task'), name='kid')\n" + DONE
    h, client = harness(tmp_path, [cell(code if child else DONE)])
    task = asyncio.create_task(solve(h))
    try:
        await asyncio.wait_for(closing.wait(), 20)
        for _ in range(cancellations):
            task.cancel()
            await asyncio.sleep(0.05)
        # Failure/no-inference is insufficient: the owner must still join its
        # unpublished close task before returning cancellation or sealing evidence.
        assert not task.done()
        assert not h.benchmark_guard.finalized
        assert not seen[0]._closed and not seen[0]._close_task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 20)
        receipt = h.benchmark_policy_receipt()
        assert receipt["phase"] == "final" and receipt["verdict"] == "fail"
        assert receipt["outcome"] == "cancelled"
        assert len(client.requests) == int(child)
        assert all(session._closed and session._close_task.done() for session in seen)
        assert all(session.id not in h.benchmark_guard.published_ids for session in seen)
        assert all(not registry._reserved and not registry.sessions for registry in registries)
    finally:
        release.set()
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.gather(
            *(session._close_task for session in seen if session._close_task),
            return_exceptions=True,
        )


@pytest.mark.parametrize(
    "operation",
    [
        "closing the agent",
        "stopping the agent's jobs",
        "closing its model client",
        "waiting for the checkpoint",
    ],
)
@pytest.mark.parametrize("child", [False, True], ids=["root", "child"])
@pytest.mark.parametrize("cancelled", [False, True], ids=["exception", "cancel"])
async def test_swallowed_cleanup_error_fails_policy_and_still_finishes_later_steps(
    tmp_path, monkeypatch, operation, child, cancelled
):
    from nooa_atom.session.session import Session
    from nooa_atom.session.store import SessionStore

    original_step = Session._close_step
    seen, steps = [], {}
    error_type = asyncio.CancelledError if cancelled else OSError

    async def inject(session, what, step):
        steps.setdefault(session.id, []).append(what)
        if bool(session.parent_id) is child and what == operation:
            seen.append(session)

            async def fail():
                raise error_type("SECRET_CLEANUP_MESSAGE_MUST_NOT_ENTER_RECEIPT")

            step = fail
        return await original_step(session, what, step)

    monkeypatch.setattr(Session, "_close_step", inject)
    code = "result=await self.jobs.launch(Subagent('child').run('task'), name='child')\n" + DONE
    h, client = harness(tmp_path, [cell(code), cell(DONE)] if child else [cell(DONE)])
    try:
        with pytest.raises(BenchmarkPolicyViolation):
            await solve(h)
        receipt = h.benchmark_policy_receipt()
        assert receipt["phase"] == "final" and receipt["verdict"] == "fail"
        [session] = seen
        row = next(row for row in receipt["sessions"] if row["session_id"] == session.id)
        assert row["cleanup"]["error_count"] == 1
        assert row["cleanup"]["errors"] == [
            {"operation": operation, "exception_type": error_type.__name__}
        ]
        assert "SECRET_CLEANUP" not in json.dumps(receipt)
        assert any(item["code"] == "session_cleanup_failed" for item in receipt["violations"])
        assert session._closed and "closing its record" in steps[session.id]
        assert not SessionStore(tmp_path / "sessions").is_active(session.id)
        before = session._cleanup_errors.report()
        await session.close()
        assert session._cleanup_errors.report() == before  # no reset/retry on idempotent close
        assert h.benchmark_policy_receipt() == receipt
        assert len(client.requests) == 1 + int(child)
    finally:
        if operation == "closing the agent":
            for session in seen:
                await session._agent.aclose()  # only this injection skipped actual agent cleanup


@pytest.mark.parametrize(
    "damage", ["missing", "object", "replacement", "counter", "samples", "overflow"]
)
async def test_missing_or_malformed_required_cleanup_evidence_is_not_clean(
    tmp_path, monkeypatch, damage
):
    from nooa_atom.session.session import Session

    original = Session._close_body

    async def damage_after_close(session):
        await original(session)
        if damage == "missing":
            session.__dict__.pop("_cleanup_errors", None)
        elif damage == "object":
            session._cleanup_errors = object()
        elif damage == "replacement":
            from nooa_atom.session.session import _CleanupErrors

            session._cleanup_errors = _CleanupErrors()  # not the guard-registered owner
        elif hasattr(session, "_cleanup_errors"):
            if damage == "counter":
                session._cleanup_errors.count = True  # bool is not a count
            elif damage == "samples":
                session._cleanup_errors.samples = (("untrusted operation", object()),)
            else:
                session._cleanup_errors.overflow = "false"

    monkeypatch.setattr(Session, "_close_body", damage_after_close)
    h, _ = harness(tmp_path, [cell(DONE)])
    with pytest.raises(BenchmarkPolicyViolation):
        await solve(h)
    receipt = h.benchmark_policy_receipt()
    assert receipt["phase"] == "final" and receipt["verdict"] == "fail"
    assert receipt["sessions"][0]["cleanup"] == {"valid": False}
    assert any(item["code"] == "cleanup_evidence_invalid" for item in receipt["violations"])


@pytest.mark.parametrize("cancelled", [False, True], ids=["exception", "cancel"])
async def test_generic_close_keeps_best_effort_and_lock_release_with_cleanup_evidence(
    make_session, sessions_dir, monkeypatch, cancelled
):
    from nooa_atom.session.session import Session
    from nooa_atom.session.store import SessionStore

    session, _ = make_session(start=False)
    original = Session._close_step
    steps = []

    async def fail_agent_close(owner, what, step):
        steps.append(what)
        if what == "closing the agent":

            async def fail():
                raise (asyncio.CancelledError if cancelled else OSError)("generic close failure")

            step = fail
        return await original(owner, what, step)

    monkeypatch.setattr(Session, "_close_step", fail_agent_close)
    try:
        await session.close()  # generic interactive close still continues instead of raising
        assert session._closed and not SessionStore(sessions_dir).is_active(session.id)
        assert "closing its model client" in steps and "closing its record" in steps
        evidence = session._cleanup_errors.report()
        assert evidence["error_count"] == 1
        await session.close()
        assert session._cleanup_errors.report() == evidence
    finally:
        await session._agent.aclose()


def test_cleanup_error_evidence_is_bounded_sanitized_detached_and_sticky():
    from nooa_atom.session.session import _CleanupErrors

    state = _CleanupErrors()
    assert state.report() == {"error_count": 0, "errors": [], "overflow": False}
    error = type("E" * 200, (OSError,), {})("SECRET_MESSAGE")
    for _ in range(200):
        state.record("SECRET_UNTRUSTED_OPERATION", error)
    report = state.report()
    assert report["error_count"] == 200 and report["overflow"] is True
    assert len(report["errors"]) == 8 and len(json.dumps(report)) < 1500
    assert "SECRET" not in json.dumps(report)
    assert report["errors"][0] == {"operation": "other", "exception_type": "E" * 64}
    report["errors"][0]["operation"] = "mutated copy"
    assert state.report()["errors"][0]["operation"] == "other"
    state.count = state.COUNT_LIMIT
    state.record("closing the agent", error)
    assert state.report()["error_count"] == state.COUNT_LIMIT  # bounded saturating count
    state.count = True  # malformed state is not coerced/reset, even by another failure
    state.record("closing the agent", OSError("SECRET"))
    assert state.report() is None and state.count is True


async def test_invalid_cleanup_evidence_never_interrupts_generic_best_effort_close(
    make_session, sessions_dir, monkeypatch
):
    from nooa_atom.session.session import Session
    from nooa_atom.session.store import SessionStore

    session, _ = make_session(start=False)
    session._cleanup_errors = None
    original = Session._close_step

    async def inject(owner, what, step):
        if what == "closing the agent":

            async def fail():
                raise OSError("generic fault with missing evidence")

            step = fail
        return await original(owner, what, step)

    monkeypatch.setattr(Session, "_close_step", inject)
    try:
        await session.close()
        assert session._closed and not SessionStore(sessions_dir).is_active(session.id)
        assert session._cleanup_errors is None  # never replace invalid evidence with clean data
    finally:
        await session._agent.aclose()


async def test_cleanup_failure_is_latched_for_a_registered_unpublished_session(
    tmp_path, monkeypatch
):
    from nooa_atom.session.session import Session

    original_install, original_step = BenchmarkPolicyGuard.install, Session._close_step
    seen = []

    def fail_readback(guard, session):
        original_install(guard, session)
        session._benchmark_guard = None

    async def fail_close(session, what, step):
        if what == "closing the agent":
            seen.append(session)

            async def fail():
                raise OSError("private cleanup fault")

            step = fail
        return await original_step(session, what, step)

    monkeypatch.setattr(BenchmarkPolicyGuard, "install", fail_readback)
    monkeypatch.setattr(Session, "_close_step", fail_close)
    h, client = harness(tmp_path, [cell(DONE)])
    try:
        with pytest.raises(BenchmarkPolicyViolation):
            await solve(h)
        receipt = h.benchmark_policy_receipt()
        assert receipt["verdict"] == "fail" and client.requests == []
        assert receipt["sessions"][0]["cleanup"]["error_count"] == 1
        assert any(item["code"] == "session_cleanup_failed" for item in receipt["violations"])
    finally:
        for session in seen:
            await session._agent.aclose()


@pytest.mark.parametrize("step_cancelled", [False, True])
async def test_caller_cancellation_survives_a_latched_cleanup_failure(
    tmp_path, monkeypatch, step_cancelled
):
    from nooa_atom.session.session import Session

    original = Session._close_step
    seen = []

    async def fail_close(session, what, step):
        if what == "closing the agent":
            seen.append(session)

            async def fail():
                raise (asyncio.CancelledError if step_cancelled else OSError)("step failure")

            step = fail
        return await original(session, what, step)

    monkeypatch.setattr(Session, "_close_step", fail_close)
    h, client = harness(tmp_path, [cell("await asyncio.sleep(1000)")])
    task = asyncio.create_task(solve(h))
    try:
        async with asyncio.timeout(20):
            while not client.requests:
                await asyncio.sleep(0.01)
        task.cancel("caller cancellation")
        with pytest.raises(asyncio.CancelledError, match="caller cancellation"):
            await task
        receipt = h.benchmark_policy_receipt()
        assert receipt["phase"] == "final" and receipt["verdict"] == "fail"
        assert receipt["outcome"] == "cancelled" and receipt["sessions"][0]["closed"]
        assert receipt["sessions"][0]["cleanup"]["error_count"] == 1
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        for session in seen:
            await session._agent.aclose()


def test_cleanup_error_type_metadata_does_not_invoke_custom_metaclass():
    from nooa_atom.session.session import _CleanupErrors

    called = []

    class Meta(type):
        @property
        def __name__(cls):
            called.append(True)
            raise asyncio.CancelledError("metadata must not abort cleanup")

    class NamedError(OSError, metaclass=Meta):
        pass

    state = _CleanupErrors()
    state.record("closing the agent", NamedError("SECRET_MESSAGE"))
    assert called == []
    assert state.report()["errors"] == [
        {"operation": "closing the agent", "exception_type": "NamedError"}
    ]


async def test_owned_model_client_failure_is_recorded_by_the_real_inner_close_step(
    make_session, sessions_dir
):
    from nooa_atom.session.store import SessionStore

    class FailingClient:
        async def aclose(self):
            raise OSError("SECRET_CLIENT_CLOSE")

    session, _ = make_session(start=False)
    session._owned_llm = FailingClient()
    await session.close()
    assert session._closed and not SessionStore(sessions_dir).is_active(session.id)
    assert session._cleanup_errors.report()["errors"] == [
        {"operation": "closing a model client", "exception_type": "OSError"}
    ]


async def test_cleanup_name_str_subclass_callbacks_never_interrupt_close(tmp_path, monkeypatch):
    from nooa_atom.session.session import Session
    from nooa_atom.session.store import SessionStore

    callbacks = []

    class Name(str):
        def __getitem__(self, key):
            callbacks.append("getitem")
            raise AssertionError("type-name callback must never run")

        def isascii(self):
            callbacks.append("isascii")
            raise AssertionError("type-name callback must never run")

        def isidentifier(self):
            callbacks.append("isidentifier")
            raise AssertionError("type-name callback must never run")

    class Failure(OSError):
        pass

    Failure.__name__ = Name("Failure")
    original = Session._close_step
    seen, steps = [], []

    async def fail_close(session, what, step):
        steps.append(what)
        if what == "closing the agent":
            seen.append(session)

            async def fail():
                raise Failure("SECRET_CLOSE_MESSAGE")

            step = fail
        return await original(session, what, step)

    monkeypatch.setattr(Session, "_close_step", fail_close)
    h, _ = harness(tmp_path, [cell(DONE)])
    try:
        with pytest.raises(BenchmarkPolicyViolation):
            await solve(h)
        assert callbacks == []
        [session] = seen
        assert session._closed and "closing its record" in steps
        assert not SessionStore(tmp_path / "sessions").is_active(session.id)
        receipt = h.benchmark_policy_receipt()
        assert receipt["verdict"] == "fail" and receipt["phase"] == "final"
        assert receipt["sessions"][0]["cleanup"]["errors"] == [
            {"operation": "closing the agent", "exception_type": "UnknownError"}
        ]
        assert receipt["sessions"][0]["cleanup"]["error_count"] == 1
        assert "SECRET_CLOSE_MESSAGE" not in json.dumps(receipt)
    finally:
        for session in seen:
            await session._agent.aclose()
            if not session._closed:  # pre-fix regression aborts before handle release
                session.handle.close()
