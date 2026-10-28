# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact-owned, bounded benchmark policy. Not a generic telemetry subsystem.

Enforcement is middleware/owner/client boundary code. Observers only latch
additional evidence; raising in an event subscriber is never enforcement.
"""

from __future__ import annotations

import contextvars
import copy
import json
import threading
from pathlib import Path
from typing import Any
from uuid import uuid4

POLICY = "no_reflection_no_titles_v1"
SCHEMA = "nooa.atom.benchmark_policy.v1"
_session_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "atom_benchmark_session", default=None
)
_FORBIDDEN = {
    "_reflect_done_plan": "reflection",
    "name_session": "title_generation",
    "rename_session": "rename",
}
_SCHEMAS = {"DoneReflectionPlan": "reflection", "NameSessionResponse": "title_generation"}


class BenchmarkPolicyViolation(RuntimeError):
    pass


def validate_benchmark_policy(values: dict[str, Any]) -> None:
    """Exact explicit values, not defaults/coercions or silently ignored kwargs."""
    if type(values.get("benchmark_policy")) is not str or values["benchmark_policy"] != POLICY:
        raise ValueError(f"benchmark_policy must explicitly be {POLICY!r}")
    if type(values.get("experimental_namespace")) is not str or values[
        "experimental_namespace"
    ] not in {"legacy", "call", "session"}:
        raise ValueError(
            "benchmark experimental_namespace must explicitly select legacy/call/session"
        )
    if values.get("experimental_done_reflection") is not False:
        raise ValueError("benchmarks require explicit experimental_done_reflection=False")
    if values.get("benchmark_allow_session_titles") is not False:
        raise ValueError("benchmarks require explicit benchmark_allow_session_titles=False")


def _config(options: Any) -> dict[str, Any]:
    return {
        key: getattr(options, key, None)
        for key in (
            "benchmark_policy",
            "experimental_namespace",
            "experimental_done_reflection",
            "benchmark_allow_session_titles",
        )
    }


class BenchmarkPolicyGuard:
    """One episode's fail-closed coverage/attempt latch and owned JSON receipt."""

    MAX_SESSIONS = 64
    MAX_INVOCATIONS = 4096
    MAX_VIOLATIONS = 64
    MAX_BYTES = 2_000_000

    def __init__(self, path: str | Path):
        if not isinstance(path, (str, Path)) or not str(path):
            raise ValueError("worker-owned benchmark_policy_path is required")
        self.path = Path(path).resolve()
        self.run_id = uuid4().hex
        self.namespace: str | None = None
        self.sessions: dict[str, dict[str, Any]] = {}
        self.invocations: dict[str, dict[str, Any]] = {}
        self.published_ids: set[str] = set()
        self.violations: list[dict[str, Any]] = []
        self.failed = False
        self._lock = threading.RLock()
        self._writing = False
        self.counts = {"reflection": 0, "title_generation": 0, "rename": 0, "title_mutation": 0}
        self.overflow = False
        self.finalized = False
        self.outcome = "not_started"
        self._by_agent: dict[int, str] = {}
        self._unsubscribe: list[Any] = []
        self._client: Any = None
        self._last_receipt: dict[str, Any] | None = None
        # Never accept/reuse an old artifact as evidence for this episode.
        with self.path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(self.snapshot()))

    def configure(self, values: dict[str, Any]) -> None:
        try:
            validate_benchmark_policy(values)
        except Exception:
            self.violate("configuration_invalid")
            self.finalized = True
            self.outcome = "construction_error"
            self.write()
            raise
        self.namespace = values["experimental_namespace"]
        self.write()

    def violate(
        self, code: str, *, session_id=None, method=None, invocation_id=None, attempt_id=None
    ) -> None:
        with self._lock:
            self.failed = True
            details = {
                "code": code,
                "session_id": session_id,
                "method": method,
                "invocation_id": invocation_id,
                "attempt_id": attempt_id,
            }
            row = {}
            for key, value in details.items():
                if value is None or type(value) is str and len(value) <= 256:
                    row[key] = value
                else:
                    row[key] = "<invalid-or-over-bound>"
                    self.overflow = True
            if row not in self.violations:
                if code in self.counts:
                    self.counts[code] += 1
                if len(self.violations) < self.MAX_VIOLATIONS:
                    self.violations.append(row)
                else:
                    self.overflow = True
            # A late rejected attempt must invalidate an already-written pass too.
            if self.finalized and not self._writing:
                self.write()

    def refuse(self, code: str, **details) -> None:
        self.violate(code, **details)
        raise BenchmarkPolicyViolation(code)

    def _label(self, value):
        if value is None or type(value) is str and len(value) <= 256:
            return value
        self.violate("identity_or_title_bound_exceeded")
        return "<invalid-or-over-bound>"

    def _receipt_value(self, value):
        if value is None or type(value) is bool or type(value) is str and len(value) <= 256:
            return value
        self.violate("receipt_value_invalid_or_over_bound")
        return "<invalid-or-over-bound>"

    def identity(self, session) -> dict[str, Any]:
        stored = session.handle.info
        return {
            key: self._label(value)
            for key, value in {
                "id": session.id,
                "parent_id": session.parent_id,
                "name": session.name,
                "info_id": session.info.id,
                "info_name": session.info.name,
                "title": session.info.title,
                "stored_id": stored.id,
                "stored_parent_id": stored.parent_id,
                "stored_name": stored.name,
                "stored_title": stored.title,
            }.items()
        }

    def expect(self, handle, options) -> None:
        sid = handle.id
        if sid in self.sessions or len(self.sessions) >= self.MAX_SESSIONS:
            self.refuse("duplicate_or_excess_session", session_id=sid)
        self.sessions[sid] = {
            "session_id": sid,
            "parent_id": handle.info.parent_id,
            "registered": False,
            "ready": False,
        }
        try:
            validate_benchmark_policy(_config(options))
            if options.experimental_namespace != self.namespace:
                raise ValueError("namespace arm changed")
        except Exception:
            self.refuse("session_configuration_invalid", session_id=sid)

    def install(self, session) -> None:
        sid = session.id
        entry = self.sessions.get(sid)
        if entry is None or entry["registered"]:
            self.refuse("unobserved_or_duplicate_session", session_id=sid)
        entry.update(
            registered=True,
            session=session,
            cleanup_owner=getattr(session, "_cleanup_errors", None),
            initial=self.identity(session),
            registered_options=_config(session.options),
            registered_runtime={
                "namespace": session._agent._namespace_lifetime,
                "reflection": session._agent._done_reflection,
                "allow_session_titles": session.options.benchmark_allow_session_titles,
            },
        )
        self._by_agent[id(session._agent)] = sid
        session._benchmark_guard = self

        def before(event):
            self.record_invocation(
                sid,
                event.call_id,
                event.method_name,
                parent=event.parent_call_id,
                generation=event.needs_generation,
            )
            if event.method_name in _FORBIDDEN:
                self.violate(
                    _FORBIDDEN[event.method_name],
                    session_id=sid,
                    method=event.method_name,
                    invocation_id=event.call_id,
                    attempt_id=event.call_id,
                )

        def generation(event):
            if event.method_name in _FORBIDDEN:
                self.violate(
                    _FORBIDDEN[event.method_name],
                    session_id=sid,
                    method=event.method_name,
                    invocation_id=session._agent.runtime._agent_call_id,
                    attempt_id=session._agent.runtime._agent_call_id,
                )

        def title(event):
            self.violate("title_mutation", session_id=sid, attempt_id=event.id)

        async def agent_call(ctx, nxt):
            method = ctx.method_name
            if method == "aclose":
                return await nxt(ctx)  # policy failure must not prevent owned teardown
            if method in _FORBIDDEN:
                call_id = ctx.agent.runtime._agent_call_id
                self.refuse(
                    _FORBIDDEN[method],
                    session_id=sid,
                    method=method,
                    invocation_id=call_id,
                    attempt_id=call_id,
                )
            self.check_session(
                session,
                require_ready=bool(
                    getattr(getattr(type(ctx.agent), method, None), "_needs_generation", False)
                ),
            )
            if self.failed:
                raise BenchmarkPolicyViolation("benchmark policy violation already latched")
            return await nxt(ctx)

        async def llm_call(ctx, nxt):
            self.check_session(session, require_ready=True)
            if ctx.client is not self._client:
                self.refuse("policy_client_drift", session_id=sid)
            self.check_model(None, ctx.params.get("output_model"))
            return await nxt(ctx)

        entry["observers"] = [
            (session._agent.event_manager, "BeforeAgentCall", before),
            (session._agent.event_manager, "BeforeTurn", generation),
            (session.handle.events, "SessionTitleUpdated", title),
        ]
        entry["agent_middleware"] = agent_call
        entry["llm_middleware"] = llm_call
        self._unsubscribe.extend(
            [
                session._agent.event_manager.on("BeforeAgentCall", before),
                session._agent.event_manager.on("BeforeTurn", generation),
                session.handle.events.on("SessionTitleUpdated", title),
                session._agent.event_manager.intercept("agent_call", agent_call),
                session._agent.event_manager.intercept("llm_call", llm_call),
            ]
        )
        self.check_session(session, require_ready=False)

    def check_session(self, session, *, require_ready: bool) -> None:
        sid = session.id
        entry = self.sessions.get(sid)
        if entry is None or not entry["registered"] or entry.get("session") is not session:
            self.refuse("session_uncovered", session_id=sid)
        if session._benchmark_guard is not self or not session._benchmark_policy_required:
            self.refuse("owner_guard_missing", session_id=sid)
        try:
            validate_benchmark_policy(_config(session.options))
            if session.options.experimental_namespace != self.namespace:
                raise ValueError("namespace drift")
            if (
                session._agent._done_reflection is not False
                or session._agent._namespace_lifetime != self.namespace
            ):
                raise ValueError("runtime drift")
        except Exception:
            self.refuse("policy_runtime_drift", session_id=sid)
        if session._agent.llm is not self._client or session.options.llm is not self._client:
            self.refuse("policy_client_drift", session_id=sid)
        if any(
            callback not in manager._handlers.get(kind, [])
            for manager, kind, callback in entry["observers"]
        ):
            self.refuse("observer_coverage_missing", session_id=sid)
        em = session._agent.event_manager
        if entry["agent_middleware"] not in em._middleware.get("agent_call", []) or entry[
            "llm_middleware"
        ] not in em._middleware.get("llm_call", []):
            self.refuse("middleware_missing", session_id=sid)
        if self.identity(session) != entry["initial"]:
            self.refuse("identity_or_title_mutation", session_id=sid)
        if require_ready and (not entry["ready"] or _session_id.get() != sid or session.closing):
            self.refuse("session_not_ready_in_loop_context", session_id=sid)

    def ready(self, session) -> None:
        self.check_session(session, require_ready=False)
        entry = self.sessions[session.id]
        if entry["ready"]:
            self.refuse("duplicate_readiness", session_id=session.id)
        _session_id.set(session.id)
        entry["ready"] = True
        entry["ready_context_session_id"] = _session_id.get()

    def title_attempt(self, session, operation="Session.set_title") -> None:
        self.refuse("rename", session_id=session.id, method=operation, attempt_id=uuid4().hex)

    def record_invocation(
        self, sid, call_id, method, *, parent=None, generation=None, hooked=False
    ):
        with self._lock:
            if not sid or sid not in self.sessions:
                self.refuse("invocation_without_session", invocation_id=call_id, method=method)
            if (
                not isinstance(call_id, str)
                or len(call_id) > 128
                or not isinstance(method, str)
                or len(method) > 160
            ):
                self.refuse("invocation_bound_exceeded", session_id=sid)
            if call_id not in self.invocations:
                if len(self.invocations) >= self.MAX_INVOCATIONS:
                    self.refuse("invocation_bound_exceeded", session_id=sid)
                self.invocations[call_id] = {
                    "invocation_id": call_id,
                    "session_id": sid,
                    "method": method,
                    "parent_call_id": parent,
                    "needs_generation": generation,
                    "hook_observed": hooked,
                    "model_attempts": 0,
                    "http_dispatches": 0,
                }
            elif self.invocations[call_id]["session_id"] != sid:
                self.refuse("invocation_owner_changed", session_id=sid, invocation_id=call_id)
            if hooked:
                self.invocations[call_id]["hook_observed"] = True

    def before_agent_call(self, *, agent, method_name, call_id, parent_call_id=None, **kwargs):
        sid = self._by_agent.get(id(agent)) or _session_id.get()
        self.record_invocation(
            sid,
            call_id,
            method_name,
            parent=parent_call_id,
            generation=kwargs.get("needs_generation"),
            hooked=True,
        )
        if method_name in _FORBIDDEN:
            self.violate(
                _FORBIDDEN[method_name],
                session_id=sid,
                method=method_name,
                invocation_id=call_id,
                attempt_id=call_id,
            )

    def check_model(self, invocation_id, output_model=None, *, count=True) -> None:
        from nooa.runtime.actor import _current_call_var

        sid = _session_id.get()
        entry = self.sessions.get(sid)
        if self.finalized or entry is None:
            self.refuse("model_without_live_session", session_id=sid, invocation_id=invocation_id)
        self.check_session(entry["session"], require_ready=True)
        if self._client is None or getattr(self._client, "_episode_policy_guard", None) is not self:
            self.refuse("client_policy_guard_missing", session_id=sid)
        call = _current_call_var.get()
        method = getattr(call, "method_name", None)
        schema = getattr(output_model, "__name__", None)
        forbidden = _FORBIDDEN.get(method) or _SCHEMAS.get(schema)
        if forbidden:
            self.refuse(
                forbidden,
                session_id=sid,
                method=method,
                invocation_id=invocation_id,
                attempt_id=getattr(call, "id", None),
            )
        if self.failed:
            raise BenchmarkPolicyViolation("benchmark policy violation already latched")
        if invocation_id is not None:
            row = self.invocations.get(invocation_id)
            if row is None or row["session_id"] != sid:
                self.refuse(
                    "model_invocation_uncovered", session_id=sid, invocation_id=invocation_id
                )
            if count:
                row["model_attempts"] += 1

    def check_dispatch(self, invocation_id, request) -> None:
        self.check_model(invocation_id, count=False)
        row = self.invocations[invocation_id]
        row["http_dispatches"] += 1
        text = request.get("text")
        format_ = text.get("format") if isinstance(text, dict) else None
        forbidden = _SCHEMAS.get(format_.get("name")) if isinstance(format_, dict) else None
        if forbidden:
            self.refuse(forbidden, session_id=row["session_id"], invocation_id=invocation_id)

    def bind_client(self, client) -> None:
        self._client = client

    def _cleanup_report(self, session) -> dict[str, Any] | None:
        # Local import: Session constructs the evidence and itself uses this
        # policy module. Require the original owner, not a replacement clean
        # object or a permissive default for an older/malformed artifact.
        from nooa_atom.session.session import _CleanupErrors

        evidence = getattr(session, "_cleanup_errors", None)
        entry = self.sessions.get(session.id, {})
        if type(evidence) is not _CleanupErrors or evidence is not entry.get("cleanup_owner"):
            return None
        return evidence.report()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            sessions = []
            for entry in self.sessions.values():
                row = {
                    k: v
                    for k, v in entry.items()
                    if k
                    not in {
                        "session",
                        "cleanup_owner",
                        "agent_middleware",
                        "llm_middleware",
                        "observers",
                    }
                }
                session = entry.get("session")
                if session is not None:
                    row.update(
                        effective_options={
                            key: self._receipt_value(value)
                            for key, value in _config(session.options).items()
                        },
                        runtime_reflection=self._receipt_value(
                            getattr(session._agent, "_done_reflection", None)
                        ),
                        runtime_namespace=self._receipt_value(
                            getattr(session._agent, "_namespace_lifetime", None)
                        ),
                        final=self.identity(session),
                        closed=session._closed,
                        cleanup=self._cleanup_report(session) or {"valid": False},
                    )
                sessions.append(row)
            return {
                "schema": SCHEMA,
                "policy": POLICY,
                "run_id": self.run_id,
                "phase": "final" if self.finalized else "pending",
                "verdict": "fail" if self.failed else "pass" if self.finalized else "incomplete",
                "outcome": self.outcome,
                "namespace": self.namespace,
                "sessions": sessions,
                "invocations": list(self.invocations.values()),
                "violations": list(self.violations),
                "observed_attempt_counts": dict(self.counts),
                "overflow": self.overflow,
                "capture_reconciliation": "external_required_not_asserted",
            }

    def _read_receipt(self) -> dict[str, Any]:
        with self.path.open("rb") as stream:
            content = stream.read(self.MAX_BYTES + 1)
        if len(content) > self.MAX_BYTES:
            raise BenchmarkPolicyViolation("policy receipt exceeds bound")
        return json.loads(content)

    def write(self) -> dict[str, Any]:
        with self._lock:
            self._writing = True
            try:
                return self._write()
            except BaseException:
                self.violate("verdict_write_failed")
                raise
            finally:
                self._writing = False

    def _write(self) -> dict[str, Any]:
        receipt = self.snapshot()
        text = json.dumps(receipt, sort_keys=True)
        if len(text.encode()) > self.MAX_BYTES:
            self.failed = True
            raise BenchmarkPolicyViolation("policy receipt bound exceeded")
        temporary = self.path.with_name(self.path.name + "." + self.run_id + ".tmp")
        try:
            existing = self._read_receipt()
            if existing.get("run_id") != self.run_id:
                raise BenchmarkPolicyViolation("policy receipt ownership changed")
            with temporary.open("x", encoding="utf-8") as stream:
                stream.write(text)
            temporary.replace(self.path)
            if self._read_receipt() != receipt:
                raise BenchmarkPolicyViolation("policy receipt readback mismatch")
        except BaseException:
            self.failed = True
            temporary.unlink(missing_ok=True)
            raise
        self._last_receipt = json.loads(text)
        return copy.deepcopy(self._last_receipt)

    def finish(self, registry, outcome: str) -> dict[str, Any]:
        self.outcome = outcome
        if not self.sessions:
            self.violate("no_session_coverage")
        if self._client is None or getattr(self._client, "_episode_policy_guard", None) is not self:
            self.violate("client_policy_guard_missing")
        if registry is not None and self.published_ids != set(self.sessions):
            self.violate("registry_coverage_mismatch")
        for entry in self.sessions.values():
            if not entry["registered"] or not entry["ready"]:
                self.violate("coverage_incomplete", session_id=entry["session_id"])
            if entry.get("session") is not None:
                session = entry["session"]
                cleanup = self._cleanup_report(session)
                if cleanup is None:
                    self.violate("cleanup_evidence_invalid", session_id=entry["session_id"])
                elif cleanup["error_count"]:
                    self.violate("session_cleanup_failed", session_id=entry["session_id"])
                if not session._closed:
                    self.violate("session_not_closed", session_id=entry["session_id"])
                if any(job.status == "running" for job in session._agent.jobs.list()):
                    self.violate("owned_work_not_closed", session_id=entry["session_id"])
                try:
                    self.check_session(entry["session"], require_ready=False)
                except BenchmarkPolicyViolation:
                    pass
        self.finalized = True
        return self.write()

    def final_receipt(self) -> dict[str, Any]:
        with self._lock:
            if (
                not self.finalized
                or self._last_receipt is None
                or self._last_receipt["phase"] != "final"
            ):
                raise BenchmarkPolicyViolation("final policy verdict missing")
            if self._read_receipt() != self._last_receipt:
                raise BenchmarkPolicyViolation("final policy verdict missing/changed")
            if self.failed and self._last_receipt["verdict"] != "fail":
                self.write()
            return copy.deepcopy(self._last_receipt)

    def detach(self) -> None:
        for remove in reversed(self._unsubscribe):
            remove()
        self._unsubscribe.clear()
