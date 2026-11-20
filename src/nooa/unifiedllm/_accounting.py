# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private numeric logical ledger. No runtime, provider or telemetry dependency.

Receiver and invocation capture are task-local; delivery happens once at terminal
caller exit, never at late provider exit. These are reporting subtotals, not caps.
"""

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
from typing import Literal
from uuid import uuid4

from ._observations import _AttemptObservation, _AttemptWriter, _NumericUsage, _ObservationOwner


@dataclass(frozen=True, slots=True)
class _Subtotal:
    known: int | float | None
    complete: bool


@dataclass(frozen=True, slots=True)
class _LogicalSnapshot:
    logical_id: str
    outcome: Literal["success", "error", "cancelled"]
    failure_kind: str | None
    attempts: tuple[_AttemptObservation, ...]
    attempt_count: int
    dropped_attempts: int
    totals: tuple[tuple[str, _Subtotal], ...]
    cost_sources: tuple[str, ...]

    def subtotal(self, name: str) -> _Subtotal:
        return dict(self.totals)[name]


# A receiver must close over its own task/session owner. Inheriting this binding
# into a provider task cannot deliver anything: only the logical account settles.
_receiver: ContextVar[Callable[[_LogicalSnapshot], None] | None] = ContextVar(
    "_logical_terminal_receiver", default=None
)
_session_receiver: ContextVar[Callable[[_LogicalSnapshot], None] | None] = ContextVar(
    "_logical_session_receiver", default=None
)
_current_account: ContextVar["_LogicalAccount | None"] = ContextVar(
    "_logical_account", default=None
)


@dataclass(slots=True)
class _InvocationReceipt:
    # Strong identity retained ONLY during canonical event delivery. Not a
    # response field, process registry or response-ID lookup.
    response: object = None
    snapshot: _LogicalSnapshot | None = None


_current_receipt: ContextVar[_InvocationReceipt | None] = ContextVar(
    "_logical_invocation_receipt", default=None
)


@contextmanager
def _invocation_delivery():
    receipt = _InvocationReceipt()
    token = _current_receipt.set(receipt)
    try:
        yield receipt
    finally:
        _current_receipt.reset(token)


def _owns_delivered_response(response) -> bool:
    receipt = _current_receipt.get()
    return receipt is not None and receipt.snapshot is not None and receipt.response is response


def _bind_terminal_receiver(receiver):
    """Install in a fresh session-loop context; account captures its owner."""
    _session_receiver.set(receiver)
    return _receiver.set(receiver)


@contextmanager
def _accounting_receiver(receiver):
    token = _receiver.set(receiver)
    try:
        yield
    finally:
        _receiver.reset(token)


class _LogicalAccount:
    def __init__(self, *, detail_limit: int = 256):
        self.logical_id = uuid4().hex
        self._receiver = _receiver.get()
        self._receipt = _current_receipt.get()
        self.accepted_response = None
        self._journal = None
        if self._receipt is not None:
            self._receipt.response = None
            self._receipt.snapshot = None
        self._limit = detail_limit
        self._next = 0
        self._folded = 0
        self._records = []
        self._cost_sources: set[str] = set()
        self._values = {f.name: 0 for f in fields(_NumericUsage) if f.name != "cost_source"}
        self._known = dict.fromkeys(self._values, False)
        self._complete = dict.fromkeys(self._values, True)
        self.snapshot: _LogicalSnapshot | None = None

    def describe_request(self, *, messages=None, model=None):
        # Request rendering is tracing-local, never part of the neutral snapshot.
        # Late shielded tasks cannot change a terminal journal or receipt.
        if self.snapshot is None and self._journal is not None:
            try:
                self._journal.describe_request(messages=messages, model=model)
            except Exception:
                pass  # Description/span failures must never prevent provider entry.

    def attempt(self) -> tuple[int, _ObservationOwner]:
        if self.snapshot is not None:
            raise RuntimeError("Logical account already terminal")
        index = self._next
        self._next += 1
        owner = _ObservationOwner(_AttemptWriter())
        owner.writer._replace(logical_id=self.logical_id, attempt_index=index)
        owner.writer.entering_adapter()
        return index, owner

    def fold(self, index: int, owner: _ObservationOwner):
        if self.snapshot is not None or index < self._folded:
            return
        if index != self._folded:
            raise RuntimeError("Attempts must settle in send order")
        owner.writer.finish()
        observation = owner.writer.record
        self._folded += 1
        if observation.usage is not None and observation.usage.cost_usd is not None:
            self._cost_sources.add(observation.usage.cost_source)
        if len(self._records) < self._limit:
            self._records.append(observation)
        # Built-in preparation failures and observed queued cancellation prove
        # no dispatch. Unknown custom errors do not acquire that free proof.
        free = not observation.response_received and (
            observation.dispatch_known and not observation.provider_entered
        )
        for name in self._values:
            value = getattr(observation.usage, name, None)
            if (
                name == "cost_usd"
                and observation.usage is not None
                and observation.usage.cost_source == "response_metadata"
            ):
                # LiteLLM's hidden response_cost is computed by its local price
                # calculator, not certified provider billing. Preserve numeric
                # calibration/final-response projections, but do not report that
                # estimate as known paid spend (even an estimated explicit zero).
                value = None
            if value is not None:
                self._values[name] += value
                self._known[name] = True
            elif not free:
                self._complete[name] = False
            if observation.hidden_retries:
                self._complete[name] = False

    def freeze(self, outcome, failure_kind=None) -> _LogicalSnapshot:
        if self.snapshot is not None:
            return self.snapshot
        totals = tuple(
            (
                name,
                _Subtotal(
                    value if self._known[name] or self._complete[name] else None,
                    self._complete[name],
                ),
            )
            for name, value in self._values.items()
        )
        self.snapshot = _LogicalSnapshot(
            self.logical_id,
            outcome,
            failure_kind,
            tuple(self._records),
            self._next,
            self._next - len(self._records),
            totals,
            tuple(sorted(self._cost_sources)),
        )
        if self._receipt is not None:
            self._receipt.response = self.accepted_response if outcome == "success" else None
            self._receipt.snapshot = self.snapshot
        # Tracing/reporting callback failures cannot alter provider outcome or
        # accounting. Snapshot is irrevocably frozen BEFORE delivery.
        if self._receiver is not None:
            try:
                self._receiver(self.snapshot)
            except BaseException:
                pass
        return self.snapshot


@contextmanager
def _logical_account_scope(*, model="", messages=()):
    account = _LogicalAccount()
    journal = None
    token = _current_account.set(account)
    try:
        try:
            from ._logical_trace import _LogicalJournal

            journal = _LogicalJournal(account.logical_id, model, messages)
            account._journal = journal
        except Exception:
            pass  # Tracing imports/sinks are not accounting dependencies.
        yield account
    except BaseException as error:
        from asyncio import CancelledError

        kind = getattr(getattr(error, "kind", None), "value", None)
        account.freeze("cancelled" if isinstance(error, CancelledError) else "error", kind)
        raise
    else:
        account.freeze("success")
    finally:
        try:
            if journal is not None:
                journal.finish(account.snapshot, account.accepted_response)
        except BaseException:
            pass
        finally:
            _current_account.reset(token)
