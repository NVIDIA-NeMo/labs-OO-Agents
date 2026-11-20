# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime ownership of a pinned route and its immutable selected settings.

This holder is not an LLM facade. Calls capture the real selected client. A
shared binding changes settings for subsequent calls, never in-flight captures.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from threading import RLock
from typing import Any
from weakref import WeakSet, ref

from nooa.unifiedllm import LLM, load_llm
from nooa.unifiedllm.requirements import _normalize_require, _validate_requirements


async def _join_tasks(tasks):
    failure = None
    for task in tasks:
        while not task.done():
            try:
                await asyncio.shield(task)
            except BaseException as exc:
                if failure is None:
                    failure = exc
        try:
            task.result()
        except BaseException as exc:
            if failure is None:
                failure = exc
    if failure is not None:
        raise failure


async def _cleanup_result(call):
    """Retained lifecycle workers carry control exceptions as joined DATA."""
    try:
        await call()
    except BaseException as error:
        return error
    return None


async def _join_cleanup_tasks(tasks):
    failure = None
    for task in tasks:
        while not task.done():
            try:
                await asyncio.shield(task)
            except BaseException as error:
                if failure is None:
                    failure = error
        try:
            result = task.result()
            if isinstance(result, BaseException) and failure is None:
                failure = result
        except BaseException as error:
            if failure is None:
                failure = error
    if failure is not None:
        raise failure


class _ViewOwners:
    """Identity-only live provenance, promoted while retired cleanup is owed."""

    def __init__(self):
        self.weak = {}
        self.strong = None
        # Weak owner disappearance is not evidence of never-owned work.
        self.had_runtime_owner = False

    def add(self, owner):
        # None denotes binding-owned work, not an unrelated holder's obligation.
        if owner is None:
            return
        self.had_runtime_owner = True
        key = id(owner)
        if self.strong is not None:
            self.strong[key] = owner
            return
        existing = self.weak.get(key)
        if existing is not None and existing() is owner:
            return

        def gone(reference):
            # A delayed weakref callback must not remove a reused identity.
            if self.weak.get(key) is reference:
                del self.weak[key]

        self.weak[key] = ref(owner, gone)

    def retire(self):
        if self.strong is None:
            self.strong = {
                key: owner
                for key, reference in self.weak.items()
                if (owner := reference()) is not None
            }
            self.weak.clear()

    def owners(self):
        self.retire()
        return tuple(self.strong.values())


class _CaptureRecord:
    __nosnapshot__ = True

    def __init__(self, binding, view, owner):
        from nooa.runtime.event_manager import _active_close_dependencies, _CloseDependency

        self.binding = binding
        self.view = view
        self.owner = owner
        self.logical = True
        self.receipts = set()
        self.enclosing = _active_close_dependencies.get()
        self.dependency = _CloseDependency()
        self.dependency.attach(getattr(owner, "event_manager", None))

    def __call__(self, task, wire_owner):
        return self.register(task, wire_owner)

    def register(self, task, wire_owner):
        base = public_base(self.view)
        if base is None or wire_owner is not getattr(base, "_lease", None):
            return None
        receipt = asyncio.get_running_loop().create_future()
        self.receipts.add(receipt)
        for dependency in self.enclosing:
            if dependency.active:
                dependency.native_started()
        key = id(self.view)
        self.binding._native[key] = self.binding._native.get(key, 0) + 1

        def finished(failure):
            # Transport invokes this only AFTER the actual resource callbacks.
            if receipt.done():
                return
            receipt.set_result(failure)
            self.binding._native[key] -= 1
            if not self.binding._native[key]:
                del self.binding._native[key]
            for dependency in self.enclosing:
                if dependency.active:
                    dependency.native_finished()
            self.settled()

        return finished

    def settled(self):
        if not self.logical and all(receipt.done() for receipt in self.receipts):
            self.dependency.finish()
            if not any(isinstance(receipt.result(), BaseException) for receipt in self.receipts):
                self.binding._records.discard(self)
            self.binding._collect(fence=not self.receipts)
            self.binding._notify_settled()


def public_base(client: Any) -> LLM | None:
    from nooa.unifiedllm.controlled import AdmissionControl

    if type(client) is LLM:
        from nooa.unifiedllm.adapters._loader import _check_shell

        try:
            _check_shell(client)
        except Exception as exc:
            raise TypeError("Public runtime requires unchanged built-in behavior") from exc
        return client
    if type(client) is AdmissionControl:
        from nooa.unifiedllm.controlled import _validate_public_bridge

        base = vars(client).get("base_llm")
        if type(base) is LLM:
            _validate_public_bridge(client)
            return base
    return None


def derive_client(client: Any, **settings: Any) -> Any:
    """Derive only a proven loaded view; preserve an exact admission controller."""
    base = public_base(client)
    if base is None:
        raise TypeError("Settings require a loaded built-in model")
    view = load_llm(base, **settings)
    if base is client:
        return view
    try:
        return client._with_public_view(view)
    except BaseException:
        try:
            view.close()
        except BaseException:
            pass
        raise


class ModelBinding:
    """A sharable runtime slot, lazily loaded if constructed with an alias.

    Injected roots are borrowed. Alias roots and prepared settings views are
    owned. The owner releases one reference; each agent/session sharer retains
    another. Retired views close only after their last capture releases.
    """

    __nosnapshot__ = True

    def __init__(
        self, source: Any = None, *, require=None, owned: bool = False, lazy: bool = False
    ):
        self._requirements = _normalize_require(require)
        self._source = source
        self._delegate = source.retain() if isinstance(source, ModelBinding) else None
        self._current = None if isinstance(source, (str, ModelBinding)) else source
        self._owned: dict[int, Any] = {}
        self._captures: dict[int, int] = {}
        self._native: dict[int, int] = {}
        self._records: set[_CaptureRecord] = set()
        self._view_owners: dict[int, _ViewOwners] = {}
        self._retirement_owners: dict[asyncio.Task, tuple[Any, ...]] = {}
        self._ownerless_retirements: set[asyncio.Task] = set()
        self._released_delegates: list[ModelBinding] = []
        self._references = 1
        self._lock = RLock()
        self._cleanup: set[asyncio.Task] = set()
        self._closed = False
        self._owner_released = False
        self._sessions = WeakSet()
        self._on_settled = None
        if isinstance(source, str) and not source:
            raise TypeError("Model alias must be nonempty")
        if self._current is not None:
            if not lazy:
                self._validate(self._current)
            if owned:
                self._owned[id(source)] = source
                self._view_owners[id(source)] = _ViewOwners()

    @property
    def requirements(self):
        return self._requirements

    def _validate(self, client):
        if not callable(getattr(client, "acall", None)):
            raise TypeError("Model slot requires a client or alias")
        required = self.requirements
        if required.flags or required.min_context is not None:
            base = public_base(client)
            facts = base.capabilities if base is not None else getattr(client, "capabilities", None)
            _validate_requirements(facts, required, label="slot")

    @property
    def client(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("Model binding is closed")
            if self._delegate is not None and self._current is None:
                selected = self._delegate.client
                self._validate(selected)
                return selected
            if self._current is None:
                if self._source is None:
                    raise RuntimeError("Model slot is unbound")
                view = load_llm(self._source, require=self.requirements)
                self._current = view
                self._owned[id(view)] = view
                self._view_owners[id(view)] = _ViewOwners()
            self._validate(self._current)
            return self._current

    @property
    def llm(self):
        return self.client

    def retain(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("Model binding is closed")
            self._references += 1
        return self

    def prepare(self, **settings):
        view = derive_client(self.client, **settings)
        try:
            self._validate(view)
        except BaseException:
            view.close()
            raise
        return view

    def _prepare_publication(self, view):
        self._validate(view)
        with self._lock:
            if self._closed:
                raise RuntimeError("Model binding is closed")
            owned = dict(self._owned)
            view_owners = dict(self._view_owners)
            if id(view) not in owned and id(view) not in view_owners:
                view_owners[id(view)] = _ViewOwners()
            owned[id(view)] = view
            return view, owned, view_owners

    def _publish_prepared(self, publication):
        # Assignment-only durable publication: no validation, allocation,
        # observers, awaits, or disposal can reinterpret the commit outcome.
        self._current, self._owned, self._view_owners = publication

    def publish(self, view):
        self._publish_prepared(self._prepare_publication(view))
        self._collect()

    def _notify_settled(self):
        if self._on_settled is None or self._records or self._owned:
            return
        if any(
            not task.done() or task.cancelled() or task.result() is not None
            for task in self._cleanup
        ):
            return
        self._cleanup.clear()
        self._retirement_owners.clear()
        self._ownerless_retirements.clear()
        callback, self._on_settled = self._on_settled, None
        callback()

    def _binding_for(self, client):
        """Find an actual matching view without re-resolving a selected request."""
        if self._closed:
            return None
        if self._current is client:
            return self
        if self._current is None and self._delegate is not None:
            return self._delegate._binding_for(client)
        return None

    def _collect(self, *, fence=False):
        failure = None
        for key, view in tuple(self._owned.items()):
            provenance = self._view_owners.get(key)
            if provenance is not None and (self._closed or view is not self._current):
                # Promote BEFORE the final native record/receipt disappears.
                provenance.retire()
            if (
                self._captures.get(key, 0)
                or self._native.get(key, 0)
                or (not self._closed and view is self._current)
            ):
                continue
            del self._owned[key]
            owners = provenance.owners() if provenance is not None else ()
            self._view_owners.pop(key, None)
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                try:
                    view.close()
                except BaseException as exc:
                    if failure is None:
                        failure = exc
                continue
            # Fence a now-unused view synchronously; disposal is still joined.
            close_failure = None
            if fence:
                try:
                    view.close()
                except BaseException as error:
                    close_failure = error

            async def dispose(view=view, close_failure=close_failure):
                failure = await _cleanup_result(view.aclose)
                return close_failure if close_failure is not None else failure

            task = loop.create_task(dispose())
            self._cleanup.add(task)
            self._retirement_owners[task] = owners
            if provenance is not None and not provenance.had_runtime_owner:
                self._ownerless_retirements.add(task)
            task.add_done_callback(self._retirement_settled)
            # Keep unfinished/failed tasks discoverable until an owner joins them.
        # Borrowed views owe no disposal, and must not leave historical identities.
        for key in tuple(self._view_owners):
            if (
                key not in self._owned
                and not self._captures.get(key, 0)
                and not self._native.get(key, 0)
                and (self._closed or key != id(self._current))
            ):
                del self._view_owners[key]
        if failure is not None:
            raise failure

    def _retirement_settled(self, task):
        # Reading successful DATA is safe accounting; no forever-growing success
        # history. Failed/uncompleted obligations retain their responsible owners.
        if not task.cancelled() and task.result() is None:
            self._cleanup.discard(task)
            self._retirement_owners.pop(task, None)
            self._ownerless_retirements.discard(task)
        self._notify_settled()

    @asynccontextmanager
    async def capture(self, *, owner=None):
        if self._delegate is not None and self._current is None:
            if self._closed:
                raise RuntimeError("Model binding is closed")
            async with self._delegate.capture(owner=owner) as view:
                self._validate(view)
                yield view
            return
        view = self.client
        async with self._capture_exact(view, owner=owner):
            yield view

    @asynccontextmanager
    async def _capture_exact(self, view, *, owner=None):
        """Capture the already-selected client, never substitute a later selection."""
        from nooa.runtime.context_vars import _parent_agent_var
        from nooa.unifiedllm._transport_lifetime import _dispatch_registrar

        with self._lock:
            key = id(view)
            actual_owner = owner if owner is not None else _parent_agent_var.get()
            provenance = self._view_owners.setdefault(key, _ViewOwners())
            provenance.add(actual_owner)
            self._captures[key] = self._captures.get(key, 0) + 1
            record = _CaptureRecord(self, view, actual_owner)
            self._records.add(record)
        scope = record.dependency.enter()
        token = _dispatch_registrar.set(record)
        try:
            yield view
        finally:
            _dispatch_registrar.reset(token)
            record.dependency.reset(scope)
            with self._lock:
                self._captures[key] -= 1
                if not self._captures[key]:
                    del self._captures[key]
                record.logical = False
                record.settled()
                self._collect()
            # Logical exit NEVER waits shielded native work or retirement.

    async def _join_receipts(self, records):
        failure = None
        for record in records:
            for receipt in tuple(record.receipts):
                while not receipt.done():
                    try:
                        await asyncio.shield(receipt)
                    except BaseException as error:
                        if failure is None:
                            failure = error
                result = receipt.result()
                if isinstance(result, BaseException) and failure is None:
                    failure = result
        # Once the complete receipt results have been observed, do not replay
        # a reported control at a later final join or retain its owner forever.
        for record in records:
            if not record.logical and all(receipt.done() for receipt in record.receipts):
                self._records.discard(record)
        self._notify_settled()
        if failure is not None:
            raise failure

    async def _join_owner_cleanup(self, owner):
        """Join only this owner's exact dispatches, never another pool sharer."""
        failure = None
        try:
            await self._join_receipts(
                tuple(record for record in self._records if record.owner is owner)
            )
        except BaseException as error:
            failure = error
        for delegate in (self._delegate, *self._released_delegates):
            if delegate is not None:
                try:
                    await delegate._join_owner_cleanup(owner)
                except BaseException as error:
                    if failure is None:
                        failure = error
        try:
            await self._join_retirement(owner=owner)
        except BaseException as error:
            if failure is None:
                failure = error
        if failure is not None:
            raise failure

    async def _join_retirement(self, *, owner=None, all_owners=False, ownerless=False):
        tasks = tuple(
            task
            for task in self._cleanup
            if all_owners
            or (
                ownerless
                and task in self._ownerless_retirements
                and self._retirement_owners.get(task) == ()
            )
            or any(responsible is owner for responsible in self._retirement_owners.get(task, ()))
        )
        try:
            await _join_cleanup_tasks(tasks)
        finally:
            for task in tasks:
                if task.done():
                    self._cleanup.discard(task)
                    self._retirement_owners.pop(task, None)
                    self._ownerless_retirements.discard(task)

    async def _join_cleanup(self):
        failure = None
        if self._closed:
            try:
                await self._join_receipts(tuple(self._records))
            except BaseException as error:
                failure = error
        for delegate in tuple(self._released_delegates):
            try:
                await delegate._join_cleanup()
            except BaseException as error:
                if failure is None:
                    failure = error
        try:
            # Nonfinal legacy joins only explicitly classified ALREADY-retired
            # idle work, never foreign native receipts or responsible owners.
            await self._join_retirement(all_owners=self._closed, ownerless=not self._closed)
        except BaseException as error:
            if failure is None:
                failure = error
        if failure is not None:
            raise failure

    def _drop_reference(self, *, owner=False):
        with self._lock:
            if owner:
                if self._owner_released:
                    return
                self._owner_released = True
            if self._references:
                self._references -= 1
            if not self._references:
                self._closed = True
                if self._delegate is not None:
                    delegate, self._delegate = self._delegate, None
                    delegate._drop_reference()
                    self._released_delegates.append(delegate)
            self._collect()
            self._notify_settled()

    async def _release(self):
        self._drop_reference()
        await self._join_cleanup()

    async def aclose(self):
        """Release the creator once, joining retained cleanup across cancellation."""
        self._drop_reference(owner=True)
        await self._join_cleanup()
