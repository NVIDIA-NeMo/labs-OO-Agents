# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LiteLLM CustomLogger that maintains a content-addressed message journal.

Intercepts the message list *before* each LLM call and posts only messages
not yet seen in this session to ``POST /v1/journal/messages``.  After each
successful call, posts a call record referencing all messages by hash to
``POST /v1/journal/calls``.

This reduces per-call data transmission from O(N) to O(delta) — in a
100-turn agentic loop only the 1–3 new messages per turn are transmitted,
not the full accumulated context window.

Usage (handled automatically by ``enable_tracing()``)::

    import litellm
    from nooa.tracing._litellm_journal import (
        MessageJournalCallback,
    )
    litellm.callbacks.append(MessageJournalCallback("http://localhost:5001"))
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from litellm.integrations.custom_logger import CustomLogger
from opentelemetry import trace as otel_trace

from nooa.tracing._journal_builder import _encode_image
from nooa.tracing._session import get_session

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_msg(msg: dict) -> str:
    """Content-address a message dict.

    Uses canonical JSON (sorted keys, no whitespace) so the hash is stable
    regardless of dict insertion order.  Returns ``"sha256:<hex>"``.
    """
    canonical = json.dumps(msg, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()


def _msg_to_dict(msg: Any) -> dict:
    """Normalise a litellm message to a plain JSON-serialisable dict."""
    if msg is None:
        return {}
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_unset=True)
    try:
        return dict(msg)
    except TypeError:
        return {"raw": str(msg)}


def _extract_output_msgs(response_obj: Any) -> list[dict]:
    """Pull assistant messages out of a litellm completion response.

    Handles both Chat Completions format (.choices) and Responses API format (.output).
    """
    msgs = []
    try:
        # Chat Completions API: response has .choices[].message
        if hasattr(response_obj, "choices") and response_obj.choices:
            for choice in response_obj.choices:
                msgs.append(_msg_to_dict(choice.message))
        # Responses API: response has .output (list of output items)
        elif hasattr(response_obj, "output") and response_obj.output:
            for item in response_obj.output:
                if hasattr(item, "model_dump"):
                    msgs.append(item.model_dump())
                elif isinstance(item, dict):
                    msgs.append(item)
                else:
                    msgs.append({"type": getattr(item, "type", "unknown"), "repr": repr(item)})
    except Exception as exc:
        log.debug("Failed to extract output messages: %s", exc)
    return msgs


def _skeleton_dict_message(msg: dict, blocks: dict[str, str]) -> dict:
    """Content-address ``msg["content"]`` and each tool_call's ``arguments``.

    Mirrors what :func:`nooa.tracing._journal_builder.build_journal_payload`
    does for ``RenderedMessage`` inputs, but takes the dict shape returned
    by :func:`_extract_output_msgs` (and anything else that lands already
    as an OpenAI-shape message dict).

    Populates *blocks* with ``hash -> content`` for every large field so
    the caller can feed those to :meth:`MessageJournalCallback._send_new_blocks`
    for per-session dedup.

    Fields transformed:

    * ``content`` (string) → replaced with ``parts=[{"block_hash": …}]``.
    * ``tool_calls[i].function.arguments`` (string) → replaced with
      ``arguments_hash`` under the same ``function`` object.
    * ``images`` (list[str]) → replaced with ``image_hashes``
      (list[str]), one hash per image.

    Unchanged fields (``role``, ``tool_call_id``, ``type``, ``name``,
    message-level extras) are carried through untouched.
    """
    entry: dict = {k: v for k, v in msg.items() if k not in ("content", "tool_calls", "images")}

    content = msg.get("content")
    if content is not None:
        content_s = str(content)
        h = _hash_str(content_s)
        blocks[h] = content_s
        entry["parts"] = [{"block_hash": h}]

    tcs = msg.get("tool_calls")
    if tcs:
        new_tcs: list[dict] = []
        for tc in tcs:
            new_tc = dict(tc)
            fn = dict(new_tc.get("function") or {})
            args = fn.get("arguments")
            if args is not None:
                args_s = str(args)
                ah = _hash_str(args_s)
                blocks[ah] = args_s
                fn.pop("arguments", None)
                fn["arguments_hash"] = ah
            new_tc["function"] = fn
            new_tcs.append(new_tc)
        entry["tool_calls"] = new_tcs

    images = msg.get("images")
    if images:
        image_hashes: list[str] = []
        for img in images:
            img_s = _encode_image(img)
            h = _hash_str(img_s)
            blocks[h] = img_s
            image_hashes.append(h)
        entry["image_hashes"] = image_hashes

    return entry


def _hash_str(s: str) -> str:
    """SHA-256 a string the same way ``_journal_builder._hash`` does."""
    return "sha256:" + hashlib.sha256(s.encode("utf-8")).hexdigest()


_LARGE_PAYLOAD_BYTES = 512 * 1024  # 512 KB — warn when payloads get this big


_POST_RETRIES = 3
_POST_RETRY_DELAYS = (1.0, 3.0, 5.0)


def _post_json(
    url: str,
    payload: Any,
    *,
    session_id: str = "",
    timeout: float = 15.0,
    on_success: Callable[[], None] | None = None,
) -> None:
    """Fire-and-forget JSON POST with retries — dispatched to a daemon thread."""

    def _send() -> None:
        try:
            _send_impl()
        finally:
            # Self-evict so ``_PENDING_THREADS`` doesn't grow without bound
            # in long-running processes that never call ``flush_pending``.
            with _PENDING_LOCK:
                _PENDING_THREADS.discard(threading.current_thread())

    def _send_impl() -> None:
        tag = f" [session={session_id}]" if session_id else ""
        n_items = len(payload) if isinstance(payload, list) else 1
        size_kb = 0.0
        try:
            data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        except (TypeError, ValueError) as exc:
            log.warning("POST %s: JSON serialization failed: %s%s", url, exc, tag)
            return
        size_kb = len(data) / 1024
        if len(data) > _LARGE_PAYLOAD_BYTES:
            log.warning(
                "POST %s: large payload %.0f KB (%d item(s))%s",
                url,
                size_kb,
                n_items,
                tag,
            )
        from nooa.tracing._viewer_auth import apply_viewer_auth

        # A remote viewer rejects unauthenticated writes with 403.
        headers = apply_viewer_auth({"Content-Type": "application/json"})
        if session_id:
            # The receiver's /v1/journal/blocks route uses this header to
            # key blocks per session.  /v1/journal/calls reads session_id
            # off the body, but it's harmless to pass on both routes.
            headers["X-Session-Id"] = session_id
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        t0 = time.monotonic()
        for attempt in range(_POST_RETRIES):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    elapsed_ms = (time.monotonic() - t0) * 1000
                    if resp.status < 300:
                        if elapsed_ms > 2000:
                            log.info(
                                "POST %s OK but slow: %.0fms (attempt %d)%s",
                                url,
                                elapsed_ms,
                                attempt + 1,
                                tag,
                            )
                        if on_success is not None:
                            on_success()
                        return
                    log.debug("POST %s returned HTTP %s%s", url, resp.status, tag)
            except Exception as exc:
                elapsed_ms = (time.monotonic() - t0) * 1000
                if attempt < _POST_RETRIES - 1:
                    log.info(
                        "POST %s attempt %d/%d failed after %.0fms: %s%s — retrying in %.0fs",
                        url,
                        attempt + 1,
                        _POST_RETRIES,
                        elapsed_ms,
                        exc,
                        tag,
                        _POST_RETRY_DELAYS[attempt],
                    )
                    time.sleep(_POST_RETRY_DELAYS[attempt])
                else:
                    log.warning(
                        "POST %s failed after %d attempts (%.0fms total): %s (%.0f KB, %d item(s))%s",
                        url,
                        _POST_RETRIES,
                        elapsed_ms,
                        exc,
                        size_kb,
                        n_items,
                        tag,
                    )

    t = threading.Thread(target=_send, daemon=True)
    # Add to the set then start, with the lock held across both.  The
    # thread's ``finally`` block also acquires this lock to self-evict,
    # so even if ``_send`` finishes before this block returns, its
    # ``discard`` blocks until we've added — no risk of a "discard runs
    # before add, then add leaves a dead-thread reference in the set"
    # leak.  And ``flush_pending`` can never see the thread as
    # not-yet-started: it joins with the lock released, but by then the
    # thread is started and either alive or already self-evicted.
    with _PENDING_LOCK:
        _PENDING_THREADS.add(t)
        t.start()


# Process-wide lock guarding the scan-and-append on ``litellm.callbacks``
# in :meth:`JournalExporter._install`.  Two ``enable_tracing`` calls for
# the same base_url racing concurrently would otherwise both scan, see
# no existing :class:`MessageJournalCallback`, and each append a fresh
# one -- the per-Destination refcount only converges shared state when
# both exporters land on the *same* callback object.
_INSTALL_LOCK = threading.Lock()


# In-flight journal POST threads.  ``flush_pending`` waits on them so
# fire-and-forget POSTs aren't truncated when the calling process exits
# (the eval pipeline's daemon-thread POSTs were getting ClientDisconnect
# at the receiver because the test process exited before their HTTP
# request body was fully sent).  Stored as a set so completed threads
# can be evicted in O(1); ``_send`` self-evicts on its way out.
_PENDING_THREADS: set[threading.Thread] = set()
_PENDING_LOCK = threading.Lock()


def flush_pending(timeout: float = 30.0) -> None:
    """Block until all in-flight journal POSTs have finished or *timeout* expires.

    Called by ``JournalExporter.force_flush`` so OTel ``force_flush()`` (the
    eval pipeline calls it before scoring) covers the journal sideband as
    well as the OTLP span exporter.  Threads that haven't joined within
    *timeout* are abandoned -- they're daemons, so process exit will kill
    them, but ``_post_json`` will already have logged a warning if their
    HTTP POST failed.
    """
    deadline = time.monotonic() + timeout
    while True:
        with _PENDING_LOCK:
            threads = [t for t in _PENDING_THREADS if t.is_alive()]
        if not threads:
            return
        for t in threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            t.join(timeout=remaining)
        # Loop once more in case new POSTs were dispatched while we were
        # joining the first batch -- otherwise an LLM call that fires
        # multiple POSTs back-to-back could partially miss the flush.


def _to_ts(t: Any) -> float:
    """Convert a datetime or numeric value to a Unix timestamp float."""
    if hasattr(t, "timestamp"):
        return t.timestamp()
    try:
        return float(t)
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Callback
# ---------------------------------------------------------------------------


class _Destination:
    """One journal sink: pair of POST URLs + per-session block dedup state
    + a refcount so multiple ``JournalExporter``s pointed at the same
    base URL share state instead of stomping on each other at shutdown.

    Each ``MessageJournalCallback`` holds an ordered list of these so the
    eval pipeline can fan a single LLM call's journal payload out to both
    the in-process headless backend and the user's running viewer without
    needing two ``litellm.callbacks`` entries (litellm only delivers
    ``log_success_event`` to one callback per class).
    """

    __slots__ = (
        "base",
        "blocks_url",
        "calls_url",
        "refcount",
        "sent_session",
        "sent_hashes",
    )

    def __init__(self, base_url: str) -> None:
        base = base_url.rstrip("/")
        self.base = base
        self.blocks_url = base + "/v1/journal/blocks"
        self.calls_url = base + "/v1/journal/calls"
        # How many ``JournalExporter`` instances reference this destination.
        # Decremented on each shutdown; the destination is removed only
        # when the last referencing exporter shuts down.
        self.refcount: int = 1
        # Per-session block hash set.  When the session changes the old
        # set is dropped, so memory stays bounded regardless of how many
        # sessions the process sees.
        self.sent_session: str = ""
        self.sent_hashes: set[str] = set()


class MessageJournalCallback(CustomLogger):
    """LiteLLM callback that streams skeletons + content-addressed blocks.

    Per LLM call:

    1. ``log_pre_api_call`` consumes the journal sideband (populated by
       the nooa actor's ``_build_journal_payload``) and POSTs
       any newly-seen blocks (by hash) to ``/v1/journal/blocks`` *on every
       configured destination*.  The skeleton — the wire message list
       with block refs replaced by hashes — is held until the call
       completes.
    2. ``log_success_event`` POSTs the full call record to
       ``/v1/journal/calls`` *on every configured destination*.  Output
       messages (assistant replies) are included inline.

    Multiple destinations live behind a single callback so they all get
    every event.  litellm dispatches ``log_pre_api_call`` to every
    callback in ``litellm.callbacks`` but only delivers
    ``log_success_event`` to one per class — putting all destinations
    inside one callback instance sidesteps that asymmetry.

    Per-destination, per-session in-memory hash sets avoid retransmitting
    blocks that were already sent for that session.
    """

    def __init__(self, base_url: str | list[str]) -> None:
        super().__init__()
        if isinstance(base_url, str):
            urls: list[str] = [base_url]
        else:
            urls = list(base_url)
        if not urls:
            raise ValueError("MessageJournalCallback requires at least one base URL")
        self._destinations: list[_Destination] = [_Destination(u) for u in urls]
        # litellm_call_id -> (input_skeleton, span_id | None). Entries
        # are removed in log_success_event / log_failure_event.
        self._call_inputs: dict[str, tuple[list[dict], str | None]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Destination management
    # ------------------------------------------------------------------

    @property
    def base_urls(self) -> list[str]:
        """Return the list of destination base URLs (no trailing slash)."""
        with self._lock:
            return [d.base for d in self._destinations]

    def add_destination(self, base_url: str) -> None:
        """Add another sink, or bump the refcount on an existing one.

        Used by ``JournalExporter`` so two ``enable_tracing`` exporters
        pointed at the same URL share state.  Each ``add_destination``
        must be paired with one ``remove_destination`` -- the destination
        sticks around until *every* exporter that added it has gone away.
        """
        normalized = base_url.rstrip("/")
        with self._lock:
            for dest in self._destinations:
                if dest.base == normalized:
                    dest.refcount += 1
                    return
            self._destinations.append(_Destination(base_url))

    def remove_destination(self, base_url: str) -> None:
        """Decrement refcount for a sink; remove only when refcount hits 0.

        Pairs with :meth:`add_destination`.  No-op if the URL isn't
        configured.
        """
        normalized = base_url.rstrip("/")
        with self._lock:
            for dest in list(self._destinations):
                if dest.base == normalized:
                    dest.refcount -= 1
                    if dest.refcount <= 0:
                        self._destinations.remove(dest)
                    return

    def has_destinations(self) -> bool:
        with self._lock:
            return bool(self._destinations)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _session(self) -> str:
        return get_session() or "unknown"

    def _send_new_blocks(self, session_id: str, blocks: dict[str, str]) -> None:
        """POST any blocks this session hasn't seen yet, to every destination.

        Per-destination, per-session hash tracking: a freshly-started
        sink doesn't free-ride on a long-lived sink's dedup state.
        Hashes are recorded after the POST succeeds, so a failed
        fire-and-forget attempt is retried on the next call.

        Session change resets the tracking set so memory stays bounded
        regardless of how many sessions a process sees.  ``on_success``
        verifies the destination's session is still the one we
        dispatched for -- a session change between dispatch and
        completion (parallel agents racing) must not clobber the new
        session's set with stale hashes.
        """
        if not blocks:
            return
        with self._lock:
            destinations = list(self._destinations)

        for dest in destinations:
            with self._lock:
                if session_id != dest.sent_session:
                    dest.sent_session = session_id
                    dest.sent_hashes = set()
                new_entries = [
                    {"hash": h, "content": c}
                    for h, c in blocks.items()
                    if h not in dest.sent_hashes
                ]
            if not new_entries:
                continue

            def _on_success(
                d: _Destination = dest,
                sid: str = session_id,
                hashes: frozenset = frozenset(e["hash"] for e in new_entries),
            ) -> None:
                with self._lock:
                    if d.sent_session != sid:
                        # Session changed under us; the new session's set
                        # is for a different conversation, don't pollute it.
                        return
                    d.sent_hashes.update(hashes)

            _post_json(
                dest.blocks_url,
                new_entries,
                session_id=session_id,
                on_success=_on_success,
            )

    # ------------------------------------------------------------------
    # Sync hooks
    # ------------------------------------------------------------------

    @staticmethod
    def _current_span_id() -> str | None:
        """Return the active OTel span_id as a 16-char hex string, or None."""
        try:
            ctx = otel_trace.get_current_span().get_span_context()
            if ctx.is_valid:
                return format(ctx.span_id, "016x")
        except Exception as exc:
            log.debug("Failed to read current span_id: %s", exc)
        return None

    def log_pre_api_call(self, model: str, messages: list, kwargs: dict) -> None:
        from nooa.tracing._context_sideband import (
            get_journal_payload,
            set_journal_payload,
        )

        session_id = self._session()
        call_id = kwargs.get("litellm_call_id", "")

        payload = get_journal_payload()
        if payload is not None:
            set_journal_payload(None)  # consume
            self._send_new_blocks(session_id, payload.blocks)
            input_skeleton = payload.skeleton
        else:
            # No sideband — publish the raw messages as the skeleton
            # with no block refs. The viewer just uses their content
            # as-is, matching what the wire shows.
            input_skeleton = [_msg_to_dict(m) for m in messages]

        span_id = self._current_span_id()
        with self._lock:
            self._call_inputs[call_id] = (input_skeleton, span_id)

    def log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        session_id = self._session()
        call_id = kwargs.get("litellm_call_id", "")
        with self._lock:
            stored = self._call_inputs.pop(call_id, None)
        if stored is None:
            log.warning(
                "[MessageJournal] log_success_event fired for call_id=%r with no prior "
                "log_pre_api_call — input_skeleton will be empty (retry or out-of-order event?)",
                call_id,
            )
            input_skeleton, span_id = [], None
        else:
            input_skeleton, span_id = stored

        raw_output = _extract_output_msgs(response_obj)
        # Content-address the output messages too. The assistant's reply
        # itself doesn't dedup across *different* calls, but a single
        # reply can already be hundreds of KB when it embeds a large
        # code block in ``tool_call.arguments`` — routing that body
        # through the blocks sideband keeps the /v1/journal/calls record
        # small and re-uses any hash that overlaps with messages the
        # agent will echo back on subsequent turns.
        output_blocks: dict[str, str] = {}
        output_messages = [_skeleton_dict_message(m, output_blocks) for m in raw_output]
        if output_blocks:
            self._send_new_blocks(session_id, output_blocks)

        usage = getattr(response_obj, "usage", None)
        tokens: dict[str, int] | None = None
        if usage:
            tokens = {
                "prompt": getattr(usage, "prompt_tokens", 0) or 0,
                "completion": getattr(usage, "completion_tokens", 0) or 0,
            }
            details = getattr(usage, "prompt_tokens_details", None)
            if details:
                cached = getattr(details, "cached_tokens", 0) or 0
                if cached:
                    tokens["cached"] = cached

        record: dict = {
            "call_id": call_id,
            "session_id": session_id,
            "model": kwargs.get("model", ""),
            "ts_start": _to_ts(start_time),
            "ts_end": _to_ts(end_time),
            "input_skeleton": input_skeleton,
            "output_messages": output_messages,
            "tokens": tokens,
        }
        if span_id:
            record["span_id"] = span_id
        self._send_call(session_id, record)

    def _send_call(self, session_id: str, record: dict) -> None:
        """Publish a completed call record to every HTTP destination."""
        with self._lock:
            destinations = list(self._destinations)
        for dest in destinations:
            _post_json(dest.calls_url, record, session_id=session_id)

    def log_failure_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        call_id = kwargs.get("litellm_call_id", "")
        with self._lock:
            self._call_inputs.pop(call_id, None)

    # ------------------------------------------------------------------
    # Async hooks — delegate to sync; ContextVar propagates correctly
    # ------------------------------------------------------------------

    async def async_log_pre_api_call(self, model: str, messages: list, kwargs: dict) -> None:
        self.log_pre_api_call(model, messages, kwargs)

    async def async_log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self.log_success_event(kwargs, response_obj, start_time, end_time)

    async def async_log_failure_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        self.log_failure_event(kwargs, response_obj, start_time, end_time)


class FileMessageJournalCallback(MessageJournalCallback):
    """Write the existing journal protocol to a :class:`JournalFileWriter`.

    The message normalization and call correlation intentionally reuse the
    HTTP callback implementation so live and file journals have identical
    payload semantics.
    """

    def __init__(self, writer: Any) -> None:
        CustomLogger.__init__(self)
        self._call_inputs: dict[str, tuple[list[dict], str | None]] = {}
        self._lock = threading.Lock()
        self._writers: dict[Any, int] = {writer: 1}
        self._sent_hashes: dict[Any, dict[str, set[str]]] = {writer: {}}

    def add_writer(self, writer: Any) -> None:
        with self._lock:
            self._writers[writer] = self._writers.get(writer, 0) + 1
            self._sent_hashes.setdefault(writer, {})

    def remove_writer(self, writer: Any) -> None:
        with self._lock:
            refcount = self._writers.get(writer, 0) - 1
            if refcount <= 0:
                self._writers.pop(writer, None)
                self._sent_hashes.pop(writer, None)
            else:
                self._writers[writer] = refcount

    def has_writers(self) -> bool:
        with self._lock:
            return bool(self._writers)

    def _send_new_blocks(self, session_id: str, blocks: dict[str, str]) -> None:
        if not blocks:
            return
        with self._lock:
            for writer in self._writers:
                sent = self._sent_hashes.setdefault(writer, {}).setdefault(session_id, set())
                new_entries = [
                    {"hash": h, "content": content}
                    for h, content in blocks.items()
                    if h not in sent
                ]
                # Mark hashes only after the append succeeds.  A disk-full or
                # serialization failure must remain retryable on the next call.
                writer.append_blocks(session_id, new_entries)
                sent.update(entry["hash"] for entry in new_entries)

    def _send_call(self, session_id: str, record: dict) -> None:
        with self._lock:
            writers = list(self._writers)
        for writer in writers:
            writer.append_call(session_id, record)
