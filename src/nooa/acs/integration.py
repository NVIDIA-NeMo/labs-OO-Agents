# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import inspect
from typing import Any, Callable, Dict, Optional
from nooa.acs.schema import ACSDecision, ACSHook
from nooa.acs.handler import ACSHandler

class ACSBlockedError(Exception):
    """Raised when an operation is blocked by the ACS Guardian."""
    def __init__(self, reason: Optional[str] = None):
        super().__init__(f"Operation blocked by governance policy. Reason: {reason or 'Denied'}")
        self.reason = reason


class ACSApprovalRequiredError(Exception):
    """Raised when an action requires approval before it can execute."""

    def __init__(
        self, reason: Optional[str] = None, decision_id: Optional[str] = None
    ):
        super().__init__(f"Approval required. Reason: {reason or 'Human approval required'}")
        self.reason = reason
        self.decision_id = decision_id


class ACSDeferExhaustedError(Exception):
    """Raised when a deferred decision exceeds its retry limit or deadline."""


def acs_thread_safe(callback: Callable[[], Any]) -> Callable[[], Any]:
    """Mark a synchronous ACS action callback safe to run in a worker thread."""
    setattr(callback, "__acs_thread_safe__", True)
    return callback


async def enforce_acs_decision(
    handler: ACSHandler,
    hook_type: str,
    payload: Dict[str, Any],
    provenance: Dict[str, Any],
    execute_func: Callable[[], Any],
    *,
    max_defer_retries: int = 5,
    defer_timeout: float | None = 30.0,
) -> Any:
    """
    Wraps an executable function with an ACS governance check.
    
    Args:
        handler: The configured ACSHandler for the agent.
        hook_type: The type of hook being emitted (e.g., 'toolCallRequest').
        payload: The payload describing the action.
        provenance: Metadata about the session/turn context.
        execute_func: An async or sync callable that actually performs the action.
        
    Returns:
        The result of execute_func() if allowed.
        
        Raises:
        ACSBlockedError: If the Guardian denies the request.
        ACSApprovalRequiredError: If the Guardian requires human approval.
        ACSDeferExhaustedError: If deferred decisions exceed configured bounds.
    """
    if max_defer_retries < 0:
        raise ValueError("max_defer_retries must be nonnegative")
    if defer_timeout is not None and defer_timeout <= 0:
        raise ValueError("defer_timeout must be positive or None")

    hook = ACSHook(
        hook_type=hook_type,  # type: ignore
        payload=payload,
        provenance=provenance
    )
    
    loop = asyncio.get_running_loop()
    deadline = loop.time() + defer_timeout if defer_timeout is not None else None
    defer_retries = 0

    # Evaluate the decision, potentially retrying if deferred.
    while True:
        remaining = deadline - loop.time() if deadline is not None else None
        if remaining is not None and remaining <= 0:
            raise ACSDeferExhaustedError("ACS defer deadline reached")
        try:
            evaluation = handler.evaluate(hook)
            if remaining is None:
                decision = await evaluation
            else:
                decision = await asyncio.wait_for(
                    evaluation, timeout=remaining
                )
        except asyncio.TimeoutError as error:
            if deadline is None or loop.time() < deadline:
                raise
            raise ACSDeferExhaustedError(
                "ACS defer deadline reached while awaiting a decision"
            ) from error

        if deadline is not None and loop.time() >= deadline:
            raise ACSDeferExhaustedError("ACS defer deadline reached")
        
        if decision.status == "allow":
            if getattr(execute_func, "__acs_thread_safe__", False):
                result = await asyncio.to_thread(execute_func)
            else:
                result = execute_func()
            if inspect.isawaitable(result):
                return await result
            return result
                
        elif decision.status == "deny":
            # Block execution and raise an exception that the agent/system can handle
            raise ACSBlockedError(reason=decision.reason)
            
        elif decision.status == "ask":
            raise ACSApprovalRequiredError(
                reason=decision.reason, decision_id=decision.decision_id
            )
            
        elif decision.status == "defer":
            if defer_retries >= max_defer_retries:
                raise ACSDeferExhaustedError(
                    f"ACS defer retry limit reached ({max_defer_retries})"
                )

            remaining = deadline - loop.time() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                raise ACSDeferExhaustedError("ACS defer deadline reached")

            wait_ms = (
                decision.retry_after_ms
                if decision.retry_after_ms is not None
                else 1000
            )
            wait_seconds = wait_ms / 1000.0
            if remaining is not None and wait_seconds >= remaining:
                await asyncio.sleep(remaining)
                raise ACSDeferExhaustedError("ACS defer deadline reached")

            await asyncio.sleep(wait_seconds)
            defer_retries += 1
