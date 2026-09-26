# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from typing import Any, Callable, Dict, Optional
from nooa.acs.schema import ACSDecision, ACSHook
from nooa.acs.handler import ACSHandler

class ACSBlockedError(Exception):
    """Raised when an operation is blocked by the ACS Guardian."""
    def __init__(self, reason: Optional[str] = None):
        super().__init__(f"Operation blocked by governance policy. Reason: {reason or 'Denied'}")
        self.reason = reason


async def enforce_acs_decision(
    handler: ACSHandler,
    hook_type: str,
    payload: Dict[str, Any],
    provenance: Dict[str, Any],
    execute_func: Callable[[], Any]
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
    """
    hook = ACSHook(
        hook_type=hook_type,  # type: ignore
        payload=payload,
        provenance=provenance
    )
    
    # Evaluate the decision, potentially retrying if deferred
    while True:
        decision = await handler.evaluate(hook)
        
        if decision.status == "allow":
            # Execute the actual logic
            if asyncio.iscoroutinefunction(execute_func):
                return await execute_func()
            else:
                return execute_func()
                
        elif decision.status == "deny":
            # Block execution and raise an exception that the agent/system can handle
            raise ACSBlockedError(reason=decision.reason)
            
        elif decision.status == "ask":
            # Human-in-the-loop fallback.
            # In a real implementation, this might raise a special 'RequiresApprovalError'
            # or trigger a blocking prompt to the console/UI. For now, we simulate asking
            # by either raising or (if we have an approval callback injected) waiting for it.
            # TODO: Integrate with NOOA's standard interactive prompt system.
            raise ACSBlockedError(reason="Requires human approval (not yet implemented in current context).")
            
        elif decision.status == "defer":
            # Backoff and retry
            wait_ms = decision.retry_after_ms or 1000
            await asyncio.sleep(wait_ms / 1000.0)
