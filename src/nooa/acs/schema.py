# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

DecisionStatus = Literal["allow", "deny", "ask", "defer"]
HookType = Literal[
    "sessionStart", 
    "sessionEnd", 
    "turnStart", 
    "turnEnd", 
    "toolCallRequest", 
    "memoryRead", 
    "memoryWrite"
]

class ACSDecision(BaseModel):
    """A governance decision returned by the Guardian in response to an ACS hook."""
    status: DecisionStatus
    reason: Optional[str] = None
    decision_id: Optional[str] = None
    retry_after_ms: Optional[int] = Field(
        default=None, 
        description="Used only when status='defer'. Wait time before retry."
    )

class ACSHook(BaseModel):
    """The structured hook emitted by NOOA for governance decisions."""
    hook_type: HookType
    payload: Dict[str, Any]
    provenance: Dict[str, Any] = Field(default_factory=dict)
