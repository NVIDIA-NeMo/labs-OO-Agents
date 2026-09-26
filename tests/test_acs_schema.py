# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from nooa.acs.schema import ACSDecision, ACSHook

def test_acs_hook_serialization():
    """Test that ACSHook can successfully round-trip to/from JSON."""
    hook = ACSHook(
        hook_type="toolCallRequest",
        payload={"tool_name": "shell", "command": "ls -la"},
        provenance={"session_id": "12345", "user": "dev"}
    )
    
    json_data = hook.model_dump_json()
    assert "toolCallRequest" in json_data
    assert "shell" in json_data
    
    # Round trip
    reconstructed = ACSHook.model_validate_json(json_data)
    assert reconstructed.hook_type == "toolCallRequest"
    assert reconstructed.payload["tool_name"] == "shell"
    assert reconstructed.provenance["session_id"] == "12345"

def test_acs_decision_serialization():
    """Test that ACSDecision can successfully round-trip to/from JSON."""
    decision = ACSDecision(
        status="defer",
        reason="Rate limit exceeded",
        retry_after_ms=5000,
        decision_id="dec-999"
    )
    
    json_data = decision.model_dump_json()
    assert "defer" in json_data
    assert "5000" in json_data
    
    # Round trip
    reconstructed = ACSDecision.model_validate_json(json_data)
    assert reconstructed.status == "defer"
    assert reconstructed.retry_after_ms == 5000
    assert reconstructed.reason == "Rate limit exceeded"

def test_malformed_hook_data_fails_validation():
    """Test that invalid hook types throw validation errors."""
    from pydantic import ValidationError
    import pytest
    
    with pytest.raises(ValidationError):
        # 'invalidType' is not in the HookType Literal
        ACSHook(hook_type="invalidType", payload={})
