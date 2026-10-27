# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact bounds, serialized control evidence and declaration persistence."""

import json
from copy import deepcopy

import httpx
import pytest
import yaml

from nooa.unifiedllm import connect, registry
from nooa.unifiedllm.connect._records import text_baseline_passed
from tests.unifiedllm.connect.connect_http import mock_http, response_body


def plan(style="chat", **kwargs):
    return connect.plan(
        "local", "model", style, "https://api.test/v1", "", reply_tokens=1000, **kwargs
    )


@pytest.mark.parametrize("declared,effective", [(128000, 64000), (48000, 48000)])
@pytest.mark.parametrize("field", ["context_window", "context_length", "max_input_tokens"])
def test_explicit_context_uses_exact_endpoint_bound(declared, effective, field):
    endpoint = {"id": "model", field: 64000, "max_output_tokens": 2000}
    before = deepcopy(endpoint)
    entry = plan(context_window=declared, endpoint_model=endpoint).entry
    assert entry["context_window"] == effective
    assert entry["max_tokens"] == 1000
    evidence = entry["provenance"]["context_window"]
    assert evidence["declared"] == declared and evidence["effective"] == effective
    assert evidence["source"] == "user"
    assert endpoint == before


@pytest.mark.parametrize(
    "endpoint",
    [
        {"id": "neighbour", "max_input_tokens": 64000},
        {"max_input_tokens": 64000},
        {"id": "model", "max_input_tokens": True, "formats": {"context_window": 64000}},
    ],
)
def test_explicit_context_never_uses_neighbours_formats_or_idless_hints(endpoint):
    assert plan(context_window=128000, endpoint_model=endpoint).entry["context_window"] == 128000


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("control", ["thinking", "chat_template_kwargs", "extra_body"])
@pytest.mark.parametrize("drop", [False, True])
@pytest.mark.asyncio
async def test_typed_probe_observes_serialized_controls(monkeypatch, selected, control, drop):
    value = {"type": "disabled"} if control == "thinking" else {"enable_thinking": False}
    settings = (
        {control: value}
        if control != "extra_body"
        else {"extra_body": {"custom_control": "strict"}}
    )
    wire_key = control if control != "extra_body" else "custom_control"
    selected_settings = settings if control != "extra_body" else {"temperature": 0.2}
    proposal = (
        plan(reasoning_levels={"custom": selected_settings}, reasoning_level="custom")
        if selected
        else plan()
    )
    if not selected or control == "extra_body":
        proposal.entry.update(settings)
    original = registry.client_from_config
    bodies = []

    def factory(*args, **kwargs):
        client = original(*args, **kwargs)

        async def mutate(request):
            body = json.loads(request.content)
            assert wire_key in body
            if drop:
                body.pop(wire_key)
                request._content = json.dumps(body).encode()
                request.headers["content-length"] = str(len(request._content))

        client._http.httpx_async.event_hooks["request"].insert(0, mutate)
        return client

    monkeypatch.setattr(registry, "client_from_config", factory)
    mock_http(
        monkeypatch,
        lambda request: (
            bodies.append(json.loads(request.content))
            or httpx.Response(200, json=response_body("chat"))
        ),
    )
    result = await connect.check_stage(proposal, "routing", api_key="fake")
    record = result.entry["provenance"]["probes"]["routing"]
    assert record["answer_correct"] and record["typed_result_validated"], record
    assert record["settings_sent"] is (not drop), record
    assert text_baseline_passed(record) is (not drop)
    assert len(bodies) == 1
    assert (wire_key in bodies[0]) is (not drop)


@pytest.mark.parametrize("store", [False, True])
@pytest.mark.asyncio
async def test_responses_save_replan_reload_body_parity(monkeypatch, tmp_path, store):
    proposal = plan("responses")
    proposal.entry.update(store=store, include=[])
    path = tmp_path / "models.yaml"
    connect.write(proposal.entry, path, alias="local")
    saved = yaml.safe_load(path.read_text())["models"]["local"]
    assert "provenance" not in saved
    replanned = plan("responses", existing_entry=saved)
    assert replanned.entry["store"] is store and replanned.entry["include"] == []
    second = tmp_path / "replanned.yaml"
    connect.write(replanned.entry, second, alias="local")
    bodies = []
    mock_http(
        monkeypatch,
        lambda request: (
            bodies.append(json.loads(request.content))
            or httpx.Response(200, json=response_body("responses"))
        ),
    )
    for file in (path, second):
        config = yaml.safe_load(file.read_text())["models"]["local"]
        client = registry.client_from_config("local", config, api_key="fake")
        try:
            await client.acall(messages=[{"role": "user", "content": "Hello"}])
        finally:
            await client.aclose()
    assert bodies[0] == bodies[1]
    # The SDK omits the empty include list; parity must not resurrect encryption.
    assert bodies[0]["store"] is store and bodies[0].get("include", []) == []


@pytest.mark.parametrize("change", ["model", "endpoint", "credential", "style"])
def test_responses_declarations_do_not_cross_route_scope(change):
    old = plan("responses").entry
    old.update(store=True, include=[])
    args = [
        "local",
        "other" if change == "model" else "model",
        "chat" if change == "style" else "responses",
        "https://other.test/v1" if change == "endpoint" else "https://api.test/v1",
        "OTHER_KEY" if change == "credential" else "",
    ]
    entry = connect.plan(*args, existing_entry=old).entry
    if change == "style":
        assert "store" not in entry and "include" not in entry
    else:
        assert entry["store"] is False and entry["include"] == ["reasoning.encrypted_content"]
