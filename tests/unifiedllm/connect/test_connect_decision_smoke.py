# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native wire integration of bounded typed decision and image fixtures."""

import json
from dataclasses import replace
from email import policy
from email.parser import BytesParser

import httpx
import pytest
import yaml

from nooa.unifiedllm import connect
from nooa.unifiedllm.connect._media import png_fixture
from tests.unifiedllm.connect.connect_http import mock_http

ENDPOINT = "https://fixture.test/v1/decisions"


def payload(request):
    if request.headers.get("content-type", "").startswith("multipart/"):
        message = BytesParser(policy=policy.default).parsebytes(
            ("Content-Type: " + request.headers["content-type"] + "\r\n\r\n").encode()
            + request.content
        )
        parts = list(message.iter_parts())
        assert parts[1].get_payload(decode=True) == png_fixture()
        assert request.headers["TypeSafe-Danger-Zone-Images"] == "true"
        return json.loads(parts[0].get_payload(decode=True))
    return json.loads(request.content)


def fixture_response(request, *, wrong=False, corrupt=None, usage=10):
    body = payload(request)
    style = "openai" if "input" in body else "systemone"
    state = body.get("state", body.get("input"))
    if isinstance(state, list):
        blocks = state[0]["content"]
        assert blocks[2]["type"] == "input_image"
        state = blocks[0]["text"]
    if isinstance(state, str):
        state = json.loads(state)
    number = state.get("inputs", {}).get("number", 7)
    answers = [] if style == "openai" else {}
    questions = (
        body["questions"]
        if style == "openai"
        else [{"name": name, **q} for name, q in body["questions"].items()]
    )
    for question in questions:
        name, kind = question["name"], question["type"]
        if kind in {"noul", "predicate"}:
            probability = 0.99 if number > 0 else 0.01
            if wrong and name == "result":
                probability = 1 - probability
            value = (
                {"type": "predicate", "probability": probability}
                if style == "openai"
                else {"noul": probability}
            )
        elif kind == "choice":
            keys = (
                [c["value"] for c in question["choices"]]
                if style == "openai"
                else list(question["criteria"])
            )
            selected = keys[0] if name == "color" else keys[2]
            if wrong and name == "color":
                selected = keys[1]
            probabilities = {key: float(key == selected) for key in keys}
            if corrupt == "distribution":
                probabilities = dict.fromkeys(keys, 0.1)
            if corrupt == "nan":
                probabilities[keys[0]] = float("nan")
            value = {"choice": selected, "probabilities": probabilities, "confidence": 1.0}
            if style == "openai":
                value.update(
                    type="choice",
                    probabilities=[
                        {"value": key, "probability": p} for key, p in probabilities.items()
                    ],
                )
        else:
            probabilities = {"0": 0.0, "1": 0.0, "2": 1.0}
            value = {
                "score": 1.0 if corrupt == "score" else 2.0,
                "probabilities": probabilities,
                "confidence": 1.0,
            }
            if style == "openai":
                value.update(
                    type="score",
                    probabilities=[
                        {"value": key, "probability": p} for key, p in probabilities.items()
                    ],
                )
            else:
                value["legend"] = dict(enumerate(question["criteria"]))
        if style == "openai":
            answers.append({"name": name, **value})
        else:
            answers[name] = value
    return httpx.Response(
        200,
        json={
            "model": "fixture",
            "answers": answers,
            "usage": {"input_tokens": usage, "output_tokens": 0},
        },
    )


def proposal(style="openai", image=False, **kwargs):
    options = (
        {"decision_image_transport": "typesafe_multipart", "typesafe_danger_zone_images": True}
        if style == "systemone" and image
        else {}
    )
    return connect.plan(
        "fixture",
        "fixture",
        style,
        ENDPOINT,
        "",
        test_media_inputs=["image"] if image else None,
        **options,
        **kwargs,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["openai", "systemone"])
async def test_actual_typed_and_image_wire_semantics(monkeypatch, style):
    seen = []

    def handle(request):
        seen.append(request)
        return fixture_response(request)

    mock_http(monkeypatch, handle)
    plan = proposal(style, image=True)
    assert "base64" not in json.dumps(plan.probes[-1].body)
    result = await connect.check_stage(plan, "media")
    assert len(seen) == 5
    records = result.entry["provenance"]["probes"]
    assert records["decision:typed"]["typed_result"]["sign"] == "positive"
    assert records["decision:typed"]["typed_result"]["score"]["value"] == 2
    assert records["decision:negative"]["typed_result"] is False
    assert records["media:image"]["outcome"] == "passed"
    assert records["media:image"]["attachment_observed"] is True
    assert records["media:image"]["answer_correct"] is True
    assert result.entry["media_inputs"] == []  # facts do not grant permission in the library
    assert result.entry["provenance"]["tokens_charged_to_budget"] == 6144
    public = connect.diagnostic_prompt("media", result.entry, records)
    assert "base64" not in public and "data:image" not in public


@pytest.mark.asyncio
async def test_wrong_native_answers_are_not_accuracy_passes(monkeypatch):
    mock_http(monkeypatch, lambda request: fixture_response(request, wrong=True))
    result = await connect.check_stage(proposal(image=True), "all")
    records = result.entry["provenance"]["probes"]
    assert records["routing"]["outcome"] == "accepted"
    assert records["decision:positive"]["request_accepted"] is True
    assert records["decision:positive"]["answer_correct"] is False
    assert records["decision:positive"]["outcome"] == "not_confirmed"
    assert records["media:image"]["outcome"] == "not_probed"


@pytest.mark.asyncio
async def test_selfreported_access_not_enough_when_color_wrong(monkeypatch):
    mock_http(
        monkeypatch,
        lambda request: fixture_response(request, wrong="image_url" in request.content.decode()),
    )
    result = await connect.check_stage(proposal(image=True), "media")
    record = result.entry["provenance"]["probes"]["media:image"]
    assert record["access_reported"] is True
    assert record["attachment_observed"] is True
    assert record["answer_correct"] is False
    assert record["outcome"] == "not_confirmed"


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["openai", "systemone"])
@pytest.mark.parametrize("corrupt", ["distribution", "score", "nan"])
async def test_schema_contract_failure_separate_from_http_acceptance(monkeypatch, style, corrupt):
    # httpx JSON encoding rejects NaN, so supply a raw malformed JSON body for that case.
    def handle(request):
        response = fixture_response(request, corrupt=None if corrupt == "nan" else corrupt)
        if corrupt == "nan" and b'"score"' in request.content:
            return httpx.Response(
                200,
                content=response.content.replace(b'"score":2.0', b'"score":NaN').replace(
                    b'"score": 2.0', b'"score": NaN'
                ),
            )
        return response

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(proposal(style), "decision")
    record = result.entry["provenance"]["probes"]["decision:typed"]
    assert record["status_code"] == 200
    assert record["request_accepted"] is True
    assert record["typed_result_validated"] is False
    assert record["outcome"] == "not_confirmed"
    assert "typed_result" not in record


@pytest.mark.asyncio
async def test_shared_budget_usage_no_retry_and_freshness(monkeypatch):
    seen = []

    def handle(request):
        seen.append(request)
        return fixture_response(request, usage=600)

    mock_http(monkeypatch, handle)
    plan = proposal(budget_tokens=1200)
    result = await connect.run(plan, approved="all")
    assert len(seen) == 2
    assert result.entry["provenance"]["tokens_charged_to_budget"] == 1200
    assert result.entry["provenance"]["probes"]["decision:negative"]["reason"] == "budget exhausted"
    full = proposal(existing_entry=result.entry)
    result = await connect.run(full, approved="all")
    assert len(seen) == 5  # routing reused, 3 fresh tests
    result = await connect.run(replace(full, entry=result.entry), approved="all")
    assert len(seen) == 8  # accuracy never reused
    await connect.check_stage(replace(full, entry=result.entry), "routing")
    assert len(seen) == 9  # explicit routing is fresh


@pytest.mark.asyncio
async def test_timeout_and_http_failure_are_bounded_no_retries(monkeypatch):
    seen = []

    def handle(request):
        seen.append(request)
        raise httpx.ReadTimeout("PRIVATE provider payload data:image/base64")

    mock_http(monkeypatch, handle)
    result = await connect.run(proposal(image=True), approved="all")
    assert len(seen) == 1
    assert result.entry["provenance"]["probes"]["routing"]["timeout_kind"] == "ReadTimeout"
    assert "PRIVATE" not in json.dumps(result.entry)
    assert result.entry["provenance"]["tokens_charged_to_budget"] == 512


@pytest.mark.parametrize("style", ["openai", "systemone"])
def test_save_reload_and_route_reset_safe(monkeypatch, tmp_path, style):
    from nooa.unifiedllm.registry import client_from_config

    plan = proposal(style, image=True)
    entry = {**plan.entry, "media_inputs": ["image"]}
    path = tmp_path / "models.yaml"
    connect.write(entry, path, alias="fixture")
    loaded = yaml.safe_load(path.read_text())["models"]["fixture"]
    assert "provenance" not in loaded
    assert "base64" not in path.read_text() and "typed_result" not in path.read_text()
    checked = connect.configure_entry(loaded)
    client = client_from_config("fixture", checked)
    assert client.media_inputs == ("image",)
    assert client.decision_image_transport == entry["decision_image_transport"]
    assert client.typesafe_danger_zone_images == entry["typesafe_danger_zone_images"]
    same = proposal(style, existing_entry=checked)
    assert same.entry["media_inputs"] == ["image"]
    changed = connect.plan("fixture", "other", style, ENDPOINT, "", existing_entry=checked)
    assert changed.entry["media_inputs"] == []
    assert changed.entry["decision_image_transport"] == "json"
    assert changed.entry["typesafe_danger_zone_images"] is False
    reset = connect.plan(
        "fixture",
        "fixture",
        style,
        ENDPOINT,
        "",
        existing_entry=checked,
        decision_image_transport="json",
        typesafe_danger_zone_images=False,
    )
    if style == "systemone":
        assert reset.entry["media_inputs"] == []


def test_json_proxy_image_tests_reject_preflight_and_manual_is_unverified():
    with pytest.raises(ValueError, match="Image testing isn't available with System One JSON"):
        connect.plan("x", "x", "systemone", ENDPOINT, "", test_media_inputs=["image"])
    plan = connect.plan("x", "x", "systemone", ENDPOINT, "", media_inputs=["image"])
    assert (
        "System One JSON blocks image requests before sending"
        in connect.entry_warnings(plan.entry)[0]
    )


@pytest.mark.asyncio
async def test_invalid_schema_usage_stops_later_cases(monkeypatch):
    seen = []

    def handle(request):
        seen.append(request)
        composite = b'"score"' in request.content
        return fixture_response(
            request,
            corrupt="distribution" if composite else None,
            usage=100000 if composite else 10,
        )

    mock_http(monkeypatch, handle)
    result = await connect.run(proposal(budget_tokens=2048), approved="all")
    assert len(seen) == 2
    records = result.entry["provenance"]["probes"]
    assert records["decision:typed"]["typed_result_validated"] is False
    assert records["decision:typed"]["reported_tokens"] == 100000
    assert records["decision:typed"]["tokens_charged_to_budget"] == 100000
    assert records["decision:positive"]["reason"] == "budget exhausted"
    assert result.entry["provenance"]["tokens_charged_to_budget"] == 100512


def test_routing_preview_matches_actual_schema_recipe():
    from nooa.unifiedllm.connect._decision import planned_body

    plan = proposal()
    assert plan.probes[0].body == {"model": "fixture", **planned_body("routing")}


@pytest.mark.asyncio
async def test_fresh_decision_stage_requires_fixture_plan(monkeypatch):
    mock_http(monkeypatch, lambda request: pytest.fail("No incomplete stage inference"))
    with pytest.raises(ValueError, match="fresh typed decision plan"):
        await connect.check_stage(proposal(decision_checks=False), "decision")


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "transport,optin,blocked",
    [
        ("json", False, True),
        ("typesafe_multipart", False, True),
        ("typesafe_multipart", True, False),
    ],
)
def test_image_transport_warning_is_actionable_and_display_only(enabled, transport, optin, blocked):
    from copy import deepcopy

    entry = connect.plan(
        "x",
        "x",
        "systemone",
        ENDPOINT,
        "",
        media_inputs=["image"] if enabled else [],
        decision_image_transport=transport,
        typesafe_danger_zone_images=optin,
    ).entry
    before = deepcopy(entry)
    warnings = connect.entry_warnings(entry)
    if enabled and blocked:
        assert len(warnings) == 1
        assert "Images are enabled" in warnings[0] and "before sending" in warnings[0]
        assert (
            "Use OpenAI Decisions or an approved direct TypeSafe multipart connection, or turn images off."
            in warnings[0]
        )
        if transport == "json":
            assert "System One JSON blocks image requests" in warnings[0]
        else:
            assert "TypeSafe multipart requires explicit experimental image opt-in" in warnings[0]
            assert "System One JSON blocks" not in warnings[0]
    else:
        assert warnings == []
    assert entry == before
