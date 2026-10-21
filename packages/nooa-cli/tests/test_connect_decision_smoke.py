# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CLI stages and wizard use native typed decision/image requests."""

import json

import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands.connect import command

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.connect_http import mock_http
from tests.unifiedllm.connect.test_connect_decision_smoke import ENDPOINT, fixture_response


def options(style="openai"):
    return [
        "fixture",
        "--as",
        "fixture",
        "--endpoint",
        ENDPOINT,
        "--api-style",
        style,
        "--api-key-env",
        "",
    ]


@pytest.mark.parametrize(
    "stage,count,image",
    [
        ("plan", 0, True),
        ("interfaces", 1, False),
        ("routing", 1, False),
        ("decision", 4, False),
        ("media", 5, True),
        ("all", 4, False),
        ("all", 5, True),
    ],
)
@pytest.mark.parametrize("style", ["openai", "systemone"])
def test_native_stage_semantics(monkeypatch, stage, count, image, style):
    seen = []

    def handle(request):
        seen.append(request)
        return fixture_response(request)

    mock_http(monkeypatch, handle)
    flags = ["--test-media-inputs", "image"] if image else []
    if style == "systemone" and image:
        flags += [
            "--decision-image-transport",
            "typesafe_multipart",
            "--typesafe-danger-zone-images",
        ]
    result = CliRunner().invoke(command, [*options(style), "--stage", stage, *flags])
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert report["ok"] is True
    assert len(seen) == count
    assert "base64" not in result.stdout and "data:image" not in result.stdout
    if stage in {"media", "all"} and image:
        assert report["data"]["entry"]["media_inputs"] == ["image"]
        assert report["checks"]["media:image"]["attachment_observed"] is True
        assert report["checks"]["media:image"]["typed_result"]["color"] == "red"


@pytest.mark.parametrize("style", ["openai", "systemone"])
def test_wizard_native_images_save_transport_reload_and_recheck(monkeypatch, tmp_path, style):
    seen = []

    def handle(request):
        seen.append(request)
        return fixture_response(request)

    mock_http(monkeypatch, handle)
    path = tmp_path / "models.yaml"
    flags = ["--test-media-inputs", "image"]
    if style == "systemone":
        flags += [
            "--decision-image-transport",
            "typesafe_multipart",
            "--typesafe-danger-zone-images",
        ]
    result = CliRunner().invoke(
        command, [*options(style), *flags, "--no-catalogue", "--yes", "--output", str(path)]
    )
    assert result.exit_code == 0, result.output
    assert len(seen) == 5
    assert "known color confirmed" in result.output
    entry = yaml.safe_load(path.read_text())["models"]["fixture"]
    assert entry["media_inputs"] == ["image"]
    assert entry["decision_image_transport"] == (
        "typesafe_multipart" if style == "systemone" else "json"
    )
    assert entry["typesafe_danger_zone_images"] is (style == "systemone")
    assert "provenance" not in entry and "base64" not in path.read_text()
    # Reload through public configuration and plan, then rerun the real wire checks.
    plan = connect.plan(
        "fixture",
        "fixture",
        style,
        ENDPOINT,
        "",
        existing_entry=connect.configure_entry(entry),
        test_media_inputs=["image"],
    )
    import asyncio

    checked = asyncio.run(connect.check_stage(plan, "media"))
    assert checked.entry["provenance"]["probes"]["media:image"]["outcome"] == "passed"
    assert len(seen) == 10


def test_save_stage_keeps_transport_not_evidence(monkeypatch, tmp_path):
    mock_http(monkeypatch, lambda request: pytest.fail("Save is offline"))
    entry = connect.plan(
        "fixture",
        "fixture",
        "systemone",
        ENDPOINT,
        "",
        media_inputs=["image"],
        decision_image_transport="typesafe_multipart",
        typesafe_danger_zone_images=True,
    ).entry
    entry["provenance"]["probes"]["media:image"] = {
        "outcome": "passed",
        "request": {"private": "data:image/png;base64,PRIVATE"},
    }
    source, target = tmp_path / "report.json", tmp_path / "models.yaml"
    source.write_text(json.dumps({"data": {"alias": "fixture", "entry": entry}}))
    result = CliRunner().invoke(
        command, ["--stage", "save", "--input", str(source), "--output", str(target)]
    )
    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(target.read_text())["models"]["fixture"]
    assert saved["typesafe_danger_zone_images"] is True
    assert saved["media_inputs"] == ["image"]
    assert "PRIVATE" not in target.read_text()
    assert "PRIVATE" not in result.stdout


@pytest.mark.parametrize("stage", ["tools", "reasoning", "session"])
def test_chat_only_stages_reject_native_decisions_preflight(monkeypatch, stage):
    mock_http(monkeypatch, lambda request: pytest.fail("No request for chat-only stage"))
    result = CliRunner().invoke(command, [*options(), "--stage", stage])
    assert result.exit_code != 0
    assert json.loads(result.stdout)["ok"] is False


def test_systemone_json_proxy_image_test_preflight(monkeypatch):
    mock_http(monkeypatch, lambda request: pytest.fail("JSON proxy media must not be sent"))
    result = CliRunner().invoke(
        command, [*options("systemone"), "--stage", "media", "--test-media-inputs", "image"]
    )
    assert result.exit_code != 0
    assert "Unverified System One JSON image transport" in result.stdout


@pytest.mark.parametrize("key", [None, "same-origin-key"])
def test_catalogue_origin_uses_effective_default_port(key):
    from nooa_cli.commands._connect_stages import listing_key

    assert listing_key("https://fixture.test:443/models", ENDPOINT, key) == key


def test_construction_notices_do_not_pollute_stage_json(monkeypatch):
    # Agent construction is synchronous; its optional tracing announcement must
    # not escape the non-interactive JSON frontend or expose exporter URLs.
    monkeypatch.setattr(
        "nooa.agent._try_auto_enable_tracing", lambda: print("PRIVATE exporter URL")
    )
    mock_http(monkeypatch, fixture_response)
    result = CliRunner().invoke(command, [*options(), "--stage", "routing"])
    assert result.exit_code == 0
    assert "PRIVATE" not in result.stdout
    assert json.loads(result.stdout)["ok"] is True


@pytest.mark.parametrize(
    "style,flags",
    [
        ("openai", ["--media-inputs", "audio"]),
        ("systemone", ["--media-inputs", "audio"]),
        ("systemone", ["--test-media-inputs", "image"]),
        (
            "systemone",
            ["--test-media-inputs", "image", "--decision-image-transport", "typesafe_multipart"],
        ),
    ],
)
def test_local_native_plan_errors_are_usage_errors(monkeypatch, style, flags):
    mock_http(monkeypatch, lambda request: pytest.fail("Invalid plans must not send HTTP"))
    result = CliRunner().invoke(command, [*options(style), "--stage", "plan", *flags])
    assert result.exit_code == 2, result.output
    report = json.loads(result.stdout)
    assert report["ok"] is False
    assert report["error"]["type"] == "UsageError"


def test_runtime_value_error_is_not_converted_to_usage_error(monkeypatch):
    async def fail(*args, **kwargs):
        raise ValueError("Runtime response failure")

    monkeypatch.setattr(connect, "check_stage", fail)
    mock_http(monkeypatch, lambda request: pytest.fail("Stubbed runtime stays offline"))
    result = CliRunner().invoke(command, [*options(), "--stage", "routing"])
    assert result.exit_code == 1, result.output
    assert json.loads(result.stdout)["error"]["type"] == "ValueError"
