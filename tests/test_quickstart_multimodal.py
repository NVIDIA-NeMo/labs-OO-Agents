# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline contracts for the runnable multimodal quickstart (no provider calls)."""

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from nooa.media import Image
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall


@pytest.fixture
def example(monkeypatch):
    # Even a fake Agent otherwise probes the local tracing endpoint.
    monkeypatch.setattr("nooa.agent._try_auto_enable_tracing", lambda: None)
    construct = Mock(side_effect=AssertionError("client constructed during import"))
    monkeypatch.setattr("nooa.unifiedllm.get_llm_client", construct)
    path = Path(__file__).resolve().parents[1] / "examples/quickstart/13_multimodal.py"
    spec = importlib.util.spec_from_file_location("quickstart_multimodal_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    construct.assert_not_called()
    return module


@pytest.mark.asyncio
async def test_both_strategies_receive_asset_images_and_close(example, monkeypatch, capsys):
    data = {"colors": ["black"], "text_content": ["Hello"], "description": "Text image"}
    client = FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="done",
                        name="return_result",
                        arguments=json.dumps({"result": "Colored shapes"}),
                    )
                ],
                finish_reason="tool_calls",
            ),
            LLMResponse(content=json.dumps(data), finish_reason="stop"),
        ],
        strict_exhaustion=True,
        media_inputs=["image"],
    )
    closed = []

    def track_constructor(cls):
        def construct(**kwargs):
            agent = cls(**kwargs)

            async def on_close():
                closed.append(type(agent).__name__)

            agent.event_manager.on_close(on_close)
            return agent

        return construct

    monkeypatch.setattr(example, "ImageDescriber", track_constructor(example.ImageDescriber))
    monkeypatch.setattr(example, "ImageAnalyzer", track_constructor(example.ImageAnalyzer))
    await example.run_example(client)
    assert closed == ["ImageDescriber", "ImageAnalyzer"]
    assert client.call_count == 2
    assert client.remaining_responses == 0
    for call, filename in zip(client.calls, ["test_image.png", "test_text.png"], strict=True):
        blocks = [
            block
            for message in call.messages
            if isinstance(message.get("content"), (tuple, list))
            for block in message["content"]
            if block.get("type") == "image_url"
        ]
        assert len(blocks) == 1
        assert blocks[0]["image_url"]["url"] == Image.from_file(example.ASSETS / filename).data_url
    assert client.calls[0].tools
    assert client.calls[1].output_model is not None
    output = capsys.readouterr().out
    assert "Colored shapes" in output
    assert '"text_content": [' in output
    assert "Hello" in output


@pytest.mark.asyncio
async def test_text_only_fails_before_inference_and_closes_client(example, monkeypatch):
    client = FakeLLMClient(strict_exhaustion=True)
    client.aclose = AsyncMock()
    monkeypatch.setattr(example, "get_llm_client", Mock(return_value=client))
    monkeypatch.setattr(
        example, "ImageDescriber", Mock(side_effect=AssertionError("agent created"))
    )
    with pytest.raises(ValueError, match=r"media_inputs: \[image\]"):
        await example.main()
    example.get_llm_client.assert_called_once_with("vision")
    assert client.call_count == 0
    client.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_path", ["describe", "analyze"])
async def test_errors_propagate_and_all_created_resources_close(example, monkeypatch, failed_path):
    client = FakeLLMClient(media_inputs=["image"])
    client.aclose = AsyncMock()
    error = RuntimeError("provider does not support images")
    describer = Mock()
    describer.describe = AsyncMock(return_value="Shapes")
    describer.aclose = AsyncMock()
    analyzer = Mock()
    analyzer.analyze = AsyncMock(
        return_value=example.ImageAnalysis(
            colors=[],
            text_content=[],
            description="Text",
        )
    )
    analyzer.aclose = AsyncMock()
    getattr(describer if failed_path == "describe" else analyzer, failed_path).side_effect = error
    monkeypatch.setattr(example, "ImageDescriber", Mock(return_value=describer))
    monkeypatch.setattr(example, "ImageAnalyzer", Mock(return_value=analyzer))
    monkeypatch.setattr(example, "get_llm_client", Mock(return_value=client))
    with pytest.raises(RuntimeError) as raised:
        await example.main()
    assert raised.value is error
    describer.aclose.assert_awaited_once()
    if failed_path == "analyze":
        analyzer.aclose.assert_awaited_once()
    else:
        example.ImageAnalyzer.assert_not_called()
    client.aclose.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_config", [False, True])
async def test_config_reload_is_explicit_and_client_closed(
    example, monkeypatch, tmp_path, explicit_config
):
    config = tmp_path / "models.yaml"
    config.write_text("models: {}\n")
    client = FakeLLMClient(media_inputs=["image"])
    client.aclose = AsyncMock()
    reload = Mock()
    run = AsyncMock()
    monkeypatch.setattr(example, "reload_registry", reload)
    monkeypatch.setattr(example, "get_llm_client", Mock(return_value=client))
    monkeypatch.setattr(example, "run_example", run)
    await example.main("custom-vision", config if explicit_config else None)
    if explicit_config:
        reload.assert_called_once_with(config)
    else:
        reload.assert_not_called()
    example.get_llm_client.assert_called_once_with("custom-vision")
    run.assert_awaited_once_with(client)
    client.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_explicit_config_fails_before_client_construction(example, tmp_path):
    with pytest.raises(FileNotFoundError, match="Config file does not exist"):
        await example.main(config=tmp_path / "absent.yaml")
    example.get_llm_client.assert_not_called()
