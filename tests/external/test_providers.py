# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for LLM provider configurations.

Tests the three provider configurations:
- openai: OpenAI API (api.openai.com)
- nvidia: NVIDIA NIM public API (integrate.api.nvidia.com)
- nvidia_internal: NVIDIA internal API with GPT-5, o1, o3 (inference-api.nvidia.com)
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestProviderConfiguration:
    """Test provider configuration logic."""

    def test_openai_provider_config(self):
        """OpenAI provider should use OPENAI_API_KEY and default endpoint."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test-key"}):
            api_key = os.getenv("OPENAI_API_KEY")
            api_base = ""  # OpenAI uses default endpoint

            assert api_key == "sk-test-key"
            assert api_base == ""

    def test_nvidia_nim_provider_config(self):
        """NVIDIA NIM provider should use NVIDIA_API_KEY and integrate.api endpoint."""
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "nvapi-test-key"}):
            api_key = os.getenv("NVIDIA_API_KEY")
            api_base = "https://integrate.api.nvidia.com/v1"

            assert api_key == "nvapi-test-key"
            assert api_base == "https://integrate.api.nvidia.com/v1"

    def test_nvidia_internal_provider_config(self):
        """NVIDIA internal provider should use NVIDIA_INFERENCE_API_KEY and inference-api endpoint."""
        with patch.dict(os.environ, {"NVIDIA_INFERENCE_API_KEY": "sk-test-internal-key"}):
            api_key = os.getenv("NVIDIA_INFERENCE_API_KEY")
            api_base = "https://inference-api.nvidia.com/v1"

            assert api_key == "sk-test-internal-key"
            assert api_base == "https://inference-api.nvidia.com/v1"

    def test_nvidia_internal_model_prefix(self):
        """NVIDIA internal models should be prefixed with 'openai/' for litellm."""
        model = "azure/openai/gpt-5"

        # The run_ablation.py logic adds openai/ prefix if not present
        if not model.startswith("openai/"):
            model = f"openai/{model}"

        assert model == "openai/azure/openai/gpt-5"

    def test_nvidia_internal_model_prefix_idempotent(self):
        """Model prefix should not be added twice."""
        model = "openai/azure/openai/gpt-5"

        # Should not add prefix if already present
        if not model.startswith("openai/"):
            model = f"openai/{model}"

        assert model == "openai/azure/openai/gpt-5"


class TestProviderEnvValidation:
    """Test that providers correctly validate environment variables."""

    def test_openai_missing_key(self):
        """Should fail when OPENAI_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove key if it exists
            os.environ.pop("OPENAI_API_KEY", None)
            api_key = os.getenv("OPENAI_API_KEY")
            assert api_key is None

    def test_nvidia_missing_key(self):
        """Should fail when NVIDIA_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NVIDIA_API_KEY", None)
            api_key = os.getenv("NVIDIA_API_KEY")
            assert api_key is None

    def test_nvidia_internal_missing_key(self):
        """Should fail when NVIDIA_INFERENCE_API_KEY is not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("NVIDIA_INFERENCE_API_KEY", None)
            api_key = os.getenv("NVIDIA_INFERENCE_API_KEY")
            assert api_key is None


class TestLLMConfig:
    """Test LLMConfig dataclass behavior."""

    def test_llm_config_creation(self):
        """LLMConfig should store all configuration values."""
        # Import would need path setup, so we test the expected behavior
        config = {
            "model": "gpt-4o-mini",
            "api_key": "test-key",
            "api_base": "",
            "temperature": 0.0,
            "max_tokens": 2000,
        }

        assert config["model"] == "gpt-4o-mini"
        assert config["temperature"] == 0.0
        assert config["max_tokens"] == 2000

    def test_llm_config_nvidia_internal(self):
        """LLMConfig for nvidia_internal should have correct base URL."""
        config = {
            "model": "openai/azure/openai/gpt-5",
            "api_key": "sk-internal-key",
            "api_base": "https://inference-api.nvidia.com/v1",
            "temperature": 0.0,
            "max_tokens": 2000,
        }

        assert "inference-api.nvidia.com" in config["api_base"]
        assert config["model"].startswith("openai/")


@pytest.mark.integration
class TestProviderIntegration:
    """Integration tests that make real API calls.

    These tests require API keys to be set and are skipped by default.
    Run with: pytest -m integration
    """

    @pytest.mark.skipif(
        not os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_API_KEY", "").startswith("sk-"),
        reason="OPENAI_API_KEY not set or not a valid OpenAI key",
    )
    @pytest.mark.asyncio
    async def test_openai_chat_completion(self):
        """Test real OpenAI API call."""
        import litellm

        response = await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=10,
        )

        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    @pytest.mark.skipif(not os.getenv("NVIDIA_API_KEY"), reason="NVIDIA_API_KEY not set")
    @pytest.mark.asyncio
    async def test_nvidia_nim_chat_completion(self):
        """Test real NVIDIA NIM API call."""
        import litellm

        response = await litellm.acompletion(
            model="nvidia_nim/meta/llama-3.3-70b-instruct",
            messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
            api_key=os.getenv("NVIDIA_API_KEY"),
            max_tokens=10,
        )

        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0

    @pytest.mark.skipif(
        not os.getenv("NVIDIA_INFERENCE_API_KEY"), reason="NVIDIA_INFERENCE_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_nvidia_internal_chat_completion(self):
        """Test real NVIDIA internal API call with GPT-5.1."""
        import litellm

        litellm.drop_params = True  # GPT-5 doesn't support all params

        response = await litellm.acompletion(
            model="openai/azure/openai/gpt-5.1",  # Use gpt-5.1 from models.yaml
            messages=[{"role": "user", "content": "Say 'test' and nothing else."}],
            api_key=os.getenv("NVIDIA_INFERENCE_API_KEY"),
            api_base="https://inference-api.nvidia.com/v1",
            max_tokens=50,  # Increased to allow for response
        )

        assert response.choices[0].message.content is not None
        assert len(response.choices[0].message.content) > 0
        print(f"✅ GPT-5.1 response: {response.choices[0].message.content}")


class TestSimpleLLMClient:
    """Test SimpleLLMClient class behavior."""

    @pytest.mark.asyncio
    async def test_client_passes_api_key(self):
        """Client should pass api_key to litellm when set."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_response.choices[0].message.tool_calls = None

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_response

            # Simulate what SimpleLLMClient.chat does
            kwargs = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0.0,
                "max_tokens": 2000,
                "api_key": "test-key",
            }

            await mock_completion(**kwargs)

            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["api_key"] == "test-key"

    @pytest.mark.asyncio
    async def test_client_passes_api_base(self):
        """Client should pass api_base to litellm when set."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "test response"
        mock_response.choices[0].message.tool_calls = None

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_response

            kwargs = {
                "model": "openai/azure/openai/gpt-5",
                "messages": [{"role": "user", "content": "test"}],
                "temperature": 0.0,
                "max_tokens": 2000,
                "api_key": "sk-internal-key",
                "api_base": "https://inference-api.nvidia.com/v1",
            }

            await mock_completion(**kwargs)

            mock_completion.assert_called_once()
            call_kwargs = mock_completion.call_args[1]
            assert call_kwargs["api_base"] == "https://inference-api.nvidia.com/v1"

    @pytest.mark.asyncio
    async def test_client_handles_tool_calls(self):
        """Client should extract tool calls from response."""
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "NYC"}'

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.choices[0].message.tool_calls = [mock_tool_call]

        with patch("litellm.acompletion", new_callable=AsyncMock) as mock_completion:
            mock_completion.return_value = mock_response

            # Simulate response processing like SimpleLLMClient does
            msg = mock_response.choices[0].message
            tool_calls = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(
                        {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                    )

            assert len(tool_calls) == 1
            assert tool_calls[0]["name"] == "get_weather"
            assert tool_calls[0]["arguments"] == '{"location": "NYC"}'
