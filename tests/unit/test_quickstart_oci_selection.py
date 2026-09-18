# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quickstart model selector: the OCI Generative AI branch."""

import importlib
import os
from unittest.mock import patch

import pytest


def _reload_quickstart(env: dict[str, str]):
    """Re-import ``nooa.util.quickstart`` under a controlled environment.

    ``get_llm_client`` is patched so no client is constructed; the mock records
    the model string and provider kwargs the selector chose.
    """
    with (
        patch.dict(os.environ, env, clear=True),
        patch("dotenv.load_dotenv"),
        patch("nooa.unifiedllm.registry.get_llm_client") as get_llm_client,
    ):
        import nooa.util.quickstart as quickstart

        importlib.reload(quickstart)
        return quickstart.MODEL, get_llm_client


@pytest.fixture(autouse=True)
def _restore_quickstart():
    yield
    _reload_quickstart({})


def test_oci_compartment_selects_oci_generative_ai() -> None:
    model, get_llm_client = _reload_quickstart(
        {"OCI_COMPARTMENT_ID": "ocid1.compartment.oc1..example", "OCI_REGION": "us-chicago-1"}
    )

    assert model == "oci/meta.llama-3.3-70b-instruct"
    get_llm_client.assert_called_with(
        "oci/meta.llama-3.3-70b-instruct",
        oci_compartment_id="ocid1.compartment.oc1..example",
        oci_region="us-chicago-1",
    )


def test_oci_model_and_dedicated_endpoint_overrides() -> None:
    model, get_llm_client = _reload_quickstart(
        {
            "OCI_COMPARTMENT_ID": "ocid1.compartment.oc1..example",
            "OCI_MODEL": "oci/my-imported-nemotron",
            "OCI_ENDPOINT_ID": "ocid1.generativeaiendpoint.oc1..example",
        }
    )

    assert model == "oci/my-imported-nemotron"
    get_llm_client.assert_called_with(
        "oci/my-imported-nemotron",
        oci_compartment_id="ocid1.compartment.oc1..example",
        oci_serving_mode="DEDICATED",
        oci_endpoint_id="ocid1.generativeaiendpoint.oc1..example",
    )


def test_nvidia_key_takes_precedence_over_oci() -> None:
    model, _ = _reload_quickstart(
        {"NVIDIA_API_KEY": "nvapi-example", "OCI_COMPARTMENT_ID": "ocid1.compartment.oc1..example"}
    )

    assert model.startswith("nvidia_nim/")


def test_oci_takes_precedence_over_openai_key() -> None:
    model, _ = _reload_quickstart(
        {"OPENAI_API_KEY": "sk-example", "OCI_COMPARTMENT_ID": "ocid1.compartment.oc1..example"}
    )

    assert model.startswith("oci/")


def test_oci_cli_profile_builds_a_signer_without_credential_variables() -> None:
    import sys
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    fake_oci = MagicMock()
    fake_oci.config.from_file.return_value = {
        "security_token_file": "/tmp/nooa-test-token",
        "key_file": "/tmp/nooa-test-key.pem",
    }
    fake_oci.auth.signers.SecurityTokenSigner.return_value = SimpleNamespace(kind="session-token")

    with (
        patch.dict(sys.modules, {"oci": fake_oci}),
        patch("builtins.open", create=True) as open_mock,
    ):
        open_mock.return_value.__enter__.return_value.read.return_value = "token-value\n"
        model, get_llm_client = _reload_quickstart(
            {
                "OCI_COMPARTMENT_ID": "ocid1.compartment.oc1..example",
                "OCI_CLI_PROFILE": "DEFAULT",
            }
        )

    assert model == "oci/meta.llama-3.3-70b-instruct"
    fake_oci.config.from_file.assert_called_once_with(profile_name="DEFAULT")
    kwargs = get_llm_client.call_args.kwargs
    assert kwargs["oci_compartment_id"] == "ocid1.compartment.oc1..example"
    assert kwargs["oci_signer"].kind == "session-token"


def test_oci_cli_profile_without_sdk_raises_a_clear_error() -> None:
    import sys

    with (
        patch.dict(sys.modules, {"oci": None}),
        pytest.raises(ModuleNotFoundError, match="uv pip install oci"),
    ):
        _reload_quickstart(
            {"OCI_COMPARTMENT_ID": "ocid1.compartment.oc1..example", "OCI_CLI_PROFILE": "DEFAULT"}
        )
