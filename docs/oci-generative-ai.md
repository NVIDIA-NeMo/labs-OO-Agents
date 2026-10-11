# OCI Generative AI

NOOA uses LiteLLM model strings through `get_llm_client()`, and LiteLLM routes
`oci/<model>` to [Oracle Cloud Infrastructure (OCI) Generative AI](https://docs.oracle.com/en-us/iaas/Content/generative-ai/home.htm).
Any chat or text-generation model in the OCI Generative AI catalog, and any such
model you import into a dedicated endpoint, can drive a NOOA agent without
changing agent code. NOOA calls LiteLLM's completion interface, so the catalog's
embedding models are not used here.

```python
from nooa.unifiedllm.registry import get_llm_client

llm = get_llm_client(
    "oci/meta.llama-3.3-70b-instruct",
    oci_region="us-chicago-1",
    oci_compartment_id="ocid1.compartment.oc1..example",
)
```

Requests go to `https://inference.generativeai.<region>.oci.oraclecloud.com`.
The region defaults to `us-ashburn-1`; set it to a region where your tenancy is
subscribed to the service. `oci_compartment_id` is required.

## Authentication

LiteLLM signs OCI requests itself. Two options:

**API-key credentials.** Set the values from your OCI user's API key as
parameters or as environment variables, which LiteLLM reads automatically:

```bash
export OCI_REGION=us-chicago-1
export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..example
export OCI_USER=ocid1.user.oc1..example
export OCI_TENANCY=ocid1.tenancy.oc1..example
export OCI_FINGERPRINT=aa:bb:cc:...
export OCI_KEY_FILE=~/.oci/oci_api_key.pem   # or OCI_KEY with the PEM contents
```

**An OCI SDK signer.** If you already use the `oci` CLI, reuse a profile from
`~/.oci/config`, including session-token profiles created by
`oci session authenticate`, by passing a signer object:

```python
import os

import oci

from nooa.unifiedllm.registry import get_llm_client

config = oci.config.from_file(profile_name="DEFAULT")
if "security_token_file" in config:
    with open(os.path.expanduser(config["security_token_file"])) as f:
        token = f.read().strip()
    private_key = oci.signer.load_private_key_from_file(
        config["key_file"], pass_phrase=config.get("pass_phrase")
    )
    signer = oci.auth.signers.SecurityTokenSigner(token, private_key)
else:
    signer = oci.signer.Signer(
        tenancy=config["tenancy"],
        user=config["user"],
        fingerprint=config["fingerprint"],
        private_key_file_location=config["key_file"],
        pass_phrase=config.get("pass_phrase"),
    )

llm = get_llm_client(
    "oci/meta.llama-3.3-70b-instruct",
    oci_signer=signer,
    oci_region=config.get("region", "us-ashburn-1"),
    oci_compartment_id=os.environ["OCI_COMPARTMENT_ID"],
)
```

The `oci` SDK is not a NOOA dependency; install it with `uv pip install oci`.
The quickstart selector implements both options behind `OCI_CLI_PROFILE` and the
`OCI_*` variables; see [Quickstart selector](#quickstart-selector).

## Choose a model

Use the catalog model id after `oci/`. The default below is the one exercised
with NOOA's Predict and CodeAct strategies; the others are catalog examples that
LiteLLM routes the same way:

| Model string | Notes |
| --- | --- |
| `oci/meta.llama-3.3-70b-instruct` | Default; tested with NOOA quickstarts 01, 02, 03, and 16 |
| `oci/meta.llama-4-maverick-17b-128e-instruct-fp8` | Multimodal |
| `oci/xai.grok-4` | Reasoning model |
| `oci/google.gemini-2.5-pro` | Reasoning model, multimodal |
| `oci/openai.gpt-oss-120b` | Open-weights MoE |
| `oci/cohere.command-a-03-2025` | Cohere family |

Availability differs by region. List the catalog for your compartment with
`oci generative-ai model-collection list-models --compartment-id <ocid> --region <region>`.

## NVIDIA Nemotron on OCI

The Generative AI catalog does not include Nemotron models by default. Two ways to run them:

**Imported model on a dedicated endpoint.** OCI Generative AI can
[import open-weights models](https://docs.oracle.com/en-us/iaas/Content/generative-ai/imported-models.htm),
including NVIDIA Nemotron 3 and Nemotron 3.5 Lightning, onto a dedicated AI
cluster behind an endpoint. Point LiteLLM at the endpoint:

```python
llm = get_llm_client(
    "oci/meta.llama-3.3-70b-instruct",   # vendor prefix selects the request format; see below
    oci_serving_mode="DEDICATED",
    oci_endpoint_id="ocid1.generativeaiendpoint.oc1..example",
    oci_region="us-chicago-1",
    oci_compartment_id="ocid1.compartment.oc1..example",
)
```

With `oci_serving_mode="DEDICATED"` and an explicit `oci_endpoint_id`, the
endpoint decides which weights serve the request, but LiteLLM still uses the
model string's vendor prefix to choose the OCI request format and parameter
mapping: `cohere.*` selects the Cohere format, anything else the generic format.
Nemotron and other Llama-style imports use the generic format, so pass a
`meta.*` identifier such as the one above; for an imported Cohere model pass a
`cohere.*` identifier.

**Self-hosted on OKE.** Serve Nemotron with vLLM on Oracle Container Engine for
Kubernetes and use LiteLLM's `hosted_vllm/` route, exactly as in
[Local models](local-models.md):

```python
llm = get_llm_client(
    "hosted_vllm/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
    api_base="http://127.0.0.1:8000/v1",   # e.g. a kubectl port-forward to the vLLM router
)
```

A reference deployment of Nemotron 3.5 Lightning on an OKE A10 node pool is in
[NVIDIA/nvidia-oci-samples](https://github.com/NVIDIA/nvidia-oci-samples).

## Quickstart selector

The quickstart examples pick a provider from your environment. When
`OCI_COMPARTMENT_ID` is set and `NVIDIA_API_KEY` is not, they use OCI Generative AI:

| Variable | Effect |
| --- | --- |
| `OCI_COMPARTMENT_ID` | Selects OCI; passed as `oci_compartment_id` |
| `OCI_REGION` | Passed as `oci_region` (default `us-ashburn-1`) |
| `OCI_MODEL` | Model string, default `oci/meta.llama-3.3-70b-instruct` |
| `OCI_ENDPOINT_ID` | Adds `oci_serving_mode="DEDICATED"` and `oci_endpoint_id` |
| `OCI_CLI_PROFILE` | Signs requests with that `~/.oci/config` profile (API key or session token); needs `uv pip install oci` |
| `OCI_USER`, `OCI_TENANCY`, `OCI_FINGERPRINT`, `OCI_KEY_FILE` | API-key credentials read by LiteLLM when no profile is given |

With these variables set, and `NVIDIA_API_KEY` unset, every quickstart in
`examples/quickstart/` runs on OCI Generative AI unchanged. `NVIDIA_API_KEY`
takes precedence over OCI in the selector.

## Aliases

Put repeated configuration in `.nooa/llm_config.yaml`:

```yaml
models:
  oci-llama:
    model_name: oci/meta.llama-3.3-70b-instruct
    temperature: 0.0
```

Registry aliases forward `model_name`, `api_base`, `api_key_env`, and a fixed set
of generation parameters to LiteLLM. Provider settings such as the region and
compartment are not part of that set, so supply them through the `OCI_REGION` and
`OCI_COMPARTMENT_ID` environment variables or as call-site keyword arguments:

```python
llm = get_llm_client("oci-llama")                                   # OCI_* variables set
llm = get_llm_client("oci-llama", oci_compartment_id="ocid1....")   # or override here
```

## Troubleshooting

- `404` or `NotAuthorizedOrNotFound`: the model is not available in `oci_region`,
  or the compartment lacks a policy allowing `generative-ai-family` access.
- `oci_compartment_id is required`: set `OCI_COMPARTMENT_ID` or pass the parameter.
- Signature errors with a session-token profile: the token expires after about an
  hour; run `oci session authenticate` again and rebuild the signer.

References: [LiteLLM OCI provider](https://docs.litellm.ai/docs/providers/oci),
[OCI Generative AI pretrained models](https://docs.oracle.com/en-us/iaas/Content/generative-ai/pretrained-models.htm).
