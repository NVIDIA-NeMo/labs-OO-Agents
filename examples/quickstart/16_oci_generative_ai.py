# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Quickstart 16: Oracle Cloud Infrastructure (OCI) Generative AI as the model provider.

NOOA is model-agnostic through litellm, which routes ``oci/<model>`` to OCI
Generative AI. The quickstart model selector picks OCI when OCI_COMPARTMENT_ID is
set, so every quickstart in this directory runs on OCI with the same variables:

  export OCI_COMPARTMENT_ID=ocid1.compartment.oc1..example
  export OCI_REGION=us-chicago-1                 # a region with the service
  export OCI_CLI_PROFILE=DEFAULT                 # an ~/.oci/config profile (needs `uv pip install oci`)
  # or, instead of a profile: OCI_USER, OCI_TENANCY, OCI_FINGERPRINT, OCI_KEY_FILE
  uv run python examples/quickstart/16_oci_generative_ai.py

Optional: OCI_MODEL (default oci/meta.llama-3.3-70b-instruct) and OCI_ENDPOINT_ID to
target a dedicated endpoint such as an imported NVIDIA Nemotron model.

The agent is a capacity planner whose deterministic helpers are its only source
of facts. See docs/oci-generative-ai.md for explicit client construction, the
model catalog, aliases, and Nemotron options.
"""

import sys

from nooa.util.quickstart import *

if not MODEL.startswith("oci/"):
    print(
        f"SKIP: the quickstart selector chose {MODEL!r}. Set OCI_COMPARTMENT_ID (and OCI credentials) to run on OCI."
    )
    sys.exit(0)

# GPU count and total GPU memory per OCI shape. The agent reads these through the
# helper methods instead of recalling specifications from training data.
GPU_SHAPES: dict[str, dict[str, float]] = {
    "VM.GPU.A10.1": {"gpus": 1, "gpu_memory_gb": 24},
    "VM.GPU.A10.2": {"gpus": 2, "gpu_memory_gb": 48},
    "BM.GPU.A10.4": {"gpus": 4, "gpu_memory_gb": 96},
    "BM.GPU.A100-v2.8": {"gpus": 8, "gpu_memory_gb": 640},
    "BM.GPU.H100.8": {"gpus": 8, "gpu_memory_gb": 640},
}


class Recommendation(BaseModel):
    shape: str = Field(description="The recommended OCI GPU shape name.")
    total_gpu_memory_gb: float = Field(description="Total GPU memory of that shape in GB.")
    rationale: str = Field(description="One sentence explaining why this is the smallest fit.")


class CapacityPlanner(Agent, llm=llm):
    """You plan GPU capacity on Oracle Cloud for serving open-weights models."""

    def gpu_shapes(self) -> dict[str, dict[str, float]]:
        """Return the available OCI GPU shapes with their GPU count and total GPU memory in GB."""
        return GPU_SHAPES

    def fits(self, weights_gb: float, shape: str, headroom: float = 1.25) -> bool:
        """Whether model weights, with headroom for the KV cache, fit a shape's total GPU memory."""
        return weights_gb * headroom <= GPU_SHAPES[shape]["gpu_memory_gb"]

    async def recommend(self, model_name: str, weights_gb: float) -> Recommendation:
        """Recommend the smallest shape whose GPU memory fits the model weights with headroom.

        Use self.gpu_shapes() and self.fits() rather than recalling shape specifications.
        """
        ...


@autorun
async def main():
    planner = CapacityPlanner()
    result = await planner.recommend(
        "NVIDIA Nemotron 3.5 Lightning 30B-A3B (NVFP4)", weights_gb=21.6
    )
    print(f"model: {MODEL}")
    print(result)
