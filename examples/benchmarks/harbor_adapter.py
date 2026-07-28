# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harbor adapter for the NOOA BenchAgent.

Registers NOOA as a Harbor "installed agent" for the open-source Harbor
benchmark harness (https://github.com/harbor-framework/harbor). Harbor calls
:meth:`install` once inside each trial container (clone the repo, create a
venv, install the ``nooa-bench`` package), then :meth:`run` invokes the
``nemo-harbor`` CLI; per-task token usage is read back from ``result.json``.

Usage — reference this file via ``import_path`` in your Harbor config
(see ``harbor_minimal.yaml`` next to this file)::

    PYTHONPATH=examples/benchmarks harbor run --config examples/benchmarks/harbor_minimal.yaml

Credentials: set ``NVIDIA_INFERENCE_API_KEY`` (inference.nvidia.com gateway)
or ``NVIDIA_API_KEY`` (public NIM endpoint) on the host — it is forwarded
into the agent process. The NVIDIA endpoints are OpenAI-compatible, so the
key is also exposed as ``OPENAI_API_KEY`` for litellm when you point
``api_base`` at them. The Copilot SDK path can use a forwarded
``COPILOT_GITHUB_TOKEN``; direct local runs normally reuse the signed-in
Copilot CLI account.

Note: the container needs network access during :meth:`install` (git clone +
dependency download). For air-gapped task containers, pre-stage the repo and
wheels yourself.
"""

from __future__ import annotations

import json
import os
import shlex

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

REPO_DIR = "/installed-agent/nooa"
VENV = "/opt/nooa-venv"
COPILOT_RUNTIME_DIR = "/opt/nooa-copilot-runtime"
COPILOT_HOME = "/logs/artifacts/copilot-home"

# Credentials forwarded from the host into the agent process. The NVIDIA
# inference endpoints are public; add other provider keys here if needed.
FORWARDED_ENV_VARS = (
    "NVIDIA_INFERENCE_API_KEY",  # inference.nvidia.com gateway
    "NVIDIA_API_KEY",  # public NIM endpoint
    "COPILOT_GITHUB_TOKEN",  # optional Copilot SDK token auth
)


class NooaBenchAgent(BaseInstalledAgent):
    """Runs the NOOA BenchAgent (``packages/nooa-bench``) under Harbor.

    YAML kwargs:
        git_url:    clone URL of the NOOA repo (or env ``NOOA_GIT_URL``)
        git_ref:    optional branch/tag/SHA (or env ``NOOA_GIT_REF``)
        agent_type: registry key passed to ``nemo-harbor`` (default ``bench``)
        api_base:   optional OpenAI-compatible endpoint override
        reasoning_effort: optional Copilot SDK reasoning effort
        context_tier: optional Copilot SDK context tier
        timeout_seconds: optional Copilot SDK session timeout
    """

    def __init__(
        self,
        logs_dir,
        git_url: str | None = None,
        git_ref: str | None = None,
        agent_type: str = "bench",
        api_base: str | None = None,
        reasoning_effort: str | None = None,
        context_tier: str | None = None,
        timeout_seconds: float | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(logs_dir, *args, **kwargs)
        self._git_url = git_url or os.environ.get("NOOA_GIT_URL")
        if not self._git_url:
            raise ValueError(
                "NOOA repo URL required: pass the git_url kwarg in the Harbor "
                "YAML or set the NOOA_GIT_URL environment variable."
            )
        self._git_ref = git_ref or os.environ.get("NOOA_GIT_REF")
        self._agent_type = agent_type
        self._api_base = api_base
        self._reasoning_effort = reasoning_effort
        self._context_tier = context_tier
        self._timeout_seconds = timeout_seconds

    @staticmethod
    def name() -> str:
        return "nooa-bench"

    async def install(self, environment: BaseEnvironment) -> None:
        """Clone the repo and install core + cli + bench into a fresh venv."""
        branch = f"--branch {shlex.quote(self._git_ref)} " if self._git_ref else ""
        copilot_runtime_install = (
            f"mkdir -p {COPILOT_RUNTIME_DIR} && chmod -R a+rwx {COPILOT_RUNTIME_DIR} && "
            f"COPILOT_CLI_EXTRACT_DIR={COPILOT_RUNTIME_DIR} "
            f"{VENV}/bin/python3 -m copilot download-runtime && "
            f"chmod -R a+rwx {COPILOT_RUNTIME_DIR} && "
            if self._agent_type == "copilot"
            else ""
        )
        await self.exec_as_root(
            environment,
            command=(
                # git may be missing from slim task images; best-effort install.
                "(which git > /dev/null 2>&1 || "
                "(apt-get update && apt-get install -y git curl)) && "
                f"git clone --depth 1 {branch}{shlex.quote(self._git_url)} {REPO_DIR} && "
                # uv gives us a venv + resolver without assuming pip exists.
                "(which uv > /dev/null 2>&1 || "
                "(curl -LsSf https://astral.sh/uv/install.sh | sh)) && "
                'export PATH="$HOME/.local/bin:$PATH" && '
                f"uv venv {VENV} && "
                f"uv pip install --python {VENV}/bin/python3 "
                f"{REPO_DIR} {REPO_DIR}/packages/nooa-cli {REPO_DIR}/packages/nooa-bench && "
                f"{copilot_runtime_install}"
                f"{VENV}/bin/python3 -c \"from nooa_bench.runner import main; print('nooa-bench installed OK')\""
            ),
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        api_base = f"--api-base {shlex.quote(self._api_base)} " if self._api_base else ""
        reasoning_effort = (
            f"--reasoning-effort {shlex.quote(self._reasoning_effort)} "
            if self._reasoning_effort
            else ""
        )
        context_tier = (
            f"--context-tier {shlex.quote(self._context_tier)} " if self._context_tier else ""
        )
        timeout_seconds = (
            f"--timeout-seconds {self._timeout_seconds:g} "
            if self._timeout_seconds is not None
            else ""
        )
        # Exit code 1 means the agent reported task failure: map it to exit 0 so
        # the verifier scores reward=0 instead of raising a harness error.
        prefix = f"mkdir -p {shlex.quote(COPILOT_HOME)} && " if self._agent_type == "copilot" else ""
        command = (
            f"({prefix}{VENV}/bin/nemo-harbor "
            f"--instruction {shlex.quote(instruction)} "
            f"--model {shlex.quote(self.model_name or '')} "
            f"--agent-type {shlex.quote(self._agent_type)} "
            f"{api_base}"
            f"{reasoning_effort}"
            f"{context_tier}"
            f"{timeout_seconds}"
            f"; EC=$?; [ $EC -eq 1 ] && exit 0 || exit $EC) "
            f"2>&1 | tee /logs/agent/nooa_bench.log"
        )
        env = {k: v for k in FORWARDED_ENV_VARS if (v := os.environ.get(k))}
        # The NVIDIA endpoints are OpenAI-compatible; litellm reads the key
        # from OPENAI_API_KEY when the model is routed via api_base.
        nvidia_key = env.get("NVIDIA_INFERENCE_API_KEY") or env.get("NVIDIA_API_KEY")
        if nvidia_key:
            env.setdefault("OPENAI_API_KEY", nvidia_key)
        if self._agent_type == "copilot":
            env.setdefault("COPILOT_CLI_EXTRACT_DIR", COPILOT_RUNTIME_DIR)
            env.setdefault("COPILOT_HOME", COPILOT_HOME)
        await self.exec_as_agent(environment, command=command, env=env, cwd=REPO_DIR)

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Surface per-task token usage from the runner's result.json."""
        result_path = self.logs_dir / "result.json"
        if not result_path.exists():
            return
        try:
            data = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            return
        context.n_input_tokens = data.get("n_input_tokens")
        context.n_output_tokens = data.get("n_output_tokens")
