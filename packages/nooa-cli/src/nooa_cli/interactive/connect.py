# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-local model setup for native and ACP command adapters.

Connect owns discovery, plans, checks and registry serialization. This control
owns the user's unfinished setup and the explicit check/save actions. It never
changes the running agent's model or the process-wide registry.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

from nooa.unifiedllm import connect, resolve_api_key_from_config
from nooa_cli.interactive.controls import (
    BehaviorControl,
    ControlMessage,
    ControlResult,
    ControlTable,
)


class _Arguments(argparse.ArgumentParser):
    def error(self, message):
        # Unknown arguments may be accidentally pasted credentials. Do not echo.
        raise ValueError("Invalid /connect arguments. Use /connect help.")


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise ValueError("Token limits must be positive integers.")
    return number


class ConnectControl(BehaviorControl):
    """Configure models with NOOA Connect: discover, preview, check, then save."""

    name = "connect"

    def __init__(self, *args, registry_path: Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        root = Path(self.workspace or self.agent.cwd).resolve()
        self.registry_path = registry_path or root / ".nooa" / "llm_config.yaml"
        self.reset()

    def reset(self) -> None:
        """Discard the unfinished setup, including any masked host-supplied key."""
        self.connection: tuple[str, str, str] | None = None
        self.discovery: connect.Discovery | None = None
        self.proposal: connect.ConnectPlan | None = None
        self._api_key: str | None = None

    @classmethod
    def help_text(cls) -> dict[str, str]:
        return {
            "/connect [PROVIDER|URL] [--api-style STYLE] [--api-key-env NAME]": "Discover models",
            "/connect model ID [--as ALIAS] [--max-tokens N] [--context-window N]": "Preview settings",
            "/connect check minimal|all": "Approve bounded model calls (may incur charges)",
            "/connect save [--replace]": "Save the previewed alias to this workspace",
            "/connect cancel": "Discard unfinished setup",
        }

    def validate_args(self, args: list[str]) -> tuple[bool, str | None]:
        # Parsing is shared by both adapters in execute(), with no process exits.
        return True, None

    async def run(self, args: list[str]) -> ControlResult:
        try:
            return await self.execute(args)
        except ValueError as exc:
            # These are local validation / sanitized discovery errors. Mask the
            # temporary credential too, including a caller-supplied invalid URL.
            message = str(exc)
            if self._api_key:
                message = message.replace(self._api_key, "[redacted]")
            return ControlResult.err(message)
        except Exception as exc:
            # Provider and filesystem exceptions can contain credentials or
            # request bodies. The library's check records carry safe diagnostics.
            return ControlResult.err(
                f"Model setup failed ({type(exc).__name__}). Nothing was applied to the running agent."
            )

    async def execute(self, args: list[str]) -> ControlResult:
        if not args or args == ["help"]:
            if self.proposal is not None and not args:
                return self.preview()
            return ControlResult.ok(
                ControlMessage(
                    "\n".join(f"{key} — {value}" for key, value in self.help_text().items())
                ),
                ControlMessage("Providers: " + ", ".join(connect.PROVIDERS)),
                ControlMessage(
                    "Keys are read from environment variables; never paste a key into a slash command."
                ),
                ControlMessage(
                    "Model options also include --budget-tokens N, --reasoning-template effort|adaptive|budget|toggle|thinking, --levels low,high, and --reasoning-default LEVEL."
                ),
            )
        action, *rest = args
        if action == "cancel" and not rest:
            self.reset()
            return ControlResult.ok(ControlMessage("Model setup cancelled."))
        if action == "check":
            if rest not in (["minimal"], ["all"]):
                raise ValueError(
                    "Use /connect check minimal or /connect check all to approve model calls."
                )
            return await self.check(rest[0])
        if action == "save":
            if rest not in ([], ["--replace"]):
                raise ValueError("Usage: /connect save [--replace]")
            return self.save(replace_existing=bool(rest))
        parser = _Arguments(add_help=False)
        if action == "model":
            parser.add_argument("model")
            parser.add_argument("--as", dest="alias")
            parser.add_argument("--max-tokens", type=_positive)
            parser.add_argument("--context-window", type=_positive)
            parser.add_argument(
                "--budget-tokens", type=_positive, default=connect.DEFAULT_CHECK_BUDGET
            )
            parser.add_argument(
                "--reasoning-template",
                choices=("effort", "adaptive", "budget", "toggle", "thinking"),
            )
            parser.add_argument("--levels")
            parser.add_argument("--reasoning-default")
            return self.select_model(**vars(parser.parse_args(rest)))
        parser.add_argument("--api-style", choices=("chat", "responses", "anthropic"))
        parser.add_argument("--api-key-env")
        options = parser.parse_args(rest)
        preset = connect.PROVIDERS.get(action)
        return await self.start(
            preset.api_base if preset else action,
            api_style=options.api_style or (preset.api_style if preset else "chat"),
            api_key_env=options.api_key_env
            if options.api_key_env is not None
            else (preset.api_key_env if preset else ""),
        )

    def _credential(self) -> str | None:
        assert self.connection is not None
        name = self.connection[2]
        key = self._api_key or resolve_api_key_from_config("connect", {"api_key_env": name})
        if name and not key:
            raise ValueError(
                f"Set {name} in the server environment and retry. Do not paste credentials into chat."
            )
        return key

    async def start(
        self,
        endpoint: str,
        *,
        api_style: str = "chat",
        api_key_env: str = "",
        api_key: str | None = None,
    ) -> ControlResult:
        """Discover models; masked native input may supply a temporary credential."""
        self.reset()
        self._api_key = api_key
        endpoint = connect.normalize_endpoint(endpoint)
        if api_key_env and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise ValueError("Use an environment variable name for the key, not the key value.")
        if api_style not in ("chat", "responses", "anthropic"):
            raise ValueError("API style must be chat, responses or anthropic.")
        self.connection = (endpoint, api_style, api_key_env)
        self.discovery = await connect.discover(
            endpoint, api_style=api_style, api_key=self._credential()
        )
        self.connection = (self.discovery.api_base, api_style, api_key_env)
        return ControlResult.ok(
            ControlTable(
                columns=["Model ID"],
                rows=[
                    [json.dumps(model["id"], ensure_ascii=False)] for model in self.discovery.models
                ],
                title="Available models",
            ),
            ControlMessage(
                "Next: /connect model MODEL_ID --as ALIAS. No model calls or files written yet."
            ),
        )

    def select_model(
        self,
        model: str,
        *,
        alias: str | None = None,
        max_tokens: int | None = None,
        context_window: int | None = None,
        budget_tokens: int = connect.DEFAULT_CHECK_BUDGET,
        reasoning_template: str | None = None,
        levels: str | None = None,
        reasoning_default: str | None = None,
    ) -> ControlResult:
        """Build an offline preview using exact endpoint IDs and shared defaults."""
        if self.connection is None:
            raise ValueError("Start with /connect PROVIDER or /connect URL first.")
        endpoint, style, key_env = self.connection
        if bool(reasoning_template) != bool(levels):
            raise ValueError("Provide both --reasoning-template and --levels.")
        settings = None
        if levels:
            labels = [label.strip() for label in levels.split(",")]
            settings = {
                label: connect.reasoning_settings(reasoning_template, style, label)
                for label in labels
            }
        metadata = (
            next((item for item in self.discovery.models if item["id"] == model), None)
            if self.discovery
            else None
        )
        proposal = connect.plan(
            alias or model.rsplit("/", 1)[-1],
            model,
            style,
            endpoint,
            key_env,
            endpoint_model=metadata,
            reasoning_levels=settings,
            reply_tokens=max_tokens,
            budget_tokens=budget_tokens,
            session_checks=True,
        )
        if context_window is not None:
            if context_window <= 0:
                raise ValueError("Context window must be positive.")
            proposal.entry["context_window"] = context_window
            proposal.entry["provenance"].setdefault("limit_sources", {})["context_window"] = "user"
        if reasoning_default is not None:
            if reasoning_default not in proposal.entry.get("reasoning_levels", {}):
                raise ValueError("The reasoning default must name a configured level.")
            proposal.entry["reasoning_default"] = reasoning_default
        self.proposal = connect.refresh_plan(proposal)
        return self.preview()

    def preview(self) -> ControlResult:
        """Show the exact pending entry and spending limits before any checks."""
        if self.proposal is None:
            raise ValueError("Choose a model first with /connect model MODEL_ID.")
        p = self.proposal
        return ControlResult.ok(
            ControlMessage(
                f"Alias: {json.dumps(p.alias)}\nSave to: {self.registry_path}\n"
                + json.dumps(p.entry, indent=2, ensure_ascii=False)
            ),
            ControlMessage(
                f"Check budget: {p.budget_tokens:,} estimated tokens; all-check estimate: {p.token_estimate:,}. These estimates are not billing caps; model calls may incur charges."
            ),
            *(ControlMessage(warning, "warning") for warning in connect.entry_warnings(p.entry)),
            ControlMessage(
                "Next: /connect check minimal (routing only), /connect check all (tools, reasoning and session checks), or /connect save (save without new checks)."
            ),
        )

    async def check(self, mode: str) -> ControlResult:
        """Run only the user's approved checks; cancellation propagates to Connect."""
        if self.proposal is None:
            raise ValueError("Preview settings with /connect model MODEL_ID first.")
        proposal = self.proposal
        result = await connect.run(proposal, approved=mode, api_key=self._credential())
        self.proposal = replace(proposal, entry=result.entry)
        verdict = connect.verdict(result.entry)
        return ControlResult.ok(
            ControlMessage("Checks passed: " + (", ".join(verdict.passed) or "none")),
            ControlMessage(
                "Needs attention: " + (", ".join(verdict.needs_attention) or "none"),
                "warning" if verdict.needs_attention else "info",
            ),
            ControlMessage("Skipped/unconfirmed: " + (", ".join(verdict.skipped) or "none")),
            ControlMessage(
                "Review with /connect; save with /connect save. Nothing has been saved or switched."
            ),
        )

    def save(self, *, replace_existing: bool = False) -> ControlResult:
        """Persist one alias after an explicit save; never replace implicitly."""
        if self.proposal is None:
            raise ValueError("Preview settings with /connect model MODEL_ID first.")
        proposal = self.proposal
        connect.write(
            proposal.entry,
            self.registry_path,
            alias=proposal.alias,
            replace_existing=replace_existing,
        )
        self.reset()
        return ControlResult.ok(
            ControlMessage(
                f"Saved model {json.dumps(proposal.alias)} to {self.registry_path}.", "success"
            ),
            ControlMessage(
                "The running model is unchanged. Select this alias in your host or launch with --model ALIAS and this config file. Unchecked capabilities remain unconfirmed."
            ),
        )
