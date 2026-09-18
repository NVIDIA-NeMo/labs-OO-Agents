# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wizard steps; loaded lazily by the Connect command."""

import asyncio
import os
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import click
import httpx
import yaml

from nooa import paths
from nooa.unifiedllm import connect

from . import _connect_prompts as prompts
from . import _connect_view as view
from ._connect_registry import credential_names, diagnostic_context, entries, shadowing_source


@dataclass
class WizardState:
    """Explicit configuration and evidence carried between wizard steps."""

    alias: Any = None
    api_key: Any = None
    api_key_env: Any = None
    api_style: Any = None
    approval: Any = None
    budget_tokens: Any = None
    candidate: Any = None
    catalogue_model: Any = None
    configured: Any = None
    context_window: Any = None
    data: Any = None
    default_style: Any = None
    discovery_endpoint: Any = None
    discovery_file: Any = None
    discovery_style: Any = None
    discovery_succeeded: Any = None
    edit_model: Any = None
    edited_settings: Any = None
    editing: Any = None
    endpoint: Any = None
    endpoint_models: Any = None
    existing: Any = None
    explicit_key_env: Any = None
    interface_spent: Any = None
    interfaces: Any = None
    levels: Any = None
    levels_file: Any = None
    model: Any = None
    no_catalogue: Any = None
    no_probe: Any = None
    output: Any = None
    output_tokens: Any = None
    patches: Any = None
    path: Any = None
    probe: Any = None
    prompt_key: Any = None
    proposal: Any = None
    provider: Any = None
    reasoning_output_tokens: Any = None
    reasoning_template: Any = None
    registry: Any = None
    remaining_estimate: Any = None
    reply_tokens: Any = None
    result: Any = None
    server_urls: Any = None
    show_config: Any = None
    unobserved: Any = None
    yes: Any = None


async def show_checks(events, *, reasoning_levels=None, summary=True):
    with view.quiet_provider_messages():
        return await display_checks(events, reasoning_levels=reasoning_levels, summary=summary)


async def display_checks(events, *, reasoning_levels=None, summary=True):
    progress = view.CheckProgress()
    try:
        async with aclosing(events) as steps:
            async for event in steps:
                if isinstance(event, (connect.ConnectResult, connect.InterfaceResult)):
                    return event
                missing = connect.unobserved_reasoning_levels(
                    {
                        "reasoning_levels": reasoning_levels or {},
                        "provenance": {"probes": {event.name: event.outcome}},
                    }
                )
                progress.update(event.name, event.outcome, missing_reasoning=bool(missing))
    finally:
        progress.finish(summary=summary)
    raise click.ClickException("Checks ended without a result.")


def select_connection(state: WizardState) -> bool:
    """Select connection; return False when the user cancels."""
    state.registry = entries(state.path if state.output else None)
    if state.edit_model is not None:
        if any(
            (
                state.model,
                state.provider,
                state.endpoint,
                state.api_style,
                state.catalogue_model,
                state.reasoning_template,
                state.levels,
            )
        ):
            raise click.UsageError(
                "--edit-model cannot also select a new connection or catalogue template"
            )
        if state.edit_model == "":
            if state.yes:
                raise click.UsageError("With --yes, --edit-model requires an alias")
            if not state.registry:
                raise click.ClickException(
                    "No registry models to edit. Run nooa connect to add one."
                )
            state.edit_model = prompts.prompt(
                "Model to edit", choices=sorted(state.registry), open_menu=True
            )
        if state.edit_model not in state.registry:
            raise click.ClickException(f"Unknown registry model {state.edit_model!r}")
        state.editing, source_path = state.registry[state.edit_model]
        state.editing = deepcopy(state.editing)
        if not state.output:
            from nooa.llm_config import bundled_config_paths

            if source_path.resolve() not in {p.resolve() for p in bundled_config_paths()}:
                state.path = source_path
        state.alias = state.alias or state.edit_model
        routed = state.editing.get("model_name", state.edit_model)
        state.api_style = state.editing.get("api_style") or (
            "responses"
            if state.editing.get("client_type") == "responses"
            else "anthropic"
            if routed.startswith("anthropic/")
            else "chat"
        )
        prefix = "anthropic/" if state.api_style == "anthropic" else "openai/"
        state.model = routed.removeprefix(prefix)
        state.endpoint = (
            state.editing.get("api_base")
            or connect.PROVIDERS[
                "anthropic" if state.api_style == "anthropic" else "openai"
            ].api_base
        )
        if state.api_key_env is None:
            state.api_key_env = state.editing.get("api_key_env", "")
        state.explicit_key_env = True
        state.no_catalogue = True
        view.line(f"Editing {state.edit_model} from {source_path}. No model discovery is needed.")
    state.data = {}
    if state.path.exists():
        with state.path.open() as source:
            state.data = yaml.safe_load(source) or {}
    if isinstance(state.data, dict) and state.data.get("models") is None:
        state.data["models"] = {}
    if not isinstance(state.data, dict) or not isinstance(state.data.get("models", {}), dict):
        raise click.ClickException("Registry must contain a models mapping.")
    for name, entry in state.data.get("models", {}).items():
        if isinstance(name, str) and isinstance(entry, dict):
            state.registry.setdefault(name, (entry, state.path))
    state.server_urls = [p.api_base for p in connect.PROVIDERS.values()]
    for entry, _ in state.registry.values():
        address = entry.get("api_base") if isinstance(entry, dict) else None
        if isinstance(address, str):
            try:
                state.server_urls.append(connect.normalize_endpoint(address))
            except ValueError:
                pass  # Do not offer malformed URLs or embedded credentials.
    state.server_urls = list(dict.fromkeys(state.server_urls))
    state.default_style = "chat"
    state.approval = "none" if state.no_probe else state.probe
    state.budget_tokens = (
        connect.DEFAULT_CHECK_BUDGET if state.budget_tokens is None else state.budget_tokens
    )
    state.interfaces = None
    view.intro(
        checks=state.approval != "none",
        output_tokens=state.output_tokens,
        budget_tokens=state.budget_tokens,
        reasoning_output_tokens=state.reasoning_output_tokens,
    )
    if state.approval != "none" and not state.yes:
        if not prompts.confirm("Approve API checks within this budget?", default=True):
            click.echo("No API checks approved. Run with --no-probe for manual setup.")
            return False
    if state.editing is None:
        view.step(1, "Connection")
    if state.provider and state.provider not in (*connect.PROVIDERS, "custom"):
        raise click.UsageError(
            "Unknown provider. Choose nvidia, openai, anthropic, google, openrouter, or custom."
        )
    if not state.provider and not state.endpoint and not state.yes:
        state.provider = prompts.prompt(
            "Choose a provider",
            choices=(*connect.PROVIDERS, "custom"),
            labels={
                **{name: preset.label for name, preset in connect.PROVIDERS.items()},
                "custom": "Custom endpoint",
            },
            open_menu=True,
        )
    if state.provider and state.provider != "custom":
        preset = connect.PROVIDERS[state.provider]
        state.endpoint = state.endpoint or preset.api_base
        state.default_style = preset.api_style
        if state.api_key_env is None:
            state.api_key_env = preset.api_key_env
    if state.yes and state.provider and state.provider != "custom":
        state.api_style = state.api_style or state.default_style
    if state.yes and not all((state.model, state.alias, state.endpoint, state.api_style)):
        raise click.UsageError(
            "With --yes supply MODEL, --as and either --provider or --endpoint plus --api-style."
        )
    state.endpoint = state.endpoint or prompts.prompt(
        "Model server URL", suggestions=state.server_urls, open_menu=True
    )
    state.endpoint = connect.normalize_endpoint(state.endpoint)
    if not state.explicit_key_env:
        saved_names = credential_names(state.registry, state.endpoint)
        if len(saved_names) == 1:
            state.api_key_env = saved_names[0]
            if not state.api_key_env:
                view.line("Using saved key variable (no authentication) for this endpoint.")
            elif os.environ.get(state.api_key_env):
                view.line(f"Using saved key variable {state.api_key_env} for this endpoint.")
            else:
                # credential_names only remembers which variable NAME this
                # endpoint used last time, not whether a value is still set —
                # saying "using" here when we're about to prompt for the same
                # key again reads as broken, not reassuring.
                view.line(
                    f"This endpoint previously used key variable {state.api_key_env}, "
                    "but it has no value set."
                )
        elif len(saved_names) > 1:
            if state.yes:
                raise click.UsageError(
                    "Several key variables are saved for this endpoint; supply --api-key-env"
                )
            state.api_key_env = prompts.prompt(
                "Saved key variable",
                choices=[name or "-" for name in saved_names],
                open_menu=True,
            )
            if state.api_key_env == "-":
                state.api_key_env = ""
    # Listing/authentication conventions do not choose the selected model's
    # generation interface. A mixed server can list all models via /models.
    state.discovery_style = state.api_style or state.default_style
    if state.api_key_env is None:
        default_env = (
            "ANTHROPIC_API_KEY" if state.discovery_style == "anthropic" else "OPENAI_API_KEY"
        )
        state.api_key_env = (
            default_env
            if state.yes
            else prompts.prompt(
                "Key environment variable (new to enter a key; - for no authentication)",
                default=default_env,
                suggestions=prompts.environment_names(
                    [p.api_key_env for p in connect.PROVIDERS.values()] + ["-", "new"]
                ),
            )
        )
        if state.api_key_env == "-":
            state.api_key_env = ""
    if state.api_key_env == "new":
        state.api_key_env = prompts.prompt(
            "Save key under variable name",
            default="NOOA_MODEL_API_KEY",
            suggestions=prompts.environment_names(),
        )
        state.prompt_key = True
    # Validate before using an endpoint or collecting a credential.
    connect.plan(
        state.alias or "candidate",
        state.model or "candidate",
        state.discovery_style,
        state.endpoint,
        state.api_key_env,
    )
    needs_key = (
        not state.yes
        and (state.approval != "none" or not state.model)
        and state.api_key_env
        and not os.environ.get(state.api_key_env)
    )
    state.api_key = (
        prompts.prompt("API key (used only for this setup)", hide_input=True)
        if state.prompt_key or needs_key
        else os.environ.get(state.api_key_env)
        if state.api_key_env
        else None
    )
    return True


def select_model(state: WizardState) -> bool:
    """Select model; return False when the user cancels."""
    if state.editing is None:
        view.step(2, "Model")
    state.discovery_succeeded = None
    state.discovery_endpoint = None
    state.endpoint_models = ()
    if state.discovery_file:
        from ._connect_stages import read_discovery

        state.endpoint_models = read_discovery(state.discovery_file, state.endpoint)
        state.discovery_endpoint = state.endpoint
    if not state.model:
        click.echo("Connecting to the server and listing models...")
        try:
            found = asyncio.run(
                connect.discover(
                    state.endpoint, api_style=state.discovery_style, api_key=state.api_key
                )
            )
        except connect.DiscoveryError as exc:
            state.discovery_succeeded = False
            state.discovery_endpoint = state.endpoint
            click.echo(f"Could not list models: {exc}", err=True)
            if exc.status_code in {401, 403}:
                raise click.ClickException(
                    "Authentication failed. Check the key and try again."
                ) from None
            state.model = prompts.prompt("Exact model ID (if known; Ctrl-C to cancel)")
        else:
            state.endpoint = found.api_base
            state.discovery_succeeded = True
            state.discovery_endpoint = state.endpoint
            state.endpoint_models = found.models
            names = [item["id"] for item in found.models]
            click.echo(
                f"Server listed {len(names)} model(s). Credentials are checked next. Type part of a name to search, then Tab to select."
            )
            state.model = prompts.prompt("Model", choices=names)
    return True


def check_interfaces(state: WizardState) -> bool:
    """Check interfaces; return False when the user cancels."""
    view.step(3, "Connection checks")
    if state.api_style == "responses":
        view.line(connect.ENCRYPTED_REASONING_EXPLANATION, dim=True)
    if not state.api_style:
        available = ("chat", "responses", "anthropic")
        if state.approval != "none":
            state.interface_spent = 0
            retry_styles = ("chat", "responses", "anthropic")
            interface_timeout = 30
            while True:
                state.interfaces = asyncio.run(
                    show_checks(
                        connect.check_interfaces(
                            state.alias or "candidate",
                            state.model,
                            state.endpoint,
                            state.api_key_env,
                            budget_tokens=max(0, state.budget_tokens - state.interface_spent),
                            output_tokens=state.output_tokens,
                            api_key=state.api_key,
                            styles=retry_styles,
                            timeout_seconds=interface_timeout,
                        ),
                        summary=False,
                    )
                )
                state.interface_spent += state.interfaces.tokens_charged_to_budget
                state.interfaces = replace(
                    state.interfaces, tokens_charged_to_budget=state.interface_spent
                )
                available = state.interfaces.accepted
                if available:
                    break
                failed_checks = {
                    style: r.entry["provenance"]["probes"]["routing"]
                    for style, r in state.interfaces.results.items()
                }
                slow = (
                    state.endpoint == state.discovery_endpoint
                    and state.discovery_succeeded
                    and all(r.get("timeout_kind") for r in failed_checks.values())
                )
                view.line(
                    "Model listing succeeded, but model responses timed out; the route may be slow."
                    if slow
                    else "Could not confirm a working connection. Listing models does not validate the key.",
                    fg="yellow",
                )
                click.echo(
                    "Agent diagnostic prompt:\n"
                    + connect.diagnostic_prompt(
                        "interfaces",
                        {
                            "model_name": state.model,
                            "api_base": state.endpoint,
                            "api_key_env": state.api_key_env,
                        },
                        failed_checks,
                        api_key=state.api_key,
                        run_context=diagnostic_context(
                            target=state.path,
                            alias=state.alias,
                            model=state.model,
                            endpoint=state.endpoint,
                            api_key_env=state.api_key_env,
                            api_key=state.api_key,
                            budget=state.budget_tokens,
                            remaining=max(0, state.budget_tokens - state.interface_spent),
                            output_tokens=state.output_tokens,
                            reasoning_output_tokens=state.reasoning_output_tokens,
                            stage="interfaces",
                            interface_timeout_seconds=interface_timeout,
                            discovery_succeeded=state.discovery_succeeded
                            if state.endpoint == state.discovery_endpoint
                            else None,
                        ),
                    )
                )
                if state.yes:
                    raise click.ClickException(
                        "Check credentials and endpoint, or run without --yes to correct them interactively."
                    )
                remaining = max(0, state.budget_tokens - state.interface_spent)
                if remaining < state.output_tokens + 512:
                    view.line(
                        "The approved check budget is exhausted. Nothing was saved; restart setup to approve a new budget."
                    )
                    return False
                view.line(
                    f"You can correct the connection here. {remaining:,} estimated tokens remain in the approved budget."
                )
                action = prompts.prompt(
                    "Next step",
                    choices=("key", "server", "retry", "longer", "cancel"),
                    default="longer" if slow else "key",
                    labels={
                        "key": "Change key",
                        "server": "Edit server and model",
                        "retry": "Try again unchanged",
                        "longer": "Retry one interface with a 120-second timeout",
                        "cancel": "Exit without saving",
                    },
                )
                if action == "cancel":
                    click.echo("Setup cancelled. Nothing was saved.")
                    return False
                if action == "longer":
                    retry_styles = (
                        prompts.prompt(
                            "Interface to retry",
                            choices=("chat", "responses", "anthropic"),
                            default=state.default_style,
                        ),
                    )
                    interface_timeout = 120
                    continue
                retry_styles = ("chat", "responses", "anthropic")
                interface_timeout = 30
                if action == "key":
                    source = prompts.prompt(
                        "Key environment variable (or paste for a temporary key)",
                        default=state.api_key_env or "paste",
                        suggestions=prompts.environment_names(["paste"]),
                    )
                    if source == "paste":
                        state.api_key = prompts.prompt(
                            "API key (used only for this setup)", hide_input=True
                        )
                    else:
                        state.api_key_env = source
                        state.api_key = os.environ.get(source)
                        if not state.api_key:
                            view.line(
                                "That variable is unset or empty. You can paste a temporary key instead."
                            )
                            state.api_key = prompts.prompt(
                                "API key (used only for this setup)", hide_input=True
                            )
                elif action == "server":
                    state.endpoint = connect.normalize_endpoint(
                        prompts.prompt(
                            "Model server URL",
                            default=state.endpoint,
                            suggestions=state.server_urls,
                            open_menu=True,
                        )
                    )
                    state.model = prompts.prompt("Exact model ID", default=state.model)
            click.echo(
                "Interfaces that returned the expected response format: " + ", ".join(available)
            )
        if state.interfaces and len(available) == 1:
            state.api_style = available[0]
            click.echo(f"Using {state.api_style} for {state.model}.")
        else:
            click.echo(f"Choose the request interface for {state.model}:")
            click.echo(
                "chat = OpenAI-compatible; responses = OpenAI Responses; anthropic = Anthropic Messages."
            )
            state.api_style = prompts.prompt(
                "API format",
                choices=available,
                default=state.default_style if state.default_style in available else available[0],
            )
        if state.api_style == "responses":
            view.line(connect.ENCRYPTED_REASONING_EXPLANATION, dim=True)
    return True


def configure_metadata(state: WizardState) -> bool:
    """Configure metadata; return False when the user cancels."""
    # An explicit --as can reuse its previous evidence. Otherwise checks use
    # a temporary label; the user names the entry only when ready to save.
    state.existing = state.editing or (
        state.data.get("models", {}).get(state.alias) if state.alias else None
    )
    state.candidate = None
    state.edited_settings = False
    if state.editing is not None:
        state.candidate = {
            "id": state.editing.get("underlying_model", state.model),
            "context_length": state.editing.get("context_window"),
            "top_provider": {
                "max_completion_tokens": state.editing.get(
                    "max_output_tokens",
                    state.editing.get("provenance", {})
                    .get("catalogue_limits", {})
                    .get("max_completion_tokens"),
                )
            },
            "reasoning": {
                "supported_efforts": list(state.editing.get("reasoning_levels", {})),
                "default_effort": state.editing.get("reasoning_default"),
            },
        }
        if not state.yes:
            state.candidate = prompts.edit_model_details(state.candidate)
        state.edited_settings = True
    if state.no_catalogue and state.catalogue_model:
        raise click.UsageError("--catalogue-model cannot be used with --no-catalogue")
    if not state.no_catalogue:
        click.echo("Looking up public model information...")
        try:
            models = asyncio.run(connect.catalogue())
        except (httpx.HTTPError, ValueError, KeyError):
            if state.catalogue_model:
                raise click.ClickException(
                    "Could not load the requested catalogue entry."
                ) from None
            click.echo("Public catalogue unavailable; continuing with unknown limits.", err=True)
            models = []
        matches = (
            [item for item in models if item.get("id") == state.catalogue_model]
            if state.catalogue_model
            else connect.match_models(state.model, models)
        )
        if state.catalogue_model and not matches:
            raise click.ClickException("Requested catalogue model was not found.")
        if len(matches) == 1:
            state.candidate = matches[0]
        elif matches:
            click.echo("Possible catalogue models: " + ", ".join(item["id"] for item in matches))
            if state.yes:
                raise click.ClickException(
                    "Ambiguous match: choose --catalogue-model or --no-catalogue."
                )
            selected = prompts.prompt(
                "Catalogue model (blank leaves it unknown)",
                default="",
                show_default=False,
                choices=[""] + [item["id"] for item in matches],
            )
            if selected:
                state.candidate = next((item for item in matches if item["id"] == selected), None)
                if state.candidate is None:
                    raise click.ClickException("Choose one of the displayed model IDs.")
        else:
            click.echo("No catalogue match; model limits and reasoning levels remain unknown.")
    endpoint_model = (
        next((item for item in state.endpoint_models if item.get("id") == state.model), None)
        if state.endpoint == state.discovery_endpoint
        else None
    )
    if (
        endpoint_model is None
        and state.candidate is not None
        and state.editing is None
        and not state.endpoint_models
    ):
        # Explicit MODEL skips the picker, not the endpoint's own limit metadata.
        try:
            limits_listing = asyncio.run(
                connect.discover(state.endpoint, api_style=state.api_style, api_key=state.api_key)
            )
        except connect.DiscoveryError:
            view.line(
                "Endpoint limits unavailable; catalogue limits are fallback information.",
                fg="yellow",
            )
        else:
            endpoint_model = next(
                (item for item in limits_listing.models if item.get("id") == state.model), None
            )
    if endpoint_model is not None and state.editing is None:
        combined = connect.model_metadata(state.model, state.candidate, endpoint_model)
        if combined.get("endpoint_limits"):
            state.candidate = combined
    if state.candidate is not None:
        while True:
            view.model_details(
                state.candidate, output_tokens=state.output_tokens, edited=state.edited_settings
            )
            action = (
                "use"
                if state.yes
                else prompts.prompt(
                    "Model settings",
                    default="use",
                    choices=("use", "edit", "skip", "cancel"),
                    labels={
                        "use": "Use these settings",
                        "edit": "Edit settings",
                        "skip": "Continue without these settings",
                        "cancel": "Cancel setup",
                    },
                    open_menu=True,
                )
            )
            if action == "cancel":
                click.echo("Setup cancelled. Nothing saved.")
                return False
            if action == "skip":
                if state.editing is not None:
                    click.echo("Edits discarded. Nothing saved.")
                    return False
                state.candidate = None
                state.edited_settings = False
                click.echo(
                    "Continuing without the published model settings. Explicit command-line settings still apply."
                )
                break
            if action == "use":
                break
            state.candidate = prompts.edit_model_details(state.candidate)
            state.edited_settings = True
        if state.edited_settings and state.editing is None:
            # Apply edits only after confirmation, not if the user skips them.
            state.context_window = None
            state.levels_file = state.levels = state.reasoning_template = None
    return True


def configure_checks(state: WizardState) -> bool:
    """Configure checks; return False when the user cancels."""
    if state.levels_file and (state.levels or state.reasoning_template):
        raise click.UsageError("Use either --levels-file or --reasoning-template with --levels.")
    state.patches = None
    if state.editing is not None:
        labels = state.candidate.get("reasoning", {}).get("supported_efforts", [])
        original_levels = state.editing.get("reasoning_levels", {})
        if not state.levels_file and any(label not in original_levels for label in labels):
            raise click.UsageError(
                "New reasoning levels need request settings; supply --levels-file"
            )
        if not state.levels_file:
            state.patches = {label: deepcopy(original_levels[label]) for label in labels}
    if state.levels_file:
        with Path(state.levels_file).open() as source:
            state.patches = yaml.safe_load(source)
    if state.levels or state.reasoning_template:
        if not state.reasoning_template or not state.levels:
            raise click.UsageError("--reasoning-template and --levels must be supplied together.")
        state.patches = {
            label.strip(): connect.reasoning_settings(
                state.reasoning_template, state.api_style, label.strip()
            )
            for label in state.levels.split(",")
        }
    state.proposal = connect.plan(
        state.alias or "candidate",
        state.model,
        state.api_style,
        state.endpoint,
        state.api_key_env,
        catalogue=state.candidate,
        reasoning_levels=state.patches,
        budget_tokens=state.budget_tokens,
        output_tokens=state.output_tokens,
        reasoning_output_tokens=state.reasoning_output_tokens,
        existing_entry=state.interfaces.results[state.api_style].entry
        if state.interfaces
        else state.existing,
        session_checks=state.approval == "all",
        reply_tokens=state.reply_tokens,
    )
    if state.editing is not None:
        # Preserve transport controls, custom parameters and exact level blocks.
        merged = deepcopy(state.editing)
        for field in (
            "context_window",
            "max_output_tokens",
            "reasoning_levels",
            "reasoning_default",
        ):
            merged.pop(field, None)
            if field in state.proposal.entry:
                merged[field] = deepcopy(state.proposal.entry[field])
        merged["api_key_env"] = state.api_key_env
        merged.setdefault("api_style", state.api_style)
        if state.proposal.entry.get("allowed_openai_params"):
            merged["allowed_openai_params"] = sorted(
                set(merged.get("allowed_openai_params", []))
                | set(state.proposal.entry["allowed_openai_params"])
            )
        if "reasoning_levels" not in state.editing and not merged.get("reasoning_levels"):
            merged.pop("reasoning_levels", None)
        merged["provenance"] = {
            **deepcopy(state.editing.get("provenance", {})),
            **state.proposal.entry["provenance"],
        }
        merged["provenance"]["probes"] = {}  # Edits must not reuse stale evidence.
        state.proposal = replace(state.proposal, entry=merged)
        if state.api_style == "responses":
            for check in state.proposal.probes:
                for field in ("store", "include"):
                    check.body.pop(field, None)
                    if field in merged:
                        check.body[field] = deepcopy(merged[field])
    if state.edited_settings:
        for field in (
            "context_window",
            "max_output_tokens",
            "reasoning_levels",
            "reasoning_default",
        ):
            state.proposal.entry["provenance"][field] = {
                "source": "user",
                "value": state.proposal.entry.get(field),
            }
    state.interface_spent = state.interfaces.tokens_charged_to_budget if state.interfaces else 0
    state.proposal = replace(
        state.proposal,
        budget_tokens=max(0, state.budget_tokens - state.interface_spent),
    )
    if state.interfaces:
        state.proposal.entry["provenance"]["interfaces"] = {
            style: state.result.entry["provenance"]["probes"]["routing"]
            for style, state.result in state.interfaces.results.items()
        }
    if state.context_window:
        state.proposal.entry["context_window"] = state.context_window
        state.proposal.entry["provenance"]["context_window"] = {
            "source": "user",
            "value": state.context_window,
        }
    if state.patches:
        state.proposal.entry["provenance"]["reasoning_levels"] = {
            "source": "user",
            "template": state.reasoning_template,
        }
    if "context_window" not in state.proposal.entry:
        click.echo(
            "No context window selected. The runtime will use its fallback; set --context-window to supply a limit."
        )
    state.configured = connect.configure_entry(
        state.proposal.entry, reply_tokens=state.reply_tokens
    )
    if state.reply_tokens is None and not state.yes:
        bounds = [
            v
            for v in (
                state.configured["context_window"] - 1
                if state.configured.get("context_window")
                else None,
                state.configured["provenance"]
                .get("catalogue_limits", {})
                .get("max_completion_tokens"),
            )
            if isinstance(v, int) and v > 0
        ]
        chosen_cap = prompts.choose_reply_limit(
            state.configured["max_tokens"],
            min(bounds) if bounds else None,
            source=state.configured["provenance"]["reply_limit"]["source"],
        )
        state.configured = connect.configure_entry(state.configured, reply_tokens=chosen_cap)
    state.proposal = connect.refresh_plan(replace(state.proposal, entry=state.configured))
    state.remaining_estimate = state.proposal.token_estimate
    view.line(
        f"Saved reply budget: {state.configured['max_tokens']:,} tokens, shared by reasoning and the answer."
    )
    for label, limit in state.configured["provenance"].get("level_reply_limits", {}).items():
        view.line(
            f"Reasoning level {label}: reply budget raised to {limit['value']:,} tokens to leave room for its thinking budget.",
            fg="yellow",
        )
    return True


def run_checks(state: WizardState) -> bool:
    """Run checks; return False when the user cancels."""
    price = (
        "unknown"
        if state.proposal.price_estimate is None
        else f"~${state.proposal.price_estimate:.6f} at catalogue prices"
    )
    view.line("Check plan", fg="bright_cyan", bold=True)
    view.line(
        f"Connection · tools · {len(state.proposal.entry.get('reasoning_levels', {}))} reasoning settings",
        dim=True,
    )
    if state.proposal.session_checks:
        view.line(
            "Then 3 conversation replies to check cache reuse and reasoning retention.",
            dim=True,
        )
    view.line(
        "Configured checks send the saved reply cap, including level overrides. Insufficient budget skips the check; caps are never reduced for testing.",
        dim=True,
    )
    view.line(
        f"Estimated tokens: {state.remaining_estimate:,} · budget remaining: {state.proposal.budget_tokens:,} · estimated price: {price}",
        dim=True,
    )
    if state.remaining_estimate > state.proposal.budget_tokens:
        click.echo(
            "Warning: the approved budget is too small for all checks. Some will be skipped. Restart with a larger --budget-tokens value to run them all.",
            err=True,
        )
    click.echo(
        "No retries or capacity probes. Estimates are not billing limits: endpoints can ignore output caps."
    )
    state.result = asyncio.run(
        show_checks(
            connect.run_steps(state.proposal, approved=state.approval, api_key=state.api_key),
            reasoning_levels=state.proposal.entry.get("reasoning_levels"),
        )
    )
    state.result.entry["provenance"]["tokens_charged_to_budget"] = (
        state.result.entry["provenance"].get("tokens_charged_to_budget", 0) + state.interface_spent
    )
    skipped = [
        name
        for name, record in state.result.entry["provenance"]["probes"].items()
        if record.get("reason") == "budget exhausted"
    ]
    if skipped:
        click.echo(
            "Warning: setup is incomplete; budget exhausted before "
            + ", ".join(skipped)
            + ". These settings have not been checked.",
            err=True,
        )
    decision = connect.verdict(state.result.entry)
    state.unobserved = decision.unobserved_levels
    checks = {
        **state.result.entry["provenance"]["probes"],
        **state.result.entry["provenance"].get("session_checks", {}),
    }
    if state.approval != "none" and (
        decision.needs_attention
        or any(checks[name].get("reason") != "not approved" for name in decision.skipped)
    ):
        click.echo(
            "Agent diagnostic prompt:\n"
            + connect.diagnostic_prompt(
                "checks",
                state.result.entry,
                checks,
                api_key=state.api_key,
                run_context=diagnostic_context(
                    target=state.path,
                    alias=state.alias,
                    model=state.model,
                    endpoint=state.endpoint,
                    api_key_env=state.api_key_env,
                    api_key=state.api_key,
                    budget=state.budget_tokens,
                    remaining=max(
                        0,
                        state.proposal.budget_tokens
                        - state.result.entry["provenance"].get("tokens_charged_to_budget", 0),
                    ),
                    output_tokens=state.output_tokens,
                    reasoning_output_tokens=state.reasoning_output_tokens,
                ),
            )
        )
    if state.unobserved:
        click.echo(
            "Warning: no reasoning information was returned for: "
            + ", ".join(state.unobserved)
            + ". These checks have not confirmed reasoning for those levels. "
            "Try another API format or review the server's reasoning settings. "
            "Some servers do not expose reasoning information.",
            err=True,
        )
    return True


def save_model(state: WizardState) -> bool:
    """Save model; return False when the user cancels."""
    view.step(4, "Save model")
    # Checks may take a while: refresh names before offering completion or
    # asking to replace an entry added since setup started.
    state.data = {}
    if state.path.exists():
        with state.path.open() as source:
            state.data = yaml.safe_load(source) or {}
    if isinstance(state.data, dict) and state.data.get("models") is None:
        state.data["models"] = {}
    if not isinstance(state.data, dict) or not isinstance(state.data.get("models", {}), dict):
        raise click.ClickException("Registry must contain a models mapping.")
    while not state.alias or not state.alias.strip():
        state.alias = prompts.prompt(
            "Save this model as",
            default=state.model.rsplit("/", 1)[-1],
            suggestions=list(state.data.get("models", {})) + [state.model.rsplit("/", 1)[-1]],
            existing=tuple(state.data.get("models", {})),
        )
        if not state.alias.strip():
            click.echo("Enter a non-empty model name.", err=True)
    if state.alias in state.data.get("models", {}):
        click.echo(
            f"Warning: saving will overwrite model {state.alias!r} in {state.path}.", err=True
        )
        if not state.yes and not prompts.confirm("Replace this model?", default=False):
            return False
    state.result = replace(state.result, alias=state.alias)
    view.line(f"{state.alias} · {state.model} · {state.api_style}", bold=True)
    if state.show_config:
        click.echo(yaml.safe_dump({"models": {state.alias: state.result.entry}}, sort_keys=False))
    else:
        view.line(
            "Full configuration is saved with the model. Use --show-config to preview the YAML.",
            dim=True,
        )
    if shadow := shadowing_source(state.alias, state.path):
        view.line(
            f"Warning: {shadow} currently defines this alias and takes precedence over this destination. Update that file or explicitly load {state.path} to use this entry.",
            fg="yellow",
        )
    view.line(
        f"Save summary: {state.result.entry['api_style']} · reply budget {state.result.entry['max_tokens']:,} tokens"
    )
    view.line(
        "Reasoning levels: "
        + (", ".join(state.result.entry.get("reasoning_levels", {})) or "none configured")
        + "; default: "
        + str(state.result.entry.get("reasoning_default", "server default"))
    )
    if state.yes or prompts.confirm(f"Write model entry to {state.path}?", default=True):
        save_key = False
        if (
            state.api_key
            and state.api_key_env
            and state.api_key != os.environ.get(state.api_key_env)
        ):
            secrets_path = paths.get_user_dir("secrets.yaml")
            view.line(
                f"This setup used a new key. It can be saved in {secrets_path} with owner-only permissions (plain text, not encrypted). Existing values are preserved; YAML formatting may change."
            )
            save_key = not state.yes and prompts.confirm(
                "Save this key for future NOOA runs?", default=True
            )
            if save_key:
                from nooa.secrets import write_secret_env

                write_secret_env(secrets_path, state.api_key_env, state.api_key)
                view.line(
                    f"Saved key as {state.api_key_env}. Existing secret values are preserved; YAML formatting may change."
                )
                if os.environ.get(state.api_key_env):
                    view.line(
                        f"Your current environment still overrides this file. Unset or update {state.api_key_env} before starting NOOA again.",
                        fg="yellow",
                    )
        connect.write(state.result.entry, state.path, alias=state.alias)
        click.echo(f"Saved {state.alias} to {state.path}.")
        view.line(
            f"Interface: {state.api_style} · max_tokens: {state.result.entry['max_tokens']:,} · reasoning levels: {', '.join(state.result.entry.get('reasoning_levels', {})) or 'unknown'} · default: {state.result.entry.get('reasoning_default', 'unknown')}"
        )
        if state.api_key and not save_key and state.api_key != os.environ.get(state.api_key_env):
            click.echo(
                f"The key was not saved. Set {state.api_key_env} (or add it to your NOOA secrets file) before using this alias."
            )
        click.echo(f'Use it in Python: get_llm_client("{state.alias}")')
        if state.output:
            click.echo(
                "For a custom path, include it in NEMO_OO_LLM_CONFIG or reload_registry(path)."
            )
    return True


def run_wizard(**options):
    """Run ordered steps with one shared budget and an explicit save decision."""
    state = WizardState(**options)
    state.path = Path(state.output) if state.output else paths.get_user_dir("llm_config.yaml")
    state.api_key = None
    state.explicit_key_env = state.api_key_env is not None
    state.editing = None
    try:
        for step in (
            select_connection,
            select_model,
            check_interfaces,
            configure_metadata,
            configure_checks,
            run_checks,
            save_model,
        ):
            if not step(state):
                return
    except (ValueError, OSError, yaml.YAMLError, httpx.HTTPError) as exc:
        detail = view.local_failure(exc, api_key=state.api_key, api_key_env=state.api_key_env)
        click.echo(
            "Agent diagnostic prompt:\n"
            + connect.diagnostic_prompt(
                "setup",
                {},
                {"setup": {"outcome": "failed", "error": type(exc).__name__, "detail": detail}},
                api_key=state.api_key,
                run_context=diagnostic_context(
                    target=state.path,
                    alias=state.alias,
                    model=state.model,
                    api_key_env=state.api_key_env,
                    api_key=state.api_key,
                    output_tokens=state.output_tokens,
                    reasoning_output_tokens=state.reasoning_output_tokens,
                ),
            ),
            err=True,
        )
        raise click.ClickException(detail) from None
