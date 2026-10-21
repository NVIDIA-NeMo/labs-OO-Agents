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
from nooa.unifiedllm.connect._records import text_baseline_passed

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
    endpoint_models_url: Any = None
    listing_api_base: Any = None
    discovery_file: Any = None
    discovery_style: Any = None
    models_url: Any = None
    listing_style: Any = None
    discovery_api_key_env: Any = None
    discovery_api_key: Any = None
    discovery_succeeded: Any = None
    edit_model: Any = None
    edited_settings: Any = None
    edited_level_patches: Any = None
    editing: Any = None
    endpoint: Any = None
    endpoint_models: Any = None
    existing: Any = None
    explicit_key_env: Any = None
    interface_spent: Any = None
    share_interface_key: bool = False
    connection_endpoint_explicit: bool = False
    interfaces: Any = None
    levels: Any = None
    levels_file: Any = None
    metadata_prepared: bool = False
    metadata_displayed: bool = False
    public_details_status: str | None = None
    guided: bool = False
    guided_media_explicit: bool = False
    guided_media_revised: bool = False
    selected_endpoint_model: Any = None
    media_hints: Any = None
    media_inputs: Any = None
    inherit_media_inputs: bool = False
    test_media_inputs: Any = None
    decision_image_transport: Any = None
    typesafe_danger_zone_images: Any = None
    model: Any = None
    no_catalogue: Any = None
    no_metadata: bool = False
    server_metadata: str = "listing"
    server_details: Any = None
    no_probe: Any = None
    output: Any = None
    output_tokens: Any = None
    patches: Any = None
    path: Any = None
    probe: Any = None
    prompt_key: Any = None
    proposal: Any = None
    provider: Any = None
    reasoning_level: Any = None
    reasoning_output_tokens: Any = None
    reasoning_template: Any = None
    registry: Any = None
    remaining_estimate: Any = None
    reply_tokens: Any = None
    cache_enabled: bool | None = None
    structured_output_mode: str | None = None
    structured_output_mode_set: bool = False
    result: Any = None
    server_urls: Any = None
    systemone_endpoint: Any = None
    interface_endpoints: Any = None
    show_config: Any = None
    show_diagnostics: Any = False
    unobserved: Any = None
    working_dir: Any = None
    yes: Any = None


def scrub_setup(state: WizardState, value):
    """Protect both active credentials in previews and troubleshooting output."""
    from nooa.unifiedllm.connect._diagnostics import scrub_report

    value = scrub_report(value, api_key=state.api_key, api_key_env=state.api_key_env)
    return scrub_report(
        value, api_key=state.discovery_api_key, api_key_env=state.discovery_api_key_env
    )


def setup_diagnostic(state: WizardState, *args, **kwargs):
    return scrub_setup(state, connect.diagnostic_prompt(*args, **kwargs))


async def show_checks(
    events, *, reasoning_levels=None, summary=True, media_inputs=None, diagnostics=False
):
    with view.quiet_provider_messages():
        return await display_checks(
            events,
            reasoning_levels=reasoning_levels,
            summary=summary,
            media_inputs=media_inputs,
            diagnostics=diagnostics,
        )


async def display_checks(
    events, *, reasoning_levels=None, summary=True, media_inputs=None, diagnostics=False
):
    progress = view.CheckProgress()
    progress.diagnostics = diagnostics
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
                progress.update(
                    event.name,
                    event.outcome,
                    missing_reasoning=bool(missing),
                    media_inputs=media_inputs,
                )
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
        connect.configure_entry(state.editing)  # Validate saved declarations before credentials.
        if not state.output:
            from nooa.llm_config import bundled_config_paths

            if source_path.resolve() not in {p.resolve() for p in bundled_config_paths()}:
                state.path = source_path
        state.alias = state.alias or state.edit_model
        routed = state.editing.get("model_name", state.edit_model)
        state.api_style = state.editing.get("api_style") or (
            "systemone"
            if state.editing.get("client_type") == "decision"
            else "responses"
            if state.editing.get("client_type") == "responses"
            else "anthropic"
            if routed.startswith("anthropic/")
            else "chat"
        )
        prefix = (
            ""
            if state.api_style in connect.DECISION_API_STYLES
            else "anthropic/"
            if state.api_style == "anthropic"
            else "openai/"
        )
        state.model = routed.removeprefix(prefix)
        state.endpoint = (
            state.editing.get("endpoint")
            if state.api_style in connect.DECISION_API_STYLES
            else state.editing.get("api_base")
            or connect.PROVIDERS[
                "anthropic" if state.api_style == "anthropic" else "openai"
            ].api_base
        )
        if state.api_key_env is None:
            state.api_key_env = state.editing.get("api_key_env", "")
        state.explicit_key_env = True
        state.no_catalogue = True
        view.line(
            scrub_setup(
                state,
                f"Editing {state.edit_model} from {source_path}. No model discovery is needed.",
            )
        )
        # --working-dir only redirects entries() to the target's own file
        # when that file already exists (it cannot read a file that isn't
        # there yet) — otherwise this alias was found via the general
        # chain (cwd/global), not target's own copy. Silently saving that
        # into target without saying so looks like an in-place edit of
        # target's own prior settings when it's actually a copy from
        # elsewhere into a project that never had this alias.
        if state.working_dir and source_path.resolve() != state.path.resolve():
            view.line(
                scrub_setup(
                    state,
                    f"{state.path} does not define {state.edit_model!r} yet; copying it in from "
                    f"{source_path} rather than editing it in place there.",
                ),
                fg="yellow",
            )
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
        address = (
            entry.get("endpoint") or entry.get("api_base") if isinstance(entry, dict) else None
        )
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
    if state.editing is None:
        view.step(1, "Connection")
    if state.provider and state.provider not in (*connect.PROVIDERS, "custom"):
        raise click.UsageError(
            f"Unknown provider. Choose {', '.join(connect.PROVIDERS)}, or custom."
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
    # Model listing/auth follows provider or an independent listing style.
    # Choose the inference interface only after selecting the model.
    explicit_endpoint = state.endpoint
    state.connection_endpoint_explicit = explicit_endpoint is not None
    state.interface_endpoints = {}
    if state.systemone_endpoint:
        state.interface_endpoints["systemone"] = connect.normalize_endpoint(
            state.systemone_endpoint
        )
    elif state.provider == "hub":
        state.interface_endpoints["systemone"] = connect.PROVIDERS["hub"].api_base + "/systemone"
    elif state.provider == "openrouter":
        state.interface_endpoints["systemone"] = "https://openrouter.ai/api/alpha/decisions"
    if state.provider and state.provider != "custom":
        preset = connect.PROVIDERS[state.provider]
        if state.api_style in connect.DECISION_API_STYLES and not explicit_endpoint:
            state.endpoint = state.interface_endpoints.get(state.api_style)
            if not state.endpoint and state.yes:
                raise click.UsageError(
                    "Decision styles require a full --endpoint URL for this provider"
                )
        else:
            state.endpoint = state.endpoint or preset.api_base
        state.default_style = preset.api_style
        if state.api_key_env is None:
            state.api_key_env = preset.api_key_env
    if state.yes and state.provider and state.provider != "custom":
        state.api_style = state.api_style or state.default_style
    reject_decision_media(state)
    if state.structured_output_mode_set and state.api_style is None:
        raise click.UsageError(
            "--structured-output-mode requires --api-style chat or anthropic, not interface discovery"
        )
    if state.yes and not all((state.model, state.alias, state.endpoint, state.api_style)):
        raise click.UsageError(
            "With --yes supply MODEL, --as and either --provider or --endpoint plus --api-style."
        )
    state.endpoint = state.endpoint or prompts.prompt(
        "Full decision request URL"
        if state.api_style in connect.DECISION_API_STYLES
        else "Model server URL",
        suggestions=state.server_urls,
        open_menu=True,
    )
    state.endpoint = connect.normalize_endpoint(state.endpoint)
    announced = False
    if not state.explicit_key_env:
        saved_names = credential_names(state.registry, state.endpoint)
        if len(saved_names) == 1:
            state.api_key_env = saved_names[0]
            announced = True
            if not state.api_key_env:
                view.line(
                    scrub_setup(state, "Using this endpoint without authentication, as saved.")
                )
            elif os.environ.get(state.api_key_env):
                view.line(
                    scrub_setup(
                        state, f"Using saved key variable {state.api_key_env} for this endpoint."
                    )
                )
            else:
                # credential_names only remembers which variable NAME this
                # endpoint used last time, not whether a value is still set —
                # saying "using" here when we're about to prompt for the same
                # key again reads as broken, not reassuring.
                view.line(
                    scrub_setup(
                        state,
                        f"This endpoint previously used key variable {state.api_key_env}, "
                        "but it has no value set.",
                    )
                )
        elif len(saved_names) > 1:
            if state.yes:
                raise click.UsageError(
                    "Several key variables are saved for this endpoint; supply --api-key-env"
                )
            state.api_key_env = prompts.prompt(
                "Saved key variable",
                choices=[name or "-" for name in saved_names],
                labels={name or "-": name or "No authentication" for name in saved_names},
                open_menu=True,
            )
            if state.api_key_env == "-":
                state.api_key_env = ""
    # Listing/authentication conventions do not choose the selected model's
    # generation interface. A mixed server can list all models via /models.
    state.discovery_style = state.listing_style or (
        state.api_style
        if state.api_style in {"chat", "responses", "anthropic"}
        else state.default_style
    )
    if state.provider in {"hub", "openrouter"}:
        # Shared provider catalogues are independent of the full decision POST URL.
        state.models_url = (
            state.models_url or connect.PROVIDERS[state.provider].api_base + "/models"
        )
    if state.models_url:
        connect.validate_models_url(state.models_url)
    if state.api_key_env is None:
        default_env = (
            "DECISION_MODEL_API_KEY"
            if state.api_style in connect.DECISION_API_STYLES
            else "ANTHROPIC_API_KEY"
            if state.discovery_style == "anthropic"
            else "OPENAI_API_KEY"
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
        state.api_style or state.default_style,
        state.endpoint,
        state.api_key_env,
        **(
            {"structured_output_mode": state.structured_output_mode}
            if state.structured_output_mode_set
            else {}
        ),
    )
    needs_key = (
        not state.yes
        and (
            state.approval != "none"
            or (
                not state.no_metadata
                and not state.model
                and not state.discovery_file
                and state.discovery_api_key_env is None
            )
        )
        and state.api_key_env
        and not os.environ.get(state.api_key_env)
    )
    asked = bool(state.prompt_key or needs_key)
    state.api_key = (
        prompts.prompt("API key (used only for this setup)", hide_input=True)
        if asked
        else os.environ.get(state.api_key_env)
        if state.api_key_env
        else None
    )
    if state.api_key and not asked and not announced:
        # A preset's variable is used silently otherwise; a stale key there
        # then looks like a broken provider.
        view.line(scrub_setup(state, f"Using the key in {state.api_key_env}."))
    return True


def select_model(state: WizardState) -> bool:
    """Select model; return False when the user cancels."""
    if state.editing is None:
        view.step(2, "Model")
    state.discovery_succeeded = None
    state.discovery_endpoint = None
    state.endpoint_models = ()
    state.endpoint_models_url = None
    state.listing_api_base = None
    if state.discovery_file:
        if state.api_style in connect.DECISION_API_STYLES and not state.models_url:
            raise click.UsageError("--discovery-file for decision routes requires --models-url")
        from ._connect_stages import read_discovery_listing

        listing = read_discovery_listing(
            state.discovery_file,
            state.endpoint,
            models_url=state.models_url,
            api_style=state.api_style,
        )
        state.endpoint_models = listing.models
        state.listing_api_base = listing.api_base
        state.endpoint_models_url = listing.models_url
        state.discovery_endpoint = state.endpoint
    if (
        not state.model
        and state.api_style in connect.DECISION_API_STYLES
        and not (state.models_url or state.discovery_file)
    ):
        # Unknown catalogue contract: never append /models to a decision POST URL.
        state.model = prompts.prompt("Exact decision model ID (Ctrl-C to cancel)")
    if not state.model and state.endpoint_models:
        state.model = prompts.prompt(
            "Search available models (type to search; Tab to select)",
            choices=[item["id"] for item in state.endpoint_models],
            searchable=True,
        )
    if state.no_metadata and not state.model:
        if state.yes:
            raise click.UsageError("--no-metadata requires an exact MODEL")
        state.model = prompts.prompt("Exact model ID (metadata is disabled; Ctrl-C to cancel)")
    while not state.model:
        view.note(scrub_setup(state, "Connecting to the server and listing models..."))
        try:
            from ._connect_stages import listing_key

            key = listing_key(
                state.models_url, state.endpoint, state.api_key, state.discovery_api_key_env
            )
            if (
                state.models_url == connect.CATALOGUE_URL
                or state.endpoint + "/models" == connect.CATALOGUE_URL
            ):
                key = None
            state.discovery_api_key = key
            found = asyncio.run(
                connect.discover(
                    state.endpoint,
                    api_style=state.api_style
                    if state.api_style in connect.DECISION_API_STYLES
                    else state.discovery_style,
                    api_key=key,
                    **(
                        {"models_url": state.models_url, "listing_style": state.discovery_style}
                        if state.models_url
                        else {}
                    ),
                )
            )
        except connect.DiscoveryError as exc:
            state.discovery_succeeded = False
            state.discovery_endpoint = state.endpoint
            view.note(scrub_setup(state, f"Could not list models: {exc}"), err=True)
            if exc.status_code in {401, 403}:
                view.line(
                    scrub_setup(
                        state,
                        "Model listing unavailable. Continue with an exact model ID or cancel.",
                    )
                )
            if state.yes:
                raise click.ClickException(
                    "Authentication failed. Supply an exact MODEL or check the listing key."
                    if exc.status_code in {401, 403}
                    else "Model listing unavailable. Supply an exact MODEL."
                ) from None
            state.model = prompts.prompt("Exact model ID (if known; Ctrl-C to cancel)")
        else:
            if not state.models_url and state.api_style not in connect.DECISION_API_STYLES:
                state.endpoint = found.api_base
            state.discovery_succeeded = True
            state.discovery_endpoint = state.endpoint
            state.endpoint_models = found.models
            state.listing_api_base = found.api_base
            state.endpoint_models_url = found.models_url
            names = [item["id"] for item in found.models]
            view.note(
                scrub_setup(
                    state, f"Found {len(names)} models. Type part of a name, then Tab to select."
                )
            )
            state.model = prompts.prompt(
                "Search available models (type to search; Tab to select)",
                choices=names,
                searchable=True,
            )
    return True


def select_request_interface(state: WizardState) -> bool:
    """Choose after the selected model/metadata, before any inference consent."""
    if not state.metadata_displayed:
        show_model_metadata(state)
    if state.api_style or state.yes or state.editing is not None:
        return True
    choices = ("discover", "chat", "responses", "anthropic", "systemone", "openai")
    # Use declared output type, not the model name, to suggest an interface.
    metadata = state.selected_endpoint_model or state.candidate or {}
    outputs = (metadata.get("architecture") or {}).get("output_modalities") or metadata.get(
        "output_modalities", []
    )
    decision_output = isinstance(outputs, list) and "decisions" in outputs and "text" not in outputs
    decision_hint = decision_output or any(
        metadata.get(field) == "decision" for field in ("model_type", "client_type")
    )
    suggested = (
        "systemone" if decision_hint and "systemone" in state.interface_endpoints else "discover"
    )
    choice = prompts.prompt(
        "Request interface",
        choices=choices,
        default=suggested,
        open_menu=True,
        labels={style: prompts.choice_label(style) for style in choices},
    )
    if choice == "discover":
        return True
    state.api_style = choice
    if choice in connect.DECISION_API_STYLES:
        preset_base = (
            connect.PROVIDERS[state.provider].api_base
            if state.provider in connect.PROVIDERS
            else None
        )
        mapped = state.interface_endpoints.get(choice)
        if mapped and (
            state.systemone_endpoint
            or not state.connection_endpoint_explicit
            or state.endpoint == preset_base
        ):
            selected_endpoint = mapped
        elif state.connection_endpoint_explicit:
            selected_endpoint = state.endpoint
        else:
            selected_endpoint = prompts.prompt(
                "Full decision request URL",
                default=mapped or state.endpoint,
                suggestions=state.server_urls,
            )
        selected_endpoint = connect.normalize_endpoint(selected_endpoint)
        if (state.api_key or state.api_key_env) and not state.share_interface_key:
            from urllib.parse import urlsplit

            def origin(url):
                address = urlsplit(url)
                return (
                    address.scheme,
                    address.hostname,
                    address.port or (443 if address.scheme == "https" else 80),
                )

            if origin(selected_endpoint) != origin(state.endpoint):
                raise click.UsageError(
                    "Changing the server origin requires --share-interface-key to reuse this key, "
                    "or supply the target directly with --endpoint and --api-style."
                )
        state.endpoint = selected_endpoint
        # The explicit shared catalogue retains descriptive selected-model facts.
        # Decision plans do not apply chat budgets/profiles from the listing.
        state.candidate = (
            connect.decision_metadata(state.model, state.candidate, state.selected_endpoint_model)
            or None
        )
        state.media_hints = state.candidate
    reject_decision_media(state)
    return True


def check_interfaces(state: WizardState) -> bool:
    """Check interfaces; return False when the user cancels."""
    if state.structured_output_mode_set and not state.api_style:
        raise click.UsageError(
            "--structured-output-mode requires an explicit Completion interface, not interface discovery"
        )
    reject_decision_media(state)
    view.step(3, "Connection checks")
    if state.api_style == "responses":
        view.line(scrub_setup(state, connect.ENCRYPTED_REASONING_EXPLANATION), dim=True)
    if not state.api_style:
        available = ("chat", "responses", "anthropic", *state.interface_endpoints)
        if state.approval != "none":
            state.interface_spent = 0
            retry_styles = available
            interface_timeout = 30
            while True:
                # Always test reasoning per interface, not just connectivity --
                # "nooa connect" with no arguments should run the best check we
                # have by default, not require the caller to already know to
                # ask for it. If the user gave us their own --reasoning-template
                # (e.g. for an Anthropic-only setup), that shape wins; "effort"
                # is a reasonable default elsewhere, understood by chat and
                # responses (and harmless if a style ignores/rejects it).
                reasoning_level = state.reasoning_level or (
                    state.levels.split(",")[0].strip() if state.levels else "medium"
                )
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
                            endpoints=state.interface_endpoints,
                            allow_cross_origin_credentials=state.share_interface_key,
                            timeout_seconds=interface_timeout,
                            reasoning_template=state.reasoning_template or "effort",
                            reasoning_level=reasoning_level,
                            reasoning_output_tokens=(
                                state.reasoning_output_tokens
                                or connect.DEFAULT_REASONING_OUTPUT_TOKENS
                            ),
                        ),
                        summary=False,
                        diagnostics=state.show_diagnostics,
                    )
                )
                state.interface_spent += state.interfaces.tokens_charged_to_budget
                state.interfaces = replace(
                    state.interfaces, tokens_charged_to_budget=state.interface_spent
                )
                available = state.interfaces.accepted
                if available:
                    if state.interfaces.recommended_style in available:
                        state.default_style = state.interfaces.recommended_style
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
                    scrub_setup(
                        state,
                        "Model listing succeeded, but model responses timed out; the route may be slow."
                        if slow
                        else "Connection check failed. Review the key, server URL and API format, then try again.",
                    ),
                    fg="yellow",
                )
                if state.show_diagnostics:
                    click.echo(
                        "Agent diagnostic prompt:\n"
                        + setup_diagnostic(
                            state,
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
                else:
                    view.line(
                        scrub_setup(
                            state, "For troubleshooting details, rerun with --show-diagnostics."
                        ),
                        dim=True,
                    )
                if state.yes:
                    raise click.ClickException(
                        "Check credentials and endpoint, or run without --yes to correct them interactively."
                    )
                remaining = max(0, state.budget_tokens - state.interface_spent)
                if remaining < state.output_tokens + 512:
                    view.line(
                        scrub_setup(
                            state,
                            "Check budget used up. Nothing was saved. Restart with a larger --budget-tokens value.",
                        )
                    )
                    return False
                view.line(scrub_setup(state, "Change the connection settings and try again."))
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
                    view.note(scrub_setup(state, "Setup cancelled. Nothing was saved."))
                    return False
                if action == "longer":
                    retry_styles = (
                        prompts.prompt(
                            "Interface to retry",
                            choices=tuple(state.interfaces.results),
                            labels={
                                style: prompts.choice_label(style)
                                for style in state.interfaces.results
                            },
                            default=state.default_style,
                        ),
                    )
                    interface_timeout = 120
                    continue
                retry_styles = ("chat", "responses", "anthropic", *state.interface_endpoints)
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
                                scrub_setup(
                                    state,
                                    "That variable is unset or empty. You can paste a temporary key instead.",
                                )
                            )
                            state.api_key = prompts.prompt(
                                "API key (used only for this setup)", hide_input=True
                            )
                elif action == "server":
                    state.interface_endpoints = {}  # Never carry decision routes to another server.
                    retry_styles = ("chat", "responses", "anthropic")
                    state.endpoint = connect.normalize_endpoint(
                        prompts.prompt(
                            "Model server URL",
                            default=state.endpoint,
                            suggestions=state.server_urls,
                            open_menu=True,
                        )
                    )
                    state.model = prompts.prompt("Exact model ID", default=state.model)
                    if state.guided:
                        state.candidate = state.media_hints = state.selected_endpoint_model = None
                        state.endpoint_models = ()
                        state.discovery_endpoint = state.endpoint_models_url = None
                        view.line(
                            scrub_setup(
                                state,
                                "Server/model changed; previous metadata no longer applies. Explicit settings are retained.",
                            )
                        )
            view.note(
                scrub_setup(
                    state,
                    "Working API formats: "
                    + ", ".join(prompts.choice_label(style) for style in available),
                )
            )
            view.note(
                scrub_setup(
                    state,
                    "Accepted client types: " + ", ".join(state.interfaces.accepted_client_types),
                )
            )
            observed = [s for s in available if state.interfaces.reasoning_observed.get(s)]
            view.note(
                scrub_setup(
                    state,
                    "API formats that returned reasoning: "
                    + (
                        ", ".join(prompts.choice_label(style) for style in observed)
                        if observed
                        else "none"
                    ),
                )
            )
        if state.interfaces and len(available) == 1:
            state.api_style = available[0]
            view.note(
                scrub_setup(
                    state, f"Using {prompts.choice_label(state.api_style)} for {state.model}."
                )
            )
        else:
            view.note(scrub_setup(state, f"Choose the request interface for {state.model}:"))
            state.api_style = prompts.prompt(
                "API format",
                choices=available,
                labels={style: prompts.choice_label(style) for style in available},
                default=state.default_style if state.default_style in available else available[0],
            )
        if state.api_style == "responses":
            view.line(scrub_setup(state, connect.ENCRYPTED_REASONING_EXPLANATION), dim=True)
        if state.interfaces:
            selected = state.interfaces.results[state.api_style].entry
            selected_endpoint = selected.get("endpoint") or selected["api_base"]
            if (
                state.api_style == "anthropic"
                and state.endpoint == state.discovery_endpoint
                and selected_endpoint == state.endpoint.removesuffix("/v1")
            ):
                # Messages adds /v1/messages at runtime: this is the same route,
                # not a new server. Keep its listing limits associated with it.
                # A shared models_url alone never authorizes unrelated routes.
                state.discovery_endpoint = selected_endpoint
            state.endpoint = selected_endpoint
        elif state.api_style in state.interface_endpoints:
            state.endpoint = state.interface_endpoints[state.api_style]
    # A default all-media choice made against a provisional chat route must be
    # reviewed explicitly when discovery selects an image-only decision route.
    if (
        state.guided
        and state.api_style in connect.DECISION_API_STYLES
        and any(
            kind != "image"
            for kind in (*(state.media_inputs or []), *(state.test_media_inputs or []))
        )
    ):
        view.line(
            scrub_setup(
                state,
                "The NOOA decision client currently supports image attachments only. Review audio, video and PDF selections before continuing.",
            )
        )
        action = prompts.labelled_choice(
            "Review media for the selected decision interface",
            (
                ("image", "Use only the selected image inputs/tests"),
                ("off", "Leave media inputs and tests off"),
                ("cancel", "Cancel without saving"),
            ),
            default="cancel",
        )
        if action == "cancel":
            if state.guided_media_explicit:
                reject_decision_media(state)
            return False
        state.media_inputs = (
            ["image"] if action == "image" and "image" in (state.media_inputs or []) else []
        )
        state.test_media_inputs = (
            ["image"] if action == "image" and "image" in (state.test_media_inputs or []) else None
        )
        state.inherit_media_inputs = False
        state.guided_media_revised = True
    # Corrections above are explicit; headless and known-style validation is unchanged.
    reject_decision_media(state)
    return True


def public_models_for_setup(state: WizardState):
    """Use a fetched public route record rather than repeating its GET."""
    public_listing = (
        state.endpoint_models_url == connect.CATALOGUE_URL
        or state.models_url == connect.CATALOGUE_URL
    )
    if state.discovery_succeeded is False and (
        public_listing or state.endpoint + "/models" == connect.CATALOGUE_URL
    ):
        return []
    if (
        public_listing
        and state.endpoint_models is not None
        and state.discovery_endpoint == state.endpoint
        and not any(
            record["schema"] == "openrouter.models"
            for record in (state.server_details.records if state.server_details else ())
        )
    ):
        from nooa.unifiedllm.connect._server_details import openrouter_facts

        return [openrouter_facts(item.get("public_model", item)) for item in state.endpoint_models]
    if state.server_details is not None and any(
        record["schema"] == "openrouter.models" for record in state.server_details.records
    ):
        return (
            [state.server_details.catalogue_model] if state.server_details.catalogue_model else []
        )
    return asyncio.run(connect.catalogue())


def prepare_metadata(state: WizardState) -> bool:
    """Fetch allowed metadata before media choice, without settings prompts."""
    state.metadata_prepared = True
    state.selected_endpoint_model = (
        next(
            (item for item in (state.endpoint_models or ()) if item.get("id") == state.model), None
        )
        if state.endpoint == state.discovery_endpoint
        else None
    )
    # An explicit --as can reuse its previous evidence. Otherwise checks use
    # a temporary label; the user names the entry only when ready to save.
    state.existing = state.editing or (
        state.data.get("models", {}).get(state.alias) if state.alias else None
    )
    state.candidate = None
    state.edited_settings = False
    state.server_details = None
    if not state.no_metadata and not state.no_catalogue and state.editing is None:
        from ._connect_stages import listing_key

        state.discovery_api_key = listing_key(
            state.models_url, state.endpoint, state.api_key, state.discovery_api_key_env
        )
        listing = (
            connect.Discovery(
                state.listing_api_base or state.endpoint,
                tuple(state.endpoint_models),
                state.endpoint_models_url,
            )
            if state.endpoint == state.discovery_endpoint and state.discovery_succeeded is not False
            else None
        )
        state.server_details = (
            connect.ModelDetails(state.model, {"id": state.model})
            if state.discovery_succeeded is False and state.endpoint == state.discovery_endpoint
            else asyncio.run(
                connect.server_details(
                    state.endpoint,
                    state.model,
                    api_style=state.api_style or state.discovery_style or "chat",
                    api_key=state.api_key,
                    listing_api_key=state.discovery_api_key,
                    models_url=state.models_url,
                    listing_style=state.discovery_style,
                    listing=listing,
                    litellm=state.server_metadata == "litellm" or state.provider == "hub",
                    catalogue_id=state.catalogue_model,
                )
            )
        )
        if state.server_details.records:
            state.selected_endpoint_model = state.server_details.model
            if listing is None:
                state.endpoint_models = (state.selected_endpoint_model,)
                state.discovery_endpoint = state.endpoint
                state.endpoint_models_url = state.server_details.records[0]["source"]
    if state.api_style in connect.DECISION_API_STYLES:
        if state.no_catalogue and state.catalogue_model:
            raise click.UsageError("--catalogue-model cannot be used with --no-catalogue")
        if not state.no_catalogue and state.editing is None:
            from ._connect_stages import listing_key

            # Validate explicit credential configuration before any public lookup.
            listing_key(
                state.models_url, state.endpoint, state.api_key, state.discovery_api_key_env
            )
            try:
                models = public_models_for_setup(state)
            except (httpx.HTTPError, TimeoutError, ValueError, KeyError):
                state.public_details_status = "Unavailable"
                models = []
            selected_id = state.catalogue_model or state.model
            matches = [item for item in models if item.get("id") == selected_id]
            if len(matches) > 1:
                raise click.ClickException("Requested catalogue model must match exactly once.")
            if len(matches) == 1:
                state.candidate = matches[0]
            else:
                if not state.yes and models and not state.catalogue_model:
                    selected = prompts.prompt(
                        "Explicit OpenRouter match (blank continues without details)",
                        default="",
                        show_default=False,
                        choices=[""] + [item["id"] for item in models],
                        searchable=True,
                    )
                    if selected:
                        matches = [item for item in models if item["id"] == selected]
                        if len(matches) != 1:
                            raise click.ClickException("Choose one explicit OpenRouter model ID.")
                        state.candidate = matches[0]
            # Keep the same bounded facts for the suggestion UI and library plan.
            if state.candidate is not None:
                state.candidate = connect.decision_metadata(state.model, state.candidate)
                state.candidate.setdefault("id", selected_id)
        state.media_hints = connect.decision_metadata(
            state.model, state.candidate, state.selected_endpoint_model
        )
        if state.media_hints.get("media_source") == "endpoint listing":
            state.media_hints["media_source"] = state.endpoint_models_url or "endpoint listing"
        show_model_metadata(state)
        return True
    if state.editing is not None:
        state.candidate = {
            "id": state.editing.get("underlying_model", state.model),
            "context_length": state.editing.get("context_window"),
            # Saved entries keep settings only, not the catalogue limits, and an
            # edit does not look the model up, so its ceiling is known here only
            # if the entry saved max_output_tokens. The saved cap is kept as is.
            "top_provider": {"max_completion_tokens": state.editing.get("max_output_tokens")},
            "reasoning": {
                "supported_efforts": list(state.editing.get("reasoning_levels", {})),
                "default_effort": state.editing.get("reasoning_default"),
            },
        }
        state.edited_settings = True
    if state.no_catalogue and state.catalogue_model:
        raise click.UsageError("--catalogue-model cannot be used with --no-catalogue")
    if not state.no_catalogue:
        if state.editing is None and not state.endpoint_models:
            # Validate explicit catalogue credential configuration before any lookup.
            from ._connect_stages import listing_key

            state.discovery_api_key = listing_key(
                state.models_url, state.endpoint, state.api_key, state.discovery_api_key_env
            )
        try:
            models = public_models_for_setup(state)
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError):
            state.public_details_status = "Unavailable"
            models = []
        matches = (
            [item for item in models if item.get("id") == state.catalogue_model]
            if state.catalogue_model
            else [item for item in models if item.get("id") == state.model]
        )
        if len(matches) == 1:
            state.candidate = matches[0]
        elif matches:
            view.note(
                scrub_setup(
                    state, "Possible catalogue models: " + ", ".join(item["id"] for item in matches)
                )
            )
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
            # An exact/suffix match found nothing — likely a gateway prefix
            # (aws/..., bedrock-..., vertex/...) the catalogue never records.
            # Never auto-select a fuzzy guess; only offer it for confirmation.
            fuzzy = (
                []
                if state.yes or state.catalogue_model
                else connect.fuzzy_match_models(state.model, models)
            )
            if fuzzy:
                view.note(
                    scrub_setup(
                        state,
                        "Model not found; did you mean one of these? "
                        + ", ".join(item["id"] for item in fuzzy),
                    )
                )
                selected = prompts.prompt(
                    "Catalogue model (blank leaves it unknown)",
                    default="",
                    show_default=False,
                    choices=[""] + [item["id"] for item in fuzzy],
                )
                if selected:
                    state.candidate = next((item for item in fuzzy if item["id"] == selected), None)
                    if state.candidate is None:
                        raise click.ClickException("Choose one of the displayed model IDs.")
    if (
        state.candidate is not None
        and (
            state.endpoint + "/models" != connect.CATALOGUE_URL
            or state.candidate.get("id") != state.model
        )
        and state.editing is None
    ):
        # OpenRouter limits/media fill gaps, but its route-specific reasoning
        # profiles/default request settings do not configure a different server.
        public_facts = deepcopy(state.candidate)
        state.candidate = deepcopy(state.candidate)
        state.candidate.pop("reasoning", None)
        state.candidate.pop("default_parameters", None)
        state.candidate.pop("pricing", None)
        state.candidate.pop("top_provider", None)
        if isinstance(state.candidate.get("limit_sources"), dict):
            state.candidate["limit_sources"].pop("max_completion_tokens", None)
        if state.server_details is not None:
            state.server_details = replace(
                state.server_details,
                records=(
                    *state.server_details.records,
                    {
                        "source": connect.CATALOGUE_URL,
                        "schema": "openrouter.fallback",
                        "id": public_facts.get("id"),
                        "selected_id": state.model,
                        "facts": public_facts,
                    },
                ),
            )
    endpoint_model = state.selected_endpoint_model
    state.selected_endpoint_model = endpoint_model
    state.media_hints = deepcopy(state.candidate)
    if state.media_hints is not None and state.editing is None:
        state.media_hints.setdefault("media_source", connect.CATALOGUE_URL)
    if endpoint_model is not None and state.editing is None:
        combined = connect.model_metadata(state.model, state.candidate, endpoint_model)
        state.media_hints = combined
        state.media_hints.setdefault(
            "media_source", connect.CATALOGUE_URL if state.candidate else "endpoint listing"
        )
        if combined.get("endpoint_limits"):
            state.candidate = combined
    show_model_metadata(state)
    return True


def show_model_metadata(state: WizardState):
    """One selected-model summary, before interface/media choices and model calls."""
    state.metadata_displayed = True
    model = state.media_hints or state.candidate or state.selected_endpoint_model or {}
    report = connect.advertised_media_suggestions(model)
    endpoint = state.selected_endpoint_model or {}
    view.section("Model details")
    rows = [("Model", state.model)]
    records = state.server_details.records if state.server_details else ()
    server_facts = any(
        record.get("schema") != "openrouter.fallback" and len(record.get("facts") or {}) > 1
        for record in records
    ) or any(key != "id" for key in endpoint)
    if state.edited_settings:
        rows.append(("Source", "Your edits"))
    elif server_facts:
        rows.append(("Source", "This server"))
        if state.candidate and (
            "catalogue" in (state.candidate.get("limit_sources") or {}).values()
            or model.get("media_source") == connect.CATALOGUE_URL
        ):
            rows.append(("Additional details", "OpenRouter"))
    elif (
        report["advertised"]
        and model.get("media_source") == connect.CATALOGUE_URL
        or state.candidate
        and any(
            key not in {"id", "limit_sources", "media_source", "media_model_id"}
            for key in state.candidate
        )
    ):
        rows.append(("Source", "OpenRouter"))
    else:
        rows.append(
            (
                "Status",
                "Details not requested"
                if state.no_catalogue or state.no_metadata
                else "No details listed",
            )
        )
    if state.public_details_status:
        rows.append(("OpenRouter", state.public_details_status))
    for record in records:
        product = record.get("product") or {}
        if product.get("title") == "LiteLLM API":
            rows.append(
                ("Server", "LiteLLM" + (" " + product["version"] if product.get("version") else ""))
            )
            break
    for key, label in (("display_name", "Name"), ("description", "Description")):
        value = endpoint.get(key) or model.get(key)
        if value:
            rows.append((label, value))
    outputs = (endpoint.get("architecture") or {}).get("output_modalities") or endpoint.get(
        "output_modalities", []
    )
    if (isinstance(outputs, list) and "decisions" in outputs and "text" not in outputs) or any(
        endpoint.get(field) == "decision" for field in ("model_type", "client_type")
    ):
        rows.append(("Model type", "Decision"))
    interfaces = endpoint.get("supported_interfaces") or []
    if interfaces:
        rows.append(
            ("Interfaces listed", ", ".join(prompts.choice_label(style) for style in interfaces))
        )
    report = connect.advertised_media_suggestions(model)
    if report["advertised"]:
        rows.append(
            ("Media listed", ", ".join(view.media_label(kind) for kind in report["advertised"]))
        )
        source = model.get("media_source")
        if source:
            rows.append(
                (
                    "Media source",
                    "OpenRouter"
                    if source == connect.CATALOGUE_URL
                    else "This server"
                    if source == "endpoint listing" or source == state.endpoint_models_url
                    else source,
                )
            )
    if (
        state.candidate
        and state.api_style not in connect.DECISION_API_STYLES
        and not any(value == "Decision" for key, value in rows if key == "Model type")
    ):
        rows.extend(
            view.model_details(
                state.candidate,
                output_tokens=state.output_tokens,
                edited=bool(state.edited_settings),
                heading=False,
                attribution=False,
                render=False,
            )
        )
    view.fields(scrub_setup(state, rows))
    if not report["advertised"]:
        view.note("No media is listed.")
    if state.server_details:
        for notice in state.server_details.notices:
            view.warning(scrub_setup(state, notice))


def configure_metadata(state: WizardState) -> bool:
    """Review settings after media choice, retaining the existing prompt order."""
    if not state.metadata_prepared:
        if not prepare_metadata(state):
            return False
    if state.api_style in connect.DECISION_API_STYLES:
        return True
    if state.editing is not None and not state.yes:
        state.candidate = prompts.edit_model_details(state.candidate)
    if state.candidate is not None:
        while True:
            if state.edited_settings:
                view.model_details(
                    scrub_setup(state, state.candidate),
                    output_tokens=state.output_tokens,
                    edited=True,
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
                view.note(scrub_setup(state, "Setup cancelled. Nothing saved."))
                return False
            if action == "skip":
                if state.editing is not None:
                    view.note(scrub_setup(state, "Edits discarded. Nothing saved."))
                    return False
                state.candidate = None
                state.edited_settings = False
                view.note(
                    scrub_setup(
                        state,
                        "Continuing without the published model settings. Explicit command-line settings still apply.",
                    )
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


def reject_decision_media(state: WizardState) -> None:
    """Validate route-local controls once the request style is known."""
    if state.structured_output_mode_set and state.api_style not in {None, "chat", "anthropic"}:
        raise click.UsageError("--structured-output-mode requires Completion (chat or anthropic)")
    if state.api_style in connect.DECISION_API_STYLES:
        if state.cache_enabled is not None:
            raise click.UsageError(
                "--no-cache does not apply to decision models; they do not configure chat caching"
            )
        from nooa.decisions.media import decision_media_inputs

        try:
            decision_media_inputs(prompts.parse_media_inputs(state.media_inputs))
            decision_media_inputs(prompts.parse_media_inputs(state.test_media_inputs))
        except ValueError as exc:
            raise click.UsageError(str(exc)) from None


def configure_media_inputs(state: WizardState) -> bool:
    """Declaration belongs to this connection, not optional published metadata."""
    from nooa.unifiedllm.media_inputs import validate_media_inputs

    from ._connect_advertised_media import offer_suggestions

    if state.api_style in connect.DECISION_API_STYLES:
        reject_decision_media(state)
        previous_entry = state.editing or (
            (state.data or {}).get("models", {}).get(state.alias) if state.alias else None
        )
        route = connect.plan(
            state.alias or "candidate",
            state.model,
            state.api_style,
            state.endpoint,
            state.api_key_env or "",
            existing_entry=previous_entry,
            decision_image_transport=state.decision_image_transport,
            typesafe_danger_zone_images=state.typesafe_danger_zone_images,
        ).entry
        previous = route.get("media_inputs", [])
        state.decision_image_transport = route["decision_image_transport"]
        state.typesafe_danger_zone_images = route["typesafe_danger_zone_images"]
        if offer_suggestions(state, previous):
            return True
        if state.test_media_inputs is not None:
            state.test_media_inputs = prompts.parse_media_inputs(state.test_media_inputs)
        if state.media_inputs is not None:
            state.media_inputs = prompts.parse_media_inputs(state.media_inputs)
        elif state.test_media_inputs:
            state.media_inputs = list(previous)
            state.inherit_media_inputs = True
        elif state.yes or state.no_probe or state.approval != "all" or state.budget_tokens < 2048:
            state.media_inputs = list(previous)
            state.inherit_media_inputs = True
        else:
            mode = prompts.choose_decision_images(previous)
            state.media_inputs = (
                list(previous) if mode == "keep" else ["image"] if mode == "manual" else []
            )
            state.test_media_inputs = ["image"] if mode == "test" else None
        if (
            state.api_style == "systemone"
            and (state.media_inputs or state.test_media_inputs)
            and not state.yes
        ):
            view.line(
                scrub_setup(
                    state,
                    "TypeSafe multipart is selected, but experimental image opt-in is missing; image requests are blocked until you confirm it."
                    if state.decision_image_transport == "typesafe_multipart"
                    and not state.typesafe_danger_zone_images
                    else "Image testing isn't available with System One JSON. Use direct TypeSafe multipart with experimental image requests, or leave images off."
                    if state.decision_image_transport == "json"
                    else "TypeSafe multipart image requests require the explicit confirmations below.",
                )
            )
            if prompts.confirm("Use direct TypeSafe multipart for images?", default=False):
                state.decision_image_transport = "typesafe_multipart"
                state.typesafe_danger_zone_images = prompts.confirm(
                    "Opt into experimental TypeSafe image requests?", default=False
                )
        view.line(
            scrub_setup(
                state,
                "The NOOA decision client currently supports image attachments only. Use Chat for audio, video or PDFs.",
            )
        )
        return True
    if state.media_inputs is not None:
        previous = ()
        try:
            previous = validate_media_inputs((state.editing or {}).get("media_inputs"))
        except ValueError:
            pass  # An explicit choice can repair an invalid legacy declaration.
        offer_suggestions(state, previous)
        state.media_inputs = prompts.parse_media_inputs(state.media_inputs)
        if state.test_media_inputs is not None:
            state.test_media_inputs = prompts.parse_media_inputs(state.test_media_inputs)
        return True
    previous_entry = state.editing or (
        (state.data or {}).get("models", {}).get(state.alias) if state.alias else None
    )
    previous = ()
    if previous_entry and state.model and state.api_style:
        route = connect.plan(
            state.alias or "candidate",
            state.model,
            state.api_style,
            state.endpoint,
            state.api_key_env or "",
            existing_entry=previous_entry,
        ).entry
        previous = validate_media_inputs(route.get("media_inputs"))
    elif state.editing:
        previous = validate_media_inputs(state.editing.get("media_inputs"))
    if offer_suggestions(state, previous):
        return True
    if state.test_media_inputs is not None:
        state.test_media_inputs = prompts.parse_media_inputs(state.test_media_inputs)
    elif not state.yes and state.approval == "all":
        mode = prompts.choose_media_mode(editing=bool(state.editing))
        if mode == "test":
            state.test_media_inputs = ["image", "audio", "video", "pdf"]
        else:
            state.media_inputs = prompts.choose_media_inputs(previous)
            return True
    if state.test_media_inputs:
        # Retesting is evidence collection, not permission revocation.
        # Resolve again after model/interface selection; this list is the menu
        # preview, not permission to forward to a subsequently changed route.
        state.inherit_media_inputs = True
        state.media_inputs = list(previous)
        if not state.guided:
            view.line(scrub_setup(state, "Media tests: " + ", ".join(state.test_media_inputs)))
    else:
        state.inherit_media_inputs = bool(state.yes)
        state.media_inputs = list(previous) if state.yes else prompts.choose_media_inputs(previous)
    return True


def configure_checks(state: WizardState) -> bool:
    """Configure checks; return False when the user cancels."""
    reject_decision_media(state)
    profile = connect.plan(
        state.alias or "candidate",
        state.model,
        state.api_style,
        state.endpoint,
        state.api_key_env,
        existing_entry=state.editing or state.existing,
        **(
            {"structured_output_mode": state.structured_output_mode}
            if state.structured_output_mode_set
            else {}
        ),
        decision_image_transport=state.decision_image_transport,
        typesafe_danger_zone_images=state.typesafe_danger_zone_images,
    ).entry
    mode_kwargs = (
        {"structured_output_mode": profile["structured_output_mode"]}
        if "structured_output_mode" in profile
        else {}
    )
    if state.api_style in connect.DECISION_API_STYLES and any(
        (state.levels_file, state.levels, state.reasoning_template, state.reply_tokens)
    ):
        raise click.UsageError("Decision models do not use reasoning-level or reply-token options")
    if state.media_inputs is None or state.inherit_media_inputs:
        previous = connect.plan(
            state.alias or "candidate",
            state.model,
            state.api_style,
            state.endpoint,
            state.api_key_env,
            existing_entry=state.existing,
            decision_image_transport=state.decision_image_transport,
            typesafe_danger_zone_images=state.typesafe_danger_zone_images,
        ).entry
        state.media_inputs = previous.get("media_inputs", [])
    if state.levels_file and (state.levels or state.reasoning_template):
        raise click.UsageError("Use either --levels-file or --reasoning-template with --levels.")
    state.patches = deepcopy(state.edited_level_patches)
    if state.editing is not None and state.api_style not in connect.DECISION_API_STYLES:
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
        catalogue_settings=bool(state.edited_settings),
        endpoint_model=state.selected_endpoint_model,
        endpoint_models_url=state.endpoint_models_url,
        media_inputs=state.media_inputs,
        test_media_inputs=state.test_media_inputs,
        decision_image_transport=state.decision_image_transport,
        typesafe_danger_zone_images=state.typesafe_danger_zone_images,
        reasoning_levels=state.patches,
        budget_tokens=state.budget_tokens,
        output_tokens=state.output_tokens,
        reasoning_output_tokens=state.reasoning_output_tokens,
        existing_entry=state.interfaces.results[state.api_style].entry
        if state.interfaces
        else state.existing,
        session_checks=state.approval == "all"
        and state.api_style not in connect.DECISION_API_STYLES,
        reply_tokens=state.reply_tokens,
        cache_enabled=state.cache_enabled,
        **mode_kwargs,
    )
    if state.server_details is not None:
        state.proposal.entry["provenance"]["server_details"] = scrub_setup(
            state, {"id": state.model, "records": list(state.server_details.records)}
        )
    if state.interfaces:
        state.proposal.entry["provenance"]["interface_endpoint"] = deepcopy(
            state.interfaces.results[state.api_style]
            .entry["provenance"]
            .get("interface_endpoint", {})
        )
    if "advertised_model" in state.proposal.entry.get("provenance", {}) and state.models_url:
        state.proposal.entry["provenance"]["models_url"] = state.models_url
    if state.editing is not None:
        # Preserve transport controls, custom parameters and exact level blocks.
        merged = deepcopy(state.editing)
        for field in (
            "structured_output_mode",
            "media_inputs",
            "decision_image_transport",
            "typesafe_danger_zone_images",
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
        if state.editing.get("structured_output_mode") != merged.get("structured_output_mode"):
            for field in ("session_checks", "requests_accepted", "reasoning_observed"):
                merged["provenance"].pop(field, None)
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
    if (
        not state.guided
        and state.api_style not in connect.DECISION_API_STYLES
        and "context_window" not in state.proposal.entry
    ):
        view.note(
            scrub_setup(
                state,
                "Context limit unknown; NOOA will use its default. Set --context-window if you know the server limit.",
            )
        )
    state.configured = connect.configure_entry(
        state.proposal.entry,
        reply_tokens=state.reply_tokens,
        cache_enabled=state.cache_enabled,
        **mode_kwargs,
    )
    if state.api_style in connect.DECISION_API_STYLES:
        state.proposal = connect.refresh_plan(replace(state.proposal, entry=state.configured))
        state.remaining_estimate = state.proposal.token_estimate
        if not state.guided:
            view.line(scrub_setup(state, "Decision calls have no reply limit."))
        return True
    if state.reply_tokens is None and not state.yes and not state.guided:
        output_ceiling = (
            state.configured["provenance"].get("catalogue_limits", {}).get("max_completion_tokens")
        )
        if not (
            isinstance(output_ceiling, int)
            and not isinstance(output_ceiling, bool)
            and output_ceiling > 0
        ):
            output_ceiling = None
        bounds = [
            v
            for v in (
                state.configured["context_window"] - 1
                if state.configured.get("context_window")
                else None,
                output_ceiling,
            )
            if isinstance(v, int) and v > 0
        ]
        chosen_cap = prompts.choose_reply_limit(
            state.configured["max_tokens"],
            min(bounds) if bounds else None,
            output_ceiling=output_ceiling,
            source=state.configured["provenance"]["reply_limit"]["source"],
        )
        state.configured = connect.configure_entry(state.configured, reply_tokens=chosen_cap)
    state.proposal = connect.refresh_plan(replace(state.proposal, entry=state.configured))
    state.remaining_estimate = state.proposal.token_estimate
    if not state.guided:
        view.line(
            scrub_setup(
                state,
                f"Saved reply limit: {state.configured['max_tokens']:,} tokens, shared by reasoning and the answer.",
            )
        )
    for label, limit in state.configured["provenance"].get("level_reply_limits", {}).items():
        view.line(
            scrub_setup(
                state,
                f"Reasoning level {label}: reply limit raised to {limit['value']:,} tokens to leave room for its thinking budget.",
            ),
            fg="yellow",
        )
    return True


def baseline_allows_save(state: WizardState) -> bool:
    """Fail closed for approved checks; manual setup is explicitly unverified."""
    if state.no_probe or state.approval == "none":
        return True
    entry = state.result.entry if state.result is not None else {}
    record = entry.get("provenance", {}).get("probes", {}).get("routing", {})
    if state.api_style in connect.DECISION_API_STYLES:
        # Decision models have no text baseline: the accepted decision check is the gate.
        if connect.check_status("routing", record) == "passed":
            return True
    elif text_baseline_passed(record):
        return True
    reason = view.connection_detail(record)
    # Only sanitized display facts: never echo raw provider errors or replies.
    view.note(
        scrub_setup(
            state,
            "Nothing was saved. Connection: " + reason.rstrip(".") + ". "
            "Review the server URL and settings, or increase --budget-tokens if checks were skipped. "
            "Rerun with --no-probe to save without testing. "
            "Add --no-catalogue and an explicit model ID to avoid listing/metadata requests. "
            "See docs/model-configuration.md. No model entry or key was written.",
        ),
        err=True,
    )
    return False


def run_checks(state: WizardState) -> bool:
    """Run checks; return False when the user cancels."""
    if state.api_style in connect.DECISION_API_STYLES:
        checks = ["decision connection", "boolean, choice and score answers"]
        if state.test_media_inputs:
            checks.append("image access and sample color")
    else:
        checks = ["connection", "tools"]
        if state.proposal.entry.get("reasoning_levels"):
            checks.append("reasoning")
        if any(p.name.startswith("media:") for p in state.proposal.probes):
            checks.append("media inputs")
    if state.proposal.session_checks:
        checks.append("conversation")
    view.line(
        scrub_setup(state, "Checking " + ", ".join(checks[:-1]) + " and " + checks[-1] + "…"),
        fg="bright_cyan",
        bold=True,
    )
    if state.remaining_estimate > state.proposal.budget_tokens:
        view.note(
            scrub_setup(
                state, "Some tests will be skipped. Increase --budget-tokens to run them all."
            ),
            err=True,
        )
    state.result = asyncio.run(
        show_checks(
            connect.run_steps(state.proposal, approved=state.approval, api_key=state.api_key),
            reasoning_levels=state.proposal.entry.get("reasoning_levels"),
            media_inputs=state.media_inputs,
            diagnostics=state.show_diagnostics,
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
        view.note(
            scrub_setup(
                state,
                "Warning: setup is incomplete; check budget used up before "
                + ", ".join(view.check_label(name) for name in skipped)
                + ". These settings have not been checked.",
            ),
            err=True,
        )
    if state.test_media_inputs:
        facts = state.result.entry["provenance"]["probes"]
        enabled = list(state.media_inputs)
        for kind in state.test_media_inputs:
            if facts.get(f"media:{kind}", {}).get("outcome") == "passed" and kind not in enabled:
                enabled.append(kind)
        state.result.entry["media_inputs"] = enabled
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
        if state.show_diagnostics:
            click.echo(
                "Agent diagnostic prompt:\n"
                + setup_diagnostic(
                    state,
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
        else:
            view.line(
                scrub_setup(state, "For troubleshooting details, rerun with --show-diagnostics."),
                dim=True,
            )
    if state.unobserved:
        view.note(
            scrub_setup(
                state,
                "Warning: no reasoning information was returned for: "
                + ", ".join(state.unobserved)
                + ". Reasoning may still be active. Try another API format or review the server settings.",
            ),
            err=True,
        )
    return baseline_allows_save(state)


def save_model(state: WizardState) -> bool:
    """Save model; return False when canceled or the baseline is unverified."""
    if not baseline_allows_save(state):
        return False
    view.step(4, "Save model")
    if state.no_probe or state.approval == "none":
        view.line(scrub_setup(state, "Connection: not tested (manual setup)"), fg="yellow")
    if state.api_style in connect.DECISION_API_STYLES:
        view.line(
            scrub_setup(
                state,
                "Image requests: "
                + (
                    "OpenAI Decisions"
                    if state.api_style == "openai"
                    else "TypeSafe multipart"
                    if state.result.entry["decision_image_transport"] == "typesafe_multipart"
                    else "System One JSON"
                )
                + (
                    " · experimental images on"
                    if state.result.entry["typesafe_danger_zone_images"]
                    else ""
                ),
            )
        )
    if "media_inputs" in state.result.entry:
        view.media_settings(state.result.entry)
    if state.test_media_inputs:
        facts = state.result.entry["provenance"]["probes"]
        unavailable = [
            kind
            for kind in state.test_media_inputs
            if facts.get(f"media:{kind}", {}).get("outcome") == "unavailable"
        ]
        if unavailable:
            view.line(scrub_setup(state, "Not available in this test: " + ", ".join(unavailable)))
            view.line(
                scrub_setup(
                    state,
                    "Keep these settings, or choose another model or API format to try the unavailable inputs.",
                ),
                dim=True,
            )
    tested_media_inputs = list(state.result.entry.get("media_inputs", []))
    if not state.yes and state.test_media_inputs:
        action = prompts.review_media_inputs(state.result.entry["media_inputs"])
        if action == "manual":
            view.line(
                scrub_setup(state, "Inputs enabled manually will be saved as not tested."),
                fg="yellow",
            )
            state.result.entry["media_inputs"] = prompts.choose_media_inputs(
                state.result.entry["media_inputs"]
            )
    if tested_media_inputs != state.result.entry.get("media_inputs", []):
        view.media_settings(state.result.entry)
    if state.test_media_inputs:
        facts = state.result.entry["provenance"]["probes"]
        for kind in state.test_media_inputs:
            record = facts.get(f"media:{kind}", {})
            if record.get("outcome") in {"inconclusive", "rejected", "unavailable"} and (
                kind in tested_media_inputs
            ) != (kind in state.result.entry["media_inputs"]):
                headline = view.check_outcome("media:" + kind, record, "attention")
                view.progress_result(
                    "–" if headline == "Not available" else "!",
                    view.check_label("media:" + kind),
                    view.media_test_detail(
                        kind, record, media_inputs=state.result.entry["media_inputs"]
                    ),
                    headline=headline,
                )
    # The effective declarations above are reviewed before the final save decision.
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
            view.note(scrub_setup(state, "Enter a non-empty model name."), err=True)
    if state.alias in state.data.get("models", {}):
        view.note(
            scrub_setup(
                state, f"Warning: saving will overwrite model {state.alias!r} in {state.path}."
            ),
            err=True,
        )
        if not state.yes and not prompts.confirm("Replace this model?", default=False):
            return False
    state.result = replace(state.result, alias=state.alias)
    view.line(
        scrub_setup(
            state, f"{state.alias} · {state.model} · {prompts.choice_label(state.api_style)}"
        ),
        bold=True,
    )
    if state.show_config:
        preview = scrub_setup(state, {"models": {state.alias: state.result.entry}})
        click.echo(yaml.safe_dump(preview, sort_keys=False))
    else:
        view.line(
            scrub_setup(state, "Use --show-config to preview the full YAML settings."),
            dim=True,
        )
    if shadow := shadowing_source(state.alias, state.path, extra_priority=bool(state.working_dir)):
        view.line(
            scrub_setup(
                state,
                f"Warning: {shadow} currently defines this alias and takes precedence over this destination. Update that file or explicitly load {state.path} to use this entry.",
            ),
            fg="yellow",
        )
    if state.api_style in connect.DECISION_API_STYLES:
        view.line(scrub_setup(state, f"Save summary: {prompts.choice_label(state.api_style)}"))
    else:
        view.line(
            scrub_setup(
                state,
                f"Save summary: {prompts.choice_label(state.result.entry['api_style'])} · reply limit {state.result.entry['max_tokens']:,} tokens",
            )
        )
        view.line(
            scrub_setup(
                state,
                "Reasoning levels: "
                + (", ".join(state.result.entry.get("reasoning_levels", {})) or "none configured")
                + "; default: "
                + str(state.result.entry.get("reasoning_default", "server default")),
            )
        )
    if state.api_style not in connect.DECISION_API_STYLES and (
        "cache_breakpoint" in state.result.entry and state.result.entry["cache_breakpoint"] is None
    ):
        view.fields([("Caching", "Off")])
    view.fields(scrub_setup(state, structured_output_rows(state.result.entry)))
    if state.yes or prompts.confirm(f"Write model entry to {state.path}?", default=True):
        save_key = False
        if (
            state.api_key
            and state.api_key_env
            and state.api_key != os.environ.get(state.api_key_env)
        ):
            secrets_path = paths.get_user_dir("secrets.yaml")
            view.line(
                scrub_setup(
                    state,
                    f"This setup used a new key. It can be saved in {secrets_path} as plain text (not encrypted), readable only by your account.",
                )
            )
            save_key = not state.yes and prompts.confirm(
                "Save this key for future NOOA runs?", default=True
            )
            if save_key:
                from nooa.secrets import write_secret_env

                write_secret_env(secrets_path, state.api_key_env, state.api_key)
                view.line(
                    scrub_setup(
                        state,
                        f"Saved key as {state.api_key_env}. Existing secret values are preserved.",
                    )
                )
                if os.environ.get(state.api_key_env):
                    view.line(
                        scrub_setup(
                            state,
                            f"Your current environment still overrides this file. Unset or update {state.api_key_env} before starting NOOA again.",
                        ),
                        fg="yellow",
                    )
        connect.write(state.result.entry, state.path, alias=state.alias)
        view.section("Saved model")
        view.fields(scrub_setup(state, [("Alias", state.alias), ("File", state.path)]))
        view.media_settings(state.result.entry)
        view.fields(scrub_setup(state, structured_output_rows(state.result.entry)))
        if state.api_style in connect.DECISION_API_STYLES:
            view.line(
                scrub_setup(
                    state,
                    f"API format: {prompts.choice_label(state.api_style)} · decision model: {state.result.entry['model_name']}",
                )
            )
        if state.api_style not in connect.DECISION_API_STYLES:
            view.line(
                scrub_setup(
                    state,
                    f"API format: {prompts.choice_label(state.api_style)} · reply limit: {state.result.entry['max_tokens']:,} tokens",
                )
            )
        if state.api_key and not save_key and state.api_key != os.environ.get(state.api_key_env):
            view.note(
                scrub_setup(
                    state,
                    f"The key was not saved. Set {state.api_key_env} (or add it to your NOOA secrets file) before using this alias.",
                )
            )
        view.note(
            scrub_setup(
                state,
                f'Use it in Python: get_decision_model("{state.alias}")'
                if state.api_style in connect.DECISION_API_STYLES
                else f'Use it in Python: get_llm_client("{state.alias}")',
            )
        )
        if state.output and not state.working_dir:
            view.note(
                scrub_setup(
                    state,
                    "For a custom path, include it in NEMO_OO_LLM_CONFIG or reload_registry(path).",
                )
            )
    return True


def structured_output_rows(entry):
    """Declaration and observed wire facts are different, bounded review fields."""
    if "structured_output_mode" not in entry:
        return []
    mode = entry["structured_output_mode"]
    rows = [("Structured output", f"{mode or 'none'} (declared)")]
    if mode == "anthropic":
        baseline = entry.get("provenance", {}).get("probes", {}).get("routing", {})
        observed = (
            baseline.get("structured_output_schema_observed") is True
            and baseline.get("structured_output_conflicts_observed") is False
            and baseline.get("settings_sent") is True
        )
        rows.append(
            ("Native carrier", "output_config.format observed" if observed else "Not confirmed")
        )
    return rows


def show_guided_proposal(state: WizardState, *, deferred_style=False, updated=False):
    """One concise summary for each current proposal, not stale initial settings."""
    entry = state.proposal.entry
    maximum = entry.get("provenance", {}).get("catalogue_limits", {}).get("max_completion_tokens")
    if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
        maximum = None
    view.section("Proposed configuration" + (" (updated)" if updated else ""))
    rows = [
        ("Model", scrub_setup(state, state.model)),
        ("Endpoint", scrub_setup(state, state.endpoint)),
        ("Interface", prompts.choice_label(state.api_style)),
    ]
    if deferred_style:
        rows.append(("Discovery", "When you choose Test now"))
    if state.api_style not in connect.DECISION_API_STYLES:
        rows.extend(
            (
                (
                    "Context window",
                    f"{entry['context_window']:,} tokens"
                    if entry.get("context_window")
                    else "Not set",
                ),
                ("Reply limit", f"{entry['max_tokens']:,} tokens"),
            )
        )
        if maximum:
            rows.append(("Reply maximum", f"{maximum:,} tokens"))
        if "cache_breakpoint" in entry and entry["cache_breakpoint"] is None:
            rows.append(("Caching", "Off"))
        rows.append(("Reasoning", ", ".join(entry.get("reasoning_levels", {})) or "Server default"))
        rows.extend(structured_output_rows(entry))
    else:
        transport = (
            "OpenAI Decisions"
            if state.api_style == "openai"
            else "TypeSafe multipart"
            if state.decision_image_transport == "typesafe_multipart"
            else "System One JSON"
        )
        rows.append(("Image requests", transport))
        if state.typesafe_danger_zone_images:
            rows.append(("Image option", "experimental images on"))
    view.fields(scrub_setup(state, rows))
    if state.edited_settings:
        view.fields(scrub_setup(state, [("Model details", "Your edits")]))
    elif (
        state.candidate is not None
        and state.editing is None
        and (
            state.candidate.get("context_length")
            or (state.candidate.get("top_provider") or {}).get("max_completion_tokens")
            or (state.candidate.get("reasoning") or {}).get("supported_efforts")
            or isinstance((state.candidate.get("reasoning") or {}).get("default_enabled"), bool)
            or state.candidate.get("description")
            or any(connect.media_metadata(state.candidate).values())
        )
    ):
        sources = set((state.candidate.get("limit_sources") or {}).values())
        endpoint_facts = bool(state.candidate.get("endpoint_limits"))
        endpoint_media = bool(state.endpoint_models_url) and (
            (state.media_hints or {}).get("media_source") == state.endpoint_models_url
        )
        server_details = endpoint_facts or (
            state.api_style in connect.DECISION_API_STYLES and endpoint_media
        )
        if server_details:
            view.fields(scrub_setup(state, [("Model details", "This server")]))
            if "catalogue" in sources:
                view.fields(scrub_setup(state, [("Additional details", "OpenRouter")]))
        elif state.candidate:
            view.fields(scrub_setup(state, [("Model details", "OpenRouter")]))
    view.section("Media inputs")
    enabled = entry.get("media_inputs", [])
    selected = state.test_media_inputs or []
    kinds = (
        ("image",)
        if state.api_style in connect.DECISION_API_STYLES
        else ("image", "audio", "video", "pdf")
    )
    media_rows = []
    for kind in kinds:
        value = (
            "Enabled; test again"
            if kind in enabled and kind in selected
            else "Test before enabling"
            if kind in selected
            else "Enabled (not tested)"
            if kind in enabled
            else "Off"
        )
        media_rows.append((view.media_label(kind), value))
    view.fields(scrub_setup(state, media_rows))
    view.section("Tests")
    if state.no_probe or state.probe == "none":
        view.fields(scrub_setup(state, [("Status", "Off")]))
    else:
        test_rows = []
        if selected:
            test_rows.append(("Media samples", ", ".join(view.media_label(k) for k in selected)))
        if state.budget_tokens is not None and state.budget_tokens < connect.DEFAULT_CHECK_BUDGET:
            test_rows.append(("Check budget", view.format_budget(state.budget_tokens) + " tokens"))
        if not test_rows:
            test_rows.append(("Status", "Optional — choose Test now to run"))
        view.fields(scrub_setup(state, test_rows))


def review_guided_proposal(state: WizardState, *, deferred_style=False):
    """Consent or edit; edits are re-summarized before the next decision."""
    while True:
        options = (("test", "Test now"),) if not state.no_probe and state.probe != "none" else ()
        profile_options = (
            (("structured", "Change structured output profile"),)
            if state.api_style in {"chat", "anthropic"}
            and (
                state.structured_output_mode_set or "structured_output_mode" in state.proposal.entry
            )
            else ()
        )
        action = prompts.labelled_choice(
            "Would you like to test these settings?",
            (
                *options,
                ("save", "Save without testing"),
                ("change", "Change settings"),
                *profile_options,
            ),
            default="test" if options else "save",
        )
        if action == "structured":
            state.structured_output_mode = prompts.choose_structured_output_mode(
                state.proposal.entry.get("structured_output_mode")
            )
            state.structured_output_mode_set = True
            if not configure_checks(state):
                return None
            show_guided_proposal(state, deferred_style=deferred_style, updated=True)
            continue
        if action != "change":
            break
        if state.api_style in connect.DECISION_API_STYLES:
            # Decision metadata never becomes chat/reasoning request parameters.
            mode = prompts.choose_decision_images(state.media_inputs)
            if mode == "text":
                state.media_inputs, state.test_media_inputs = [], None
                state.inherit_media_inputs = False
            elif mode == "manual":
                state.media_inputs, state.test_media_inputs = ["image"], None
                state.inherit_media_inputs = False
            elif mode == "test":
                state.test_media_inputs = ["image"]
            if not configure_media_inputs(state):
                return None
        else:
            current = deepcopy(state.candidate or {})
            current.setdefault("id", state.model)
            entry = state.proposal.entry
            current["context_length"] = entry.get("context_window")
            current["top_provider"] = {
                "max_completion_tokens": entry.get("max_output_tokens")
                or entry.get("provenance", {})
                .get("catalogue_limits", {})
                .get("max_completion_tokens")
            }
            maximum = current["top_provider"]["max_completion_tokens"]
            if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
                current["top_provider"]["max_completion_tokens"] = None
            current["reasoning"] = {
                "supported_efforts": list(entry.get("reasoning_levels", {})),
                "default_effort": entry.get("reasoning_default"),
            }
            state.candidate = prompts.edit_model_details(current)
            state.context_window = state.candidate.get("context_length")
            state.levels_file = state.levels = state.reasoning_template = None
            state.edited_settings = True
            labels = state.candidate.get("reasoning", {}).get("supported_efforts", [])
            original = entry.get("reasoning_levels", {})
            state.edited_level_patches = {
                label: deepcopy(original[label])
                if label in original
                else connect.reasoning_settings("effort", state.api_style, label)
                for label in labels
            } or None
            limits = [
                v
                for v in (
                    state.context_window - 1 if state.context_window else None,
                    state.candidate.get("top_provider", {}).get("max_completion_tokens"),
                )
                if isinstance(v, int) and not isinstance(v, bool) and v > 0
            ]
            ceiling = min(limits) if limits else None
            state.reply_tokens = prompts.choose_reply_limit(
                min(entry["max_tokens"], ceiling) if ceiling else entry["max_tokens"],
                ceiling,
                output_ceiling=state.candidate.get("top_provider", {}).get("max_completion_tokens"),
                source=entry.get("provenance", {}).get("reply_limit", {}).get("source", "user"),
            )
            # Profile/media declarations and explicit test selections remain unchanged.
        if not configure_checks(state):
            return None
        show_guided_proposal(state, deferred_style=deferred_style, updated=True)
    return action


def guided_setup(state: WizardState) -> bool:
    """Prepare a proposal using GETs only; the action below is model-call consent."""
    if (
        not select_connection(state)
        or not select_model(state)
        or not prepare_metadata(state)
        or not select_request_interface(state)
    ):
        return False
    deferred_style = state.api_style is None
    if deferred_style and state.structured_output_mode_set:
        raise click.UsageError(
            "--structured-output-mode requires --api-style chat or anthropic, not interface discovery"
        )
    explicit_reply_tokens = state.reply_tokens
    # A provider preset is a proposal, never evidence of supported interfaces.
    state.api_style = state.api_style or state.default_style
    if not configure_media_inputs(state) or not configure_checks(state):
        return False
    show_guided_proposal(state, deferred_style=deferred_style)
    action = review_guided_proposal(state, deferred_style=deferred_style)
    if action is None:
        return False
    state.approval = state.probe if action == "test" else "none"
    if action == "save":
        state.test_media_inputs = None
    if deferred_style:
        if state.api_style not in connect.DECISION_API_STYLES:
            state.reply_tokens = state.proposal.entry["max_tokens"]
        state.api_style = None
        if not check_interfaces(state):
            return False
        if state.api_style in connect.DECISION_API_STYLES:
            state.reply_tokens = explicit_reply_tokens
            state.candidate = (
                connect.decision_metadata(
                    state.model, state.candidate, state.selected_endpoint_model
                )
                or None
            )
            if not configure_media_inputs(state):
                return False
        # Rebuild only the request plan after interface selection, not prompt rounds.
        if not configure_checks(state):
            return False
    elif action == "save":
        if not configure_checks(state):
            return False
    if state.guided_media_revised:
        show_guided_proposal(state, updated=True)
        action = review_guided_proposal(state)
        if action is None:
            return False
        state.approval = state.probe if action == "test" else "none"
        if action == "save":
            state.test_media_inputs = None
        if not configure_checks(state):
            return False
    return run_checks(state) and save_model(state)


def run_wizard(**options):
    """Run ordered steps with one shared budget and an explicit save decision."""
    state = WizardState(**options)
    state.path = Path(state.output) if state.output else paths.get_user_dir("llm_config.yaml")
    state.api_key = None
    state.explicit_key_env = state.api_key_env is not None
    state.editing = None
    try:
        if not state.yes:
            state.guided = True
            state.guided_media_explicit = (
                state.media_inputs is not None or state.test_media_inputs is not None
            )
            guided_setup(state)
            return
        for step in (
            select_connection,
            select_model,
            check_interfaces,
            prepare_metadata,
            configure_media_inputs,
            configure_metadata,
            configure_checks,
            run_checks,
            save_model,
        ):
            if not step(state):
                return
    except (ValueError, OSError, yaml.YAMLError, httpx.HTTPError) as exc:
        detail = scrub_setup(
            state, view.local_failure(exc, api_key=state.api_key, api_key_env=state.api_key_env)
        )
        if state.show_diagnostics:
            click.echo(
                "Agent diagnostic prompt:\n"
                + setup_diagnostic(
                    state,
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
        else:
            view.line(
                scrub_setup(state, "For troubleshooting details, rerun with --show-diagnostics."),
                dim=True,
            )
        raise click.ClickException(detail) from None
