# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wizard steps; loaded lazily by the Connect command."""

import asyncio
import os
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass, field, replace
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
    last_step: int = 0  # highest step header shown in this attempt
    api_key: Any = None
    api_key_env: Any = None
    api_style: Any = None
    approval: Any = None
    budget_tokens: Any = None
    candidate: Any = None
    catalogue_model: Any = None
    configured: Any = None
    context_window: Any = None
    manual_limits: Any = None
    manual_limits_required: bool = False
    manual_limits_deferred: bool = False
    settings_prefill: Any = field(default_factory=dict, init=False)
    route_settings_prefill: Any = field(default_factory=dict, init=False)
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
    tokens_spent: int = 0
    test_failed: bool = False
    failure_ack_snapshot: Any = None
    route_revised: bool = False
    manual_reasoning_owned: bool = False
    manual_reasoning_origin: bool = False
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
    guided_reasoning_choice: Any = None
    guided_reasoning_levels: Any = None
    guided_reasoning_route: Any = None
    guided_reasoning_active: Any = None
    guided_reasoning_current: Any = None
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

    def __post_init__(self):
        self.settings_prefill = {
            key: getattr(self, key)
            for key in ("context_window", "levels_file", "levels", "reasoning_template")
        }
        self.route_settings_prefill = {
            **self.settings_prefill,
            **{
                key: getattr(self, key)
                for key in (
                    "reply_tokens",
                    "media_inputs",
                    "test_media_inputs",
                    "reasoning_level",
                    "structured_output_mode",
                )
            },
        }


def scrub_setup(state: WizardState, value):
    """Protect both active credentials in previews and troubleshooting output."""
    from nooa.unifiedllm.connect._diagnostics import scrub_report

    value = scrub_report(value, api_key=state.api_key, api_key_env=state.api_key_env)
    return scrub_report(
        value, api_key=state.discovery_api_key, api_key_env=state.discovery_api_key_env
    )


def setup_diagnostic(state: WizardState, *args, **kwargs):
    return scrub_setup(state, connect.diagnostic_prompt(*args, **kwargs))


def reasoning_disabled(params) -> bool:
    """True when a level's settings ask the model not to reason."""
    thinking = params.get("thinking")
    template = params.get("chat_template_kwargs")
    reasoning = params.get("reasoning")
    efforts = (
        params.get("reasoning_effort"),
        reasoning.get("effort") if isinstance(reasoning, dict) else None,
    )
    return (
        (isinstance(thinking, dict) and thinking.get("type") == "disabled")
        or (isinstance(template, dict) and template.get("enable_thinking") is False)
        or any(isinstance(e, str) and e.strip().lower() in {"none", "off"} for e in efforts)
    )


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
                ignored_off = (
                    event.name.startswith("level:")
                    and event.outcome.get("outcome") == "accepted"
                    and bool(event.outcome.get("reasoning_observed"))
                    and reasoning_disabled((reasoning_levels or {}).get(event.name[6:], {}))
                )
                progress.update(
                    event.name,
                    event.outcome,
                    missing_reasoning=bool(missing),
                    ignored_off=ignored_off,
                    media_inputs=media_inputs,
                )
    finally:
        progress.finish(summary=summary)
    raise click.ClickException("Checks ended without a result.")


STEP_TITLES = {1: "Connection", 2: "Model", 3: "Settings", 4: "Test", 5: "Save model"}


def show_step(state: WizardState, number: int) -> None:
    """Print each step header once per attempt, in order; skipped steps keep the total."""
    if number > state.last_step:
        state.last_step = number
        view.step(number, STEP_TITLES[number], total=len(STEP_TITLES))


def select_connection(state: WizardState) -> bool:
    """Select connection; return False when the user cancels."""
    from ._connect_input_policy import validate_native_declaration

    if state.api_style:
        validate_native_declaration(state)
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
        from ._connect_input_policy import repair_native_media

        current = (
            prompts.parse_media_inputs(state.media_inputs)
            if state.media_inputs is not None
            else state.editing.get("media_inputs", [])
        )
        repaired = repair_native_media(
            state, current, entry=state.editing, explicit=state.media_inputs is not None
        )
        if repaired is None:
            return False
        # Validate a detached candidate; preserve the raw source profile/evidence.
        connect.configure_entry({**state.editing, "media_inputs": repaired})
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
        show_step(state, 1)
    if state.provider and state.provider not in (*connect.PROVIDERS, "custom"):
        raise click.UsageError(
            f"Unknown provider. Choose {', '.join(connect.PROVIDERS)}, or custom."
        )
    if not state.provider and not state.endpoint and not state.yes:
        # Preselect the first provider whose key is already set, so Enter works.
        keyed = next(
            (
                name
                for name, preset in connect.PROVIDERS.items()
                if os.environ.get(preset.api_key_env)
            ),
            None,
        )
        state.provider = prompts.prompt(
            "Choose a provider",
            default=keyed,
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
    if state.endpoint == connect.PROVIDERS["anthropic"].api_base:
        state.default_style = "anthropic"
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
            else prompts.key_source(
                default_env, [p.api_key_env for p in connect.PROVIDERS.values()]
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
        show_step(state, 2)
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


def known_decision_model(state: WizardState) -> bool:
    """Only explicit selected-model type facts can bypass provisional chat limits."""
    metadata = state.selected_endpoint_model or state.candidate or {}
    outputs = (metadata.get("architecture") or {}).get("output_modalities") or metadata.get(
        "output_modalities", []
    )
    return (
        (isinstance(outputs, list) and "decisions" in outputs and "text" not in outputs)
        or any(metadata.get(field) == "decision" for field in ("model_type", "client_type"))
        or (metadata.get("mode") == "evaluation")
    )


def discard_chat_limits(state: WizardState):
    """A later native Decision selection drops provisional completion declarations."""
    # Even an explicit context prefill is a chat bound, not Decision state.
    state.context_window = None
    state.manual_limits = None
    state.manual_limits_required = False
    state.manual_limits_deferred = False
    state.candidate = (
        connect.decision_metadata(state.model, state.candidate, state.selected_endpoint_model)
        or None
    )
    state.media_hints = state.candidate


def select_request_interface(state: WizardState) -> bool:
    """Choose after the selected model/metadata, before any inference consent."""
    if not state.metadata_displayed:
        show_model_metadata(state)
    if state.api_style or state.yes or state.editing is not None:
        return True
    # The native provider/endpoint contract is known, unlike a Claude model on a gateway.
    if state.provider == "anthropic" or state.endpoint == connect.PROVIDERS["anthropic"].api_base:
        state.api_style = "anthropic"
        return True
    choices = ("discover", "chat", "responses", "anthropic", "systemone", "openai")
    # Use declared output/type facts, not the model name, to suggest an interface.
    decision_hint = known_decision_model(state)
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
        discard_chat_limits(state)
    # An explicit Completion override of a native type still needs chat bounds.
    if choice not in connect.DECISION_API_STYLES:
        require_manual_limits(state)
    reject_decision_media(state)
    return True


def keep_untested_media(state: WizardState) -> None:
    """Saving without testing keeps formats that were waiting for a test, as not tested."""
    if state.test_media_inputs:
        current = list(state.media_inputs or [])
        state.media_inputs = current + [k for k in state.test_media_inputs if k not in current]
        state.inherit_media_inputs = False
    state.test_media_inputs = None


class RestartSetup(Exception):
    """The user chose to start setup again from the first step."""


def offer_restart(state: WizardState) -> bool:
    """After a failure, the only choice is to start over; return False to stop."""
    if state.yes:
        return False
    if prompts.confirm("Start over?", default=True):
        raise RestartSetup
    view.note("Nothing was saved. Give the prompt above to an agent, or run nooa connect again.")
    return False


def check_interfaces(state: WizardState) -> bool:
    """Check interfaces; return False when the user cancels."""
    if state.structured_output_mode_set and not state.api_style:
        raise click.UsageError(
            "--structured-output-mode requires an explicit Completion interface, not interface discovery"
        )
    reject_decision_media(state)
    from ._connect_input_policy import validate_native_declaration

    if state.api_style:
        validate_native_declaration(state)
    show_step(state, 3)
    if state.api_style == "responses":
        view.line(scrub_setup(state, connect.ENCRYPTED_REASONING_EXPLANATION), dim=True)
    if not state.api_style:
        available = ("chat", "responses", "anthropic", *state.interface_endpoints)
        if state.approval != "none":
            state.interface_spent = state.interface_spent or 0
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
                            budget_tokens=max(0, state.budget_tokens - state.tokens_spent),
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
                state.tokens_spent += state.interfaces.tokens_charged_to_budget
                state.interface_spent += state.interfaces.tokens_charged_to_budget
                state.interfaces = replace(
                    state.interfaces, tokens_charged_to_budget=state.interface_spent
                )
                available = state.interfaces.accepted
                if available:
                    if state.interfaces.recommended_style in available:
                        state.default_style = state.interfaces.recommended_style
                    break
                state.test_failed = True
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
                                remaining=max(0, state.budget_tokens - state.tokens_spent),
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
                remaining = max(0, state.budget_tokens - state.tokens_spent)
                if remaining < state.output_tokens + 512:
                    view.line(
                        scrub_setup(
                            state,
                            "Check budget used up. Nothing was saved. Restart with a larger --budget-tokens value.",
                        )
                    )
                    return False
                # No working interface: show the result and offer a fresh start.
                return offer_restart(state)
            if state.interfaces is None:
                view.note("Replacement API formats have not been tested; choose one explicitly.")
            else:
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
                        "Accepted client types: "
                        + ", ".join(state.interfaces.accepted_client_types),
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
            default="image",
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


def usable_server_metadata(state: WizardState) -> bool:
    """Exact allowlisted route facts, not an ID, empty containers or neighbours."""
    from nooa.unifiedllm.connect._server_details import listing_facts, positive_integer

    selected = state.selected_endpoint_model or {}
    if state.model and selected.get("id") == state.model:
        if known_decision_model(state):
            return True
        facts = listing_facts(selected)
        for key, value in facts.items():
            if key == "architecture":
                if any(value.get(field) for field in ("input_modalities", "output_modalities")):
                    return True
            elif key != "id" and value:
                return True
    for record in state.server_details.records if state.server_details else ():
        if record.get("schema") == "openrouter.fallback" or record.get("id") != state.model:
            continue
        facts = record.get("facts") or {}
        if any(
            positive_integer(facts.get(key))
            for key in (
                "max_input_tokens",
                "max_output_tokens",
                "context_length",
                "context_window",
            )
        ):
            return True
        if any(
            isinstance(facts.get(key), bool)
            for key in (
                "supports_vision",
                "supports_function_calling",
                "supports_reasoning",
                "supports_parallel_function_calling",
                "supports_system_messages",
                "supports_response_schema",
                "supports_prompt_caching",
                "supports_audio_input",
                "supports_audio_output",
                "supports_web_search",
            )
        ):
            return True
        if (
            facts.get("mode")
            or facts.get("supported_reasoning_efforts")
            or facts.get("supported_openai_params")
        ):
            return True
    return False


def prepare_metadata(state: WizardState) -> bool:
    """Fetch allowed metadata and collect missing manual limits before the first summary."""
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
                if (
                    not state.yes
                    and models
                    and not state.catalogue_model
                    and not usable_server_metadata(state)
                ):
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
        # A native OpenAI raw ID has one exact provider-qualified catalogue ID.
        # This is supplementary public metadata, never a fuzzy model-name guess.
        if (
            not matches
            and not state.catalogue_model
            and (
                state.provider == "openai" or state.endpoint == connect.PROVIDERS["openai"].api_base
            )
        ):
            matches = [item for item in models if item.get("id") == "openai/" + state.model]
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
            if not selected:
                state.manual_limits_required = True
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
                if state.yes or state.catalogue_model or usable_server_metadata(state)
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
                if not selected:
                    state.manual_limits_required = True
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
    require_manual_limits(state)
    show_model_metadata(state)
    return True


def model_media_hints(state):
    """Detached advertisement for display and suggestions, never permissions.

    Exact selected listing media (including empty/false declarations) wins.
    Only when that listing has no media facts may a selected LiteLLM group
    supports_vision flag describes image advertisement. It never implies PDF,
    audio/video, native profiles or a tested result.
    """
    hints = deepcopy(state.media_hints or state.candidate or state.selected_endpoint_model or {})
    endpoint = state.selected_endpoint_model or {}
    if endpoint.get("id") == state.model and connect.media_metadata(endpoint):
        return connect.model_metadata(state.model, hints, endpoint)
    records = state.server_details.records if state.server_details else ()
    for record in records:
        if record.get("schema") != "litellm.model_group.info" or record.get("id") != state.model:
            continue
        flag = (record.get("facts") or {}).get("supports_vision")
        if isinstance(flag, bool):
            # An explicit false overrides lower-priority public image hints but
            # says nothing about the other independently advertised formats.
            original_source = hints.get("media_source", "published model information")
            other_types = [
                kind
                for kind in connect.advertised_media_suggestions(hints)["advertised"]
                if kind != "image"
            ]
            hints.setdefault("capabilities", {})["image_input"] = {"supported": flag}
            if flag:
                image_source = record.get("source") or "This server"
                hints["media_source"] = (
                    image_source + " (Images); " + original_source + " (other listed inputs)"
                    if other_types
                    else image_source
                )
            hints["media_model_id"] = state.model
            break
    return hints


def show_model_metadata(state: WizardState):
    """One selected-model summary, before interface/media choices and model calls."""
    state.metadata_displayed = True
    model = model_media_hints(state)
    report = connect.advertised_media_suggestions(model)
    endpoint = state.selected_endpoint_model or {}
    view.section("Model details")
    rows = [("Model", state.model)]
    records = state.server_details.records if state.server_details else ()
    server_facts = usable_server_metadata(state)
    if state.edited_settings or state.manual_limits:
        rows.append(("Source", "Your settings"))
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
        value = (
            endpoint.get(key)
            or (endpoint.get("name") if key == "display_name" else None)
            or model.get(key)
        )
        if value:
            rows.append((label, value))
    if known_decision_model(state):
        rows.append(("Model type", "Decision"))
    elif endpoint.get("model_type") or endpoint.get("client_type"):
        rows.append(("Model type", endpoint.get("model_type") or endpoint.get("client_type")))
    for record in records:
        if record.get("schema") != "litellm.model_group.info" or record.get("id") != state.model:
            continue
        facts = record.get("facts") or {}
        for key, label in (
            ("supports_function_calling", "Tools listed"),
            ("supports_reasoning", "Reasoning listed"),
        ):
            if isinstance(facts.get(key), bool):
                rows.append((label, "Yes" if facts[key] else "No"))
        if facts.get("supported_reasoning_efforts"):
            rows.append(
                ("Reasoning efforts listed", ", ".join(facts["supported_reasoning_efforts"]))
            )
    interfaces = endpoint.get("supported_interfaces") or []
    if interfaces:
        rows.append(
            ("Interfaces listed", ", ".join(prompts.choice_label(style) for style in interfaces))
        )
    report = connect.advertised_media_suggestions(model)
    if report["advertised"]:
        rows.append(
            ("Media types", ", ".join(view.media_label(kind) for kind in report["advertised"]))
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
    if state.manual_limits:
        rows.extend(
            [
                ("Context window (declared)", f"{state.manual_limits['context_window']:,} tokens"),
                (
                    "Input token limit (declared)",
                    f"{state.manual_limits['max_input_tokens']:,} tokens",
                ),
                ("Effective context bound", f"{state.context_window:,} tokens"),
            ]
        )
    view.fields(scrub_setup(state, rows))
    if not report["advertised"]:
        view.note("No media is listed.")
    if state.server_details:
        for notice in state.server_details.notices:
            view.warning(scrub_setup(state, notice))


def require_manual_limits(state: WizardState, *, edit=False):
    """Bound only explicit human manual continuations; never synthesize catalogue facts.

    Raw declarations live in wizard/proposal provenance. The registry supports only
    a conservative effective context_window, not separate literal input metadata.
    """
    if (
        state.yes
        or state.manual_limits_deferred
        or state.api_style in connect.DECISION_API_STYLES
        or state.api_style is None
        and known_decision_model(state)
        and not edit
    ):
        return
    if not (edit or state.manual_limits_required or state.no_catalogue and state.editing is None):
        return
    if state.manual_limits is not None and not edit:
        return
    server = state.selected_endpoint_model or {}
    known = (
        connect.model_metadata(state.model, None, server) if server.get("id") == state.model else {}
    )
    if known.get("context_length") and not edit:
        # A reported context or input bound is already conservative and exact.
        # Preserve it, including its separate raw endpoint_limits provenance.
        return
    previous = state.manual_limits or {}
    raw_context = previous.get("context_window", state.context_window)
    raw_input = previous.get("max_input_tokens")
    context, input_limit = prompts.manual_model_limits(raw_context, raw_input, edit=edit)
    state.manual_limits = {"context_window": context, "max_input_tokens": input_limit}
    state.context_window = min(context, input_limit)
    if known.get("context_length"):
        state.context_window = min(state.context_window, known["context_length"])
    if not state.guided:
        view.fields(
            [
                ("Source", "Your settings"),
                ("Context window (declared)", f"{context:,} tokens"),
                ("Input token limit (declared)", f"{input_limit:,} tokens"),
                ("Effective context bound", f"{state.context_window:,} tokens"),
            ]
        )


def discard_catalogue_settings(state: WizardState):
    """Explicit continuation discards public guesses, not exact selected-server facts."""
    server = state.selected_endpoint_model
    state.candidate = (
        connect.model_metadata(state.model, None, server)
        if server and server.get("id") == state.model
        else None
    )
    if state.edited_settings:
        for key, value in state.settings_prefill.items():
            setattr(state, key, value)
    state.edited_level_patches = None
    state.manual_limits_required = True
    state.edited_settings = False


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
                discard_catalogue_settings(state)
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
    require_manual_limits(state)
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


def select_manual_media(state: WizardState, previous, *, available=None):
    """Plan explicitly selected tests; selection alone never grants new permissions."""
    from ._connect_input_policy import picker_options, repair_native_media

    repaired = repair_native_media(state, previous)
    if repaired is None:
        raise click.Abort()
    options = {"available": available} if available is not None else picker_options(state)
    selected = prompts.choose_media_inputs(repaired, **options)
    repair_native_media(state, selected, explicit=True)
    state.test_media_inputs = None
    state.inherit_media_inputs = False
    state.media_inputs = selected
    if not selected:
        return
    report = connect.advertised_media_suggestions(
        {"id": state.model, "input_modalities": selected},
        api_style=state.api_style,
        decision_image_transport=state.decision_image_transport or "json",
        typesafe_danger_zone_images=state.typesafe_danger_zone_images,
    )
    from ._connect_input_policy import available_media

    supported = available_media(state)
    for kind, reason in report["unavailable_checks"].items():
        if kind in supported:
            view.fields([(view.media_label(kind) + " test", reason)])
    can_test = not state.no_probe and state.approval == "all" and bool(report["checkable"])
    if not can_test:
        return
    names = ", ".join(view.media_label(kind) for kind in selected)
    action = prompts.labelled_choice(
        f"Would you like to test these media types: {names}?",
        (("test", f"Test {names}"), ("enable", f"Enable {names} without testing")),
        default="test",
    )
    if action == "test":
        state.test_media_inputs = list(selected)
        state.media_inputs = list(previous)
        state.inherit_media_inputs = True


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
        from ._connect_input_policy import validate_native_declaration

        validate_native_declaration(state)
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
            media_inputs=[],  # route preview only; no mutation or permission revocation
        ).entry
        saved_inputs = validate_media_inputs(previous_entry.get("media_inputs"))
        from ._connect_input_policy import available_media

        if connect.same_media_route(previous_entry, route) or state.editing is not None:
            previous = saved_inputs
        elif len(available_media(state, entry=route)) == 2 and {"audio", "video"}.intersection(
            saved_inputs
        ):
            # A GUI route change cannot silently erase an invalid previous
            # selection. Obtain an explicit repair for the replacement route.
            previous = saved_inputs
        else:
            previous = ()
    elif state.editing:
        previous = validate_media_inputs(state.editing.get("media_inputs"))
    from ._connect_input_policy import repair_native_media

    repaired = repair_native_media(state, previous)
    if repaired is None:
        return False
    previous = tuple(repaired)
    if state.media_inputs is not None:
        return True  # explicit repair; do not restore inherited unsupported permissions
    if offer_suggestions(state, previous):
        return True
    if state.test_media_inputs is not None:
        state.test_media_inputs = prompts.parse_media_inputs(state.test_media_inputs)
    elif not state.yes and not state.no_probe and state.approval == "all":
        mode = prompts.choose_media_mode(editing=bool(state.editing))
        if mode == "test":
            state.test_media_inputs = ["image", "audio", "video", "pdf"]
        else:
            select_manual_media(state, previous)
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
        if state.yes:
            state.media_inputs = list(previous)
        else:
            select_manual_media(state, previous)
    return True


def server_effort_levels(state: WizardState):
    """Effort levels the selected model's own server lists, as request settings.

    Listing is a hint; the chosen level is still tested before it is trusted.
    """
    efforts, supported = None, None
    for record in state.server_details.records if state.server_details else ():
        if record.get("schema") == "openrouter.fallback" or record.get("id") != state.model:
            continue
        facts = record.get("facts") or {}
        if isinstance(facts.get("supports_reasoning"), bool):
            supported = facts["supports_reasoning"]
        if facts.get("supported_reasoning_efforts"):
            efforts = list(facts["supported_reasoning_efforts"])
    if supported is False:
        return {}
    if not efforts:
        return None
    template = "adaptive" if state.api_style == "anthropic" else "effort"
    return {
        label: connect.reasoning_settings(template, state.api_style, label)
        for label in efforts
        if isinstance(label, str) and label
    } or None


def discovered_effort_levels(state: WizardState):
    """Standard effort choices when interface discovery saw reasoning respond to effort.

    Discovery sends one effort request per interface. If the chosen interface
    returned reasoning for it, low/medium/high are offered; the chosen one is
    tested before saving.
    """
    result = (state.interfaces.results or {}).get(state.api_style) if state.interfaces else None
    if result is None:
        return None
    for name, record in result.entry.get("provenance", {}).get("probes", {}).items():
        if (
            name.startswith("level:")
            and record.get("outcome") == "accepted"
            and record.get("reasoning_observed")
        ):
            template = state.reasoning_template or "effort"
            return {
                label: connect.reasoning_settings(template, state.api_style, label)
                for label in ("low", "medium", "high")
            }
    return None


def discovered_toggle_levels(state: WizardState):
    """On/Off thinking for chat models that reason by default but ignore effort.

    Many open models served by vLLM/SGLang (Qwen, for example) switch thinking
    with ``chat_template_kwargs.enable_thinking``. Offered only when discovery
    saw reasoning on this Chat interface and no effort request changed it; the
    chosen setting is tested before saving.
    """
    if state.api_style != "chat" or state.reasoning_template:
        return None
    result = (state.interfaces.results or {}).get("chat") if state.interfaces else None
    if result is None:
        return None
    probes = result.entry.get("provenance", {}).get("probes", {})
    effort_worked = any(
        name.startswith("level:")
        and record.get("outcome") == "accepted"
        and record.get("reasoning_observed")
        for name, record in probes.items()
    )
    reasons = any(record.get("reasoning_observed") for record in probes.values())
    if effort_worked or not reasons:
        return None
    return {
        "on": connect.reasoning_settings("toggle", "chat", "on"),
        "off": connect.reasoning_settings("toggle", "chat", "off"),
    }


def choose_reasoning(state: WizardState, *, deferred=False, edit=False) -> bool:
    """Plain route-specific selection. Nothing is activated by merely listing levels."""
    if state.api_style in connect.DECISION_API_STYLES:
        state.guided_reasoning_choice = state.guided_reasoning_levels = None
        state.guided_reasoning_route = state.guided_reasoning_active = None
        return True
    if deferred:
        view.note("Reasoning: choose after the request interface is known.")
        return True
    from nooa.unifiedllm.connect._reasoning import native_levels

    route = (state.model, state.api_style, state.endpoint)
    if (
        not edit
        and state.guided_reasoning_route == route
        and state.guided_reasoning_choice is not None
    ):
        return True
    saved = state.editing or state.existing or {}
    routed = {
        "model_name": ("anthropic/" if state.api_style == "anthropic" else "openai/") + state.model,
        "api_base": connect.normalize_endpoint(state.endpoint),
        "api_style": state.api_style,
        "client_type": "responses" if state.api_style == "responses" else "completion",
    }
    if saved and not connect.same_media_route(saved, routed):
        saved = {}
    same_previous_route = state.guided_reasoning_route in {None, route}
    levels = deepcopy(state.edited_level_patches) if same_previous_route else None
    if state.levels_file:
        with Path(state.levels_file).open() as source:
            levels = yaml.safe_load(source)
    elif state.levels or state.reasoning_template:
        if not state.levels or not state.reasoning_template:
            raise click.UsageError("--reasoning-template and --levels must be supplied together.")
        levels = {
            label.strip(): connect.reasoning_settings(
                state.reasoning_template, state.api_style, label.strip()
            )
            for label in state.levels.split(",")
        }
    elif levels is None and saved:
        levels = deepcopy(saved.get("reasoning_levels"))
    native_manual = (
        state.manual_reasoning_origin
        and same_previous_route
        and not (state.levels_file or state.levels or state.reasoning_template)
    )
    if levels is None or native_manual:
        previous_on = deepcopy((levels or {}).get("on")) if native_manual else None
        levels = native_levels(
            state.selected_endpoint_model,
            model_id=state.model,
            api_style=state.api_style,
            source=state.endpoint_models_url,
            endpoint=state.endpoint,
        )
        native_manual = bool(levels and "on" in levels)
        state.manual_reasoning_origin = native_manual
        if native_manual and previous_on is not None:
            levels["on"] = previous_on  # retain the wizard's reviewed allowance across edits
    if levels is None:
        levels = server_effort_levels(state)
    if not levels:
        # What this interface actually did in discovery wins over a server
        # listing that says the model doesn't reason.
        levels = discovered_effort_levels(state) or discovered_toggle_levels(state) or levels
    if levels is None:
        # Only exact OpenRouter metadata retains known effort templates here.
        candidate = state.candidate or {}
        if (
            state.endpoint + "/models" == connect.CATALOGUE_URL
            and candidate.get("id") == state.model
        ):
            labels = (candidate.get("reasoning") or {}).get("supported_efforts") or []
            levels = {
                label: connect.reasoning_settings("effort", state.api_style, label)
                for label in labels
            } or None
    if levels is not None:
        from nooa.unifiedllm.reasoning import ReasoningConfig

        ReasoningConfig(levels=levels)
    current = state.proposal.entry if edit and state.proposal else saved
    if state.reasoning_level is not None and not edit:
        from nooa.unifiedllm.reasoning import ReasoningConfig

        ReasoningConfig(levels=levels).settings(state.reasoning_level)
        current = {**current, "reasoning_level": state.reasoning_level, "reasoning_levels": levels}
    if state.guided_reasoning_route not in {None, route}:
        current = {}
        state.guided_reasoning_active = None
    raw_controls = {
        "thinking",
        "reasoning",
        "reasoning_effort",
        "output_config",
        "chat_template_kwargs",
    }
    keep = bool(
        current.get("reasoning_level")
        or raw_controls & current.keys()
        or raw_controls & (current.get("extra_body") or {}).keys()
    )
    if not levels and not keep:
        view.note(
            "Reasoning: Model default (controls unknown)."
            if levels is None
            else "Reasoning: Model default (selection unsupported)."
        )
        state.guided_reasoning_choice = None
        state.guided_reasoning_levels = None
        state.guided_reasoning_route = route
        return True
    options = [("keep", "Keep current settings")] if keep else []
    options.append(("default", "Model default (no new override)"))
    selections = {}
    occupied = {"keep", "default", "cancel"}
    for label in levels or {}:
        token = label
        while token in occupied:
            token = "level:" + token
        occupied.add(token)
        selections[token] = label
        options.append((token, "Off" if label in {"off", "disabled", "none"} else label.title()))
    options = tuple(options)  # Ctrl-C cancels setup
    default = "keep" if keep else "default"
    if state.guided_reasoning_route == route and state.guided_reasoning_choice in dict(options):
        default = state.guided_reasoning_choice
    view.note("Choose how much reasoning to request. Listed choices are not yet tested.")
    choice = prompts.labelled_choice("Reasoning", options, default=default)
    if choice == "cancel":
        view.note("Setup cancelled. Nothing saved.")
        return False
    state.guided_reasoning_current = deepcopy(current) if choice == "keep" else None
    state.guided_reasoning_active = (
        current.get("reasoning_level")
        if choice == "keep"
        else None
        if choice == "default"
        else selections[choice]
    )
    if choice == "keep" and current.get("reasoning_levels"):
        levels = deepcopy(current["reasoning_levels"])
    if native_manual and (
        choice == "on" or choice == "keep" and current.get("reasoning_level") == "on"
    ):
        state.manual_reasoning_owned = True
        current_budget = levels["on"]["thinking"]["budget_tokens"]
        budget = prompts.prompt(
            "Reasoning token allowance (minimum 1,024)", default=str(current_budget)
        )
        try:
            budget = int(budget)
        except (TypeError, ValueError):
            raise click.UsageError(
                "Reasoning token allowance must be an integer of at least 1,024."
            ) from None
        if budget < 1024:
            raise click.UsageError("Reasoning token allowance must be at least 1,024.")
        levels["on"] = {"thinking": {"type": "enabled", "budget_tokens": budget}}
    elif choice != "keep":
        state.manual_reasoning_owned = False
        if native_manual:
            levels.pop("on", None)  # materialize a manual budget only after selecting On
    state.guided_reasoning_choice = choice
    state.guided_reasoning_levels = levels
    state.guided_reasoning_route = route
    return True


def bound_simple_reasoning(state: WizardState, levels) -> bool:
    """Only wizard-owned manual On uses this UI; advanced profiles stay untouched."""
    if not state.manual_reasoning_owned or state.guided_reasoning_active != "on":
        return True
    budget = levels["on"]["thinking"]["budget_tokens"]
    # Resolve the ordinary reply recommendation without thinking patches first.
    base = connect.plan(
        state.alias or "candidate",
        state.model,
        state.api_style,
        state.endpoint,
        state.api_key_env,
        catalogue=state.candidate,
        endpoint_model=state.selected_endpoint_model,
        endpoint_models_url=state.endpoint_models_url,
        context_window=state.context_window,
        reply_tokens=state.reply_tokens,
        media_inputs=[],
        existing_entry=None,
    ).entry
    cap = connect.effective_config_snapshot(base)["params"]["max_tokens"]
    while budget >= cap:
        view.note(
            f"Reasoning needs at least 1,024 tokens and must be below the saved reply limit "
            f"({cap:,}); reasoning and the answer share that limit. No cap was changed."
        )
        if (
            prompts.labelled_choice(
                "The reply limit cannot fit this reasoning allowance",
                (("adjust", "Adjust reply limit"), ("cancel", "Cancel setup")),
                default="adjust",
            )
            == "cancel"
        ):
            return False
        maximum = (
            base.get("provenance", {}).get("catalogue_limits", {}).get("max_completion_tokens")
        )
        bounds = [
            v
            for v in (maximum, base.get("context_window", 0) - 1)
            if isinstance(v, int) and not isinstance(v, bool) and v > 0
        ]
        ceiling = min(bounds) if bounds else None
        if ceiling is not None and budget >= ceiling:
            view.note(
                f"The server limit ({ceiling:,}) cannot fit that allowance. Choose a smaller allowance or cancel."
            )
            action = prompts.labelled_choice(
                "Reasoning allowance exceeds the server limit",
                (("allowance", "Change reasoning allowance"), ("cancel", "Cancel setup")),
                default="allowance",
            )
            if action == "cancel":
                return False
            budget = int(
                prompts.prompt("Reasoning token allowance (minimum 1,024)", default="1024")
            )
            if budget < 1024:
                raise click.UsageError("Reasoning token allowance must be at least 1,024.")
            levels["on"]["thinking"]["budget_tokens"] = budget
            continue
        cap = prompts.choose_reply_limit(cap, ceiling, output_ceiling=maximum)
        state.reply_tokens = cap
    return True


def configure_checks(state: WizardState) -> bool:
    """Configure checks; return False when the user cancels."""
    reject_decision_media(state)
    from ._connect_input_policy import repair_native_media

    current = (
        prompts.parse_media_inputs(state.media_inputs)
        if state.media_inputs is not None
        else (state.editing or state.existing or {}).get("media_inputs", [])
    )
    repaired = repair_native_media(
        state, current, explicit=state.guided_media_explicit or state.yes
    )
    if repaired is None:
        return False
    require_manual_limits(state)
    profile = connect.plan(
        state.alias or "candidate",
        state.model,
        state.api_style,
        state.endpoint,
        state.api_key_env,
        existing_entry=state.editing or state.existing,
        media_inputs=[] if state.editing is not None else repaired,
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
            media_inputs=[] if state.editing is not None else state.media_inputs,
            decision_image_transport=state.decision_image_transport,
            typesafe_danger_zone_images=state.typesafe_danger_zone_images,
        ).entry
        state.media_inputs = (
            list(repaired) if state.editing is not None else previous.get("media_inputs", [])
        )
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
    if state.guided_reasoning_levels is not None and state.guided_reasoning_choice is not None:
        state.patches = deepcopy(state.guided_reasoning_levels)

    if not bound_simple_reasoning(state, state.patches):
        return False
    planning_candidate = state.candidate
    if state.manual_limits and planning_candidate is not None:
        # User edits must use the current manual bound, not a stale provisional
        # candidate context. Exact endpoint limits are still merged by plan().
        planning_candidate = deepcopy(planning_candidate)
        planning_candidate["context_length"] = state.context_window
        planning_candidate.setdefault("limit_sources", {})["context_length"] = "user"
        if planning_candidate.get("top_provider"):
            planning_candidate["top_provider"].pop("context_length", None)
    state.proposal = connect.plan(
        state.alias or "candidate",
        state.model,
        state.api_style,
        state.endpoint,
        state.api_key_env,
        catalogue=planning_candidate,
        catalogue_settings=bool(state.edited_settings),
        endpoint_model=state.selected_endpoint_model,
        endpoint_models_url=state.endpoint_models_url,
        media_inputs=[] if state.editing is not None else state.media_inputs,
        test_media_inputs=state.test_media_inputs,
        decision_image_transport=state.decision_image_transport,
        typesafe_danger_zone_images=state.typesafe_danger_zone_images,
        reasoning_levels=state.patches,
        **({"reasoning_level": state.reasoning_level} if state.reasoning_level is not None else {}),
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
        # Guided setup tests only the reasoning level the user chose.
        checked_levels=([state.guided_reasoning_active] if state.guided_reasoning_active else [])
        if state.guided
        and state.editing is None
        and not (state.levels or state.levels_file or state.reasoning_template)
        else None,
        **mode_kwargs,
    )
    if (
        state.guided
        and state.editing is None
        and state.interfaces
        and state.guided_reasoning_choice is None
        and not (state.levels or state.levels_file or state.reasoning_template)
    ):
        # No reasoning choice was offered; the single discovery probe level is
        # not a saved choice.
        state.proposal.entry.pop("reasoning_levels", None)
        state.proposal.entry.pop("reasoning_default", None)
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
        # The temporary proposal did not claim permissions for its synthetic
        # route. Restore the explicit selection on the preserved actual route;
        # refresh_plan below validates it before any check or save.
        merged["media_inputs"] = list(state.media_inputs or [])
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
    choice = state.guided_reasoning_choice
    if state.api_style not in connect.DECISION_API_STYLES and choice is not None:
        entry = state.proposal.entry
        if choice == "keep":
            current = state.guided_reasoning_current or {}
            for field in (
                "thinking",
                "reasoning",
                "reasoning_effort",
                "output_config",
                "chat_template_kwargs",
            ):
                if field in current:
                    entry[field] = deepcopy(current[field])
            if current.get("extra_body"):
                entry.setdefault("extra_body", {}).update(deepcopy(current["extra_body"]))
            if state.guided_reasoning_active is not None:
                entry["reasoning_level"] = state.guided_reasoning_active
        elif choice == "default":
            entry.pop("reasoning_level", None)
        else:
            entry["reasoning_level"] = state.guided_reasoning_active
        entry["provenance"]["reasoning_selection"] = {
            "source": "user",
            "choice": choice,
            "active": entry.get("reasoning_level"),
        }
    state.interface_spent = state.interfaces.tokens_charged_to_budget if state.interfaces else 0
    state.proposal = replace(
        state.proposal,
        budget_tokens=max(0, state.budget_tokens - state.tokens_spent),
    )
    if state.interfaces:
        state.proposal.entry["provenance"]["interfaces"] = {
            style: state.result.entry["provenance"]["probes"]["routing"]
            for style, state.result in state.interfaces.results.items()
        }
    if state.context_window:
        state.proposal = replace(
            state.proposal,
            entry=connect.apply_context_window(
                state.proposal.entry,
                state.context_window,
                model=state.model,
                endpoint_model=state.selected_endpoint_model,
            ),
        )
    if state.manual_limits and state.api_style not in connect.DECISION_API_STYLES:
        provenance = state.proposal.entry["provenance"]
        provenance["manual_limits"] = {"source": "user", **state.manual_limits}
        provenance["context_window"].update(
            value=state.proposal.entry["context_window"],
            declared=state.manual_limits["context_window"],
            effective=state.proposal.entry["context_window"],
        )
        # User declarations are not a public catalogue mapping or tested evidence.
        provenance["limit_sources"] = {"context_length": "user"}
        provenance.pop("catalogue", None)
        provenance["probes"] = {}
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
    candidate = connect.configure_entry(entry)
    # A separately edited configured candidate must not borrow the result's proof.
    configured = state.configured if state.configured is not None else candidate
    if connect.routing_allows_save(entry, candidate) and connect.routing_allows_save(
        entry, configured
    ):
        return True
    state.test_failed = True
    reason = view.connection_detail(record)
    # Only sanitized display facts: never echo raw provider errors or replies.
    view.note(
        scrub_setup(
            state,
            "Nothing was saved. Connection: " + reason.rstrip(".") + ". "
            "Review the server URL and settings, or increase --budget-tokens if checks were skipped. "
            "Review the current configuration before retrying or explicitly saving untested. "
            "Add --no-catalogue and an explicit model ID to avoid listing/metadata requests. "
            "See docs/model-configuration.md. No model entry or key was written.",
        ),
        err=True,
    )
    return False


def run_checks(state: WizardState) -> bool:
    """Run checks; return False when the user cancels."""
    if not state.no_probe and state.approval != "none":
        show_step(state, 4)
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
    if state.budget_tokens is None:
        state.budget_tokens = state.proposal.budget_tokens
    state.proposal = replace(
        state.proposal, budget_tokens=max(0, state.budget_tokens - state.tokens_spent)
    )
    state.result = asyncio.run(
        show_checks(
            connect.run_steps(state.proposal, approved=state.approval, api_key=state.api_key),
            reasoning_levels=state.proposal.entry.get("reasoning_levels"),
            media_inputs=state.media_inputs,
            diagnostics=state.show_diagnostics,
        )
    )
    state.tokens_spent += state.result.entry["provenance"].get("tokens_charged_to_budget", 0)
    state.result.entry["provenance"]["tokens_charged_to_budget"] = state.tokens_spent
    # Settings the server refused during the checks (and were turned off) are
    # part of what was tested, so the save candidate must carry them too.
    for name, record_key in (
        ("cache_breakpoint", "explicit_cache"),
        ("include", "encrypted_reasoning"),
    ):
        record = state.result.entry["provenance"].get(record_key) or {}
        if (
            record.get("outcome") == "rejected"
            and state.configured is not None
            and name in state.result.entry
        ):
            state.configured[name] = deepcopy(state.result.entry[name])
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
        from ._connect_input_policy import available_media

        available = available_media(state, entry=state.result.entry)
        for kind in state.test_media_inputs:
            if (
                kind in available
                and facts.get(f"media:{kind}", {}).get("outcome") == "passed"
                and kind not in enabled
            ):
                enabled.append(kind)
        state.result.entry["media_inputs"] = enabled
    decision = connect.verdict(state.result.entry)
    state.unobserved = decision.unobserved_levels
    checks = {
        **state.result.entry["provenance"]["probes"],
        **state.result.entry["provenance"].get("session_checks", {}),
    }
    baseline_ok = baseline_allows_save(state)
    if state.approval != "none" and (
        decision.needs_attention
        or any(checks[name].get("reason") != "not approved" for name in decision.skipped)
    ):
        # A failed baseline always gets the agent prompt; optional gaps keep the hint.
        if state.show_diagnostics or not baseline_ok:
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
                            state.budget_tokens - state.tokens_spent,
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
    return baseline_ok


def save_model(state: WizardState) -> bool:
    """Save model; return False when canceled or the baseline is unverified."""
    if not baseline_allows_save(state):
        return False
    show_step(state, 5)
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
            from ._connect_input_policy import available_media

            view.line(scrub_setup(state, "Not available in this test: " + ", ".join(unavailable)))
            view.line(
                scrub_setup(
                    state,
                    "Inputs not supported by this request interface cannot be enabled. Other implemented inputs may be selected manually as not tested."
                    if len(available_media(state, entry=state.result.entry)) == 2
                    else "Unavailable inputs were not tested on this connection. Keep these settings, or enable inputs manually as unverified.",
                ),
                dim=True,
            )
    from ._connect_input_policy import repair_native_media

    repaired = repair_native_media(
        state, state.result.entry.get("media_inputs", []), entry=state.result.entry
    )
    if repaired is None:
        return False
    state.result.entry["media_inputs"] = repaired
    connect.configure_entry(state.result.entry)  # local policy gate before any save side effect
    tested_media_inputs = list(state.result.entry.get("media_inputs", []))
    media_facts = state.result.entry.get("provenance", {}).get("probes", {})
    enabled_now = state.result.entry.get("media_inputs", [])
    all_passed = bool(state.test_media_inputs) and all(
        (media_facts.get(f"media:{kind}", {}).get("outcome") == "passed" and kind in enabled_now)
        or (
            media_facts.get(f"media:{kind}", {}).get("outcome") == "rejected"
            and kind not in enabled_now
        )
        for kind in state.test_media_inputs
    )
    # Nothing to decide when every tested format either passed (and is on) or
    # was refused by the server (and is off). If the model only said it couldn't
    # see a sample, the user may still choose to enable that format.
    if not state.yes and state.test_media_inputs and not all_passed:
        action = prompts.review_media_inputs(state.result.entry["media_inputs"])
        if action == "manual":
            view.line(
                scrub_setup(state, "Inputs enabled manually will be saved as not tested."),
                fg="yellow",
            )
            from ._connect_input_policy import picker_options

            selected = prompts.choose_media_inputs(
                state.result.entry["media_inputs"],
                **picker_options(state, entry=state.result.entry),
            )
            connect.configure_entry({**state.result.entry, "media_inputs": selected})
            state.result.entry["media_inputs"] = selected
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
    if not baseline_allows_save(state):
        return False
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
    expected_alias = connect.alias_target(state.data.get("models", {}), state.alias)
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
                "Reasoning: "
                + (
                    state.result.entry["reasoning_level"].title()
                    if state.result.entry.get("reasoning_level")
                    else "Keep current settings"
                    if state.guided_reasoning_choice == "keep"
                    else "Model default (no new override)"
                ),
            )
        )
    if state.api_style not in connect.DECISION_API_STYLES and (
        "cache_breakpoint" in state.result.entry and state.result.entry["cache_breakpoint"] is None
    ):
        view.fields([("Caching", "Off")])
    view.fields(scrub_setup(state, structured_output_rows(state.result.entry)))
    from nooa.unifiedllm.connect._verification import save_verification

    confidence, _ = save_verification(state.result.entry)
    view.fields([("Verification", confidence["state"] + " (recorded checks only)")])
    reviewed_entry = deepcopy(state.result.entry)
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
        if state.result.entry != reviewed_entry:
            raise ValueError("Model settings changed after review; review and retest before saving")
        if not baseline_allows_save(state):
            return False
        if save_key:
            from nooa.secrets import new_secret_env

            def model_may_be_saved():
                # The normal registry loader deliberately swallows read/parse
                # errors. Rollback needs a STRICT read: unknown is not absent.
                try:
                    saved_document = yaml.safe_load(state.path.read_text())
                except FileNotFoundError:
                    return False
                if not isinstance(saved_document, dict) or not isinstance(
                    saved_document.get("models", {}), dict
                ):
                    raise ValueError("Cannot determine whether the model was committed")
                # A concurrent edit may change parameters/alias yet still
                # depend on the new variable. Do not remove its credential.
                return any(
                    isinstance(profile, dict) and profile.get("api_key_env") == state.api_key_env
                    for profile in saved_document.get("models", {}).values()
                )

            with new_secret_env(
                secrets_path,
                state.api_key_env,
                state.api_key,
                model_may_be_saved=model_may_be_saved,
                rollback_guard=lambda: connect.registry_write_lock(state.path),
            ):
                connect.write(
                    state.result.entry, state.path, alias=state.alias, expected_alias=expected_alias
                )
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
        else:
            connect.write(
                state.result.entry, state.path, alias=state.alias, expected_alias=expected_alias
            )
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
    profile = entry.get("provenance", {}).get("structured_output_profile", {})
    rows = [
        (
            "Answer format",
            "Native JSON (recommended by this server)"
            if mode == "anthropic" and profile.get("recommended")
            else "Native JSON (your choice)"
            if mode == "anthropic"
            else "Standard JSON",
        ),
        ("Structured output", f"{mode or 'none'} (declared)"),
    ]
    if mode == "anthropic":
        baseline = entry.get("provenance", {}).get("probes", {}).get("routing", {})
        observed = (
            connect.verification_is_current(baseline, entry)
            and baseline.get("structured_output_schema_observed") is True
            and baseline.get("structured_output_conflicts_observed") is False
            and baseline.get("settings_sent") is True
        )
        rows.append(("Native JSON", "Observed" if observed else "Not confirmed"))
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
                    "Effective context bound" if state.manual_limits else "Context window",
                    f"{entry['context_window']:,} tokens"
                    if entry.get("context_window")
                    else "Not set",
                ),
                ("Reply limit", f"{entry['max_tokens']:,} tokens"),
            )
        )
        if state.manual_limits:
            rows.extend(
                [
                    (
                        "Context window (declared)",
                        f"{state.manual_limits['context_window']:,} tokens",
                    ),
                    (
                        "Input token limit (declared)",
                        f"{state.manual_limits['max_input_tokens']:,} tokens",
                    ),
                ]
            )
        if maximum:
            rows.append(("Reply maximum", f"{maximum:,} tokens"))
        if "cache_breakpoint" in entry and entry["cache_breakpoint"] is None:
            rows.append(("Caching", "Off"))
        active = entry.get("reasoning_level")
        label = active.title() if active else "Model default (no new override)"
        if state.guided_reasoning_choice == "keep" and active is None:
            label = "Keep current settings"
        rows.append(("Reasoning", label))
        rows.append(("Reasoning check", "Not tested yet"))
        if entry.get("reasoning_levels"):
            rows.append(("Other available choices", ", ".join(entry["reasoning_levels"])))
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
    if state.manual_limits and state.api_style not in connect.DECISION_API_STYLES:
        view.fields(scrub_setup(state, [("Model details", "Your settings")]))
        view.note(
            "Saved context bound is no greater than the smaller declared limit; raw limits remain in this review, not the saved registry."
        )
    elif state.edited_settings:
        view.fields(scrub_setup(state, [("Model details", "Your settings")]))
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
    selected_report = connect.advertised_media_suggestions(
        {"id": state.model, "input_modalities": list(selected)},
        api_style=state.api_style,
        decision_image_transport=state.decision_image_transport or "json",
        typesafe_danger_zone_images=state.typesafe_danger_zone_images,
    )
    unavailable = selected_report["unavailable_checks"]
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
        from ._connect_input_policy import available_media

        if kind not in available_media(state, entry=entry):
            value = (
                "Repair required; interface doesn’t support this format"
                if kind in enabled
                else "Off; interface doesn’t support this format"
            )
        elif kind in unavailable:
            value = "Enabled; test unavailable" if kind in enabled else "Off; test unavailable"
        media_rows.append((view.media_label(kind), value))
    view.fields(scrub_setup(state, media_rows))
    # Rows above already say when the interface doesn't support a format.
    for kind, reason in unavailable.items():
        if kind in available_media(state, entry=entry):
            view.fields(scrub_setup(state, [(view.media_label(kind) + " test", reason)]))
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


def edit_guided_limits(state: WizardState) -> None:
    """Change settings → Reply limit and context; reasoning is a separate choice."""
    if not state.manual_limits_deferred and (state.manual_limits or state.candidate is None):
        require_manual_limits(state, edit=True)
        maximum = (
            state.proposal.entry.get("provenance", {})
            .get("catalogue_limits", {})
            .get("max_completion_tokens")
        )
        limits = [state.context_window - 1]
        if isinstance(maximum, int) and not isinstance(maximum, bool) and maximum > 0:
            limits.append(maximum)
        else:
            maximum = None
        ceiling = min(limits)
        state.reply_tokens = prompts.choose_reply_limit(
            min(state.proposal.entry["max_tokens"], ceiling),
            ceiling,
            output_ceiling=maximum,
            source="user",
        )
    else:
        current = deepcopy(state.candidate or {})
        current.setdefault("id", state.model)
        entry = state.proposal.entry
        current["context_length"] = entry.get("context_window")
        current["top_provider"] = {
            "max_completion_tokens": entry.get("max_output_tokens")
            or entry.get("provenance", {}).get("catalogue_limits", {}).get("max_completion_tokens")
        }
        maximum = current["top_provider"]["max_completion_tokens"]
        if not isinstance(maximum, int) or isinstance(maximum, bool) or maximum <= 0:
            current["top_provider"]["max_completion_tokens"] = None
        current["reasoning"] = {
            "supported_efforts": list(entry.get("reasoning_levels", {})),
            "default_effort": entry.get("reasoning_default"),
        }
        state.candidate = prompts.edit_model_details(current, reasoning=False)
        state.context_window = state.candidate.get("context_length")
        state.edited_settings = True
        state.edited_level_patches = deepcopy(entry.get("reasoning_levels"))
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


def edit_guided_decision_media(state: WizardState) -> bool:
    """Change settings → Media for decision interfaces (images only)."""
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
        return False
    return True


def edit_guided_media(state: WizardState) -> bool:
    """Change settings → Media: pick formats, then test or enable them as usual."""
    if state.api_style in connect.DECISION_API_STYLES:
        return edit_guided_decision_media(state)
    current = (state.proposal.entry.get("media_inputs") if state.proposal else None) or []
    select_manual_media(state, list(current))
    return True


def change_settings_options(state: WizardState):
    """The plain Change settings submenu, in order, with only options that apply."""
    decision = state.api_style in connect.DECISION_API_STYLES
    options = []
    if not decision:
        options += [("limits", "Reply limit and context"), ("reasoning", "Reasoning")]
    options.append(("media", "Media"))
    if state.api_style in {"chat", "anthropic"}:
        options.append(("structured", "Answer format"))
    if (
        not decision
        and state.candidate is not None
        and state.editing is None
        and not state.manual_limits_required
        and not state.manual_limits
    ):
        options.append(("skip", "Continue without catalogue settings"))
    options.append(("back", "Back"))
    return tuple(options), "media" if decision else "limits"


def review_guided_proposal(state: WizardState, *, deferred_style=False):
    """Consent or edit; edits are re-summarized before the next decision."""
    while True:
        options = (("test", "Test now"),) if not state.no_probe and state.probe != "none" else ()
        # Keep the main decision short; every edit lives under Change settings.
        # Ctrl-C cancels without saving.
        action = prompts.labelled_choice(
            "Would you like to test these settings?",
            (*options, ("save", "Save without testing"), ("change", "Change settings")),
            default="test" if options else "save",
        )
        if action == "cancel":
            view.note("Setup cancelled. Nothing was saved.")
            return None
        if action == "save" and state.test_failed:
            snapshot = connect.effective_config_snapshot(state.proposal.entry)
            if state.failure_ack_snapshot != snapshot:
                if not prompts.confirm(
                    "An approved connection test failed. Save current settings explicitly as untested anyway?",
                    default=False,
                ):
                    continue
                state.failure_ack_snapshot = snapshot
        if action != "change":
            break
        change_options, change_default = change_settings_options(state)
        change = prompts.labelled_choice(
            "What would you like to change?", change_options, default=change_default
        )
        if change == "cancel":
            view.note("Setup cancelled. Nothing was saved.")
            return None
        if change == "back":
            continue
        if change == "structured":
            state.structured_output_mode = prompts.choose_structured_output_mode(
                state.proposal.entry.get("structured_output_mode")
            )
            state.structured_output_mode_set = True
        elif change == "skip":
            discard_catalogue_settings(state)
        elif change == "media":
            if not edit_guided_media(state):
                return None
        elif change == "reasoning":
            if not choose_reasoning(state, deferred=deferred_style, edit=True):
                return None
        else:
            edit_guided_limits(state)
            # Profile/media declarations and explicit test selections remain unchanged.
        if not configure_checks(state):
            return None
        show_guided_proposal(state, deferred_style=deferred_style, updated=True)
    return action


def guided_setup(state: WizardState) -> bool:
    """Prepare a proposal using GETs only; the action below is model-call consent."""
    if not select_connection(state) or not select_model(state) or not prepare_metadata(state):
        return False
    if not state.metadata_displayed:
        show_model_metadata(state)  # model details close the Model step
    show_step(state, 3)
    if not select_request_interface(state):
        return False
    deferred_style = state.api_style is None
    if deferred_style and state.structured_output_mode_set:
        raise click.UsageError(
            "--structured-output-mode requires --api-style chat or anthropic, not interface discovery"
        )
    explicit_reply_tokens = state.reply_tokens
    if deferred_style and not state.no_probe and state.probe != "none":
        # Choosing Discover approves these small interface checks. Run them now
        # so media and reasoning are set up for the interface actually chosen.
        state.approval = state.probe
        if not check_interfaces(state):
            return False
        if state.api_style in connect.DECISION_API_STYLES:
            state.reply_tokens = explicit_reply_tokens
            discard_chat_limits(state)
        deferred_style = False
    # A provider preset is a proposal, never evidence of supported interfaces.
    state.api_style = state.api_style or state.default_style
    # Unknown interfaces already have their provisional literal limits upfront.
    state.manual_limits_deferred = deferred_style and known_decision_model(state)
    if (
        not configure_media_inputs(state)
        or not choose_reasoning(state, deferred=deferred_style)
        or not configure_checks(state)
    ):
        return False
    # Revisions made before this first proposal are already shown in it; only
    # later revisions need a second review.
    state.route_revised = state.guided_media_revised = False
    show_guided_proposal(state, deferred_style=deferred_style)
    action = review_guided_proposal(state, deferred_style=deferred_style)
    if action is None:
        return False
    approved_snapshot = connect.effective_config_snapshot(state.proposal.entry)
    state.approval = state.probe if action == "test" else "none"
    if action == "save":
        keep_untested_media(state)
    elif not state.no_probe and state.approval != "none":
        show_step(state, 4)  # approved discovery checks are part of the test
    if deferred_style:
        state.manual_limits_deferred = False
        state.api_style = None
        if not check_interfaces(state):
            return False
        if state.api_style in connect.DECISION_API_STYLES:
            state.reply_tokens = explicit_reply_tokens
            discard_chat_limits(state)
            if not configure_media_inputs(state):
                return False
        # No provisional native control can carry to another discovered interface.
        if not choose_reasoning(state):
            return False
        # Re-plan for the selected route using the already reviewed limits.
        if not configure_checks(state):
            return False
    elif action == "save":
        if not configure_checks(state):
            return False
    if (
        state.route_revised
        or state.guided_media_revised
        or deferred_style
        and connect.effective_config_snapshot(state.proposal.entry) != approved_snapshot
    ):
        show_guided_proposal(state, updated=True)
        action = review_guided_proposal(state)
        if action is None:
            return False
        state.approval = state.probe if action == "test" else "none"
        if action == "save":
            keep_untested_media(state)
        if not configure_checks(state):
            return False
    if not run_checks(state):
        # The failure and an agent prompt are already shown.
        return offer_restart(state)
    return save_model(state)


def new_state(options) -> WizardState:
    state = WizardState(**options)
    state.path = Path(state.output) if state.output else paths.get_user_dir("llm_config.yaml")
    state.api_key = None
    state.explicit_key_env = state.api_key_env is not None
    state.editing = None
    return state


def run_wizard(**options):
    """Run ordered steps with one shared budget and an explicit save decision."""
    state = new_state(options)
    try:
        if not state.yes:
            while True:
                state.guided = True
                state.guided_media_explicit = (
                    state.media_inputs is not None or state.test_media_inputs is not None
                )
                try:
                    succeeded = guided_setup(state)
                    break
                except RestartSetup:
                    click.echo()
                    view.note("Starting again.")
                    state = new_state(options)
            if not succeeded and state.test_failed:
                view.note(
                    "Approved connection test failed or was cancelled. Nothing saved.", err=True
                )
                raise click.ClickException("Test failed.")
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
                if state.test_failed:
                    view.note("Approved connection test failed. Nothing saved.", err=True)
                    raise click.ClickException("Test failed.")
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
