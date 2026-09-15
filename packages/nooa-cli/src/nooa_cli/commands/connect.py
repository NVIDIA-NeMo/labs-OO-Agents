# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Terminal approval and display for the shared nooa.connect library."""

import click

from ._connect_stages import STAGES


@click.command()
@click.argument("model", required=False)
@click.option(
    "--edit-model",
    is_flag=False,
    flag_value="",
    default=None,
    help="Edit an existing alias; omit NAME to select from the registry.",
)
@click.option(
    "--stage", type=click.Choice(STAGES), help="Run one non-interactive stage; emit JSON and exit."
)
@click.option(
    "--input",
    "input_file",
    type=click.Path(exists=True, dir_okay=False),
    help="JSON plan/result to save with --stage save.",
)
@click.option(
    "--provider",
    help="Connection preset: nvidia, openai, anthropic, google, openrouter, or custom.",
)
@click.option("--as", "alias", help="Local model alias to save (otherwise prompted).")
@click.option("--endpoint", help="API base URL (otherwise prompted).")
@click.option("--api-style", type=click.Choice(["chat", "responses", "anthropic"]))
@click.option(
    "--api-key-env",
    help="Environment variable name, never the key itself (otherwise prompted).",
)
@click.option("--catalogue-model", help="Explicit OpenRouter model ID to use as metadata.")
@click.option("--prompt-key", is_flag=True, help="Read a masked key; offer to save it at the end.")
@click.option("--no-catalogue", is_flag=True, help="Do not fetch public model metadata.")
@click.option(
    "--reasoning-template",
    type=click.Choice(["effort", "adaptive", "budget", "toggle", "thinking"]),
)
@click.option("--levels", help="Comma-separated candidate labels to probe.")
@click.option(
    "--levels-file",
    type=click.Path(exists=True, dir_okay=False),
    help="YAML mapping from labels to complete request settings.",
)
@click.option(
    "--context-window",
    type=click.IntRange(min=1),
    help="User-supplied context limit, not a request allocation.",
)
@click.option(
    "--probe", type=click.Choice(["all", "minimal", "none"]), default="all", show_default=True
)
@click.option("--no-probe", is_flag=True, help="Save an untested entry without model calls.")
@click.option(
    "--budget-tokens",
    type=click.IntRange(min=1),
    help="Shared estimated-token budget for all checks (default: 131072); never increased after approval.",
)
@click.option("--output-tokens", type=click.IntRange(1, 4096), default=200, show_default=True)
@click.option(
    "--reasoning-output-tokens",
    type=click.IntRange(1, 32768),
    default=4096,
    show_default=True,
    help="Output cap for each reasoning check, including thinking; separate from basic checks.",
)
@click.option(
    "--max-tokens",
    "--reply-tokens",
    "reply_tokens",
    type=click.IntRange(min=1),
    help="Reply budget saved for agents (not the smaller setup-check cap).",
)
@click.option("--show-config", is_flag=True, help="Show full YAML details before saving.")
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    help="Registry path; defaults to the user llm_config.yaml.",
)
@click.option(
    "--yes",
    is_flag=True,
    help="Approve the checks and save without prompting; supply the connection options.",
)
def command(
    model,
    edit_model,
    stage,
    input_file,
    provider,
    alias,
    endpoint,
    api_style,
    api_key_env,
    prompt_key,
    catalogue_model,
    no_catalogue,
    reasoning_template,
    levels,
    levels_file,
    context_window,
    probe,
    no_probe,
    budget_tokens,
    output_tokens,
    reasoning_output_tokens,
    reply_tokens,
    show_config,
    output,
    yes,
):
    """Walk through model setup, check the connection, and save an alias.

    Run `uv run nooa connect` with no arguments for guided setup. Flags prefill
    the answers; --yes requires MODEL, --as and either --provider or an
    explicit --endpoint and --api-style.
    MODEL is the exact endpoint model ID, without a LiteLLM routing prefix.
    """
    if stage:
        from ._connect_stages import run_stage

        code = run_stage(
            stage,
            model=model,
            alias=alias,
            endpoint=endpoint,
            api_style=api_style,
            api_key_env=api_key_env,
            budget_tokens=budget_tokens,
            output_tokens=output_tokens,
            reasoning_output_tokens=reasoning_output_tokens,
            reply_tokens=reply_tokens,
            levels_file=levels_file,
            context_window=context_window,
            input_file=input_file,
            output=output,
            yes=yes,
            prompt_key=prompt_key,
            invalid_options=[
                name
                for name, used in {
                    "--no-probe": no_probe,
                    "--probe": probe != "all",
                    "--provider": provider,
                    "--catalogue-model": catalogue_model,
                    "--reasoning-template": reasoning_template,
                    "--levels": levels,
                    "--show-config": show_config,
                    "--edit-model": edit_model is not None,
                }.items()
                if used
            ],
        )
        raise click.exceptions.Exit(code)
    if input_file:
        raise click.UsageError("--input requires --stage save")
    import asyncio
    import os
    from contextlib import aclosing
    from copy import deepcopy
    from dataclasses import replace
    from pathlib import Path

    import httpx
    import yaml

    from nooa import connect
    from nooa.paths import get_user_dir

    from . import _connect_view as view
    from ._connect_prompts import (
        choose_reply_limit,
        confirm,
        edit_model_details,
        environment_names,
        prompt,
    )
    from ._connect_registry import credential_names, diagnostic_context, entries, shadowing_source

    async def show_checks(events, *, reasoning_levels=None):
        with view.quiet_provider_messages():
            return await display_checks(events, reasoning_levels=reasoning_levels)

    async def display_checks(events, *, reasoning_levels=None):
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
            progress.finish()
        raise click.ClickException("Checks ended without a result.")

    path = Path(output) if output else get_user_dir("llm_config.yaml")
    api_key = None
    explicit_key_env = api_key_env is not None
    editing = None
    try:
        registry = entries(path if output else None)
        if edit_model is not None:
            if any(
                (model, provider, endpoint, api_style, catalogue_model, reasoning_template, levels)
            ):
                raise click.UsageError(
                    "--edit-model cannot also select a new connection or catalogue template"
                )
            if edit_model == "":
                if yes:
                    raise click.UsageError("With --yes, --edit-model requires an alias")
                if not registry:
                    raise click.ClickException(
                        "No registry models to edit. Run nooa connect to add one."
                    )
                edit_model = prompt("Model to edit", choices=sorted(registry), open_menu=True)
            if edit_model not in registry:
                raise click.ClickException(f"Unknown registry model {edit_model!r}")
            editing, source_path = registry[edit_model]
            editing = deepcopy(editing)
            if not output:
                from nooa.llm_config import bundled_config_paths

                if source_path.resolve() not in {p.resolve() for p in bundled_config_paths()}:
                    path = source_path
            alias = alias or edit_model
            routed = editing.get("model_name", edit_model)
            api_style = editing.get("api_style") or (
                "responses"
                if editing.get("client_type") == "responses"
                else "anthropic"
                if routed.startswith("anthropic/")
                else "chat"
            )
            prefix = "anthropic/" if api_style == "anthropic" else "openai/"
            model = routed.removeprefix(prefix)
            endpoint = (
                editing.get("api_base")
                or connect.PROVIDERS["anthropic" if api_style == "anthropic" else "openai"].api_base
            )
            if api_key_env is None:
                api_key_env = editing.get("api_key_env", "")
            explicit_key_env = True
            no_catalogue = True
            view.line(f"Editing {edit_model} from {source_path}. No model discovery is needed.")
        data = {}
        if path.exists():
            with path.open() as source:
                data = yaml.safe_load(source) or {}
        if not isinstance(data, dict) or not isinstance(data.get("models", {}), dict):
            raise click.ClickException("Registry must contain a models mapping.")
        for name, entry in data.get("models", {}).items():
            if isinstance(name, str) and isinstance(entry, dict):
                registry.setdefault(name, (entry, path))
        server_urls = [p.api_base for p in connect.PROVIDERS.values()]
        for entry, _ in registry.values():
            address = entry.get("api_base") if isinstance(entry, dict) else None
            if isinstance(address, str):
                try:
                    server_urls.append(connect.normalize_endpoint(address))
                except ValueError:
                    pass  # Do not offer malformed URLs or embedded credentials.
        server_urls = list(dict.fromkeys(server_urls))
        default_style = "chat"
        approval = "none" if no_probe else probe
        budget_tokens = connect.DEFAULT_CHECK_BUDGET if budget_tokens is None else budget_tokens
        interfaces = None
        view.intro(
            checks=approval != "none",
            output_tokens=output_tokens,
            budget_tokens=budget_tokens,
            reasoning_output_tokens=reasoning_output_tokens,
        )
        if approval != "none" and not yes:
            if not confirm("Approve API checks within this budget?", default=True):
                click.echo("No API checks approved. Run with --no-probe for manual setup.")
                return
        if editing is None:
            view.step(1, "Connection")
        if provider and provider not in (*connect.PROVIDERS, "custom"):
            raise click.UsageError(
                "Unknown provider. Choose nvidia, openai, anthropic, google, openrouter, or custom."
            )
        if not provider and not endpoint and not yes:
            provider = prompt(
                "Choose a provider",
                choices=(*connect.PROVIDERS, "custom"),
                labels={
                    **{name: preset.label for name, preset in connect.PROVIDERS.items()},
                    "custom": "Custom endpoint",
                },
                open_menu=True,
            )
        if provider and provider != "custom":
            preset = connect.PROVIDERS[provider]
            endpoint = endpoint or preset.api_base
            default_style = preset.api_style
            if api_key_env is None:
                api_key_env = preset.api_key_env
        if yes and provider and provider != "custom":
            api_style = api_style or default_style
        if yes and not all((model, alias, endpoint, api_style)):
            raise click.UsageError("With --yes supply MODEL, --endpoint, --api-style and --as.")
        endpoint = endpoint or prompt("Model server URL", suggestions=server_urls, open_menu=True)
        endpoint = connect.normalize_endpoint(endpoint)
        if not explicit_key_env:
            saved_names = credential_names(registry, endpoint)
            if len(saved_names) == 1:
                api_key_env = saved_names[0]
                view.line(
                    f"Using saved key variable {api_key_env or '(no authentication)'} for this endpoint."
                )
            elif len(saved_names) > 1:
                if yes:
                    raise click.UsageError(
                        "Several key variables are saved for this endpoint; supply --api-key-env"
                    )
                api_key_env = prompt(
                    "Saved key variable",
                    choices=[name or "-" for name in saved_names],
                    open_menu=True,
                )
                if api_key_env == "-":
                    api_key_env = ""
        # Listing/authentication conventions do not choose the selected model's
        # generation interface. A mixed server can list all models via /models.
        discovery_style = api_style or default_style
        if api_key_env is None:
            default_env = (
                "ANTHROPIC_API_KEY" if discovery_style == "anthropic" else "OPENAI_API_KEY"
            )
            api_key_env = (
                default_env
                if yes
                else prompt(
                    "Key environment variable (new to enter a key; - for no authentication)",
                    default=default_env,
                    suggestions=environment_names(
                        [p.api_key_env for p in connect.PROVIDERS.values()] + ["-", "new"]
                    ),
                )
            )
            if api_key_env == "-":
                api_key_env = ""
        if api_key_env == "new":
            api_key_env = prompt(
                "Save key under variable name",
                default="NOOA_MODEL_API_KEY",
                suggestions=environment_names(),
            )
            prompt_key = True
        # Validate before using an endpoint or collecting a credential.
        connect.plan(
            alias or "candidate", model or "candidate", discovery_style, endpoint, api_key_env
        )
        needs_key = (
            not yes and approval != "none" and api_key_env and not os.environ.get(api_key_env)
        )
        api_key = (
            prompt("API key (used only for this setup)", hide_input=True)
            if prompt_key or needs_key
            else os.environ.get(api_key_env)
            if api_key_env
            else None
        )
        if editing is None:
            view.step(2, "Model")
        discovery_succeeded = None
        discovery_endpoint = None
        if not model:
            click.echo("Connecting to the server and listing models...")
            try:
                found = asyncio.run(
                    connect.discover(endpoint, api_style=discovery_style, api_key=api_key)
                )
            except connect.DiscoveryError as exc:
                discovery_succeeded = False
                discovery_endpoint = endpoint
                click.echo(f"Could not list models: {exc}", err=True)
                if exc.status_code in {401, 403}:
                    raise click.ClickException(
                        "Authentication failed. Check the key and try again."
                    ) from None
                model = prompt("Exact model ID (if known; Ctrl-C to cancel)")
            else:
                endpoint = found.api_base
                discovery_succeeded = True
                discovery_endpoint = endpoint
                names = [item["id"] for item in found.models]
                click.echo(
                    f"Server listed {len(names)} model(s). Credentials are checked next. Type part of a name to search, then Tab to select."
                )
                model = prompt("Model", choices=names)
        view.step(3, "Connection checks")
        if api_style == "responses":
            view.line(connect.ENCRYPTED_REASONING_EXPLANATION, dim=True)
        if not api_style:
            available = ("chat", "responses", "anthropic")
            if approval != "none":
                interface_spent = 0
                while True:
                    interfaces = asyncio.run(
                        show_checks(
                            connect.check_interfaces(
                                alias or "candidate",
                                model,
                                endpoint,
                                api_key_env,
                                budget_tokens=max(0, budget_tokens - interface_spent),
                                output_tokens=output_tokens,
                                api_key=api_key,
                            )
                        )
                    )
                    interface_spent += interfaces.tokens_charged_to_budget
                    interfaces = replace(interfaces, tokens_charged_to_budget=interface_spent)
                    available = interfaces.accepted
                    if available:
                        break
                    view.line(
                        "Could not confirm a working connection. Listing models does not validate the key.",
                        fg="yellow",
                    )
                    failed_checks = {
                        style: r.entry["provenance"]["probes"]["routing"]
                        for style, r in interfaces.results.items()
                    }
                    click.echo(
                        "Agent diagnostic prompt:\n"
                        + connect.diagnostic_prompt(
                            "interfaces",
                            {"model_name": model, "api_base": endpoint, "api_key_env": api_key_env},
                            failed_checks,
                            run_context=diagnostic_context(
                                target=path,
                                alias=alias,
                                model=model,
                                endpoint=endpoint,
                                api_key_env=api_key_env,
                                api_key=api_key,
                                budget=budget_tokens,
                                remaining=max(0, budget_tokens - interface_spent),
                                output_tokens=output_tokens,
                                reasoning_output_tokens=reasoning_output_tokens,
                                stage="interfaces",
                                discovery_succeeded=discovery_succeeded
                                if endpoint == discovery_endpoint
                                else None,
                            ),
                        )
                    )
                    if yes:
                        raise click.ClickException(
                            "Check credentials and endpoint, or run without --yes to correct them interactively."
                        )
                    remaining = max(0, budget_tokens - interface_spent)
                    if remaining < output_tokens + 512:
                        view.line(
                            "The approved check budget is exhausted. Nothing was saved; restart setup to approve a new budget."
                        )
                        return
                    view.line(
                        f"You can correct the connection here. {remaining:,} estimated tokens remain in the approved budget."
                    )
                    action = prompt(
                        "Next step",
                        choices=("key", "server", "retry", "cancel"),
                        default="key",
                        labels={
                            "key": "Change key",
                            "server": "Edit server and model",
                            "retry": "Try again unchanged",
                            "cancel": "Exit without saving",
                        },
                    )
                    if action == "cancel":
                        click.echo("Setup cancelled. Nothing was saved.")
                        return
                    if action == "key":
                        source = prompt(
                            "Key environment variable (or paste for a temporary key)",
                            default=api_key_env or "paste",
                            suggestions=environment_names(["paste"]),
                        )
                        if source == "paste":
                            api_key = prompt("API key (used only for this setup)", hide_input=True)
                        else:
                            api_key_env = source
                            api_key = os.environ.get(source)
                            if not api_key:
                                view.line(
                                    "That variable is unset or empty. You can paste a temporary key instead."
                                )
                                api_key = prompt(
                                    "API key (used only for this setup)", hide_input=True
                                )
                    elif action == "server":
                        endpoint = connect.normalize_endpoint(
                            prompt(
                                "Model server URL",
                                default=endpoint,
                                suggestions=server_urls,
                                open_menu=True,
                            )
                        )
                        model = prompt("Exact model ID", default=model)
                click.echo(
                    "Interfaces that returned the expected response format: " + ", ".join(available)
                )
            if interfaces and len(available) == 1:
                api_style = available[0]
                click.echo(f"Using {api_style} for {model}.")
            else:
                click.echo(f"Choose the request interface for {model}:")
                click.echo(
                    "chat = OpenAI-compatible; responses = OpenAI Responses; anthropic = Anthropic Messages."
                )
                api_style = prompt(
                    "API format",
                    choices=available,
                    default=default_style if default_style in available else available[0],
                )
            if api_style == "responses":
                view.line(connect.ENCRYPTED_REASONING_EXPLANATION, dim=True)
        # An explicit --as can reuse its previous evidence. Otherwise checks use
        # a temporary label; the user names the entry only when ready to save.
        existing = editing or (data.get("models", {}).get(alias) if alias else None)
        candidate = None
        edited_settings = False
        if editing is not None:
            candidate = {
                "id": editing.get("underlying_model", model),
                "context_length": editing.get("context_window"),
                "top_provider": {
                    "max_completion_tokens": editing.get(
                        "max_output_tokens",
                        editing.get("provenance", {})
                        .get("catalogue_limits", {})
                        .get("max_completion_tokens"),
                    )
                },
                "reasoning": {
                    "supported_efforts": list(editing.get("reasoning_levels", {})),
                    "default_effort": editing.get("reasoning_default"),
                },
            }
            if not yes:
                candidate = edit_model_details(candidate)
            edited_settings = True
        if no_catalogue and catalogue_model:
            raise click.UsageError("--catalogue-model cannot be used with --no-catalogue")
        if not no_catalogue:
            click.echo("Looking up public model information...")
            try:
                models = asyncio.run(connect.catalogue())
            except (httpx.HTTPError, ValueError, KeyError):
                if catalogue_model:
                    raise click.ClickException(
                        "Could not load the requested catalogue entry."
                    ) from None
                click.echo(
                    "Public catalogue unavailable; continuing with unknown limits.", err=True
                )
                models = []
            matches = (
                [item for item in models if item.get("id") == catalogue_model]
                if catalogue_model
                else connect.match_models(model, models)
            )
            if catalogue_model and not matches:
                raise click.ClickException("Requested catalogue model was not found.")
            if len(matches) == 1:
                candidate = matches[0]
            elif matches:
                click.echo(
                    "Possible catalogue models: " + ", ".join(item["id"] for item in matches)
                )
                if yes:
                    raise click.ClickException(
                        "Ambiguous match: choose --catalogue-model or --no-catalogue."
                    )
                selected = prompt(
                    "Catalogue model (blank leaves it unknown)",
                    default="",
                    show_default=False,
                    choices=[""] + [item["id"] for item in matches],
                )
                if selected:
                    candidate = next((item for item in matches if item["id"] == selected), None)
                    if candidate is None:
                        raise click.ClickException("Choose one of the displayed model IDs.")
            else:
                click.echo("No catalogue match; model limits and reasoning levels remain unknown.")
        if candidate is not None:
            while True:
                view.model_details(candidate, output_tokens=output_tokens, edited=edited_settings)
                action = (
                    "use"
                    if yes
                    else prompt(
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
                    return
                if action == "skip":
                    if editing is not None:
                        click.echo("Edits discarded. Nothing saved.")
                        return
                    candidate = None
                    edited_settings = False
                    click.echo(
                        "Continuing without the published model settings. Explicit command-line settings still apply."
                    )
                    break
                if action == "use":
                    break
                candidate = edit_model_details(candidate)
                edited_settings = True
            if edited_settings and editing is None:
                # Apply edits only after confirmation, not if the user skips them.
                context_window = None
                levels_file = levels = reasoning_template = None
        if levels_file and (levels or reasoning_template):
            raise click.UsageError(
                "Use either --levels-file or --reasoning-template with --levels."
            )
        patches = None
        if editing is not None:
            labels = candidate.get("reasoning", {}).get("supported_efforts", [])
            original_levels = editing.get("reasoning_levels", {})
            if not levels_file and any(label not in original_levels for label in labels):
                raise click.UsageError(
                    "New reasoning levels need request settings; supply --levels-file"
                )
            patches = {label: deepcopy(original_levels[label]) for label in labels}
        if levels_file:
            with Path(levels_file).open() as source:
                patches = yaml.safe_load(source)
        if levels or reasoning_template:
            if not reasoning_template or not levels:
                raise click.UsageError(
                    "--reasoning-template and --levels must be supplied together."
                )
            patches = {
                label.strip(): connect.reasoning_settings(
                    reasoning_template, api_style, label.strip()
                )
                for label in levels.split(",")
            }
        proposal = connect.plan(
            alias or "candidate",
            model,
            api_style,
            endpoint,
            api_key_env,
            catalogue=candidate,
            reasoning_levels=patches,
            budget_tokens=budget_tokens,
            output_tokens=output_tokens,
            reasoning_output_tokens=reasoning_output_tokens,
            existing_entry=interfaces.results[api_style].entry if interfaces else existing,
            session_checks=approval == "all",
            reply_tokens=reply_tokens,
        )
        if editing is not None:
            # Preserve transport controls, custom parameters and exact level blocks.
            merged = deepcopy(editing)
            for field in (
                "context_window",
                "max_output_tokens",
                "reasoning_levels",
                "reasoning_default",
            ):
                merged.pop(field, None)
                if field in proposal.entry:
                    merged[field] = deepcopy(proposal.entry[field])
            merged["api_key_env"] = api_key_env
            merged.setdefault("api_style", api_style)
            if proposal.entry.get("allowed_openai_params"):
                merged["allowed_openai_params"] = sorted(
                    set(merged.get("allowed_openai_params", []))
                    | set(proposal.entry["allowed_openai_params"])
                )
            if "reasoning_levels" not in editing and not merged.get("reasoning_levels"):
                merged.pop("reasoning_levels", None)
            merged["provenance"] = {
                **deepcopy(editing.get("provenance", {})),
                **proposal.entry["provenance"],
            }
            merged["provenance"]["probes"] = {}  # Edits must not reuse stale evidence.
            proposal = replace(proposal, entry=merged)
            if api_style == "responses":
                for check in proposal.probes:
                    for field in ("store", "include"):
                        check.body.pop(field, None)
                        if field in merged:
                            check.body[field] = deepcopy(merged[field])
        if edited_settings:
            for field in (
                "context_window",
                "max_output_tokens",
                "reasoning_levels",
                "reasoning_default",
            ):
                proposal.entry["provenance"][field] = {
                    "source": "user",
                    "value": proposal.entry.get(field),
                }
        interface_spent = interfaces.tokens_charged_to_budget if interfaces else 0
        # Once the model is selected we know how many levels need checking.
        # An explicit user limit stays shared and is never increased.
        remaining_estimate = sum(
            p.token_estimate
            for p in proposal.probes
            if (approval == "all" or approval == "minimal" and p.name == "routing")
            and not (
                proposal.entry["provenance"]["probes"].get(p.name, {}).get("outcome") == "accepted"
                and proposal.entry["provenance"]["probes"][p.name].get("request") == p.body
            )
        )
        proposal = replace(
            proposal,
            budget_tokens=max(0, budget_tokens - interface_spent),
        )
        if interfaces:
            proposal.entry["provenance"]["interfaces"] = {
                style: result.entry["provenance"]["probes"]["routing"]
                for style, result in interfaces.results.items()
            }
        if context_window:
            proposal.entry["context_window"] = context_window
            proposal.entry["provenance"]["context_window"] = {
                "source": "user",
                "value": context_window,
            }
        if patches:
            proposal.entry["provenance"]["reasoning_levels"] = {
                "source": "user",
                "template": reasoning_template,
            }
        if "context_window" not in proposal.entry:
            click.echo(
                "No context window selected. The runtime will use its fallback; set --context-window to supply a limit."
            )
        configured = connect.configure_entry(proposal.entry, reply_tokens=reply_tokens)
        if reply_tokens is None and not yes:
            bounds = [
                v
                for v in (
                    configured.get("context_window"),
                    configured["provenance"]
                    .get("catalogue_limits", {})
                    .get("max_completion_tokens"),
                )
                if isinstance(v, int) and v > 0
            ]
            chosen_cap = choose_reply_limit(
                configured["max_tokens"],
                min(bounds) if bounds else None,
                source=configured["provenance"]["reply_limit"]["source"],
            )
            configured = connect.configure_entry(configured, reply_tokens=chosen_cap)
        proposal = replace(proposal, entry=configured)
        for check in proposal.probes:
            if check.name.startswith("level:"):
                check.body.update(configured["reasoning_levels"][check.name.removeprefix("level:")])
        if proposal.session_checks:
            from nooa._connect_session import reply_budget

            remaining_estimate += (
                3 * reply_budget(configured, max(0, proposal.budget_tokens - remaining_estimate))[2]
            )
        view.line(
            f"Saved reply budget: {configured['max_tokens']:,} tokens, shared by reasoning and the answer."
        )
        for label, limit in configured["provenance"].get("level_reply_limits", {}).items():
            view.line(
                f"Reasoning level {label}: reply budget raised to {limit['value']:,} tokens to leave room for its thinking budget.",
                fg="yellow",
            )
        price = (
            "unknown"
            if proposal.price_estimate is None
            else f"~${proposal.price_estimate:.6f} at catalogue prices"
        )
        view.line("Check plan", fg="bright_cyan", bold=True)
        view.line(
            f"Connection · tools · {len(proposal.entry.get('reasoning_levels', {}))} reasoning settings",
            dim=True,
        )
        if proposal.session_checks:
            view.line(
                "Then 3 conversation replies to check cache reuse and reasoning retention.",
                dim=True,
            )
        view.line(
            f"Reply caps: {output_tokens:,} for connection/tool checks; {reasoning_output_tokens:,} for reasoning checks; conversation checks use the saved cap when the approved budget allows, otherwise at most 2,048.",
            dim=True,
        )
        view.line(
            f"Estimated tokens: {remaining_estimate:,} · budget remaining: {proposal.budget_tokens:,} · estimated price: {price}",
            dim=True,
        )
        if remaining_estimate > proposal.budget_tokens:
            click.echo(
                "Warning: the approved budget is too small for all checks. Some will be skipped. Restart with a larger --budget-tokens value to run them all.",
                err=True,
            )
        click.echo(
            "Only truncated conversation replies retry, within the approved budget. No network-error retries or capacity probes. Estimates are not billing limits: endpoints can ignore output caps."
        )
        result = asyncio.run(
            show_checks(
                connect.run_steps(proposal, approved=approval, api_key=api_key),
                reasoning_levels=proposal.entry.get("reasoning_levels"),
            )
        )
        result.entry["provenance"]["tokens_charged_to_budget"] = (
            result.entry["provenance"].get("tokens_charged_to_budget", 0) + interface_spent
        )
        skipped = [
            name
            for name, record in result.entry["provenance"]["probes"].items()
            if record.get("reason") == "budget exhausted"
        ]
        if skipped:
            click.echo(
                "Warning: setup is incomplete; budget exhausted before "
                + ", ".join(skipped)
                + ". These settings have not been checked.",
                err=True,
            )
        unobserved = connect.unobserved_reasoning_levels(result.entry)
        checks = {
            **result.entry["provenance"]["probes"],
            **result.entry["provenance"].get("session_checks", {}),
        }
        if approval != "none" and (
            unobserved
            or any(
                r.get("outcome") in {"rejected", "not_confirmed", "not_probed"}
                and r.get("reason") != "not approved"
                for r in checks.values()
            )
        ):
            click.echo(
                "Agent diagnostic prompt:\n"
                + connect.diagnostic_prompt(
                    "checks",
                    result.entry,
                    checks,
                    run_context=diagnostic_context(
                        target=path,
                        alias=alias,
                        model=model,
                        endpoint=endpoint,
                        api_key_env=api_key_env,
                        api_key=api_key,
                        budget=budget_tokens,
                        remaining=max(
                            0,
                            proposal.budget_tokens
                            - result.entry["provenance"].get("tokens_charged_to_budget", 0),
                        ),
                        output_tokens=output_tokens,
                        reasoning_output_tokens=reasoning_output_tokens,
                    ),
                )
            )
        if unobserved:
            click.echo(
                "Warning: no reasoning information was returned for: "
                + ", ".join(unobserved)
                + ". These checks have not confirmed reasoning for those levels. "
                "Try another API format or review the server's reasoning settings. "
                "Some servers do not expose reasoning information.",
                err=True,
            )
        view.step(4, "Save model")
        # Checks may take a while: refresh names before offering completion or
        # asking to replace an entry added since setup started.
        data = {}
        if path.exists():
            with path.open() as source:
                data = yaml.safe_load(source) or {}
        if not isinstance(data, dict) or not isinstance(data.get("models", {}), dict):
            raise click.ClickException("Registry must contain a models mapping.")
        while not alias or not alias.strip():
            alias = prompt(
                "Save this model as",
                default=model.rsplit("/", 1)[-1],
                suggestions=list(data.get("models", {})) + [model.rsplit("/", 1)[-1]],
                existing=tuple(data.get("models", {})),
            )
            if not alias.strip():
                click.echo("Enter a non-empty model name.", err=True)
        if alias in data.get("models", {}):
            click.echo(f"Warning: saving will overwrite model {alias!r} in {path}.", err=True)
            if not yes and not confirm("Replace this model?", default=False):
                return
        result = replace(result, alias=alias)
        view.line(f"{alias} · {model} · {api_style}", bold=True)
        if show_config:
            click.echo(yaml.safe_dump({"models": {alias: result.entry}}, sort_keys=False))
        else:
            view.line(
                "Full configuration is saved with the model. Use --show-config to preview the YAML.",
                dim=True,
            )
        if shadow := shadowing_source(alias, path):
            view.line(
                f"Warning: {shadow} currently defines this alias and takes precedence over this destination. Update that file or explicitly load {path} to use this entry.",
                fg="yellow",
            )
        if yes or confirm(f"Write model entry to {path}?", default=True):
            save_key = False
            if api_key and api_key_env and api_key != os.environ.get(api_key_env):
                secrets_path = get_user_dir("secrets.yaml")
                view.line(
                    f"This setup used a new key. It can be saved in {secrets_path} with owner-only permissions (plain text, not encrypted). Existing values are preserved; YAML formatting may change."
                )
                save_key = not yes and confirm("Save this key for future NOOA runs?", default=True)
                if save_key:
                    from nooa.secrets import write_secret_env

                    write_secret_env(secrets_path, api_key_env, api_key)
                    view.line(
                        f"Saved key as {api_key_env}. Existing secret values are preserved; YAML formatting may change."
                    )
                    if os.environ.get(api_key_env):
                        view.line(
                            f"Your current environment still overrides this file. Unset or update {api_key_env} before starting NOOA again.",
                            fg="yellow",
                        )
            connect.write(result.entry, path, alias=alias)
            click.echo(f"Saved {alias} to {path}.")
            if api_key and not save_key and api_key != os.environ.get(api_key_env):
                click.echo(
                    f"The key was not saved. Set {api_key_env} (or add it to your NOOA secrets file) before using this alias."
                )
            click.echo(f'Use it in Python: get_llm_client("{alias}")')
            if output:
                click.echo(
                    "For a custom path, include it in NEMO_OO_LLM_CONFIG or reload_registry(path)."
                )
    except (ValueError, OSError, yaml.YAMLError, httpx.HTTPError) as exc:
        detail = view.local_failure(exc, api_key=api_key, api_key_env=api_key_env)
        click.echo(
            "Agent diagnostic prompt:\n"
            + connect.diagnostic_prompt(
                "setup",
                {},
                {"setup": {"outcome": "failed", "error": type(exc).__name__, "detail": detail}},
                run_context=diagnostic_context(
                    target=path,
                    alias=alias,
                    model=model,
                    api_key_env=api_key_env,
                    api_key=api_key,
                    output_tokens=output_tokens,
                    reasoning_output_tokens=reasoning_output_tokens,
                ),
            ),
            err=True,
        )
        raise click.ClickException(detail) from None
